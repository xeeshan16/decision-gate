"""The drift monitor: measure each routed question, tighten its auto tier, send alerts.

One check of one question version compares two windows of stored decisions:

* the **current window**, the last ``window_hours``;
* the **reference window**, ``reference_hours`` long, ending where the active routing rule
  was selected (or where the current window starts, if that is earlier). The thresholds
  in force were selected on traffic like the reference, so that is what today's traffic
  must still resemble.

Four signals (see :mod:`decision_gate.drift` for the first three):

* ``score_psi``: PSI of the raw confidence of every decision, labeled or not;
* ``label_psi``: PSI of the true answers among labeled decisions (not for score questions);
* ``ece``: calibration error of the current window's labeled decisions, scored with the
  active calibrator, judged on the lower end of a bootstrap interval;
* ``auto_error``: of the current window's labeled decisions whose score clears the active
  auto threshold, how many were wrong. These labels come mostly from the random verify
  sample, so they are an unbiased view of the auto tier. The check fires only when the
  error rate is *confidently* over the error budget: the lower confidence bound (the
  policy's bound and confidence) exceeds ``max_error_rate``.

On drift, with ``action: tighten``, the monitor re-selects the auto threshold on the
current window's labels alone and installs the result only if it is stricter than the one
in force. When the window has too few labels to select anything, the auto tier is
suspended (no auto threshold, so those decisions go to verify or to a human). A tightened
rule keeps the policy fingerprint and calibrator of the rule it replaces, so the router
applies it at once, and a routing refresh will not loosen it while the drift persists.

An alert goes out when a question enters drift, when the monitor changes its routing,
and when a question that was drifting checks out ``ok`` again.
"""

from __future__ import annotations

import uuid
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

import numpy as np

from decision_gate.alerts import AlertError, AlertSink
from decision_gate.backends.base import answer_key
from decision_gate.bounds import interval, upper_bound
from decision_gate.calibrators import CalibrationError, load_calibrator
from decision_gate.db.models import DriftCheckRecord, RoutingRuleRecord
from decision_gate.db.repository import (
    CalibratorRepository,
    DecisionRepository,
    DriftCheckRepository,
    LabeledRow,
    OutcomeRepository,
    RoutingRuleRepository,
)
from decision_gate.drift import (
    DriftPolicy,
    Signal,
    categorical_psi,
    overall_status,
    population_stability_index,
)
from decision_gate.metrics import ece_interval, expected_calibration_error
from decision_gate.questions import ScoreQuestion
from decision_gate.refit import calibration_data
from decision_gate.reviews import Clock, as_utc, utc_now
from decision_gate.routing import RoutingConfig, RoutingPolicy
from decision_gate.selection import AnyQuestion, GateAnalyzer, SelectionError

EVENT_DETECTED = "drift.detected"
EVENT_RECOVERED = "drift.recovered"


@dataclass(frozen=True, slots=True)
class _Scores:
    """Scores of labeled rows as the router sees them, or why they are unavailable."""

    values: np.ndarray | None
    correct: np.ndarray
    source: str
    calibrator_id: uuid.UUID | None
    problem: str | None = None


class DriftMonitor:
    def __init__(
        self,
        config: RoutingConfig,
        *,
        decisions: DecisionRepository,
        outcomes: OutcomeRepository,
        calibrators: CalibratorRepository,
        rules: RoutingRuleRepository,
        checks: DriftCheckRepository,
        analyzer: GateAnalyzer,
        alerter: AlertSink | None = None,
        clock: Clock = utc_now,
    ) -> None:
        self.config = config
        self._decisions = decisions
        self._outcomes = outcomes
        self._calibrators = calibrators
        self._rules = rules
        self._checks = checks
        self._analyzer = analyzer
        self._alerter = alerter
        self._clock = clock

    def monitored(self, question: AnyQuestion) -> bool:
        return (
            self.config.policy_for(question.id) is not None
            and self.config.drift.policy_for(question.id).enabled
        )

    async def check(
        self, question: AnyQuestion, *, now: datetime | None = None
    ) -> DriftCheckRecord:
        """Run every signal for one question version, act on the result and store it."""
        policy = self.config.policy_for(question.id)
        if policy is None:
            raise SelectionError(f"no routing policy covers {question.id}")
        drift = self.config.drift.policy_for(question.id)
        if not drift.enabled:
            raise SelectionError(f"drift monitoring is disabled for {question.id}")

        now = as_utc(now or self._clock())
        window_start = now - timedelta(hours=drift.window_hours)
        rule = await self._rules.active(question.id, question.version)
        reference_end = window_start
        if rule is not None:
            reference_end = min(reference_end, as_utc(rule.created_at))
        reference_start = reference_end - timedelta(hours=drift.reference_hours)

        key = (question.id, question.version)
        ref_scores = await self._decisions.confidences(
            *key, since=reference_start, until=reference_end
        )
        cur_scores = await self._decisions.confidences(*key, since=window_start, until=now)
        ref_rows = await self._outcomes.labeled_rows(
            question_id=question.id,
            question_version=question.version,
            since=reference_start,
            until=reference_end,
        )
        cur_rows = await self._outcomes.labeled_rows(
            question_id=question.id,
            question_version=question.version,
            since=window_start,
            until=now,
        )

        record = await self._calibrators.active(question.id, question.version)
        scores = self._scores(cur_rows, record, policy)
        rule_current = (
            rule is not None
            and rule.policy_fingerprint == policy.fingerprint()
            and rule.calibrator_id == (record.id if record else None)
        )
        signals = [
            score_psi_signal(ref_scores, cur_scores, drift),
            label_psi_signal(question, ref_rows, cur_rows, drift),
            ece_signal(scores, drift),
            auto_error_signal(scores, rule if rule_current else None, policy, drift),
        ]
        status = overall_status(signals)

        check = DriftCheckRecord(
            id=uuid.uuid4(),
            question_id=question.id,
            question_version=question.version,
            status=status,
            checked_at=now,
            window_start=window_start,
            window_end=now,
            reference_start=reference_start,
            reference_end=reference_end,
            policy=drift.model_dump(mode="json"),
            signals=[s.as_dict() for s in signals],
            summary={
                "reference": await self._decisions.window_counts(
                    *key, since=reference_start, until=reference_end
                ),
                "current": await self._decisions.window_counts(*key, since=window_start, until=now),
                "scores": scores.source,
                "rule_id": str(rule.id) if rule else None,
                "rule_current": rule_current,
            },
            action="none",
            action_reason="no drift" if status != "drift" else "",
            previous_rule_id=rule.id if rule else None,
        )
        tightened: RoutingRuleRecord | None = None
        if status == "drift":
            tightened = await self._act(
                question, check, rule if rule_current else None, rule, policy, drift
            )
        previous = await self._checks.latest(question.id, question.version)
        stored, _installed = await self._checks.add(check, tightened)
        await self._alert(stored, previous)
        return stored

    async def check_all(
        self, questions: Sequence[AnyQuestion], *, now: datetime | None = None
    ) -> list[DriftCheckRecord]:
        moment = now or self._clock()
        return [await self.check(q, now=moment) for q in questions if self.monitored(q)]

    # --- Scores ---------------------------------------------------------------------------

    @staticmethod
    def _scores(rows: list[LabeledRow], record: Any, policy: RoutingPolicy) -> _Scores:
        correct = np.array([float(r.correct) for r in rows], dtype=np.float64)
        if record is None:
            if policy.require_calibrator:
                return _Scores(None, correct, "none", None, "no active calibrator")
            raw = np.array([r.confidence for r in rows], dtype=np.float64)
            return _Scores(raw, correct, "raw confidence", None)
        if not rows:
            return _Scores(correct, correct, f"calibrator v{record.version}", record.id)
        try:
            calibrator = load_calibrator(record.method, record.params)
            values = calibrator.predict(calibration_data(rows))
        except CalibrationError as exc:
            return _Scores(
                None, correct, f"calibrator v{record.version}", record.id, f"cannot score: {exc}"
            )
        return _Scores(values, correct, f"calibrator v{record.version}", record.id)

    # --- Acting on drift ------------------------------------------------------------------

    async def _act(
        self,
        question: AnyQuestion,
        check: DriftCheckRecord,
        current: RoutingRuleRecord | None,
        active: RoutingRuleRecord | None,
        policy: RoutingPolicy,
        drift: DriftPolicy,
    ) -> RoutingRuleRecord | None:
        if drift.action == "alert":
            check.action_reason = "drift action is 'alert'; routing is unchanged"
            return None
        if current is None:
            check.action = "skipped"
            check.action_reason = (
                "no routing rule in force (missing, or selected under another policy or "
                "calibrator), so decisions already go to a human"
            )
            return None
        if current.auto_threshold is None:
            check.action = "already_tightened" if current.drift_check_id else "none"
            check.action_reason = "the rule has no auto threshold; nothing is automated"
            return None

        old = current.auto_threshold
        threshold, why = await self._window_threshold(question, policy, check)
        if threshold is not None and threshold <= old:
            check.action = "not_tightened"
            check.action_reason = (
                f"the window's labels support the current auto threshold: {why} "
                f"({threshold:.4f} <= {old:.4f})"
            )
            return None

        verify = current.verify_threshold
        if verify is not None and threshold is not None:
            verify = min(verify, threshold)
        if threshold is None:
            check.action = "suspended"
            check.action_reason = f"auto tier suspended (was {old:.4f}): {why}"
        else:
            check.action = "tightened"
            check.action_reason = f"auto threshold raised from {old:.4f} to {threshold:.4f}: {why}"
        drifted = [s["name"] for s in check.signals if s["status"] == "drift"]
        rule = RoutingRuleRecord(
            id=uuid.uuid4(),
            question_id=current.question_id,
            question_version=current.question_version,
            policy_fingerprint=current.policy_fingerprint,
            policy=current.policy,
            calibrator_id=current.calibrator_id,
            auto_threshold=threshold,
            verify_threshold=verify,
            conformal=current.conformal,
            labeled=current.labeled,
            eligible=current.eligible,
            reason=(
                f"drift ({', '.join(drifted)}) at {check.checked_at:%Y-%m-%d %H:%M}Z: "
                f"{check.action_reason}"
            )[:1000],
            evaluation={
                **(current.evaluation or {}),
                "drift": {"check_id": str(check.id), "replaced_rule_id": str(current.id)},
            },
            drift_check_id=check.id,
        )
        check.previous_rule_id = active.id if active else None
        check.rule_id = rule.id
        return rule

    async def _window_threshold(
        self, question: AnyQuestion, policy: RoutingPolicy, check: DriftCheckRecord
    ) -> tuple[float | None, str]:
        """Select the auto threshold on the current window's labels only."""
        try:
            report = await self._analyzer.thresholds(
                question,
                target_precision=policy.target_precision,
                confidence=policy.confidence,
                bound=policy.bound,
                min_accepted=policy.min_accepted,
                holdout=0.0,
                since=as_utc(check.window_start),
                until=as_utc(check.window_end),
            )
        except SelectionError as exc:
            return None, f"no threshold can be selected on the window: {exc}"
        eligible = report.basis.selection_count
        if eligible < policy.min_labeled:
            return None, (
                f"{eligible} eligible labeled decisions in the window, fewer than "
                f"min_labeled={policy.min_labeled}"
            )
        selection = report.selection
        assert selection is not None
        if selection.threshold is None:
            return None, f"no threshold on the window's labels: {selection.reason}"
        return selection.threshold, f"re-selected on {eligible} labeled decisions in the window"

    # --- Alerts ---------------------------------------------------------------------------

    async def _alert(self, check: DriftCheckRecord, previous: DriftCheckRecord | None) -> None:
        event: str | None = None
        if check.status == "drift" and (
            previous is None
            or previous.status != "drift"
            or check.action in ("tightened", "suspended")
        ):
            event = EVENT_DETECTED
        elif check.status == "ok" and previous is not None and previous.status == "drift":
            event = EVENT_RECOVERED
        if event is None:
            return
        if self._alerter is None:
            check.alert, check.alert_event, check.alert_error = "unconfigured", event, None
        else:
            try:
                await self._alerter.send(event, alert_payload(event, check))
            except AlertError as exc:
                check.alert, check.alert_event, check.alert_error = "failed", event, str(exc)[:500]
            else:
                check.alert, check.alert_event, check.alert_error = "sent", event, None
        await self._checks.set_alert(
            check.id, alert=check.alert, event=event, error=check.alert_error
        )


def alert_payload(event: str, check: DriftCheckRecord) -> dict[str, Any]:
    """The webhook body: enough to act on without opening the API."""
    drifted = [s for s in check.signals if s["status"] == "drift"]
    if event == EVENT_RECOVERED:
        headline = f"{check.question_key}: no drift any more"
        advice = "run a routing refresh to re-select thresholds on recent labels"
    else:
        names = ", ".join(s["name"] for s in drifted)
        headline = f"{check.question_key}: drift in {names}"
        advice = check.action_reason
    return {
        "event": event,
        "text": f"{headline}. {advice}",
        "check_id": str(check.id),
        "question": check.question_id,
        "question_version": check.question_version,
        "status": check.status,
        "checked_at": as_utc(check.checked_at).isoformat(),
        "window": {
            "start": as_utc(check.window_start).isoformat(),
            "end": as_utc(check.window_end).isoformat(),
        },
        "signals": [
            {k: s[k] for k in ("name", "status", "value", "limit", "message")}
            for s in check.signals
        ],
        "action": check.action,
        "action_reason": check.action_reason,
        "previous_rule_id": str(check.previous_rule_id) if check.previous_rule_id else None,
        "rule_id": str(check.rule_id) if check.rule_id else None,
    }


# --- Signals -----------------------------------------------------------------------------


def score_psi_signal(
    reference: Sequence[float], current: Sequence[float], drift: DriftPolicy
) -> Signal:
    if len(reference) < drift.min_decisions or len(current) < drift.min_decisions:
        return Signal(
            "score_psi",
            "insufficient",
            f"{len(reference)} reference and {len(current)} current decisions; "
            f"need {drift.min_decisions} in each",
            limit=drift.max_score_psi,
        )
    result = population_stability_index(reference, current, bins=drift.psi_bins)
    detail = result.as_dict()
    detail["reference_mean"] = float(np.mean(reference))
    detail["current_mean"] = float(np.mean(current))
    drifted = result.value > drift.max_score_psi
    return Signal(
        "score_psi",
        "drift" if drifted else "ok",
        f"confidence PSI {result.value:.3f} {'>' if drifted else '<='} {drift.max_score_psi} "
        f"(mean {detail['reference_mean']:.3f} -> {detail['current_mean']:.3f})",
        value=result.value,
        limit=drift.max_score_psi,
        detail=detail,
    )


def label_psi_signal(
    question: AnyQuestion,
    reference: Sequence[LabeledRow],
    current: Sequence[LabeledRow],
    drift: DriftPolicy,
) -> Signal:
    if isinstance(question, ScoreQuestion):
        return Signal(
            "label_psi",
            "insufficient",
            "not measured for score questions",
            limit=drift.max_label_psi,
        )
    if len(reference) < drift.min_labeled or len(current) < drift.min_labeled:
        return Signal(
            "label_psi",
            "insufficient",
            f"{len(reference)} reference and {len(current)} current labeled decisions; "
            f"need {drift.min_labeled} in each",
            limit=drift.max_label_psi,
        )
    result = categorical_psi(
        Counter(answer_key(r.outcome) for r in reference),
        Counter(answer_key(r.outcome) for r in current),
    )
    biggest = max(result.bins, key=lambda b: b.contribution)
    drifted = result.value > drift.max_label_psi
    return Signal(
        "label_psi",
        "drift" if drifted else "ok",
        f"outcome PSI {result.value:.3f} {'>' if drifted else '<='} {drift.max_label_psi} "
        f"(largest change: {biggest.label!r} {biggest.reference_share:.1%} -> "
        f"{biggest.current_share:.1%})",
        value=result.value,
        limit=drift.max_label_psi,
        detail=result.as_dict(),
    )


def ece_signal(scores: _Scores, drift: DriftPolicy) -> Signal:
    if scores.values is None:
        return Signal("ece", "insufficient", f"no scores: {scores.problem}", limit=drift.max_ece)
    n = scores.values.size
    if n < drift.min_labeled:
        return Signal(
            "ece",
            "insufficient",
            f"{n} labeled decisions in the window; need {drift.min_labeled}",
            limit=drift.max_ece,
        )
    ece = expected_calibration_error(scores.values, scores.correct, n_bins=drift.ece_bins)
    low, high = ece_interval(scores.values, scores.correct, n_bins=drift.ece_bins)
    mean_score = float(scores.values.mean())
    accuracy = float(scores.correct.mean())
    drifted = low > drift.max_ece
    return Signal(
        "ece",
        "drift" if drifted else "ok",
        f"ECE {ece:.3f} (95% interval {low:.3f} to {high:.3f}) on {n} labeled decisions, "
        f"{'lower end above' if drifted else 'within'} {drift.max_ece}; mean "
        f"{scores.source} score {mean_score:.3f} vs accuracy {accuracy:.3f}",
        value=ece,
        limit=drift.max_ece,
        detail={
            "labeled": n,
            "interval": [low, high],
            "mean_score": mean_score,
            "accuracy": accuracy,
            "scores": scores.source,
        },
    )


def auto_error_signal(
    scores: _Scores,
    rule: RoutingRuleRecord | None,
    policy: RoutingPolicy,
    drift: DriftPolicy,
) -> Signal:
    limit = policy.max_error_rate
    if rule is None:
        return Signal("auto_error", "insufficient", "no routing rule in force", limit=limit)
    if rule.auto_threshold is None:
        return Signal("auto_error", "insufficient", "the rule has no auto threshold", limit=limit)
    if scores.values is None:
        return Signal("auto_error", "insufficient", f"no scores: {scores.problem}", limit=limit)
    above = scores.values >= rule.auto_threshold
    n = int(above.sum())
    if n < drift.min_auto_labeled:
        return Signal(
            "auto_error",
            "insufficient",
            f"{n} labeled decisions above the auto threshold {rule.auto_threshold:.4f}; "
            f"need {drift.min_auto_labeled}",
            limit=limit,
        )
    correct = int(scores.correct[above].sum())
    errors = n - correct
    # The error rate is at least 1 - (upper bound on accuracy), at the policy's confidence.
    error_low = 1.0 - upper_bound(correct, n, policy.confidence, policy.bound)
    acc_low, acc_high = interval(correct, n, 0.95, policy.bound)
    drifted = error_low > limit
    rate = errors / n
    return Signal(
        "auto_error",
        "drift" if drifted else "ok",
        f"{errors} of {n} labeled decisions above the auto threshold "
        f"{rule.auto_threshold:.4f} were wrong ({rate:.1%}); at {policy.confidence:.0%} "
        f"confidence the error rate is at least {error_low:.1%}, "
        f"{'above' if drifted else 'within'} the {limit:.1%} budget",
        value=rate,
        limit=limit,
        detail={
            "threshold": rule.auto_threshold,
            "labeled_above": n,
            "errors": errors,
            "error_lower_bound": error_low,
            "error_interval_95": [1.0 - acc_high, 1.0 - acc_low],
            "confidence": policy.confidence,
            "bound": policy.bound,
        },
    )


__all__ = [
    "EVENT_DETECTED",
    "EVENT_RECOVERED",
    "DriftMonitor",
    "alert_payload",
    "auto_error_signal",
    "ece_signal",
    "label_psi_signal",
    "score_psi_signal",
]
