"""Routing tiers: whether a decision is automated, checked by a person, or decided by one.

Every decision lands in one of three tiers:

* ``auto``: the calibrated probability that the answer is correct clears the threshold
  selected for the question's error budget, so the answer is used as is;
* ``verify``: a person confirms the proposed answer (cheap: they check, not decide). This
  covers scores between the verify and auto thresholds, plus a random sample of
  automatable decisions, which keeps an unbiased stream of labels flowing from the auto
  tier. Without it the only labels would come from decisions people already looked at,
  and calibration would drift toward the hard cases;
* ``human``: a person decides from scratch. Anything the gate cannot vouch for goes here:
  no policy, no calibrator yet, too few labels, a rule selected under a different policy
  or calibrator, or a score below every threshold.

The policy (error budget, confidence, minimum sample sizes, sampling rate) is
configuration, kept in a YAML file. The thresholds are *data*: a refresh job selects
them from labeled decisions with the fixed-sequence test in :mod:`decision_gate.thresholds`
and stores them as a routing rule. A rule remembers the policy fingerprint and the
calibrator it was selected under, and the router refuses to apply it once either has
changed: a threshold only means something on the scores it was selected on.
"""

from __future__ import annotations

import hashlib
import json
import time
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from decision_gate.backends.base import answer_key
from decision_gate.bounds import BoundMethod
from decision_gate.calibrators import (
    AnyCalibrator,
    CalibrationError,
    TemperatureScaler,
    load_calibrator,
)
from decision_gate.conformal import ConformalError, ConformalMethod, ConformalPredictor
from decision_gate.db.models import CalibratorRecord, RoutingRuleRecord
from decision_gate.db.repository import CalibratorRepository, RoutingRuleRepository
from decision_gate.questions import Answer, ChoiceQuestion, QuestionCatalog, ScoreQuestion
from decision_gate.selection import AnyQuestion, GateAnalyzer, SelectionError

Tier = Literal["auto", "verify", "human"]
TIERS: tuple[Tier, ...] = ("auto", "verify", "human")


class RoutingConfigError(ValueError):
    """The routing file is missing, unreadable or invalid."""


class RoutingPolicy(BaseModel):
    """How one question is routed. Every field has a conservative default."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    max_error_rate: float = Field(
        default=0.05,
        gt=0,
        lt=1,
        description="Error budget of the auto tier: the highest tolerated error rate.",
    )
    confidence: float = Field(
        default=0.95,
        ge=0.5,
        lt=1,
        description="Confidence that the auto tier's error rate is within the budget.",
    )
    bound: BoundMethod = "clopper-pearson"
    min_labeled: int = Field(
        default=200,
        ge=1,
        description="Fewest eligible labeled decisions before anything is automated.",
    )
    min_accepted: int = Field(
        default=100, ge=1, description="Fewest decisions a candidate threshold must accept."
    )
    require_calibrator: bool = Field(
        default=True,
        description="Send everything to a human until a calibrator is active. Turning this "
        "off routes on raw backend confidence, which is only safe for a backend already "
        "known to be calibrated.",
    )
    verify_max_error_rate: float | None = Field(
        default=None,
        gt=0,
        lt=1,
        description="Looser error budget for the verify tier. Unset: no verify threshold, so "
        "decisions below the auto threshold go to a human.",
    )
    verify_sample_rate: float = Field(
        default=0.05,
        ge=0,
        le=1,
        description="Fraction of automatable decisions sent to verify anyway, for labels.",
    )
    conformal_alpha: float | None = Field(
        default=None,
        gt=0,
        lt=1,
        description="Attach split-conformal answer sets (miss rate alpha) to decisions a "
        "person handles. Choice and predicate questions only.",
    )
    conformal_method: ConformalMethod = "lac"

    @model_validator(mode="after")
    def _check_budgets(self) -> RoutingPolicy:
        if self.verify_max_error_rate is not None and (
            self.verify_max_error_rate <= self.max_error_rate
        ):
            raise ValueError("verify_max_error_rate must be larger than max_error_rate")
        return self

    @property
    def target_precision(self) -> float:
        return 1.0 - self.max_error_rate

    def fingerprint(self) -> str:
        """Stable hash of the policy; a rule is only valid under the policy it was made for."""
        canonical = json.dumps(self.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode()).hexdigest()[:16]


class RoutingConfig(BaseModel):
    """Routing policies by question id. ``default`` covers questions not listed."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    default: RoutingPolicy | None = None
    questions: dict[str, RoutingPolicy] = Field(default_factory=dict)

    def policy_for(self, question_id: str) -> RoutingPolicy | None:
        return self.questions.get(question_id, self.default)

    def unknown_questions(self, catalog: QuestionCatalog) -> list[str]:
        """Policies that name a question the catalog does not have (likely a typo)."""
        known = {q.id for q in catalog.questions}
        return sorted(set(self.questions) - known)


def parse_routing(raw: Any, *, source: str = "<data>") -> RoutingConfig:
    """Build the config, merging ``defaults`` into every per-question policy.

    The file has an optional ``defaults`` mapping (fields shared by every policy), an
    optional ``default`` policy for unlisted questions (``default: {}`` routes every
    question with the defaults; leave it out and unlisted questions always go to a human),
    and ``questions``, a mapping of question id to policy overrides.
    """
    if raw is None:
        return RoutingConfig()
    if not isinstance(raw, dict):
        raise RoutingConfigError(f"{source}: expected a mapping")
    unknown = set(raw) - {"defaults", "default", "questions"}
    if unknown:
        raise RoutingConfigError(f"{source}: unknown keys {sorted(unknown)}")
    shared = raw.get("defaults") or {}
    questions = raw.get("questions") or {}
    if not isinstance(shared, dict) or not isinstance(questions, dict):
        raise RoutingConfigError(f"{source}: 'defaults' and 'questions' must be mappings")
    data: dict[str, Any] = {}
    if "default" in raw:
        data["default"] = {**shared, **(raw["default"] or {})}
    data["questions"] = {qid: {**shared, **(policy or {})} for qid, policy in questions.items()}
    try:
        return RoutingConfig.model_validate(data)
    except ValidationError as exc:
        raise RoutingConfigError(f"{source}: invalid routing policy\n{exc}") from exc


def load_routing(path: Path) -> RoutingConfig:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise RoutingConfigError(f"cannot read {path}: {exc.strerror or exc}") from exc
    try:
        raw = json.loads(text) if path.suffix == ".json" else yaml.safe_load(text)
    except (json.JSONDecodeError, yaml.YAMLError) as exc:
        raise RoutingConfigError(f"{path}: not valid YAML or JSON: {exc}") from exc
    return parse_routing(raw, source=str(path))


def sample_unit(decision_id: uuid.UUID) -> float:
    """A uniform number in [0, 1) derived from the decision id.

    Verify sampling uses this rather than a random draw so it is reproducible: the same
    decision always gets the same tier, and an auditor can recompute who was sampled.
    """
    digest = hashlib.sha256(b"dgate-verify-sample:" + decision_id.bytes).digest()
    return int.from_bytes(digest[:8], "big") / 2.0**64


# --- Refreshing rules -------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class RefreshResult:
    question_key: str
    rule: RoutingRuleRecord


class RoutingRefresher:
    """Select thresholds (and conformal sets) for each routed question and store them."""

    def __init__(
        self,
        config: RoutingConfig,
        analyzer: GateAnalyzer,
        calibrators: CalibratorRepository,
        rules: RoutingRuleRepository,
    ) -> None:
        self._config = config
        self._analyzer = analyzer
        self._calibrators = calibrators
        self._rules = rules

    async def active_calibrator_id(self, question: AnyQuestion) -> uuid.UUID | None:
        record = await self._calibrators.active(question.id, question.version)
        return record.id if record else None

    async def refresh(self, question: AnyQuestion) -> RefreshResult:
        policy = self._config.policy_for(question.id)
        if policy is None:
            raise SelectionError(f"no routing policy covers {question.id}")
        rule = await self._select(question, policy)
        stored = await self._rules.add_active(rule)
        return RefreshResult(question_key=question.key, rule=stored)

    async def _select(self, question: AnyQuestion, policy: RoutingPolicy) -> RoutingRuleRecord:
        rule = RoutingRuleRecord(
            id=uuid.uuid4(),
            question_id=question.id,
            question_version=question.version,
            policy_fingerprint=policy.fingerprint(),
            policy=policy.model_dump(mode="json"),
            labeled=0,
            eligible=0,
            evaluation={},
        )
        active = await self._calibrators.active(question.id, question.version)
        if active is None and policy.require_calibrator:
            rule.calibrator_id = None
            rule.reason = "no active calibrator; every decision goes to a human until one is fitted"
            return rule

        report = await self._analyzer.thresholds(
            question,
            target_precision=policy.target_precision,
            confidence=policy.confidence,
            bound=policy.bound,
            min_accepted=policy.min_accepted,
            holdout=0.0,
        )
        basis = report.basis
        rule.calibrator_id = basis.calibrator.id if basis.calibrator else None
        rule.labeled = basis.labeled
        rule.eligible = basis.selection_count
        scores = basis.score_source
        notes: list[str] = []
        if rule.eligible < policy.min_labeled:
            rule.reason = (
                f"{rule.eligible} eligible labeled decisions, fewer than min_labeled="
                f"{policy.min_labeled}; nothing is automated yet"
            )
            return rule

        selection = report.selection
        assert selection is not None  # eligible >= min_labeled >= 1
        rule.auto_threshold = selection.threshold
        rule.evaluation = {"auto": _selection_summary(selection)}
        if selection.threshold is None:
            notes.append(f"auto: none ({selection.reason})")
        else:
            notes.append(f"auto: {selection.reason}")

        if policy.verify_max_error_rate is not None:
            verify = await self._analyzer.thresholds(
                question,
                target_precision=1.0 - policy.verify_max_error_rate,
                confidence=policy.confidence,
                bound=policy.bound,
                min_accepted=policy.min_accepted,
                holdout=0.0,
            )
            assert verify.selection is not None
            threshold = verify.selection.threshold
            if threshold is not None and rule.auto_threshold is not None:
                threshold = min(threshold, rule.auto_threshold)
            rule.verify_threshold = threshold
            rule.evaluation["verify"] = _selection_summary(verify.selection)
            notes.append(
                f"verify: {verify.selection.reason}" if threshold is not None else "verify: none"
            )

        if policy.conformal_alpha is not None and not isinstance(question, ScoreQuestion):
            try:
                conformal = await self._analyzer.conformal(
                    question,
                    alpha=policy.conformal_alpha,
                    method=policy.conformal_method,
                    holdout=0.0,
                )
            except SelectionError as exc:
                notes.append(f"answer sets: none ({exc})")
            else:
                if conformal.predictor is not None and not conformal.predictor.is_trivial:
                    rule.conformal = conformal.predictor.params()
                notes.append(f"answer sets: {conformal.reason}")

        rule.reason = _clip(f"scores from {scores}; " + "; ".join(notes), 1000)
        return rule


def _clip(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[: limit - 3] + "..."


def _selection_summary(selection: Any) -> dict[str, Any]:
    point = selection.selected
    return {
        "target_precision": selection.target_precision,
        "confidence": selection.confidence,
        "bound": selection.bound,
        "total": selection.total,
        "tested": selection.tested,
        "threshold": selection.threshold,
        "accepted": point.accepted if point else 0,
        "coverage": point.coverage if point else 0.0,
        "precision": point.precision if point else None,
        "lower_bound": point.lower_bound if point else None,
    }


# --- Routing at decision time -----------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Route:
    tier: Tier
    reason: str
    calibrated_confidence: float | None = None
    calibrator_id: uuid.UUID | None = None
    rule_id: uuid.UUID | None = None
    answer_set: list[Answer] | None = None


@dataclass(frozen=True, slots=True)
class _State:
    rule: RoutingRuleRecord | None
    calibrator_record: CalibratorRecord | None
    calibrator: AnyCalibrator | None
    calibrator_error: str | None
    predictor: ConformalPredictor | None


class Router:
    """Assign a tier to each new decision. Never raises for a routing problem: it routes
    the decision to a human and says why, because the gate failing closed must not mean
    the caller's request failing."""

    def __init__(
        self,
        config: RoutingConfig,
        rules: RoutingRuleRepository,
        calibrators: CalibratorRepository,
        *,
        cache_ttl_s: float = 5.0,
    ) -> None:
        self.config = config
        self._rules = rules
        self._calibrators = calibrators
        self._ttl = cache_ttl_s
        self._cache: dict[str, tuple[float, _State]] = {}

    def invalidate(self) -> None:
        """Forget cached rules and calibrators, e.g. right after a refresh or activation."""
        self._cache.clear()

    async def _state(self, question: AnyQuestion) -> _State:
        now = time.monotonic()
        cached = self._cache.get(question.key)
        if cached is not None and cached[0] > now:
            return cached[1]
        rule = await self._rules.active(question.id, question.version)
        record = await self._calibrators.active(question.id, question.version)
        calibrator: AnyCalibrator | None = None
        error: str | None = None
        if record is not None:
            try:
                calibrator = load_calibrator(record.method, record.params)
            except CalibrationError as exc:
                error = str(exc)
        predictor: ConformalPredictor | None = None
        if rule is not None and rule.conformal is not None:
            try:
                predictor = ConformalPredictor.from_params(rule.conformal)
            except (ConformalError, KeyError, TypeError, ValueError):
                predictor = None
        state = _State(rule, record, calibrator, error, predictor)
        if self._ttl > 0:
            self._cache[question.key] = (now + self._ttl, state)
        return state

    async def route(
        self,
        question: AnyQuestion,
        decision_id: uuid.UUID,
        answer: Answer,
        confidence: float,
        probabilities: Mapping[str, float] | None,
    ) -> Route:
        policy = self.config.policy_for(question.id)
        if policy is None:
            return Route("human", f"no routing policy covers {question.id}")
        state = await self._state(question)
        record = state.calibrator_record

        calibrated: float | None = None
        if record is not None:
            if state.calibrator is None:
                return Route(
                    "human",
                    f"the active calibrator v{record.version} cannot be loaded: "
                    f"{state.calibrator_error}",
                    calibrator_id=record.id,
                )
            try:
                calibrated = state.calibrator.calibrate(
                    confidence, probabilities, answer_key(answer)
                )
            except CalibrationError as exc:
                return Route(
                    "human",
                    f"calibrator v{record.version} cannot score this decision: {exc}",
                    calibrator_id=record.id,
                )
        elif policy.require_calibrator:
            return Route("human", f"{question.key} is not calibrated yet")

        def route(tier: Tier, reason: str, *, with_set: bool = True) -> Route:
            answer_set = None
            if with_set and state.predictor is not None and probabilities is not None:
                answer_set = self._answer_set(question, state, probabilities)
            return Route(
                tier,
                reason,
                calibrated_confidence=calibrated,
                calibrator_id=record.id if record else None,
                rule_id=state.rule.id if state.rule else None,
                answer_set=answer_set,
            )

        rule = state.rule
        if rule is None:
            return route("human", f"no routing rule for {question.key} yet; refresh routing")
        if rule.policy_fingerprint != policy.fingerprint():
            return route("human", "the routing policy changed since the rule was selected")
        if rule.calibrator_id != (record.id if record else None):
            return route(
                "human", "the active calibrator changed since the thresholds were selected"
            )

        score = calibrated if calibrated is not None else confidence
        label = "calibrated probability" if calibrated is not None else "raw confidence"
        if rule.auto_threshold is not None and score >= rule.auto_threshold:
            if sample_unit(decision_id) < policy.verify_sample_rate:
                return route(
                    "verify",
                    f"{label} {score:.4f} clears the auto threshold "
                    f"{rule.auto_threshold:.4f}, but the decision was sampled for "
                    f"verification ({policy.verify_sample_rate:.0%} of automatable decisions)",
                )
            return route(
                "auto",
                f"{label} {score:.4f} >= auto threshold {rule.auto_threshold:.4f} "
                f"(error budget {policy.max_error_rate:.1%} at {policy.confidence:.0%} "
                "confidence)",
                with_set=False,
            )
        if rule.verify_threshold is not None and score >= rule.verify_threshold:
            return route(
                "verify",
                f"{label} {score:.4f} >= verify threshold {rule.verify_threshold:.4f}",
            )
        if rule.auto_threshold is None:
            if rule.eligible < policy.min_labeled:
                why = (
                    f"only {rule.eligible} eligible labeled decisions "
                    f"(min_labeled {policy.min_labeled})"
                )
            else:
                why = f"no auto threshold meets the {policy.max_error_rate:.1%} error budget"
            if rule.verify_threshold is not None:
                why += f"; {label} {score:.4f} is below the verify threshold"
                why += f" {rule.verify_threshold:.4f}"
            return route("human", why)
        floor = rule.verify_threshold if rule.verify_threshold is not None else rule.auto_threshold
        return route("human", f"{label} {score:.4f} is below every threshold ({floor:.4f})")

    @staticmethod
    def _answer_set(
        question: AnyQuestion, state: _State, probabilities: Mapping[str, float]
    ) -> list[Answer] | None:
        assert state.predictor is not None
        distribution = dict(probabilities)
        try:
            if isinstance(state.calibrator, TemperatureScaler):
                distribution = state.calibrator.scale(distribution)
            labels = state.predictor.predict_set(distribution)
        except (CalibrationError, ConformalError):
            return None
        if isinstance(question, ChoiceQuestion):
            return list(labels)
        return [label == "true" for label in labels]


__all__ = [
    "TIERS",
    "RefreshResult",
    "Route",
    "Router",
    "RoutingConfig",
    "RoutingConfigError",
    "RoutingPolicy",
    "RoutingRefresher",
    "Tier",
    "load_routing",
    "parse_routing",
    "sample_unit",
]
