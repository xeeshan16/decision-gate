"""Turn a simulation into numbers, and the numbers into a report (text, Markdown, JSON).

Everything here is measured against the truth the simulation knows for every decision,
so the auto tier's error rate is the real one, not the one the gate can estimate from
its labels.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from decision_gate.bounds import interval
from decision_gate.metrics import expected_calibration_error
from decision_gate.simulator.datasets import SEGMENTS
from decision_gate.simulator.engine import DecisionTrace, QuestionRun, SimulationResult

PHASES = ("history", "steady", "drift")
TIERS = ("auto", "verify", "human")
ACTIONS = ("tightened", "suspended")


@dataclass(frozen=True, slots=True)
class AutoError:
    """The auto tier's true error rate over some decisions, with a 95% interval."""

    decisions: int
    errors: int

    @property
    def rate(self) -> float | None:
        return self.errors / self.decisions if self.decisions else None

    @property
    def interval(self) -> tuple[float, float] | None:
        if not self.decisions:
            return None
        low, high = interval(self.decisions - self.errors, self.decisions, 0.95)
        return 1.0 - high, 1.0 - low

    def as_dict(self) -> dict[str, Any]:
        return {
            "decisions": self.decisions,
            "errors": self.errors,
            "rate": self.rate,
            "interval_95": list(self.interval) if self.interval else None,
        }


def auto_error(traces: Sequence[DecisionTrace]) -> AutoError:
    auto = [t for t in traces if t.tier == "auto"]
    return AutoError(len(auto), sum(1 for t in auto if not t.correct))


def _ece(pairs: Sequence[tuple[float, bool]]) -> float | None:
    if len(pairs) < 50:
        return None
    return expected_calibration_error([p for p, _ in pairs], [c for _, c in pairs])


def summarize(traces: Sequence[DecisionTrace]) -> dict[str, Any]:
    n = len(traces)
    tiers = Counter(t.tier for t in traces)
    segments = Counter(t.segment for t in traces)
    calibrated = [(t.calibrated, t.correct) for t in traces if t.calibrated is not None]
    return {
        "decisions": n,
        "tiers": {tier: tiers.get(tier, 0) / n if n else 0.0 for tier in TIERS},
        "segments": {s: segments.get(s, 0) / n if n else 0.0 for s in SEGMENTS},
        "accuracy": sum(t.correct for t in traces) / n if n else None,
        "auto_error": auto_error(traces).as_dict(),
        "reviewed": sum(1 for t in traces if t.labeled_by == "review"),
        "labeled": sum(1 for t in traces if t.labeled_by is not None),
        "ece_raw": _ece([(t.confidence, t.correct) for t in traces]),
        "ece_calibrated": _ece(calibrated),
    }


def _first(days: Sequence[int]) -> int | None:
    return min(days) if days else None


def analyze_question(run: QuestionRun, result: SimulationResult) -> dict[str, Any]:
    cfg = result.config
    policy = result.routing.policy_for(run.question.id)
    by_day: dict[int, list[DecisionTrace]] = {}
    for trace in run.traces:
        by_day.setdefault(trace.day, []).append(trace)
    phases: dict[str, Any] = {}
    for phase in PHASES:
        days = [d for d in range(cfg.days) if cfg.phase(d) == phase]
        if not days:
            continue
        traces = [t for d in days for t in by_day.get(d, [])]
        phases[phase] = {"first_day": days[0] + 1, "last_day": days[-1] + 1, **summarize(traces)}

    daily = []
    for day in range(cfg.days):
        traces = by_day.get(day, [])
        auto, verify = run.thresholds[day] if day < len(run.thresholds) else (None, None)
        check = run.checks.get(day)
        daily.append(
            {
                "day": day + 1,
                "phase": cfg.phase(day),
                "auto_threshold": auto,
                "verify_threshold": verify,
                "decisions": len(traces),
                "auto_share": sum(t.tier == "auto" for t in traces) / len(traces)
                if traces
                else 0.0,
                "auto_error": auto_error(traces).as_dict(),
                "check": None
                if check is None
                else {
                    "status": check["status"],
                    "action": check["action"],
                    "drifted": sorted(n for n, s in check["signals"].items() if s == "drift"),
                },
            }
        )

    drift_start = cfg.history_days + cfg.steady_days  # zero-based first drift day
    drift_days = [d for d, c in run.checks.items() if c["status"] == "drift"]
    action_days = [d for d, c in run.checks.items() if c["action"] in ACTIONS]
    detected = _first([d for d in drift_days if d >= drift_start]) if cfg.drift_days else None
    acted = _first([d for d in action_days if d >= drift_start]) if cfg.drift_days else None
    findings: dict[str, Any] = {
        "budget": policy.max_error_rate if policy else None,
        "false_alarm_days": [d + 1 for d in drift_days if d < drift_start],
        "drift_starts_day": drift_start + 1 if cfg.drift_days else None,
        # A check runs at the end of a day; one at the end of day d protects day d + 1.
        "detected_day": detected + 1 if detected is not None else None,
        "action_day": acted + 1 if acted is not None else None,
        "action": run.checks[acted]["action"] if acted is not None else None,
        "detection_delay_days": detected - drift_start + 1 if detected is not None else None,
    }
    if cfg.drift_days:
        end = acted + 1 if acted is not None else cfg.days
        before = [t for d in range(drift_start, end) for t in by_day.get(d, [])]
        after = [t for d in range(end, cfg.days) for t in by_day.get(d, [])]
        findings["auto_error_before_action"] = auto_error(before).as_dict()
        findings["auto_error_after_action"] = auto_error(after).as_dict()

    return {
        "question": run.question.key,
        "kind": run.question.kind,
        "policy": policy.model_dump(mode="json") if policy else None,
        "phases": phases,
        "findings": findings,
        "daily": daily,
        "events": [
            {"day": e.day + 1, "kind": e.kind, **_event_detail(e.kind, e.detail)}
            for e in run.events
        ],
    }


def _event_detail(kind: str, detail: dict[str, Any]) -> dict[str, Any]:
    if kind == "drift_check":
        return {
            "status": detail["status"],
            "action": detail["action"],
            "action_reason": detail["action_reason"],
            "signals": detail["signals"],
            "messages": detail["messages"],
        }
    return dict(detail)


def analyze(result: SimulationResult) -> dict[str, Any]:
    cfg = result.config
    return {
        "config": {
            "datasets": list(cfg.datasets),
            "seed": cfg.seed,
            "per_day": cfg.per_day,
            "history_days": cfg.history_days,
            "steady_days": cfg.steady_days,
            "drift_days": cfg.drift_days,
            "overconfidence": cfg.overconfidence,
            "reviewer_accuracy": cfg.reviewer_accuracy,
            "outcome_rate": cfg.outcome_rate,
            "outcome_delay_days": cfg.outcome_delay_days,
            "refit_every_days": cfg.refit_every_days,
            "start": cfg.start.isoformat(),
        },
        "database_url": result.database_url,
        "alerts": Counter(event for event, _ in result.alerts),
        "questions": [analyze_question(run, result) for run in result.runs.values()],
    }


# --- Rendering ----------------------------------------------------------------------------


def _pct(value: float | None, digits: int = 1) -> str:
    return "-" if value is None else f"{value:.{digits}%}"


def _num(value: float | None, spec: str = ".3f") -> str:
    return "-" if value is None else format(value, spec)


def _err(stats: dict[str, Any]) -> str:
    if not stats["decisions"]:
        return "-"
    low, high = stats["interval_95"]
    return (
        f"{stats['errors']}/{stats['decisions']} = {stats['rate']:.1%} "
        f"(95% CI {low:.1%}-{high:.1%})"
    )


def _event_line(event: dict[str, Any]) -> str:
    kind = event["kind"]
    if kind == "refit":
        if event["outcome"] == "activated":
            return (
                f"calibrator v{event['version']} activated ({event['method']}, "
                f"{event['labeled']} labeled decisions)"
            )
        return f"refit {event['outcome']}: {event['reason']}"
    if kind == "routing":
        return (
            f"routing refreshed: auto >= {_num(event['auto_threshold'], '.4f')}, verify >= "
            f"{_num(event['verify_threshold'], '.4f')}"
        )
    drifted = ", ".join(n for n, s in event["signals"].items() if s == "drift") or "none"
    line = f"drift check: {event['status']} (signals: {drifted}); action {event['action']}"
    if event["action"] in ACTIONS:
        line += f": {event['action_reason']}"
    return line


def _verdict(stats: dict[str, Any], budget: float) -> str:
    """Compare the true error rate with the budget, allowing for sampling noise."""
    if not stats["decisions"]:
        return "nothing automated"
    low, high = stats["interval_95"]
    if high <= budget:
        return f"within the {budget:.0%} budget"
    if low > budget:
        return f"ABOVE the {budget:.0%} budget"
    return f"not distinguishable from the {budget:.0%} budget"


def _headline(q: dict[str, Any]) -> list[str]:
    f = q["findings"]
    budget = f["budget"]
    lines = []
    steady = q["phases"].get("steady")
    if steady is not None and budget is not None:
        e = steady["auto_error"]
        lines.append(
            f"steady: {_pct(steady['tiers']['auto'])} automated, true auto error "
            f"{_err(e)}, {_verdict(e, budget)}"
        )
    if f["drift_starts_day"] is not None:
        if f["detected_day"] is None:
            lines.append(f"drift from day {f['drift_starts_day']}: not detected")
        else:
            lines.append(
                f"drift from day {f['drift_starts_day']}: detected at the end of day "
                f"{f['detected_day']} ({f['detection_delay_days']} day(s) of drifted traffic)"
                + (
                    f", auto tier {f['action']} at the end of day {f['action_day']}"
                    if f["action_day"] is not None
                    else ", routing not tightened"
                )
            )
        lines.append(
            f"drift, until the action: true auto error {_err(f['auto_error_before_action'])}"
        )
        if f["action_day"] is not None:
            after = f["auto_error_after_action"]
            lines.append(
                f"drift, after the action: true auto error {_err(after)}"
                if after["decisions"]
                else "drift, after the action: nothing automated"
            )
    if f["false_alarm_days"]:
        lines.append(f"drift reported before any drift on days {f['false_alarm_days']}")
    return lines


def render_text(report: dict[str, Any]) -> str:
    cfg = report["config"]
    out = [
        f"decision-gate simulation: seed {cfg['seed']}, {cfg['per_day']} decisions per day "
        f"per question, {cfg['history_days']} + {cfg['steady_days']} + {cfg['drift_days']} "
        "days (history + steady + drift)",
        "",
    ]
    for q in report["questions"]:
        budget = q["findings"]["budget"]
        out.append(
            f"{q['question']} ({q['kind']}"
            + (f", auto error budget {budget:.0%})" if budget is not None else ", no policy)")
        )
        out.extend(f"  {line}" for line in _headline(q))
        out.append("")
        out.append(
            f"  {'phase':<8} {'days':>7} {'decisions':>9} {'auto':>6} {'verify':>6} "
            f"{'human':>6} {'hard+novel':>10} {'ECE raw':>7} {'ECE cal':>7}  true auto error"
        )
        for name, p in q["phases"].items():
            out.append(
                f"  {name:<8} {p['first_day']:>3}-{p['last_day']:<3} {p['decisions']:>9} "
                f"{_pct(p['tiers']['auto']):>6} {_pct(p['tiers']['verify']):>6} "
                f"{_pct(p['tiers']['human']):>6} "
                f"{_pct(p['segments']['hard'] + p['segments']['novel']):>10} "
                f"{_num(p['ece_raw']):>7} {_num(p['ece_calibrated']):>7}  {_err(p['auto_error'])}"
            )
        out.append("")
        out.append("  events:")
        out.extend(f"    day {e['day']:>3}  {_event_line(e)}" for e in q["events"])
        out.append("")
        out.append(f"  {'day':>5} {'auto >=':>8} {'auto':>6} {'auto err':>9}  check")
        for d in q["daily"]:
            if d["phase"] == "history":
                continue
            check = d["check"]
            check_text = (
                "-"
                if check is None
                else f"{check['status']}"
                + (f" ({', '.join(check['drifted'])})" if check["drifted"] else "")
                + (f" -> {check['action']}" if check["action"] in ACTIONS else "")
            )
            out.append(
                f"  {d['day']:>5} {_num(d['auto_threshold'], '.4f'):>8} "
                f"{_pct(d['auto_share']):>6} {_pct(d['auto_error']['rate']):>9}  "
                f"{'*' if d['phase'] == 'drift' else ' '}{check_text}"
            )
        out.append("")
    if report["alerts"]:
        alerts = ", ".join(f"{k} x{v}" for k, v in sorted(report["alerts"].items()))
        out.append(f"webhook alerts: {alerts}")
    if report["database_url"]:
        out.append(f"database kept: {report['database_url']}")
    return "\n".join(out).rstrip() + "\n"


def render_markdown(report: dict[str, Any]) -> str:
    cfg = report["config"]
    out = [
        "# decision-gate simulation report",
        "",
        f"Seed {cfg['seed']}; {cfg['per_day']} decisions per day per question; "
        f"{cfg['history_days']} days of history, {cfg['steady_days']} steady, "
        f"{cfg['drift_days']} drifted. Model overconfidence {cfg['overconfidence']}, "
        f"reviewer accuracy {cfg['reviewer_accuracy']:.0%}, "
        f"{cfg['outcome_rate']:.0%} of auto decisions get their true outcome after "
        f"{cfg['outcome_delay_days']:g} days, refit and refresh every "
        f"{cfg['refit_every_days']} days. All error rates are measured against the "
        "simulation's ground truth.",
        "",
    ]
    for q in report["questions"]:
        out.append(f"## `{q['question']}`")
        out.append("")
        out.extend(f"- {line}" for line in _headline(q))
        out.append("")
        out.append(
            "| phase | days | decisions | auto | verify | human | hard + novel | ECE raw "
            "| ECE calibrated | true auto error |"
        )
        out.append("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |")
        for name, p in q["phases"].items():
            out.append(
                f"| {name} | {p['first_day']}-{p['last_day']} | {p['decisions']} | "
                f"{_pct(p['tiers']['auto'])} | {_pct(p['tiers']['verify'])} | "
                f"{_pct(p['tiers']['human'])} | "
                f"{_pct(p['segments']['hard'] + p['segments']['novel'])} | "
                f"{_num(p['ece_raw'])} | {_num(p['ece_calibrated'])} | {_err(p['auto_error'])} |"
            )
        out.append("")
        out.append("Events:")
        out.append("")
        out.extend(f"- day {e['day']}: {_event_line(e)}" for e in q["events"])
        out.append("")
    return "\n".join(out).rstrip() + "\n"


__all__ = [
    "AutoError",
    "analyze",
    "analyze_question",
    "auto_error",
    "render_markdown",
    "render_text",
    "summarize",
]
