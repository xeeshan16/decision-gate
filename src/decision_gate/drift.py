"""Drift statistics and the drift policy.

Thresholds are selected on yesterday's traffic and applied to today's. They stay valid only
while three things hold, and each has its own check:

* **The score distribution** the backend produces. If the inputs change (a new product
  line, a new ticket source) or the model behind the backend changes, the distribution of
  its confidence moves. The population stability index (PSI) compares the current window
  with a reference window. It needs no labels, so it is the earliest warning.
* **The outcome mix.** If the share of each true answer changes (more refunds are
  ineligible this month), a calibrator fitted on the old mix is off even when the scores
  look the same. PSI over the outcome labels measures that, on labeled decisions.
* **Calibration itself.** Expected calibration error (ECE) over the labeled decisions of
  the current window, scored with the active calibrator. Small samples inflate ECE, so the
  check fires on the lower end of a bootstrap interval, not the point estimate.

The monitor (:mod:`decision_gate.drift_monitor`) adds the check that matters most for the
error budget: the observed error rate of decisions that cleared the auto threshold.

Everything here is pure computation over arrays; the monitor reads the windows from the
database.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

PSI_EPSILON = 1e-4
"""Floor for an empty bin's share; without it one empty bin makes the PSI infinite."""

SignalName = Literal["score_psi", "label_psi", "ece", "auto_error"]
SignalStatus = Literal["ok", "drift", "insufficient"]
CheckStatus = Literal["ok", "drift", "insufficient"]
DriftAction = Literal["tighten", "alert"]


class DriftConfigError(ValueError):
    """The ``drift`` section of the routing file is invalid."""


# --- Population stability index ----------------------------------------------------------


@dataclass(frozen=True, slots=True)
class PsiBin:
    """One bin: its range (``None`` for an open end, or a label) and both shares."""

    label: str
    reference_share: float
    current_share: float

    @property
    def contribution(self) -> float:
        p = max(self.reference_share, PSI_EPSILON)
        q = max(self.current_share, PSI_EPSILON)
        return (q - p) * math.log(q / p)


@dataclass(frozen=True, slots=True)
class PsiResult:
    value: float
    reference_count: int
    current_count: int
    bins: tuple[PsiBin, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "value": self.value,
            "reference_count": self.reference_count,
            "current_count": self.current_count,
            "bins": [
                {
                    "label": b.label,
                    "reference_share": b.reference_share,
                    "current_share": b.current_share,
                    "contribution": b.contribution,
                }
                for b in self.bins
            ],
        }


def _psi(labels: Sequence[str], reference: np.ndarray, current: np.ndarray) -> PsiResult:
    ref_total = int(reference.sum())
    cur_total = int(current.sum())
    bins = tuple(
        PsiBin(label, float(r / ref_total), float(c / cur_total))
        for label, r, c in zip(labels, reference, current, strict=True)
    )
    return PsiResult(
        value=float(sum(b.contribution for b in bins)),
        reference_count=ref_total,
        current_count=cur_total,
        bins=bins,
    )


def population_stability_index(
    reference: Sequence[float] | np.ndarray,
    current: Sequence[float] | np.ndarray,
    *,
    bins: int = 10,
) -> PsiResult:
    """PSI of ``current`` against ``reference``, over quantile bins of the reference.

    ``PSI = sum((q - p) * ln(q / p))`` over bins, where ``p`` and ``q`` are the reference
    and current shares. The bin edges are the reference's inner quantiles, so each bin
    holds about the same share of the reference and the outer bins are open-ended (scores
    outside the reference's range still land somewhere). Tied quantiles merge, so heavily
    clustered scores give fewer bins. Because the bins come from ranks, the PSI is the same
    on raw and on monotonically calibrated scores.

    Rule of thumb from credit scoring: below 0.1 the populations match, 0.1 to 0.25 is a
    moderate shift, above 0.25 a major one.
    """
    if bins < 2:
        raise ValueError("bins must be at least 2")
    ref = np.asarray(reference, dtype=np.float64)
    cur = np.asarray(current, dtype=np.float64)
    if ref.ndim != 1 or cur.ndim != 1 or ref.size == 0 or cur.size == 0:
        raise ValueError("reference and current must be non-empty 1-d sequences")
    if not (np.isfinite(ref).all() and np.isfinite(cur).all()):
        raise ValueError("scores must be finite")
    inner = np.unique(np.quantile(ref, np.linspace(0.0, 1.0, bins + 1)[1:-1]))
    ref_counts = np.bincount(np.searchsorted(inner, ref, side="left"), minlength=inner.size + 1)
    cur_counts = np.bincount(np.searchsorted(inner, cur, side="left"), minlength=inner.size + 1)
    edges = [-math.inf, *inner.tolist(), math.inf]
    labels = [_range_label(edges[i], edges[i + 1]) for i in range(len(edges) - 1)]
    return _psi(labels, ref_counts, cur_counts)


def _range_label(lower: float, upper: float) -> str:
    low = "-inf" if math.isinf(lower) else f"{lower:.4f}"
    high = "inf" if math.isinf(upper) else f"{upper:.4f}"
    return f"({low}, {high}]"


def categorical_psi(reference: Mapping[str, int], current: Mapping[str, int]) -> PsiResult:
    """PSI over category counts, e.g. how often each true answer occurs.

    A category seen in only one window gets the epsilon share in the other, so a brand-new
    outcome label shows up as a large contribution instead of being ignored.
    """
    labels = sorted(set(reference) | set(current))
    ref = np.array([reference.get(label, 0) for label in labels], dtype=np.float64)
    cur = np.array([current.get(label, 0) for label in labels], dtype=np.float64)
    if (ref < 0).any() or (cur < 0).any():
        raise ValueError("counts must not be negative")
    if ref.sum() == 0 or cur.sum() == 0:
        raise ValueError("both windows need at least one observation")
    return _psi(labels, ref, cur)


# --- Results -----------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Signal:
    """One drift check's verdict, with the measured value and the limit it was held to."""

    name: SignalName
    status: SignalStatus
    message: str
    value: float | None = None
    limit: float | None = None
    detail: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status,
            "message": self.message,
            "value": self.value,
            "limit": self.limit,
            "detail": self.detail,
        }


def overall_status(signals: Sequence[Signal]) -> CheckStatus:
    """``drift`` if any signal drifted, ``ok`` if at least one could be measured."""
    if any(s.status == "drift" for s in signals):
        return "drift"
    if any(s.status == "ok" for s in signals):
        return "ok"
    return "insufficient"


# --- Policy ------------------------------------------------------------------------------


class DriftPolicy(BaseModel):
    """How one question's drift is measured and what happens when it is found.

    Kept apart from :class:`decision_gate.routing.RoutingPolicy` on purpose: a routing rule
    is tied to its policy's fingerprint, so tuning a drift limit must not invalidate every
    rule and send all traffic to a human.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    enabled: bool = True
    window_hours: float = Field(
        default=168.0, gt=0, description="The current window: the last N hours of decisions."
    )
    reference_hours: float = Field(
        default=672.0,
        gt=0,
        description="Length of the reference window, which ends where the active routing "
        "rule was selected (or where the current window starts, if that is earlier).",
    )
    min_decisions: int = Field(
        default=200, ge=10, description="Fewest decisions per window for the score PSI."
    )
    min_labeled: int = Field(
        default=100, ge=10, description="Fewest labeled decisions per window for label PSI/ECE."
    )
    min_auto_labeled: int = Field(
        default=30,
        ge=5,
        description="Fewest labeled decisions above the auto threshold for the error check.",
    )
    psi_bins: int = Field(default=10, ge=2, le=100)
    max_score_psi: float = Field(default=0.25, gt=0)
    max_label_psi: float = Field(default=0.25, gt=0)
    max_ece: float = Field(
        default=0.05,
        gt=0,
        lt=1,
        description="Drift when the lower end of the 95% bootstrap interval of the "
        "window's ECE is above this.",
    )
    ece_bins: int = Field(default=10, ge=2, le=100)
    action: DriftAction = Field(
        default="tighten",
        description="'tighten': raise or suspend the auto threshold on drift, and alert. "
        "'alert': only record the check and alert.",
    )

    @model_validator(mode="after")
    def _check_windows(self) -> DriftPolicy:
        if self.min_auto_labeled > self.min_labeled:
            raise ValueError("min_auto_labeled must not exceed min_labeled")
        return self


class DriftConfig(BaseModel):
    """Drift policies: ``default`` for every question, ``questions`` for overrides."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    default: DriftPolicy = Field(default_factory=DriftPolicy)
    questions: dict[str, DriftPolicy] = Field(default_factory=dict)

    def policy_for(self, question_id: str) -> DriftPolicy:
        return self.questions.get(question_id, self.default)


def parse_drift(raw: Any, *, source: str = "<data>") -> DriftConfig:
    """Parse the ``drift`` section: policy fields shared by every question, plus
    ``questions``, a mapping of question id to overrides of those fields."""
    if raw is None:
        return DriftConfig()
    if not isinstance(raw, dict):
        raise DriftConfigError(f"{source}: 'drift' must be a mapping")
    shared = {k: v for k, v in raw.items() if k != "questions"}
    questions = raw.get("questions") or {}
    if not isinstance(questions, dict):
        raise DriftConfigError(f"{source}: 'drift.questions' must be a mapping")
    try:
        return DriftConfig(
            default=DriftPolicy.model_validate(shared),
            questions={
                qid: DriftPolicy.model_validate({**shared, **(policy or {})})
                for qid, policy in questions.items()
            },
        )
    except ValidationError as exc:
        raise DriftConfigError(f"{source}: invalid drift policy\n{exc}") from exc


__all__ = [
    "PSI_EPSILON",
    "CheckStatus",
    "DriftAction",
    "DriftConfig",
    "DriftConfigError",
    "DriftPolicy",
    "PsiBin",
    "PsiResult",
    "Signal",
    "SignalName",
    "SignalStatus",
    "categorical_psi",
    "overall_status",
    "parse_drift",
    "population_stability_index",
]
