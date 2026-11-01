"""Calibration reports: metrics for each question version, optionally per backend and model.

Calibration is always measured per question *version* (a new prompt or option set is a
new model as far as calibration goes). Reports can be split further by ``backend`` and
``model_version``, which is how a team compares a new model or prompt against the old one.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from itertools import groupby
from typing import Literal, get_args

import numpy as np

from decision_gate.backends.base import answer_key
from decision_gate.db.repository import LabeledRow
from decision_gate.metrics import BinStrategy, CalibrationMetrics, calibration_metrics

GroupField = Literal["backend", "model_version"]
GROUP_FIELDS: tuple[GroupField, ...] = get_args(GroupField)


@dataclass(frozen=True, slots=True)
class CalibrationReport:
    """Metrics for one group of labeled decisions."""

    question_id: str
    question_version: int
    backend: str | None
    """Set when the report is split by backend, otherwise ``None`` (all backends)."""
    model_version: str | None
    metrics: CalibrationMetrics

    @property
    def question_key(self) -> str:
        return f"{self.question_id}@v{self.question_version}"


def _group_key(row: LabeledRow, group_by: Sequence[GroupField]) -> tuple[str, int, str, str]:
    return (
        row.question_id,
        row.question_version,
        row.backend if "backend" in group_by else "",
        row.model_version if "model_version" in group_by else "",
    )


def _distribution(rows: Sequence[LabeledRow]) -> tuple[np.ndarray, np.ndarray] | None:
    """The full predicted distribution and true label index per row, when every row has one.

    Labels are the union of every key seen plus the true labels, so a true label the
    backend gave no probability counts as probability 0 (and costs the most log loss).
    Score questions, and groups where any decision lacks a distribution, return ``None``.
    """
    if not rows or any(
        row.probabilities is None or not isinstance(row.outcome, bool | str) for row in rows
    ):
        return None
    truths = [answer_key(row.outcome) for row in rows]
    labels = sorted({key for row in rows for key in row.probabilities or {}} | set(truths))
    column = {label: i for i, label in enumerate(labels)}
    matrix = np.zeros((len(rows), len(labels)), dtype=np.float64)
    for i, row in enumerate(rows):
        for label, p in (row.probabilities or {}).items():
            matrix[i, column[label]] = p
    return matrix, np.array([column[t] for t in truths], dtype=np.intp)


def build_reports(
    rows: Iterable[LabeledRow],
    *,
    group_by: Sequence[GroupField] = (),
    n_bins: int = 10,
    strategy: BinStrategy = "uniform",
    bootstrap: int = 0,
    seed: int = 0,
    min_count: int = 1,
) -> list[CalibrationReport]:
    """One report per group with at least ``min_count`` labeled decisions, sorted by key."""
    unknown = set(group_by) - set(GROUP_FIELDS)
    if unknown:
        raise ValueError(f"cannot group by {sorted(unknown)}; choose from {list(GROUP_FIELDS)}")
    if min_count < 1:
        raise ValueError("min_count must be at least 1")
    ordered = sorted(rows, key=lambda row: _group_key(row, group_by))
    reports: list[CalibrationReport] = []
    for (qid, version, backend, model_version), members in groupby(
        ordered, key=lambda row: _group_key(row, group_by)
    ):
        group = list(members)
        if len(group) < min_count:
            continue
        distribution = _distribution(group)
        metrics = calibration_metrics(
            [row.confidence for row in group],
            [row.correct for row in group],
            n_bins=n_bins,
            strategy=strategy,
            bootstrap=bootstrap,
            seed=seed,
            probabilities=distribution[0] if distribution else None,
            labels=distribution[1] if distribution else None,
        )
        reports.append(
            CalibrationReport(
                question_id=qid,
                question_version=version,
                backend=backend if "backend" in group_by else None,
                model_version=model_version if "model_version" in group_by else None,
                metrics=metrics,
            )
        )
    return reports
