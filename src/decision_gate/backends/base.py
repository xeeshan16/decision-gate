"""The contract every decision backend implements.

A backend is anything that can answer a :mod:`decision_gate.questions` question about a
piece of input and report how sure it is: a hosted decision API, a prompted LLM that
returns JSON, a classic classifier, or the deterministic fake used in tests and demos.
The gate never trusts the reported confidence directly; it is the raw score that later
gets calibrated against real outcomes.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field, model_validator

from decision_gate.questions import Answer, ChoiceQuestion, PredicateQuestion, ScoreQuestion

AnyQuestion = PredicateQuestion | ChoiceQuestion | ScoreQuestion
DecisionInput = Mapping[str, Any]

# Probabilities exactly 0 or 1 break log loss and logit-based calibrators, and no real
# model is ever that certain. Backends clip into this open interval.
PROBABILITY_EPSILON = 1e-6


def clip_probability(p: float) -> float:
    if not math.isfinite(p):
        raise ValueError("probability must be finite")
    return min(max(p, PROBABILITY_EPSILON), 1.0 - PROBABILITY_EPSILON)


class RawDecision(BaseModel):
    """What a backend proposes, before calibration and routing."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    answer: Answer
    confidence: float = Field(
        gt=0.0, lt=1.0, description="Backend-reported probability that the answer is correct."
    )
    probabilities: dict[str, float] | None = Field(
        default=None,
        description="Full distribution for predicate ('true'/'false') and choice questions.",
    )
    backend: str
    model_version: str

    @model_validator(mode="after")
    def _check_distribution(self) -> RawDecision:
        if self.probabilities is None:
            return self
        if any(not 0.0 <= p <= 1.0 for p in self.probabilities.values()):
            raise ValueError("probabilities must be within [0, 1]")
        total = sum(self.probabilities.values())
        if not math.isclose(total, 1.0, abs_tol=1e-6):
            raise ValueError(f"probabilities must sum to 1 (got {total:.6f})")
        return self


@runtime_checkable
class DecisionBackend(Protocol):
    """Something that answers questions. Implementations must be safe to call concurrently."""

    @property
    def name(self) -> str: ...

    async def decide(self, question: AnyQuestion, data: DecisionInput) -> RawDecision: ...


def answer_key(answer: Answer) -> str:
    """The key an answer uses in a ``probabilities`` mapping."""
    if isinstance(answer, bool):
        return "true" if answer else "false"
    return str(answer)


class BackendError(RuntimeError):
    """The backend failed, timed out or returned something that is not a valid decision."""


def spread_distribution(
    question: AnyQuestion, answer: Answer, confidence: float
) -> dict[str, float] | None:
    """A full distribution that puts ``confidence`` on ``answer`` and spreads the rest evenly.

    Used when a backend reports only the probability of its own answer. Score questions
    have no finite label set, so they get ``None``.
    """
    if isinstance(question, PredicateQuestion):
        p_true = confidence if answer is True else 1.0 - confidence
        return {"true": p_true, "false": 1.0 - p_true}
    if isinstance(question, ChoiceQuestion):
        rest = (1.0 - confidence) / (len(question.options) - 1)
        chosen = answer_key(answer)
        return {o: (confidence if o == chosen else rest) for o in question.options}
    return None
