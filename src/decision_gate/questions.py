"""Question definitions: what a decision backend is asked, and what a valid answer looks like.

decision-gate supports three kinds of question. Each one reduces to the same thing for
calibration purposes: the backend proposes an answer and a probability that the answer is
correct, and later an outcome tells us whether it was.

* ``predicate``: a yes/no question ("is this refund eligible?"). The answer is a bool.
* ``choice``: pick one label from a fixed set ("which queue does this ticket belong to?").
* ``score``: estimate a number in a range ("expected handling minutes"). The answer counts
  as correct when it lands within ``tolerance`` of the true value.
"""

from __future__ import annotations

import math
import re
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

QUESTION_ID_PATTERN = re.compile(r"^[a-z][a-z0-9_.-]{0,62}[a-z0-9]$")

Answer = bool | str | float


class _QuestionBase(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str = Field(description="Stable slug, e.g. 'refund.eligible'.")
    version: int = Field(default=1, ge=1, description="Bump when the prompt or options change.")
    prompt: str = Field(min_length=1, max_length=4000)
    description: str = ""

    @field_validator("id")
    @classmethod
    def _check_id(cls, value: str) -> str:
        if not QUESTION_ID_PATTERN.fullmatch(value):
            raise ValueError(
                "id must be 2-64 chars of lowercase letters, digits, '.', '_' or '-', "
                "start with a letter and end with a letter or digit"
            )
        return value

    @property
    def key(self) -> str:
        """Identifier of this exact question version, used to key calibrators."""
        return f"{self.id}@v{self.version}"


class PredicateQuestion(_QuestionBase):
    kind: Literal["predicate"] = "predicate"

    def validate_answer(self, answer: Answer) -> bool:
        if not isinstance(answer, bool):
            raise ValueError(f"{self.key}: predicate answers must be true or false")
        return answer

    def is_correct(self, answer: Answer, outcome: Answer) -> bool:
        return self.validate_answer(answer) == self.validate_answer(outcome)


class ChoiceQuestion(_QuestionBase):
    kind: Literal["choice"] = "choice"
    options: tuple[str, ...] = Field(min_length=2, max_length=256)

    @field_validator("options")
    @classmethod
    def _check_options(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if any(not option.strip() for option in value):
            raise ValueError("options must be non-empty strings")
        if len(set(value)) != len(value):
            raise ValueError("options must be unique")
        return value

    def validate_answer(self, answer: Answer) -> str:
        if not isinstance(answer, str) or answer not in self.options:
            raise ValueError(f"{self.key}: answer must be one of {list(self.options)}")
        return answer

    def is_correct(self, answer: Answer, outcome: Answer) -> bool:
        return self.validate_answer(answer) == self.validate_answer(outcome)


class ScoreQuestion(_QuestionBase):
    kind: Literal["score"] = "score"
    min_value: float
    max_value: float
    tolerance: float = Field(gt=0, description="An answer within this distance is correct.")

    @model_validator(mode="after")
    def _check_range(self) -> ScoreQuestion:
        if not (math.isfinite(self.min_value) and math.isfinite(self.max_value)):
            raise ValueError("min_value and max_value must be finite")
        if self.min_value >= self.max_value:
            raise ValueError("min_value must be below max_value")
        if self.tolerance >= self.max_value - self.min_value:
            raise ValueError("tolerance must be smaller than the value range")
        return self

    def validate_answer(self, answer: Answer) -> float:
        # bool is a subclass of int; a True/False score is almost certainly a client bug.
        if isinstance(answer, bool) or not isinstance(answer, int | float):
            raise ValueError(f"{self.key}: score answers must be numbers")
        value = float(answer)
        if not math.isfinite(value) or not self.min_value <= value <= self.max_value:
            raise ValueError(
                f"{self.key}: score must be within [{self.min_value}, {self.max_value}]"
            )
        return value

    def is_correct(self, answer: Answer, outcome: Answer) -> bool:
        return abs(self.validate_answer(answer) - self.validate_answer(outcome)) <= self.tolerance


Question = Annotated[
    PredicateQuestion | ChoiceQuestion | ScoreQuestion,
    Field(discriminator="kind"),
]


class QuestionCatalog(BaseModel):
    """The set of questions a deployment knows about, loaded from config."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    questions: tuple[Question, ...] = ()

    @model_validator(mode="after")
    def _unique_keys(self) -> QuestionCatalog:
        seen: set[str] = set()
        for question in self.questions:
            if question.key in seen:
                raise ValueError(f"duplicate question {question.key}")
            seen.add(question.key)
        return self

    def get(
        self, question_id: str, version: int | None = None
    ) -> PredicateQuestion | ChoiceQuestion | ScoreQuestion:
        """Return a question by id; the highest version when ``version`` is omitted."""
        matches = [q for q in self.questions if q.id == question_id]
        if version is not None:
            matches = [q for q in matches if q.version == version]
        if not matches:
            suffix = f" version {version}" if version is not None else ""
            raise KeyError(f"unknown question {question_id!r}{suffix}")
        return max(matches, key=lambda q: q.version)
