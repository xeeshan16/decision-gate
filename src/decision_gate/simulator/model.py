"""The simulated decision model: skill and miscalibration that depend on the segment.

A real model is good at the cases it has seen many of and bad, but no less sure of
itself, at the ones it has not. This backend reproduces that: each segment of a dataset
(see :mod:`decision_gate.simulator.datasets`) is answered by a
:class:`~decision_gate.backends.fake.FakeBackend` with its own skill floor and its own
overconfidence, and the truth comes from the dataset, so "correct" means correct.

Because the hard and novel profiles report confidences as high as the easy one while
being right less often, the meaning of a confidence score depends on the mix of segments.
A calibrator fitted on one mix is wrong for another, which is exactly what drift does to
a deployed model.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from decision_gate.backends.base import AnyQuestion, DecisionInput, RawDecision
from decision_gate.backends.fake import FakeBackend
from decision_gate.questions import Answer, ChoiceQuestion
from decision_gate.simulator.datasets import SEGMENTS, Segment


@dataclass(frozen=True, slots=True)
class Profile:
    """How the model behaves on one segment.

    ``skill`` sets the floor of the per-item probability of being right, as a share of the
    way from chance to certainty: 0 means the floor is chance, 0.6 means the item is right
    with a probability drawn uniformly between ``chance + 0.6 * (1 - chance)`` and 1.
    ``overconfidence`` multiplies the log-odds distance of the reported confidence from
    chance (1 is calibrated).
    """

    skill: float
    overconfidence: float

    def __post_init__(self) -> None:
        if not 0.0 <= self.skill < 1.0:
            raise ValueError("skill must be within [0, 1)")
        if self.overconfidence <= 0:
            raise ValueError("overconfidence must be positive")


def default_profiles(overconfidence: float = 2.0) -> dict[Segment, Profile]:
    """The model the simulator uses unless told otherwise.

    Hard items are answered with less skill and more bravado; novel ones barely better
    than chance, with the most bravado of all.
    """
    return {
        "easy": Profile(skill=0.9, overconfidence=overconfidence),
        "hard": Profile(skill=0.3, overconfidence=overconfidence * 1.5),
        "novel": Profile(skill=0.0, overconfidence=overconfidence * 2.5),
    }


class SimulatedModel:
    """A :class:`~decision_gate.backends.base.DecisionBackend` for simulations."""

    def __init__(
        self,
        truths: Mapping[str, Answer],
        segments: Mapping[str, Segment],
        profiles: Mapping[Segment, Profile] | None = None,
        *,
        seed: str = "simulator",
        model_version: str = "sim-1",
    ) -> None:
        """``truths`` and ``segments`` are keyed by the ``id`` field of each input.

        They are mappings the simulator keeps filling as it generates items, so the model
        can answer items created after it was constructed.
        """
        self._truths = truths
        self._segments = segments
        self._profiles = dict(profiles or default_profiles())
        missing = set(SEGMENTS) - set(self._profiles)
        if missing:
            raise ValueError(f"no profile for segments {sorted(missing)}")
        self._model_version = model_version
        self._seed = seed
        self._backends: dict[tuple[Segment, str], FakeBackend] = {}

    @property
    def name(self) -> str:
        return "simulated"

    def _backend(self, segment: Segment, question: AnyQuestion) -> FakeBackend:
        key = (segment, question.key)
        backend = self._backends.get(key)
        if backend is None:
            profile = self._profiles[segment]
            chance = 1.0 / len(question.options) if isinstance(question, ChoiceQuestion) else 0.5
            floor = chance + profile.skill * (1.0 - chance)
            backend = FakeBackend(
                overconfidence=profile.overconfidence,
                min_skill=floor,
                oracle=self._oracle,
                seed=f"{self._seed}|{segment}",
                model_version=self._model_version,
            )
            self._backends[key] = backend
        return backend

    def _oracle(self, _question: AnyQuestion, data: DecisionInput) -> Answer:
        return self._truths[str(data["id"])]

    def decide_sync(self, question: AnyQuestion, data: DecisionInput) -> RawDecision:
        item_id = str(data["id"])
        if item_id not in self._truths:
            raise KeyError(f"the simulated model has no truth for input {item_id!r}")
        raw = self._backend(self._segments[item_id], question).decide_sync(question, data)
        return raw.model_copy(update={"backend": self.name})

    async def decide(self, question: AnyQuestion, data: DecisionInput) -> RawDecision:
        return self.decide_sync(question, data)


__all__ = ["Profile", "SimulatedModel", "default_profiles"]
