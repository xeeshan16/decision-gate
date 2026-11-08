"""Synthetic datasets with known ground truth.

Each dataset generates items for one question: the input a model would see, the true
answer and a *segment* that says how hard the item is for the simulated model. The truth
follows written rules (a returns policy, the wording of a ticket), so it is known for
every item, labeled by the gate or not. That is what makes a simulation useful: it can
report the auto tier's real error rate, which production never sees.

Every dataset has a normal mix and a drifted one. Drift changes the population (more old
orders and a new marketplace line for refunds; a release that floods the technical queue
with tickets about a feature the model has never seen) and therefore what a given
confidence score means, without anyone telling the gate.

Segments:

* ``easy``: clear-cut items;
* ``hard``: items near a policy boundary or with ambiguous wording;
* ``novel``: items from a part of the world that appeared only after drift.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal, Protocol

import numpy as np

from decision_gate.backends.base import AnyQuestion
from decision_gate.questions import Answer, ChoiceQuestion, PredicateQuestion

Segment = Literal["easy", "hard", "novel"]
SEGMENTS: tuple[Segment, ...] = ("easy", "hard", "novel")


@dataclass(frozen=True, slots=True)
class Item:
    """One synthetic case: what the model sees, the truth and how hard it is."""

    data: dict[str, Any]
    truth: Answer
    segment: Segment


class Dataset(Protocol):
    @property
    def name(self) -> str: ...

    @property
    def question(self) -> AnyQuestion: ...

    def generate(self, rng: np.random.Generator, item_id: str, *, drifted: bool) -> Item: ...


def _pick(rng: np.random.Generator, weights: Mapping[str, float]) -> str:
    names = list(weights)
    p = np.array([weights[n] for n in names], dtype=np.float64)
    return names[int(rng.choice(len(names), p=p / p.sum()))]


# --- Refund eligibility -------------------------------------------------------------------

REFUND_CATEGORIES = {
    "apparel": 0.35,
    "electronics": 0.25,
    "home": 0.20,
    "grocery": 0.10,
    "digital": 0.10,
}
# Holiday season: more electronics and digital gifts, and a new marketplace line
# (third-party sellers) with its own, shorter return window.
REFUND_CATEGORIES_DRIFTED = {
    "apparel": 0.25,
    "electronics": 0.20,
    "home": 0.10,
    "grocery": 0.05,
    "digital": 0.15,
    "marketplace": 0.25,
}
REFUND_REASONS = {"changed_mind": 0.40, "damaged": 0.25, "wrong_item": 0.15, "late": 0.20}
REFUND_TIERS = {"standard": 0.7, "plus": 0.25, "business": 0.05}
# Days within this distance of a return window's limit are boundary cases.
BOUNDARY_DAYS = 3


def refund_window(category: str, reason: str, opened: bool) -> int | None:
    """Return window in days under the synthetic policy, or ``None`` if never eligible.

    * grocery: only damaged or wrong items, within 7 days;
    * digital: only unopened, within 14 days;
    * marketplace: 14 days, whatever the reason;
    * everything else: 60 days for damaged or wrong items, 30 days otherwise.
    """
    defect = reason in ("damaged", "wrong_item")
    if category == "grocery":
        return 7 if defect else None
    if category == "digital":
        return None if opened else 14
    if category == "marketplace":
        return 14
    return 60 if defect else 30


def refund_eligible(category: str, reason: str, opened: bool, order_age_days: int) -> bool:
    window = refund_window(category, reason, opened)
    return window is not None and order_age_days <= window


class RefundDataset:
    """``refund.eligible``: is a refund request eligible under the returns policy?"""

    name = "refund"

    def __init__(self) -> None:
        self.question = PredicateQuestion(
            id="refund.eligible",
            prompt="Is this refund request eligible under the returns policy? Answer true or "
            "false.",
            description="Synthetic refund-eligibility check (simulator).",
        )

    def generate(self, rng: np.random.Generator, item_id: str, *, drifted: bool) -> Item:
        category = _pick(rng, REFUND_CATEGORIES_DRIFTED if drifted else REFUND_CATEGORIES)
        reason = _pick(rng, REFUND_REASONS)
        opened = bool(rng.random() < 0.5)
        # Gifts come back late: order ages stretch out in the holiday season.
        age = int(min(120, rng.gamma(2.0, 15.0 if drifted else 9.0)))
        window = refund_window(category, reason, opened)
        if category == "marketplace":
            segment: Segment = "novel"
        elif category == "digital" or (window is not None and abs(age - window) <= BOUNDARY_DAYS):
            segment = "hard"
        else:
            segment = "easy"
        data = {
            "id": item_id,
            "category": category,
            "reason": reason,
            "opened": opened,
            "order_age_days": age,
            "amount": round(float(rng.lognormal(3.5, 0.8)), 2),
            "customer_tier": _pick(rng, REFUND_TIERS),
        }
        return Item(data, refund_eligible(category, reason, opened, age), segment)


# --- Ticket triage ------------------------------------------------------------------------

QUEUES = ("billing", "technical", "account", "shipping", "other")
QUEUE_MIX = {"billing": 0.30, "technical": 0.25, "account": 0.20, "shipping": 0.20, "other": 0.05}
# Release 5.0 ships: technical tickets surge and most of them are about the new sync.
QUEUE_MIX_DRIFTED = {
    "billing": 0.15,
    "technical": 0.55,
    "account": 0.12,
    "shipping": 0.13,
    "other": 0.05,
}
NOVEL_TECHNICAL_SHARE = 0.6

# (subject template, segment). Hard templates read plausibly as another queue.
TICKET_TEMPLATES: dict[str, Sequence[tuple[str, Segment]]] = {
    "billing": (
        ("Charged twice for my {plan} subscription", "easy"),
        ("Invoice {n} shows the wrong amount", "easy"),
        ("Please refund the renewal from last week", "easy"),
        ("Card declined and now I can't sign in", "hard"),
    ),
    "technical": (
        ("App crashes when I open the {screen} screen", "easy"),
        ("Error {code} when exporting a report", "easy"),
        ("Page keeps loading forever on {screen}", "easy"),
        ("Two-factor code never arrives", "hard"),
    ),
    "account": (
        ("How do I change the email on my account?", "easy"),
        ("Please delete my account and data", "easy"),
        ("Add a second admin to our {plan} workspace", "easy"),
        ("Locked out after too many attempts, was I billed?", "hard"),
    ),
    "shipping": (
        ("Order {order} has not arrived yet", "easy"),
        ("Tracking for {order} has not updated in a week", "easy"),
        ("Package arrived damaged", "easy"),
        ("Order {order} delivered but app shows error", "hard"),
    ),
    "other": (
        ("Do you have a partnership program?", "easy"),
        ("Feedback about your support hours", "hard"),
    ),
}
NOVEL_TEMPLATES = (
    "Sync stuck at {pct}% since the 5.0 update",
    "After updating to 5.0 my {screen} data is duplicated",
    "5.0 offline mode loses changes on {screen}",
)
SCREENS = ("dashboard", "reports", "settings", "calendar", "inbox")
PLANS = ("free", "pro", "team", "enterprise")
CHANNELS = {"email": 0.5, "chat": 0.35, "phone": 0.15}


class TicketDataset:
    """``ticket.queue``: which support queue should handle a ticket?"""

    name = "ticket"

    def __init__(self) -> None:
        self.question = ChoiceQuestion(
            id="ticket.queue",
            prompt="Which support queue should handle this ticket?",
            options=QUEUES,
            description="Synthetic ticket triage (simulator).",
        )

    def generate(self, rng: np.random.Generator, item_id: str, *, drifted: bool) -> Item:
        queue = _pick(rng, QUEUE_MIX_DRIFTED if drifted else QUEUE_MIX)
        slots = {
            "plan": PLANS[int(rng.integers(len(PLANS)))],
            "screen": SCREENS[int(rng.integers(len(SCREENS)))],
            "n": f"INV-{int(rng.integers(10_000, 99_999))}",
            "order": f"#{int(rng.integers(100_000, 999_999))}",
            "code": f"E{int(rng.integers(100, 999))}",
            "pct": int(rng.integers(5, 95)),
        }
        segment: Segment
        if drifted and queue == "technical" and rng.random() < NOVEL_TECHNICAL_SHARE:
            template = NOVEL_TEMPLATES[int(rng.integers(len(NOVEL_TEMPLATES)))]
            segment = "novel"
        else:
            templates = TICKET_TEMPLATES[queue]
            template, segment = templates[int(rng.integers(len(templates)))]
        data = {
            "id": item_id,
            "subject": template.format(**slots),
            "channel": _pick(rng, CHANNELS),
            "plan": slots["plan"],
        }
        return Item(data, queue, segment)


DATASETS: dict[str, type[RefundDataset] | type[TicketDataset]] = {
    "refund": RefundDataset,
    "ticket": TicketDataset,
}


def make_dataset(name: str) -> Dataset:
    try:
        return DATASETS[name]()
    except KeyError:
        raise ValueError(f"unknown dataset {name!r}; choose from {sorted(DATASETS)}") from None


__all__ = [
    "DATASETS",
    "QUEUES",
    "SEGMENTS",
    "Dataset",
    "Item",
    "RefundDataset",
    "Segment",
    "TicketDataset",
    "make_dataset",
    "refund_eligible",
    "refund_window",
]
