"""End to end against a running stack: API, worker and PostgreSQL (e.g. `docker compose up`).

Skipped unless ``DGATE_E2E_URL`` points at the API. The stack must use the fake backend and
the example catalog and routing, start from an empty database, and run the worker with a
short interval. Ground truth comes from the fake backend's own oracle, so the test can label
its decisions the way a real system would label them later.

The flow is the product's main loop: decisions go to people while nothing is calibrated,
labels arrive, the worker refits a calibrator and selects thresholds on its own, and new
decisions start to be automated.
"""

from __future__ import annotations

import os
import time
from collections.abc import Callable, Iterator
from typing import Any

import httpx
import pytest

from decision_gate.backends.fake import hash_oracle
from decision_gate.catalog import load_catalog
from tests.conftest import EXAMPLES

BASE_URL = os.environ.get("DGATE_E2E_URL")
pytestmark = pytest.mark.skipif(not BASE_URL, reason="set DGATE_E2E_URL to run against a stack")

QUESTION = "refund.eligible"
HISTORY = 600
WAIT_S = float(os.environ.get("DGATE_E2E_WAIT_S", "180"))


@pytest.fixture(scope="module")
def api() -> Iterator[httpx.Client]:
    assert BASE_URL is not None
    with httpx.Client(base_url=BASE_URL, timeout=30) as client:
        deadline = time.monotonic() + 60
        while True:
            try:
                if client.get("/readyz").status_code == 200:
                    break
            except httpx.TransportError:
                pass
            if time.monotonic() > deadline:  # pragma: no cover - stack never came up
                pytest.fail("API not ready after 60 s")
            time.sleep(1)
        yield client


def _until[T](what: str, probe: Callable[[], T | None]) -> T:
    deadline = time.monotonic() + WAIT_S
    while True:
        value = probe()
        if value is not None:
            return value
        if time.monotonic() > deadline:
            pytest.fail(f"timed out after {WAIT_S:.0f} s waiting for {what}")
        time.sleep(2)


def _decide(api: httpx.Client, item: int, key: str) -> dict[str, Any]:
    response = api.post(
        "/v1/decide",
        json={"question": QUESTION, "input": {"item": item}},
        headers={"Idempotency-Key": key},
    )
    assert response.status_code in (200, 201), response.text
    body: dict[str, Any] = response.json()
    return body


def test_the_worker_turns_labels_into_automation(api: httpx.Client) -> None:
    question = load_catalog(EXAMPLES / "questions.yaml").get(QUESTION)

    # 1. Nothing is calibrated: every decision goes to a person.
    history = [_decide(api, i, f"e2e-history-{i}") for i in range(HISTORY)]
    assert {d["tier"] for d in history} == {"human"}
    queues = api.get("/v1/reviews/stats", params={"question": QUESTION}).json()["queues"]
    assert sum(q["available"] for q in queues) >= HISTORY, queues

    # 2. Ground truth arrives for all of them, in batches.
    for start in range(0, HISTORY, 200):
        batch = [
            {
                "decision_id": d["id"],
                "outcome": hash_oracle(question, {"item": i}),
                "source": "e2e",
            }
            for i, d in enumerate(history[start : start + 200], start=start)
        ]
        result = api.post("/v1/outcomes/batch", json={"outcomes": batch}).json()
        assert result["counts"]["created"] == len(batch), result["counts"]

    # 3. The worker, on its own timer, fits a calibrator and stores a routing rule.
    def calibrated() -> dict[str, Any] | None:
        for c in api.get("/v1/calibrators").json()["calibrators"]:
            if c["question"] == QUESTION and c["status"] == "active":
                return dict(c)
        return None

    calibrator = _until("an active calibrator", calibrated)

    def current_rule() -> dict[str, Any] | None:
        for s in api.get("/v1/routing/status").json()["questions"]:
            if s["question"] == QUESTION and s["state"] == "current":
                return dict(s["rule"])
        return None

    rule = _until("a routing rule on the new calibrator", current_rule)
    assert rule["auto_threshold"] is not None, rule["reason"]

    # 4. New decisions are routed by that rule; the API picks it up after its cache TTL.
    def automated() -> list[dict[str, Any]] | None:
        fresh = [
            _decide(api, HISTORY + i, f"e2e-new-{time.monotonic_ns()}-{i}") for i in range(100)
        ]
        return fresh if any(d["tier"] == "auto" for d in fresh) else None

    fresh = _until("an automated decision", automated)
    auto = [d for d in fresh if d["tier"] == "auto"]
    assert all(d["calibrated_confidence"] >= rule["auto_threshold"] for d in auto)
    assert all(d["routing_rule_id"] is not None for d in fresh)

    # 5. The worker also ran drift checks, and the metrics saw automated decisions.
    _until(
        "a drift check",
        lambda: api.get("/v1/drift/checks").json().get("checks") or None,
    )
    metrics = api.get("/metrics").text
    assert f'dgate_decisions_total{{question="{QUESTION}",replay="false",tier="auto"}}' in metrics
    assert calibrator["method"] in ("platt", "isotonic", "temperature")
