"""Circuit breaker and rate limiter, on a stepped clock."""

from __future__ import annotations

import asyncio

import pytest

from decision_gate.backends.base import BackendError
from decision_gate.resilience import (
    CircuitBreaker,
    CircuitOpenError,
    CircuitState,
    RateLimiter,
    retry_after_header,
)


class StepClock:
    def __init__(self) -> None:
        self.now = 100.0

    def __call__(self) -> float:
        return self.now


async def ok() -> str:
    return "ok"


async def fail() -> str:
    raise BackendError("boom")


async def time_out() -> str:
    raise TimeoutError


async def bug() -> str:
    raise KeyError("caller bug")


def state(breaker: CircuitBreaker) -> CircuitState:
    """Read the state through a call, so the type checker does not narrow it across awaits."""
    return breaker.state


def make_breaker(
    clock: StepClock, transitions: list[tuple[CircuitState, CircuitState]] | None = None
) -> CircuitBreaker:
    def listen(old: CircuitState, new: CircuitState) -> None:
        if transitions is not None:
            transitions.append((old, new))

    return CircuitBreaker(
        "fake", failure_threshold=3, reset_after_s=10, clock=clock, on_state_change=listen
    )


async def test_opens_after_consecutive_failures_and_rejects_fast() -> None:
    clock = StepClock()
    transitions: list[tuple[CircuitState, CircuitState]] = []
    breaker = make_breaker(clock, transitions)
    for _ in range(2):
        with pytest.raises(BackendError):
            await breaker.call(fail)
    assert state(breaker) is CircuitState.CLOSED
    assert breaker.consecutive_failures == 2
    with pytest.raises(TimeoutError):
        await breaker.call(time_out)  # timeouts count too
    assert state(breaker) is CircuitState.OPEN
    assert transitions == [(CircuitState.CLOSED, CircuitState.OPEN)]

    clock.now += 4
    calls = 0

    async def counted() -> str:
        nonlocal calls
        calls += 1
        return "ok"

    with pytest.raises(CircuitOpenError) as info:
        await breaker.call(counted)
    assert calls == 0
    assert info.value.retry_after_s == pytest.approx(6)
    assert "3 consecutive failures" in str(info.value)
    assert isinstance(info.value, BackendError)


async def test_success_resets_the_failure_count() -> None:
    breaker = make_breaker(StepClock())
    for _ in range(2):
        with pytest.raises(BackendError):
            await breaker.call(fail)
    assert await breaker.call(ok) == "ok"
    assert breaker.consecutive_failures == 0
    for _ in range(2):
        with pytest.raises(BackendError):
            await breaker.call(fail)
    assert state(breaker) is CircuitState.CLOSED


async def test_successful_probe_closes_the_circuit() -> None:
    clock = StepClock()
    transitions: list[tuple[CircuitState, CircuitState]] = []
    breaker = make_breaker(clock, transitions)
    for _ in range(3):
        with pytest.raises(BackendError):
            await breaker.call(fail)
    clock.now += 10
    assert state(breaker) is CircuitState.HALF_OPEN  # reported before any call
    assert await breaker.call(ok) == "ok"
    assert state(breaker) is CircuitState.CLOSED
    assert [new for _, new in transitions] == [
        CircuitState.OPEN,
        CircuitState.HALF_OPEN,
        CircuitState.CLOSED,
    ]


async def test_failed_probe_reopens_for_a_full_period() -> None:
    clock = StepClock()
    breaker = make_breaker(clock)
    for _ in range(3):
        with pytest.raises(BackendError):
            await breaker.call(fail)
    clock.now += 11
    with pytest.raises(BackendError):
        await breaker.call(fail)  # one failure is enough in half-open
    assert state(breaker) is CircuitState.OPEN
    clock.now += 9
    with pytest.raises(CircuitOpenError) as info:
        await breaker.call(ok)
    assert info.value.retry_after_s == pytest.approx(1)


async def test_only_one_probe_at_a_time() -> None:
    clock = StepClock()
    breaker = make_breaker(clock)
    for _ in range(3):
        with pytest.raises(BackendError):
            await breaker.call(fail)
    clock.now += 10
    release = asyncio.Event()

    async def slow() -> str:
        await release.wait()
        return "probe"

    probe = asyncio.create_task(breaker.call(slow))
    await asyncio.sleep(0)
    with pytest.raises(CircuitOpenError, match="waiting for a probe"):
        await breaker.call(ok)
    release.set()
    assert await probe == "probe"
    assert await breaker.call(ok) == "ok"


async def test_other_errors_and_cancellation_free_the_probe_without_judging() -> None:
    clock = StepClock()
    breaker = make_breaker(clock)
    for _ in range(3):
        with pytest.raises(BackendError):
            await breaker.call(fail)
    clock.now += 10
    with pytest.raises(KeyError):
        await breaker.call(bug)
    assert state(breaker) is CircuitState.HALF_OPEN

    async def hang() -> str:
        await asyncio.sleep(60)
        raise AssertionError("unreachable")  # pragma: no cover

    task = asyncio.create_task(breaker.call(hang))
    await asyncio.sleep(0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert state(breaker) is CircuitState.HALF_OPEN
    assert await breaker.call(ok) == "ok"
    assert state(breaker) is CircuitState.CLOSED


async def test_other_errors_do_not_count_while_closed() -> None:
    breaker = make_breaker(StepClock())
    for _ in range(5):
        with pytest.raises(KeyError):
            await breaker.call(bug)
    assert state(breaker) is CircuitState.CLOSED
    assert breaker.consecutive_failures == 0


def test_breaker_validates_its_settings() -> None:
    with pytest.raises(ValueError, match="failure_threshold"):
        CircuitBreaker("x", failure_threshold=0)
    with pytest.raises(ValueError, match="reset_after_s"):
        CircuitBreaker("x", reset_after_s=0)


def test_gauge_values_rise_with_severity() -> None:
    values = [s.gauge_value for s in (CircuitState.CLOSED, CircuitState.HALF_OPEN)]
    assert [*values, CircuitState.OPEN.gauge_value] == [0, 1, 2]


def test_rate_limiter_allows_a_burst_then_the_rate() -> None:
    clock = StepClock()
    limiter = RateLimiter(rate=2.0, burst=3, clock=clock)
    assert [limiter.acquire("a") for _ in range(3)] == [0.0, 0.0, 0.0]
    assert limiter.acquire("a") == pytest.approx(0.5)
    assert limiter.acquire("b") == 0.0  # buckets are per key
    clock.now += 0.5
    assert limiter.acquire("a") == 0.0
    assert limiter.acquire("a") == pytest.approx(0.5)
    clock.now += 100
    assert [limiter.acquire("a") for _ in range(4)][-1] > 0  # refills only up to the burst


def test_rate_limiter_forgets_the_least_recently_used_key() -> None:
    clock = StepClock()
    limiter = RateLimiter(rate=1.0, burst=1, max_keys=2, clock=clock)
    assert limiter.acquire("a") == 0.0
    assert limiter.acquire("b") == 0.0
    assert limiter.acquire("a") > 0  # touches "a", so "b" is now the oldest
    assert limiter.acquire("c") == 0.0
    assert len(limiter) == 2
    assert limiter.acquire("b") == 0.0  # forgotten, so it starts with a full bucket
    assert limiter.acquire("a") == 0.0  # and evicted "a" in turn


def test_rate_limiter_validates_its_settings() -> None:
    with pytest.raises(ValueError, match="rate"):
        RateLimiter(rate=0, burst=1)
    with pytest.raises(ValueError, match="burst"):
        RateLimiter(rate=1, burst=0)
    with pytest.raises(ValueError, match="max_keys"):
        RateLimiter(rate=1, burst=1, max_keys=0)


@pytest.mark.parametrize(("seconds", "header"), [(0.01, "1"), (1.0, "1"), (1.2, "2"), (29.5, "30")])
def test_retry_after_header_rounds_up(seconds: float, header: str) -> None:
    assert retry_after_header(seconds) == header
