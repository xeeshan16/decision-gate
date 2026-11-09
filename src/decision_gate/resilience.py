"""Protection against a failing backend and against callers that send too much.

* :class:`CircuitBreaker` stops calling a backend that keeps failing. After
  ``failure_threshold`` consecutive failures (errors or timeouts) it *opens*: calls are
  rejected at once with :class:`CircuitOpenError` instead of each waiting out the timeout.
  After ``reset_after_s`` it lets exactly one probe call through (*half-open*). A
  successful probe closes the circuit; a failed one opens it again for another period.
* :class:`RateLimiter` is a token bucket per caller key: ``rate`` requests per second on
  average, with bursts of up to ``burst``.

Both are in-process and per replica. That is deliberate for the breaker (each replica
sees its own connection to the backend fail). For rate limits it means the effective
limit is ``replicas * rate``; a shared limit belongs in the gateway in front of the
service.

Neither knows about HTTP. Both take an injectable monotonic clock so the tests can step
time instead of sleeping.
"""

from __future__ import annotations

import enum
import math
import time
from collections import OrderedDict
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import TypeVar

from decision_gate.backends.base import BackendError

T = TypeVar("T")
MonotonicClock = Callable[[], float]


class CircuitState(enum.Enum):
    CLOSED = "closed"
    HALF_OPEN = "half_open"
    OPEN = "open"

    @property
    def gauge_value(self) -> int:
        """0 closed, 1 half-open, 2 open: a gauge that rises as things get worse."""
        return {"closed": 0, "half_open": 1, "open": 2}[self.value]


class CircuitOpenError(BackendError):
    """The backend is not being called because it has been failing."""

    def __init__(self, message: str, retry_after_s: float) -> None:
        super().__init__(message)
        self.retry_after_s = retry_after_s


StateListener = Callable[[CircuitState, CircuitState], None]


class CircuitBreaker:
    """Consecutive-failure circuit breaker with a single half-open probe.

    Only ``failure_types`` count as failures. Anything else (a cancelled request, a bug in
    the caller) neither opens nor closes the circuit; a probe that ends that way simply
    frees the probe slot.

    Not thread-safe; it is meant for one asyncio event loop, where the state changes
    between two ``await`` points are atomic.
    """

    def __init__(
        self,
        name: str,
        *,
        failure_threshold: int = 5,
        reset_after_s: float = 30.0,
        failure_types: tuple[type[BaseException], ...] = (BackendError, TimeoutError),
        clock: MonotonicClock = time.monotonic,
        on_state_change: StateListener | None = None,
    ) -> None:
        if failure_threshold < 1:
            raise ValueError("failure_threshold must be at least 1")
        if reset_after_s <= 0:
            raise ValueError("reset_after_s must be positive")
        self.name = name
        self.failure_threshold = failure_threshold
        self.reset_after_s = reset_after_s
        self._failure_types = failure_types
        self._clock = clock
        self._on_state_change = on_state_change
        self._state = CircuitState.CLOSED
        self._failures = 0
        self._opened_at = 0.0
        self._probe_in_flight = False

    @property
    def state(self) -> CircuitState:
        """The current state; an open circuit whose wait is over reports half-open."""
        if self._state is CircuitState.OPEN and self._wait_left() <= 0:
            return CircuitState.HALF_OPEN
        return self._state

    @property
    def consecutive_failures(self) -> int:
        return self._failures

    def _wait_left(self) -> float:
        return self._opened_at + self.reset_after_s - self._clock()

    def _set_state(self, new: CircuitState) -> None:
        old, self._state = self._state, new
        if old is not new and self._on_state_change is not None:
            self._on_state_change(old, new)

    def _acquire(self) -> bool:
        """Admit a call or raise. Returns whether the call is the half-open probe."""
        if self._state is CircuitState.CLOSED:
            return False
        if self._state is CircuitState.OPEN:
            wait = self._wait_left()
            if wait > 0:
                raise CircuitOpenError(
                    f"{self.name}: circuit open after {self._failures} consecutive "
                    f"failures; retry in {wait:.1f}s",
                    retry_after_s=wait,
                )
            self._set_state(CircuitState.HALF_OPEN)
        if self._probe_in_flight:
            raise CircuitOpenError(
                f"{self.name}: circuit half-open, waiting for a probe call to finish",
                retry_after_s=min(1.0, self.reset_after_s),
            )
        self._probe_in_flight = True
        return True

    def _success(self, probe: bool) -> None:
        if probe:
            self._probe_in_flight = False
        self._failures = 0
        self._set_state(CircuitState.CLOSED)

    def _failure(self, probe: bool) -> None:
        if probe:
            self._probe_in_flight = False
        self._failures += 1
        if probe or self._failures >= self.failure_threshold:
            self._opened_at = self._clock()
            self._set_state(CircuitState.OPEN)

    async def call(self, fn: Callable[[], Awaitable[T]]) -> T:
        """Run ``fn`` if the circuit admits it, and record how it went."""
        probe = self._acquire()
        try:
            result = await fn()
        except self._failure_types:
            self._failure(probe)
            raise
        except BaseException:
            if probe:
                self._probe_in_flight = False
            raise
        self._success(probe)
        return result


@dataclass
class _Bucket:
    tokens: float
    updated: float


class RateLimiter:
    """A token bucket per key, with a bounded number of keys.

    ``acquire(key)`` returns 0 when the request may go ahead, otherwise how many seconds
    until a token is available. When more than ``max_keys`` keys are tracked, the least
    recently used is forgotten; a forgotten key starts again with a full bucket, which errs
    on the side of letting traffic through.
    """

    def __init__(
        self,
        rate: float,
        burst: int,
        *,
        max_keys: int = 10_000,
        clock: MonotonicClock = time.monotonic,
    ) -> None:
        if rate <= 0:
            raise ValueError("rate must be positive")
        if burst < 1:
            raise ValueError("burst must be at least 1")
        if max_keys < 1:
            raise ValueError("max_keys must be at least 1")
        self.rate = rate
        self.burst = burst
        self.max_keys = max_keys
        self._clock = clock
        self._buckets: OrderedDict[str, _Bucket] = OrderedDict()

    def __len__(self) -> int:
        return len(self._buckets)

    def acquire(self, key: str) -> float:
        now = self._clock()
        bucket = self._buckets.get(key)
        if bucket is None:
            bucket = _Bucket(tokens=float(self.burst), updated=now)
            self._buckets[key] = bucket
            while len(self._buckets) > self.max_keys:
                self._buckets.popitem(last=False)
        else:
            self._buckets.move_to_end(key)
            elapsed = max(0.0, now - bucket.updated)
            bucket.tokens = min(float(self.burst), bucket.tokens + elapsed * self.rate)
            bucket.updated = now
        if bucket.tokens >= 1.0:
            bucket.tokens -= 1.0
            return 0.0
        return (1.0 - bucket.tokens) / self.rate


def retry_after_header(seconds: float) -> str:
    """Whole seconds for a ``Retry-After`` header, never less than one."""
    return str(max(1, math.ceil(seconds)))
