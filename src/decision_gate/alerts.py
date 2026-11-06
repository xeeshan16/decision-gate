"""Alert delivery: a signed JSON webhook, with retries.

The receiver can be anything that accepts a POST: a chat incoming-webhook relay, an
incident tool, a small service of your own. Each request carries:

* ``X-Dgate-Event``: the event name (``drift.detected`` or ``drift.recovered``);
* ``X-Dgate-Delivery``: a unique id per alert, the same on every retry, so the receiver
  can drop duplicates;
* ``X-Dgate-Signature: sha256=<hex>`` when a secret is configured: HMAC-SHA256 of the raw
  body with the secret, so the receiver can check the alert came from this service.

Delivery is at-least-once within one call: a timeout or a 5xx/429 response is retried with
exponential backoff; any other 4xx is a configuration problem and is not retried.
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import uuid
from typing import Any, Protocol

import httpx


class AlertError(RuntimeError):
    """The alert could not be delivered."""


class AlertSink(Protocol):
    async def send(self, event: str, payload: dict[str, Any]) -> None:
        """Deliver one alert or raise :class:`AlertError`."""


def sign(secret: str, body: bytes) -> str:
    """The ``X-Dgate-Signature`` header value for ``body``."""
    digest = hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()
    return f"sha256={digest}"


class WebhookAlerter:
    def __init__(
        self,
        url: str,
        *,
        secret: str | None = None,
        timeout_s: float = 5.0,
        attempts: int = 3,
        backoff_s: float = 0.5,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        if attempts < 1:
            raise ValueError("attempts must be at least 1")
        self.url = url
        self._secret = secret
        self._timeout_s = timeout_s
        self._attempts = attempts
        self._backoff_s = backoff_s
        self._transport = transport

    async def send(self, event: str, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode()
        headers = {
            "Content-Type": "application/json",
            "User-Agent": "decision-gate-alerts",
            "X-Dgate-Event": event,
            "X-Dgate-Delivery": str(uuid.uuid4()),
        }
        if self._secret:
            headers["X-Dgate-Signature"] = sign(self._secret, body)
        last = "no attempt made"
        async with httpx.AsyncClient(timeout=self._timeout_s, transport=self._transport) as client:
            for attempt in range(self._attempts):
                if attempt:
                    await asyncio.sleep(self._backoff_s * 2 ** (attempt - 1))
                try:
                    response = await client.post(self.url, content=body, headers=headers)
                except httpx.HTTPError as exc:
                    last = f"{type(exc).__name__}: {exc}"
                    continue
                if response.is_success:
                    return
                last = f"HTTP {response.status_code}"
                if response.status_code < 500 and response.status_code != 429:
                    break
        raise AlertError(f"webhook delivery failed after {attempt + 1} attempt(s): {last}")


__all__ = ["AlertError", "AlertSink", "WebhookAlerter", "sign"]
