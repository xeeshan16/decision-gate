"""Adapter for any decision service that already speaks JSON over HTTP.

Many teams already run a classifier or a hosted decision endpoint. This adapter posts the
question and the input to a single URL and expects the answer back, so such a service can
sit behind decision-gate without code changes on either side beyond matching this contract.

Request body::

    {"question": {<question definition>}, "input": {...}}

Expected response (``probabilities`` and ``model_version`` are optional)::

    {"answer": true, "confidence": 0.83, "probabilities": {"true": 0.83, "false": 0.17},
     "model_version": "refund-clf-2026-10"}
"""

from __future__ import annotations

from typing import Any

import httpx
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from decision_gate.backends.base import (
    AnyQuestion,
    BackendError,
    DecisionInput,
    RawDecision,
    clip_probability,
    spread_distribution,
)
from decision_gate.questions import Answer


class _Reply(BaseModel):
    model_config = ConfigDict(extra="ignore")

    answer: Answer
    confidence: float = Field(ge=0.0, le=1.0)
    probabilities: dict[str, float] | None = None
    model_version: str | None = None


def build_decision(
    question: AnyQuestion,
    *,
    answer: Any,
    confidence: float,
    probabilities: dict[str, float] | None,
    backend: str,
    model_version: str,
) -> RawDecision:
    """Validate a backend's answer against the question and assemble a :class:`RawDecision`.

    Shared by the HTTP adapters so every backend is held to the same rules: the answer must
    be valid for the question, and the confidence is clipped away from exactly 0 and 1.
    """
    try:
        checked = question.validate_answer(answer)
        p = clip_probability(confidence)
        expected = spread_distribution(question, checked, p)
        if probabilities is not None and (expected is None or set(probabilities) != set(expected)):
            raise ValueError("probabilities must cover exactly the question's labels")
        return RawDecision(
            answer=checked,
            confidence=p,
            probabilities=probabilities if probabilities is not None else expected,
            backend=backend,
            model_version=model_version,
        )
    except (ValueError, ValidationError) as exc:
        raise BackendError(f"{backend}: invalid decision for {question.key}: {exc}") from exc


class HTTPDecisionBackend:
    """Calls a JSON decision endpoint. See the module docstring for the contract."""

    def __init__(
        self,
        url: str,
        *,
        token: str | None = None,
        timeout_s: float = 10.0,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        headers = {"Authorization": f"Bearer {token}"} if token else {}
        self._url = url
        self._client = client or httpx.AsyncClient(timeout=timeout_s)
        self._headers = headers

    @property
    def name(self) -> str:
        return "http"

    async def aclose(self) -> None:
        await self._client.aclose()

    async def decide(self, question: AnyQuestion, data: DecisionInput) -> RawDecision:
        body = {"question": question.model_dump(mode="json"), "input": dict(data)}
        try:
            response = await self._client.post(self._url, json=body, headers=self._headers)
            response.raise_for_status()
            reply = _Reply.model_validate(response.json())
        except httpx.HTTPError as exc:
            raise BackendError(f"http: request failed: {exc!r}") from exc
        except (ValueError, ValidationError) as exc:
            raise BackendError(f"http: unexpected response: {exc}") from exc
        return build_decision(
            question,
            answer=reply.answer,
            confidence=reply.confidence,
            probabilities=reply.probabilities,
            backend=self.name,
            model_version=reply.model_version or "unknown",
        )
