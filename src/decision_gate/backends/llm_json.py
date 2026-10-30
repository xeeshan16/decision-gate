"""Adapter for a prompted LLM behind an OpenAI-compatible chat completions endpoint.

Works with any server that implements ``POST {base_url}/chat/completions`` (hosted APIs,
or local servers such as Ollama, vLLM and llama.cpp). The model is asked to reply with a
JSON object holding its answer and its confidence. That self-reported confidence is
exactly the kind of number that tends to be overconfident, which is why the gate
calibrates it against outcomes before trusting it.

Nothing here is needed to build, test or demo the project: the default backend is the
deterministic fake, and tests drive this adapter through a mock transport.
"""

from __future__ import annotations

import json
import re
from typing import Any

import httpx

from decision_gate.backends.base import AnyQuestion, BackendError, DecisionInput, RawDecision
from decision_gate.backends.http_api import build_decision
from decision_gate.questions import ChoiceQuestion, PredicateQuestion

SYSTEM_PROMPT = (
    "You answer one question about one input. Reply with a single JSON object and nothing "
    'else: {"answer": <answer>, "confidence": <probability from 0 to 1 that your answer '
    "is correct>}. Do not explain."
)

_FENCE = re.compile(r"^```(?:json)?\s*(.*?)\s*```$", re.DOTALL)


def answer_format(question: AnyQuestion) -> str:
    if isinstance(question, PredicateQuestion):
        return "answer must be true or false"
    if isinstance(question, ChoiceQuestion):
        return f"answer must be exactly one of {json.dumps(list(question.options))}"
    return (
        f"answer must be a number between {question.min_value:g} and {question.max_value:g}; "
        f"it counts as correct within {question.tolerance:g} of the true value"
    )


def build_messages(question: AnyQuestion, data: DecisionInput) -> list[dict[str, str]]:
    user = {
        "question": question.prompt,
        "format": answer_format(question),
        "input": dict(data),
    }
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": json.dumps(user, sort_keys=True, default=str)},
    ]


def parse_reply(content: str) -> dict[str, Any]:
    """Extract the JSON object from a model reply, tolerating a Markdown code fence."""
    text = content.strip()
    fenced = _FENCE.match(text)
    if fenced:
        text = fenced.group(1)
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        raise BackendError(f"llm_json: reply is not JSON: {text[:200]!r}") from exc
    if not isinstance(parsed, dict) or "answer" not in parsed or "confidence" not in parsed:
        raise BackendError("llm_json: reply must be an object with 'answer' and 'confidence'")
    return parsed


class LLMJSONBackend:
    """Asks a chat model for ``{"answer", "confidence"}``. See the module docstring."""

    def __init__(
        self,
        base_url: str,
        model: str,
        *,
        api_key: str | None = None,
        temperature: float = 0.0,
        timeout_s: float = 30.0,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._url = base_url.rstrip("/") + "/chat/completions"
        self._model = model
        self._temperature = temperature
        self._headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self._client = client or httpx.AsyncClient(timeout=timeout_s)

    @property
    def name(self) -> str:
        return "llm_json"

    async def aclose(self) -> None:
        await self._client.aclose()

    async def decide(self, question: AnyQuestion, data: DecisionInput) -> RawDecision:
        body = {
            "model": self._model,
            "temperature": self._temperature,
            "response_format": {"type": "json_object"},
            "messages": build_messages(question, data),
        }
        try:
            response = await self._client.post(self._url, json=body, headers=self._headers)
            response.raise_for_status()
            content = response.json()["choices"][0]["message"]["content"]
        except httpx.HTTPError as exc:
            raise BackendError(f"llm_json: request failed: {exc!r}") from exc
        except (ValueError, KeyError, IndexError, TypeError) as exc:
            raise BackendError(f"llm_json: unexpected response shape: {exc!r}") from exc
        if not isinstance(content, str):
            raise BackendError("llm_json: message content is not text")
        reply = parse_reply(content)
        confidence = reply["confidence"]
        if isinstance(confidence, bool) or not isinstance(confidence, int | float):
            raise BackendError("llm_json: confidence must be a number")
        if not 0.0 <= confidence <= 1.0:
            raise BackendError("llm_json: confidence must be within [0, 1]")
        return build_decision(
            question,
            answer=reply["answer"],
            confidence=float(confidence),
            probabilities=None,
            backend=self.name,
            model_version=self._model,
        )
