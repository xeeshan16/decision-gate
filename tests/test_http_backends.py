from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

import httpx
import pytest

from decision_gate.backends import BackendError, HTTPDecisionBackend, LLMJSONBackend
from decision_gate.backends.factory import build_backend
from decision_gate.backends.llm_json import build_messages, parse_reply
from decision_gate.config import Settings
from decision_gate.questions import ChoiceQuestion, PredicateQuestion, ScoreQuestion

Handler = Callable[[httpx.Request], httpx.Response]


def client_for(handler: Handler) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


def json_reply(body: Any, status: int = 200) -> Handler:
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(status, json=body)

    return handler


class TestHTTPDecisionBackend:
    async def test_posts_question_and_input(self, predicate: PredicateQuestion) -> None:
        seen: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            seen.append(request)
            return httpx.Response(
                200, json={"answer": True, "confidence": 0.8, "model_version": "clf-7"}
            )

        backend = HTTPDecisionBackend(
            "http://decider/decide", token="t0k", client=client_for(handler)
        )
        decision = await backend.decide(predicate, {"order_age_days": 3})
        await backend.aclose()

        assert decision.answer is True
        assert decision.confidence == 0.8
        assert decision.probabilities == {"true": 0.8, "false": pytest.approx(0.2)}
        assert decision.backend == "http"
        assert decision.model_version == "clf-7"
        body = json.loads(seen[0].content)
        assert body["question"]["id"] == "refund.eligible"
        assert body["input"] == {"order_age_days": 3}
        assert seen[0].headers["authorization"] == "Bearer t0k"

    async def test_keeps_reported_distribution(self, choice: ChoiceQuestion) -> None:
        probs = {"a": 0.1, "b": 0.6, "c": 0.2, "d": 0.1}
        reply = {"answer": "b", "confidence": 0.6, "probabilities": probs}
        backend = HTTPDecisionBackend("http://x", client=client_for(json_reply(reply)))
        decision = await backend.decide(choice, {})
        assert decision.probabilities == probs
        assert decision.model_version == "unknown"

    async def test_no_auth_header_without_token(self, score: ScoreQuestion) -> None:
        seen: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            seen.append(request)
            return httpx.Response(200, json={"answer": 42, "confidence": 1.0})

        backend = HTTPDecisionBackend("http://x", client=client_for(handler))
        decision = await backend.decide(score, {})
        assert decision.answer == 42.0
        assert decision.confidence < 1.0  # clipped away from certainty
        assert decision.probabilities is None
        assert "authorization" not in seen[0].headers

    @pytest.mark.parametrize(
        ("handler", "match"),
        [
            (json_reply({"error": "boom"}, status=500), "request failed"),
            (lambda _: httpx.Response(200, text="not json"), "unexpected response"),
            (json_reply({"answer": True}), "unexpected response"),
            (json_reply({"answer": True, "confidence": 1.5}), "unexpected response"),
            (json_reply({"answer": "maybe", "confidence": 0.9}), "invalid decision"),
            (
                json_reply({"answer": True, "confidence": 0.9, "probabilities": {"yes": 1.0}}),
                "cover exactly",
            ),
            (
                json_reply(
                    {"answer": True, "confidence": 0.9, "probabilities": {"true": 1, "false": 1}}
                ),
                "sum to 1",
            ),
        ],
    )
    async def test_rejects_bad_replies(
        self, predicate: PredicateQuestion, handler: Handler, match: str
    ) -> None:
        backend = HTTPDecisionBackend("http://x", client=client_for(handler))
        with pytest.raises(BackendError, match=match):
            await backend.decide(predicate, {})

    async def test_probabilities_rejected_for_score(self, score: ScoreQuestion) -> None:
        reply = {"answer": 10, "confidence": 0.5, "probabilities": {"10": 1.0}}
        backend = HTTPDecisionBackend("http://x", client=client_for(json_reply(reply)))
        with pytest.raises(BackendError, match="cover exactly"):
            await backend.decide(score, {})

    async def test_transport_error(self, predicate: PredicateQuestion) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("refused", request=request)

        backend = HTTPDecisionBackend("http://x", client=client_for(handler))
        with pytest.raises(BackendError, match="request failed"):
            await backend.decide(predicate, {})


def chat(content: Any) -> Handler:
    return json_reply({"choices": [{"message": {"role": "assistant", "content": content}}]})


class TestLLMJSONBackend:
    async def test_round_trip(self, choice: ChoiceQuestion) -> None:
        seen: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            seen.append(request)
            return chat('{"answer": "c", "confidence": 0.7}')(request)

        backend = LLMJSONBackend(
            "http://llm/v1/", "small-model", api_key="k", client=client_for(handler)
        )
        decision = await backend.decide(choice, {"subject": "refund"})
        await backend.aclose()

        assert decision.answer == "c"
        assert decision.confidence == 0.7
        assert decision.probabilities is not None
        assert decision.probabilities["a"] == pytest.approx(0.1)
        assert decision.backend == "llm_json"
        assert decision.model_version == "small-model"
        request = seen[0]
        assert str(request.url) == "http://llm/v1/chat/completions"
        assert request.headers["authorization"] == "Bearer k"
        body = json.loads(request.content)
        assert body["model"] == "small-model"
        assert body["temperature"] == 0.0
        assert body["response_format"] == {"type": "json_object"}

    async def test_fenced_reply(self, predicate: PredicateQuestion) -> None:
        reply = '```json\n{"answer": false, "confidence": 0.9}\n```'
        backend = LLMJSONBackend("http://llm", "m", client=client_for(chat(reply)))
        decision = await backend.decide(predicate, {})
        assert decision.answer is False
        assert decision.probabilities == {"true": pytest.approx(0.1), "false": 0.9}

    @pytest.mark.parametrize(
        ("handler", "match"),
        [
            (json_reply({}, status=429), "request failed"),
            (json_reply({"choices": []}), "unexpected response shape"),
            (lambda _: httpx.Response(200, text="<html>"), "unexpected response shape"),
            (chat(None), "not text"),
            (chat("sure! the answer is yes"), "not JSON"),
            (chat('{"answer": true}'), "must be an object"),
            (chat("[1, 2]"), "must be an object"),
            (chat('{"answer": true, "confidence": "high"}'), "must be a number"),
            (chat('{"answer": true, "confidence": true}'), "must be a number"),
            (chat('{"answer": true, "confidence": 90}'), "within"),
            (chat('{"answer": "yes", "confidence": 0.9}'), "invalid decision"),
        ],
    )
    async def test_rejects_bad_replies(
        self, predicate: PredicateQuestion, handler: Handler, match: str
    ) -> None:
        backend = LLMJSONBackend("http://llm", "m", client=client_for(handler))
        with pytest.raises(BackendError, match=match):
            await backend.decide(predicate, {})

    def test_messages_describe_the_answer_format(
        self, predicate: PredicateQuestion, choice: ChoiceQuestion, score: ScoreQuestion
    ) -> None:
        def user(question: Any) -> dict[str, Any]:
            loaded: dict[str, Any] = json.loads(build_messages(question, {"k": 1})[1]["content"])
            return loaded

        assert user(predicate)["format"] == "answer must be true or false"
        assert '["a", "b", "c", "d"]' in user(choice)["format"]
        assert "between 0 and 100" in user(score)["format"]
        assert "within 5" in user(score)["format"]
        assert user(score)["input"] == {"k": 1}
        assert build_messages(predicate, {})[0]["role"] == "system"

    def test_parse_reply_plain(self) -> None:
        assert parse_reply(' {"answer": 1, "confidence": 0.5} ') == {
            "answer": 1,
            "confidence": 0.5,
        }


class TestFactory:
    def test_fake_is_default(self) -> None:
        assert build_backend(Settings()).name == "fake"

    async def test_http(self) -> None:
        backend = build_backend(Settings(backend="http", http_url="http://x", http_token="t"))
        assert isinstance(backend, HTTPDecisionBackend)
        await backend.aclose()

    async def test_llm_json(self) -> None:
        settings = Settings(
            backend="llm_json", llm_base_url="http://llm", llm_model="m", llm_api_key="k"
        )
        backend = build_backend(settings)
        assert isinstance(backend, LLMJSONBackend)
        await backend.aclose()

    async def test_without_secrets(self) -> None:
        for settings in (
            Settings(backend="http", http_url="http://x"),
            Settings(backend="llm_json", llm_base_url="http://llm", llm_model="m"),
        ):
            backend = build_backend(settings)
            assert isinstance(backend, HTTPDecisionBackend | LLMJSONBackend)
            await backend.aclose()

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"backend": "http"}, "DGATE_HTTP_URL"),
            ({"backend": "llm_json", "llm_base_url": "http://llm"}, "DGATE_LLM_MODEL"),
        ],
    )
    def test_missing_settings(self, kwargs: dict[str, Any], match: str) -> None:
        with pytest.raises(ValueError, match=match):
            Settings(**kwargs)
