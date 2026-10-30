"""Build the configured backend adapter from settings."""

from __future__ import annotations

from decision_gate.backends.base import DecisionBackend
from decision_gate.backends.fake import FakeBackend
from decision_gate.backends.http_api import HTTPDecisionBackend
from decision_gate.backends.llm_json import LLMJSONBackend
from decision_gate.config import Settings


def build_backend(settings: Settings) -> DecisionBackend:
    if settings.backend == "http":
        assert settings.http_url is not None  # enforced by Settings
        return HTTPDecisionBackend(
            settings.http_url,
            token=settings.http_token.get_secret_value() if settings.http_token else None,
            timeout_s=settings.backend_timeout_s,
        )
    if settings.backend == "llm_json":
        assert settings.llm_base_url is not None
        assert settings.llm_model is not None
        return LLMJSONBackend(
            settings.llm_base_url,
            settings.llm_model,
            api_key=settings.llm_api_key.get_secret_value() if settings.llm_api_key else None,
            temperature=settings.llm_temperature,
            timeout_s=settings.backend_timeout_s,
        )
    return FakeBackend(overconfidence=settings.fake_overconfidence, seed=settings.fake_seed)
