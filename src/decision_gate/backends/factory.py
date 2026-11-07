"""Build the configured backend adapters (primary and shadow candidate) from settings."""

from __future__ import annotations

from pydantic import SecretStr

from decision_gate.backends.base import DecisionBackend
from decision_gate.backends.fake import FakeBackend
from decision_gate.backends.http_api import HTTPDecisionBackend
from decision_gate.backends.llm_json import LLMJSONBackend
from decision_gate.config import Settings


def _secret(value: SecretStr | None) -> str | None:
    return value.get_secret_value() if value else None


def build_backend(settings: Settings) -> DecisionBackend:
    if settings.backend == "http":
        assert settings.http_url is not None  # enforced by Settings
        return HTTPDecisionBackend(
            settings.http_url,
            token=_secret(settings.http_token),
            timeout_s=settings.backend_timeout_s,
        )
    if settings.backend == "llm_json":
        assert settings.llm_base_url is not None
        assert settings.llm_model is not None
        return LLMJSONBackend(
            settings.llm_base_url,
            settings.llm_model,
            api_key=_secret(settings.llm_api_key),
            temperature=settings.llm_temperature,
            timeout_s=settings.backend_timeout_s,
        )
    return FakeBackend(overconfidence=settings.fake_overconfidence, seed=settings.fake_seed)


def build_shadow_backend(settings: Settings) -> DecisionBackend | None:
    """The candidate backend for shadow mode, or ``None`` when it is not configured."""
    if settings.shadow_backend is None:
        return None
    if settings.shadow_backend == "http":
        assert settings.shadow_http_url is not None  # enforced by Settings
        return HTTPDecisionBackend(
            settings.shadow_http_url,
            token=_secret(settings.shadow_http_token),
            timeout_s=settings.shadow_timeout_s,
        )
    if settings.shadow_backend == "llm_json":
        assert settings.shadow_llm_base_url is not None
        assert settings.shadow_llm_model is not None
        return LLMJSONBackend(
            settings.shadow_llm_base_url,
            settings.shadow_llm_model,
            api_key=_secret(settings.shadow_llm_api_key),
            temperature=settings.shadow_llm_temperature,
            timeout_s=settings.shadow_timeout_s,
        )
    return FakeBackend(
        overconfidence=settings.shadow_fake_overconfidence,
        min_skill=settings.shadow_fake_min_skill,
        seed=settings.shadow_fake_seed,
        model_version=settings.shadow_fake_model_version,
    )
