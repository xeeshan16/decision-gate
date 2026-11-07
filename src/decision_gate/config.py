"""Runtime settings, read from ``DGATE_*`` environment variables."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from decision_gate.alerts import WebhookAlerter
from decision_gate.reviews import ReviewSla

DEFAULT_DATABASE_URL = "postgresql+asyncpg://dgate:dgate@localhost:5432/dgate"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="DGATE_", extra="ignore")

    questions_file: Path | None = Field(
        default=None, description="YAML or JSON file with the question catalog."
    )
    database_url: str = Field(
        default=DEFAULT_DATABASE_URL,
        description="SQLAlchemy async URL. PostgreSQL (asyncpg) in production.",
    )
    routing_file: Path | None = Field(
        default=None,
        description="YAML or JSON routing policies. Without one every decision goes to a human.",
    )
    routing_cache_ttl_s: float = Field(
        default=5.0,
        ge=0,
        description="How long the router caches a question's rule and calibrator (0 = never).",
    )
    review_verify_sla_minutes: float = Field(
        default=60.0, gt=0, description="Deadline for resolving a verify item."
    )
    review_human_sla_minutes: float = Field(
        default=240.0, gt=0, description="Deadline for resolving a human item."
    )
    review_claim_minutes: float = Field(
        default=15.0,
        gt=0,
        description="How long a claim lasts before the item returns to the queue.",
    )
    drift_webhook_url: str | None = Field(
        default=None, description="Where drift alerts are POSTed. Unset: alerts are not sent."
    )
    drift_webhook_secret: SecretStr | None = Field(
        default=None, description="HMAC-SHA256 key for the X-Dgate-Signature header."
    )
    drift_webhook_timeout_s: float = Field(default=5.0, gt=0)
    drift_webhook_attempts: int = Field(default=3, ge=1, le=10)
    store_inputs: bool = Field(
        default=True,
        description="Keep each decision's input. When false only its SHA-256 hash is stored.",
    )

    backend: Literal["fake", "http", "llm_json"] = Field(
        default="fake", description="Decision backend adapter."
    )
    backend_timeout_s: float = Field(default=10.0, gt=0, description="Per-decision time limit.")

    fake_overconfidence: float = Field(
        default=2.0,
        gt=0,
        description="How overconfident the fake backend is (1.0 means perfectly calibrated).",
    )
    fake_seed: str = "decision-gate"

    http_url: str | None = Field(default=None, description="Endpoint for the 'http' backend.")
    http_token: SecretStr | None = Field(default=None, description="Optional bearer token.")

    llm_base_url: str | None = Field(
        default=None, description="OpenAI-compatible base URL for the 'llm_json' backend."
    )
    llm_model: str | None = None
    llm_api_key: SecretStr | None = None
    llm_temperature: float = Field(default=0.0, ge=0.0, le=2.0)

    shadow_backend: Literal["fake", "http", "llm_json"] | None = Field(
        default=None,
        description="Candidate backend answered in shadow next to the primary. Unset: off.",
    )
    shadow_name: str | None = Field(
        default=None,
        min_length=1,
        max_length=64,
        description="Name the candidate's answers are stored under, e.g. 'gpt-x-prompt-v3'. "
        "Defaults to the shadow backend type.",
    )
    shadow_sample_rate: float = Field(
        default=1.0, ge=0, le=1, description="Fraction of new decisions shadowed."
    )
    shadow_questions: str = Field(
        default="", description="Comma-separated question ids to shadow. Empty: all."
    )
    shadow_timeout_s: float = Field(default=10.0, gt=0, description="Per-call time limit.")
    shadow_max_pending: int = Field(
        default=100, ge=1, description="Most shadow calls in flight; beyond it, skip."
    )
    shadow_fake_overconfidence: float = Field(default=1.0, gt=0)
    shadow_fake_min_skill: float | None = Field(
        default=None,
        gt=0,
        lt=1,
        description="Lowest probability of a correct answer for the fake candidate; higher "
        "makes a more accurate model.",
    )
    shadow_fake_seed: str = "decision-gate"
    shadow_fake_model_version: str = "fake-2"
    shadow_http_url: str | None = None
    shadow_http_token: SecretStr | None = None
    shadow_llm_base_url: str | None = None
    shadow_llm_model: str | None = None
    shadow_llm_api_key: SecretStr | None = None
    shadow_llm_temperature: float = Field(default=0.0, ge=0.0, le=2.0)

    kafka_bootstrap_servers: str = Field(
        default="localhost:9092", description="Kafka brokers for the outcome consumer."
    )
    kafka_outcomes_topic: str = "dgate.outcomes"
    kafka_group_id: str = "decision-gate-outcomes"
    kafka_dead_letter_topic: str | None = Field(
        default="dgate.outcomes.dlq",
        description="Where rejected outcome events go. Empty disables dead-lettering.",
    )

    log_level: Literal["debug", "info", "warning", "error"] = "info"

    @field_validator("shadow_backend", "shadow_name", mode="before")
    @classmethod
    def _blank_is_none(cls, value: object) -> object:
        return None if value == "" else value

    @field_validator("kafka_dead_letter_topic", "drift_webhook_url")
    @classmethod
    def _empty_is_none(cls, value: str | None) -> str | None:
        return value or None

    @model_validator(mode="after")
    def _check_backend_settings(self) -> Settings:
        if self.backend == "http" and not self.http_url:
            raise ValueError("DGATE_HTTP_URL is required when DGATE_BACKEND=http")
        if self.backend == "llm_json" and not (self.llm_base_url and self.llm_model):
            raise ValueError(
                "DGATE_LLM_BASE_URL and DGATE_LLM_MODEL are required when DGATE_BACKEND=llm_json"
            )
        if self.shadow_backend == "http" and not self.shadow_http_url:
            raise ValueError("DGATE_SHADOW_HTTP_URL is required when DGATE_SHADOW_BACKEND=http")
        if self.shadow_backend == "llm_json" and not (
            self.shadow_llm_base_url and self.shadow_llm_model
        ):
            raise ValueError(
                "DGATE_SHADOW_LLM_BASE_URL and DGATE_SHADOW_LLM_MODEL are required when "
                "DGATE_SHADOW_BACKEND=llm_json"
            )
        return self

    @property
    def shadow_candidate(self) -> str | None:
        """The configured candidate's name: ``DGATE_SHADOW_NAME``, else the backend type."""
        return self.shadow_name or self.shadow_backend

    @property
    def shadow_question_ids(self) -> list[str]:
        return [q.strip() for q in self.shadow_questions.split(",") if q.strip()]

    def drift_alerter(self) -> WebhookAlerter | None:
        if not self.drift_webhook_url:
            return None
        return WebhookAlerter(
            self.drift_webhook_url,
            secret=self.drift_webhook_secret.get_secret_value()
            if self.drift_webhook_secret
            else None,
            timeout_s=self.drift_webhook_timeout_s,
            attempts=self.drift_webhook_attempts,
        )

    @property
    def review_sla(self) -> ReviewSla:
        return ReviewSla(
            verify_minutes=self.review_verify_sla_minutes,
            human_minutes=self.review_human_sla_minutes,
            claim_minutes=self.review_claim_minutes,
        )
