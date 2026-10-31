"""Runtime settings, read from ``DGATE_*`` environment variables."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

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

    @field_validator("kafka_dead_letter_topic")
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
        return self
