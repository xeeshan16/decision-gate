"""Runtime settings, read from ``DGATE_*`` environment variables."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="DGATE_", extra="ignore")

    questions_file: Path | None = Field(
        default=None, description="YAML or JSON file with the question catalog."
    )
    backend: Literal["fake"] = Field(
        default="fake", description="Decision backend. More adapters arrive with the decide API."
    )
    fake_overconfidence: float = Field(
        default=2.0,
        gt=0,
        description="How overconfident the fake backend is (1.0 means perfectly calibrated).",
    )
    fake_seed: str = "decision-gate"
    log_level: Literal["debug", "info", "warning", "error"] = "info"
