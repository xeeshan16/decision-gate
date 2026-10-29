"""Loading the question catalog from a YAML or JSON file."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from decision_gate.questions import QuestionCatalog


class CatalogError(ValueError):
    """The questions file is missing, unreadable or invalid."""


def parse_catalog(raw: Any, *, source: str = "<data>") -> QuestionCatalog:
    if raw is None:
        return QuestionCatalog()
    if isinstance(raw, list):
        raw = {"questions": raw}
    if not isinstance(raw, dict):
        raise CatalogError(f"{source}: expected a mapping with a 'questions' list")
    try:
        return QuestionCatalog.model_validate(raw)
    except ValidationError as exc:
        raise CatalogError(f"{source}: invalid question catalog\n{exc}") from exc


def load_catalog(path: Path) -> QuestionCatalog:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise CatalogError(f"cannot read {path}: {exc.strerror or exc}") from exc
    try:
        raw = json.loads(text) if path.suffix == ".json" else yaml.safe_load(text)
    except (json.JSONDecodeError, yaml.YAMLError) as exc:
        raise CatalogError(
            f"{path}: not valid {'JSON' if path.suffix == '.json' else 'YAML'}: {exc}"
        ) from exc
    return parse_catalog(raw, source=str(path))
