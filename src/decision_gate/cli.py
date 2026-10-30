"""``dgate`` command-line interface."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer

from decision_gate import __version__
from decision_gate.backends import FakeBackend
from decision_gate.backends.fake import hash_oracle
from decision_gate.catalog import CatalogError, load_catalog

app = typer.Typer(
    help="decision-gate: calibrated thresholds and human-in-the-loop routing.",
    no_args_is_help=True,
    add_completion=False,
)
questions_app = typer.Typer(help="Inspect and validate question catalogs.", no_args_is_help=True)
app.add_typer(questions_app, name="questions")
db_app = typer.Typer(help="Manage the database schema (Alembic migrations).", no_args_is_help=True)
app.add_typer(db_app, name="db")

CatalogPath = Annotated[
    Path, typer.Option("--questions", "-q", help="Question catalog (YAML or JSON).")
]


@app.command()
def version() -> None:
    """Print the version."""
    typer.echo(__version__)


@app.command()
def serve(
    host: Annotated[str, typer.Option(help="Bind address.")] = "127.0.0.1",
    port: Annotated[int, typer.Option(help="Port.")] = 8080,
) -> None:  # pragma: no cover - thin wrapper around uvicorn
    """Run the HTTP API (settings come from DGATE_* environment variables)."""
    import uvicorn

    from decision_gate.app import create_app
    from decision_gate.config import Settings

    settings = Settings()
    uvicorn.run(create_app(settings), host=host, port=port, log_level=settings.log_level)


DatabaseURL = Annotated[
    str | None,
    typer.Option("--url", help="Database URL. Defaults to DGATE_DATABASE_URL."),
]


def _database_url(url: str | None) -> str:
    from decision_gate.config import Settings

    return url or Settings().database_url


@db_app.command("upgrade")
def db_upgrade(
    revision: Annotated[str, typer.Argument(help="Target revision.")] = "head",
    url: DatabaseURL = None,
) -> None:
    """Apply migrations up to a revision (default: the latest)."""
    from decision_gate.db import migrate

    migrate.upgrade(_database_url(url), revision)


@db_app.command("downgrade")
def db_downgrade(
    revision: Annotated[str, typer.Argument(help="Target revision, e.g. 'base'.")],
    url: DatabaseURL = None,
) -> None:
    """Revert migrations down to a revision."""
    from decision_gate.db import migrate

    migrate.downgrade(_database_url(url), revision)


@db_app.command("current")
def db_current(url: DatabaseURL = None) -> None:
    """Show the revision the database is at."""
    from decision_gate.db import migrate

    migrate.current(_database_url(url))


@questions_app.command("check")
def questions_check(path: Annotated[Path, typer.Argument(help="Catalog file.")]) -> None:
    """Validate a catalog and list its questions."""
    try:
        catalog = load_catalog(path)
    except CatalogError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    for q in catalog.questions:
        typer.echo(f"{q.key:<32} {q.kind}")
    typer.echo(f"{len(catalog.questions)} question(s) OK")


@app.command()
def ask(
    question_id: Annotated[str, typer.Argument(help="Question id.")],
    questions: CatalogPath,
    data: Annotated[str, typer.Option("--input", "-i", help="Input as a JSON object.")] = "{}",
    overconfidence: Annotated[
        float, typer.Option(help="Fake backend overconfidence (1.0 = calibrated).")
    ] = 2.0,
) -> None:
    """Ask the deterministic fake backend a question and print its raw decision."""
    try:
        catalog = load_catalog(questions)
        question = catalog.get(question_id)
    except (CatalogError, KeyError) as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    try:
        payload = json.loads(data)
    except json.JSONDecodeError as exc:
        typer.echo(f"error: --input is not valid JSON: {exc}", err=True)
        raise typer.Exit(code=2) from exc
    if not isinstance(payload, dict):
        typer.echo("error: --input must be a JSON object", err=True)
        raise typer.Exit(code=2)
    decision = FakeBackend(overconfidence=overconfidence).decide_sync(question, payload)
    typer.echo(json.dumps({"question": question.key, **decision.model_dump(mode="json")}, indent=2))


@app.command("fake-gap")
def fake_gap(
    questions: CatalogPath,
    samples: Annotated[
        int, typer.Option("--samples", "-n", min=1, help="Inputs per question.")
    ] = 5000,
    overconfidence: Annotated[
        float, typer.Option(help="Fake backend overconfidence (1.0 = calibrated).")
    ] = 2.0,
) -> None:
    """Measure how far the fake backend's confidence is from its real accuracy.

    Uses synthetic inputs and the fake's own ground truth, so it needs nothing external.
    """
    try:
        catalog = load_catalog(questions)
    except CatalogError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    backend = FakeBackend(overconfidence=overconfidence)
    typer.echo(f"{'question':<32} {'accuracy':>9} {'confidence':>11} {'gap':>7}")
    for question in catalog.questions:
        correct = 0
        confidence = 0.0
        for i in range(samples):
            data = {"item": i}
            decision = backend.decide_sync(question, data)
            correct += question.is_correct(decision.answer, hash_oracle(question, data))
            confidence += decision.confidence
        accuracy, mean_conf = correct / samples, confidence / samples
        typer.echo(
            f"{question.key:<32} {accuracy:>9.3f} {mean_conf:>11.3f} {mean_conf - accuracy:>+7.3f}"
        )
