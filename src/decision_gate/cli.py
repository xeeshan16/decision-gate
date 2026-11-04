"""``dgate`` command-line interface."""

from __future__ import annotations

import asyncio
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any

import typer

from decision_gate import __version__
from decision_gate.backends import FakeBackend
from decision_gate.backends.fake import hash_oracle
from decision_gate.catalog import CatalogError, load_catalog

if TYPE_CHECKING:
    from decision_gate.calibration import CalibrationReport, GroupField
    from decision_gate.ingest import IngestStats, OutcomeSource
    from decision_gate.metrics import BinStrategy
    from decision_gate.questions import QuestionCatalog
    from decision_gate.refit import RefitResult

app = typer.Typer(
    help="decision-gate: calibrated thresholds and human-in-the-loop routing.",
    no_args_is_help=True,
    add_completion=False,
)
questions_app = typer.Typer(help="Inspect and validate question catalogs.", no_args_is_help=True)
app.add_typer(questions_app, name="questions")
db_app = typer.Typer(help="Manage the database schema (Alembic migrations).", no_args_is_help=True)
app.add_typer(db_app, name="db")
outcomes_app = typer.Typer(help="Ingest outcomes (ground truth).", no_args_is_help=True)
app.add_typer(outcomes_app, name="outcomes")
calibration_app = typer.Typer(
    help="Measure calibration on labeled decisions.", no_args_is_help=True
)
app.add_typer(calibration_app, name="calibration")
calibrators_app = typer.Typer(
    help="Fit, list and activate calibrators (the versioned registry).", no_args_is_help=True
)
app.add_typer(calibrators_app, name="calibrators")
thresholds_app = typer.Typer(
    help="Select automation thresholds with statistical guarantees.", no_args_is_help=True
)
app.add_typer(thresholds_app, name="thresholds")
conformal_app = typer.Typer(
    help="Calibrate conformal prediction sets for choice questions.", no_args_is_help=True
)
app.add_typer(conformal_app, name="conformal")

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
    from decision_gate.metrics import calibration_metrics

    backend = FakeBackend(overconfidence=overconfidence)
    typer.echo(
        f"{'question':<32} {'accuracy':>9} {'confidence':>11} {'gap':>7} {'ECE':>7} {'Brier':>7}"
    )
    for question in catalog.questions:
        confidences: list[float] = []
        hits: list[bool] = []
        for i in range(samples):
            data = {"item": i}
            decision = backend.decide_sync(question, data)
            confidences.append(decision.confidence)
            hits.append(question.is_correct(decision.answer, hash_oracle(question, data)))
        m = calibration_metrics(confidences, hits)
        typer.echo(
            f"{question.key:<32} {m.accuracy:>9.3f} {m.mean_confidence:>11.3f} "
            f"{m.overconfidence:>+7.3f} {m.ece:>7.3f} {m.brier:>7.3f}"
        )


OptionalCatalogPath = Annotated[
    Path | None,
    typer.Option("--questions", "-q", help="Question catalog. Defaults to DGATE_QUESTIONS_FILE."),
]


def _catalog_for_ingest(path: Path | None) -> QuestionCatalog:
    from decision_gate.config import Settings

    path = path or Settings().questions_file
    if path is None:
        typer.echo("error: pass --questions or set DGATE_QUESTIONS_FILE", err=True)
        raise typer.Exit(code=2)
    try:
        return load_catalog(path)
    except CatalogError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=1) from exc


async def _ingest(
    database_url: str, catalog: QuestionCatalog, source: OutcomeSource
) -> IngestStats:
    from decision_gate.db import make_engine, make_sessionmaker
    from decision_gate.db.repository import DecisionRepository, OutcomeRepository
    from decision_gate.ingest import OutcomeIngestor
    from decision_gate.service import OutcomeService

    engine = make_engine(database_url)
    try:
        sessions = make_sessionmaker(engine)
        service = OutcomeService(catalog, DecisionRepository(sessions), OutcomeRepository(sessions))
        return await OutcomeIngestor(service, source).run()
    finally:
        await engine.dispose()


def _report(stats: IngestStats) -> None:
    rejected = sum(stats.rejected.values())
    typer.echo(f"created {stats.created}, duplicate {stats.duplicate}, rejected {rejected}")
    for reason, count in sorted(stats.rejected.items()):
        typer.echo(f"  {reason}: {count}")


@outcomes_app.command("ingest")
def outcomes_ingest(
    path: Annotated[str, typer.Argument(help="JSON Lines file of outcome events, or '-'.")],
    questions: OptionalCatalogPath = None,
    url: DatabaseURL = None,
    dead_letter: Annotated[
        Path | None,
        typer.Option(help="Append rejected events here (JSON Lines, with the reason)."),
    ] = None,
    strict: Annotated[
        bool, typer.Option(help="Exit with status 1 if any event was rejected.")
    ] = False,
) -> None:
    """Record outcome events from a JSON Lines file (a backfill, or a demo).

    Each line is {"decision_id": ..., "outcome": ..., "source"?: ..., "observed_at"?: ...}.
    Re-running the same file is safe: outcomes already recorded count as duplicates.
    """
    from decision_gate.ingest import JsonLinesSource, RejectReason, SourceMessage

    catalog = _catalog_for_ingest(questions)

    def warn(message: SourceMessage, reason: RejectReason, detail: str) -> None:
        typer.echo(f"{message.position}: {reason}: {detail}", err=True)

    if path == "-":
        source = JsonLinesSource(sys.stdin, dead_letter_path=dead_letter, on_reject=warn)
        stats = asyncio.run(_ingest(_database_url(url), catalog, source))
    else:
        try:
            stream = Path(path).open(encoding="utf-8")  # noqa: SIM115 - closed below
        except OSError as exc:
            typer.echo(f"error: {exc}", err=True)
            raise typer.Exit(code=1) from exc
        with stream:
            source = JsonLinesSource(stream, dead_letter_path=dead_letter, on_reject=warn)
            stats = asyncio.run(_ingest(_database_url(url), catalog, source))
    _report(stats)
    if strict and stats.rejected:
        raise typer.Exit(code=1)


@outcomes_app.command("consume")
def outcomes_consume(
    questions: OptionalCatalogPath = None,
) -> None:  # pragma: no cover - needs a Kafka broker; the source is unit-tested
    """Consume outcome events from Kafka until interrupted (needs the 'kafka' extra).

    Reads DGATE_KAFKA_BOOTSTRAP_SERVERS, DGATE_KAFKA_OUTCOMES_TOPIC, DGATE_KAFKA_GROUP_ID
    and DGATE_KAFKA_DEAD_LETTER_TOPIC.
    """
    from decision_gate.config import Settings
    from decision_gate.kafka import from_settings

    settings = Settings()
    catalog = _catalog_for_ingest(questions)

    async def run() -> IngestStats:
        async with from_settings(settings) as source:
            return await _ingest(settings.database_url, catalog, source)

    try:
        _report(asyncio.run(run()))
    except KeyboardInterrupt:
        typer.echo("stopped")


async def _calibration_reports(
    database_url: str,
    *,
    question: str | None,
    version: int | None,
    group_by: list[GroupField],
    n_bins: int,
    strategy: BinStrategy,
    since: datetime | None,
    until: datetime | None,
    bootstrap: int,
    min_count: int,
) -> list[CalibrationReport]:
    from decision_gate.calibration import build_reports
    from decision_gate.db import make_engine, make_sessionmaker
    from decision_gate.db.repository import OutcomeRepository

    engine = make_engine(database_url)
    try:
        repository = OutcomeRepository(make_sessionmaker(engine))
        rows = await repository.labeled_rows(
            question_id=question, question_version=version, since=since, until=until
        )
    finally:
        await engine.dispose()
    return build_reports(
        rows,
        group_by=group_by,
        n_bins=n_bins,
        strategy=strategy,
        bootstrap=bootstrap,
        min_count=min_count,
    )


def _fmt(value: float | None, spec: str = ".3f") -> str:
    return "-" if value is None else format(value, spec)


def _print_report_table(reports: list[CalibrationReport], group_by: list[GroupField]) -> None:
    group_header = "".join(f" {field:<14}" for field in group_by)
    typer.echo(
        f"{'question':<28}{group_header} {'n':>6} {'acc':>6} {'conf':>6} {'ECE':>6} "
        f"{'ECE 95%':>13} {'MCE':>6} {'Brier':>6} {'logloss':>7}"
    )
    for r in reports:
        m = r.metrics
        groups = "".join(
            f" {(r.backend if field == 'backend' else r.model_version) or '':<14.14}"
            for field in group_by
        )
        interval = f"{m.ece_interval[0]:.3f}-{m.ece_interval[1]:.3f}" if m.ece_interval else "-"
        typer.echo(
            f"{r.question_key:<28.28}{groups} {m.count:>6} {m.accuracy:>6.3f} "
            f"{m.mean_confidence:>6.3f} {m.ece:>6.3f} {interval:>13} {m.mce:>6.3f} "
            f"{m.brier:>6.3f} {m.log_loss:>7.3f}"
        )


def _print_bins(report: CalibrationReport) -> None:
    label = " ".join(x for x in (report.question_key, report.backend, report.model_version) if x)
    typer.echo(f"\n{label}: reliability ({report.metrics.strategy} bins)")
    typer.echo(f"  {'bin':<13} {'n':>6} {'conf':>6} {'acc':>6} {'gap':>7}  accuracy")
    for b in report.metrics.bins:
        bar = "#" * round((b.accuracy or 0.0) * 20)
        typer.echo(
            f"  {b.lower:.2f}-{b.upper:.2f}{'':<4} {b.count:>6} {_fmt(b.mean_confidence):>6} "
            f"{_fmt(b.accuracy):>6} {_fmt(b.gap, '+.3f'):>7}  {bar}".rstrip()
        )


@calibration_app.command("report")
def calibration_report(
    question: Annotated[str | None, typer.Option(help="Only this question id.")] = None,
    version: Annotated[int | None, typer.Option(min=1, help="Only this question version.")] = None,
    by: Annotated[
        list[str] | None,
        typer.Option("--by", help="Also split by 'backend' and/or 'model_version'."),
    ] = None,
    bins: Annotated[int, typer.Option(min=1, max=100, help="Number of bins.")] = 10,
    strategy: Annotated[
        str, typer.Option(help="'uniform' (equal width) or 'quantile' (equal count) bins.")
    ] = "uniform",
    since: Annotated[
        datetime | None, typer.Option(help="Decisions made at or after this time (UTC).")
    ] = None,
    until: Annotated[
        datetime | None, typer.Option(help="Decisions made before this time (UTC).")
    ] = None,
    bootstrap: Annotated[
        int, typer.Option(min=0, max=10_000, help="Resamples for a 95% ECE interval.")
    ] = 0,
    min_count: Annotated[int, typer.Option(min=1, help="Skip smaller groups.")] = 1,
    show_bins: Annotated[
        bool, typer.Option("--show-bins", help="Print each reliability table.")
    ] = False,
    as_json: Annotated[bool, typer.Option("--json", help="Print JSON instead.")] = False,
    url: DatabaseURL = None,
) -> None:
    """Calibration metrics over labeled decisions, per question version."""
    from decision_gate.calibration import GROUP_FIELDS

    fields: list[GroupField] = []
    for field in by or []:
        if field not in GROUP_FIELDS:
            typer.echo(f"error: --by must be one of {list(GROUP_FIELDS)}", err=True)
            raise typer.Exit(code=2)
        if field not in fields:
            fields.append(field)
    if strategy not in ("uniform", "quantile"):
        typer.echo("error: --strategy must be 'uniform' or 'quantile'", err=True)
        raise typer.Exit(code=2)
    reports = asyncio.run(
        _calibration_reports(
            _database_url(url),
            question=question,
            version=version,
            group_by=fields,
            n_bins=bins,
            strategy=strategy,  # type: ignore[arg-type]
            since=_as_utc(since),
            until=_as_utc(until),
            bootstrap=bootstrap,
            min_count=min_count,
        )
    )
    if as_json:
        from decision_gate.app import CalibrationReportOut

        payload = [CalibrationReportOut.from_report(r).model_dump(mode="json") for r in reports]
        typer.echo(json.dumps({"group_by": fields, "reports": payload}, indent=2))
        return
    if not reports:
        typer.echo("no labeled decisions match")
        return
    _print_report_table(reports, fields)
    if show_bins:
        for report in reports:
            _print_bins(report)


def _as_utc(moment: datetime | None) -> datetime | None:
    if moment is None or moment.tzinfo is not None:
        return moment
    return moment.replace(tzinfo=UTC)


@app.command("fake-calibrate")
def fake_calibrate(
    questions: CatalogPath,
    samples: Annotated[
        int, typer.Option("--samples", "-n", min=20, help="Inputs per question.")
    ] = 5000,
    overconfidence: Annotated[
        float, typer.Option(help="Fake backend overconfidence (1.0 = calibrated).")
    ] = 2.0,
    holdout: Annotated[float, typer.Option(min=0.05, max=0.95, help="Held-out fraction.")] = 0.3,
) -> None:
    """Fit every calibrator on the fake backend and compare them on held-out inputs.

    Needs no database: decisions and ground truth come from the fake backend, the first
    part of the inputs trains and the rest is held out.
    """
    try:
        catalog = load_catalog(questions)
    except CatalogError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    from decision_gate.calibrators import fit_calibrator
    from decision_gate.db.repository import LabeledRow
    from decision_gate.refit import calibration_data, candidate_methods, score

    backend = FakeBackend(overconfidence=overconfidence)
    n_train = samples - max(1, round(samples * holdout))
    typer.echo(
        f"{'question':<32} {'method':<12} {'ECE':>7} {'Brier':>7} {'logloss':>8}  parameters"
    )
    for question in catalog.questions:
        rows: list[LabeledRow] = []
        for i in range(samples):
            data = {"item": i}
            decision = backend.decide_sync(question, data)
            truth = hash_oracle(question, data)
            rows.append(
                LabeledRow(
                    question_id=question.id,
                    question_version=question.version,
                    backend=decision.backend,
                    model_version=decision.model_version,
                    confidence=decision.confidence,
                    correct=question.is_correct(decision.answer, truth),
                    probabilities=decision.probabilities,
                    outcome=truth,
                    answer=decision.answer,
                )
            )
        everything = calibration_data(rows)
        train = everything.subset(slice(0, n_train))
        test = everything.subset(slice(n_train, None))
        raw = score(test.confidences, test.correct)
        typer.echo(
            f"{question.key:<32} {'raw':<12} {raw['ece']:>7.3f} {raw['brier']:>7.3f} "
            f"{raw['log_loss']:>8.4f}"
        )
        for method in candidate_methods("auto", train):
            calibrator = fit_calibrator(method, train)
            m = score(calibrator.predict(test), test.correct)
            params = calibrator.params()
            shown = (
                f"{len(params['x'])} knots"
                if method == "isotonic"
                else " ".join(f"{k}={v:.3f}" for k, v in params.items())
            )
            typer.echo(
                f"{'':<32} {method:<12} {m['ece']:>7.3f} {m['brier']:>7.3f} "
                f"{m['log_loss']:>8.4f}  {shown}"
            )


async def _with_refit_job(database_url: str, work: Any) -> Any:
    from decision_gate.db import make_engine, make_sessionmaker
    from decision_gate.db.repository import CalibratorRepository, OutcomeRepository
    from decision_gate.refit import RefitJob

    engine = make_engine(database_url)
    try:
        sessions = make_sessionmaker(engine)
        calibrators = CalibratorRepository(sessions)
        return await work(RefitJob(OutcomeRepository(sessions), calibrators), calibrators)
    finally:
        await engine.dispose()


def _print_refit(result: RefitResult) -> None:
    line = f"{result.question_key:<28} {result.outcome:<10} n={result.labeled:<6} {result.reason}"
    if result.record is not None:
        line += f"  [v{result.record.version} {result.record.id}]"
    typer.echo(line)


@calibrators_app.command("refit")
def calibrators_refit(
    question: Annotated[
        str | None, typer.Argument(help="Question id. Omit with --all to refit everything.")
    ] = None,
    version: Annotated[
        int | None, typer.Option(min=1, help="Question version (default: latest in catalog).")
    ] = None,
    refit_all: Annotated[
        bool, typer.Option("--all", help="Refit every question version with labels.")
    ] = False,
    method: Annotated[str, typer.Option(help="auto, platt, isotonic or temperature.")] = "auto",
    holdout: Annotated[
        float, typer.Option(min=0.01, max=0.99, help="Newest fraction held out.")
    ] = 0.3,
    min_count: Annotated[int, typer.Option(min=2, help="Skip with fewer labels.")] = 100,
    since: Annotated[
        datetime | None, typer.Option(help="Decisions made at or after this time (UTC).")
    ] = None,
    until: Annotated[
        datetime | None, typer.Option(help="Decisions made before this time (UTC).")
    ] = None,
    activate: Annotated[
        bool, typer.Option("--activate/--no-activate", help="Activate a fit that helps.")
    ] = True,
    questions: OptionalCatalogPath = None,
    as_json: Annotated[bool, typer.Option("--json", help="Print JSON instead.")] = False,
    url: DatabaseURL = None,
) -> None:
    """Fit calibrators on labeled decisions; activate the best one if it beats the rest."""
    from decision_gate.calibrators import CalibrationError
    from decision_gate.refit import METHOD_CHOICES

    if method not in METHOD_CHOICES:
        typer.echo(f"error: --method must be one of {list(METHOD_CHOICES)}", err=True)
        raise typer.Exit(code=2)
    if (question is None) == (not refit_all):
        typer.echo("error: give a question id or --all (not both)", err=True)
        raise typer.Exit(code=2)
    if question is not None and version is None:
        catalog = _catalog_for_ingest(questions)
        try:
            version = catalog.get(question).version
        except KeyError as exc:
            typer.echo(f"error: {exc.args[0]}", err=True)
            raise typer.Exit(code=1) from exc
    options: dict[str, Any] = {
        "method": method,
        "holdout": holdout,
        "min_count": min_count,
        "since": _as_utc(since),
        "until": _as_utc(until),
        "activate": activate,
    }

    async def work(job: Any, _calibrators: Any) -> list[RefitResult]:
        if question is None:
            results: list[RefitResult] = await job.refit_all(**options)
            return results
        assert version is not None
        return [await job.refit(question, version, **options)]

    try:
        results: list[RefitResult] = asyncio.run(_with_refit_job(_database_url(url), work))
    except CalibrationError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    if as_json:
        from decision_gate.app import RefitOut

        payload = [RefitOut.from_result(r).model_dump(mode="json") for r in results]
        typer.echo(json.dumps({"results": payload}, indent=2))
        return
    if not results:
        typer.echo("no labeled decisions")
        return
    for result in results:
        _print_refit(result)


@calibrators_app.command("list")
def calibrators_list(
    question: Annotated[str | None, typer.Option(help="Only this question id.")] = None,
    status: Annotated[
        str | None, typer.Option(help="active, candidate, rejected or retired.")
    ] = None,
    url: DatabaseURL = None,
) -> None:
    """The calibrator registry, newest fit first within each question version."""

    async def work(_job: Any, calibrators: Any) -> Any:
        return await calibrators.find(question_id=question, status=status)

    records = asyncio.run(_with_refit_job(_database_url(url), work))
    if not records:
        typer.echo("no calibrators")
        return
    typer.echo(
        f"{'question':<28} {'ver':>4} {'method':<12} {'status':<10} {'train':>6} "
        f"{'holdout':>7} {'raw LL':>8} {'cal LL':>8} {'cal ECE':>8}  id"
    )
    for r in records:
        raw, cal = r.evaluation.get("raw", {}), r.evaluation.get("calibrated", {})
        typer.echo(
            f"{r.question_key:<28.28} {r.version:>4} {r.method:<12} {r.status:<10} "
            f"{r.train_count:>6} {r.holdout_count:>7} {_fmt(raw.get('log_loss'), '.4f'):>8} "
            f"{_fmt(cal.get('log_loss'), '.4f'):>8} {_fmt(cal.get('ece')):>8}  {r.id}"
        )


@calibrators_app.command("show")
def calibrators_show(
    calibrator_id: Annotated[str, typer.Argument(help="Calibrator id.")],
    url: DatabaseURL = None,
) -> None:
    """Print one calibrator as JSON, with its parameters and held-out evaluation."""
    import uuid

    from decision_gate.app import CalibratorOut

    try:
        key = uuid.UUID(calibrator_id)
    except ValueError as exc:
        typer.echo("error: not a calibrator id", err=True)
        raise typer.Exit(code=2) from exc

    async def work(_job: Any, calibrators: Any) -> Any:
        return await calibrators.get(key)

    record = asyncio.run(_with_refit_job(_database_url(url), work))
    if record is None:
        typer.echo("error: unknown calibrator", err=True)
        raise typer.Exit(code=1)
    typer.echo(json.dumps(CalibratorOut.from_record(record).model_dump(mode="json"), indent=2))


@calibrators_app.command("activate")
def calibrators_activate(
    calibrator_id: Annotated[str, typer.Argument(help="Calibrator id.")],
    url: DatabaseURL = None,
) -> None:
    """Make a stored calibrator the active one (e.g. roll back); retires the current one."""
    import uuid

    try:
        key = uuid.UUID(calibrator_id)
    except ValueError as exc:
        typer.echo("error: not a calibrator id", err=True)
        raise typer.Exit(code=2) from exc

    from decision_gate.db.repository import ActivationConflictError

    async def work(_job: Any, calibrators: Any) -> Any:
        return await calibrators.activate(key)

    try:
        record = asyncio.run(_with_refit_job(_database_url(url), work))
    except ActivationConflictError as exc:  # pragma: no cover - needs a concurrent race
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    if record is None:
        typer.echo("error: unknown calibrator", err=True)
        raise typer.Exit(code=1)
    typer.echo(f"activated {record.question_key} v{record.version} ({record.method}) {record.id}")


async def _with_analyzer(database_url: str, work: Any) -> Any:
    from decision_gate.db import make_engine, make_sessionmaker
    from decision_gate.db.repository import CalibratorRepository, OutcomeRepository
    from decision_gate.selection import GateAnalyzer

    engine = make_engine(database_url)
    try:
        sessions = make_sessionmaker(engine)
        analyzer = GateAnalyzer(OutcomeRepository(sessions), CalibratorRepository(sessions))
        return await work(analyzer)
    finally:
        await engine.dispose()


def _question(question_id: str, version: int | None, questions: Path | None) -> Any:
    catalog = _catalog_for_ingest(questions)
    try:
        return catalog.get(question_id, version)
    except KeyError as exc:
        typer.echo(f"error: {exc.args[0]}", err=True)
        raise typer.Exit(code=1) from exc


HoldoutOption = Annotated[
    float, typer.Option(min=0.0, max=0.95, help="Newest fraction held out to check the result.")
]
SinceOption = Annotated[
    datetime | None, typer.Option(help="Decisions made at or after this time (UTC).")
]
UntilOption = Annotated[
    datetime | None, typer.Option(help="Decisions made before this time (UTC).")
]
CalibratorOption = Annotated[
    bool,
    typer.Option("--calibrator/--raw", help="Score with the active calibrator, or raw scores."),
]


def _basis_line(basis: Any) -> str:
    line = (
        f"{basis.question_key}: {basis.labeled} labeled, scores from {basis.score_source}; "
        f"{basis.selection_count} used, {basis.holdout_count} held out"
    )
    if basis.excluded_training:
        line += f", {basis.excluded_training} skipped (calibrator training data)"
    if basis.reused_training:
        line += f", {basis.reused_training} reused (the calibrator preserves their order)"
    return line


@thresholds_app.command("select")
def thresholds_select(
    question: Annotated[str, typer.Argument(help="Question id.")],
    version: Annotated[
        int | None, typer.Option(min=1, help="Question version (default: latest in catalog).")
    ] = None,
    target_precision: Annotated[
        float | None,
        typer.Option(min=0.0, max=1.0, help="Minimum precision of automated decisions."),
    ] = None,
    max_error_rate: Annotated[
        float | None, typer.Option(min=0.0, max=1.0, help="Alternative: 1 - precision.")
    ] = None,
    confidence: Annotated[float, typer.Option(min=0.5, max=0.9999)] = 0.95,
    bound: Annotated[str, typer.Option(help="clopper-pearson or wilson.")] = "clopper-pearson",
    min_accepted: Annotated[
        int, typer.Option(min=1, help="Fewest decisions a candidate must accept.")
    ] = 100,
    holdout: HoldoutOption = 0.3,
    since: SinceOption = None,
    until: UntilOption = None,
    use_calibrator: CalibratorOption = True,
    curve_points: Annotated[
        int, typer.Option("--curve", min=0, help="Curve rows to show (JSON: 10 by default).")
    ] = 0,
    questions: OptionalCatalogPath = None,
    as_json: Annotated[bool, typer.Option("--json", help="Print JSON instead.")] = False,
    url: DatabaseURL = None,
) -> None:
    """Pick the lowest threshold whose precision meets the target with confidence."""
    from pydantic import ValidationError

    from decision_gate.gating_api import ThresholdOut, ThresholdSelectIn
    from decision_gate.selection import SelectionError

    try:
        request = ThresholdSelectIn(
            question=question,
            version=version,
            target_precision=target_precision,
            max_error_rate=max_error_rate,
            confidence=confidence,
            bound=bound,
            min_accepted=min_accepted,
            holdout=holdout,
            since=_as_utc(since),
            until=_as_utc(until),
            use_calibrator=use_calibrator,
            curve_points=(curve_points or 10) if as_json else curve_points,
        )
    except ValidationError as exc:
        typer.echo(f"error: {exc.errors()[0]['msg']}", err=True)
        raise typer.Exit(code=2) from exc
    target = _question(question, version, questions)

    async def work(analyzer: Any) -> Any:
        return await analyzer.thresholds(
            target,
            target_precision=request.target,
            confidence=request.confidence,
            bound=request.bound,
            min_accepted=request.min_accepted,
            holdout=request.holdout,
            since=request.since,
            until=request.until,
            use_calibrator=request.use_calibrator,
        )

    try:
        report = asyncio.run(_with_analyzer(_database_url(url), work))
    except SelectionError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    out = ThresholdOut.from_report(report, request)
    if as_json:
        typer.echo(out.model_dump_json(indent=2))
        return
    typer.echo(_basis_line(report.basis))
    typer.echo(out.reason)
    if out.holdout is not None:
        precision = _fmt(out.holdout.precision, ".4f")
        typer.echo(
            f"held out: {out.holdout.accepted}/{out.holdout.total} accepted "
            f"({out.holdout.coverage:.1%}), precision {precision}"
        )
    if out.curve:
        typer.echo(
            f"{'threshold':>10} {'accepted':>9} {'coverage':>9} {'precision':>10} {'bound':>8}"
        )
        for p in out.curve:
            mark = "  <- selected" if out.selected and p.threshold == out.selected.threshold else ""
            typer.echo(
                f"{p.threshold:>10.4f} {p.accepted:>9} {p.coverage:>9.1%} "
                f"{p.precision:>10.4f} {p.lower_bound:>8.4f}{mark}"
            )


@conformal_app.command("fit")
def conformal_fit(
    question: Annotated[str, typer.Argument(help="Question id (choice or predicate).")],
    version: Annotated[
        int | None, typer.Option(min=1, help="Question version (default: latest in catalog).")
    ] = None,
    alpha: Annotated[float, typer.Option(min=0.001, max=0.999, help="Allowed miss rate.")] = 0.1,
    method: Annotated[str, typer.Option(help="lac or aps.")] = "lac",
    holdout: HoldoutOption = 0.3,
    since: SinceOption = None,
    until: UntilOption = None,
    use_calibrator: CalibratorOption = True,
    questions: OptionalCatalogPath = None,
    as_json: Annotated[bool, typer.Option("--json", help="Print JSON instead.")] = False,
    url: DatabaseURL = None,
) -> None:
    """Calibrate prediction sets that contain the true label with probability >= 1 - alpha."""
    from pydantic import ValidationError

    from decision_gate.gating_api import ConformalFitIn, ConformalOut
    from decision_gate.selection import SelectionError

    try:
        request = ConformalFitIn(
            question=question,
            version=version,
            alpha=alpha,
            method=method,
            holdout=holdout,
            since=_as_utc(since),
            until=_as_utc(until),
            use_calibrator=use_calibrator,
        )
    except ValidationError as exc:
        typer.echo(f"error: {exc.errors()[0]['msg']}", err=True)
        raise typer.Exit(code=2) from exc
    target = _question(question, version, questions)

    async def work(analyzer: Any) -> Any:
        return await analyzer.conformal(
            target,
            alpha=request.alpha,
            method=request.method,
            holdout=request.holdout,
            since=request.since,
            until=request.until,
            use_calibrator=request.use_calibrator,
        )

    try:
        report = asyncio.run(_with_analyzer(_database_url(url), work))
    except SelectionError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    out = ConformalOut.from_report(report, request)
    if as_json:
        typer.echo(out.model_dump_json(indent=2))
        return
    typer.echo(_basis_line(report.basis))
    typer.echo(out.reason)
    if out.holdout is not None:
        h = out.holdout
        sizes = " ".join(f"{size}:{count}" for size, count in h.size_counts.items())
        typer.echo(
            f"held out: coverage {h.coverage:.3f} (target {out.target_coverage:.3f}), "
            f"mean set size {h.mean_size:.2f}, singletons {h.singleton_rate:.1%} "
            f"(accuracy {_fmt(h.singleton_accuracy)}), empty {h.empty_rate:.1%}; sizes {sizes}"
        )


@app.command("fake-gate")
def fake_gate(
    questions: CatalogPath,
    samples: Annotated[
        int, typer.Option("--samples", "-n", min=100, help="Inputs per question.")
    ] = 6000,
    overconfidence: Annotated[
        float, typer.Option(help="Fake backend overconfidence (1.0 = calibrated).")
    ] = 2.0,
    target_precision: Annotated[float, typer.Option(min=0.5, max=0.999)] = 0.95,
    alpha: Annotated[float, typer.Option(min=0.01, max=0.5)] = 0.1,
) -> None:
    """Thresholds and conformal sets on the fake backend, with no database.

    Inputs are split in order: 40% fit a calibrator (temperature scaling when the backend
    returns distributions, Platt otherwise), 30% select the threshold and calibrate the
    conformal quantile, and the last 30% check what they achieve.
    """
    try:
        catalog = load_catalog(questions)
    except CatalogError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    import numpy as np

    from decision_gate.backends.base import answer_key
    from decision_gate.calibrators import Method, fit_calibrator
    from decision_gate.conformal import distribution_matrix, evaluate, fit_conformal
    from decision_gate.db.repository import LabeledRow
    from decision_gate.questions import ChoiceQuestion, ScoreQuestion
    from decision_gate.refit import calibration_data
    from decision_gate.selection import calibrated_distributions
    from decision_gate.thresholds import apply_threshold, select_threshold

    backend = FakeBackend(overconfidence=overconfidence)
    a, b = int(samples * 0.4), int(samples * 0.7)
    for question in catalog.questions:
        rows: list[LabeledRow] = []
        for i in range(samples):
            data = {"item": i}
            decision = backend.decide_sync(question, data)
            truth = hash_oracle(question, data)
            rows.append(
                LabeledRow(
                    question_id=question.id,
                    question_version=question.version,
                    backend=decision.backend,
                    model_version=decision.model_version,
                    confidence=decision.confidence,
                    correct=question.is_correct(decision.answer, truth),
                    probabilities=decision.probabilities,
                    outcome=truth,
                    answer=decision.answer,
                )
            )
        everything = calibration_data(rows)
        train = everything.subset(slice(0, a))
        method: Method = "temperature" if train.has_distributions else "platt"
        calibrator = fit_calibrator(method, train)
        typer.echo(f"{question.key}  (calibrator: {method}, target precision {target_precision})")
        for name, scores in (
            ("raw", everything.confidences),
            ("calibrated", calibrator.predict(everything)),
        ):
            selection = select_threshold(
                scores[a:b], everything.correct[a:b], target_precision=target_precision
            )
            if selection.threshold is None:
                typer.echo(f"  {name:<11} no threshold: {selection.reason}")
                continue
            check = apply_threshold(selection.threshold, scores[b:], everything.correct[b:])
            typer.echo(
                f"  {name:<11} threshold {selection.threshold:.4f}  "
                f"selection: {selection.selected.coverage:.1%} automated, "  # type: ignore[union-attr]
                f"bound {selection.selected.lower_bound:.4f}  "  # type: ignore[union-attr]
                f"check: {check.coverage:.1%} automated, precision {_fmt(check.precision, '.4f')}"
            )
        if isinstance(question, ScoreQuestion):
            continue
        labels = question.options if isinstance(question, ChoiceQuestion) else ("false", "true")
        index = {label: i for i, label in enumerate(labels)}
        matrix = distribution_matrix(calibrated_distributions(rows, calibrator), labels)
        truths = np.array([index[answer_key(row.outcome)] for row in rows], dtype=np.intp)
        for conformal_method in ("lac", "aps"):
            predictor = fit_conformal(
                matrix[a:b], truths[a:b], labels, alpha=alpha, method=conformal_method
            )
            result = evaluate(predictor, matrix[b:], truths[b:])
            typer.echo(
                f"  {conformal_method:<11} sets for {1 - alpha:.0%} coverage: "
                f"quantile {predictor.quantile:.4f}  check: coverage {result.coverage:.3f}, "
                f"mean size {result.mean_size:.2f}, singletons {result.singleton_rate:.1%}"
            )
