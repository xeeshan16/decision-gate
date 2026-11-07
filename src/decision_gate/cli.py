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
routing_app = typer.Typer(
    help="Routing tiers: check policies, refresh rules, show status.", no_args_is_help=True
)
app.add_typer(routing_app, name="routing")
drift_app = typer.Typer(
    help="Drift monitoring: run checks (tightening routing on drift), list results.",
    no_args_is_help=True,
)
app.add_typer(drift_app, name="drift")
reviews_app = typer.Typer(
    help="The human review queue: list, claim, resolve and measure.", no_args_is_help=True
)
app.add_typer(reviews_app, name="reviews")
shadow_app = typer.Typer(
    help="Shadow mode: a candidate backend answers on the side; compare it before promoting.",
    no_args_is_help=True,
)
app.add_typer(shadow_app, name="shadow")

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


RoutingPath = Annotated[
    Path | None,
    typer.Option("--routing", "-r", help="Routing policies. Defaults to DGATE_ROUTING_FILE."),
]


def _routing_config(path: Path | None) -> Any:
    from decision_gate.config import Settings
    from decision_gate.routing import RoutingConfigError, load_routing

    path = path or Settings().routing_file
    if path is None:
        typer.echo("error: pass --routing or set DGATE_ROUTING_FILE", err=True)
        raise typer.Exit(code=2)
    try:
        return load_routing(path)
    except RoutingConfigError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=1) from exc


def _policy_line(policy: Any) -> str:
    line = (
        f"auto error <= {policy.max_error_rate:.1%} at {policy.confidence:.0%} confidence, "
        f"min {policy.min_labeled} labeled"
    )
    if policy.verify_max_error_rate is not None:
        line += f", verify error <= {policy.verify_max_error_rate:.1%}"
    line += f", {policy.verify_sample_rate:.0%} of auto sampled to verify"
    if policy.conformal_alpha is not None:
        line += f", answer sets at {1 - policy.conformal_alpha:.0%} coverage"
    if not policy.require_calibrator:
        line += ", raw confidence allowed"
    return line


@routing_app.command("check")
def routing_check(
    path: Annotated[Path, typer.Argument(help="Routing policy file (YAML or JSON).")],
    questions: OptionalCatalogPath = None,
) -> None:
    """Validate a routing file and show the policy each catalog question gets."""
    config = _routing_config(path)
    catalog = _catalog_for_ingest(questions)
    for question in catalog.questions:
        policy = config.policy_for(question.id)
        if policy is None:
            typer.echo(f"{question.key}: no policy (every decision goes to a human)")
        else:
            typer.echo(f"{question.key}: [{policy.fingerprint()}] {_policy_line(policy)}")
    unknown = config.unknown_questions(catalog)
    if unknown:
        typer.echo(f"error: policies for questions not in the catalog: {unknown}", err=True)
        raise typer.Exit(code=1)


async def _refresh(database_url: str, config: Any, targets: list[Any]) -> list[Any]:
    from decision_gate.db import make_engine, make_sessionmaker
    from decision_gate.db.repository import (
        CalibratorRepository,
        DriftCheckRepository,
        OutcomeRepository,
        RoutingRuleRepository,
    )
    from decision_gate.routing import RoutingRefresher
    from decision_gate.selection import GateAnalyzer

    engine = make_engine(database_url)
    try:
        sessions = make_sessionmaker(engine)
        calibrators = CalibratorRepository(sessions)
        refresher = RoutingRefresher(
            config,
            GateAnalyzer(OutcomeRepository(sessions), calibrators),
            calibrators,
            RoutingRuleRepository(sessions),
            DriftCheckRepository(sessions),
        )
        return [(await refresher.refresh(q)).rule for q in targets]
    finally:
        await engine.dispose()


def _fmt_threshold(value: float | None) -> str:
    return "none" if value is None else f"{value:.4f}"


@routing_app.command("refresh")
def routing_refresh(
    question: Annotated[
        str | None, typer.Argument(help="Question id. Omit to refresh every routed question.")
    ] = None,
    version: Annotated[int | None, typer.Option(min=1, help="Question version.")] = None,
    routing: RoutingPath = None,
    questions: OptionalCatalogPath = None,
    url: DatabaseURL = None,
) -> None:
    """Select thresholds from labeled decisions and store them as the active rules.

    A running server caches rules for DGATE_ROUTING_CACHE_TTL_S seconds; the HTTP refresh
    endpoint applies them at once.
    """
    from decision_gate.selection import SelectionError

    config = _routing_config(routing)
    catalog = _catalog_for_ingest(questions)
    if question is not None:
        targets = [_question(question, version, questions)]
        if config.policy_for(question) is None:
            typer.echo(f"error: no routing policy covers {question}", err=True)
            raise typer.Exit(code=1)
    else:
        targets = [q for q in catalog.questions if config.policy_for(q.id) is not None]
    try:
        rules = asyncio.run(_refresh(_database_url(url), config, targets))
    except SelectionError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    for rule in rules:
        typer.echo(
            f"{rule.question_key}: auto {_fmt_threshold(rule.auto_threshold)}, verify "
            f"{_fmt_threshold(rule.verify_threshold)}, "
            f"answer sets {'yes' if rule.conformal else 'no'} "
            f"({rule.eligible} of {rule.labeled} labeled decisions used)"
        )
        typer.echo(f"  {rule.reason}")


# --- Review queue ------------------------------------------------------------------------


async def _with_reviews(database_url: str, catalog: Any, work: Any) -> Any:
    from decision_gate.config import Settings
    from decision_gate.db import make_engine, make_sessionmaker
    from decision_gate.db.repository import ReviewRepository
    from decision_gate.questions import QuestionCatalog
    from decision_gate.reviews import ReviewService

    engine = make_engine(database_url)
    try:
        service = ReviewService(
            catalog or QuestionCatalog(),
            ReviewRepository(make_sessionmaker(engine)),
            Settings().review_sla,
        )
        return await work(service)
    finally:
        await engine.dispose()


def _run_review(url: str | None, work: Any, catalog: Any = None) -> Any:
    """Run one queue action; not-found and conflicts exit with status 1."""
    from decision_gate.db.repository import ReviewConflictError, ReviewNotFoundError
    from decision_gate.reviews import InvalidResolutionError

    try:
        return asyncio.run(_with_reviews(_database_url(url), catalog, work))
    except (ReviewNotFoundError, ReviewConflictError, InvalidResolutionError) as exc:
        typer.echo(f"error: {exc.args[0] if exc.args else exc}", err=True)
        raise typer.Exit(code=1) from exc


def _minutes(value: float | None) -> str:
    if value is None:
        return "-"
    if value < 120:
        return f"{value:.0f}m"
    return f"{value / 60:.1f}h"


def _item_line(item: Any, decision: Any, now: datetime) -> str:
    from decision_gate.reviews import as_utc, is_overdue, item_state

    state = item_state(item, now)
    due = as_utc(item.due_at)
    late = " OVERDUE" if is_overdue(item, now) else ""
    who = f" by {item.resolved_by or item.claimed_by}" if state in ("claimed", "resolved") else ""
    calibrated = (
        "-" if decision.calibrated_confidence is None else f"{decision.calibrated_confidence:.3f}"
    )
    line = (
        f"{item.id}  {item.question_key:<28} {item.tier:<6} {state}{who}{late}  "
        f"due {due:%Y-%m-%d %H:%M}Z  answer {json.dumps(decision.answer)} "
        f"(calibrated {calibrated})"
    )
    if item.resolution is not None:
        line += f" -> {json.dumps(item.resolution)} [{item.verdict}]"
    return line


ReviewerOption = Annotated[str, typer.Option("--reviewer", help="Who is acting.")]
ReviewIdArgument = Annotated[str, typer.Argument(help="Review item id.")]


def _item_id(raw: str) -> Any:
    import uuid

    try:
        return uuid.UUID(raw)
    except ValueError as exc:
        typer.echo(f"error: not a review item id: {raw}", err=True)
        raise typer.Exit(code=2) from exc


def _check_choice(value: str | None, allowed: tuple[str, ...], option: str) -> None:
    if value is not None and value not in allowed:
        typer.echo(f"error: {option} must be one of {list(allowed)}", err=True)
        raise typer.Exit(code=2)


@reviews_app.command("list")
def reviews_list(
    state: Annotated[
        str | None,
        typer.Option(help="available, claimed, resolved or superseded."),
    ] = None,
    question: Annotated[str | None, typer.Option(help="Only this question id.")] = None,
    tier: Annotated[str | None, typer.Option(help="verify or human.")] = None,
    overdue: Annotated[
        bool | None, typer.Option("--overdue/--on-time", help="Only late (or on-time) items.")
    ] = None,
    limit: Annotated[int, typer.Option(min=1, max=10_000)] = 50,
    url: DatabaseURL = None,
) -> None:
    """Review items, earliest deadline first."""
    from decision_gate.reviews import REVIEW_STATES, REVIEW_TIERS

    _check_choice(state, REVIEW_STATES, "--state")
    _check_choice(tier, REVIEW_TIERS, "--tier")

    async def work(service: Any) -> tuple[list[Any], datetime]:
        found = await service.find(
            state=state, question_id=question, tier=tier, overdue=overdue, limit=limit
        )
        return found, service.now()

    found, now = _run_review(url, work)
    if not found:
        typer.echo("no review items match")
        return
    for item, decision in found:
        typer.echo(_item_line(item, decision, now))


@reviews_app.command("claim")
def reviews_claim(
    reviewer: ReviewerOption,
    item: Annotated[
        str | None, typer.Argument(help="Item id. Omit to claim the most urgent one.")
    ] = None,
    question: Annotated[str | None, typer.Option(help="Only items for this question.")] = None,
    tier: Annotated[str | None, typer.Option(help="verify or human.")] = None,
    lease: Annotated[
        float | None, typer.Option(min=0.1, max=480, help="Claim length in minutes.")
    ] = None,
    url: DatabaseURL = None,
) -> None:
    """Claim an item under a lease (the most urgent one unless an id is given)."""
    from decision_gate.reviews import REVIEW_TIERS

    _check_choice(tier, REVIEW_TIERS, "--tier")
    item_id = _item_id(item) if item is not None else None

    async def work(service: Any) -> tuple[Any, datetime]:
        if item_id is not None:
            found = await service.claim(item_id, reviewer, lease_minutes=lease)
        else:
            found = await service.claim_next(
                reviewer, question_id=question, tier=tier, lease_minutes=lease
            )
        return found, service.now()

    found, now = _run_review(url, work)
    if found is None:
        typer.echo("nothing is waiting")
        return
    claimed, decision = found
    typer.echo(_item_line(claimed, decision, now))
    if decision.answer_set is not None:
        typer.echo(f"  answer set: {json.dumps(decision.answer_set)}")
    if decision.input is not None:
        typer.echo(f"  input: {json.dumps(decision.input, sort_keys=True)}")


@reviews_app.command("release")
def reviews_release(
    item: ReviewIdArgument, reviewer: ReviewerOption, url: DatabaseURL = None
) -> None:
    """Give a claimed item back to the queue."""
    item_id = _item_id(item)

    async def work(service: Any) -> Any:
        return await service.release(item_id, reviewer)

    released, _decision = _run_review(url, work)
    typer.echo(f"released {released.id}")


@reviews_app.command("resolve")
def reviews_resolve(
    item: ReviewIdArgument,
    reviewer: ReviewerOption,
    outcome: Annotated[
        str | None, typer.Option(help='The correct answer as JSON, e.g. true, "billing", 25.')
    ] = None,
    confirm: Annotated[
        bool, typer.Option("--confirm", help="The backend's answer is right.")
    ] = False,
    note: Annotated[str | None, typer.Option(help="Optional note for the audit trail.")] = None,
    questions: OptionalCatalogPath = None,
    url: DatabaseURL = None,
) -> None:
    """Resolve an item; the answer is recorded as the decision's outcome."""
    item_id = _item_id(item)
    if confirm == (outcome is not None):
        typer.echo("error: pass exactly one of --outcome or --confirm", err=True)
        raise typer.Exit(code=2)
    value: Any = None
    if outcome is not None:
        try:
            value = json.loads(outcome)
        except json.JSONDecodeError:
            value = outcome  # a bare word such as billing is a string answer
    catalog = _catalog_for_ingest(questions)

    async def work(service: Any) -> Any:
        return await service.resolve(item_id, reviewer, outcome=value, confirm=confirm, note=note)

    resolved, _decision, recorded, created = _run_review(url, work, catalog)
    typer.echo(
        f"{'resolved' if created else 'already resolved'} {resolved.id}: "
        f"{json.dumps(recorded.outcome)} [{resolved.verdict}], backend "
        f"{'right' if recorded.correct else 'wrong'}"
    )


@reviews_app.command("stats")
def reviews_stats(
    question: Annotated[str | None, typer.Option(help="Only this question id.")] = None,
    since: Annotated[
        datetime | None, typer.Option(help="Only items created at or after this time (UTC).")
    ] = None,
    as_json: Annotated[bool, typer.Option("--json", help="Print JSON instead.")] = False,
    url: DatabaseURL = None,
) -> None:
    """Backlog, SLA performance and backend agreement per question version and tier."""
    from decision_gate.reviews_api import QueueStatsOut

    async def work(service: Any) -> tuple[list[Any], Any]:
        return await service.stats(question_id=question, since=_as_utc(since)), service.sla

    stats, sla = _run_review(url, work)
    if as_json:
        payload = [QueueStatsOut.from_stats(s).model_dump(mode="json") for s in stats]
        typer.echo(json.dumps({"queues": payload}, indent=2))
        return
    if not stats:
        typer.echo("no review items")
        return
    typer.echo(
        f"SLA: verify {_minutes(sla.verify_minutes)}, human {_minutes(sla.human_minutes)}, "
        f"claims last {_minutes(sla.claim_minutes)}"
    )
    typer.echo(
        f"{'queue':<36} {'avail':>6} {'claim':>6} {'late':>5} {'oldest':>7} {'done':>6} "
        f"{'on-time':>8} {'median':>7} {'p90':>6} {'agree':>6}"
    )
    for s in stats:
        label = f"{s.question_id}@v{s.question_version} {s.tier}"
        typer.echo(
            f"{label:<36} {s.available:>6} {s.claimed:>6} {s.overdue:>5} "
            f"{_minutes(s.oldest_waiting_minutes):>7} {s.resolved:>6} "
            f"{_fmt(s.sla_hit_rate, '.1%'):>8} {_minutes(s.median_minutes):>7} "
            f"{_minutes(s.p90_minutes):>6} {_fmt(s.agreement_rate, '.1%'):>6}"
        )


@reviews_app.command("backfill")
def reviews_backfill(url: DatabaseURL = None) -> None:
    """Enqueue verify and human decisions recorded before the review queue existed."""

    async def work(service: Any) -> int:
        return int(await service.backfill())

    typer.echo(f"enqueued {_run_review(url, work)} decision(s)")


# --- Drift monitoring --------------------------------------------------------------------


async def _with_drift(database_url: str, config: Any, work: Any) -> Any:
    from decision_gate.config import Settings
    from decision_gate.db import make_engine, make_sessionmaker
    from decision_gate.db.repository import (
        CalibratorRepository,
        DecisionRepository,
        DriftCheckRepository,
        OutcomeRepository,
        RoutingRuleRepository,
    )
    from decision_gate.drift_monitor import DriftMonitor
    from decision_gate.selection import GateAnalyzer

    engine = make_engine(database_url)
    try:
        sessions = make_sessionmaker(engine)
        outcomes = OutcomeRepository(sessions)
        calibrators = CalibratorRepository(sessions)
        checks = DriftCheckRepository(sessions)
        monitor = DriftMonitor(
            config,
            decisions=DecisionRepository(sessions),
            outcomes=outcomes,
            calibrators=calibrators,
            rules=RoutingRuleRepository(sessions),
            checks=checks,
            analyzer=GateAnalyzer(outcomes, calibrators),
            alerter=Settings().drift_alerter(),
        )
        return await work(monitor, checks)
    finally:
        await engine.dispose()


def _print_check(check: Any, *, verbose: bool = True) -> None:
    from decision_gate.reviews import as_utc

    alert = f", alert {check.alert_event} {check.alert}" if check.alert else ""
    checked = as_utc(check.checked_at)
    typer.echo(
        f"{check.question_key}: {check.status.upper()} at {checked:%Y-%m-%d %H:%M}Z "
        f"(window {as_utc(check.window_start):%m-%d %H:%M} to "
        f"{as_utc(check.window_end):%m-%d %H:%M}, reference "
        f"{as_utc(check.reference_start):%m-%d %H:%M} to "
        f"{as_utc(check.reference_end):%m-%d %H:%M}); action {check.action}{alert}"
    )
    if not verbose:
        return
    for signal in check.signals:
        typer.echo(f"  {signal['name']:<10} {signal['status']:<12} {signal['message']}")
    if check.action != "none":
        typer.echo(f"  -> {check.action_reason}")
    if check.alert_error:
        typer.echo(f"  alert error: {check.alert_error}")


@drift_app.command("check")
def drift_check(
    question: Annotated[
        str | None, typer.Argument(help="Question id. Omit to check every monitored question.")
    ] = None,
    version: Annotated[int | None, typer.Option(min=1, help="Question version.")] = None,
    at: Annotated[
        str | None,
        typer.Option(help="Evaluate as of this ISO 8601 time instead of now (e.g. backfills)."),
    ] = None,
    routing: RoutingPath = None,
    questions: OptionalCatalogPath = None,
    url: DatabaseURL = None,
) -> None:
    """Measure drift; on drift, tighten routing (per the drift policy) and send alerts.

    Exits with status 3 when any question is drifting, so a scheduler can flag it.
    A running server caches rules for DGATE_ROUTING_CACHE_TTL_S seconds.
    """
    from decision_gate.selection import SelectionError

    config = _routing_config(routing)
    catalog = _catalog_for_ingest(questions)
    if question is not None:
        targets = [_question(question, version, questions)]
    else:
        targets = list(catalog.questions)
    now = None
    if at is not None:
        try:
            now = datetime.fromisoformat(at)
        except ValueError as exc:
            typer.echo(f"error: --at is not an ISO 8601 time: {at}", err=True)
            raise typer.Exit(code=2) from exc
        if now.tzinfo is None:
            now = now.replace(tzinfo=UTC)

    async def work(monitor: Any, _checks: Any) -> list[Any]:
        if question is not None:
            return [await monitor.check(targets[0], now=now)]
        return list(await monitor.check_all(targets, now=now))

    try:
        records = asyncio.run(_with_drift(_database_url(url), config, work))
    except SelectionError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    if not records:
        typer.echo("no monitored questions (drift needs a routing policy)")
        return
    for record in records:
        _print_check(record)
    if any(r.status == "drift" for r in records):
        raise typer.Exit(code=3)


@drift_app.command("list")
def drift_list(
    question: Annotated[str | None, typer.Option(help="Only this question id.")] = None,
    status: Annotated[str | None, typer.Option(help="ok, drift or insufficient.")] = None,
    limit: Annotated[int, typer.Option(min=1, max=10_000)] = 20,
    verbose: Annotated[bool, typer.Option("--verbose", "-v", help="Show every signal.")] = False,
    url: DatabaseURL = None,
) -> None:
    """Drift check history, newest first."""
    from decision_gate.routing import RoutingConfig

    _check_choice(status, ("ok", "drift", "insufficient"), "--status")

    async def work(_monitor: Any, checks: Any) -> list[Any]:
        return list(await checks.find(question_id=question, status=status, limit=limit))

    records = asyncio.run(_with_drift(_database_url(url), RoutingConfig(), work))
    if not records:
        typer.echo("no drift checks")
        return
    for record in records:
        _print_check(record, verbose=verbose)


# --- Shadow mode --------------------------------------------------------------------------


async def _with_shadow(database_url: str, work: Any, *, runner: bool = False) -> Any:
    from decision_gate.backends.factory import build_shadow_backend
    from decision_gate.config import Settings
    from decision_gate.db import make_engine, make_sessionmaker
    from decision_gate.db.shadow import ShadowRepository
    from decision_gate.shadow_runner import ShadowRunner

    engine = make_engine(database_url)
    shadow: ShadowRunner | None = None
    try:
        repository = ShadowRepository(make_sessionmaker(engine))
        if runner:
            settings = Settings()
            backend = build_shadow_backend(settings)
            if backend is None:
                typer.echo("error: set DGATE_SHADOW_BACKEND to choose a candidate", err=True)
                raise typer.Exit(code=2)
            shadow = ShadowRunner(
                backend,
                repository,
                candidate=settings.shadow_candidate or backend.name,
                timeout_s=settings.shadow_timeout_s,
            )
        return await work(repository, shadow)
    finally:
        if shadow is not None:
            await shadow.aclose()
        await engine.dispose()


def _optional_routing(path: Path | None) -> Any:
    from decision_gate.config import Settings
    from decision_gate.routing import RoutingConfig

    if path is None and Settings().routing_file is None:
        return RoutingConfig()
    return _routing_config(path)


@shadow_app.command("status")
def shadow_status(
    question: Annotated[str | None, typer.Option(help="Only this question id.")] = None,
    url: DatabaseURL = None,
) -> None:
    """Stored shadow answers per question version and candidate."""

    async def work(repository: Any, _runner: Any) -> list[Any]:
        return list(await repository.summaries(question_id=question))

    summaries = asyncio.run(_with_shadow(_database_url(url), work))
    if not summaries:
        typer.echo("no shadow answers")
        return
    typer.echo(
        f"{'question':<28} {'candidate':<20} {'total':>7} {'ok':>7} {'failed':>7} {'labeled':>8}"
    )
    for s in summaries:
        typer.echo(
            f"{s.question_id + '@v' + str(s.question_version):<28} {s.candidate:<20} "
            f"{s.total:>7} {s.ok:>7} {s.errors + s.timeouts:>7} {s.labeled:>8}"
        )


@shadow_app.command("replay")
def shadow_replay(
    question: Annotated[str, typer.Argument(help="Question id.")],
    version: Annotated[int | None, typer.Option(min=1, help="Question version.")] = None,
    limit: Annotated[int, typer.Option(min=1, max=1_000_000)] = 1000,
    labeled_only: Annotated[
        bool,
        typer.Option("--labeled-only/--all", help="Only decisions that have an outcome."),
    ] = True,
    concurrency: Annotated[int, typer.Option(min=1, max=64)] = 8,
    questions: OptionalCatalogPath = None,
    url: DatabaseURL = None,
) -> None:
    """Run the configured candidate (DGATE_SHADOW_*) over stored decisions, oldest first."""
    target = _question(question, version, questions)

    async def work(_repository: Any, runner: Any) -> Any:
        return await runner.replay(
            target, limit=limit, labeled_only=labeled_only, concurrency=concurrency
        )

    result = asyncio.run(_with_shadow(_database_url(url), work, runner=True))
    typer.echo(
        f"{result.question_id}@v{result.question_version} {result.candidate}: "
        f"{result.attempted} replayed, {result.ok} ok, {result.failed} failed, "
        f"{result.duplicates} already answered, "
        f"{result.skipped_without_input} skipped (no stored input)"
    )


def _print_side(label: str, side: Any) -> None:
    auto = side.auto
    threshold = "none" if auto.threshold is None else f"{auto.threshold:.3f}"
    typer.echo(
        f"  {label:<10} {side.method:<12} raw ece {side.raw['ece']:.3f}  "
        f"calibrated ece {side.calibrated['ece']:.3f}  brier {side.calibrated['brier']:.3f}  "
        f"accuracy {side.calibrated['accuracy']:.3f}  auto {auto.coverage:.1%} "
        f"(threshold {threshold})  p95 {_fmt(side.latency_p95_ms, '.1f')} ms"
    )


@shadow_app.command("compare")
def shadow_compare(
    question: Annotated[str, typer.Argument(help="Question id.")],
    candidate: Annotated[
        str | None,
        typer.Option(help="Candidate name. Defaults to DGATE_SHADOW_NAME or the backend type."),
    ] = None,
    version: Annotated[int | None, typer.Option(min=1, help="Question version.")] = None,
    mode: Annotated[str | None, typer.Option(help="Only 'live' or only 'replay' answers.")] = None,
    holdout: Annotated[float, typer.Option(min=0.05, max=0.95)] = 0.5,
    method: Annotated[str, typer.Option(help="auto, platt, isotonic or temperature.")] = "auto",
    min_paired: Annotated[int, typer.Option(min=40)] = 200,
    alpha: Annotated[float, typer.Option(min=0.001, max=0.5)] = 0.05,
    min_coverage_gain: Annotated[float, typer.Option(min=0.0, max=0.99)] = 0.02,
    max_failure_rate: Annotated[float, typer.Option(min=0.0, max=1.0)] = 0.01,
    as_json: Annotated[bool, typer.Option("--json", help="Print the full report as JSON.")] = False,
    routing: RoutingPath = None,
    questions: OptionalCatalogPath = None,
    url: DatabaseURL = None,
) -> None:
    """Compare a candidate with the primary backend on paired, labeled decisions.

    Exits 0 on 'promote', 3 on 'keep' and 4 on 'insufficient', so a pipeline can gate a
    rollout on it.
    """
    from decision_gate.calibrators import CalibrationError
    from decision_gate.config import Settings
    from decision_gate.refit import METHOD_CHOICES
    from decision_gate.shadow import ShadowCompareError, compare_stored
    from decision_gate.shadow_api import comparison_out

    _check_choice(mode, ("live", "replay"), "--mode")
    _check_choice(method, METHOD_CHOICES, "--method")
    target = _question(question, version, questions)
    config = _optional_routing(routing)
    name = candidate or Settings().shadow_candidate
    if name is None:
        typer.echo("error: pass --candidate or set DGATE_SHADOW_NAME", err=True)
        raise typer.Exit(code=2)

    async def work(repository: Any, _runner: Any) -> Any:
        return await compare_stored(
            repository,
            target,
            name,
            policy=config.policy_for(target.id),
            mode=mode,
            holdout=holdout,
            method=method,
            min_paired=min_paired,
            alpha=alpha,
            min_coverage_gain=min_coverage_gain,
            max_failure_rate=max_failure_rate,
        )

    try:
        result = asyncio.run(_with_shadow(_database_url(url), work))
    except (ShadowCompareError, CalibrationError) as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    if as_json:
        typer.echo(json.dumps(comparison_out(result), indent=2, default=str))
    else:
        typer.echo(
            f"{result.question_key} primary vs {result.candidate}: {result.verdict.upper()} "
            f"({result.paired} paired: {result.train} train, {result.holdout} held out)"
        )
        for reason in result.reasons:
            typer.echo(f"  - {reason}")
        if result.accuracy is not None:
            a = result.accuracy
            low, high = a.interval
            typer.echo(
                f"  accuracy delta {a.delta:+.3f} [{low:+.3f}, {high:+.3f}]  "
                f"primary-only {a.primary_only}, candidate-only {a.candidate_only}, "
                f"p={a.p_value:.3g}"
            )
        if result.primary is not None and result.challenger is not None:
            _print_side("primary", result.primary)
            _print_side("candidate", result.challenger)
        if result.reliability is not None:
            r = result.reliability
            typer.echo(f"  candidate failures {r.failures}/{r.total} ({r.rate:.1%})")
    codes = {"promote": 0, "keep": 3, "insufficient": 4}
    if codes[result.verdict]:
        raise typer.Exit(code=codes[result.verdict])
