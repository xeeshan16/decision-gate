"""Migrations must produce exactly the schema the ORM models describe."""

from __future__ import annotations

import asyncio
from typing import Any

from alembic.autogenerate import compare_metadata
from alembic.migration import MigrationContext
from sqlalchemy import inspect
from sqlalchemy.engine import Connection
from typer.testing import CliRunner

from decision_gate.cli import app as cli
from decision_gate.db import make_engine
from decision_gate.db.models import Base

runner = CliRunner()


def run_sync(database_url: str, fn: Any) -> Any:
    async def run() -> Any:
        engine = make_engine(database_url)
        try:
            async with engine.connect() as connection:
                return await connection.run_sync(fn)
        finally:
            await engine.dispose()

    return asyncio.run(run())


def schema_diff(connection: Connection) -> list[Any]:
    return list(compare_metadata(MigrationContext.configure(connection), Base.metadata))


def table_names(connection: Connection) -> list[str]:
    return sorted(inspect(connection).get_table_names())


def test_migrations_match_models(database_url: str) -> None:
    assert run_sync(database_url, schema_diff) == []


def test_cli_downgrade_and_upgrade(database_url: str) -> None:
    assert "decisions" in run_sync(database_url, table_names)

    result = runner.invoke(cli, ["db", "downgrade", "base", "--url", database_url])
    assert result.exit_code == 0, result.output
    assert "decisions" not in run_sync(database_url, table_names)

    result = runner.invoke(cli, ["db", "upgrade", "--url", database_url])
    assert result.exit_code == 0, result.output
    assert "decisions" in run_sync(database_url, table_names)

    result = runner.invoke(cli, ["db", "current", "--url", database_url])
    assert result.exit_code == 0, result.output


def test_cli_reads_database_url_from_env(sqlite_url: str, monkeypatch: Any) -> None:
    monkeypatch.setenv("DGATE_DATABASE_URL", sqlite_url)
    result = runner.invoke(cli, ["db", "current"])
    assert result.exit_code == 0, result.output
