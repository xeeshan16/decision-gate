# Progress log

## Day 1 (2026-10-29): scaffold, question schema, fake backend

**Done**

- Project layout (`src/decision_gate`), `pyproject.toml` with uv, hatchling build, ruff,
  strict mypy and pytest with coverage. MIT license and `.gitignore`.
- Question schema (`questions.py`) for the three kinds: `predicate` (bool), `choice` (one of
  a fixed option set) and `score` (a number in a range, correct within a tolerance). Each
  question is keyed by `id@vN`, because calibration is per question version. Each type
  validates answers and decides whether an answer matches an outcome, so every later stage
  can treat all three as "proposed answer + probability it is correct".
- Catalog loader for YAML or JSON with clear errors; `examples/questions.yaml` has a
  refund-eligibility, a ticket-queue and a handling-time question.
- `DecisionBackend` protocol and the `RawDecision` model (answer, reported confidence,
  full distribution for predicate and choice questions).
- Deterministic `FakeBackend`: hashed per-item skill, answers correct with exactly that
  probability, and an `overconfidence` knob that distorts the reported confidence in
  log-odds space around the chance level. A pluggable oracle supplies ground truth, so the
  day 11 simulator can drive it with its own datasets.
- FastAPI app factory with `/healthz`, `/readyz`, `GET /v1/questions` and
  `GET /v1/questions/{id}`; `DGATE_*` settings.
- `dgate` CLI: `version`, `serve`, `questions check`, `ask` (one raw decision from the fake)
  and `fake-gap` (measured accuracy against mean reported confidence).
- GitHub Actions CI on Python 3.12 and 3.13: ruff lint and format check, mypy strict,
  pytest with a 95% coverage floor, `uv build` and a CLI smoke test.

**Tests:** 90 passing locally on Python 3.13 and 3.12, 100% line and branch coverage.
The fake backend's statistical behaviour is tested: with `overconfidence=1.0` its mean
confidence matches its accuracy within 0.03 over 4,000 items; at 3.0 it overstates by
more than 0.08; below 1.0 it understates.

**Next (day 2):** `POST /v1/decide`, backend adapters (fake, optional OpenAI Decisions,
generic LLM-JSON), PostgreSQL schema with Alembic, and integration tests against a
Postgres service container in CI.

## Day 2 (2026-10-30): decide API, backend adapters, PostgreSQL

**Done**

- `POST /v1/decide`: resolves the question (latest version unless pinned), calls the
  backend under a time limit, validates the answer against the question, stores the
  decision and returns it. `GET /v1/decisions/{id}` reads one back. Errors map to clear
  statuses: `404` unknown question, `409` idempotency key reused for a different
  question or input, `502` backend failure or invalid answer, `504` timeout.
- Idempotency: a key in the body or the `Idempotency-Key` header makes retries return the
  original decision (`200`) instead of a new one (`201`). Concurrent retries are settled
  by a unique constraint, not by a check-then-insert race; the losers read the winner's row.
- Persistence: SQLAlchemy 2 async with asyncpg, `decisions` table with JSONB answer,
  distribution and input, the input's SHA-256, backend, model version and latency, and an
  index on (question, version, time) for the calibration and drift jobs. Alembic
  migrations ship inside the package and run with `dgate db upgrade|downgrade|current`.
  `DGATE_STORE_INPUTS=false` keeps only the hash.
- Backend adapters: `http` (any JSON decision service) and `llm_json` (any
  OpenAI-compatible chat completions endpoint, asked for `{"answer", "confidence"}`,
  tolerant of a Markdown code fence). Both share one validation path, so no backend can
  store an answer that is invalid for its question. `/readyz` now checks the database.
- CI: a PostgreSQL 17 service container; the database tests run against both SQLite and
  PostgreSQL, and a new smoke step migrates the database, starts the API and checks an
  idempotent replay over HTTP.

**Plan change:** day 2 listed an "OpenAI Decisions" adapter. I did not have a verified
wire format for that API to test against, so instead of guessing at one I shipped the
generic `http` adapter, which any decision service can satisfy, and the `llm_json` adapter.
A vendor-specific adapter can be added behind the same interface once its contract is
pinned down from the official docs; it is not needed to build, test or demo the project.

**Tests:** 171 passing locally on Python 3.13 and 3.12 with `DGATE_TEST_DATABASE_URL` set
to a local PostgreSQL 16 server (149 pass and 22 PostgreSQL variants skip without it).
100% line and branch coverage. New tests cover both adapters through a mock transport
(happy paths and 20 failure or malformed-reply cases), every API error path, that migrations produce
exactly the ORM schema (Alembic `compare_metadata` reports no diff), migrate down and up
through the CLI, and concurrent retries with one idempotency key storing exactly one row.

**Next (day 3):** idempotent `POST /v1/outcomes` for labels that arrive later, joined to
their decisions, and an optional Kafka outcome consumer behind an interface.
