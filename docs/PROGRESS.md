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

## Day 3 (2026-10-31): outcomes over HTTP, files and Kafka

**First: CI was red.** Day 2's run passed every check, then failed in the setup-uv
post-job step: the API server from the smoke test was still running under `uv run`,
holding uv's cache lock, so the cache prune waited five minutes and failed the job. The
smoke step now starts the installed entry point directly and kills it on exit
([3740887](https://github.com/xeeshan16/decision-gate/commit/3740887)).

**Done**

- `outcomes` table (Alembic `0002`): one row per decision, keyed by the decision id with a
  foreign key, holding the outcome, `correct` (computed once by the question's own rule),
  `source`, `observed_at` and `recorded_at`. SQLite now enforces foreign keys too, so the
  tests behave like PostgreSQL.
- `POST /v1/outcomes`: `201` new, `200` the same outcome again (an idempotent replay keeps
  the original row, source included), `404` unknown decision, `409` a different outcome
  than the one recorded, `422` not a valid answer for the question (or the question
  version has left the catalog). Concurrent duplicates are settled by the primary key.
- `POST /v1/outcomes/batch` (up to 1,000, per-item status) for backfills;
  `GET /v1/decisions/{id}` now includes the outcome; `GET /v1/outcomes/stats` reports per
  question version the label rate, raw accuracy, mean raw confidence on the same labeled
  decisions, and the gap between them.
- `OutcomeRepository.labeled(...)`: decisions joined to outcomes for one question version
  over a time window. This is the data the day 4 metrics and day 5 calibrators read.
- `OutcomeIngestor` over a pluggable `OutcomeSource`: commit after handling (at-least-once),
  dead-letter permanent rejects with a reason, and let transient errors propagate
  uncommitted. Sources: `JsonLinesSource` (`dgate outcomes ingest FILE|-`, with
  `--dead-letter` and `--strict`) and `KafkaOutcomeSource` (`dgate outcomes consume`,
  optional `kafka` extra on aiokafka) with manual next-offset commits and a dead-letter
  topic carrying reason headers.
- CI installs all extras, and the API smoke test now records an outcome over HTTP, replays
  it through the CLI as a duplicate and checks the stats count one label.

**Not done / honest gaps:** the Kafka source is tested with in-memory consumer and
producer fakes, not a real broker (no broker was available here, and I did not want to add
an unverified Kafka service to CI). A real-broker test is planned with the Docker Compose
stack on day 13.

**Tests:** 216 passing locally on Python 3.13 and 3.12 against SQLite and a local
PostgreSQL 16 server, 100% line and branch coverage (coverage now traces greenlets, which
SQLAlchemy's async layer runs callbacks in). New tests cover every outcome status code,
score tolerance scoring, UTC normalisation, batch statuses and limits, the stats
arithmetic, the labeled join and windows, five concurrent duplicate outcomes storing one
row, orphan outcomes rejected by the database, JSON Lines ingestion with every reject
reason and the dead-letter file, Kafka offsets (`offset + 1` per partition), tombstones and
dead-letter headers, a dead database propagating without a commit, and the CLI.

**Next (day 4):** calibration metrics: ECE, Brier score, log loss and reliability bins per
question, backend and version, tested against hand-computed fixtures.

