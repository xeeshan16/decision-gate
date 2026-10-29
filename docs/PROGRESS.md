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
