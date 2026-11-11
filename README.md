# decision-gate

[![CI](https://github.com/xeeshan16/decision-gate/actions/workflows/ci.yml/badge.svg)](https://github.com/xeeshan16/decision-gate/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.12%20%7C%203.13-blue)
![License: MIT](https://img.shields.io/badge/license-MIT-green)

**Calibrated thresholds and human-in-the-loop routing for AI decision models.**

decision-gate sits between your application and any model that answers a question with a
probability: a hosted decision API, a prompted LLM that returns JSON, or a classic
classifier. It records every decision, learns from the real outcomes how much each reported
probability is actually worth, and routes each new decision to one of three tiers:

- **auto**: confident enough, after calibration, to act on without a person;
- **verify**: act, but have a person confirm;
- **human**: send to a review queue to be decided from scratch.

The thresholds between the tiers are not guessed. They are selected from labeled decisions
so that the auto tier meets an error budget with a stated statistical confidence, stored as
audited rules, and tightened automatically when the model drifts.

## The problem

Decision models now return typed answers with probabilities, but those probabilities are
not calibrated to *your* data. A model that says "0.9" may be right 70% of the time on your
tickets, and model-stated confidence is known to be overconfident. Teams end up picking a
threshold by gut feel, never revisit it, and only notice drift after the error rate has
already climbed.

decision-gate turns that into an engineering loop: log decisions, join outcomes, fit a
calibrator per question version, pick thresholds that meet an error budget with
statistical bounds, send the rest to people whose answers become new labels, and tighten
automatically when calibration drifts.

## Results on synthetic traffic

The one number that matters, the auto tier's true error rate, is exactly what production
cannot see, because most automated decisions never get a label. The built-in simulator
runs the real service, review queue, refit job, routing refresh and drift monitor on
synthetic traffic whose ground truth is known, over 21 days of history, 14 steady days and
14 days of unannounced drift. From the committed seed-7 run
([full report](docs/reports/simulation-seed-7.md)):

| Question | Budget | Steady: automated | Steady: true auto error | Drift detected after | Auto error before the action |
| --- | ---: | ---: | --- | ---: | --- |
| `refund.eligible` (yes/no) | 10% | 94.4% | 4.9% (95% CI 4.1%–5.8%) | 7 days | 11.5% (95% CI 9.8%–13.3%) |
| `ticket.queue` (5 options) | 5% | 67.4% | 5.3% (95% CI 4.3%–6.4%) | 4 days | 10.9% (95% CI 8.1%–14.2%) |

The model behind it is deliberately overconfident (raw ECE 0.045 and 0.102 in the steady
phase; 0.007 and 0.016 after calibration). On `ticket.queue` the steady error is not
statistically distinguishable from its 5% budget, which is the honest reading of a
guarantee that holds with 95% confidence. When drift is detected, the monitor suspends or
raises the auto threshold, and nothing more is automated until the evidence supports it.
[docs/STATISTICS.md](docs/STATISTICS.md) explains each guarantee and where it stops.

| `ticket.queue` reliability, steady | `ticket.queue` reliability, drift |
| --- | --- |
| ![ticket.queue, steady phase](docs/reports/diagrams/ticket.queue-steady.svg) | ![ticket.queue, drift phase](docs/reports/diagrams/ticket.queue-drift.svg) |

## How it works

```mermaid
flowchart LR
    client[Client] -->|POST /v1/decide| api[decision-gate API]
    api --> backend[Backend adapter<br/>fake / HTTP decision API / LLM-JSON]
    backend --> cal[Calibrator<br/>per question version]
    cal --> policy[Routing rule]
    policy -->|auto| client
    policy -->|verify / human| queue[Review queue]
    outcomes[Outcomes<br/>HTTP, JSON Lines or Kafka] --> store[(PostgreSQL)]
    queue -->|resolution = outcome| store
    api --> store
    api -.->|in the background| shadow[Shadow candidate] -.-> store
    store --> worker[Worker: refit, routing refresh, drift checks]
    worker --> cal
    worker --> policy
```

Every question kind (`predicate`, `choice`, `score`) reduces to the same problem: the
backend proposes an answer and a probability that it is correct, and an outcome later says
whether it was. One decision goes through these steps:

```mermaid
sequenceDiagram
    participant C as Client
    participant A as API
    participant B as Backend
    participant R as Router
    participant D as PostgreSQL
    participant Q as Review queue
    C->>A: POST /v1/decide (question, input, Idempotency-Key)
    A->>D: same key seen? return the stored decision
    A->>B: ask (circuit breaker, timeout)
    B-->>A: answer + confidence
    A->>A: validate the answer against the question
    A->>R: route
    R->>D: active calibrator and routing rule (cached)
    R-->>A: calibrated confidence, tier, reason, answer set
    A->>D: store the decision with its tier
    A->>Q: verify / human: enqueue with an SLA deadline
    A-->>C: decision, tier and reason
```

The loop closes in the background:

1. **Outcomes** arrive over HTTP, from JSON Lines files or a Kafka topic (with a
   dead-letter topic), and every review resolution is recorded as an outcome in the same
   transaction. A label is written once; a conflicting one is a `409`.
2. **Refit**: Platt, isotonic and temperature calibrators are fitted per question version
   and a new one is activated only if it beats the active one on held-out decisions.
3. **Routing refresh**: thresholds are selected with a fixed-sequence test on
   Clopper-Pearson (or Wilson) bounds, so the auto tier's precision meets the target with
   the configured confidence. A rule records the policy and calibrator it was selected
   under and goes stale, sending everything to a person, when either changes.
4. **Drift checks**: score PSI, outcome-mix PSI, windowed ECE with a bootstrap interval
   and the auto tier's observed error. On drift the auto threshold is raised or the tier
   suspended, a refresh will not undo it while the drift lasts, and a signed webhook
   alert is sent.

A separate `dgate worker` process runs steps 2–4 on a timer; a PostgreSQL advisory lock
keeps extra replicas from doing the work twice. A candidate model can answer the same
traffic in **shadow mode** and be compared with the primary on the same labeled decisions
(McNemar's exact test plus auto-tier coverage) before it is promoted.

**Design choices worth calling out**

- **Fail closed, and say why.** A routing problem never fails the request: the decision
  goes to `human` with a reason (`refund.eligible@v1 is not calibrated yet`). A backend
  failure is a `502`/`503`/`504` carrying `"fallback_tier": "human"`.
- **Thresholds are data, policies are configuration.** The error budget lives in YAML; the
  threshold that meets it is selected from evidence and kept with its history for audit.
- **Verify sampling keeps the labels honest.** A fixed share of automatable decisions is
  sent to verify, chosen by a hash of the decision id, so calibration does not only ever
  see the hard cases.
- **Everything is reproducible.** The fake backend, the simulator and the diagrams are
  deterministic for a seed, so reports and SVGs can be committed and diffed.

## Features

- Question catalog with versioned `predicate`, `choice` and `score` questions.
- Backends: a deterministic, configurably miscalibrated fake (no keys, no network), any
  JSON decision API, and any OpenAI-compatible `/chat/completions` endpoint (hosted, or
  local such as Ollama or vLLM) asked for `{"answer", "confidence"}`.
- `POST /v1/decide` with idempotency keys; PostgreSQL with Alembic migrations.
- Outcome ingestion over HTTP (single and batch), JSON Lines and Kafka with a dead-letter
  topic.
- Calibration metrics: ECE with a bootstrap interval, MCE, Brier, log loss, multi-class
  scores and reliability bins, per question version, backend and model.
- Calibrators (Platt, isotonic, temperature) written from scratch on numpy, with a
  versioned registry, held-out promotion and rollback.
- Threshold selection with precision guarantees, and split-conformal answer sets (LAC,
  APS) for the people who handle the rest.
- Routing tiers from a per-question policy, stored as audited rules.
- Review queue: SLA deadlines, leased claims with `SKIP LOCKED`, resolution as outcome,
  backlog, SLA and agreement statistics.
- Drift monitoring with automatic tightening or suspension and signed webhook alerts.
- Shadow mode, live or replayed over stored decisions, with a promote / keep /
  insufficient verdict.
- Simulator with known ground truth, segment-dependent miscalibration and injected drift.
- Prometheus `/metrics`, OpenTelemetry traces, a circuit breaker and per-caller rate
  limits.
- Maintenance worker, SVG reliability diagrams, and a Docker Compose stack tested end to
  end in CI.

## Quick start

Requires Python 3.12+ and [uv](https://docs.astral.sh/uv/). Everything below was run from
a fresh clone.

```bash
git clone https://github.com/xeeshan16/decision-gate.git
cd decision-gate
uv sync
```

Validate the example question catalog and ask the fake backend for a raw, uncalibrated
decision (the same input always gives the same answer):

```console
$ uv run dgate questions check examples/questions.yaml
refund.eligible@v1               predicate
ticket.queue@v1                  choice
ticket.handling_minutes@v1       score
3 question(s) OK

$ uv run dgate ask refund.eligible -q examples/questions.yaml -i '{"order_age_days": 12, "item": "headphones"}'
{
  "question": "refund.eligible@v1",
  "answer": true,
  "confidence": 0.7618304815842805,
  "probabilities": {
    "true": 0.7618304815842805,
    "false": 0.23816951841571954
  },
  "backend": "fake",
  "model_version": "fake-1"
}
```

The fake is overconfident by default, which is the problem the rest of the system exists
to correct. `fake-gap` measures how much, on 5,000 synthetic inputs per question:

```console
$ uv run dgate fake-gap -q examples/questions.yaml
question                          accuracy  confidence     gap     ECE   Brier
refund.eligible@v1                   0.755       0.849  +0.094   0.098   0.179
ticket.queue@v1                      0.606       0.788  +0.182   0.182   0.231
ticket.handling_minutes@v1           0.750       0.845  +0.096   0.096   0.178
```

### See the whole loop in about a minute and a half

The simulator needs no database server (it uses a temporary SQLite file). A shorter run
than the committed report, 14 days of history, 7 steady and 7 drifted, first lines:

```console
$ uv run dgate simulate run -n 200 --history-days 14 --steady-days 7 --drift-days 7 --refit-every 7
decision-gate simulation: seed 7, 200 decisions per day per question, 14 + 7 + 7 days (history + steady + drift)

refund.eligible@v1 (predicate, auto error budget 10%)
  steady: 94.7% automated, true auto error 60/1326 = 4.5% (95% CI 3.5%-5.8%), within the 10% budget
  drift from day 22: detected at the end of day 26 (5 day(s) of drifted traffic), auto tier suspended at the end of day 26
  drift, until the action: true auto error 81/895 = 9.1% (95% CI 7.3%-11.1%)
  drift, after the action: nothing automated

  phase       days decisions   auto verify  human hard+novel ECE raw ECE cal  true auto error
  history    1-14       2800   0.0%   0.0% 100.0%      15.9%   0.042       -  -
  steady    15-21       1400  94.7%   5.3%   0.0%      15.1%   0.041   0.006  60/1326 = 4.5% (95% CI 3.5%-5.8%)
  drift     22-28       1400  63.9%  31.5%   4.6%      44.8%   0.090   0.038  81/895 = 9.1% (95% CI 7.3%-11.1%)
```

Add `--markdown report.md --json report.json --diagrams diagrams/` to keep the report and
reliability diagrams.

### Run the API

With PostgreSQL running locally (any recent version; CI uses 17):

```bash
export DGATE_DATABASE_URL=postgresql+asyncpg://dgate:dgate@localhost:5432/dgate
export DGATE_QUESTIONS_FILE=examples/questions.yaml
export DGATE_ROUTING_FILE=examples/routing.yaml
uv run dgate db upgrade          # apply Alembic migrations
uv run dgate serve --port 8080
```

On a fresh database nothing is calibrated yet, so the first decision goes to a person and
says why (output wrapped for readability):

```console
$ curl -s -X POST localhost:8080/v1/decide -H 'content-type: application/json' \
    -H 'Idempotency-Key: order-1042' \
    -d '{"question": "refund.eligible", "input": {"order_age_days": 12, "item": "headphones"}}'
{"id":"f13ef5f8-2004-4ff0-ba82-1f3757226c5c","question":"refund.eligible","question_version":1,
 "answer":true,"confidence":0.7618304815842805,"calibrated_confidence":null,
 "tier":"human","reason":"refund.eligible@v1 is not calibrated yet","answer_set":null,
 "routing_rule_id":null,"probabilities":{"true":0.7618304815842805,"false":0.23816951841571954},
 "backend":"fake","model_version":"fake-1","latency_ms":0.27013499999384294,
 "idempotency_key":"order-1042","created_at":"2026-11-11T09:58:38.078616Z","outcome":null}
```

Retrying with the same `Idempotency-Key` returns the stored decision (`200` instead of
`201`). From here, outcomes, refits, routing refreshes, the review queue and drift checks
are walked through with real output in the [guide](docs/GUIDE.md).

### Run the whole stack (Docker Compose)

```bash
docker compose up --build            # API on http://localhost:8080, worker, PostgreSQL
docker compose --profile kafka up --build                         # add Kafka outcome ingestion
DGATE_OTEL_ENABLED=true docker compose --profile otel up --build  # add tracing
```

| Service | What it does |
| --- | --- |
| `postgres` | PostgreSQL 17 with a named volume |
| `migrate` | Runs `dgate db upgrade` once, then exits; everything else waits for it |
| `api` | `dgate serve` (fake backend, example catalog and routing), as a non-root user |
| `worker` | `dgate worker`: refit, routing refresh and drift checks every `DGATE_WORKER_INTERVAL_S` seconds (default 300) |
| `kafka`, `outcomes-consumer` | Profile `kafka`: a single-node KRaft broker and `dgate outcomes consume` |
| `otel-collector` | Profile `otel`: receives the API's traces and logs them |

The image is a two-stage build: uv resolves the locked dependencies in the first stage, and
the runtime stage holds only the virtual environment and the example configuration, run as
uid 10001. CI builds the image and runs the stack on every push.

## Backends

| `DGATE_BACKEND` | What it calls |
| --- | --- |
| `fake` (default) | Deterministic, configurably miscalibrated fake. No network, no keys. |
| `http` | Any JSON decision service: `POST $DGATE_HTTP_URL` with `{"question", "input"}`, expecting `{"answer", "confidence", "probabilities"?, "model_version"?}`. |
| `llm_json` | Any OpenAI-compatible `/chat/completions` endpoint. The model is asked for `{"answer", "confidence"}`. |

Whatever the backend, its answer is validated against the question (a choice must be one of
the options, a score must be in range) before it is stored; an invalid answer is a `502`,
never a silently stored decision. No API key is needed for anything in this repository,
including the tests and CI.

## Documentation

| Document | What is in it |
| --- | --- |
| [docs/GUIDE.md](docs/GUIDE.md) | Every component in use, with commands and real output: outcomes, calibration, thresholds, routing, reviews, drift, shadow mode, simulator, observability, worker, diagrams |
| [docs/REFERENCE.md](docs/REFERENCE.md) | All HTTP endpoints and status codes, every `DGATE_*` setting, the question catalog format |
| [docs/STATISTICS.md](docs/STATISTICS.md) | Each statistical method, what it guarantees, its assumptions and what is not guaranteed |
| [docs/reports/](docs/reports/simulation-seed-7.md) | The committed seed-7 simulation report and reliability diagrams |
| [docs/PLAN.md](docs/PLAN.md), [docs/PROGRESS.md](docs/PROGRESS.md) | The 14-day build plan and the daily log, including design decisions and known gaps |

## Project layout

```text
src/decision_gate/
  app.py, *_api.py      FastAPI app and routers
  service.py            decide: backend call, validation, routing, storage
  backends/             fake, HTTP decision API, OpenAI-compatible LLM
  calibration.py        metrics (ECE, MCE, Brier, log loss, reliability bins)
  calibrators.py        Platt, isotonic, temperature (numpy only)
  refit.py              calibrator registry and held-out promotion
  bounds.py, thresholds.py, selection.py   precision bounds and threshold selection
  conformal.py          split-conformal answer sets
  routing.py            tiers, rules, staleness
  reviews.py            review queue
  drift.py, drift_monitor.py, alerts.py    drift signals, actions, webhooks
  shadow*.py            shadow candidates and the paired comparison
  simulator/            synthetic datasets, simulated model, clock and reports
  worker.py             the maintenance loop
  resilience.py, telemetry.py, metrics.py  breaker, rate limits, traces, Prometheus
  ingest.py, kafka.py   outcome ingestion
  db/, migrations/      SQLAlchemy models and Alembic migrations
tests/                  unit and integration tests (SQLite, and PostgreSQL when configured)
tests/e2e/              end-to-end test against a running stack
```

## Development

```bash
uv sync --all-extras                  # the otel and kafka extras are needed for type checking
uv run ruff check . && uv run ruff format --check .
uv run mypy
uv run pytest --cov
```

Database tests run against SQLite by default. Point `DGATE_TEST_DATABASE_URL` at an empty
PostgreSQL database to run the same tests against PostgreSQL as well (they reset its
schema):

```bash
DGATE_TEST_DATABASE_URL=postgresql+asyncpg://dgate:dgate@localhost:5432/dgate_test uv run pytest
```

CI runs on every push:

- **check** (Python 3.12 and 3.13, PostgreSQL 17 service): ruff, mypy (strict), the test
  suite on SQLite and PostgreSQL with a 95% coverage floor, `uv build`, CLI smoke tests, a
  short simulation against PostgreSQL, and an API smoke test covering idempotent replay,
  outcomes over HTTP and JSON Lines, calibration metrics, refit, threshold selection,
  conformal fits, routing, the review queue, drift checks and `/metrics`.
- **stack**: validates the Compose file with every profile, builds the image, starts the
  stack and runs [tests/e2e/test_stack.py](tests/e2e/test_stack.py): 600 decisions all go
  to a person, their outcomes arrive in batches, the worker fits a calibrator and stores a
  routing rule on its own timer, and new decisions start to be automated, each at or above
  the rule's threshold. It then sends an outcome through a real Kafka broker and checks
  that a malformed event lands on the dead-letter topic.

## What I'd do next

- **A minimum improvement margin for refits.** A calibrator is promoted when it is
  strictly better on held-out log loss, so a negligible gain still makes the routing rule
  stale and sends traffic to people until the next refresh. A margin, or a paired test,
  would avoid needless churn.
- **Keep the drift hold across a new calibrator.** The floor a drift check puts on the auto
  threshold is tied to the calibrator it was set under. When a refit activates a new
  calibrator while drift is still open, the next refresh may loosen the threshold, and only
  the drift check that follows in the same pass tightens it again (in the short simulation
  above, day 28: 0.4192, then 0.9851). Keeping the tier suspended until a drift check on the
  new calibrator passes would close that window entirely.
- **A reviewer UI.** The queue has a complete API (claim, release, resolve, stats) but no
  front end; a small page per question with the answer set highlighted would make it
  usable by a support team.
- **Per-segment calibration.** The simulator shows the main failure mode: a calibrator fitted
  on one mix of easy and hard items is wrong for another. Calibrating per declared segment
  (customer tier, ticket source) would narrow that gap.
- **Worker metrics and rule-history retention.** The worker reports through JSON log lines
  only, and every routing refresh stores a new rule row; a metrics endpoint and a retention
  policy for superseded rules would make it easier to run for months.
- **Label delay in the guarantees.** Thresholds assume the labeled decisions represent
  current traffic; outcomes that arrive days late are weighted the same as fresh ones.

## License

[MIT](LICENSE)
