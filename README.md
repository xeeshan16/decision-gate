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
- **verify**: act, but sample for review;
- **human**: send to a review queue.

> **Status:** under active development (day 4 of a 14-day build). See
> [docs/PLAN.md](docs/PLAN.md) for the roadmap and [docs/PROGRESS.md](docs/PROGRESS.md) for
> the daily log. Sections below marked *planned* do not exist yet.

## The problem

Decision models now return typed answers with probabilities, but those probabilities are
not calibrated to *your* data. A model that says "0.9" may be right 70% of the time on your
tickets, and model-stated confidence is known to be overconfident. Teams end up picking a
threshold by gut feel, never revisit it, and only notice drift after the error rate has
already climbed.

decision-gate turns that into an engineering loop: log decisions, join outcomes, fit a
calibrator per question version, pick thresholds that meet an error budget with
statistical bounds, and tighten automatically when calibration drifts.

## Features

| Feature | Status |
| --- | --- |
| Question schema: `predicate`, `choice`, `score`, versioned per question | ✅ day 1 |
| `DecisionBackend` interface and a deterministic, configurably miscalibrated fake | ✅ day 1 |
| Health, readiness and question catalog API; `dgate` CLI | ✅ day 1 |
| Backend adapters: fake, generic HTTP decision API, OpenAI-compatible LLM returning JSON | ✅ day 2 |
| `POST /v1/decide` with idempotency keys, PostgreSQL persistence and Alembic migrations | ✅ day 2 |
| Idempotent outcomes over HTTP (single and batch), label coverage and raw accuracy stats | ✅ day 3 |
| Streaming outcome ingestion: JSON Lines CLI and an optional Kafka consumer with a dead-letter topic | ✅ day 3 |
| Calibration metrics: ECE (with a bootstrap interval), MCE, Brier, log loss, multi-class scores and reliability bins, per question version, backend and model | ✅ day 4 |
| Platt, isotonic and temperature calibrators | planned (day 5) |
| Thresholds with Wilson/Clopper-Pearson bounds, conformal sets | planned (day 6) |
| Per-question routing policy and human review queue | planned (days 7-8) |
| Drift detection, shadow mode, simulator, observability | planned (days 9-12) |

## Architecture

```mermaid
flowchart LR
    client[Client] -->|POST /v1/decide| api[decision-gate API]
    api --> backend[Backend adapter<br/>fake / HTTP decision API / LLM-JSON]
    backend --> cal[Calibrator<br/>per question version]
    cal --> policy[Threshold policy]
    policy -->|auto| client
    policy -->|verify / human| queue[Review queue]
    outcomes[Outcomes<br/>HTTP or Kafka] --> store[(PostgreSQL)]
    queue --> outcomes
    api --> store
    store --> refit[Refit worker] --> cal
    store --> drift[Drift monitor] --> policy
```

Every question kind reduces to the same calibration problem: the backend proposes an answer
and a probability that it is correct, and an outcome later says whether it was. For `score`
questions an answer is correct when it lands within the question's `tolerance`.

## Quick start

Requires Python 3.12+ and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/xeeshan16/decision-gate.git
cd decision-gate
uv sync
uv run pytest
```

Validate a question catalog:

```console
$ uv run dgate questions check examples/questions.yaml
refund.eligible@v1               predicate
ticket.queue@v1                  choice
ticket.handling_minutes@v1       score
3 question(s) OK
```

Ask the deterministic fake backend for a raw (uncalibrated) decision. The same input always
gives the same answer:

```console
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

The fake is deliberately overconfident by default (`overconfidence=2.0`), which is the
problem the rest of the system exists to correct. `fake-gap` measures it on 5,000 synthetic
inputs per question, with no database involved:

```console
$ uv run dgate fake-gap -q examples/questions.yaml
question                          accuracy  confidence     gap     ECE   Brier
refund.eligible@v1                   0.755       0.849  +0.094   0.098   0.179
ticket.queue@v1                      0.606       0.788  +0.182   0.182   0.231
ticket.handling_minutes@v1           0.750       0.845  +0.096   0.096   0.178
```

Run the API. It needs PostgreSQL (any recent version; CI uses 17):

```bash
export DGATE_DATABASE_URL=postgresql+asyncpg://dgate:dgate@localhost:5432/dgate
export DGATE_QUESTIONS_FILE=examples/questions.yaml
uv run dgate db upgrade          # apply Alembic migrations
uv run dgate serve --port 8080
```

Ask for a decision. Every decision is stored with the backend's raw confidence, so that
outcomes can later be joined to it. Retrying with the same `Idempotency-Key` returns the
original decision (`200`) instead of creating a new one (`201`):

```console
$ curl -s -X POST localhost:8080/v1/decide -H 'content-type: application/json' \
    -H 'Idempotency-Key: order-1042' \
    -d '{"question": "refund.eligible", "input": {"order_age_days": 12, "item": "headphones"}}'
{"id":"a1ce214c-b985-445f-b9f5-5eaf3afb86f4","question":"refund.eligible","question_version":1,
 "answer":true,"confidence":0.7618304815842805,
 "probabilities":{"true":0.7618304815842805,"false":0.23816951841571954},
 "backend":"fake","model_version":"fake-1","latency_ms":0.139,
 "idempotency_key":"order-1042","created_at":"2026-10-30T09:58:06.815632Z"}
```

(Output wrapped for readability.) `GET /v1/decisions/{id}` returns a stored decision,
including its outcome once one is recorded.

### Outcomes

The ground truth usually arrives later, from another system. Record it once per decision;
decision-gate scores it with the question's own rule (`correct`). Sending the same outcome
again is a harmless `200`; sending a *different* one is a `409`, because silently
rewriting a label would rewrite the history calibration is fitted on:

```console
$ curl -s -X POST localhost:8080/v1/outcomes -H 'content-type: application/json' \
    -d '{"decision_id": "ad9d1aff-f579-47b8-9b1b-0baf3497505d", "outcome": false, "source": "refund-desk"}'
{"decision_id":"ad9d1aff-f579-47b8-9b1b-0baf3497505d","outcome":false,"correct":false,
 "source":"refund-desk","observed_at":"2026-10-31T10:00:51.053205Z",
 "recorded_at":"2026-10-31T10:00:51.055387Z"}
```

For backfills, `POST /v1/outcomes/batch` takes up to 1,000 outcomes and reports each one
(`created`, `duplicate`, `conflict`, `unknown_decision` or `invalid`). Outcome events in a
file go through the same path with the CLI, which is safe to re-run:

```console
$ uv run dgate outcomes ingest outcomes.jsonl --dead-letter rejected.jsonl
line 4: invalid: ticket.queue@v1: answer must be one of ['billing', 'technical', 'account', 'shipping', 'other']
created 3, duplicate 0, rejected 1
  invalid: 1
$ uv run dgate outcomes ingest outcomes.jsonl
line 4: invalid: ticket.queue@v1: answer must be one of ['billing', 'technical', 'account', 'shipping', 'other']
created 0, duplicate 3, rejected 1
  invalid: 1
```

Each line is `{"decision_id", "outcome", "source"?, "observed_at"?}`. `GET
/v1/outcomes/stats` then shows label coverage and how raw confidence compares with raw
accuracy, per question version (this is what the calibrators will fix):

```console
$ curl -s localhost:8080/v1/outcomes/stats | jq '.questions[1]'
{
  "question": "ticket.queue",
  "question_version": 1,
  "decisions": 4,
  "labeled": 3,
  "label_rate": 0.75,
  "accuracy": 0.6666666666666666,
  "mean_confidence": 0.7819558140025135,
  "confidence_gap": 0.11528914733584683
}
```

#### Kafka

With the optional extra (`uv sync --extra kafka`, or `pip install 'decision-gate[kafka]'`),
`dgate outcomes consume` reads outcome events from a topic. Offsets are committed manually,
one record at a time and only after it is handled, so delivery is at-least-once and
idempotent recording makes redeliveries harmless. Events that can never succeed as sent
(malformed, unknown decision, invalid or conflicting label) are published to a dead-letter
topic with `dgate-reason`, `dgate-detail` and `dgate-origin` headers, then committed, so
one bad event cannot block a partition. Anything else (say, the database is down) stops
the consumer without committing, and the event is redelivered on restart.

### Calibration metrics

Once outcomes arrive, `GET /v1/calibration/metrics` and `dgate calibration report` measure
how far the raw confidence is from the real accuracy, per question version. Every metric
is computed on the pair every question kind reduces to: the confidence the backend gave
its answer, and whether the answer was right.

| Metric | What it tells you |
| --- | --- |
| Reliability bins | Mean confidence against accuracy in each confidence bin: the reliability diagram. `uniform` bins (equal width, right-closed, empty bins kept so models line up) or `quantile` bins (equal count). |
| ECE | Sample-weighted mean of `\|accuracy - confidence\|` over the bins. `bootstrap=N` adds a seeded 95% percentile-bootstrap interval. |
| MCE | The worst bin's gap. |
| Brier score, log loss | Binning-free scores of the same pairs. Log loss punishes confident mistakes hardest. |
| Multi-class Brier and log loss | For predicate and choice decisions with a full `probabilities` distribution, scored against the true label. |

Split by `backend` and/or `model_version` to compare models or prompt versions. Below,
the fake backend answered 1,000 synthetic inputs per question as `m1` (overconfident,
`overconfidence=2.0`) and again as `m2` (calibrated, `1.0`), with outcomes from the fake's
own ground truth recorded through the API:

```console
$ uv run dgate calibration report --by model_version --bootstrap 200
question                     model_version       n    acc   conf    ECE       ECE 95%    MCE  Brier logloss
refund.eligible@v1           m1               1000  0.754  0.850  0.098   0.079-0.129  0.175  0.178   0.569
refund.eligible@v1           m2               1000  0.754  0.752  0.010   0.009-0.040  0.016  0.166   0.498
ticket.handling_minutes@v1   m1               1000  0.754  0.829  0.075   0.056-0.103  0.105  0.168   0.521
ticket.handling_minutes@v1   m2               1000  0.754  0.735  0.022   0.017-0.052  0.050  0.162   0.485
ticket.queue@v1              m1               1000  0.585  0.787  0.210   0.187-0.240  0.332  0.252   0.826
ticket.queue@v1              m2               1000  0.585  0.597  0.032   0.025-0.068  0.068  0.198   0.575
```

Same answers, same accuracy; only the stated confidence differs, and every score sees it.
The calibrated model's interval sits above its point estimate because binned ECE is
biased upwards on finite samples (even a perfectly calibrated model shows a gap in each
bin), and resampling adds the same bias again. Read the interval as "how noisy is this",
not as a confidence interval for the true calibration error.

`--show-bins` prints each reliability table (empty bins trimmed here):

```console
$ uv run dgate calibration report --question refund.eligible --by model_version --show-bins
...
refund.eligible@v1 m1: reliability (uniform bins)
  bin                n   conf    acc     gap  accuracy
  0.50-0.60         96  0.550  0.562  -0.012  ###########
  0.60-0.70        109  0.652  0.550  +0.101  ###########
  0.70-0.80        127  0.756  0.685  +0.071  ##############
  0.80-0.90        160  0.857  0.681  +0.175  ##############
  0.90-1.00        508  0.970  0.874  +0.096  #################
```

The endpoint returns the same metrics as JSON, with every bin. Without `group_by`, a
report covers every backend and model version of the question (both `m1` and `m2` here):

```console
$ curl -s 'localhost:8080/v1/calibration/metrics?question=ticket.queue&bins=2' | jq '.reports[0] | del(.bins)'
{
  "question": "ticket.queue",
  "question_version": 1,
  "backend": null,
  "model_version": null,
  "count": 2000,
  "accuracy": 0.585,
  "mean_confidence": 0.692359296785496,
  "overconfidence": 0.10735929678549605,
  "ece": 0.10735929678549704,
  "ece_interval": null,
  "mce": 0.14741396512146432,
  "brier": 0.2251036955756147,
  "log_loss": 0.7002494720984769,
  "multiclass_brier": 0.5926296194695184,
  "multiclass_log_loss": 1.2755616319632312
}
```

Parameters: `question`, `version`, `group_by` (repeatable: `backend`, `model_version`),
`bins` (1-100, default 10), `strategy` (`uniform` or `quantile`), `since`/`until` (decision
time), `bootstrap` (0-1000 resamples) and `min_count` (skip smaller groups). The CLI takes
the same options (`--by`, `--bins`, `--strategy`, `--since`, `--until`, `--bootstrap`,
`--min-count`, `--json`). Metrics are plain numpy, implemented from scratch and tested
against hand-worked fixtures in [tests/test_metrics.py](tests/test_metrics.py).

### Endpoints

| Endpoint | Purpose |
| --- | --- |
| `GET /healthz` | Liveness |
| `GET /readyz` | Readiness, including a database check (`503` when it is unreachable) |
| `GET /v1/questions`, `GET /v1/questions/{id}` | Question catalog |
| `POST /v1/decide` | Ask the backend, store and return the decision. `404` unknown question, `409` idempotency key reused for a different request, `502` backend error or invalid answer, `504` backend timeout |
| `GET /v1/decisions/{id}` | A stored decision, with its outcome once known |
| `POST /v1/outcomes` | Record the ground truth for a decision. `201` created, `200` same outcome already recorded, `404` unknown decision, `409` a different outcome was recorded, `422` not a valid answer |
| `POST /v1/outcomes/batch` | Up to 1,000 outcomes with a per-item status |
| `GET /v1/outcomes/stats` | Per question version: decisions, labeled, label rate, raw accuracy, mean raw confidence and the gap (`?question=` filters) |
| `GET /v1/calibration/metrics` | ECE, MCE, Brier, log loss, multi-class scores and reliability bins per question version, optionally per backend and model version |

## Backends

| `DGATE_BACKEND` | What it calls |
| --- | --- |
| `fake` (default) | Deterministic, configurably miscalibrated fake. No network, no keys. |
| `http` | Any JSON decision service: `POST $DGATE_HTTP_URL` with `{"question", "input"}`, expecting `{"answer", "confidence", "probabilities"?, "model_version"?}`. |
| `llm_json` | Any OpenAI-compatible `/chat/completions` endpoint (hosted, or local servers such as Ollama or vLLM). The model is asked for `{"answer", "confidence"}`. |

Whatever the backend, its answer is validated against the question (a choice must be one of
the options, a score must be in range) before it is stored; an invalid answer is a `502`,
never a silently stored decision.

## Configuration

| Variable | Default | Meaning |
| --- | --- | --- |
| `DGATE_QUESTIONS_FILE` | unset | YAML or JSON question catalog |
| `DGATE_DATABASE_URL` | `postgresql+asyncpg://dgate:dgate@localhost:5432/dgate` | SQLAlchemy async database URL |
| `DGATE_STORE_INPUTS` | `true` | Store decision inputs; when `false` only a SHA-256 of the input is kept |
| `DGATE_BACKEND` | `fake` | `fake`, `http` or `llm_json` |
| `DGATE_BACKEND_TIMEOUT_S` | `10` | Time limit for one backend call |
| `DGATE_HTTP_URL`, `DGATE_HTTP_TOKEN` | unset | Endpoint and optional bearer token for `http` |
| `DGATE_LLM_BASE_URL`, `DGATE_LLM_MODEL` | unset | OpenAI-compatible base URL and model for `llm_json` |
| `DGATE_LLM_API_KEY` | unset | Optional API key for `llm_json` (local servers usually need none) |
| `DGATE_LLM_TEMPERATURE` | `0` | Sampling temperature for `llm_json` |
| `DGATE_FAKE_OVERCONFIDENCE` | `2.0` | Fake backend miscalibration; `1.0` is perfectly calibrated |
| `DGATE_FAKE_SEED` | `decision-gate` | Seed for the fake backend |
| `DGATE_KAFKA_BOOTSTRAP_SERVERS` | `localhost:9092` | Brokers for `dgate outcomes consume` |
| `DGATE_KAFKA_OUTCOMES_TOPIC` | `dgate.outcomes` | Topic with outcome events |
| `DGATE_KAFKA_GROUP_ID` | `decision-gate-outcomes` | Consumer group |
| `DGATE_KAFKA_DEAD_LETTER_TOPIC` | `dgate.outcomes.dlq` | Where rejected events go; empty disables it |
| `DGATE_LOG_LEVEL` | `info` | `debug`, `info`, `warning` or `error` |

## Question catalog

```yaml
questions:
  - id: ticket.queue
    version: 1            # bump when the prompt or options change
    kind: choice
    prompt: Which support queue should handle this ticket?
    options: [billing, technical, account, shipping, other]
  - id: ticket.handling_minutes
    kind: score
    prompt: Estimate how many minutes an agent will need.
    min_value: 0
    max_value: 240
    tolerance: 15         # within 15 minutes counts as correct
```

## Development

```bash
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

CI runs the checks on Python 3.12 and 3.13 with a PostgreSQL 17 service container, plus
`uv build`, a CLI smoke test, and an API smoke test that migrates the database, checks an
idempotent replay, records an outcome over HTTP, replays it through `dgate outcomes ingest`
and checks the stats count it once, then checks calibration metrics over HTTP and the CLI. The Kafka source is tested with in-memory consumer and
producer fakes; a test against a real broker comes with the Docker Compose stack (day 13).

## License

[MIT](LICENSE)
