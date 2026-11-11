# decision-gate reference

The HTTP endpoints, every configuration variable and the question catalog format. The
[guide](GUIDE.md) shows each of them in use.

## Endpoints

| Endpoint | Purpose |
| --- | --- |
| `GET /healthz` | Liveness |
| `GET /readyz` | Readiness, including a database check (`503` when it is unreachable) and the circuit state |
| `GET /metrics` | Prometheus metrics |
| `GET /v1/questions`, `GET /v1/questions/{id}` | Question catalog |
| `POST /v1/decide` | Ask the backend, store and return the decision. `404` unknown question, `409` idempotency key reused for a different request, `429` rate limited, `502` backend error or invalid answer, `503` circuit open, `504` backend timeout; the three backend failures carry `"fallback_tier": "human"` |
| `GET /v1/decisions/{id}` | A stored decision, with its outcome once known |
| `POST /v1/outcomes` | Record the ground truth for a decision. `201` created, `200` same outcome already recorded, `404` unknown decision, `409` a different outcome was recorded, `422` not a valid answer |
| `POST /v1/outcomes/batch` | Up to 1,000 outcomes with a per-item status |
| `GET /v1/outcomes/stats` | Per question version: decisions, labeled, label rate, raw accuracy, mean raw confidence and the gap (`?question=` filters) |
| `GET /v1/calibration/metrics` | ECE, MCE, Brier, log loss, multi-class scores and reliability bins per question version, optionally per backend and model version |
| `POST /v1/calibrators/refit` | Fit calibrators on labeled decisions and activate the best if it wins on held-out data. `404` unknown question, `422` invalid options (e.g. `temperature` on a score question) |
| `GET /v1/calibrators`, `GET /v1/calibrators/{id}` | The calibrator registry (`?question=`, `?version=`, `?status=` filter) with parameters and held-out evaluation |
| `POST /v1/calibrators/{id}/activate` | Make a stored calibrator active (roll back); the current one is retired |
| `POST /v1/thresholds/select` | The lowest automation threshold whose precision meets the target with confidence, the candidate curve and a held-out check. `404` unknown question, `422` invalid options or a calibrator that cannot score the data |
| `POST /v1/conformal/fit` | A split-conformal predictor for a choice or predicate question, with held-out coverage and set sizes. `422` for score questions |
| `POST /v1/routing/refresh` | Select and store routing rules for one question (`{"question": ...}`) or every routed one (`{}`). `404` unknown or unrouted question, `422` data that cannot be analysed |
| `GET /v1/routing/status` | Per question version: its policy, its active rule and whether the rule is `current`, or why decisions are going to a human (`no_policy`, `no_rule`, `policy_changed`, `calibrator_changed`) |
| `GET /v1/routing/rules`, `GET /v1/routing/rules/{id}` | Rule history, newest first (`?question=`, `?version=`, `?status=` filter) |
| `GET /v1/reviews`, `GET /v1/reviews/{id}` | Review items with the decision a reviewer needs (answer, calibrated probability, answer set, input), earliest deadline first (`?state=available\|claimed\|resolved\|superseded`, `?question=`, `?tier=`, `?claimed_by=`, `?overdue=`) |
| `POST /v1/reviews/claim` | Claim the most urgent available item (`{"reviewer", "question"?, "tier"?, "lease_minutes"?}`); `204` when nothing is waiting |
| `POST /v1/reviews/{id}/claim`, `POST /v1/reviews/{id}/release` | Claim (or renew) one item; give it back. `404` unknown, `409` held by someone else, resolved or superseded |
| `POST /v1/reviews/{id}/resolve` | Resolve with `{"reviewer", "outcome"}` or `{"reviewer", "confirm": true}`; the answer becomes the decision's outcome. `201` resolved, `200` same resolution replayed, `409` conflict, `422` invalid answer |
| `GET /v1/reviews/stats` | Backlog, overdue count, SLA hit rate, median and p90 resolution time, and backend agreement with a Wilson interval, per question version and tier (`?question=`, `?since=`) |
| `POST /v1/reviews/backfill` | Enqueue verify and human decisions recorded before the queue existed |
| `POST /v1/drift/check` | Run drift checks for one question (`{"question": ...}`) or every monitored one (`{}`); tightens routing and sends alerts per the drift policy. `404` unknown or unmonitored question |
| `GET /v1/drift/status` | The latest drift check of every monitored question version |
| `GET /v1/drift/checks`, `GET /v1/drift/checks/{id}` | Check history with every signal, the action taken and the alert outcome, newest first (`?question=`, `?version=`, `?status=`) |
| `GET /v1/shadow/status` | The configured candidate, its in-process counters (submitted, skipped, dropped, completed, failed) and stored answer counts per question version and candidate (`?question=`) |
| `GET /v1/shadow/decisions/{id}` | Every candidate's answer to one decision |
| `POST /v1/shadow/replay` | Run the configured candidate over stored decisions it has not answered (`{"question", "limit"?, "labeled_only"?}`). `409` when shadow mode is not configured |
| `GET /v1/shadow/compare` | Paired comparison and verdict for one question (`?question=`, `?candidate=`, `?mode=live\|replay`, `?since=`, `?until=`, `?holdout=`, `?method=`, `?min_paired=`, `?alpha=`, `?min_coverage_gain=`, `?max_failure_rate=`). `422` for invalid options |

## Configuration

| Variable | Default | Meaning |
| --- | --- | --- |
| `DGATE_QUESTIONS_FILE` | unset | YAML or JSON question catalog |
| `DGATE_DATABASE_URL` | `postgresql+asyncpg://dgate:dgate@localhost:5432/dgate` | SQLAlchemy async database URL |
| `DGATE_ROUTING_FILE` | unset | YAML or JSON routing policies; without one every decision goes to a human |
| `DGATE_ROUTING_CACHE_TTL_S` | `5` | How long the router caches a question's rule and calibrator (`0` disables the cache) |
| `DGATE_REVIEW_VERIFY_SLA_MINUTES` | `60` | Deadline for resolving a verify item |
| `DGATE_REVIEW_HUMAN_SLA_MINUTES` | `240` | Deadline for resolving a human item |
| `DGATE_REVIEW_CLAIM_MINUTES` | `15` | How long a claim lasts before the item returns to the queue |
| `DGATE_DRIFT_WEBHOOK_URL` | unset | Where drift alerts are POSTed; unset means checks record `unconfigured` |
| `DGATE_DRIFT_WEBHOOK_SECRET` | unset | HMAC-SHA256 key for the `X-Dgate-Signature` header |
| `DGATE_DRIFT_WEBHOOK_TIMEOUT_S` | `5` | Time limit for one webhook attempt |
| `DGATE_DRIFT_WEBHOOK_ATTEMPTS` | `3` | Attempts per alert (timeouts, 5xx and 429 are retried) |
| `DGATE_STORE_INPUTS` | `true` | Store decision inputs; when `false` only a SHA-256 of the input is kept |
| `DGATE_BACKEND` | `fake` | `fake`, `http` or `llm_json` |
| `DGATE_BACKEND_TIMEOUT_S` | `10` | Time limit for one backend call |
| `DGATE_HTTP_URL`, `DGATE_HTTP_TOKEN` | unset | Endpoint and optional bearer token for `http` |
| `DGATE_LLM_BASE_URL`, `DGATE_LLM_MODEL` | unset | OpenAI-compatible base URL and model for `llm_json` |
| `DGATE_LLM_API_KEY` | unset | Optional API key for `llm_json` (local servers usually need none) |
| `DGATE_LLM_TEMPERATURE` | `0` | Sampling temperature for `llm_json` |
| `DGATE_FAKE_OVERCONFIDENCE` | `2.0` | Fake backend miscalibration; `1.0` is perfectly calibrated |
| `DGATE_FAKE_SEED` | `decision-gate` | Seed for the fake backend |
| `DGATE_SHADOW_BACKEND` | unset | Candidate backend for shadow mode: `fake`, `http` or `llm_json`; unset turns shadow mode off |
| `DGATE_SHADOW_NAME` | backend type | Name the candidate's answers are stored and compared under |
| `DGATE_SHADOW_SAMPLE_RATE` | `1.0` | Fraction of new decisions shadowed |
| `DGATE_SHADOW_QUESTIONS` | all | Comma-separated question ids to shadow |
| `DGATE_SHADOW_TIMEOUT_S` | `10` | Time limit for one candidate call |
| `DGATE_SHADOW_MAX_PENDING` | `100` | Most candidate calls in flight; decisions beyond it are not shadowed |
| `DGATE_SHADOW_HTTP_URL`, `DGATE_SHADOW_HTTP_TOKEN` | unset | Endpoint and token for an `http` candidate |
| `DGATE_SHADOW_LLM_BASE_URL`, `DGATE_SHADOW_LLM_MODEL`, `DGATE_SHADOW_LLM_API_KEY`, `DGATE_SHADOW_LLM_TEMPERATURE` | unset, `0` | An `llm_json` candidate |
| `DGATE_SHADOW_FAKE_OVERCONFIDENCE`, `DGATE_SHADOW_FAKE_MIN_SKILL`, `DGATE_SHADOW_FAKE_SEED`, `DGATE_SHADOW_FAKE_MODEL_VERSION` | `1.0`, unset, `decision-gate`, `fake-2` | A `fake` candidate; a higher skill floor makes a more accurate model |
| `DGATE_KAFKA_BOOTSTRAP_SERVERS` | `localhost:9092` | Brokers for `dgate outcomes consume` |
| `DGATE_KAFKA_OUTCOMES_TOPIC` | `dgate.outcomes` | Topic with outcome events |
| `DGATE_KAFKA_GROUP_ID` | `decision-gate-outcomes` | Consumer group |
| `DGATE_KAFKA_DEAD_LETTER_TOPIC` | `dgate.outcomes.dlq` | Where rejected events go; empty disables it |
| `DGATE_CIRCUIT_BREAKER_ENABLED` | `true` | Stop calling a backend that keeps failing |
| `DGATE_CIRCUIT_FAILURE_THRESHOLD` | `5` | Consecutive failures (errors or timeouts) that open the circuit |
| `DGATE_CIRCUIT_RESET_S` | `30` | How long it stays open before one probe call |
| `DGATE_RATE_LIMIT_PER_S` | unset | Average `POST /v1/decide` requests per second per caller; unset means no limit |
| `DGATE_RATE_LIMIT_BURST` | `20` | Largest burst per caller |
| `DGATE_RATE_LIMIT_KEY_HEADER` | unset | Header that identifies the caller (set by a trusted gateway); unset means the client address |
| `DGATE_METRICS_ENABLED` | `true` | Serve `/metrics` |
| `DGATE_OTEL_ENABLED` | `false` | Export traces over OTLP/HTTP (needs the `otel` extra) |
| `DGATE_OTEL_SERVICE_NAME` | `decision-gate` | `service.name` on exported spans |
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
