# decision-gate: 14-day plan

One milestone per day. Each day ships working code with tests and a green CI run. If the
plan changes, this file is updated and the reason is logged in [PROGRESS.md](PROGRESS.md).

| Day | Milestone | Status |
| --- | --- | --- |
| 1 | **Scaffold.** FastAPI app, settings, health checks, README, MIT license, CI (ruff, mypy, pytest, build), PLAN and PROGRESS docs. Schema for the three question types (predicate, choice, score) and a deterministic fake backend. | done |
| 2 | **Decide API and persistence.** `POST /v1/decide`, a backend adapter interface (fake, generic HTTP decision API, generic LLM-JSON), and PostgreSQL with Alembic. Integration tests run against real Postgres in CI. A vendor-specific decision-API adapter is deferred (see PROGRESS, day 2). | done |
| 3 | **Outcomes.** Idempotent `POST /v1/outcomes` for labels that arrive later, joined to their decisions, plus an optional Kafka outcome consumer behind an interface. | done |
| 4 | **Calibration metrics.** ECE, Brier score, log loss and reliability bins for each question, backend and version, tested against hand-computed fixtures. | |
| 5 | **Calibrators.** Platt scaling, isotonic regression (pool-adjacent-violators) and temperature scaling for multi-class questions. A versioned calibrator registry and a refit job. | |
| 6 | **Threshold selection.** Pick thresholds that meet a target precision or error rate using Wilson/Clopper-Pearson bounds. Split-conformal prediction sets for choice questions, so the answer set covers the true label at a guaranteed rate. | |
| 7 | **Routing tiers.** YAML policy for each question (error budget, minimum sample count). Questions that aren't calibrated yet go to a human. Each response includes the calibrated probability, the tier and the reason. | |
| 8 | **Human review queue.** List, claim and resolve APIs with SLA timestamps. Resolutions feed back in as outcomes, which closes the loop. | |
| 9 | **Drift monitoring.** Rolling-window ECE, population stability index (PSI) on the score distribution, and outcome-label rate. Thresholds tighten automatically and a webhook alert fires when drift is detected. | |
| 10 | **Shadow mode.** Run a candidate backend or prompt version in shadow and compare calibrated performance before promoting it. | |
| 11 | **Simulator.** Synthetic ticket-triage and refund-eligibility datasets with known ground truth and deliberately injected miscalibration and drift. `dgate simulate` produces real reports. | |
| 12 | **Observability and resilience.** Prometheus metrics, OpenTelemetry, backend timeouts and a circuit breaker that falls back to the human tier, plus rate limits. | |
| 13 | **Docker Compose and end-to-end test.** API, worker, Postgres and optional Kafka. Reliability-diagram reports and a statistics write-up. | |
| 14 | **Finish.** README polish, Mermaid diagram, real output, setup verified from a clean clone, and "What I'd do next". | |

## Ground rules

- No paid APIs or secrets are needed to build, test or demo. Hosted model adapters are
  optional and sit behind the `DecisionBackend` interface; tests use the deterministic fake.
- All data is synthetic.
- Calibrators and statistics are implemented from scratch (numpy only) and tested against
  hand-computed fixtures.
