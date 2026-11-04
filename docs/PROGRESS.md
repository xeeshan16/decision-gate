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


## Day 4 (2026-11-01): calibration metrics

CI on main was green (day 3's run passed), so the day went to the milestone.

**Done**

- `decision_gate.metrics`, numpy only: reliability bins (`uniform`: equal width,
  right-closed, empty bins kept so diagrams line up; `quantile`: equal count), ECE, MCE,
  top-label Brier score and log loss, multi-class Brier score and log loss over a full
  distribution, and a seeded percentile-bootstrap interval for the ECE. Inputs are
  validated (lengths, ranges, rows summing to 1) with clear errors.
- `decision_gate.calibration.build_reports`: one report per question version, optionally
  split by `backend` and/or `model_version`, with a `min_count` floor. Multi-class scores
  are filled in only when every decision in the group has a distribution; a true label the
  backend gave no probability counts as 0.
- `OutcomeRepository.labeled_rows(...)`: labeled decisions as plain column tuples (no ORM
  objects), filtered by question, version and decision-time window.
- `GET /v1/calibration/metrics` (`question`, `version`, repeatable `group_by`, `bins`,
  `strategy`, `since`, `until`, `bootstrap`, `min_count`) and `dgate calibration report`
  with the same options, a table view, `--show-bins` reliability tables and `--json`.
- `dgate fake-gap` now prints ECE and Brier next to the raw gap.
- The CI smoke test checks the new endpoint and CLI against PostgreSQL after recording an
  outcome.

**Observation worth keeping:** on 1,000 labeled decisions per question, the calibrated
fake (`m2`) has ECE 0.010 for `refund.eligible`, but its bootstrap interval is
0.009-0.040, above the point estimate. That is the known upward bias of binned ECE on
finite samples, applied twice by resampling. The README says to read the interval as a
noise indicator, not a confidence interval for the true error. Day 6's threshold
selection will rely on Wilson/Clopper-Pearson bounds on accuracy, not on ECE.

**Tests:** 274 passing locally on Python 3.13 and 3.12 against SQLite and a local
PostgreSQL 16 server, 100% line and branch coverage. Each metric has a fixture worked out
by hand in the test file (Brier 0.222, log loss 0.6997, ECE 0.34 and MCE 0.4 on a
five-decision set, quantile ECE 0.22, multi-class Brier 0.5), plus a zero-ECE calibrated
case, a large-sample overconfidence check, and every validation error. Report tests check
that the endpoint agrees with the metrics module on data recorded through the API, the
grouping by model version and backend, time windows, `min_count`, bootstrap intervals,
422s for bad parameters, and the CLI table, bins and JSON output.

**Next (day 5):** calibrators: Platt scaling, isotonic regression (pool-adjacent-violators)
and temperature scaling for multi-class questions, a versioned calibrator registry and a
refit job.


## Day 5 (2026-11-02): calibrators, registry and refit job

CI on main was green (day 4's run passed), so the day went to the milestone.

**Done**

- `decision_gate.calibrators`, numpy only:
  - Platt scaling by Newton's method with a backtracking line search, starting from the
    identity, on Platt's smoothed targets (finite parameters for separable or one-class
    data).
  - Isotonic regression: a linear-time, stack-based pool-adjacent-violators, ties pooled
    into weighted points, redundant knots dropped, linear interpolation between blocks,
    outputs kept within [0.001, 0.999].
  - Temperature scaling for full distributions, fitted by golden-section search on
    `1/T` (the NLL is convex in it), bounded to T in [0.05, 20].
  - Every calibrator returns the calibrated probability of the decided answer and
    round-trips its parameters through JSON.
- Calibrator registry: a `calibrators` table (migration 0003) with per-question-version
  fit numbers, status (`active`, `candidate`, `rejected`, `retired`), parameters, the
  training window and the held-out evaluation. A partial unique index allows one active
  calibrator per question version; activation retires the old one in the same
  transaction, and concurrent fits retry on the version unique constraint.
- Refit job: chronological split (newest 30% held out by default), every candidate method
  fitted on the older part, the best by held-out log loss (Brier breaks ties) stored, and
  activated only if it strictly beats the raw confidences and the active calibrator on
  the same held-out decisions. Too few labels store nothing.
- `POST /v1/calibrators/refit`, `GET /v1/calibrators`, `GET /v1/calibrators/{id}`,
  `POST /v1/calibrators/{id}/activate`; `dgate calibrators refit [QUESTION | --all]`,
  `list`, `show`, `activate`; `dgate fake-calibrate` for a database-free comparison.
- `LabeledRow` now carries the decided answer and the decision time (temperature scaling
  needs the answer key, the split needs the time).
- CI's API smoke test runs the refit job over HTTP and the CLI against PostgreSQL.

**Real run (in the README):** 1,000 labeled decisions per question from the fake with
overconfidence 2, recorded through the API into PostgreSQL 16. The refit job activated
temperature scaling for `ticket.queue` with T = 2.15 (held-out log loss 0.823 to 0.567,
ECE 0.225 to 0.043) and for `refund.eligible` (0.550 to 0.495). A later isotonic fit was
rejected because it did not beat the active calibrator.

**Honest gaps:**

- For `ticket.handling_minutes` Platt won by 0.001 log loss on 300 held-out decisions and
  was promoted, because the rule is "strictly better". That is within noise. A minimum
  improvement margin (or a paired bootstrap test) belongs with the routing policy on
  day 7, where promotion actually changes behaviour.
- The stored calibrator is the one fitted on 70% of the data, so the recorded held-out
  numbers describe exactly the deployed model. Refitting on all data after selection
  would use more labels but leave the deployed model unevaluated; I chose the former.
- Calibrated probabilities are not applied to `/v1/decide` responses yet (day 7).
- The concurrent-version test runs on PostgreSQL only; SQLite serialises writers. The
  exact-race branches (repeated lost version races, a concurrent activation) are marked
  `no cover`.

**Tests:** 343 passing locally on Python 3.13 and 3.12 against SQLite and a local
PostgreSQL 16 server, 100% line and branch coverage. Hand-worked fixtures: PAV on small
sequences (including a cascade merge), isotonic knots and interpolation, Platt on a
constant input (41/60 from the smoothed targets), temperature scaling at T = 2 (0.8/0.2
becomes 2/3 and 1/3, NLL log 1.5). Optimality checks: Platt's gradient is zero at the fit,
the temperature fit is a minimum. Recovery checks: Platt recovers a = 0.5 and temperature
scaling recovers T = 2 from synthetic data with a known distortion. Registry and job tests
cover versions, rollback, the one-active index, rejection against raw and against the
incumbent, candidates, an incumbent that cannot be scored, windows, `--all`, the HTTP
endpoints and the CLI.

**Next (day 6):** threshold selection: thresholds that meet a target precision or error
rate using Wilson and Clopper-Pearson bounds, and split-conformal prediction sets for
choice questions.



## Day 6 (2026-11-03): threshold selection and conformal prediction sets

CI on main was green (day 5's run passed), so the day went to the milestone.

**Done**

- `decision_gate.bounds`, stdlib only: the regularized incomplete beta function (continued
  fraction, modified Lentz), beta quantiles by bisection, and one-sided and two-sided
  Clopper-Pearson and Wilson bounds on a binomial proportion. Checked against scipy on
  3,000 random cases during development (largest difference 5e-13); scipy is not a
  dependency.
- `decision_gate.thresholds`: candidate thresholds are the distinct scores accepting at
  least `min_accepted` decisions (thinned to at most 100), tested as a fixed sequence from
  the highest down. Each must have a lower confidence bound on precision at or above the
  target; the search stops at the first failure. Because the order depends only on the
  scores, the family-wise error stays at `1 - confidence` with no correction.
- `decision_gate.conformal`: split-conformal predictors with the finite-sample
  `ceil((n + 1)(1 - alpha))` quantile, LAC and non-randomised APS scores (APS always keeps
  the top label, so its sets are never empty), set prediction, evaluation (coverage, mean
  size, singleton rate and accuracy, empty rate, size counts) and JSON parameters.
- `decision_gate.selection.GateAnalyzer`: runs both over stored decisions, scoring with
  the active calibrator (temperature scaling also rescales the distributions conformal
  sets use) and skipping the calibrator's own training decisions. The rest is split by
  time into a selection part and a held-out check.
- `POST /v1/thresholds/select` and `POST /v1/conformal/fit` (in a separate router module
  to keep `app.py` from growing further), `dgate thresholds select`, `dgate conformal fit`,
  and `dgate fake-gate` for a database-free run of the whole pipeline. CI smoke tests the
  endpoints and commands against PostgreSQL.

**Real run (in the README):** 3,000 labeled decisions per question from the fake with
overconfidence 2, recorded through the API into PostgreSQL 16, then `refit --all`. For
`refund.eligible` the calibrated threshold for 90% precision is 0.867 (24.4% automated,
lower bound 0.908); on the 270 held-out decisions it automated 23.3% at precision 0.905.
For `ticket.queue` no calibrated threshold qualified on the 630 eligible decisions, while
raw scores with all 2,100 older decisions met the target at 19% automation.

**Honest gaps:**

- Skipping the calibrator's training decisions is the cautious rule for every calibrator,
  but for a strictly increasing calibrator (Platt, and temperature scaling on two-label or
  evenly spread distributions) the accepted sets are the same as on raw scores, so the
  rule only costs statistical power, as `ticket.queue` shows. Day 7 will use all decisions
  when the active calibrator is strictly increasing and keep the exclusion for isotonic
  (whose pooled steps depend on the labels).
- The `ticket.queue` LAC held-out coverage in the real run was 0.859 against a 0.9 target,
  about two standard deviations low for 630 calibration and 270 held-out decisions. The
  guarantee is an average over calibration sets; the tests check that average over 150
  synthetic runs of a deliberately miscalibrated model, allowing 0.006 for noise.
- APS is close to useless for two-label questions (it returned both labels for every
  `refund.eligible` decision in `fake-gate`); LAC is the default.
- The guarantee assumes the labeled decisions are representative of future traffic. Label
  delay or selective labeling (only reviewed decisions get outcomes) would violate that;
  drift monitoring (day 9) and sampling in the verify tier (day 7) are the mitigations.
- Results are computed on request and not stored; the routing policy on day 7 will persist
  the threshold and quantile it acts on.
- `min_accepted` now defaults to 100. With 30, a single error in the first candidate
  (29/30 has a 95% lower bound of 0.851) ended the search, and a simulation selected a
  threshold in only 69 of 100 runs.

**Tests:** 445 passing locally on Python 3.13 and 3.12 against SQLite and a local
PostgreSQL 16 server, 100% line and branch coverage. Hand-worked fixtures: the binomial
tail identity for the incomplete beta, Clopper-Pearson 5/10 (0.1871, 0.8129) and the
closed forms for 0/n and n/n, Wilson 5/10 (0.2366, 0.7634), candidate thresholds with ties,
a fixed sequence that stops on its first candidate even though a later one would pass, the
conformal rank `ceil((n + 1)(1 - alpha))`, and LAC and APS sets on a five-decision example.
Property checks: exact Clopper-Pearson coverage is never below 95% across 25 proportions,
Wilson's dips below it somewhere, the selected threshold misses its target in at most 10%
of 100 simulated runs (nominal 5%), and conformal coverage averages within bounds on a
miscalibrated model. Integration tests cover the analyzer, both endpoints and the CLI on
SQLite and PostgreSQL, including calibrator exclusion, unloadable calibrators and invalid
data.

**Next (day 7):** routing tiers: a YAML policy per question (error budget, confidence,
minimum sample count, verify sampling rate), uncalibrated questions routed to a human, and
`/v1/decide` responses that carry the calibrated probability, the tier and the reason.

## Day 7 (2026-11-04): routing tiers

**Done:**

- `decision_gate.routing`: a YAML routing policy per question (`max_error_rate`,
  `confidence`, `bound`, `min_labeled`, `min_accepted`, `require_calibrator`,
  `verify_max_error_rate`, `verify_sample_rate`, `conformal_alpha`, `conformal_method`),
  with shared `defaults`, an optional `default` policy for unlisted questions, and a stable
  fingerprint. `examples/routing.yaml` covers the three example questions.
- `RoutingRefresher` selects the auto threshold (and optionally a verify threshold under
  a looser budget, and conformal answer-set parameters) from labeled decisions and stores
  them in a new `routing_rules` table (migration 0004), one active rule per question
  version and the rest retired for audit. Each rule records the policy fingerprint and the
  calibrator it was selected under.
- `Router` assigns `auto`, `verify` or `human` to every new decision before it is stored:
  calibrated probability against the thresholds, a reproducible verify sample (a hash of
  the decision id) from the automatable decisions, and `human` with a reason whenever
  there is no policy, no calibrator, too few labels, a stale rule (policy or calibrator
  changed), or a calibrator that cannot be loaded or cannot score the decision. Rules and
  calibrators are cached per question for `DGATE_ROUTING_CACHE_TTL_S`; refreshes, refits
  and activations through the API clear the cache.
- `/v1/decide` responses (and stored decisions) now carry `calibrated_confidence`, `tier`,
  `reason`, `answer_set` and `routing_rule_id`. Replays return the stored tier.
- `POST /v1/routing/refresh`, `GET /v1/routing/status`, `GET /v1/routing/rules[/{id}]`,
  `dgate routing check` and `dgate routing refresh`. CI smoke tests the routing flow
  against PostgreSQL.
- Day 6 follow-up: threshold selection and conformal fits now reuse the calibrator's
  training decisions when the calibrator cannot reorder them (Platt with a positive slope,
  temperature scaling on two labels; conformal sets also with Platt and isotonic, which do
  not touch the distributions). Isotonic, and temperature over three or more labels, keep
  the exclusion. Responses report `reused_training` next to `excluded_training`.

**Real run (in the README):** 3,000 labeled decisions per question through the API into
PostgreSQL 16, `refit --all`, `dgate routing refresh`, then 2,000 fresh decisions per
question scored against the fake's ground truth. `refund.eligible` (10% budget): 21.4%
automated at accuracy 0.953. `ticket.handling_minutes` (20% budget): 73.4% automated at
0.793, 0.7 standard errors below the 0.80 target on this sample (the guarantee is on the
long-run rate, and the selection's lower bound was 0.8026). `ticket.queue` (5% budget):
nothing automated, because 900 eligible decisions cannot prove a 5% error rate.

**Honest gaps:**

- The verify threshold bounds the precision of everything at or above it, not of the
  verify band on its own: for `refund.eligible` the band ran at 0.712 while auto plus
  verify ran at 0.768 against a 0.75 target. The README says so; a band-specific bound
  would need a different test.
- Rules are refreshed on request (HTTP or CLI), not on a schedule. Drift monitoring (day
  9) is where automatic tightening belongs; until then an operator or a cron job calls
  the refresh endpoint.
- A CLI refresh against a running server takes effect after the cache TTL (5 s by
  default); the HTTP refresh applies at once.
- Sampled decisions go to `verify`, but nothing yet turns verify or human tiers into work
  items. That is the review queue (day 8), whose resolutions will flow back as outcomes.

**Tests:** 492 passing locally on Python 3.13 against SQLite and a local PostgreSQL 16
server, 100% line and branch coverage. New: policy parsing, merging, validation and
fingerprints; sampling is deterministic and uniform; end-to-end routing on the fake with
a refit calibrator (every decision's tier is checked against the stored thresholds and the
recomputed sample, and auto accuracy on fresh decisions); stale rules after a new
calibrator or a policy change; replays keep their tier; too few labels; raw-confidence
routing; calibrators that cannot load or score; trivial, failed and mismatched answer
sets; the router cache and its invalidation; the HTTP and CLI error paths; and which
calibrators may reuse their training decisions.

**Next (day 8):** the human review queue: list, claim and resolve verify and human
decisions with SLA timestamps, and feed resolutions back as outcomes.
