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

## Day 8 (2026-11-05): human review queue

**Shipped:**

- `review_items` table (migration 0005): one item per `verify` or `human` decision, with
  `created_at`, an SLA `due_at`, claim fields (`claimed_by`, `claimed_at`,
  `claim_expires_at`, `claims`) and resolution fields (`resolved_by`, `resolved_at`,
  `resolution`, `verdict`, `agreed`, `note`). The item is inserted in the same transaction
  as its decision; an idempotent replay never enqueues twice.
- SLAs per tier (`DGATE_REVIEW_VERIFY_SLA_MINUTES` 60, `DGATE_REVIEW_HUMAN_SLA_MINUTES`
  240) and claim leases (`DGATE_REVIEW_CLAIM_MINUTES` 15, or `lease_minutes` per claim).
  An expired lease puts the item back in the queue without any background job: "available"
  is computed from the lease time.
- `POST /v1/reviews/claim` gives out the earliest-deadline available item, using
  `FOR UPDATE SKIP LOCKED` on PostgreSQL plus a conditional `UPDATE` as the real guard (the
  only one on SQLite). Claim one item, renew your own claim, release it; a lost race is a
  `409`.
- `POST /v1/reviews/{id}/resolve` with an `outcome` or `confirm: true` validates the answer,
  then stores the resolution and the outcome (`source: "review"`) in one transaction under
  a row lock. Verdicts: `confirmed` / `corrected` (verify), `decided` (human). Same
  resolution again is a `200`; a different one is a `409`.
- Outcomes recorded any other way (HTTP, batch, JSON Lines, Kafka) supersede the waiting
  item in the same transaction as the outcome insert.
- `GET /v1/reviews` (filters: state, question, tier, claimed_by, overdue) shows each item
  with what a reviewer needs (answer, raw and calibrated confidence, answer set, routing
  reason, input). `GET /v1/reviews/stats` gives backlog, overdue, oldest waiting, SLA hit
  rate, median and p90 resolution minutes, and agreement with a 95% Wilson interval, per
  question version and tier.
- `POST /v1/reviews/backfill` for verify and human decisions stored on day 7, before the
  queue existed.
- CLI: `dgate reviews list | claim | release | resolve | stats | backfill`. CI smoke tests
  the queue against PostgreSQL: a superseded item, then claim, resolve and the resulting
  outcome.

**Real run (in the README):** empty PostgreSQL 16 database; 3,000 decisions per question
through the API (all human); four concurrent scripted reviewers answering with the fake's
ground truth resolved all 9,000; refit and routing refresh on those review labels gave the
same thresholds as day 7; then 2,000 more decisions per question, of which 4,024 went to
review and were resolved. 13,024 resolutions in total, no `409`, no item claimed twice.
Verify-tier agreement on `refund.eligible` was 70.8% (95% interval 68.3% to 73.1%),
consistent with the 71.2% the day 7 run measured for that band directly.

**Honest gaps:**

- Reviewers are identified by a free-text `reviewer` field. There is no authentication or
  authorization yet, so anyone who can reach the API can resolve items; this needs to sit
  behind an authenticating proxy until the service has its own auth.
- SLAs are global per tier, not per question; per-question SLAs would belong in the
  routing policy file, but adding fields there changes every policy fingerprint and makes
  all rules stale, so it needs a separate section rather than new policy fields.
- Nothing escalates or alerts on an overdue item yet; the stats show it. Alerts arrive
  with the drift webhook (day 9).
- Review labels are treated as ground truth. A second-reviewer audit (double-labeling a
  sample to measure reviewer error) is not implemented.
- The real run's SLA figures are trivial because the reviewers are a script.

**Tests:** 559 passing locally on Python 3.13 against SQLite and a local PostgreSQL 16
server (67 new), 100% line and branch coverage. New:
enqueueing on decide and on replay; per-tier deadlines; earliest-deadline-first order and
filters; leases that lapse and renew; release rules; ten concurrent claims over six items
on both databases; confirm, correct and decide verdicts; idempotent and conflicting
resolutions; resolving after someone's lease lapsed and without a claim; invalid answers;
superseding by an outcome from elsewhere; overdue filtering and stats with a controlled
clock; backfill; and the CLI workflow and its error exits.

**Next (day 9):** drift monitoring: rolling-window ECE, PSI on the score distribution and
the outcome-label rate, automatic threshold tightening and a webhook alert.

## Day 9 (2026-11-06): drift monitoring

CI on main was green from day 8, so the day went to the milestone.

**Shipped:**

- `decision_gate.drift`: population stability index over quantile bins of the reference
  (open-ended outer bins, merged ties, an epsilon floor so an empty bin cannot make it
  infinite; rank-based, so the same on raw and monotonically calibrated scores),
  categorical PSI for the outcome mix, and the drift policy.
- `decision_gate.drift_monitor`: one check per question version compares the current
  window with a reference window that ends where the active routing rule was selected.
  Four signals: `score_psi` (every decision, no labels needed), `label_psi` (outcome
  mix), `ece` (window's labeled decisions under the active calibrator, judged on the
  lower end of a 95% bootstrap interval) and `auto_error` (labeled decisions above the
  auto threshold; drift only when the lower confidence bound on their error rate is over
  the policy's budget).
- Automatic tightening: on drift, the auto threshold is re-selected on the window's labels
  alone and installed only if stricter; with too few labels or no threshold meeting the
  budget, the auto tier is suspended. The tightened rule keeps the old rule's policy
  fingerprint and calibrator, links to its check (`routing_rules.drift_check_id`), and
  replaces the old rule only if it is still active, in the same transaction as the check.
- A routing refresh no longer undoes a tightening while the latest check still reports
  drift (it keeps the stricter auto threshold); after an `ok` check it may loosen again.
  The router says when the auto tier is suspended and by which check.
- Signed webhook alerts (`decision_gate.alerts`): HMAC-SHA256 signature header, a
  delivery id stable across retries, retries with backoff on timeouts, 5xx and 429.
  Alerts go out on entering drift, on a routing change and on recovery; the outcome
  (`sent`, `failed`, `unconfigured`) is stored on the check.
- Drift settings in a `drift:` section of the routing file with per-question overrides,
  kept out of policy fingerprints so tuning them never invalidates a rule.
- Migration `0006` (`drift_checks`, `routing_rules.drift_check_id`); `POST /v1/drift/check`,
  `GET /v1/drift/status`, `GET /v1/drift/checks[/{id}]`; `dgate drift check` (exit 3 on
  drift, `--at` for backfills) and `dgate drift list`; CI smoke-tests the drift endpoints
  and CLI against PostgreSQL.

**Real run (in the README):** PostgreSQL 16 through the HTTP API. 3,000 decisions per
question from the fake at overconfidence 2.0, all labeled, then refit and refresh. Then
the fake was switched to overconfidence 5.0 and 3,000 more decisions per question were
routed on the old thresholds, labeled only through the review queue (2,174 items). The
check flagged both questions on `score_psi`, `ece` and `auto_error` (for
`refund.eligible`, 17 of 108 labeled auto-tier decisions wrong, at least 10.3% at 95%
confidence against a 10% budget), left `label_psi` quiet (the ground truth did not
change), suspended both auto tiers and delivered two signed alerts. The next 500
decisions per question had no auto decisions, and a refresh that would have re-automated
`refund.eligible` kept the suspension.

**Plan note:** the plan's "outcome-label rate" is implemented as the outcome mix (PSI of
the true answers). The share of decisions that are labeled is reported in every check's
summary (with the tier split) but not alerted on, because it mostly reflects routing and
review throughput rather than drift.

**Honest gaps:**

- Detection power is limited by labels. A test with a model that is wrong about 4% more
  often did not trip `auto_error` or `ece` on 3,000 window decisions; it is tightened only
  when another signal fires. The verify sample rate is the lever.
- ECE and the outcome-mix PSI are computed on labeled decisions, which are mostly the
  review queue's hard cases. Calibration given a score is unaffected by which scores get
  labeled, but the mix is, so `label_psi` can move when routing changes even if the world
  did not. Comparing like with like (both windows labeled the same way) helps; it is not
  enforced.
- Nothing runs the checks on a schedule yet; use cron with `dgate drift check` until the
  Compose worker (day 13).
- Overdue review items still do not alert (noted on day 8); the webhook exists now, so this
  is a small follow-up.
- Recovery is not automatic: an `ok` check sends `drift.recovered` but leaves routing
  tightened until someone runs a refresh, by design.

**Tests:** 640 passing locally on Python 3.13 against SQLite and a local PostgreSQL 16
server (81 new), 100% line and branch coverage. New: hand-computed PSI fixtures,
monotone invariance, open bins and ties; drift config parsing and fingerprint isolation;
each signal's branches; monitor runs for steady traffic, a broken model (tighten, route,
alert), persistent drift without a second alert then recovery, refresh keeping a
suspension and a raised threshold until drift clears, a score shift without labels, labels
that vindicate the threshold, a slightly worse model getting a higher threshold, alert-only
policy, failed and unconfigured alerts, no rule, raw scores, a calibrator that cannot
score, and a rule replaced mid-check; webhook signing, retries and error handling; API and
CLI.

**Next (day 10):** shadow mode: run a candidate backend or prompt version in shadow and
compare calibrated performance before promoting it.

## Day 10 (2026-11-07): shadow mode

CI on main was green from day 9, so the day went to the milestone.

**Shipped:**

- `decision_gate.shadow_runner`: after a new decision is stored, a configured candidate
  backend (`DGATE_SHADOW_BACKEND`, any of fake, http or llm_json, named by
  `DGATE_SHADOW_NAME`) answers the same input in a background task. It never touches the
  response: timeouts, exceptions and invalid answers are stored as `timeout` or `error`.
  Bounded in-flight work (`DGATE_SHADOW_MAX_PENDING`, excess counted as `dropped`), a
  sample rate hashed from the decision id with its own salt (independent of verify
  sampling, so not biased towards any tier), a question filter, in-process counters,
  and a drain with a timeout on shutdown. Idempotent replays are not shadowed again.
- Replay: run the candidate over stored, labeled decisions it has not answered, oldest
  first, with bounded concurrency. Decisions stored without their input (which hold JSON
  `null`, not SQL `NULL`; the query handles both) are counted and skipped.
- Migration `0007` (`shadow_decisions`, one row per decision and candidate, enforced by a
  unique constraint) and `ShadowRepository` (paired rows, per-candidate summaries, replay
  targets).
- `decision_gate.shadow`: the comparison. Pairs are split by time; each side gets its own
  calibrator fitted on the older half (`auto` chooses the method on an inner split of that
  half) and is scored on the newer half: McNemar's exact test and a paired interval for
  accuracy, the router's fixed-sequence threshold selection under the question's routing
  policy for auto-tier coverage, plus raw and calibrated ECE, Brier, log loss and latency
  percentiles. Verdict `promote` / `keep` / `insufficient` with the reasons, including a
  failure-rate gate. The binomial CDF reuses the incomplete beta from `bounds`.
- `GET /v1/shadow/status`, `GET /v1/shadow/decisions/{id}`, `POST /v1/shadow/replay`,
  `GET /v1/shadow/compare`; `dgate shadow status | replay | compare` (compare exits 0, 3 or
  4 by verdict). CI smoke-tests the endpoints and CLI against PostgreSQL with a fake
  candidate on the server.

**Real run (in the README):** PostgreSQL 16 through the HTTP API, 3,000 decisions per
question shadowed live (6,000 submitted and completed, none dropped) and labeled. A
candidate with a higher skill floor but *worse* raw calibration was promoted on both
questions: `refund.eligible` +12.3% accuracy (McNemar p=8.2e-56) and auto coverage 65.1%
against 31.2% at the 10% budget; `ticket.queue` +25.9% and 23.3% against 9.5% at 5%. A
second candidate replayed over the same 3,000 stored decisions (same model, overconfidence
4.0) was kept: raw ECE 0.159 against 0.098, but identical accuracy and coverage after
calibration.

**Plan note:** "before promoting it" is implemented as a verdict, not an automatic switch.
Backends are configured by environment, and the active calibrator and routing rule belong
to the question version rather than to a model, so switching the backend at runtime
without a refit would route the new model's scores through the old model's calibrator.
The README says to refit and refresh routing right after promoting.

**Honest gaps:**

- No calibrator is fitted for the candidate in the registry on promotion; the comparison
  reports its calibrator parameters, but the first refit after the switch should use
  only the new model's decisions. Calibrators keyed by model version would fix this
  properly.
- In the fake, both models share a seed, so the candidate is right wherever the primary
  is (`primary-only 0`). The test handles disagreement in both directions (covered by the
  synthetic-pair tests); the demo just does not show it.
- Auto-tier coverage is selected and evaluated on the same held-out half for both sides.
  It is a fair comparison, not an estimate of production coverage.
- Live shadow calls run in the API process. Under heavy load a separate worker consuming
  decisions from a queue would isolate the candidate further; `max_pending` bounds the
  damage meanwhile.
- Shadow answers are not visible in `GET /v1/decisions/{id}`; they have their own endpoint.

**Tests:** 684 passing locally on Python 3.13 against SQLite and a local PostgreSQL 16
server (44 new), 100% line and branch coverage. New: binomial CDF against exact sums,
McNemar and the paired interval against hand-computed values; comparisons that promote a
better candidate, keep an identical, a worse, an unreliable and a less-automating one,
and report too few pairs; method choice with and without distributions; sampling
independence from verify sampling; backpressure, drain timeouts, timeouts, exceptions,
invalid answers and storage failures in the runner; idempotent storage and replay
(including a lost race and decisions without inputs); settings and the backend factory;
API and CLI, including that `/v1/decide` still answers when the candidate fails.

**Next (day 11):** simulator: synthetic ticket-triage and refund-eligibility datasets with
known ground truth and injected miscalibration and drift; `dgate simulate` produces real
reports.

## Day 11 (2026-11-08): simulator

**Shipped:** `dgate simulate run`, which runs the whole gate on synthetic traffic whose
ground truth is known for every decision, and reports the auto tier's *true* error rate
before and after drift. Nothing is re-implemented: decisions go through the real
`DecisionService` and router, reviews through the queue, labels through outcome recording,
and the refit job, routing refresh and drift monitor run nightly against a real database
on a simulated clock.

- `decision_gate.simulator.datasets`: refund requests judged by a written returns policy
  and support tickets routed by their subject, each item tagged `easy`, `hard` (policy
  boundary, ambiguous wording) or `novel`. Each has a drifted population: holiday order
  ages plus a new marketplace line for refunds; a release that floods the technical queue
  with tickets about a new feature.
- `SimulatedModel`: a `DecisionBackend` that answers each segment with its own skill floor
  and overconfidence (one `FakeBackend` per segment, truth from the dataset). Hard and novel
  items are less accurate but just as confident, so what a score means depends on the mix.
- `decision_gate.simulator.engine`: history (no calibrator, so every decision goes to a
  person whose answer is the label), then refit and first routing rule, steady days, drift
  days. Reviewers resolve within SLA (configurable accuracy); a share of auto decisions get
  their true outcome a few days later; refit and refresh every N days, then the drift check.
- `decision_gate.simulator.report`: per phase tiers, segment mix, raw and calibrated ECE
  and the true auto error with a Clopper-Pearson interval; detection day, action and error
  before and after; events and a daily table. Text, Markdown and JSON.
- `dgate simulate export` prints items as JSON lines.
- To run on a simulated clock, `DecisionService` and `OutcomeService` take a `clock`, the
  routing refresher stamps rules with its clock, and a tightened rule is stamped with the
  check time. `create_app(clock=...)` now passes its clock to all of them, so decisions and
  outcomes are stamped consistently with reviews and drift checks. `DecisionService` also
  takes a `new_id` factory: decision ids drive verify sampling, and the simulator derives
  them from the seed so the same seed gives the same run. `make_engine(...,
  single_connection=True)` keeps one SQLite connection with WAL for long sequential jobs.
- CI runs a short simulation against its own PostgreSQL database and checks the JSON.

**Real run (default settings, seed 7, in the README and
[docs/reports/simulation-seed-7.md](reports/simulation-seed-7.md)):** 200 decisions per
question per day, 21 + 14 + 14 days.

- Refunds: 94.4% automated in the steady phase with a true error of 4.9% (10% budget).
  Drift began on day 36; only the outcome-mix PSI fired, at the end of day 42, and the
  auto tier was suspended. Until then the true auto error was 11.5% (95% CI 9.8%-13.3%).
- Tickets: 67.4% automated at 5.3% (4.3%-6.4%) against a 5% budget. Drift was caught at
  the end of day 39 by the auto-error check and the ECE signal, and the tier suspended; true
  auto error until then 10.9%.
- Raw ECE 0.045 / 0.102 fell to 0.007 / 0.016 after calibration (steady phase).

**What the simulator taught me (worth more than the numbers):**

- The auto-error check has little power against a small overshoot. With 5% verify sampling
  and 10% delayed outcomes, a 1.5-point overshoot on refunds was never proven; the label-mix
  PSI caught the drift instead, a week in. The trade-off (verify rate against detection
  speed) is now measurable instead of guessed.
- The ticket steady-state error sits on the budget rather than comfortably under it. The
  threshold is chosen so the bound holds on the labels it was selected on (fixed-sequence
  testing over all labeled decisions); on fresh traffic the realised rate came out at the
  budget. The report says "not distinguishable" instead of calling a pass or fail on
  noise. Selecting on a held-out split by default would give more margin at the cost of
  coverage; worth measuring with the simulator.
- Order of nightly jobs matters. A refresh that installs a *new* calibrator during drift
  can reopen a suspended auto tier, because a threshold under a new calibrator is not
  comparable with the old floor (`_respect_drift` deliberately skips that case). Running
  the drift check after the refit and refresh, as the simulator does, re-selects the
  threshold on the drift window's labels at once (day 49 of the refund run: refresh reopened at 0.7652, the check re-tightened to 0.7668 on
  the window's labels). A scheduler that runs them in the other order leaves a gap of one
  cycle. Documented here; the Compose stack's worker (day 13) will run them in the safe order.
- The plan's "deliberately injected miscalibration" is per segment rather than a single
  knob: a global overconfidence is removed by any calibrator, so it would not show drift.

**Honest gaps:**

- Throughput is about 9 ms per decision on average on SQLite through the ORM (19,600
  decisions in about three minutes here), so the default run takes a few minutes. Fine for a report; not a load test.
- Reviewers answer instantly and never abandon a claim; lease expiry and backlog under
  load are tested in the review-queue tests, not simulated.
- Only predicate and choice datasets; the score question has no simulated dataset yet.
- Shadow mode is not part of the simulation.

**Tests:** 734 passing locally on Python 3.13 against SQLite and a local PostgreSQL 16
server (50 new), 100% line and branch coverage. New tests cover the policy rules and drifted
populations of both datasets, segment-dependent skill and confidence of the model,
configuration validation, an end-to-end run (history all human, refits, refreshes, nightly
checks, determinism by seed), simulation into a kept SQLite and PostgreSQL database (and
refusal to mix with existing decisions), unreliable reviewers and delayed outcomes, report
findings (false alarms, detection, action, suspension), verdict wording, the CLI and a
clock test for decisions and outcomes over the API.

**Next (day 12):** observability and resilience: Prometheus metrics, OpenTelemetry, backend
timeouts and a circuit breaker that falls back to the human tier, plus rate limits.

## Day 12 (2026-11-09): observability and resilience

**Done**

- `GET /metrics` (Prometheus, per-app registry): HTTP requests and latency by route
  template, decisions by question, tier and replay, backend calls by result (`ok`, `error`,
  `timeout`, `cancelled`, `rejected`) and latency, circuit state and transitions, rate
  limited requests, outcomes by source, build info. Labels are bounded: question ids come
  from the catalog, routes are templates, unknown paths are `unmatched`.
- OpenTelemetry tracing on the API alone (`opentelemetry-api`, a no-op until a provider
  exists): a server span per request that continues the caller's `traceparent` (no extra
  instrumentation package needed), and `dgate.decide` > `dgate.backend.decide`,
  `dgate.route`, `dgate.store` inside it. The `otel` extra plus `DGATE_OTEL_ENABLED`
  installs the SDK with an OTLP/HTTP exporter in `dgate serve`, configured by the
  standard `OTEL_EXPORTER_OTLP_*` variables, and flushes it on shutdown.
- `resilience.py`: a consecutive-failure circuit breaker with a single half-open probe
  (errors, invalid answers and timeouts count; cancellations and caller bugs free the
  probe slot without judging the backend), and a per-key token-bucket rate limiter with
  LRU-bounded keys. Both take an injected monotonic clock, so the tests step time.
- `POST /v1/decide`: the backend call goes through the breaker with the existing time
  limit. An open circuit answers `503` at once with `Retry-After`. `502`, `503` and `504`
  bodies carry `"fallback_tier": "human"`. Rate limits (`DGATE_RATE_LIMIT_PER_S`, burst,
  optional key header) answer `429` with `Retry-After`. `/readyz` reports the circuit.
- CI's API smoke test checks `/metrics` counts a decision and its replay.

**Design decision: the fallback is a signal, not a stored decision.** The plan said the
breaker "falls back to the human tier". I considered storing a human-tier decision with no
answer so it would land in the review queue. I didn't, because a decision row without an answer and
confidence would need a schema change and a filter in every calibration, refit,
threshold, drift and shadow query, and the review flow records the reviewer's answer as an
outcome scored against the model's answer, which would not exist. Instead the failure
response tells the caller to send the case to a person (`fallback_tier: human`). The
contract stays simple: anything but a decision means human. Idempotent replays keep working while the
circuit is open because they never reach the backend.

**Why `/readyz` stays ready with the circuit open:** readiness failing would take every
replica out of the load balancer during a backend outage, which would also stop replays,
outcomes and the review queue, the parts that still work.

**Real output** (in the README): with the `http` backend pointed at a closed port and a
threshold of 3, three `502`s, then `503` with `retry-after: 30`; `/metrics` showed
`result="error"` 3 and `result="rejected"` 1, and `dgate_circuit_state` 2.

**Honest gaps**

- Breaker and limits are per replica and in memory. With N replicas the effective rate
  limit is N times the setting; a shared limit belongs in the gateway.
- No retries on the backend call: a retry inside a 10 s time limit mostly doubles load on
  a struggling backend. The caller can retry with an idempotency key.
- The shadow candidate has its own time limit and in-flight cap, but no breaker of its own yet.
- Not load-tested; no latency numbers are claimed.

**Tests:** 30 new (breaker states, single probe, cancellation, reset on success, token
bucket refill and LRU eviction, metrics and spans through the API, `traceparent`
continuation, `429`/`503`/`504` bodies and headers, breaker disabled, cancelled backend
calls). Ruff, mypy (strict) and the full suite pass locally on Python 3.13 against SQLite
and a local PostgreSQL 16 server: 764 passing, 100% line and branch coverage.

**Next (day 13):** Docker Compose (API, worker running refit, routing refresh and drift
checks in the safe order, PostgreSQL, optional Kafka and an OTel collector),
an end-to-end test against the stack, reliability-diagram reports and a statistics
write-up.
