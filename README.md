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

> **Status:** under active development (day 13 of a 14-day build). See
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
| Platt, isotonic and temperature calibrators, a versioned calibrator registry and a refit job that promotes a fit only if it wins on held-out decisions | ✅ day 5 |
| Automation thresholds that meet a precision target with Clopper-Pearson or Wilson bounds (fixed-sequence testing), and split-conformal prediction sets (LAC, APS) for choice and predicate questions | ✅ day 6 |
| Routing tiers (auto, verify, human) from a per-question YAML policy: an error budget, minimum sample sizes, verify sampling and conformal answer sets; thresholds stored as audited rules that fail closed when the policy or calibrator changes | ✅ day 7 |
| Human review queue: every verify and human decision is enqueued with an SLA deadline; reviewers claim (leased, `SKIP LOCKED`), release and resolve; a resolution is recorded as the decision's outcome in the same transaction; backlog, SLA and agreement stats | ✅ day 8 |
| Drift monitoring: score PSI, outcome-mix PSI, windowed ECE with a bootstrap interval and the auto tier's observed error rate; automatic threshold tightening or suspension that a refresh will not undo while drift lasts; signed webhook alerts | ✅ day 9 |
| Shadow mode: a candidate backend answers live traffic in the background (or replays stored decisions); both are calibrated identically and compared on the same labeled decisions with McNemar's exact test and auto-tier coverage under the routing policy, for a `promote` / `keep` / `insufficient` verdict | ✅ day 10 |
| Simulator: synthetic refund and ticket traffic with known truth, segment-dependent miscalibration and injected drift, run through the real service, review queue, refit, routing refresh and drift monitor on a simulated clock; reports the auto tier's true error rate before and after drift (text, Markdown, JSON) | ✅ day 11 |
| Observability and resilience: Prometheus `/metrics`, OpenTelemetry traces (optional OTLP export), a circuit breaker on the backend that answers `503` with `fallback_tier: human`, and per-caller rate limits on `/v1/decide` | ✅ day 12 |
| Maintenance worker (refit, routing refresh, drift checks in that order, on a timer; a PostgreSQL advisory lock keeps replicas from overlapping), SVG reliability diagrams, a Docker Compose stack (API, worker, PostgreSQL, optional Kafka and OpenTelemetry collector) with an end-to-end test in CI, and a [statistics write-up](docs/STATISTICS.md) | ✅ day 13 |

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
    api -.->|in the background| shadow[Shadow candidate] -.-> store
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
export DGATE_ROUTING_FILE=examples/routing.yaml
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

(Output wrapped for readability; the routing fields are left out here and described under
[Routing tiers](#routing-tiers).) `GET /v1/decisions/{id}` returns a stored decision,
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

### Calibrators and the refit job

Metrics say how wrong the raw confidence is; calibrators fix it. Three are implemented
from scratch in [calibrators.py](src/decision_gate/calibrators.py) (numpy only):

| Method | Fits | Use it when |
| --- | --- | --- |
| `platt` | `p = sigmoid(a * logit(c) + b)` by Newton's method, with Platt's smoothed targets so separable or one-class data still give finite parameters | Little data, or a distortion that is a stretch and shift in log-odds. Works for every question kind. |
| `isotonic` | The best non-decreasing step function (pool-adjacent-violators), interpolated linearly, kept within [0.001, 0.999] | Plenty of data and an irregular distortion. `auto` only tries it with 200+ training decisions. |
| `temperature` | One temperature `T` for the full distribution, `p_i ∝ p_i^(1/T)`, by golden-section search on `1/T` (the log loss is convex there) | Predicate and choice questions with a `probabilities` distribution. Never changes which answer ranks first. |

The refit job (`POST /v1/calibrators/refit`, `dgate calibrators refit`) works per question
version:

1. load the labeled decisions, oldest first, and hold out the newest 30% (`holdout`). A
   chronological split asks the question that matters: does a calibrator fitted on the
   past help on decisions it has never seen?
2. fit every candidate method on the older part (`method=auto`, or name one);
3. score the raw confidences, each candidate and the currently active calibrator on the
   held-out part, and keep the candidate with the lowest held-out log loss;
4. store it as the next version in the registry. It becomes `active` only if it beats
   both the raw confidences and the active calibrator on the same held-out decisions;
   otherwise it is stored as `rejected` with the reason (or `candidate` with
   `activate=false`). Fewer than `min_count` labels (default 100) stores nothing.

The stored calibrator is the one fitted on the training split, so its recorded held-out
numbers describe exactly the model that runs. A partial unique index allows only one
active calibrator per question version, and `POST /v1/calibrators/{id}/activate` rolls
back to any earlier fit.

With the same fake setup as above (1,000 labeled decisions per question from an
overconfident fake, `overconfidence=2.0`, recorded through the API into PostgreSQL):

```console
$ uv run dgate calibrators refit --all
refund.eligible@v1           activated  n=1000   temperature held-out log loss 0.4952 beats raw 0.5502  [v1 eba83f41-2b10-444c-bc04-15628886680d]
ticket.handling_minutes@v1   activated  n=1000   platt held-out log loss 0.4466 beats raw 0.4476  [v1 c0c9562b-0f8d-4ac5-8a14-586976069e71]
ticket.queue@v1              activated  n=1000   temperature held-out log loss 0.5667 beats raw 0.8228  [v1 dd6b5a54-d2cc-438a-9e32-5e603883daa2]
$ uv run dgate calibrators refit refund.eligible --method isotonic
refund.eligible@v1           rejected   n=1000   isotonic held-out log loss 0.5144 does not beat active v1 (temperature) 0.4952  [v2 3298abd8-02b7-43e0-a262-c75815593c3e]
$ uv run dgate calibrators list
question                      ver method       status      train holdout   raw LL   cal LL  cal ECE  id
refund.eligible@v1              2 isotonic     rejected      700     300   0.5502   0.5144    0.056  3298abd8-02b7-43e0-a262-c75815593c3e
refund.eligible@v1              1 temperature  active        700     300   0.5502   0.4952    0.023  eba83f41-2b10-444c-bc04-15628886680d
ticket.handling_minutes@v1      1 platt        active        700     300   0.4476   0.4466    0.041  c0c9562b-0f8d-4ac5-8a14-586976069e71
ticket.queue@v1                 1 temperature  active        700     300   0.8228   0.5667    0.043  dd6b5a54-d2cc-438a-9e32-5e603883daa2
```

The fitted temperature for `ticket.queue` is 2.15, close to the fake's built-in
overconfidence of 2 (for the fake's distributions, overconfidence `k` in log-odds is
exactly temperature `k`). Held-out ECE drops from 0.225 to 0.043. The score question shows
the honest flip side: on these 300 held-out decisions Platt scaling barely beats the raw
scores (0.4466 against 0.4476) and is still promoted, because the rule is "strictly
better". A minimum improvement margin is on the list for the routing policy (day 7).

`dgate fake-calibrate` runs the same comparison without a database, on fresh fake
decisions (5,000 per question, 30% held out):

```console
$ uv run dgate fake-calibrate -q examples/questions.yaml
question                         method           ECE   Brier  logloss  parameters
refund.eligible@v1               raw            0.087   0.170   0.5379
                                 platt          0.027   0.161   0.4849  a=0.439 b=0.102
                                 isotonic       0.024   0.161   0.4879  145 knots
                                 temperature    0.022   0.161   0.4849  temperature=2.113
ticket.queue@v1                  raw            0.185   0.230   0.7556
                                 platt          0.026   0.187   0.5509  a=0.480 b=-0.606
                                 isotonic       0.036   0.188   0.5537  152 knots
                                 temperature    0.025   0.187   0.5508  temperature=1.974
ticket.handling_minutes@v1       raw            0.114   0.191   0.6174
                                 platt          0.026   0.176   0.5241  a=0.503 b=0.049
                                 isotonic       0.033   0.177   0.5335  212 knots
```

Refit options: `question`, `version` (default: latest in the catalog), `method` (`auto`,
`platt`, `isotonic`, `temperature`), `holdout` (fraction, default 0.3), `min_count`
(default 100), `since`/`until` (decision time) and `activate` (default true). The CLI adds
`--all` to refit every question version that has labels, which is what a scheduled job
runs. Hand-worked fixtures, optimality checks and recovery of known distortions are in
[tests/test_calibrators.py](tests/test_calibrators.py). Calibrated probabilities are not yet applied to `/v1/decide` responses; that comes
with the routing tiers on day 7.

### Thresholds and conformal sets

A calibrated probability says how often a decision like this one is right. The routing
question is different: *from which score up can decisions be automated, so that the
automated ones are right at least 95% of the time, and how sure are we of that?*

**Threshold selection** (`POST /v1/thresholds/select`, `dgate thresholds select`) answers
it with a statistical guarantee, implemented from scratch in
[bounds.py](src/decision_gate/bounds.py) and [thresholds.py](src/decision_gate/thresholds.py):

- A decision is accepted when its score is at least the threshold. For each candidate
  threshold, `k` of the `n` accepted decisions were correct, and a one-sided lower
  confidence bound on that proportion is computed: **Clopper-Pearson** (exact, from the
  beta quantile, never below its nominal coverage) by default, or **Wilson** (closed form,
  tighter, but its coverage dips below nominal for some proportions, which a test
  demonstrates).
- Picking "the lowest threshold whose bound clears the target" after looking at all of
  them would be cherry-picking. Candidates are instead tested as a **fixed sequence**, from
  the highest threshold down, and the search stops at the first one whose bound misses
  the target. The order depends only on the scores, never on the labels, so the
  family-wise error stays at `1 - confidence` with no multiplicity correction.
- The guarantee: if the labeled decisions are representative (i.i.d.) of the traffic the
  threshold will run on, then with probability at least `confidence` the automated
  decisions have precision at least `target_precision`. Drift breaks that assumption;
  detecting it is day 9.
- `min_accepted` (default 100) is the smallest group a candidate may accept. It also sets
  where the sequence starts: starting on 30 decisions, a single error (29/30, bound 0.85)
  would end the search with nothing selected.

**Conformal prediction sets** (`POST /v1/conformal/fit`, `dgate conformal fit`,
[conformal.py](src/decision_gate/conformal.py)) cover the other case: instead of one answer,
return a set of labels that contains the true one with probability at least `1 - alpha`,
however miscalibrated the model is. The quantile is the `ceil((n + 1)(1 - alpha))`-th
smallest nonconformity score of `n` calibration decisions (too few decisions gives an
infinite quantile: every label is in every set, and the response says so).

| Score | Nonconformity of label `y` | Behaviour |
| --- | --- | --- |
| `lac` (default) | `1 - p(y)` | Smallest sets on average. Can be empty when the model is unsure of everything, which routing treats as "send to a human". |
| `aps` | `p(y)` plus the mass of every more probable label | Sets adapt to difficulty and always include the top label. Larger on average, and of little use for two-label (predicate) questions, where it tends to return both labels. |

A singleton set is a confident answer; a larger set narrows down what a reviewer has to
consider. Both feed the routing policy on day 7.

Both analyses score decisions with the **active calibrator** (`use_calibrator`, default
true) and skip the decisions it was trained on, since a score function evaluated on its
own training data looks better than it is. The remaining decisions are split by time: the
older 70% select the threshold (or calibrate the quantile) and the newest 30% (`holdout`)
check what it actually achieves. Only temperature scaling changes the distributions that
conformal sets use; Platt and isotonic recalibrate just the decided answer.

Real output, from 3,000 labeled decisions per question from the overconfident fake
(`overconfidence=2.0`), recorded through the API into PostgreSQL 16, after
`dgate calibrators refit --all` activated temperature scaling (`refund.eligible`,
`ticket.queue`) and Platt scaling (`ticket.handling_minutes`):

```console
$ uv run dgate thresholds select refund.eligible --target-precision 0.9 --curve 6
refund.eligible@v1: 3000 labeled, scores from calibrator v1 (temperature); 630 used, 270 held out, 2100 skipped (calibrator training data)
accept score >= 0.8668: 24.4% of decisions, precision 0.9481 (146/154), 95% lower bound 0.9082 >= target 0.9000; the next candidate 0.8613 failed
held out: 63/270 accepted (23.3%), precision 0.9048
 threshold  accepted  coverage  precision    bound
    0.9049       100     15.9%     0.9800   0.9384
    0.8668       154     24.4%     0.9481   0.9082  <- selected
    0.8613       159     25.2%     0.9371   0.8957
    0.8189       207     32.9%     0.9179   0.8794
    0.7391       314     49.8%     0.8599   0.8235
    0.6655       416     66.0%     0.8221   0.7884
    0.5898       523     83.0%     0.7801   0.7482
    0.5024       630    100.0%     0.7556   0.7257
$ uv run dgate thresholds select ticket.queue --target-precision 0.9
ticket.queue@v1: 3000 labeled, scores from calibrator v1 (temperature); 630 used, 270 held out, 2100 skipped (calibrator training data)
even the highest threshold 0.8678 has a 95% lower bound of 0.8482 on precision (91/100), below the target 0.9000
$ uv run dgate thresholds select ticket.queue --target-precision 0.9 --raw
ticket.queue@v1: 3000 labeled, scores from raw; 2100 used, 900 held out
accept score >= 0.9916: 19.2% of decisions, precision 0.9330 (376/403), 95% lower bound 0.9088 >= target 0.9000; the next candidate 0.9907 failed
held out: 169/900 accepted (18.8%), precision 0.9290
$ uv run dgate conformal fit refund.eligible --alpha 0.1
refund.eligible@v1: 3000 labeled, scores from calibrator v1 (temperature); 630 used, 270 held out, 2100 skipped (calibrator training data)
lac sets with quantile 0.6960 cover the true label with probability >= 90%
held out: coverage 0.904 (target 0.900), mean set size 1.43, singletons 56.7% (accuracy 0.830), empty 0.0%; sizes 1:153 2:117
$ uv run dgate conformal fit ticket.queue --alpha 0.1
ticket.queue@v1: 3000 labeled, scores from calibrator v1 (temperature); 630 used, 270 held out, 2100 skipped (calibrator training data)
lac sets with quantile 0.8914 cover the true label with probability >= 90%
held out: coverage 0.859 (target 0.900), mean set size 2.78, singletons 55.6% (accuracy 0.747), empty 0.0%; sizes 1:150 5:120
```

What this shows, warts included:

- For `refund.eligible` the calibrated scores now read as probabilities: automate at 0.87
  rather than at a raw 0.99-something. On the held-out decisions the threshold automated
  23% of traffic at precision 0.905, just above the 0.9 target.
- For `ticket.queue`, skipping the calibrator's 2,100 training decisions leaves 630 to
  select on, too few to prove 90% precision; on raw scores, with all 2,100 older
  decisions, the same target is met at 19% automation. Excluding training data is the
  cautious choice for any calibrator, but for a strictly increasing one (Platt, or
  temperature scaling on the fake's distributions, where the other options share the
  remaining probability equally) the accepted sets are exactly those of the raw scores, so
  the exclusion costs power without buying validity. Since day 7, decisions are reused
  when the calibrator cannot reorder them (see [Routing tiers](#routing-tiers)); this run
  is kept as it was recorded on day 6.
- The `ticket.queue` held-out coverage of 0.859 is below the 0.9 target. The guarantee is
  on average over calibration sets; with 630 calibration and 270 held-out decisions the
  standard deviation of this number is about 0.02, so this run is roughly two standard
  deviations low. The test suite checks the average over 150 synthetic runs instead
  ([tests/test_conformal.py](tests/test_conformal.py)). Sets are size 1 or 5 because the
  fake spreads the remaining probability evenly over the other options.

`dgate fake-gate` runs the whole pipeline without a database: on 6,000 fresh fake decisions
per question, 40% fit the calibrator, 30% select and 30% check.

```console
$ uv run dgate fake-gate -q examples/questions.yaml --target-precision 0.9
refund.eligible@v1  (calibrator: temperature, target precision 0.9)
  raw         threshold 0.9664  selection: 29.4% automated, bound 0.9027  check: 32.5% automated, precision 0.9385
  calibrated  threshold 0.8341  selection: 29.4% automated, bound 0.9027  check: 32.5% automated, precision 0.9385
  lac         sets for 90% coverage: quantile 0.6842  check: coverage 0.896, mean size 1.36, singletons 63.7%
  aps         sets for 90% coverage: quantile 1.0000  check: coverage 1.000, mean size 2.00, singletons 0.0%
ticket.queue@v1  (calibrator: temperature, target precision 0.9)
  raw         threshold 0.9966  selection: 14.2% automated, bound 0.9017  check: 13.0% automated, precision 0.9487
  calibrated  threshold 0.8896  selection: 14.2% automated, bound 0.9017  check: 13.0% automated, precision 0.9487
  lac         sets for 90% coverage: quantile 0.8974  check: coverage 0.909, mean size 3.07, singletons 48.2%
  aps         sets for 90% coverage: quantile 0.9196  check: coverage 0.993, mean size 4.50, singletons 12.5%
ticket.handling_minutes@v1  (calibrator: platt, target precision 0.9)
  raw         no threshold: even the highest threshold 0.9989 has a 95% lower bound of 0.8725 on precision (93/100), below the target 0.9000
  calibrated  no threshold: even the highest threshold 0.9744 has a 95% lower bound of 0.8725 on precision (93/100), below the target 0.9000
```

Raw and calibrated rows select the same decisions because both calibrators here are
strictly increasing; calibration changes what the threshold *means*, not the ranking.

Threshold options: `question`, `version`, `target_precision` (default 0.95) or
`max_error_rate`, `confidence` (0.95), `bound` (`clopper-pearson`, `wilson`),
`min_accepted` (100), `holdout` (0.3), `since`/`until`, `use_calibrator` (true) and
`curve_points` (10). Conformal options: `question`, `version`, `alpha` (0.1), `method`
(`lac`, `aps`), `holdout`, `since`/`until` and `use_calibrator`; the response includes the
`params` needed to rebuild the predictor. These endpoints only analyse; routing rules
(below) store the thresholds and quantile that decisions are actually routed with.

### Routing tiers

Every decision gets a tier, decided before it is stored and returned with it:

| Tier | Meaning | When |
| --- | --- | --- |
| `auto` | Use the answer as is | The calibrated probability is at or above the rule's auto threshold, and the decision was not sampled |
| `verify` | A person confirms the proposed answer | At or above the verify threshold but below auto, or sampled from the automatable decisions (`verify_sample_rate`) |
| `human` | A person decides from scratch | Everything the gate cannot vouch for: no policy, not calibrated, too few labels, a stale rule, or a score below every threshold |

The policy is configuration ([examples/routing.yaml](examples/routing.yaml)); the
thresholds are data. `POST /v1/routing/refresh` (or `dgate routing refresh`) selects them
from labeled decisions with the fixed-sequence test above and stores them as the active
**routing rule** for each question version, keeping the old ones for audit:

```yaml
defaults:
  confidence: 0.95          # confidence that the auto tier meets its error budget
  min_labeled: 200          # nothing is automated on fewer eligible labeled decisions
  min_accepted: 100         # a threshold must accept at least this many of them
  verify_sample_rate: 0.05  # 5% of automatable decisions go to verify anyway, for labels
  require_calibrator: true  # route to a human until a calibrator is active
questions:
  refund.eligible:
    max_error_rate: 0.10        # auto tier: at most 10% wrong
    verify_max_error_rate: 0.25 # looser budget for the verify threshold
    conformal_alpha: 0.10       # answer sets for the people handling the rest
```

Design choices worth calling out:

- **Fail closed, and say why.** A routing problem never fails the request: the decision
  goes to `human` with a `reason` (`refund.eligible@v1 is not calibrated yet`, `the active
  calibrator changed since the thresholds were selected`, ...).
- **A threshold is only valid on the scores it was selected on.** Each rule records the
  policy fingerprint and the calibrator it was selected under. A refit that activates a
  new calibrator, a rollback, or an edited policy makes the rule stale, and every decision
  goes to a human until the rule is refreshed. `GET /v1/routing/status` shows which
  questions are `current` and which are not.
- **Verify sampling keeps the labels honest.** If only decisions people looked at got
  labels, calibration would only ever see the hard cases. A fixed fraction of automatable
  decisions is sent to `verify`; the draw is a hash of the decision id, so it is
  reproducible and auditable.
- **Replays are stable.** The tier is stored with the decision, so an idempotent retry
  returns the original tier even if the rule changed in between.
- **Training decisions are reused when that is safe.** A strictly increasing calibrator
  (Platt with a positive slope, temperature scaling on two labels) cannot change which
  decisions a threshold accepts, so threshold selection uses its training decisions too.
  Isotonic regression, and temperature scaling over three or more labels, keep the
  exclusion. The analysis responses report both counts (`excluded_training`,
  `reused_training`).

A real run against PostgreSQL 16: 3,000 decisions per question recorded through the API
with their fake ground truth, `dgate calibrators refit --all`, then a refresh with the
example policies:

```console
$ uv run dgate routing refresh
refund.eligible@v1: auto 0.8688, verify 0.5379, answer sets yes (3000 of 3000 labeled decisions used)
  scores from calibrator v1 (temperature); auto: accept score >= 0.8688: 24.8% of decisions, precision 0.9207 (685/744), 95% lower bound 0.9025 >= target 0.9000; the next candidate 0.8645 failed; verify: accept score >= 0.5379: 93.2% of decisions, precision 0.7639 (2135/2795), 95% lower bound 0.7503 >= target 0.7500; the next candidate 0.5340 failed; answer sets: lac sets with quantile 0.6902 cover the true label with probability >= 90%
ticket.queue@v1: auto none, verify 0.6554, answer sets yes (900 of 3000 labeled decisions used)
  scores from calibrator v1 (temperature); auto: none (even the highest threshold 0.8991 has a 95% lower bound of 0.9108 on precision (96/100), below the target 0.9500); verify: accept score >= 0.6554: 39.9% of decisions, precision 0.8384 (301/359), 95% lower bound 0.8030 >= target 0.8000; the next candidate 0.6488 failed; answer sets: lac sets with quantile 0.8980 cover the true label with probability >= 90%
ticket.handling_minutes@v1: auto 0.6109, verify none, answer sets no (3000 of 3000 labeled decisions used)
  scores from calibrator v1 (platt); auto: accept score >= 0.6109: 78.5% of decisions, precision 0.8162 (1923/2356), 95% lower bound 0.8026 >= target 0.8000; the next candidate 0.6046 failed
```

A decision now carries its routing:

```console
$ curl -s -X POST localhost:8080/v1/decide -H 'content-type: application/json' \
    -d '{"question": "refund.eligible", "input": {"order_age_days": 12, "item": "headphones"}}'
{"id":"2e1fd38f-97b2-431d-bd3a-402cccb672db","question":"refund.eligible","question_version":1,
 "answer":true,"confidence":0.7618304815842805,"calibrated_confidence":0.6344386674137995,
 "tier":"verify","reason":"calibrated probability 0.6344 >= verify threshold 0.5379",
 "answer_set":[true,false],"routing_rule_id":"a3455b09-3cf0-40b4-9982-ea0458ef5bb4", ...}
```

Then 2,000 fresh decisions per question, scored against the fake's ground truth (which the
gate never saw for these):

| Question | auto | verify | human | Answer sets |
| --- | --- | --- | --- | --- |
| `refund.eligible` (budget 10%) | 21.4%, accuracy 0.953 | 70.9%, accuracy 0.712 | 7.8%, accuracy 0.542 | mean size 1.53, contain the truth 90.5% |
| `ticket.queue` (budget 5%) | none | 41.9%, accuracy 0.840 | 58.1%, accuracy 0.448 | mean size 3.01, contain the truth 91.0% |
| `ticket.handling_minutes` (budget 20%) | 73.4%, accuracy 0.793 | 3.5%, accuracy 0.843 | 23.2%, accuracy 0.603 | n/a (score question) |

Reading it honestly:

- `refund.eligible` automates a fifth of the traffic at 95% accuracy against a 90% target.
- `ticket.handling_minutes` automated at 0.793 against a 0.80 target. The guarantee is
  that, with 95% confidence, the *long-run* precision above the threshold is at least
  0.80; one sample of 1,467 decisions has a standard error of about 0.011, so this is
  0.7 standard errors low, which is unremarkable. The selection itself was tight (lower
  bound 0.8026 against 0.80).
- `ticket.queue` automates nothing: 900 eligible decisions cannot prove a 5% error rate,
  and temperature scaling over five labels may reorder decisions, so its 2,100 training
  decisions stay excluded. More labels, or a looser budget, would change that.
- The verify threshold bounds the precision of *everything at or above it*, not of the
  verify band alone: for `refund.eligible` the band itself ran at 0.712, while auto and
  verify together ran at 0.768, above the 0.75 target. A verify threshold says "worth a
  quick check", not "almost right".

Policy fields (all optional): `max_error_rate` (0.05), `confidence` (0.95), `bound`
(`clopper-pearson`), `min_labeled` (200), `min_accepted` (100), `require_calibrator`
(true), `verify_max_error_rate` (unset: no verify threshold), `verify_sample_rate` (0.05),
`conformal_alpha` (unset: no answer sets) and `conformal_method` (`lac`). `defaults` is
merged into every question's policy; a `default` policy covers questions that are not
listed (without one they always go to a human). `dgate routing check` validates a file
against the catalog and flags policies for unknown questions. A running server caches each
question's rule and calibrator for `DGATE_ROUTING_CACHE_TTL_S` seconds; a refresh, refit or
activation through the API clears the cache at once.

### Review queue

Routing decides who handles a decision; the review queue is where people handle it. Every
`verify` and `human` decision gets a review item **in the same transaction as the
decision**, so nothing a person must see can be stored without its place in the queue.

```text
 decide ──► verify / human ──► review item (open, due in 60m / 4h)
                                   │ claim (lease, 15m)        ▲ lease lapses or release
                                   ▼                           │
                                claimed ───────────────────────┘
                                   │ resolve {outcome | confirm}
                                   ▼
                    resolved ──► outcome (source "review") ──► refit, refresh
 an outcome from anywhere else ──► superseded (leaves the queue)
```

- **Earliest deadline first.** `POST /v1/reviews/claim` hands out the available item with
  the earliest SLA deadline, so overdue items come first and a 60-minute verify item is
  served before a 4-hour human item that arrived at the same time. Filter by `question`
  or `tier` to build per-team queues.
- **Claims are leases.** A claim lasts `DGATE_REVIEW_CLAIM_MINUTES` (or `lease_minutes`
  per request); an item abandoned mid-review returns to the queue on its own, and the
  holder can renew by claiming again. `claims` counts how often an item changed hands.
- **No double work.** On PostgreSQL the next item is picked with `FOR UPDATE SKIP LOCKED`,
  so concurrent reviewers get different items instead of queueing on one row; every state
  change is a conditional `UPDATE`, so a lost race is a `409`, never a silent overwrite.
- **Resolving records the truth.** `POST /v1/reviews/{id}/resolve` takes the correct
  answer (`outcome`) or `confirm: true` for the backend's answer, validates it against the
  question, and writes the outcome (`source: "review"`) and the resolution in one
  transaction. Verify items are `confirmed` or `corrected`, human items `decided`; every
  item records whether the backend was right (`agreed`). Retrying the same resolution is a
  `200`; a different answer for a resolved item is a `409`, because rewriting a label would
  rewrite what calibration was fitted on.
- **Labels from elsewhere win the race.** If an outcome arrives over HTTP or Kafka first
  (a chargeback, a ticket that was actually moved), the item is `superseded` in the same
  transaction and leaves the queue.
- **The verify sample closes the loop without bias.** Review labels come mostly from the
  hard cases. The `verify_sample_rate` slice of automatable decisions is what keeps
  labels flowing for the auto band, which is why the router samples it.

`GET /v1/reviews/stats` (and `dgate reviews stats`) reports, per question version and tier:
the backlog (available, claimed, overdue, oldest waiting), SLA performance (share resolved
on time, median and p90 minutes from decision to resolution) and how often reviewers agreed
with the backend, with a 95% Wilson interval. Agreement on the verify tier is a direct,
continuously measured check of the band's real accuracy.

A real run against PostgreSQL 16, starting from an empty database: 3,000 decisions per
question through the API (all `human`, nothing is calibrated), worked off by four
concurrent scripted reviewers that answer with the fake's ground truth, then
`refit` and `routing refresh` on **labels that all came from the review queue**, then
2,000 more decisions per question and the same reviewers again:

```console
$ uv run dgate reviews stats --since 2026-11-05T10:08:30
SLA: verify 60m, human 4.0h, claims last 15m
queue                                 avail  claim  late  oldest   done  on-time  median    p90  agree
refund.eligible@v1 human                  2      0     0      0m    170   100.0%      1m     1m  59.4%
refund.eligible@v1 verify                15      0     0      0m   1358   100.0%      0m     0m  70.8%
ticket.handling_minutes@v1 human          3      0     0      0m    417   100.0%      1m     1m  53.2%
ticket.handling_minutes@v1 verify         1      0     0      0m     79   100.0%      0m     0m  87.3%
ticket.queue@v1 human                    14      0     0      0m   1179   100.0%      1m     1m  43.0%
ticket.queue@v1 verify                    6      0     0      0m    821   100.0%      0m     0m  85.1%

$ uv run dgate reviews claim --reviewer ann --question ticket.queue
b926aba6-b8a4-4939-bba7-609810548c2f  ticket.queue@v1              verify claimed by ann  due 2026-11-05 11:09Z  answer "other" (calibrated 0.776)
  answer set: ["other"]
  input: {"item": 200004}
```

What it shows, and what it does not:

- The loop closes: the 13,024 outcomes in that database all have `source: "review"`, and
  the calibrators and routing rules fitted on them are the same ones the day 7 run fitted
  on labels posted directly (same thresholds: 0.8688 auto and 0.5379 verify for
  `refund.eligible`).
- Four reviewers claimed concurrently and resolved 13,024 items with no `409`, and no
  item was claimed more than once.
- Reviewer agreement separates the tiers as the thresholds intend: on `refund.eligible`
  the verify band ran at 70.8% (95% interval 68.3% to 73.1%; the day 7 run measured the
  same band at 71.2%), the human band at 59.4%.
- The SLA columns are trivially perfect here because the reviewers are a script: the
  median human item waited about a minute, behind the verify items whose deadlines were
  earlier. The 20 decisions left at the end are the backlog.

`POST /v1/reviews/backfill` (or `dgate reviews backfill`) enqueues verify and human
decisions recorded before the queue existed; their deadlines count from the decision time,
so they arrive overdue, which is the truth about how long they waited.

### Drift monitoring

Thresholds are selected on yesterday's traffic and applied to today's. A drift check
compares two windows of stored decisions for one question version:

- the **current window**: the last `window_hours` (default a week);
- the **reference window**: `reference_hours` long (default four weeks), ending where the
  active routing rule was selected, or where the current window starts if that is
  earlier. The thresholds in force were chosen on traffic like the reference, so that is
  what today's traffic has to resemble.

Four signals, each `ok`, `drift` or `insufficient` (too little data to say):

| Signal | What it measures | Needs labels | Fires when |
| --- | --- | --- | --- |
| `score_psi` | Population stability index of the backend's raw confidence over every decision, in quantile bins of the reference. Rank-based, so it reads the same on calibrated scores | no | PSI > `max_score_psi` (0.25) |
| `label_psi` | PSI of the true answers among labeled decisions (the outcome mix). Not for score questions | yes | PSI > `max_label_psi` (0.25) |
| `ece` | Calibration error of the window's labeled decisions, scored with the active calibrator, with a 95% bootstrap interval | yes | the interval's *lower* end > `max_ece` (0.05): small samples inflate ECE, so the point estimate alone would cry wolf |
| `auto_error` | Error rate of the window's labeled decisions that clear the active auto threshold. These labels come mostly from the random verify sample, so they are an unbiased look at the auto tier | yes | the lower confidence bound on the error rate (the policy's bound and confidence) exceeds the policy's `max_error_rate`: the budget is *confidently* broken |

A check is `drift` if any signal drifted. With `action: tighten` (the default) the monitor
then re-selects the auto threshold **on the current window's labels alone** and installs
it only if it is stricter than the one in force. When the window has too few labels to
select anything (a score shift seen before labels arrive), or no threshold meets the
budget on recent traffic, the auto tier is **suspended**: the new rule has no auto
threshold, so those decisions go to verify or to a human. If the window's labels support
the current threshold, nothing changes (`not_tightened`). The tightened rule keeps the
policy fingerprint and calibrator of the rule it replaces, so the router applies it at
once; it records which check installed it (`drift_check_id`), and it replaces the old rule
only if that rule is still the active one, in the same transaction as the check is stored.

A routing refresh selects on every label, most of them from before the drift, so on its
own it would undo the tightening. While the latest check for a question still says
`drift`, a refresh keeps the stricter of the two auto thresholds. Once a check comes back
`ok`, the next refresh may loosen again.

**Alerts.** Set `DGATE_DRIFT_WEBHOOK_URL` and the monitor POSTs JSON when a question enters
drift, when it changes routing, and when a drifting question checks out `ok` again
(`drift.detected`, `drift.recovered`); repeated checks of the same drift stay quiet. With
`DGATE_DRIFT_WEBHOOK_SECRET` set, `X-Dgate-Signature: sha256=<hex>` is the HMAC-SHA256 of
the body. `X-Dgate-Delivery` is unique per alert and stable across retries; timeouts, 5xx
and 429 are retried with backoff. Whether the alert was `sent`, `failed` or
`unconfigured` is stored on the check.

Drift settings live in a `drift:` section of the routing file (see
[`examples/routing.yaml`](examples/routing.yaml)), with per-question overrides. They are
deliberately not part of any routing policy, so tuning a limit never changes a policy
fingerprint and never sends traffic to a human. Run checks on a schedule with
`dgate drift check` (exit status 3 when anything drifts, so a cron job can flag it) or
`POST /v1/drift/check`.

A real run against PostgreSQL 16, through the HTTP API. Phase 1: 3,000 decisions per
question from the fake backend at its default miscalibration, every one labeled by an
outcome event, then `refit` and `routing refresh`. Phase 2: the backend is "updated" to a
much more overconfident model (`DGATE_FAKE_OVERCONFIDENCE` 2.0 to 5.0) and 3,000 more
decisions per question are routed with the phase 1 thresholds; the only labels are the
review queue's, resolved by a scripted reviewer (2,174 items: the verify and human tiers
plus the 5% verify sample of automatable decisions). Then one drift check, with windows
shrunk to fit a run that took minutes and a local webhook receiver:

```console
$ curl -s -X POST localhost:8088/v1/drift/check -H 'content-type: application/json' -d '{}'
refund.eligible  drift  action suspended  alert sent (drift.detected)
  score_psi   drift  confidence PSI 0.852 > 0.25 (mean 0.848 -> 0.934)
  label_psi   ok     outcome PSI 0.000 <= 0.25 (largest change: 'true' 48.1% -> 48.7%)
  ece         drift  ECE 0.092 (95% interval 0.067 to 0.118) on 918 labeled decisions,
                     lower end above 0.05; mean calibrator v1 score 0.697 vs accuracy 0.605
  auto_error  drift  17 of 108 labeled decisions above the auto threshold 0.8054 were wrong
                     (15.7%); at 95% confidence the error rate is at least 10.3%, above the
                     10.0% budget
  -> auto tier suspended (was 0.8054): no threshold on the window's labels: even the highest
     threshold 0.8501 has a 95% lower bound of 0.8013 on precision (87/100), below the
     target 0.9000
ticket.queue  drift  action suspended  alert sent (drift.detected)
  score_psi   drift  confidence PSI 1.267 > 0.25 (mean 0.792 -> 0.924)
  label_psi   ok     outcome PSI 0.002 <= 0.25 (largest change: 'other' 19.5% -> 20.9%)
  ece         drift  ECE 0.238 (95% interval 0.214 to 0.264) on 1256 labeled decisions,
                     lower end above 0.05; mean calibrator v1 score 0.617 vs accuracy 0.379
  auto_error  drift  16 of 77 labeled decisions above the auto threshold 0.8888 were wrong
                     (20.8%); at 95% confidence the error rate is at least 13.5%, above the
                     5.0% budget
  -> auto tier suspended (was 0.8888): no threshold on the window's labels: even the highest
     threshold 0.8828 has a 95% lower bound of 0.6687 on precision (75/100), below the
     target 0.9500
```

(The JSON response, condensed to its signal messages.) The receiver got two
`drift.detected` alerts, both with valid signatures. The next 500 decisions per question:

| Question | Tiers in phase 2 (old thresholds) | Tiers after the check |
| --- | --- | --- |
| `refund.eligible` | 2,082 auto, 845 verify, 73 human | 0 auto, 488 verify, 12 human |
| `ticket.queue` | 1,744 auto, 605 verify, 651 human | 0 auto, 405 verify, 95 human |

A `routing refresh` right after the check, on all 4,000-odd labels, would have
re-automated `refund.eligible`; it kept the suspension instead:
`auto threshold suspended: drift check 52cc9d50-... still reports drift; scores from
calibrator v1 (temperature); auto: accept ...`.

What the run does and does not show:

- Phase 2 automated 2,082 `refund.eligible` decisions with a threshold that no longer
  held: on the 108 that were labeled, 15.7% were wrong against a 10% budget. The check
  catches that from the verify sample alone, without labeling the auto tier.
- The label mix did not move (the fake's ground truth is the same in both phases), and
  `label_psi` correctly stayed quiet.
- The ECE is measured on labeled decisions, which here are mostly the hard cases the
  review queue sees. Calibration *given a score* is unaffected by which scores get
  labeled, but the ECE weights its bins by the labeled mix, so it is not the ECE of all
  traffic. The `auto_error` check is the one tied to the error budget.
- The windows were minutes long because the whole run was; the defaults are a week
  against the four weeks before the rule.

### Shadow mode

Swapping the model behind a question (a new version, a new prompt, another vendor) is the
riskiest change this service sees: the calibrator and the thresholds were fitted to the old
model's scores. Shadow mode answers "is the candidate better?" with the same evidence the
gate already collects, before anything is switched.

**Collecting.** Set `DGATE_SHADOW_BACKEND` (and `DGATE_SHADOW_NAME`, e.g. `prompt-v3`).
After each new decision is stored, the candidate is asked the same input in a background
task. It never affects the response: a slow, failing or invalid candidate is recorded as
`timeout` or `error`, and `/v1/decide` answers exactly as before. At most
`DGATE_SHADOW_MAX_PENDING` calls are in flight; beyond that decisions are not shadowed
(counted as `dropped`) rather than queued without bound. `DGATE_SHADOW_SAMPLE_RATE`
shadows a fraction of decisions by a hash of the decision id with its own salt, so the
sample is reproducible and independent of verify sampling. Idempotent replays are not
shadowed twice; each candidate answers each decision at most once.

To compare without waiting for new traffic, `dgate shadow replay` (or `POST
/v1/shadow/replay`) runs the candidate over stored decisions that already have outcomes,
oldest first. It needs stored inputs (`DGATE_STORE_INPUTS`); decisions without one are
counted and skipped.

**Comparing.** A candidate's answers are scored against the primary decision's outcome,
so both models are judged on the same inputs and the same ground truth. Raw confidences
are not comparable across models, so each side is treated identically: the paired
decisions are split by time, a calibrator is fitted to each side on the older half (`auto`
picks the method with an inner split of that half, never by peeking at the rest), and the
newer half is scored:

- **Accuracy**, with McNemar's exact test: only the decisions where exactly one model was
  right carry information, which makes paired comparison far more sensitive than two
  separate accuracy figures. Reported with a paired 95% interval.
- **Auto-tier coverage**: the router's own fixed-sequence threshold selection under the
  question's routing policy, on each side's calibrated scores. Both sides are selected and
  evaluated on the same held-out half, which flatters both equally; the difference is what
  counts.
- Raw and calibrated ECE, Brier and log loss, and latency percentiles, for information.
  Per-answer Brier is not a verdict input: each model forecasts the correctness of its own
  answer, which are different events.

The verdict is `promote` only if the candidate's failure rate is within
`max_failure_rate` (1%), it is not significantly less accurate, it does not automate more
than `min_coverage_gain` (2 points) less, and it is either significantly more accurate or
automates at least that much more. Otherwise `keep`, with the reasons; `insufficient`
below `min_paired` (200) labeled pairs. `dgate shadow compare` exits 0, 3 or 4 for those, so
a pipeline can gate a rollout on it. Promotion itself is a configuration change
(`DGATE_BACKEND`); refit the calibrator and refresh routing right after it.

**Real run** (PostgreSQL 16, through the HTTP API): the primary is the fake at
overconfidence 2.0; the candidate `fake-2` is a fake with a higher skill floor
(`DGATE_SHADOW_FAKE_MIN_SKILL=0.75`) and overconfidence 3.0. 3,000 decisions per question,
every one shadowed live (6,000 submitted, 6,000 completed, none dropped), then all labeled
through `/v1/outcomes/batch`:

```console
$ dgate shadow compare refund.eligible --candidate fake-2
refund.eligible@v1 primary vs fake-2: PROMOTE (3000 paired: 1500 train, 1500 held out)
  - significantly more accurate: +12.3% accuracy (McNemar p=8.16e-56)
  - automates more: auto coverage 65.1% vs 31.2% at the error budget
  accuracy delta +0.123 [+0.106, +0.139]  primary-only 0, candidate-only 184, p=8.16e-56
  primary    temperature  raw ece 0.098  calibrated ece 0.011  brier 0.166  accuracy 0.759  auto 31.2% (threshold 0.840)  p95 0.1 ms
  candidate  temperature  raw ece 0.111  calibrated ece 0.007  brier 0.100  accuracy 0.881  auto 65.1% (threshold 0.839)  p95 0.1 ms
  candidate failures 0/3000 (0.0%)
ticket.queue@v1 primary vs fake-2: PROMOTE (3000 paired: 1500 train, 1500 held out)
  - significantly more accurate: +25.9% accuracy (McNemar p=1.59e-117)
  - automates more: auto coverage 23.3% vs 9.5% at the error budget
  accuracy delta +0.259 [+0.237, +0.282]  primary-only 0, candidate-only 389, p=1.59e-117
  primary    platt        raw ece 0.176  calibrated ece 0.028  brier 0.187  accuracy 0.615  auto 9.5% (threshold 0.919)  p95 0.1 ms
  candidate  temperature  raw ece 0.125  calibrated ece 0.017  brier 0.104  accuracy 0.875  auto 23.3% (threshold 0.938)  p95 0.1 ms
  candidate failures 0/3000 (0.0%)
```

The candidate is *more* overconfident raw (ECE 0.111 against 0.098 on `refund.eligible`)
and is still the better model once both are calibrated: it doubles the share of traffic
that can be automated within the 10% error budget. `primary-only 0` is a property of the
fake, not of the test: both fakes share a seed, so the candidate is right wherever the
primary is; real models disagree in both directions, and McNemar uses both counts.

A second candidate replayed over the same stored decisions: the same model, only louder
(overconfidence 4.0, same skill):

```console
$ DGATE_SHADOW_BACKEND=fake DGATE_SHADOW_NAME=fake-loud DGATE_SHADOW_FAKE_OVERCONFIDENCE=4.0 \
    dgate shadow replay refund.eligible --limit 5000
refund.eligible@v1 fake-loud: 3000 replayed, 3000 ok, 0 failed, 0 already answered, 0 skipped (no stored input)
$ dgate shadow status
question                     candidate              total      ok  failed  labeled
refund.eligible@v1           fake-2                  3000    3000       0     3000
refund.eligible@v1           fake-loud               3000    3000       0     3000
ticket.queue@v1              fake-2                  3000    3000       0     3000
$ dgate shadow compare refund.eligible
refund.eligible@v1 primary vs fake-loud: KEEP (3000 paired: 1500 train, 1500 held out)
  - no meaningful improvement
  - no significant accuracy difference: +0.0% accuracy (McNemar p=1)
  - similar automation: auto coverage 31.2% vs 31.2% at the error budget
  accuracy delta +0.000 [+0.000, +0.000]  primary-only 0, candidate-only 0, p=1
  primary    temperature  raw ece 0.098  calibrated ece 0.011  brier 0.166  accuracy 0.759  auto 31.2% (threshold 0.840)  p95 0.1 ms
  candidate  temperature  raw ece 0.159  calibrated ece 0.010  brier 0.166  accuracy 0.759  auto 31.2% (threshold 0.844)  p95 0.1 ms
  candidate failures 0/3000 (0.0%)
```

Its raw calibration is much worse (ECE 0.159), but calibration removes the difference and
it automates exactly as much, so it is kept: there is nothing to gain from switching, and
a raw-calibration comparison would have called it worse when it is not.

### Simulator

A gate like this is hard to judge in production, because the one number that matters, the
auto tier's real error rate, is exactly what production cannot see: most automated
decisions never get a label. `dgate simulate run` runs the whole system on synthetic
traffic whose ground truth is known for every decision, and reports what really happened.

Nothing is re-implemented for the simulation. Decisions go through the real
`DecisionService` and router, reviews through the real queue, labels through outcome
ingestion, and the refit job, routing refresh and drift monitor run as they would from a
scheduler, against a real database (a temporary SQLite file, or PostgreSQL with
`--database-url`). Only the clock, the traffic, the model and the people are simulated:

- **Datasets** (`decision_gate.simulator.datasets`): refund requests judged by a written
  returns policy (category, reason, opened, order age) and support tickets whose subject
  determines the queue. Each item has a segment: `easy`, `hard` (near a policy boundary or
  ambiguously worded) or `novel`.
- **Drift**: after the steady phase the population changes without warning. Refunds get
  holiday-season order ages and a new marketplace line with its own 14-day window;
  tickets get a release that floods the technical queue with questions about a feature the
  model has never seen.
- **The model** (`SimulatedModel`) answers each segment with its own skill and
  overconfidence: hard items with less skill and more bravado, novel ones at chance and
  louder still. A calibrator fitted on one mix of segments is wrong for another, which is
  what drift does to a deployed model.
- **The timeline**: `history` days with no calibrator (everything goes to a person, whose
  answers are the first labels), then a refit and the first routing rule, `steady` days of
  routed traffic, then `drift` days. Reviewers resolve verify and human items within
  their SLA; the true outcome of a share of auto decisions arrives a few days later. The
  drift check runs every night, and the refit and refresh on a schedule.

Same seed, same run: traffic, model answers and decision ids (which drive verify
sampling) are all derived from the seed. A real run with the defaults (seed 7, 200
decisions per question per day, 21 + 14 + 14 days; about three minutes on SQLite), first
lines of each question's report:

```text
$ dgate simulate run --markdown report.md --json report.json
refund.eligible@v1 (predicate, auto error budget 10%)
  steady: 94.4% automated, true auto error 130/2643 = 4.9% (95% CI 4.1%-5.8%), within the 10% budget
  drift from day 36: detected at the end of day 42 (7 day(s) of drifted traffic), auto tier suspended at the end of day 42
  drift, until the action: true auto error 153/1332 = 11.5% (95% CI 9.8%-13.3%)
  drift, after the action: nothing automated

  phase       days decisions   auto verify  human hard+novel ECE raw ECE cal  true auto error
  history    1-21       4200   0.0%   0.0% 100.0%      15.4%   0.043       -  -
  steady    22-35       2800  94.4%   5.6%   0.0%      15.7%   0.045   0.007  130/2643 = 4.9% (95% CI 4.1%-5.8%)
  drift     36-49       2800  47.6%  52.4%   0.0%      45.1%   0.092   0.034  153/1332 = 11.5% (95% CI 9.8%-13.3%)

  [events omitted; daily rows from day 34, * = drifted traffic]
    day  auto >=   auto  auto err  check
     34   0.6667  95.5%      4.7%   ok
     35   0.6667  94.5%      3.2%   ok
     36   0.6667  96.5%     11.9%  *ok
     37   0.6667  95.5%     13.1%  *ok
     38   0.6667  96.0%      8.9%  *ok
     39   0.6667  94.0%     11.7%  *ok
     40   0.6667  95.5%     13.1%  *ok
     41   0.6667  94.5%     10.1%  *ok
     42   0.6667  94.0%     11.7%  *drift (label_psi) -> suspended
     43        -   0.0%         -  *drift (label_psi)
     44        -   0.0%         -  *drift (label_psi)
     45        -   0.0%         -  *drift (label_psi)
     46        -   0.0%         -  *ok
     47        -   0.0%         -  *drift (label_psi)
     48        -   0.0%         -  *drift (label_psi)
     49        -   0.0%         -  *drift (label_psi) -> tightened

ticket.queue@v1 (choice, auto error budget 5%)
  steady: 67.4% automated, true auto error 100/1888 = 5.3% (95% CI 4.3%-6.4%), not distinguishable from the 5% budget
  drift from day 36: detected at the end of day 39 (4 day(s) of drifted traffic), auto tier suspended at the end of day 39
  drift, until the action: true auto error 47/433 = 10.9% (95% CI 8.1%-14.2%)
  drift, after the action: nothing automated

  phase       days decisions   auto verify  human hard+novel ECE raw ECE cal  true auto error
  history    1-21       4200   0.0%   0.0% 100.0%      26.6%   0.094       -  -
  steady    22-35       2800  67.4%  32.6%   0.0%      26.9%   0.102   0.016  100/1888 = 5.3% (95% CI 4.3%-6.4%)
  drift     36-49       2800  15.5%  84.5%   0.0%      50.8%   0.175   0.075  47/433 = 10.9% (95% CI 8.1%-14.2%)
  [...]
```

The full report is in [docs/reports/simulation-seed-7.md](docs/reports/simulation-seed-7.md).
What it shows:

- **Steady state works.** Refunds: 94% automated with a true error of 4.9% against a 10%
  budget. Tickets: 67% automated at 5.3% (95% interval 4.3%-6.4%) against 5%: the
  thresholds are selected so the error rate is within budget with 95% confidence on the
  labels they were selected on, and on fresh traffic the realised rate sits at the budget.
  The report says "not distinguishable" rather than calling a pass or a fail on noise.
- **Calibration matters.** Raw ECE 0.045 and 0.102 drop to 0.007 and 0.016 after
  calibration in the steady phase.
- **Drift is caught, but not instantly.** Tickets: the auto tier's own error check and
  the ECE signal fired after 4 days and suspended the auto tier. Refunds: only the
  outcome-mix PSI fired, after 7 days; meanwhile the true auto error was 11.5% against a
  10% budget. The labeled sample above the threshold (5% verify sampling plus delayed
  outcomes) was too small for the error check to prove a 1.5-point overshoot. The levers
  are in the routing file: a higher `verify_sample_rate` buys faster detection with
  reviewer time.
- **Suspension, not guesswork.** Once drift is confirmed and the window's labels cannot
  support any threshold, the auto tier is suspended rather than tuned on too little data;
  decisions go to verify and human until enough new labels exist.

Options: `--dataset/-d` (repeatable), `--per-day`, `--history-days`, `--steady-days`,
`--drift-days`, `--seed`, `--overconfidence`, `--reviewer-accuracy`, `--outcome-rate`,
`--outcome-delay-days`, `--refit-every`, `--routing` (defaults to the example policies),
`--database-url` (must hold no decisions for the simulated questions; afterwards `dgate
serve` against it shows the whole history), `--markdown` and `--json`.
`dgate simulate export refund -n 1000 [--drifted]` prints items as JSON lines (input,
truth, segment) for use elsewhere.

### Observability and resilience

**Metrics.** `GET /metrics` serves Prometheus metrics for the process
(`DGATE_METRICS_ENABLED=false` turns it off):

| Metric | Labels | What it measures |
| --- | --- | --- |
| `dgate_http_requests_total`, `dgate_http_request_duration_seconds` | `method`, `route`, `status` | Every request, by route *template* (`/v1/decisions/{decision_id}`); unknown paths are `unmatched` |
| `dgate_decisions_total` | `question`, `tier`, `replay` | Decisions per tier; idempotent replays are counted separately |
| `dgate_backend_calls_total` | `backend`, `result` | `ok`, `error`, `timeout`, `cancelled`, or `rejected` by the circuit breaker |
| `dgate_backend_latency_seconds` | `backend` | Duration of the backend calls that were made |
| `dgate_circuit_state`, `dgate_circuit_transitions_total` | `backend`, `to` | `0` closed, `1` half-open, `2` open; and every change |
| `dgate_rate_limited_total` | `route` | Requests answered `429` |
| `dgate_outcomes_recorded_total` | `source`, `created` | Outcomes over HTTP; `created="false"` for duplicates |
| `dgate_build_info` | `version` | |

Labels stay low-cardinality: question ids come from the catalog and routes are templates,
never ids. Real output after one decision and its replay (from the CI smoke test's steps):

```text
dgate_build_info{version="0.1.0"} 1.0
dgate_http_requests_total{method="POST",route="/v1/decide",status="201"} 1.0
dgate_http_requests_total{method="POST",route="/v1/decide",status="200"} 1.0
dgate_decisions_total{question="ticket.queue",replay="false",tier="human"} 1.0
dgate_decisions_total{question="ticket.queue",replay="true",tier="human"} 1.0
dgate_backend_calls_total{backend="fake",result="ok"} 1.0
dgate_circuit_state{backend="fake"} 0.0
```

**Traces.** Each request is a server span that continues the caller's W3C `traceparent`,
with `dgate.decide` inside it and `dgate.backend.decide`, `dgate.route` and `dgate.store`
inside that (question, tier, decision id and backend as attributes). The service depends
only on the OpenTelemetry API, which does nothing until a provider is installed. Install
the `otel` extra (`uv sync --extra otel` or `pip install 'decision-gate[otel]'`) and set `DGATE_OTEL_ENABLED=true` to export over OTLP/HTTP; the endpoint
and headers come from the standard `OTEL_EXPORTER_OTLP_*` variables.

**Circuit breaker.** After `DGATE_CIRCUIT_FAILURE_THRESHOLD` consecutive backend failures
(errors, invalid answers or timeouts; default 5), calls are rejected at once for
`DGATE_CIRCUIT_RESET_S` seconds (default 30) instead of each waiting out the timeout. Then
one probe call is let through: success closes the circuit, failure opens it for another
period. Real output with the `http` backend pointed at a port where nothing listens and a
threshold of 3:

```text
HTTP/1.1 502 Bad Gateway
{"detail":"http: request failed: ConnectError('All connection attempts failed')","fallback_tier":"human"}
HTTP/1.1 502 Bad Gateway
{"detail":"http: request failed: ConnectError('All connection attempts failed')","fallback_tier":"human"}
HTTP/1.1 502 Bad Gateway
{"detail":"http: request failed: ConnectError('All connection attempts failed')","fallback_tier":"human"}
HTTP/1.1 503 Service Unavailable
retry-after: 30
{"detail":"http: circuit open after 3 consecutive failures; retry in 30.0s","fallback_tier":"human"}
```

**Falling back to a human.** When no answer can be had from the backend (`502`, `503`,
`504`), the body says `"fallback_tier": "human"`, so a caller needs one rule: anything but
a decision means the case goes to a person. The gate does not store a decision in that case.
A row with no model answer and no confidence would have nothing to calibrate and would
have to be filtered out of every calibration, threshold and drift query. Idempotent replays of
stored decisions keep working while the circuit is open, because they never reach the
backend. `/readyz` reports the circuit state but stays ready while it is open: taking every
replica out of the load balancer because the backend is down would also stop replays,
outcomes and the review queue.

**Rate limits.** Set `DGATE_RATE_LIMIT_PER_S` (and `DGATE_RATE_LIMIT_BURST`, default 20)
to limit `POST /v1/decide` per caller with a token bucket; excess requests get `429` with
`Retry-After`. The caller is the client address, or the header named by
`DGATE_RATE_LIMIT_KEY_HEADER` (e.g. `X-Client-Id`) when a trusted gateway sets it. Limits
and the breaker are per replica, in memory: with N replicas the overall limit is N times
the setting, and a shared limit belongs in the gateway.

### Maintenance worker

Refits, routing refreshes and drift checks can each be run by hand (CLI or HTTP), but in
production they have to happen on their own and in the right order. `dgate worker` runs a
pass every `--interval` seconds:

1. **refit** every question version (a fit is activated only if it wins on held-out data);
2. **refresh routing** for every routed question, so thresholds are selected on the newest
   calibrator;
3. **check drift** for every monitored question, last, so a tightening lands on the rule
   just stored instead of being replaced by a refresh later in the same pass.

A failure in one question's step is logged and recorded, and the pass carries on with the
rest. On PostgreSQL each pass takes a session advisory lock (`pg_try_advisory_lock`), so
extra replicas skip a pass instead of running it twice. SIGTERM finishes the pass in
progress before stopping. Each pass prints one JSON line; `--once` runs a single pass and
exits `1` if a step failed or `3` if a question is drifting, for a cron job or a
Kubernetes CronJob.

Real output from the end-to-end flow (600 labeled `refund.eligible` decisions, nothing
labeled for the ticket questions yet), one step per line and details shortened:

```text
refit   refund.eligible@v1          activated     temperature held-out log loss 0.4677 beats raw 0.5011
refit   ticket.queue@v1             skipped       0 labeled decisions, fewer than min_count=100
refit   ticket.handling_minutes@v1  skipped       0 labeled decisions, fewer than min_count=100
routing refund.eligible@v1          refreshed     auto 0.8976, verify 0.5690: scores from calibrator v1 (temperature); ...
routing ticket.queue@v1             refreshed     auto none, verify none: no active calibrator; every decision goes to a human ...
routing ticket.handling_minutes@v1  refreshed     auto none, verify none: no active calibrator; ...
drift   refund.eligible@v1          ok            action none
drift   ticket.queue@v1             insufficient  action none
drift   ticket.handling_minutes@v1  insufficient  action none
```

### Reliability diagrams

`dgate calibration diagram DIR` writes one SVG per question version: raw confidence and,
when one is active, the calibrated confidence, with the number of decisions in each bin
underneath. No plotting library is involved and the output is deterministic, so diagrams
can be committed and diffed. `dgate simulate run --diagrams DIR` draws one per question and
phase, scored with the calibrator that was active at each decision. From the committed
seed-7 simulation, `ticket.queue` before and after drift:

| steady | drift |
| --- | --- |
| ![ticket.queue, steady phase](docs/reports/diagrams/ticket.queue-steady.svg) | ![ticket.queue, drift phase](docs/reports/diagrams/ticket.queue-drift.svg) |

[docs/STATISTICS.md](docs/STATISTICS.md) explains every statistical method in the project,
what each one guarantees and where each guarantee stops.

### Endpoints

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

## Run the whole stack (Docker Compose)

```bash
docker compose up --build            # API on http://localhost:8080, worker, PostgreSQL
```

| Service | What it does |
| --- | --- |
| `postgres` | PostgreSQL 17 with a named volume |
| `migrate` | Runs `dgate db upgrade` once, then exits; everything else waits for it |
| `api` | `dgate serve` (fake backend, example catalog and routing), as a non-root user |
| `worker` | `dgate worker`: refit, routing refresh and drift checks every `DGATE_WORKER_INTERVAL_S` seconds (default 300) |
| `kafka`, `outcomes-consumer` | Profile `kafka`: a single-node KRaft broker and `dgate outcomes consume` |
| `otel-collector` | Profile `otel`: receives the API's traces (set `DGATE_OTEL_ENABLED=true`) and logs them |

```bash
docker compose --profile kafka up --build                       # add Kafka outcome ingestion
DGATE_OTEL_ENABLED=true docker compose --profile otel up --build  # add tracing
```

The image is a two-stage build: the locked dependencies are resolved with uv in the first
stage, and the runtime stage holds only the virtual environment and the example
configuration. CI builds it and runs the stack on every push (see
[Development](#development)).

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
and checks the stats count it once, then checks calibration metrics over HTTP and the CLI
and runs the refit job (skipped on one label) against the calibrator registry, threshold
selection and a conformal fit over HTTP and the CLI, `dgate fake-gate`, `dgate routing
check`, and routing: the first decision goes to a human because nothing is calibrated, and
a refresh over HTTP stores a current rule for every question, the review queue round trip,
and drift checks over HTTP and the CLI (too little data, so every signal is
`insufficient` and routing is untouched), plus `/metrics` counting a decision and its replay. A short simulation runs against its own
PostgreSQL database and its JSON report is checked. The Kafka
source is unit-tested with in-memory consumer and producer fakes.

A second CI job builds the image and starts the Compose stack, then runs
[tests/e2e/test_stack.py](tests/e2e/test_stack.py) against it: 600 decisions all go to a
person, their outcomes arrive in batches, the worker fits a calibrator and stores a
routing rule on its own timer, and new decisions start to be automated, with the
calibrated confidence of every automated one at or above the rule's threshold. The same
job sends an outcome through a real Kafka broker to the consumer and checks that a
malformed event lands on the dead-letter topic. The end-to-end test runs against any stack
(`DGATE_E2E_URL=http://localhost:8080 uv run pytest tests/e2e`) and is skipped otherwise.

## License

[MIT](LICENSE)
