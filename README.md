# Recoup

**A cost-aware recovery policy for failed UPI AutoPay mandate debits.**
Razorpay AI Buildathon · Track 03 — AI Revenue Recovery.

> **Result (400 simulated mandate failures, seed 42):** Recoup recovered **₹717k** against
> **₹302k** for fixed-schedule retry, on **45% fewer debit attempts** (544 vs 994) and
> **~⅕ the messages**, while preserving **25 more mandates**. Net of action cost, the
> missed-cycle penalty and lost-mandate LTV, that is **+₹2,681 per case (95% CI
> [+₹1,427, +₹4,162])**. It escalated **27** cases it judged not worth pursuing, each with
> a logged rationale. Checked against 5 other world seeds (not just 42) and 6 perturbed
> prior sets, the gain is never a coincidence of one lucky draw — see
> [Seed and prior robustness](#evidence--does-it-hold-up-resultsphase9json) below.

## The problem, in three sentences

UPI AutoPay debits fail 8–15% of the time, mostly because an account is briefly short of
funds, and roughly 20 million mandates are revoked every month as a result. Merchants
respond with a fixed retry schedule that is blind to *why* a debit failed, *when* the
customer will have money, and what each retry or nudge *costs* — including the risk of
irritating a recoverable customer into cancelling. Recoup replaces that schedule with a
per-case decision: infer the cause with calibrated confidence, predict a funding window,
price each permitted intervention against its cost, and stop when nothing is worth doing.

Full write-up: this README + the numbers reproduced live at `python tasks.py serve`.

---

## Status

| Phase | Component | State |
|------:|-----------|-------|
| **1** | Simulator — generator, customers, hidden response functions, priors | ✅ done |
| **2** | Harness (competing-risk rollout) + baselines (never-act, fixed-schedule, always-nudge) | ✅ done |
| **3** | Cost model — `config/costs.yaml` + `agent/costs.py` (action costs, LTV estimate, missed-cycle penalty, compounding message fatigue) | ✅ done |
| **4** | Calibrated cause classifier (`agent/classifier.py`) + `cause_aware` escalate branch | ✅ done |
| **5** | Liquidity-window model (`agent/liquidity.py`) — quantile funding-day prediction + `liquidity_aware` policy | ✅ done |
| **6** | Constraint filter (`harness/constraints.py` + `config/constraints.yaml`) — retry cap, min gaps, quiet hours, message cap; enforced structurally on every plan + property tests | ✅ done |
| **7** | Cost-aware EV policy (`agent/ev_policy.py`, `recoup`) — per-case expected-value decision over retry / re-auth / do-nothing | ✅ done |
| **8** | Audit trail (`agent/audit.py`, JSONL + SQLite) + optional cached LLM prose (`agent/llm_explain.py`) | ✅ done |
| **9** | Evidence (`experiments/phase9.py`) — oracle ceiling, ablation ladder, prior-sensitivity sweep, fairness slice | ✅ done |
| **10** | Service (`service/`) — Razorpay webhook ingestion → decision API → action executors → SQLite state → one React app (results overview + live console) | ✅ done |
| **2.0** | Closed loop (`service/events.py`, `state.py`, `loop.py`, `agent/ros.py`, `recoup_v2`) — event-sourced customer state, webhook idempotency, Recovery Opportunity Score, adaptive re-planning, outcome/reward recording, model-health API | ✅ done |

### Where things stand (400 cases, seed 42)

The scoreboard reports **net value** — rupees recovered, *minus* action spend, the
missed-cycle penalty on late recoveries, and the believed LTV of every revoked mandate.
That is the objective `recoup` optimises; raw recovery rate is a proxy that hides the
revocation losses.

| Policy | Net value | Recovered | Rate | Attempts | Msg/case | On-time | Preserved | Esc |
|--------|----------:|----------:|-----:|---------:|---------:|--------:|----------:|----:|
| `never_act` (floor) | −₹901,690 | ₹0 | 0.0% | 0 | 0.00 | 0.0% | 88.0% | 0 |
| `fixed_schedule` (Baseline A) | −₹848,435 | ₹302,303 | 32.8% | 994 | 0.73 | 30.0% | 88.0% | 0 |
| `always_nudge` (Baseline B) | −₹823,183 | ₹290,712 | 31.0% | 966 | 1.81 | 25.5% | 85.5% | 0 |
| `cause_aware` (Phase 4 — escalate branch) | −₹630,945 | ₹386,543 | 42.5% | 804 | 0.74 | 39.8% | 89.2% | 24 |
| `liquidity_aware` (Phase 5 — window timing) | +₹68,912 | ₹724,340 | 73.5% | 553 | 0.17 | 26.0% | 94.2% | 24 |
| **`recoup` (Phase 7 — EV decision)** | **+₹110,237** | ₹717,314 | 71.8% | 544 | 0.17 | 30.0% | 94.2% | 27 |

**Every plan** first passes the structural constraint filter (`config/constraints.yaml`):
≤ 3 retries ≥ 24 h apart, ≤ 4 messages ≥ 20 h apart, nothing 21:00–09:00 IST. Illegal
actions are rescheduled or dropped and counted; even a deliberately greedy policy is
reined in to these caps.

Each policy adds **one idea**, so the harness prices it in isolation:

- **`always_nudge`** — "message harder": ≈ `fixed_schedule` on money (−₹29/case, CI crosses
  zero) but **worse on mandate preservation** (−0.03/case, CI [−0.05, −0.00]). More dunning
  contact does not recover more; it churns customers.
- **`cause_aware`** — a calibrated classifier flags probably-dead mandates (P ≥ 0.30, ~90%
  precision) → stop retrying, one re-auth, hand off. **+₹84k recovered**, and a re-auth
  revives ~57% of dead mandates retries never touch.
- **`liquidity_aware`** — retries at the model's predicted **p50 / p85 funding days**
  instead of a fixed calendar. **+₹422k recovered vs `fixed_schedule`** on **fewer
  attempts** (553 vs 994) and near-zero messages — the biggest single jump. It flips net
  value from −₹848k to **+₹69k**.
- **`recoup`** — prices every candidate action:
  `EV = P(success)·recovery_value − action_cost − P(revocation)·LTV − missed_cycle_penalty·P(miss)`,
  using the classifier for cause, the liquidity model for timing, `costs.yaml` for the
  economics. It recovers about the same money as `liquidity_aware` but **sooner and more
  on-time** (30% vs 26%, 16 days vs 17) because it explicitly weighs the missed-cycle
  penalty; it stops or escalates the cases where nothing has positive EV; and **every
  decision carries a logged rationale** (`audit/`). Paired vs `fixed_schedule`:
  **+₹2,681/case net value, 95% CI [+₹1,427, +₹4,162]** · +₹1,038/case recovered ·
  +0.06 mandates/case preserved. See [seed robustness](#evidence--does-it-hold-up-resultsphase9json)
  below for how this holds up across 6 independent world seeds, not just this one.

**Classifier** (`results/classifier_report.json`): test accuracy **0.90** vs a 0.60
majority prior; **ECE 0.047** calibrated (0.085 uncalibrated); holdout batch 0.885 / 0.056.

**Liquidity model** (`results/liquidity_report.json`): funding-day **MAE 6.1 days** vs
11.8 for a modal-day heuristic and 11.8 for fixed-day-7; p85 quantile coverage 0.83
(0.84 on the holdout batch).

### Evidence — does it hold up? (`results/phase9.json`)

**Ablation ladder** — remove one capability at a time, measure the loss. `net Δ/case` is
paired vs `fixed_schedule`; `% gap` is how much of the recovery gap to the oracle is closed.

| policy | recovery | on-time | preserved | net Δ/case | % recovery gap |
|--------|---------:|--------:|----------:|-----------:|---------------:|
| `fixed_schedule` | 32.8% | 30.0% | 88.0% | — | 0% |
| `no_cause` (EV policy, no classifier) | 63.2% | 15.0% | 93.0% | +₹1,958 | 70% |
| `no_timing` (EV policy, fixed window guess) | 69.0% | 33.8% | 94.0% | +₹2,438 | 84% |
| `liquidity_aware` (no cost/EV layer) | 73.5% | 26.0% | 94.2% | +₹2,570 | 94% |
| **`recoup`** | 71.8% | 30.0% | 94.2% | **+₹2,681** | 90% |
| _oracle_ (perfect timing, recovery-max) | _76.0%_ | _33.0%_ | _93.2%_ | _+₹2,443_ | _100%_ |

The classifier is worth ~20 points of the gap (and drags `no_cause`'s on-time rate to
15% — without it, dead mandates get retried on the funding schedule and never escalated).
The timing model adds ~6 more. **`recoup` exceeds the oracle's net value by ~₹95k**: the
oracle maximises raw recovery, `recoup` trades ~4 points of it to beat the billing date
more often.

**Prior sensitivity** — the trained models are frozen; only the *world* is perturbed, and
each variant is a fresh 400-case draw. `recoup` beats `fixed_schedule` on net value under
**every** perturbation, CI excluding zero:

| world | net Δ/case (recoup − fixed) |
|-------|---------------------------:|
| baseline priors | +₹2,491 |
| fewer cash-flow failures, more dead mandates | +₹2,768 |
| churn hazard ×1.8 | +₹2,127 |
| LTV halved (6 months, lower retention) | +₹1,269 |
| bank outages 3× longer | +₹2,535 |
| deeper balance shortfalls | +₹2,131 |

**Seed robustness** (final-audit addition) — the sensitivity sweep above re-draws the world
under different *priors* but always at one seed pairing. This asks the more basic question:
is world seed 42 itself a lucky draw? Five more independent seeds, same (unperturbed)
priors, fresh 400-case batches each:

| seed | recovery | recoup net Δ/case | 95% CI | recoup_v2 net Δ/case |
|-----:|---------:|-------------------:|--------|----------------------:|
| 7 | 67.8% | +₹1,257 | [+₹240, +₹2,183] | +₹1,503 |
| 42 (headline) | 71.8% | +₹2,681 | [+₹1,427, +₹4,162] | +₹2,599 |
| 99 | 72.0% | +₹2,032 | [+₹793, +₹3,476] | +₹1,815 |
| 123 | 71.8% | +₹2,316 | [+₹1,191, +₹3,697] | +₹2,334 |
| 2024 | 72.8% | +₹2,465 | [+₹1,464, +₹3,722] | +₹2,498 |
| 31337 | 67.5% | +₹1,676 | [+₹718, +₹2,898] | +₹1,805 |

Recoup beats `fixed_schedule` with the CI excluding zero at **every one of the 6 seeds**.
Honestly: seed 42 is the *highest* of the six (mean across all six ≈ **+₹2,071/case**), so
treat +₹2,681 as the top of a +₹1,257–+₹2,681 range this approach produces, not a number
that holds to the rupee — the sensitivity and seed checks together are the actual claim.
Reproduce with `python -m experiments.phase9` (`report["seed_robustness"]` in
`results/phase9.json`).

**Fairness** — by customer income pattern. Every group gains significantly vs the baseline
(all CIs exclude zero); no group is disproportionately escalated (disparity 0.03):

| group | n | recovery | Δ vs fixed | on-time | escalated | net Δ/case |
|-------|--:|---------:|-----------:|--------:|----------:|-----------:|
| salaried | 202 | 77.2% | +48.5 pts | 18.3% | 7.9% | +₹2,470 |
| gig | 129 | 66.7% | +21.7 pts | 43.4% | 4.7% | +₹2,381 |
| business | 69 | 65.2% | +43.5 pts | 39.1% | 7.2% | +₹3,862 |

Salaried customers recover 10 points more — a regular monthly payday makes the
funding-window model most accurate for them. The gap is in *model accuracy by income
regularity*, not the decision logic; it is the clearest target for future work.

### Audit trail

`recoup` logs a structured record for every case (`audit/audit_42.jsonl` + `.db`):
cause posterior, funding window, the EV of each candidate action, the decision, and the
realised outcome — plus a plain-English note:

> _Diagnosis: most likely insufficient balance (87% confidence) — the account was short of
> funds. Predicted funding window 22–29 days out (billing date at day 21). Decision: retry
> on day 22, retry on day 29, retry on day 35 (best EV ≈ ₹2,675). Outcome: recovered ₹3,519
> by retry, late (26 days); 3 attempts, 0 messages._

The note is a deterministic template — no network, always runs. An LLM can rewrite it
(`--llm`, needs `ANTHROPIC_API_KEY`); calls are cached and never touched by `reproduce`.
`python -m agent.audit --seed 42 --case c0011` prints one case in full.

---

## Recoup 2.0 — the closed loop

Phases 1–9 build and prove the *decision*. Phase 10 wraps it in a service. **Recoup 2.0**
turns that service into a closed-loop, event-driven recovery agent — without touching the
Phase 1–9 engine or its evidence (every earlier number reproduces to the digit).

```
payment event ─▶ webhook gateway ─▶ customer state engine ─▶ diagnose ─▶ funding window
     ▲                (HMAC + idempotency)     (event-sourced)              │
     │                                                                     ▼
  outcome ◀─ execute (dry-run / test mode) ◀─ select ◀─ guardrails ◀─ Recovery Opportunity Score
     │                                                                     (candidate actions)
     └──────────────── update state ─▶ re-plan (or stop when nothing clears the EV floor)
```

| Piece | File | What it adds |
|-------|------|--------------|
| Event log | `service/events.py` | 10 typed event kinds, timestamped, append-only, `dedup_key`-idempotent |
| Customer state engine | `service/state.py` | `CustomerState` folded from the event log — failure/funding/recovery history, `churn_risk`, `recovery_stage`. **Observable-only**: no hidden truth, verified by test |
| Webhook idempotency | `service/store.py` + `service/app.py` | `idempotency_keys` table; a duplicate `payment.failed` returns the original case, creates nothing |
| Recovery Opportunity Score | `agent/ros.py` | `ROS(a) = P(recovery)·value·timeliness·retention − action_cost − churn_cost − missed_cycle`. A **decomposition of** the Phase-7 EV (every term comes off an `EVPolicy` instance), plus an explicit retention term that leans on `CustomerState.churn_risk`. The original `EV(a)` is untouched. |
| Adaptive planner | `agent/policies.py::RecoupV2` (`recoup_v2`) | commits **one** action, then re-decides from the updated engagement history (`terminal="replan"`) |
| Closed loop | `service/loop.py` | `open_recovery → schedule → execute → record_action_result → re-plan → …`; stop-loss when the best candidate ≤ the EV floor. A `payment.captured` webhook is correlated back to its case via the `notes.recoup_case` tag Recoup itself attached to the re-auth Payment Link (`webhook.recovery_ref`) — a Payment Link's own payment isn't otherwise linked to the subscription it was recovering — and is only accepted while the case is still open, so a later, unrelated successful charge on the same subscription can't be mistaken for a recovery. |
| Outcome / reward | `service/store.py::outcomes` | per-action `reward = recovered_value − action_cost − missed_cycle − Δchurn·LTV`, with `state_before` / `state_after`. For offline evaluation only — **no online model updates** |
| Monitoring | `service/monitoring.py` | `/api/models/health` (classifier ECE, liquidity MAE/coverage, policy stats — from `results/*.json`) and `/api/metrics` (the running service's own numbers) |

**`recoup_v2` vs `recoup` (400 cases, seed 42):** `+₹2,599/case` net value vs fixed-schedule
(CI `[+₹1,285, +₹4,114]`), 71.2 % recovery — statistically indistinguishable from one-shot
`recoup` (`+₹2,681`). In the *frozen simulator* a failed retry carries no new signal, so
re-planning cannot beat committing the ladder up front; its payoff is operational — in the
live service it reacts to real late-arriving `payment.failed` and funding events. `recoup`
stays the headline batch policy; `recoup_v2` is what the service runs.

### API (Recoup 2.0 additions)

```
POST /webhook                      idempotent; payment.failed / subscription.pending → open/continue, captured → recover, halted → revoke
GET  /api/cases/{id}/timeline      the event log for one recovery
GET  /api/cases/{id}/decision      the latest structured decision (+ ROS breakdown)
POST /api/cases/{id}/replan        force a re-plan from current state
POST /demo/outcome/{id}            demo/testing only — feed one action's result, watch it observe + re-plan
GET  /api/metrics                  live service metrics (cases, recoveries, reward, cost/recovery)
GET  /api/models/health            classifier / liquidity / policy health from the batch reports
```

### Four deterministic demo scenarios

Each is a fixed seed or fixed webhook body — same input, same output, every run (verified:
`STORE.reset_case` clears any prior demo run on the same seed first, so replaying one is
also deterministic, not just the first call). Start the server (`python tasks.py serve`)
first.

**1 — insufficient balance → funding-window prediction → retry → recovery**
```bash
curl -X POST "localhost:8000/demo/random?cause=insufficient_balance&seed=1"
# case demo_4732, ₹2,905 — recovered by retry on day 26, via the predicted funding window
```

**2 — dead mandate → negative retry EV → re-auth → escalate**
```bash
curl -X POST "localhost:8000/demo/random?cause=mandate_dead&seed=2"
# case demo_1879 — classifier says 100% dead; ROS table shows retry EV = −₹0.4 (both
# candidate days), re-auth EV = +₹25,738 → re-auth attempted, then handed to a human
```

**3 — duplicate webhook → idempotency**
```bash
curl -X POST localhost:8000/webhook -d '{"id":"evt_x","event":"payment.failed",
  "payload":{"payment":{"entity":{"amount":99900,"error_code":"U30","subscription_id":"sub_x"}}}}'
# -> {"duplicate": false, ...}
curl -X POST localhost:8000/webhook -d '{"id":"evt_x", ...same body... }'
# -> {"duplicate": true, ...} — no second case, no second event
```

**4 — failed action → new event → state update → re-plan**
```bash
curl -X POST localhost:8000/webhook -d '{"id":"evt_y","event":"payment.failed",
  "payload":{"payment":{"entity":{"amount":349900,"error_code":"U30","subscription_id":"sub_y"}}}}'
# -> decision schedules one retry (terminal: "replan")
curl -X POST "localhost:8000/demo/outcome/sub_y?kind=retry&result=fail"
# -> customer_state.recovery_attempts: 1, retry_results: [false]; timeline now shows
#    AutoPay debit failed -> retry scheduled -> retry executed -> retry scheduled (the re-plan)
```

### Example end-to-end flow

```bash
python tasks.py serve
# 1. a failed debit arrives
curl -X POST localhost:8000/webhook -H 'content-type: application/json' -d '{
  "id": "evt_001", "event": "payment.failed",
  "payload": {"payment": {"entity": {"amount": 299900, "error_code": "U30",
    "error_description": "insufficient balance", "subscription_id": "sub_42"}}}}'
#    → customer state built, cause diagnosed, ROS scores every action,
#      one retry scheduled for the predicted funding day, terminal="replan"
curl -X POST localhost:8000/webhook -d '{ "id": "evt_001", ... }'   # 2. duplicate → {"duplicate": true}, nothing created
curl localhost:8000/api/cases/sub_42/timeline                        # 3. PAYMENT_FAILED → RETRY_SCHEDULED
# 4. the retry fires (POST /tick), the gateway later reports payment.captured:
curl -X POST localhost:8000/webhook -d '{ "event": "payment.captured",
  "payload": {"payment": {"entity": {"subscription_id": "sub_42", "status": "captured"}}}}'
#    → PAYMENT_RECOVERED, reward recorded, state = recovered, loop closed
curl localhost:8000/api/metrics
```

The Live console shows all of this: the ROS table (every candidate, `P(clear)`, retention,
score vs EV), the customer-state strip (stage, churn risk, prior attempts), the event
timeline, and a Model-health panel.

---

## Reproduce

```bash
pip install -r requirements.txt

python tasks.py reproduce        # data + train + harness + audit + tests
#   or, with make:  make reproduce

python -m simulator.generate --n 400 --seed 42 --out data
python -m simulator.show c0142 --ledger
python -m agent.train_classifier          # -> agent/models/cause_clf.pkl + results/classifier_report.json
python -m agent.train_liquidity           # -> agent/models/liquidity.pkl + results/liquidity_report.json
python -m harness.run --seed 42
python -m harness.run --trace c0142 --policy recoup
python -m agent.audit --seed 42 --case c0011      # one decision, in full
python -m experiments.phase9                       # oracle + ablations + sensitivity + fairness
python -m pytest -q
```

### The app

```bash
python tasks.py train            # once — trains the two models the console needs
python tasks.py serve            # -> http://127.0.0.1:8000
```

One React app (no build step — React + htm are vendored), two views:

- **Results** — the thesis, the scoreboard, the ablation ladder, the sensitivity sweep and
  the fairness slice, pulled live from `results/*.json` (or a baked snapshot when the
  numbers haven't been generated yet). Deployable as static files too.
- **Live console** — synthesise a failed mandate (with its hidden outcome, so it can be
  scored) and watch Recoup decide: the cause posterior, the funding-window prediction, the
  expected value of *every* candidate action, the plan, the plain-English reason — then
  replay the 45-day window side by side against the fixed schedule, with a running
  session scoreboard.

It also accepts a real Razorpay webhook:

```bash
curl -X POST http://127.0.0.1:8000/webhook -H 'content-type: application/json' -d '{
  "event": "payment.failed",
  "payload": { "payment": { "entity": {
    "amount": 299900, "error_code": "U30", "error_description": "insufficient balance", "id": "pay_demo" }}}}'
```

`POST /webhook` verifies the `X-Razorpay-Signature` HMAC when `RAZORPAY_WEBHOOK_SECRET` is
set, maps the event to a case, runs the policy, persists the decision, and schedules the
actions. Executors run in **dry-run** by default; set `RECOUP_EXECUTE_MODE=razorpay_test`
with `rzp_test_*` keys to have the re-auth action create a real test-mode Payment Link.
See `service/`.

**Environment variables** — everything is optional; the whole project runs with none of
them set:

| Variable | Effect when set |
|----------|------------------|
| `RAZORPAY_WEBHOOK_SECRET` | enforces the `X-Razorpay-Signature` HMAC on `/webhook` (unset = local/demo mode, unsigned) |
| `RECOUP_EXECUTE_MODE=razorpay_test` | executors call the real Razorpay test-mode API instead of dry-run intent-logging |
| `RAZORPAY_KEY_ID` / `RAZORPAY_KEY_SECRET` | test-mode credentials for the above — rejected unless `RAZORPAY_KEY_ID` starts with `rzp_test_` |
| `ANTHROPIC_API_KEY` | enables the optional cached LLM audit-note rewrite (`agent/llm_explain.py`) |
| `RECOUP_API_KEY` | requires `Authorization: Bearer <key>` on every route except `/`, `/healthz`, the static assets, and `/webhook` (which authenticates itself via the HMAC signature instead) — a hackathon judge running this locally never needs it; a real deployment should set it |
| `RECOUP_TICK_INTERVAL_SECONDS` | how often the background loop runs `/tick`'s logic on its own (default 30; set to `0` to go back to calling `POST /tick` manually — tests do this so a stray background task never touches a torn-down test database) |
| `RECOUP_RATE_LIMIT_PER_MIN` | per-client-IP request cap (default 300/min) — generous enough that no normal demo or test run ever hits it; a single-process in-memory guard against one misbehaving client, not a substitute for a real reverse proxy |

None of these are read anywhere near `agent/` or `simulator/` — they only gate the service's
edges (signature check, execution mode, API access, rate), never the decision logic.

**Hardening added in the final audit pass:** `/tick` now runs on its own on a background
schedule instead of needing a human or cron to remember it; SQLite runs in WAL mode so a
webhook write no longer blocks a dashboard read; every route enforces a body-size cap and a
per-IP rate limit; `service/store.py::claim_action` makes scheduled-action execution
race-safe against concurrent `/tick` calls (verified: `tests/test_service.py`). A `Dockerfile`
is included for a from-scratch deployment path (written and reviewed, not build-tested in
this environment — no Docker daemon available here).

Outputs — `data/`, `agent/models/` and `audit/` are git-ignored and regenerated;
`results/*.json` (seed 42) are committed as a record and overwritten by each run:

| file | contents |
|------|----------|
| `data/cases.jsonl` | one observable record per line — **everything the agent sees** |
| `data/truth.jsonl` | one hidden-truth record per line — **harness only** |
| `data/summary.json` | batch statistics vs the priors |
| `results/harness_42.json` | full per-case outcomes + summaries + exception lists per policy |
| `results/classifier_report.json` | classifier accuracy, calibration (ECE / reliability bins), confusion, escalate-branch precision/recall |
| `results/liquidity_report.json` | funding-day MAE / median AE / bias, p85 quantile coverage, vs naive baselines |
| `agent/models/*.pkl` | the trained classifier + liquidity models (regenerated by `train`) |
| `audit/audit_42.jsonl` / `.db` | one decision record per case — belief, EV of each candidate, choice, rationale, outcome |
| `results/phase9.json` | ablation ladder, oracle gap, prior-sensitivity sweep, fairness-by-income slice |

---

## What is synthetic, and why

Real UPI mandate / bank data is confidential and PII-bearing under DPDP; NPCI publishes
only aggregates. So every case is generated. Each carries a **hidden true cause** and a
**hidden response function** (how P(debit clears) moves over the horizon, per cause) that
the agent never sees — only the harness does. Priors live in `config/priors.yaml`, each
with a source tag or an explicit `ESTIMATE` marker.

The claim this project tests does not depend on the absolute numbers being right — all
policies (Recoup and both baselines) face an identical simulated world, so the **gap**
between them is what is measured, and a sensitivity check (Phase 9) re-runs everything
under different priors to show the ordering holds.

A final-audit pass specifically checked the *agent's own belief constants*
(`agent/ev_policy.py::EVParams`) against the simulator's hidden truth for exact matches — the
one place a policy could get an unfair edge without ever importing `simulator/response.py`.
It found two (`max_exec_prob` had literally copied `priors.max_success_prob`;
`limit_p_before_reset` matched `p_success_before_reset` to the decimal) and de-tuned both to
independent, round estimates. The harness and Phase 9 numbers above are unchanged to the
rupee after that fix — reassuring, not just because the numbers survived, but because it
means Recoup's edge was never resting on that leak in the first place.

### Cause priors (assumptions table)

| Cause | Prior | Basis |
|-------|------:|-------|
| `insufficient_balance` | 0.60 | Dominant UPI AutoPay failure mode. Public reporting puts the real share nearer 70-74% ([Business Standard, Sep 2026](https://www.business-standard.com/finance/news/upi-autopay-revocations-hit-20-mn-monthly-over-low-customer-balances-125090700500_1.html)) — kept conservative here so the three harder minority causes stay well-represented for the classifier to learn from |
| `bank_downtime` | 0.15 | Bank/gateway-side outage share; directional carve-out, no isolating NPCI aggregate found |
| `limit_breach` | 0.10 | Estimate — per-account AutoPay cap / per-transaction limit |
| `mandate_dead` | 0.15 | Estimate — revoked-at-bank / not-found; rises with mandate age |

Failure **codes are reconciled with Razorpay's actual `error_reason` taxonomy**
(`service/webhook.py → _CODE_TO_TOKEN`, sourced from Razorpay's
[rainy-day](https://razorpay.com/docs/payments/payment-gateway/rainy-day/errors/error-reasons/)
and [eMandate](https://razorpay.com/docs/payments/recurring-payments/emandate/errors/) error
docs); the four engineered causes are still **deliberately ambiguous** at the code level
(`config/priors.yaml → failure_code_emission`) — one generic decline code is reachable from
several causes, so the classifier cannot cheat off the code string alone.

---

## What this does **not** do

- Voice / Hinglish calling — large failure surface, no effect on the metric.
- A dashboard as the *headline* — the console (Phase 10) is there to operate and inspect the
  engine, not to stand in for the measured result.
- Multi-agent orchestration frameworks — this is one decision chain.
- Other failure classes — checkout abandonment, receivables, one-time payments.
- Real production data — impossible to obtain, and not expected. The webhook path is wired
  to the Razorpay event shape and test-mode APIs; it is not pointed at a live account.

---

## Layout

```
config/priors.yaml       cause priors + simulator parameters, each sourced  (hidden-world truth)
config/costs.yaml        what the AGENT believes actions cost + a mandate is worth, incl. revocation-risk beliefs  (estimates, not truth)
config/constraints.yaml  hard operating limits (retry cap, min gaps, quiet hours, message cap) — enforced structurally
agent/
  costs.py               loads costs.yaml: action_cost, ltv_estimate, recovery_value, missed_cycle_penalty, message_fatigue_factor
  features.py             observable case -> numeric feature vector (never touches truth)
  classifier.py           calibrated multinomial cause classifier + should_escalate (the escalate branch)
  calibration.py          reliability bins / ECE / Brier — plain numpy
  liquidity.py            quantile funding-day model (p50 / p85 days-from-now)
  train_classifier.py     CLI: generate training batch (separate seed), fit + calibrate, tune threshold, report
  train_liquidity.py      CLI: train the funding-day quantile regressors, eval vs naive baselines
  policies.py             cause_aware (Phase 4) + liquidity_aware (Phase 5) + recoup_v2 (2.0, adaptive re-plan)
  ev_policy.py            recoup (Phase 7) — the cost-aware expected-value decision + explain()
  ros.py                  Recoup 2.0 — Recovery Opportunity Score: a decomposition of the Phase-7 EV + a retention term
  ablations.py            Phase 9 — recoup with the classifier / timing model stubbed out
  audit.py                Phase 8 — per-case decision record (JSONL + SQLite) + template narration
  llm_explain.py          optional cached LLM rewrite of the narration (offline-safe fallback)
simulator/
  customers.py           income patterns, competing debits, cash ledger, back-sim history
  generator.py           assembles one case = observable record + hidden truth
  response.py            hidden response functions + revocation hazard  (never imported by agent/)
  generate.py            batch CLI
  show.py                human-readable single-case dump
harness/
  plan.py                the policy <-> harness contract: ScheduledAction, Plan
  constraints.py         the constraint filter: enforce(plan) -> legal plan + violations  (RBI/NPCI/TRAI + anti-harassment)
  engine.py              competing-risk rollout: per-case World, run_case; runs every plan through constraints.enforce
  metrics.py             batch summary, net-value, paired bootstrap CIs, per-cause/-income splits, exceptions
  run.py                 runner CLI + comparison table + --trace <case_id>
  oracle.py              perfect-timing upper bound (reads truth) — the ceiling, not a competitor
experiments/
  phase9.py              ablation ladder + oracle + prior-sensitivity sweep + fairness slice
service/                 Phase 10 + Recoup 2.0 — Recoup as a running closed-loop service (FastAPI, no new engine logic)
  webhook.py             Razorpay payment.failed -> a case; HMAC signature check
  events.py              2.0 — typed Event + EventType (append-only, dedup_key-idempotent)
  state.py               2.0 — CustomerState folded from the event log (observable-only)
  loop.py                2.0 — the closed loop: open_recovery / record_action_result / re-plan / stop-loss
  monitoring.py          2.0 — /api/models/health + /api/metrics from results/*.json and the outcomes table
  bridge.py              thin wrappers: decide() / random_case() / simulate() over the engine
  executors.py           retry / re-auth link / escalate  (dry-run, or Razorpay test mode)
  store.py               SQLite: cases, decisions, actions, escalations + events, customers, mandates, outcomes, idempotency_keys
  app.py                 FastAPI routes + serves the app; GET /api/results feeds the Results view
  webui/                 one React app, no build step (React + htm vendored)
    index.html           shell + all CSS (Razorpay-styled, both themes)
    app.js               <App> = <Overview> (results brief) + <Console> (decision UI + ROS table, customer-state strip, event timeline, model-health panel)
    data.js              baked results snapshot — the Overview's fallback with no backend
baselines/
  never_act.py           floor
  fixed_schedule.py      Baseline A — retry d+1/d+3/d+7 + SMS
  always_nudge.py        Baseline B — nudge, retry, nudge, retry, retry
tests/
  test_generator.py      Phase-1 invariants (day-0 failure is real, no leakage, reproducible, ...)
  test_harness.py        Phase-2 invariants (budget, all-or-nothing, determinism, paired property, ...)
  test_costs.py          Phase-3 invariants (channel-cost ordering, conservative LTV, gentle delay discount, fatigue compounds, no leakage)
  test_classifier.py     Phase-4 invariants (accuracy beats prior, ECE < 0.1, calibration helps, escalate precision, no leakage)
  test_liquidity.py      Phase-5 invariants (observable-only, beats naive MAE, p85 quantile coverage, roundtrip)
  test_constraints.py    Phase-6 property tests (fuzzed plans come out legal, idempotent, greedy policy reined in)
  test_ev_policy.py      Phase-7 invariants (well-formed plans, restraint, dead->reauth, beats fixed_schedule on net value)
  test_audit.py          Phase-8 invariants (structured record, deterministic narration, SQLite roundtrip, LLM fallback)
  test_phase9.py         Phase-9 invariants (oracle is the ceiling, ablations degrade, ordering survives perturbed priors, wins hold at independent seeds, no group left behind)
  test_service.py        Phase-10 + 2.0 (webhook signature/mapping, store round-trip, executors dry-run, idempotency, timeline/replan/metrics endpoints, malformed-input handling, no raw-exception leakage, race-safe action claiming)
  test_events_state.py   2.0 (event dedup, idempotency-key binding, state folding, recovery transitions, churn monotonicity, no leakage)
  test_ros.py            2.0 (ROS wraps EV, candidates well-formed + ranked, dead mandate never tops a retry, churn lowers retention)
  test_loop.py           2.0 (ingest + schedule, duplicate webhook ignored, recovered closes the loop, failed retry re-plans, revocation terminal, reward recorded, payment.captured correlated by Payment-Link notes, stale webhook on a closed case ignored)
```
