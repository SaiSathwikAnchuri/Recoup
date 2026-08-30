# Recoup

**A cost-aware recovery policy for failed UPI AutoPay mandate debits.**
Razorpay AI Buildathon · Track 03 — AI Revenue Recovery.

> **Result (400 simulated mandate failures, seed 42):** Recoup recovered **₹717k** against
> **₹302k** for fixed-schedule retry, on **45% fewer debit attempts** (544 vs 994) and
> **~⅕ the messages**, while preserving **25 more mandates**. Net of action cost, the
> missed-cycle penalty and lost-mandate LTV, that is **+₹2,397 per case (95% CI
> [+₹1,315, +₹3,648])**. It escalated **27** cases it judged not worth pursuing, each with
> a logged rationale.

## The problem, in three sentences

UPI AutoPay debits fail 8–15% of the time, mostly because an account is briefly short of
funds, and roughly 20 million mandates are revoked every month as a result. Merchants
respond with a fixed retry schedule that is blind to *why* a debit failed, *when* the
customer will have money, and what each retry or nudge *costs* — including the risk of
irritating a recoverable customer into cancelling. Recoup replaces that schedule with a
per-case decision: infer the cause with calibrated confidence, predict a funding window,
price each permitted intervention against its cost, and stop when nothing is worth doing.

Full write-up: **`docs/`** / the project report artifact.

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
| 9 | Oracle, ablations, sensitivity, fairness slice, video | ⬜ |

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
  **+₹2,397/case net value, 95% CI [+₹1,315, +₹3,648]** · +₹1,038/case recovered ·
  +0.06 mandates/case preserved. Across 4 world seeds it beats `liquidity_aware` on net
  value on 3, and on timeliness on all 4.

**Classifier** (`results/classifier_report.json`): test accuracy **0.90** vs a 0.60
majority prior; **ECE 0.047** calibrated (0.085 uncalibrated); holdout batch 0.885 / 0.056.

**Liquidity model** (`results/liquidity_report.json`): funding-day **MAE 6.1 days** vs
11.8 for a modal-day heuristic and 11.8 for fixed-day-7; p85 quantile coverage 0.83
(0.84 on the holdout batch).

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
python -m pytest -q
```

Outputs (git-ignored):

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

---

## What is synthetic, and why

Real UPI mandate / bank data is confidential and PII-bearing under DPDP; NPCI publishes
only aggregates. So every case is generated. Each carries a **hidden true cause** and a
**hidden response function** (how P(debit clears) moves over the horizon, per cause) that
the agent never sees — only the harness does. Priors live in `config/priors.yaml`, each
with a source tag; figures tagged `(VERIFY)` still need a working citation before
submission.

The claim this project tests does not depend on the absolute numbers being right — all
policies (Recoup and both baselines) face an identical simulated world, so the **gap**
between them is what is measured, and a sensitivity check (Phase 9) re-runs everything
under different priors to show the ordering holds.

### Cause priors (assumptions table)

| Cause | Prior | Basis |
|-------|------:|-------|
| `insufficient_balance` | 0.60 | Dominant UPI AutoPay failure mode; NPCI monthly data + press *(VERIFY)* |
| `bank_downtime` | 0.15 | NPCI beneficiary/remitter-bank-offline share; incident spikes *(VERIFY)* |
| `limit_breach` | 0.10 | Estimate — per-account AutoPay cap / per-transaction limit |
| `mandate_dead` | 0.15 | Estimate — revoked-at-bank / not-found; rises with mandate age |

Failure **codes are deliberately ambiguous** (`config/priors.yaml → failure_code_emission`):
one generic decline code is reachable from several causes, so the classifier cannot cheat.
Code strings are placeholders pending reconciliation with the real Razorpay Subscriptions /
NPCI taxonomy.

---

## What this does **not** do

- Voice / Hinglish calling — large failure surface, no effect on the metric.
- A dashboard as the primary artifact — the brief never asks for one.
- Multi-agent orchestration frameworks — this is one decision chain.
- Other failure classes — checkout abandonment, receivables, one-time payments.
- Real production data — impossible to obtain, and not expected.

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
  policies.py             cause_aware (Phase 4) + liquidity_aware (Phase 5) — staged, one idea each
  ev_policy.py            recoup (Phase 7) — the cost-aware expected-value decision + explain()
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
  metrics.py             batch summary, paired bootstrap CIs, per-cause/-income splits, exceptions
  run.py                 runner CLI + comparison table + --trace <case_id>
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
```
