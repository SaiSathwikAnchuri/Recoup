# Recoup

**A cost-aware recovery policy for failed UPI AutoPay mandate debits.**
Razorpay AI Buildathon · Track 03 — AI Revenue Recovery.

> **Result:** _`Across N simulated mandate failures, Recoup recovered ₹X against ₹Y for
> fixed-schedule retry — using Z% fewer attempts and revoking M fewer mandates. It
> escalated K cases it could not resolve.`_
> *(placeholder — filled from real runs once the policy and harness land)*

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
| 6 | Constraint layer + property tests | ⬜ |
| 7 | Cost-aware policy | ⬜ |
| 8 | Audit trail + LLM explain layer | ⬜ |
| 9 | Oracle, ablations, sensitivity, fairness slice, video | ⬜ |

### Where things stand (400 cases, seed 42)

| Policy | Recovered | Rate | Attempts | Msg/case | On-time | Preserved | Escalated |
|--------|----------:|-----:|---------:|---------:|--------:|----------:|----------:|
| `never_act` (floor) | ₹0 | 0.0% | 0 | 0.00 | 0.0% | 88.0% | 0 |
| `fixed_schedule` (Baseline A) | ₹302,303 | 32.8% | 994 | 0.73 | 30.0% | 88.0% | 0 |
| `always_nudge` (Baseline B) | ₹301,066 | 31.5% | 970 | 1.45 | 25.8% | 86.5% | 0 |
| `cause_aware` (Phase 4 — + escalate branch) | ₹386,543 | 42.5% | 804 | 0.74 | 39.8% | 89.2% | 24 |
| `liquidity_aware` (Phase 5 — + funding-window timing) | ₹724,340 | 73.5% | 553 | 0.17 | 26.0% | 94.2% | 24 |
| _perfect-timing reference (single retry, no re-auth)_ | _₹695,950_ | _67.5%_ | _298_ | _0.00_ | — | _93.0%_ | — |

Each policy adds **one idea** so the harness can price it in isolation:

- **`cause_aware`** — when the calibrated classifier is confident the mandate is dead
  (P ≥ 0.30, ~90% precision), stop retrying, send one re-auth, hand to a human.
  **+₹84k** (paired 95% CI [+₹116, +₹329]/case, excludes zero); a re-auth revives ~57%
  of dead mandates that retries never touch.
- **`liquidity_aware`** — for everything not dead, schedule retries at the model's
  predicted **p50 / p85 funding days** instead of days 1/3/7; a limit-breach waits for
  the cap to reset on the 1st. **+₹422k vs fixed_schedule** (CI [+₹795, +₹1,353]/case),
  on **fewer attempts** (553 vs 994) and **fewer messages**, with mandate preservation
  up to 94%. The cost: recovery lands ~17 days out, so on-time rate drops to 26% — the
  trade the Phase 7 EV policy will make deliberately, against the missed-cycle penalty.

**Classifier** (`results/classifier_report.json`): test accuracy **0.90** vs a 0.60
majority prior; **ECE 0.047** calibrated (0.085 uncalibrated); holdout batch 0.885 / 0.056.

**Liquidity model** (`results/liquidity_report.json`): funding-day **MAE 6.1 days** vs
11.8 for a modal-day heuristic and 11.8 for fixed-day-7; p85 quantile coverage 0.83
(0.84 on the holdout batch).

---

## Reproduce

```bash
pip install -r requirements.txt

python tasks.py reproduce        # data + train (classifier + liquidity) + harness + tests
#   or, with make:  make reproduce

python -m simulator.generate --n 400 --seed 42 --out data
python -m simulator.show c0142 --ledger
python -m agent.train_classifier          # -> agent/models/cause_clf.pkl + results/classifier_report.json
python -m agent.train_liquidity           # -> agent/models/liquidity.pkl + results/liquidity_report.json
python -m harness.run --seed 42
python -m harness.run --trace c0142 --policy liquidity_aware
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
config/costs.yaml        what the AGENT believes actions cost + a mandate is worth  (estimates, not truth)
agent/
  costs.py               loads costs.yaml: action_cost, ltv_estimate, recovery_value, missed_cycle_penalty, message_fatigue_factor
  features.py             observable case -> numeric feature vector (never touches truth)
  classifier.py           calibrated multinomial cause classifier + should_escalate (the escalate branch)
  calibration.py          reliability bins / ECE / Brier — plain numpy
  liquidity.py            quantile funding-day model (p50 / p85 days-from-now)
  train_classifier.py     CLI: generate training batch (separate seed), fit + calibrate, tune threshold, report
  train_liquidity.py      CLI: train the funding-day quantile regressors, eval vs naive baselines
  policies.py             cause_aware (Phase 4) + liquidity_aware (Phase 5) — staged, one idea each
simulator/
  customers.py           income patterns, competing debits, cash ledger, back-sim history
  generator.py           assembles one case = observable record + hidden truth
  response.py            hidden response functions + revocation hazard  (never imported by agent/)
  generate.py            batch CLI
  show.py                human-readable single-case dump
harness/
  engine.py              competing-risk rollout: Plan/Action protocol, per-case World, run_case
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
```
