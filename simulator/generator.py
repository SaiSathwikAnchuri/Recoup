"""Build one synthetic case = an observable record + its hidden ground truth.

A case is a failed UPI AutoPay debit on `observed_at`. The generator:
  1. samples a customer (cashflow, competing debits) and a mandate,
  2. samples the true cause from the priors (age/amount-adjusted),
  3. engineers the latent state so that cause is genuinely BINDING on day 0
     (the observed failure is real, not incidental),
  4. emits an ambiguous failure code (report R5),
  5. derives the observable payment history from a cashflow back-simulation.

The `case` dict is all the agent ever sees. The `truth` dict is harness-only.
"""

from __future__ import annotations

from datetime import timedelta

import numpy as np

from .calendar_utils import add_months, days_in_month, ist, iso
from .customers import (
    balance_at,
    build_ledger,
    funding_day_profile,
    sample_customer,
)
from .response import Truth, p_success

CAUSES = ("insufficient_balance", "bank_downtime", "limit_breach", "mandate_dead")

# debit amount ranges (INR) by subscription category
CATEGORIES = {
    "ott":       (149, 799),
    "sip":       (500, 5000),
    "emi":       (1000, 15000),
    "insurance": (300, 2500),
    "utility":   (200, 3000),
}
CATEGORY_P = [0.30, 0.20, 0.20, 0.15, 0.15]

REF_YEAR, REF_MONTH = 2026, 3   # the failed debit happens in March 2026

# Placeholder code tokens -> (npci_code, human description).
# RECONCILE with the real Razorpay Subscriptions / NPCI decline taxonomy (report R6).
CODE_META = {
    "U30_insufficient_funds": ("U30", "insufficient funds"),
    "U69_generic_decline":    ("U69", "transaction declined by the bank"),
    "U88_bank_offline":       ("U88", "remitter / beneficiary bank offline"),
    "U67_limit_exceeded":     ("U67", "per-transaction or mandate limit exceeded"),
    "U16_risk_declined":      ("U16", "declined by risk / fraud check"),
    "ER_mandate_revoked":     ("BAD_REQUEST_ERROR", "mandate revoked or cancelled at the bank"),
    "ER_mandate_issue":       ("BAD_REQUEST_ERROR", "mandate not found / not active"),
}


def sample_mandate(rng: np.random.Generator) -> dict:
    cat = str(rng.choice(list(CATEGORIES), p=CATEGORY_P))
    lo, hi = CATEGORIES[cat]
    amount = round(float(rng.uniform(lo, hi)))
    return dict(
        category=cat,
        amount=float(amount),
        debit_day=int(rng.integers(1, 29)),
        age_months=int(rng.integers(1, 40)),
        authorised_ceiling=round(amount * float(rng.choice([1.0, 1.25, 1.5, 2.0]))),
    )


def adjust_cause_probs(base: dict, mandate: dict) -> dict:
    """Older mandates die more often; larger debits breach limits more often."""
    p = dict(base)
    age = mandate["age_months"]
    p["mandate_dead"] *= 0.4 + min(age, 36) / 36 * 1.4          # 0.4 .. 1.8
    p["limit_breach"] *= 0.6 + mandate["amount"] / 15_000 * 1.2
    s = sum(p.values())
    return {k: v / s for k, v in p.items()}


def sample_failure_code(true_cause: str, rng: np.random.Generator, priors: dict) -> dict:
    emit = priors["failure_code_emission"][true_cause]
    keys = list(emit)
    token = str(rng.choice(keys, p=[emit[k] for k in keys]))
    npci, desc = CODE_META[token]
    return {
        "token": token,
        "npci_code": npci,
        "error_code": "BAD_REQUEST_ERROR",
        "error_reason": "payment_failed",
        "description": desc,
    }


def build_truth_and_case(idx: int, rng: np.random.Generator, priors: dict) -> tuple[dict, dict]:
    horizon = priors["horizon_days"]
    mandate = sample_mandate(rng)

    dd = min(mandate["debit_day"], days_in_month(REF_YEAR, REF_MONTH))
    observed_at = ist(REF_YEAR, REF_MONTH, dd, int(rng.integers(9, 12)), int(rng.integers(0, 60)))
    start_dt = ist(REF_YEAR, REF_MONTH, dd, 0, 0)

    cause_p = adjust_cause_probs(priors["cause_mix"], mandate)
    true_cause = str(rng.choice(CAUSES, p=[cause_p[c] for c in CAUSES]))

    # cash-flow-constrained customers: mostly the liquidity failures, a few others
    tight = rng.random() < (0.75 if true_cause == "insufficient_balance" else 0.15)
    cust = sample_customer(rng, priors, tight=tight)

    ledger = build_ledger(cust, rng, start_dt, horizon)
    opening = cust.monthly_income * cust.buffer_frac
    amount = mandate["amount"]
    scale = amount * priors["insufficient_balance"]["success_logistic_scale_frac"]
    maxp = priors["max_success_prob"]

    bal0 = balance_at(opening, ledger, observed_at)

    outages: list[tuple] = []
    limit_reset_dt = None
    p_before_reset = priors["limit_breach"]["p_success_before_reset"]
    p_during = priors["bank_downtime"]["p_success_during_outage"]
    reauth_p = maxp * 0.9

    ib_unrecoverable = False
    if true_cause == "insufficient_balance":
        ib = priors["insufficient_balance"]
        target = amount * float(rng.uniform(*ib["shortfall_frac_range"]))
        opening += target - bal0                       # shift the whole trajectory down
        if rng.random() < ib["unrecoverable_frac"]:
            ib_unrecoverable = True
            # a large unexpected debit the next day: the account never recovers in-horizon
            fut_max = max(balance_at(opening, ledger, observed_at + timedelta(days=k))
                          for k in range(1, horizon))
            shock = (fut_max - amount) + amount * 0.5
            if shock > 0:
                ledger.append((observed_at + timedelta(hours=30), -shock, "shock"))
                ledger.sort(key=lambda e: e[0])
    else:
        need = amount * 1.20                            # genuinely funded: failure is purely the other cause
        if bal0 < need:
            opening += need - bal0
        if true_cause == "bank_downtime":
            bd = priors["bank_downtime"]
            dur = max(bd["outage_min_hours"], float(rng.exponential(bd["outage_mean_hours"])))
            # start the outage so that observed_at falls strictly inside [o_start, o_start+dur]
            o_start = observed_at - timedelta(hours=float(rng.uniform(0, dur * 0.8)))
            outages.append((o_start, o_start + timedelta(hours=dur)))
            if rng.random() < bd["recurring_outage_prob"]:
                d2 = observed_at + timedelta(days=float(rng.uniform(2, horizon - 2)))
                dur2 = max(0.5, float(rng.exponential(bd["outage_mean_hours"] * 0.5)))
                outages.append((d2, d2 + timedelta(hours=dur2)))
        elif true_cause == "limit_breach":
            ny, nm = add_months(REF_YEAR, REF_MONTH, 1)
            limit_reset_dt = ist(ny, nm, 1, 0, 0)
        elif true_cause == "mandate_dead":
            a, b = priors["mandate_dead"]["reauth_success_prob_beta"]
            reauth_p = float(rng.beta(a, b))

    truth = Truth(
        true_cause=true_cause, amount=amount, opening_balance=opening, ledger=ledger,
        logistic_scale=scale, max_success_prob=maxp, reauth_success_prob=reauth_p,
        outages=outages, p_during_outage=p_during,
        limit_reset_dt=limit_reset_dt, p_before_reset=p_before_reset,
        h_base=cust.h_base,
        ltv_true=amount * priors["ltv"]["months"]
        * float(rng.uniform(*priors["ltv"]["retention_factor_range"])),
    )

    bal_observed = balance_at(opening, ledger, observed_at)
    day0_p = p_success(truth, "retry", observed_at)

    def scan() -> tuple[float, object]:
        best_p, best_dt = 0.0, None
        t, end = observed_at, observed_at + timedelta(days=horizon)
        while t <= end:
            pp = p_success(truth, "retry", t)
            if pp > best_p:
                best_p, best_dt = pp, t
            t += timedelta(hours=6)
        return best_p, best_dt

    best_p, best_dt = scan()

    # An IB case we did NOT mark unrecoverable must actually fund within the horizon
    # (otherwise it is indistinguishable from a chronic case). Inject a catch-up credit
    # shortly after the next pay-day to guarantee a real funding window.
    if true_cause == "insufficient_balance" and not ib_unrecoverable and best_p < 0.5:
        pay = cust.pay_days[0] if cust.pay_days else 3
        ny, nm = add_months(REF_YEAR, REF_MONTH, 1)
        catch_dt = ist(ny, nm, min(pay, 28), 11, 0) + timedelta(days=float(rng.uniform(0, 3)))
        if catch_dt > observed_at + timedelta(days=horizon):
            catch_dt = observed_at + timedelta(days=horizon - 2)
        ledger.append((catch_dt, amount * float(rng.uniform(1.3, 2.2)), "catch_up"))
        ledger.sort(key=lambda e: e[0])
        best_p, best_dt = scan()

    retry_recoverable = best_p >= 0.5

    # observable payment history from a continuous cashflow back-simulation
    profile = funding_day_profile(cust, rng, amount, mandate["debit_day"],
                                  n=priors["history_months"], end=(REF_YEAR, REF_MONTH))
    hits = [d for d in profile if d is not None]

    # trailing run of successful cycles, before any cause-specific recent-failure streak
    trailing = 0
    for d in reversed(profile):
        if d is None:
            break
        trailing += 1
    fail_streak = 0
    if true_cause == "mandate_dead":
        fail_streak = int(rng.integers(1, 3))
    elif true_cause in ("bank_downtime", "limit_breach") and rng.random() < 0.2:
        fail_streak = 1
    consecutive = max(0, trailing - fail_streak)

    last_ok_day = min(hits[-1] if hits else mandate["debit_day"], 28)
    ly, lm = add_months(REF_YEAR, REF_MONTH, -(1 + fail_streak))
    last_success = ist(ly, lm, last_ok_day)
    days_since = (observed_at.date() - last_success.date()).days

    cy, cm = add_months(REF_YEAR, REF_MONTH, -mandate["age_months"])
    mandate_created = ist(cy, cm, min(dd, 28))

    case = {
        "case_id": f"c{idx:04d}",
        "observed_at": iso(observed_at),
        "failure": sample_failure_code(true_cause, rng, priors),
        "mandate": {
            "category": mandate["category"],
            "amount": amount,
            "debit_day": mandate["debit_day"],
            "age_months": mandate["age_months"],
            "authorised_ceiling": mandate["authorised_ceiling"],
            "retries_used_this_cycle": 0,
            "created_at": iso(mandate_created),
        },
        "history": {
            "consecutive_successes": consecutive,
            "last_success": last_success.date().isoformat(),
            "days_since_last_success": days_since,
            "success_days_of_month": hits,
            "cycles_observed": priors["history_months"],
        },
        "horizon_days": horizon,
    }

    truth_rec = {
        "case_id": case["case_id"],
        "true_cause": true_cause,
        "amount": amount,
        "cashflow_tight": bool(tight),
        "income_pattern": cust.income_pattern,
        "monthly_income": round(cust.monthly_income),
        "pay_days": cust.pay_days,
        "competing_debits": [dict(day=d.day, amount=round(d.amount), kind=d.kind)
                             for d in cust.recurring_debits],
        "opening_balance": round(opening, 2),
        "balance_at_observed": round(bal_observed, 2),
        "day0_p_success": round(float(day0_p), 4),
        "retry_recoverable_within_horizon": bool(retry_recoverable),
        "best_retry_p": round(float(best_p), 4),
        "best_retry_at": iso(best_dt) if best_dt else None,
        "fail_streak_before_day0": fail_streak,
        "p_during_outage": p_during,
        "p_before_reset": p_before_reset,
        "h_base": cust.h_base,
        "ltv_true": round(truth.ltv_true, 2),
        "cause_params": {
            "logistic_scale": round(scale, 2),
            "max_success_prob": maxp,
            "reauth_success_prob": round(reauth_p, 4),
            "outages": [[iso(s), iso(e)] for (s, e) in outages],
            "limit_reset_dt": iso(limit_reset_dt) if limit_reset_dt else None,
        },
        "ledger": [[iso(dt), round(a, 2), k] for (dt, a, k) in ledger],
    }
    return case, truth_rec
