"""Synthetic customers: income patterns, competing debits, and a dated cash ledger.

The ledger is the spine of the `insufficient_balance` cause. A customer's realised
cashflow over the horizon is: opening_balance + a time-ordered list of signed amounts
(credits positive, debits negative). `balance_at(t)` is then a running sum.

This module is imported by both the generator and `response.py` (the hidden dynamics),
but NEVER by anything under `agent/`.
"""

from __future__ import annotations

import bisect
from dataclasses import dataclass
from datetime import datetime, timedelta

import numpy as np

from .calendar_utils import (
    CREDIT_HOUR,
    SCHED_DEBIT_HOUR,
    add_months,
    days_in_month,
    ist,
    months_between,
)

PATTERNS = ("salaried", "gig", "business")

# A ledger entry is a plain tuple: (datetime, signed_amount, kind)
LedgerEntry = tuple[datetime, float, str]


@dataclass
class Debit:
    day: int
    amount: float
    kind: str  # "emi" | "sip" | "supplier"


@dataclass
class Customer:
    income_pattern: str
    monthly_income: float
    pay_days: list[int]              # empty for gig/business (credits are irregular)
    rent_day: int
    rent_amount: float
    recurring_debits: list[Debit]
    discretionary_monthly: float
    buffer_frac: float               # opening balance as a fraction of monthly income
    h_base: float                    # baseline daily revocation hazard
    n_credits: int = 0               # irregular credits per month (gig/business)
    credit_lumpiness: float = 1.0    # >1 => a few large credits; <1 => many small ones


# ---------------------------------------------------------------------------
# sampling
# ---------------------------------------------------------------------------
def sample_customer(rng: np.random.Generator, priors: dict, tight: bool = False) -> Customer:
    """`tight=True` draws a cash-flow-constrained customer: lower income percentile,
    thinner buffer, heavier rent. These are the cases where retry *timing* matters and
    the observable success history shows a real funding window (the Priya scenario)."""
    p = str(rng.choice(PATTERNS, p=[priors["income_pattern_mix"][k] for k in PATTERNS]))
    h_base = priors["revocation"]["h_base_daily"][p]

    if p == "salaried":
        inc = (float(np.clip(rng.lognormal(np.log(30_000), 0.40), 16_000, 70_000)) if tight
               else float(np.clip(rng.lognormal(np.log(52_000), 0.5), 22_000, 180_000)))
        pay_days = [int(rng.choice([1, 1, 1, 1, 2, 3, 5, 7, 7]))]
        rent_day = int(rng.choice([1, 2, 3, 4, 5, 25, 27, 28]))
        rent_amount = inc * (rng.uniform(0.30, 0.46) if tight else rng.uniform(0.20, 0.40))
        n_emi = int(rng.choice([0, 1, 1, 2], p=[0.35, 0.35, 0.20, 0.10]))
        emis = [Debit(int(rng.choice([2, 3, 4, 5, 6, 7])), inc * rng.uniform(0.06, 0.22), "emi")
                for _ in range(n_emi)]
        n_sip = int(rng.choice([0, 1, 1, 2], p=[0.40, 0.30, 0.20, 0.10]))
        sips = [Debit(int(rng.choice([1, 2, 3, 5, 7, 10])),
                      float(rng.choice([500, 1000, 1500, 2000, 2500, 3000, 5000])), "sip")
                for _ in range(n_sip)]
        rec = emis + sips
        savings_frac = float(rng.uniform(-0.04, 0.06) if tight else rng.uniform(0.02, 0.15))
        disc = inc * (1 - savings_frac) - rent_amount - sum(d.amount for d in rec)
        buf = float(rng.uniform(0.005, 0.05) if tight else rng.uniform(0.03, 0.18))
        return Customer(p, inc, pay_days, rent_day, rent_amount, rec,
                        max(disc, inc * 0.10), buf, h_base)

    if p == "gig":
        inc = (float(np.clip(rng.lognormal(np.log(21_000), 0.45), 11_000, 45_000)) if tight
               else float(np.clip(rng.lognormal(np.log(34_000), 0.55), 15_000, 90_000)))
        rent_day = int(rng.choice([1, 2, 3, 4, 5, 6, 7]))
        rent_amount = inc * (rng.uniform(0.30, 0.48) if tight else rng.uniform(0.22, 0.42))
        n_emi = int(rng.choice([0, 1], p=[0.6, 0.4]))
        emis = [Debit(int(rng.choice([3, 5, 7, 10])), inc * rng.uniform(0.05, 0.12), "emi")
                for _ in range(n_emi)]
        savings_frac = float(rng.uniform(-0.10, 0.02) if tight else rng.uniform(-0.05, 0.08))
        disc = inc * (1 - savings_frac) - rent_amount - sum(d.amount for d in emis)
        buf = float(rng.uniform(0.003, 0.04) if tight else rng.uniform(0.01, 0.10))
        c = Customer(p, inc, [], rent_day, rent_amount, emis,
                     max(disc, inc * 0.10), buf, h_base)
        c.n_credits = int(rng.integers(8, 21))
        c.credit_lumpiness = 0.6
        return c

    # business
    inc = (float(np.clip(rng.lognormal(np.log(55_000), 0.5), 30_000, 130_000)) if tight
           else float(np.clip(rng.lognormal(np.log(90_000), 0.6), 40_000, 300_000)))
    rent_day = int(rng.choice([1, 2, 3, 5, 7, 10]))
    rent_amount = inc * rng.uniform(0.10, 0.25)
    n_deb = int(rng.integers(1, 4))
    debs = [Debit(int(rng.integers(1, 28)), inc * rng.uniform(0.10, 0.30), "supplier")
            for _ in range(n_deb)]
    savings_frac = float(rng.uniform(-0.15, 0.05) if tight else rng.uniform(-0.10, 0.20))
    disc = inc * (1 - savings_frac) - rent_amount - sum(d.amount for d in debs)
    buf = float(rng.uniform(0.02, 0.10) if tight else rng.uniform(0.05, 0.25))
    c = Customer(p, inc, [], rent_day, rent_amount, debs,
                 max(disc, inc * 0.10), buf, h_base)
    c.n_credits = int(rng.integers(1, 5))
    c.credit_lumpiness = 2.5
    return c


# ---------------------------------------------------------------------------
# ledger construction
# ---------------------------------------------------------------------------
def _month_entries(cust: Customer, rng: np.random.Generator, y: int, m: int) -> list[LedgerEntry]:
    dim = days_in_month(y, m)
    out: list[LedgerEntry] = []

    # --- credits ---
    if cust.pay_days:
        per = cust.monthly_income / len(cust.pay_days)
        for pd in cust.pay_days:
            out.append((ist(y, m, min(pd, dim), CREDIT_HOUR), per, "salary"))
    else:
        n = max(1, cust.n_credits)
        w = rng.gamma(1.0 / cust.credit_lumpiness, cust.credit_lumpiness, size=n)
        w = w / w.sum()
        days = rng.integers(1, dim + 1, size=n)
        for i in range(n):
            out.append((ist(y, m, int(days[i]), int(rng.integers(9, 20))),
                        float(cust.monthly_income * w[i]), "credit"))

    # --- scheduled debits ---
    out.append((ist(y, m, min(cust.rent_day, dim), SCHED_DEBIT_HOUR), -cust.rent_amount, "rent"))
    for d in cust.recurring_debits:
        out.append((ist(y, m, min(d.day, dim), SCHED_DEBIT_HOUR), -d.amount, d.kind))

    # --- discretionary spend, distributed across the month, boosted just after credits ---
    credit_days = sorted(e[0].day for e in out if e[1] > 0)
    weights = np.empty(dim)
    for i in range(dim):
        day = i + 1
        prev = [cd for cd in credit_days if cd <= day]
        since = (day - prev[-1]) if prev else 6
        weights[i] = (1.0 + 1.2 * np.exp(-since / 4.0)) * rng.gamma(2.0, 1.0)
    weights = weights / weights.sum() * cust.discretionary_monthly
    for i in range(dim):
        amt = float(weights[i])
        if amt <= 0:
            continue
        ntx = int(rng.integers(1, 4))
        for _ in range(ntx):
            out.append((ist(y, m, i + 1, int(rng.integers(8, 23))), -amt / ntx, "spend"))

    return out


def build_ledger(cust: Customer, rng: np.random.Generator,
                 start_dt: datetime, days: int) -> list[LedgerEntry]:
    end_dt = start_dt + timedelta(days=days)
    entries: list[LedgerEntry] = []
    for (y, m) in months_between(start_dt, end_dt):
        entries.extend(_month_entries(cust, rng, y, m))
    entries = [e for e in entries if start_dt <= e[0] <= end_dt]
    entries.sort(key=lambda e: e[0])
    return entries


def balance_at(opening: float, ledger: list[LedgerEntry], t: datetime) -> float:
    """Running balance at time `t`. Assumes `ledger` is sorted ascending by datetime."""
    b = opening
    for (dt, amt, _kind) in ledger:
        if dt <= t:
            b += amt
        else:
            break
    return b


def funding_day_profile(cust: Customer, rng: np.random.Generator, amount: float,
                        debit_day: int, n: int = 11,
                        end: tuple[int, int] = (2026, 3)) -> list[int | None]:
    """Continuous back-simulation of `n` cycles ending just before month `end`.

    For each cycle: from `debit_day`, find the first day the running balance covers
    `amount`; record that day-of-month and deduct `amount` (the debit cleared), so
    carry-over between months creates realistic variation. A cycle that never clears
    within ~12 days is recorded as None (a genuine miss). This list is the observable
    signal the liquidity model (Phase 5) learns from."""
    sy, sm = add_months(end[0], end[1], -n)
    start_dt = ist(sy, sm, 1, 0, 0)
    end_dt = ist(end[0], end[1], 1, 0, 0)

    ledger: list[LedgerEntry] = []
    for (y, m) in months_between(start_dt, end_dt - timedelta(seconds=1)):
        ledger.extend(_month_entries(cust, rng, y, m))
    ledger.sort(key=lambda e: e[0])
    ts = [e[0] for e in ledger]
    cum: list[float] = []
    running = 0.0
    for e in ledger:
        running += e[1]
        cum.append(running)

    opening = cust.monthly_income * cust.buffer_frac
    past_debits: list[tuple[datetime, float]] = []

    def bal(t: datetime) -> float:
        i = bisect.bisect_right(ts, t)
        b = opening + (cum[i - 1] if i > 0 else 0.0)
        for (dt, amt) in past_debits:
            if dt <= t:
                b += amt
        return b

    profile: list[int | None] = []
    cy, cm = sy, sm
    for _ in range(n):
        dim = days_in_month(cy, cm)
        dd = min(debit_day, dim)
        hit = None
        for off in range(0, 13):
            py, pm, pdd = cy, cm, dd + off
            if pdd > dim:
                pdd -= dim
                pm += 1
                if pm == 13:
                    pm, py = 1, py + 1
            probe_dt = ist(py, pm, pdd, SCHED_DEBIT_HOUR + 3)
            if bal(probe_dt) >= amount:
                hit = pdd
                past_debits.append((probe_dt, -amount))
                break
        profile.append(hit)
        cy, cm = add_months(cy, cm, 1)
    return profile
