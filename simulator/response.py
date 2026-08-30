"""Hidden response functions — the part the agent must never see.

`p_success(truth, action_kind, when)` is the simulator's private answer to
"if this action executed at this datetime, what is P(the debit clears)?".
The agent has its OWN, separate belief model (`agent/success_model.py`, Phase 7);
if these two ever share code, ground truth has leaked.

`revocation_hazard(...)` exposes the competing-risk hazard the Phase 2 harness
rolls forward alongside recovery.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import numpy as np

from .calendar_utils import parse_dt
from .customers import LedgerEntry, balance_at


@dataclass
class Truth:
    true_cause: str
    amount: float
    opening_balance: float
    ledger: list[LedgerEntry]
    logistic_scale: float
    max_success_prob: float
    reauth_success_prob: float
    outages: list[tuple[datetime, datetime]]
    p_during_outage: float
    limit_reset_dt: datetime | None
    p_before_reset: float
    h_base: float
    ltv_true: float


def _funded_prob(t: Truth, when: datetime) -> float:
    bal = balance_at(t.opening_balance, t.ledger, when)
    x = float(np.clip((bal - t.amount) / max(t.logistic_scale, 1.0), -30.0, 30.0))
    return t.max_success_prob / (1.0 + float(np.exp(-x)))


def _in_outage(t: Truth, when: datetime) -> bool:
    return any(s <= when <= e for (s, e) in t.outages)


def p_success(t: Truth, action_kind: str, when: datetime) -> float:
    """action_kind: 'retry' (attempt the debit) or 'reauth' (replace the mandate).
    Messages (nudge/SMS) never debit and are not valid here."""
    if action_kind == "reauth":
        return float(t.reauth_success_prob)
    if action_kind != "retry":
        raise ValueError(f"not a debiting action: {action_kind!r}")

    c = t.true_cause
    if c == "mandate_dead":
        return 0.0
    if c == "bank_downtime":
        if _in_outage(t, when):
            return float(t.p_during_outage)
        return _funded_prob(t, when)
    if c == "limit_breach":
        if t.limit_reset_dt is not None and when < t.limit_reset_dt:
            return float(t.p_before_reset)
        return _funded_prob(t, when)
    # insufficient_balance
    return _funded_prob(t, when)


def revocation_hazard(h_base: float, messages: list[tuple[int, str]],
                      day_index: int, rev_cfg: dict) -> float:
    """Daily P(customer revokes the mandate) on `day_index` (days since the failure).

    messages: list of (day_sent, kind) already dispatched, kind keyed into
    rev_cfg['message_bump'].
    """
    h = h_base
    prior = sorted((d, k) for (d, k) in messages if d <= day_index)
    for (d_sent, kind) in prior:
        cfg = rev_cfg["message_bump"].get(kind)
        if cfg is None:
            continue
        age = day_index - d_sent
        if age < 0 or age > cfg["decay_days"]:
            continue
        decay = max(0.0, 1.0 - age / (cfg["decay_days"] + 1.0))
        k_within_7 = sum(1 for (d2, _) in prior if 0 <= d_sent - d2 <= 7)
        fatigue = rev_cfg["fatigue_multiplier"] ** max(0, k_within_7 - 1)
        h += cfg["bump"] * decay * fatigue
    return min(h, 0.6)


def truth_from_record(tr: dict) -> Truth:
    """Rebuild a Truth from a serialized `truth.jsonl` row (used by tests and the harness)."""
    cp = tr["cause_params"]
    return Truth(
        true_cause=tr["true_cause"],
        amount=float(tr["amount"]),
        opening_balance=float(tr["opening_balance"]),
        ledger=[(parse_dt(x[0]), float(x[1]), x[2]) for x in tr["ledger"]],
        logistic_scale=float(cp["logistic_scale"]),
        max_success_prob=float(cp["max_success_prob"]),
        reauth_success_prob=float(cp["reauth_success_prob"]),
        outages=[(parse_dt(s), parse_dt(e)) for s, e in cp["outages"]],
        p_during_outage=float(tr["p_during_outage"]),
        limit_reset_dt=parse_dt(cp["limit_reset_dt"]) if cp["limit_reset_dt"] else None,
        p_before_reset=float(tr["p_before_reset"]),
        h_base=float(tr["h_base"]),
        ltv_true=float(tr["ltv_true"]),
    )
