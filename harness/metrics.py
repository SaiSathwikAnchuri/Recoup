"""Batch scoring: headline metrics per policy, paired deltas with bootstrap CIs,
per-cause and per-income-pattern breakdowns, and the exception list.
"""

from __future__ import annotations

import numpy as np

from .engine import Outcome


def summarise(outcomes: list[Outcome]) -> dict:
    n = len(outcomes)
    rec = [o for o in outcomes if o.recovered]
    recovered_rs = sum(o.amount_recovered for o in outcomes)
    attempts = sum(o.attempts_used for o in outcomes)
    messages = sum(o.messages_sent for o in outcomes)
    revoked = sum(o.revoked for o in outcomes)
    dtr = [o.days_to_recovery for o in rec if o.days_to_recovery is not None]

    return {
        "n": n,
        "recovered_rs": round(recovered_rs, 0),
        "recovery_rate": round(len(rec) / n, 4),
        "on_time_rate": round(sum(o.recovered_on_time for o in outcomes) / n, 4),
        "attempts": attempts,
        "attempts_per_1k_recovered": round(attempts / (recovered_rs / 1000), 2) if recovered_rs else None,
        "messages": messages,
        "messages_per_case": round(messages / n, 3),
        "mandates_preserved_rate": round(sum(o.mandate_preserved for o in outcomes) / n, 4),
        "mandates_revoked": int(revoked),
        "recovered_late": int(sum(o.recovered and not o.recovered_on_time for o in outcomes)),
        "cycles_missed": int(sum(o.cycle_missed for o in outcomes)),
        "escalated": int(sum(o.escalated for o in outcomes)),
        "unresolved": int(sum(o.stop_reason in ("plan_exhausted", "max_rounds", "unresolved")
                              and not o.recovered and not o.revoked for o in outcomes)),
        "mean_days_to_recovery": round(float(np.mean(dtr)), 2) if dtr else None,
        "reauth_recoveries": int(sum(o.via_reauth for o in outcomes)),
    }


def by_key(outcomes: list[Outcome], key: str) -> dict:
    groups: dict[str, list[Outcome]] = {}
    for o in outcomes:
        groups.setdefault(getattr(o, key), []).append(o)
    return {
        k: {
            "n": len(v),
            "recovered_rs": round(sum(o.amount_recovered for o in v), 0),
            "recovery_rate": round(sum(o.recovered for o in v) / len(v), 3),
            "preserved_rate": round(sum(o.mandate_preserved for o in v) / len(v), 3),
            "attempts": sum(o.attempts_used for o in v),
            "messages": sum(o.messages_sent for o in v),
        }
        for k, v in sorted(groups.items())
    }


def paired_delta(treatment: list[Outcome], control: list[Outcome],
                 field: str, seed: int = 0, resamples: int = 2000) -> dict:
    """Bootstrap 95% CI on the mean per-case (treatment - control) difference.
    `treatment` and `control` must be aligned by case_id."""
    c = {o.case_id: o for o in control}
    diffs = np.array([getattr(t, field) - getattr(c[t.case_id], field)
                      for t in treatment if t.case_id in c], dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(diffs), size=(resamples, len(diffs)))
    means = diffs[idx].mean(axis=1)
    return {
        "mean": round(float(diffs.mean()), 2),
        "total": round(float(diffs.sum()), 2),
        "ci95": [round(float(np.percentile(means, 2.5)), 2),
                 round(float(np.percentile(means, 97.5)), 2)],
        "excludes_zero": bool(np.percentile(means, 2.5) > 0 or np.percentile(means, 97.5) < 0),
    }


def exception_list(outcomes: list[Outcome]) -> list[dict]:
    """Every case the policy could not resolve — escalated, or lapsed without recovery."""
    out = []
    for o in outcomes:
        if o.recovered:
            continue
        out.append({
            "case_id": o.case_id,
            "true_cause": o.true_cause,
            "outcome": "revoked" if o.revoked else ("escalated" if o.escalated else "lapsed"),
            "attempts_used": o.attempts_used,
            "messages_sent": o.messages_sent,
            "stop_reason": o.stop_reason,
        })
    return out
