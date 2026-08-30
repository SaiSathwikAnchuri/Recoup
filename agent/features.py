"""Turn one observable case record into a numeric feature vector.

Reads ONLY the fields the agent can see at decision time — `case["failure"]`,
`case["mandate"]`, `case["history"]`. It never touches truth. Keep it that way:
the no-leakage boundary (report R3) is enforced by `tests/test_classifier.py`.
"""

from __future__ import annotations

import math

import numpy as np

# observed failure-code tokens (config/priors.yaml -> failure_code_emission).
# an unknown token maps to the all-zero block — the model still has every other feature.
TOKENS = (
    "U30_insufficient_funds",
    "U69_generic_decline",
    "U88_bank_offline",
    "U67_limit_exceeded",
    "U16_risk_declined",
    "ER_mandate_revoked",
    "ER_mandate_issue",
)
CATEGORIES = ("ott", "sip", "emi", "insurance", "utility")

FEATURE_NAMES: tuple[str, ...] = (
    *(f"tok::{t}" for t in TOKENS),
    "log_amount",
    "age_months",
    "debit_day",
    "ceiling_util",
    "ceiling_util_ge_1",
    *(f"cat::{c}" for c in CATEGORIES),
    "consecutive_successes",
    "days_since_last_success",
    "cycles_observed",
    "n_funding_hits",
    "funding_hit_rate",
    "no_funding_hits",
    "zero_consecutive",
    "funding_day_std",
)


def featurize(case: dict) -> np.ndarray:
    f = case["failure"]
    m = case["mandate"]
    h = case["history"]

    token = f.get("token", "")
    category = m.get("category", "")
    amount = float(m.get("amount", 0.0))
    ceiling = float(m.get("authorised_ceiling", 0.0)) or 1.0
    hits = list(h.get("success_days_of_month", []) or [])
    cycles = int(h.get("cycles_observed", 0))
    dsls = h.get("days_since_last_success")
    dsls = float(dsls) if dsls is not None else 999.0

    row: list[float] = []
    row += [1.0 if token == t else 0.0 for t in TOKENS]
    row.append(math.log1p(max(amount, 0.0)))
    row.append(float(m.get("age_months", 0)))
    row.append(float(m.get("debit_day", 0)))
    row.append(amount / ceiling)
    row.append(1.0 if amount >= ceiling else 0.0)
    row += [1.0 if category == c else 0.0 for c in CATEGORIES]
    row.append(float(h.get("consecutive_successes", 0)))
    row.append(min(dsls, 400.0))
    row.append(float(cycles))
    row.append(float(len(hits)))
    row.append(len(hits) / cycles if cycles else 0.0)
    row.append(1.0 if not hits else 0.0)
    row.append(1.0 if int(h.get("consecutive_successes", 0)) == 0 else 0.0)
    row.append(float(np.std(hits)) if len(hits) >= 2 else 0.0)

    return np.asarray(row, dtype=float)


def featurize_batch(cases: list[dict]) -> np.ndarray:
    return np.vstack([featurize(c) for c in cases]) if cases else np.empty((0, len(FEATURE_NAMES)))
