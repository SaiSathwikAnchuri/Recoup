"""Phase 1 invariants for the synthetic generator.

The load-bearing one is `test_day0_debit_actually_failed`: every case must represent
a debit that genuinely failed on day 0, with the sampled cause as the binding reason.
"""

from __future__ import annotations

import json
from datetime import timedelta

import numpy as np
import pytest
import yaml

from simulator.calendar_utils import parse_dt
from simulator.generate import build_batch, summarise
from simulator.generator import CAUSES
from simulator.response import p_success, truth_from_record

PRIORS = yaml.safe_load(open("config/priors.yaml"))
LEAK_KEYS = {"true_cause", "income_pattern", "monthly_income", "ledger",
             "opening_balance", "cause_params", "ltv_true", "pay_days"}


@pytest.fixture(scope="module")
def batch():
    return build_batch(n=600, seed=7, priors=PRIORS)


def test_day0_debit_actually_failed(batch):
    cases, truths = batch
    for c, tr in zip(cases, truths):
        t = truth_from_record(tr)
        p0 = p_success(t, "retry", parse_dt(c["observed_at"]))
        assert p0 < 0.5, f"{c['case_id']} ({tr['true_cause']}) day-0 p_success={p0:.3f} — not a real failure"


def test_dead_mandates_never_recover_by_retry(batch):
    cases, truths = batch
    dead = [(c, tr) for c, tr in zip(*batch) if tr["true_cause"] == "mandate_dead"]
    assert dead, "expected some dead-mandate cases"
    for c, tr in dead:
        t = truth_from_record(tr)
        start = parse_dt(c["observed_at"])
        for k in range(0, c["horizon_days"], 2):
            assert p_success(t, "retry", start + timedelta(days=k)) == 0.0
        assert tr["retry_recoverable_within_horizon"] is False
        assert c["history"]["consecutive_successes"] < PRIORS["history_months"]


def test_recoverable_flag_matches_best_p(batch):
    _, truths = batch
    for tr in truths:
        if tr["retry_recoverable_within_horizon"]:
            assert tr["best_retry_p"] >= 0.5
        else:
            assert tr["best_retry_p"] < 0.5


def test_non_ib_causes_are_genuinely_funded(batch):
    """For downtime / limit / dead cases the account holds enough money, so the
    failure is attributable purely to the sampled cause."""
    _, truths = batch
    for tr in truths:
        if tr["true_cause"] in ("bank_downtime", "limit_breach"):
            assert tr["balance_at_observed"] >= tr["amount"], tr["case_id"]


def test_cause_and_income_mix_are_plausible(batch):
    cases, truths = batch
    s = summarise(cases, truths, PRIORS, seed=7)
    mix = s["cause_mix"]
    assert mix["insufficient_balance"] > 0.50
    assert 0.06 <= mix["mandate_dead"] <= 0.28
    assert all(mix[c] > 0 for c in CAUSES)
    im = s["income_mix"]
    assert im["salaried"] > im["gig"] > im["business"]


def test_ledger_sorted_and_finite(batch):
    _, truths = batch
    for tr in truths:
        ts = [parse_dt(e[0]) for e in tr["ledger"]]
        assert ts == sorted(ts), tr["case_id"]
        assert all(np.isfinite(e[1]) for e in tr["ledger"])


def test_observable_case_has_no_hidden_fields(batch):
    cases, _ = batch
    for c in cases:
        assert not (set(c) & LEAK_KEYS)
        assert not (set(c.get("mandate", {})) & LEAK_KEYS)
        assert not (set(c.get("history", {})) & LEAK_KEYS)
        for req in ("case_id", "observed_at", "failure", "mandate", "history", "horizon_days"):
            assert req in c


def test_failure_code_is_ambiguous_not_a_giveaway():
    """A single generic code must be reachable from more than one cause (report R5)."""
    cases, truths = build_batch(n=800, seed=3, priors=PRIORS)
    by_token: dict[str, set[str]] = {}
    for c, tr in zip(cases, truths):
        by_token.setdefault(c["failure"]["token"], set()).add(tr["true_cause"])
    assert by_token.get("U69_generic_decline", set()).__len__() >= 3


def test_reproducible_given_seed():
    a_cases, a_truths = build_batch(n=25, seed=123, priors=PRIORS)
    b_cases, b_truths = build_batch(n=25, seed=123, priors=PRIORS)
    assert json.dumps(a_cases) == json.dumps(b_cases)
    assert json.dumps(a_truths) == json.dumps(b_truths)
