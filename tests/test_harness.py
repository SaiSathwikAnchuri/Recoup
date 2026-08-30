"""Phase 2 invariants for the rollout engine and baselines."""

from __future__ import annotations

import json

import pytest
import yaml

from baselines import AlwaysNudge, FixedSchedule, NeverAct
from harness.engine import (
    EngagementState,
    HarnessConfig,
    Plan,
    ScheduledAction,
    run_batch,
    run_case,
)
from simulator.calendar_utils import parse_dt

PRIORS = yaml.safe_load(open("config/priors.yaml"))
CFG = HarnessConfig(rev_cfg=PRIORS["revocation"])


def _load(path):
    with open(path) as f:
        return [json.loads(x) for x in f]


@pytest.fixture(scope="module")
def data():
    cases = _load("data/cases.jsonl")
    truth = {t["case_id"]: t for t in _load("data/truth.jsonl")}
    return cases, truth


# --- baseline sanity -------------------------------------------------------
def test_never_act_recovers_nothing(data):
    cases, truth = data
    outs = run_batch(cases, truth, NeverAct(), CFG, run_seed=1)
    assert all(o.amount_recovered == 0 for o in outs)
    assert all(o.attempts_used == 0 and o.messages_sent == 0 for o in outs)


def test_dead_mandates_never_recovered_by_retry_policies(data):
    cases, truth = data
    for pol in (FixedSchedule(), AlwaysNudge()):
        outs = run_batch(cases, truth, pol, CFG, run_seed=3)
        dead = [o for o in outs if o.true_cause == "mandate_dead"]
        assert dead
        assert all(not o.recovered for o in dead), pol.name


# --- engine invariants ---------------------------------------------------
def test_attempt_budget_never_exceeded(data):
    cases, truth = data
    for pol in (FixedSchedule(), AlwaysNudge()):
        outs = run_batch(cases, truth, pol, CFG, run_seed=5)
        assert all(o.attempts_used <= CFG.attempt_budget for o in outs)


def test_recovery_amount_is_all_or_nothing(data):
    cases, truth = data
    outs = run_batch(cases, truth, FixedSchedule(), CFG, run_seed=7)
    for o in outs:
        assert o.amount_recovered == (o.amount if o.recovered else 0.0)
        assert o.mandate_preserved == (not o.revoked)
        if o.revoked:
            assert not o.recovered


def test_deterministic_same_seed(data):
    cases, truth = data
    a = run_batch(cases, truth, FixedSchedule(), CFG, run_seed=11)
    b = run_batch(cases, truth, FixedSchedule(), CFG, run_seed=11)
    assert [o.as_row() for o in a] == [o.as_row() for o in b]


def test_identical_plans_give_identical_outcomes(data):
    """Two different policy objects that emit the same plan must produce the same
    outcome for a case — the property paired comparison relies on."""
    cases, truth = data

    class P1:
        name = "p1"
        def plan(self, case, state: EngagementState):
            o = state.observed_at
            return Plan([ScheduledAction(o.replace(hour=11), "retry")], terminal="stop")

    class P2:
        name = "p2"
        def plan(self, case, state: EngagementState):
            o = state.observed_at
            return Plan([ScheduledAction(o.replace(hour=11), "retry")], terminal="stop")

    for c in cases[:80]:
        o1 = run_case(c, truth[c["case_id"]], P1(), CFG, run_seed=99)
        o2 = run_case(c, truth[c["case_id"]], P2(), CFG, run_seed=99)
        d1, d2 = o1.as_row(), o2.as_row()
        d1["policy"] = d2["policy"] = "x"
        assert d1 == d2, c["case_id"]


def test_oracle_timing_recovers_most_recoverable_cases(data):
    """Engine resolution check: a policy that retries exactly at the hidden best
    moment should recover almost every case flagged retry-recoverable."""
    cases, truth = data

    class Oracle:
        name = "oracle_timing"
        def plan(self, case, state: EngagementState):
            t = truth[case["case_id"]]
            if not t["retry_recoverable_within_horizon"]:
                return Plan([], terminal="escalate")
            return Plan([ScheduledAction(parse_dt(t["best_retry_at"]), "retry")], terminal="stop")

    outs = run_batch(cases, truth, Oracle(), CFG, run_seed=13)
    recoverable = [o for o in outs
                   if truth[o.case_id]["retry_recoverable_within_horizon"]]
    hit = sum(o.recovered for o in recoverable) / len(recoverable)
    assert hit > 0.85, hit


def test_baselines_beat_never_act_on_money(data):
    cases, truth = data
    na = sum(o.amount_recovered for o in run_batch(cases, truth, NeverAct(), CFG, 17))
    fs = sum(o.amount_recovered for o in run_batch(cases, truth, FixedSchedule(), CFG, 17))
    assert fs > na == 0


# --- cause_aware policy (Phase 4 escalate branch) ------------------------
def test_cause_aware_escalates_and_never_retries_a_flagged_mandate(data):
    cases, truth = data
    clf = pytest.importorskip("agent.classifier").CauseClassifier
    if not clf.default_exists():
        pytest.skip("classifier not trained — run `python -m agent.train_classifier`")
    from agent.policies import CauseAwareRetry

    pol = CauseAwareRetry()
    outs = run_batch(cases, truth, pol, CFG, run_seed=21)
    esc = [o for o in outs if o.escalated]
    assert esc, "expected some mandates to be escalated"
    # an escalated case spends no debit retries — it went straight to reauth + human
    assert all(o.attempts_used == 0 for o in esc)
    # and it should beat fixed_schedule on recovered money over the batch
    fs = sum(o.amount_recovered for o in run_batch(cases, truth, FixedSchedule(), CFG, 21))
    assert sum(o.amount_recovered for o in outs) > fs
