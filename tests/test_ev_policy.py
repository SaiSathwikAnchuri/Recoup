"""Phase 7 — the cost-aware EV policy (`recoup`)."""

from __future__ import annotations

import inspect
import json
from datetime import timedelta
from pathlib import Path

import pytest
import yaml

from agent.classifier import CauseClassifier
from agent.liquidity import LiquidityModel
from harness.engine import EngagementState, HarnessConfig, run_batch
from harness.plan import ACTION_KINDS

pytestmark = pytest.mark.skipif(
    not (CauseClassifier.default_exists() and LiquidityModel.default_exists()),
    reason="models not trained — run `python -m agent.train_classifier && python -m agent.train_liquidity`",
)

PRIORS = yaml.safe_load(Path("config/priors.yaml").read_text())
CFG = HarnessConfig(rev_cfg=PRIORS["revocation"])


@pytest.fixture(scope="module")
def policy():
    from agent.ev_policy import EVPolicy
    return EVPolicy()


@pytest.fixture(scope="module")
def data():
    cases = [json.loads(x) for x in open("data/cases.jsonl")]
    truth = {t["case_id"]: t for t in (json.loads(x) for x in open("data/truth.jsonl"))}
    return cases, truth


def _state(case):
    return EngagementState(observed_at=__import__("simulator.calendar_utils", fromlist=["parse_dt"])
                           .parse_dt(case["observed_at"]),
                           horizon_days=case["horizon_days"],
                           attempt_budget=CFG.attempt_budget, now=None)


# -- no leakage -----------------------------------------------------------
def test_policy_reads_no_hidden_state():
    import agent.ev_policy as m
    file_src = Path(m.__file__).read_text(encoding="utf-8")
    assert "simulator.response" not in file_src          # the hidden-dynamics module
    assert "truth_from_record" not in file_src
    # plan() is handed the observable case + engagement state, nothing else
    assert list(inspect.signature(m.EVPolicy.plan).parameters) == ["self", "case", "state"]
    body = inspect.getsource(m.EVPolicy.plan)
    assert 'case["mandate"]["amount"]' in body and 'truth' not in body


# -- plan shape ---------------------------------------------------------
def test_plans_are_well_formed(policy, data):
    cases, _ = data
    for c in cases[:120]:
        p = policy.plan(c, _state(c))
        assert p.terminal in ("stop", "escalate", "replan")
        assert p.note
        obs = _state(c).observed_at
        end = obs + timedelta(days=c["horizon_days"])
        for a in p.actions:
            assert a.kind in ACTION_KINDS
            assert obs <= a.at <= end
        # retries never exceed the budget
        assert sum(a.kind == "retry" for a in p.actions) <= CFG.attempt_budget


def test_restraint_vs_fixed_schedule(policy, data):
    """The restraint the whole project argues for: never retry a dead mandate,
    escalate instead; and far shorter action lists than the fixed calendar."""
    cases, truth = data
    plans = [policy.plan(c, _state(c)) for c in cases]
    escalated = [p for p in plans if p.terminal == "escalate"]
    assert escalated, "policy never escalates — it should refuse some cases"
    # fixed_schedule always emits 4 actions; recoup's mean must be well below that
    mean_actions = sum(len(p.actions) for p in plans) / len(plans)
    assert mean_actions < 3.0
    # a dead-mandate escalation carries at most one action (a single re-auth)
    assert all(len(p.actions) <= 1 for p in escalated)


def test_dead_mandates_go_to_reauth_or_escalation(policy, data):
    cases, truth = data
    for c in cases:
        if truth[c["case_id"]]["true_cause"] != "mandate_dead":
            continue
        p = policy.plan(c, _state(c))
        # never burn the retry budget on a mandate the classifier is sure is dead
        if p.actions:
            assert p.actions[0].kind == "reauth"
        else:
            assert p.terminal == "escalate"


# -- it beats the industry baseline ---------------------------------
def test_recoup_beats_fixed_schedule_on_net_value(policy, data):
    from baselines import FixedSchedule
    from harness.metrics import attach_net_value, paired_delta
    from agent.costs import CostModel

    cases, truth = data
    costs = CostModel.from_yaml()
    r = run_batch(cases, truth, policy, CFG, run_seed=42)
    f = run_batch(cases, truth, FixedSchedule(), CFG, run_seed=42)
    attach_net_value(r, costs)
    attach_net_value(f, costs)

    d = paired_delta(r, f, "net_value", seed=42)
    assert d["mean"] > 0 and d["ci95"][0] > 0        # significantly better net value
    assert sum(o.amount_recovered for o in r) > sum(o.amount_recovered for o in f)
    assert sum(o.mandate_preserved for o in r) >= sum(o.mandate_preserved for o in f)


def test_recoup_sends_almost_no_messages(policy, data):
    cases, truth = data
    r = run_batch(cases, truth, policy, CFG, run_seed=42)
    assert sum(o.messages_sent for o in r) / len(r) < 0.5      # < 0.5 msgs per case
