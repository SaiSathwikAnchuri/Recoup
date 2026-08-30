"""Phase 6 — the structural constraint filter.

Property tests: whatever a policy throws at `enforce`, the plan that comes out is
legal, and running the harness end to end no policy ever exceeds the caps.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta

import numpy as np
import pytest
import yaml

from baselines import ALL as BASELINES
from harness.constraints import Constraints, enforce
from harness.engine import HarnessConfig, run_batch
from harness.plan import MESSAGE_KINDS, Plan, ScheduledAction

C = Constraints.from_yaml()
OBS = datetime.fromisoformat("2026-03-10T09:30:00+05:30")
HZ_END = OBS + timedelta(days=45)
KINDS = ("retry", "nudge", "sms", "reauth")


def _random_plan(rng: np.random.Generator, n: int) -> Plan:
    acts = []
    for _ in range(n):
        day = int(rng.integers(-2, 50))
        hour = int(rng.integers(0, 24))
        kind = KINDS[int(rng.integers(0, len(KINDS)))]
        acts.append(ScheduledAction(OBS + timedelta(days=day, hours=hour - OBS.hour), kind))
    return Plan(acts, terminal="stop")


def _assert_legal(plan: Plan):
    retries = [a.at for a in plan.actions if a.kind == "retry"]
    msgs = [a.at for a in plan.actions if a.kind in MESSAGE_KINDS]
    assert len(retries) <= C.max_retries
    assert len(msgs) <= C.max_messages
    for a in plan.actions:
        assert OBS <= a.at <= HZ_END
    for earlier, later in zip(sorted(retries), sorted(retries)[1:]):
        assert (later - earlier) >= timedelta(hours=C.retry_min_gap_hours)
    for earlier, later in zip(sorted(msgs), sorted(msgs)[1:]):
        assert (later - earlier) >= timedelta(hours=C.message_min_gap_hours)
    for m in msgs:
        assert not C.in_quiet_hours(m)


# -- property: enforce always yields a legal plan -------------------------
def test_enforce_fuzz_output_is_always_legal():
    rng = np.random.default_rng(0)
    for _ in range(400):
        raw = _random_plan(rng, int(rng.integers(0, 12)))
        legal, viol = enforce(raw, observed_at=OBS, horizon_end=HZ_END, c=C)
        _assert_legal(legal)
        assert len(legal.actions) + sum(v["resolution"] == "dropped" for v in viol) <= len(raw.actions)


def test_enforce_is_idempotent():
    rng = np.random.default_rng(1)
    for _ in range(200):
        raw = _random_plan(rng, int(rng.integers(0, 12)))
        once, _ = enforce(raw, observed_at=OBS, horizon_end=HZ_END, c=C)
        twice, viol2 = enforce(once, observed_at=OBS, horizon_end=HZ_END, c=C)
        assert [(a.at, a.kind) for a in once.actions] == [(a.at, a.kind) for a in twice.actions]
        assert viol2 == []


def test_terminal_mode_is_preserved():
    for term in ("stop", "escalate", "replan"):
        raw = Plan([ScheduledAction(OBS + timedelta(days=1), "retry")], terminal=term)
        legal, _ = enforce(raw, observed_at=OBS, horizon_end=HZ_END, c=C)
        assert legal.terminal == term


def test_already_legal_plan_passes_through_untouched():
    raw = Plan([
        ScheduledAction(OBS.replace(hour=11) + timedelta(days=1), "retry"),
        ScheduledAction(OBS.replace(hour=11) + timedelta(days=3), "sms"),
        ScheduledAction(OBS.replace(hour=11) + timedelta(days=5), "retry"),
    ], terminal="stop")
    legal, viol = enforce(raw, observed_at=OBS, horizon_end=HZ_END, c=C)
    assert viol == []
    assert [(a.at, a.kind) for a in legal.actions] == [(a.at, a.kind) for a in raw.actions]


# -- specific rules ---------------------------------------------------
def test_retry_cap_drops_the_excess():
    raw = Plan([ScheduledAction(OBS.replace(hour=11) + timedelta(days=2 * i), "retry")
                for i in range(1, 8)], terminal="stop")
    legal, viol = enforce(raw, observed_at=OBS, horizon_end=HZ_END, c=C)
    assert sum(a.kind == "retry" for a in legal.actions) == C.max_retries
    assert sum(v["rule"] == "retry_cap" for v in viol) == 7 - C.max_retries


def test_retry_min_gap_reschedules_forward():
    raw = Plan([
        ScheduledAction(OBS.replace(hour=11) + timedelta(days=1), "retry"),
        ScheduledAction(OBS.replace(hour=11) + timedelta(days=1, hours=2), "retry"),
    ], terminal="stop")
    legal, viol = enforce(raw, observed_at=OBS, horizon_end=HZ_END, c=C)
    ts = sorted(a.at for a in legal.actions if a.kind == "retry")
    assert (ts[1] - ts[0]) >= timedelta(hours=C.retry_min_gap_hours)
    assert any(v["rule"] == "retry_min_gap" and v["resolution"] == "moved" for v in viol)


def test_quiet_hours_message_is_moved_to_morning():
    raw = Plan([ScheduledAction(OBS.replace(hour=23) + timedelta(days=1), "sms")], terminal="stop")
    legal, viol = enforce(raw, observed_at=OBS, horizon_end=HZ_END, c=C)
    assert len(legal.actions) == 1
    assert not C.in_quiet_hours(legal.actions[0].at)
    assert legal.actions[0].at.hour == C.quiet_end
    assert viol[0]["rule"] == "quiet_hours"


def test_out_of_window_actions_are_dropped():
    raw = Plan([
        ScheduledAction(OBS - timedelta(days=1), "retry"),
        ScheduledAction(OBS + timedelta(days=99), "retry"),
    ], terminal="stop")
    legal, viol = enforce(raw, observed_at=OBS, horizon_end=HZ_END, c=C)
    assert legal.actions == []
    assert all(v["rule"] == "out_of_window" for v in viol)


# -- harness-level: no policy escapes the filter -----------------------
@pytest.fixture(scope="module")
def data():
    cases = [json.loads(x) for x in open("data/cases.jsonl")]
    truth = {t["case_id"]: t for t in (json.loads(x) for x in open("data/truth.jsonl"))}
    return cases, truth


def test_no_policy_exceeds_caps_through_the_harness(data):
    cases, truth = data
    cfg = HarnessConfig(rev_cfg=yaml.safe_load(open("config/priors.yaml"))["revocation"])
    policies = list(BASELINES)
    try:
        from agent.policies import load_all
        policies += load_all()
    except Exception:
        pass
    for pol in policies:
        outs = run_batch(cases, truth, pol, cfg, run_seed=42)
        assert all(o.attempts_used <= cfg.constraints.max_retries for o in outs), pol.name
        assert all(o.messages_sent <= cfg.constraints.max_messages for o in outs), pol.name


def test_a_greedy_policy_is_reined_in_by_the_filter(data):
    cases, truth = data
    cfg = HarnessConfig(rev_cfg=yaml.safe_load(open("config/priors.yaml"))["revocation"])

    class Greedy:
        name = "greedy"

        def plan(self, case, state):
            o = state.observed_at
            acts = [ScheduledAction(o.replace(hour=11) + timedelta(days=i), "retry")
                    for i in range(1, 15)]
            acts += [ScheduledAction(o.replace(hour=3) + timedelta(days=i), "sms")
                     for i in range(1, 15)]
            return Plan(acts, terminal="stop")

    outs = run_batch(cases, truth, Greedy(), cfg, run_seed=42)
    assert all(o.attempts_used <= cfg.constraints.max_retries for o in outs)
    assert all(o.messages_sent <= cfg.constraints.max_messages for o in outs)
    assert sum(o.blocked_actions for o in outs) > 0
