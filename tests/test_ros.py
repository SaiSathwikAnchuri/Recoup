"""Recoup 2.0 — the Recovery Opportunity Score."""

from __future__ import annotations

import inspect
import json
from pathlib import Path

import pytest

from agent.classifier import CauseClassifier
from agent.liquidity import LiquidityModel

pytestmark = pytest.mark.skipif(
    not (CauseClassifier.default_exists() and LiquidityModel.default_exists()),
    reason="models not trained")

from agent.ev_policy import EVPolicy          # noqa: E402
from agent.ros import RecoveryOpportunity     # noqa: E402
from harness.engine import EngagementState    # noqa: E402
from simulator.calendar_utils import parse_dt  # noqa: E402


@pytest.fixture(scope="module")
def data():
    cases = [json.loads(x) for x in open("data/cases.jsonl")]
    truth = {t["case_id"]: t for t in (json.loads(x) for x in open("data/truth.jsonl"))}
    return cases, truth


def _state(c):
    return EngagementState(observed_at=parse_dt(c["observed_at"]),
                           horizon_days=c["horizon_days"], attempt_budget=3)


def test_ros_wraps_ev_not_replaces_it():
    """Every economic term must be sourced from an EVPolicy instance."""
    src = inspect.getsource(RecoveryOpportunity)
    assert "self.p._" in src or "self.p.costs" in src
    assert "simulator.response" not in inspect.getsource(RecoveryOpportunity.score_candidates)


def test_candidates_are_well_formed(data):
    cases, _ = data
    ro = RecoveryOpportunity(EVPolicy())
    for c in cases[:60]:
        cands = ro.score_candidates(c, _state(c))
        assert cands and cands == sorted(cands, key=lambda x: x.score, reverse=True)
        kinds = {x.action for x in cands}
        assert {"stop", "wait", "escalate"} <= kinds
        for x in cands:
            assert 0.0 <= x.retention <= 1.0
            assert 0.0 <= x.p_success <= 1.0
            d = x.as_dict()
            assert set(d) >= {"action", "score", "ev", "reason"}


def test_dead_mandates_never_top_rank_a_retry(data):
    cases, truth = data
    ro = RecoveryOpportunity(EVPolicy())
    seen = 0
    for c in cases:
        if truth[c["case_id"]]["true_cause"] != "mandate_dead":
            continue
        probs = ro.p.clf.predict_proba_one(c)
        if probs["mandate_dead"] < 0.5:
            continue
        seen += 1
        best = ro.best(c, _state(c))
        assert best.action in ("reauth", "escalate", "stop", "wait"), c["case_id"]
    assert seen > 0


def test_churn_pressure_lowers_retention(data):
    cases, _ = data
    ro = RecoveryOpportunity(EVPolicy())

    class _CS:
        churn_risk = 0.6
    c = cases[0]
    calm = [x for x in ro.score_candidates(c, _state(c)) if x.action == "retry"]
    hot = [x for x in ro.score_candidates(c, _state(c), _CS()) if x.action == "retry"]
    if calm and hot:
        assert hot[0].retention < calm[0].retention
