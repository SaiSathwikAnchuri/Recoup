"""Phase 9 — oracle, ablations, sensitivity, fairness."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from agent.classifier import CauseClassifier
from agent.liquidity import LiquidityModel

pytestmark = pytest.mark.skipif(
    not (CauseClassifier.default_exists() and LiquidityModel.default_exists()),
    reason="models not trained",
)

PRIORS = yaml.safe_load(Path("config/priors.yaml").read_text())


@pytest.fixture(scope="module")
def batch():
    cases = [json.loads(x) for x in open("data/cases.jsonl")]
    truth = {t["case_id"]: t for t in (json.loads(x) for x in open("data/truth.jsonl"))}
    return cases, truth


@pytest.fixture(scope="module")
def bits(batch):
    from agent.costs import CostModel
    from harness.engine import HarnessConfig
    cases, truth = batch
    return cases, truth, HarnessConfig(rev_cfg=PRIORS["revocation"]), CostModel.from_yaml()


# -- oracle -----------------------------------------------------------
def test_oracle_is_the_recovery_ceiling(bits):
    from harness.engine import run_batch
    from harness.oracle import Oracle
    from agent.ev_policy import EVPolicy
    cases, truth, cfg, _ = bits
    orc = sum(o.recovered for o in run_batch(cases, truth, Oracle(truth), cfg, 42))
    rec = sum(o.recovered for o in run_batch(cases, truth, EVPolicy(), cfg, 42))
    assert orc >= rec                       # nobody out-recovers perfect timing
    assert orc / len(cases) > 0.72


# -- ablation ladder ------------------------------------------------
@pytest.fixture(scope="module")
def ladder(bits):
    from experiments.phase9 import ablation_ladder
    cases, truth, cfg, costs = bits
    return ablation_ladder(cases, truth, cfg, costs, 42)


def test_every_capability_earns_its_place(ladder):
    t = ladder["table"]
    # removing the classifier or the timing model must hurt net value vs recoup
    assert t["no_cause"]["delta_vs_fixed_per_case"] < t["recoup"]["delta_vs_fixed_per_case"]
    assert t["no_timing"]["delta_vs_fixed_per_case"] < t["recoup"]["delta_vs_fixed_per_case"]
    # recoup closes most of the recovery gap to the oracle
    assert t["recoup"]["pct_of_recovery_gap_closed"] > 80
    # and it beats every non-oracle policy on net value
    for name in ("never_act", "fixed_schedule", "no_cause", "no_timing", "liquidity_aware"):
        assert t["recoup"]["net_value"] >= t[name]["net_value"]


def test_recoup_matches_or_beats_the_recovery_oracle_on_net_value(ladder):
    # the oracle maximises recovery, not net value; recoup should not trail it
    assert ladder["recoup_net_vs_oracle_net"] >= -1.0


# -- sensitivity ---------------------------------------------------
def test_ordering_holds_under_perturbed_priors(bits):
    from experiments.phase9 import sensitivity
    cases, truth, cfg, costs = bits
    rows = sensitivity(PRIORS, n=300, seed=42, cfg=cfg, costs=costs)
    assert len(rows) >= 5
    # recoup's mean net gain is positive in every perturbed world ...
    assert all(r["delta_net_per_case"] > 0 for r in rows.values()), rows
    # ... and significant (CI excludes zero) in all but at most one borderline case
    signif = sum(r["ci95"][0] > 0 for r in rows.values())
    assert signif >= len(rows) - 1


# -- fairness ----------------------------------------------------
def test_no_income_group_is_left_behind(bits):
    from experiments.phase9 import fairness
    cases, truth, cfg, costs = bits
    f = fairness(cases, truth, cfg, costs, 42)
    assert set(f["by_income"]) == {"salaried", "gig", "business"}
    assert f["every_group_gains_vs_fixed"]
    assert f["every_group_gain_significant"]
    assert f["escalation_rate_disparity"] < 0.10      # no group disproportionately abandoned
