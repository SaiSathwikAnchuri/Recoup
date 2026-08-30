"""Phase 5 — liquidity-window model."""

from pathlib import Path

import numpy as np
import pytest
import yaml

from agent.liquidity import LIQ_FEATURES, LiquidityModel, liquidity_features, train_liquidity
from simulator.calendar_utils import parse_dt
from simulator.generate import build_batch

PRIORS = yaml.safe_load(Path("config/priors.yaml").read_text())
HORIZON = int(PRIORS["horizon_days"])


def _days(case, truth):
    return (parse_dt(truth["best_retry_at"]) - parse_dt(case["observed_at"])).total_seconds() / 86400.0


def _ib(cases, truths):
    cs, ys = [], []
    for c, t in zip(cases, truths):
        if t["true_cause"] == "insufficient_balance" and t["retry_recoverable_within_horizon"]:
            cs.append(c)
            ys.append(_days(c, t))
    return cs, np.array(ys)


@pytest.fixture(scope="module")
def model(cached_batch):
    cases, truths = cached_batch(2200, 4321)
    cs, ys = _ib(cases, truths)
    cut = int(len(cs) * 0.7)
    m50, m85 = train_liquidity(cs[:cut], ys[:cut], seed=1)
    mdl = LiquidityModel(m50, m85, horizon_days=HORIZON)
    return mdl, cs[cut:], ys[cut:]


def test_features_are_observable_only():
    import inspect

    import agent.liquidity as liq
    body = inspect.getsource(liq.liquidity_features)
    assert "truth" not in body and "true_cause" not in body and "best_retry" not in body


def test_feature_vector_shape_and_finite():
    cases, _ = build_batch(20, seed=9, priors=PRIORS)
    for c in cases:
        v = liquidity_features(c)
        assert v.shape == (len(LIQ_FEATURES),)
        assert np.isfinite(v).all()


def test_predict_window_is_ordered_and_in_horizon(model):
    mdl, cs, _ = model
    for c in cs[:200]:
        w = mdl.predict_window(c)
        assert 0.0 <= w["days_p50"] <= w["days_p85"] <= c["horizon_days"]
        assert w["date_p85"] >= w["date_p50"] >= parse_dt(c["observed_at"])


def test_beats_naive_modal_day_on_mae(model):
    mdl, cs, ys = model
    d50 = np.array([mdl.predict_window(c)["days_p50"] for c in cs])
    idx = {n: i for i, n in enumerate(LIQ_FEATURES)}["days_to_next_modal"]
    naive = np.array([liquidity_features(c)[idx] for c in cs])
    assert np.mean(np.abs(d50 - ys)) < 0.75 * np.mean(np.abs(naive - ys))


def test_p85_quantile_is_roughly_calibrated(model):
    mdl, cs, ys = model
    d85 = np.array([mdl.predict_window(c)["days_p85"] for c in cs])
    coverage = np.mean(ys <= d85)
    assert 0.72 <= coverage <= 0.95      # aiming for ~0.85, allow slack


def test_save_load_roundtrip(model, tmp_path):
    mdl, cs, _ = model
    p = tmp_path / "liq.pkl"
    mdl.save(p)
    back = LiquidityModel.load(p)
    for c in cs[:20]:
        assert back.predict_window(c)["days_p50"] == pytest.approx(
            mdl.predict_window(c)["days_p50"])
