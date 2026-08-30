"""Phase 4 — cause classifier + calibration + escalate branch."""

from pathlib import Path

import numpy as np
import pytest
import yaml

from agent.calibration import brier_multiclass, reliability
from agent.classifier import CAUSES, CauseClassifier, train
from agent.features import FEATURE_NAMES, featurize
from simulator.generate import build_batch

PRIORS = yaml.safe_load(Path("config/priors.yaml").read_text())


@pytest.fixture(scope="module")
def model():
    cases, truths = build_batch(3000, seed=2024, priors=PRIORS)
    y = [t["true_cause"] for t in truths]
    cut = 2100
    est = train(cases[:cut], y[:cut], seed=1, calibrate=True)
    raw = train(cases[:cut], y[:cut], seed=1, calibrate=False)
    te_cases, te_y = cases[cut:], np.array(y[cut:])
    clf = CauseClassifier(est, dead_threshold=0.5)
    return clf, raw, te_cases, te_y


# -- features --------------------------------------------------------------
def test_featurize_uses_only_observable_fields():
    import inspect

    import agent.features as feat

    body = inspect.getsource(feat.featurize) + inspect.getsource(feat.featurize_batch)
    assert "truth" not in body and "true_cause" not in body
    assert "import simulator" not in Path(feat.__file__).read_text(encoding="utf-8")
    assert list(inspect.signature(feat.featurize).parameters) == ["case"]


def test_feature_vector_shape_and_finite():
    cases, _ = build_batch(20, seed=5, priors=PRIORS)
    for c in cases:
        v = featurize(c)
        assert v.shape == (len(FEATURE_NAMES),)
        assert np.isfinite(v).all()


# -- accuracy / calibration ---------------------------------------------
def test_beats_the_prior_baseline_by_a_wide_margin(model):
    clf, _, te_cases, te_y = model
    pred = np.array([CAUSES[i] for i in clf.predict_proba(te_cases).argmax(axis=1)])
    acc = (pred == te_y).mean()
    assert acc > max(PRIORS["cause_mix"].values()) + 0.15   # > 0.75


def test_probabilities_are_a_distribution(model):
    clf, _, te_cases, _ = model
    p = clf.predict_proba(te_cases)
    assert p.shape[1] == 4
    assert np.allclose(p.sum(axis=1), 1.0, atol=1e-6)
    assert (p >= 0).all()


def test_calibration_error_is_small_and_calibration_helps(model):
    clf, raw, te_cases, te_y = model
    yi = np.array([CAUSES.index(c) for c in te_y])
    order = [list(raw.classes_).index(c) for c in CAUSES]
    from agent.features import featurize_batch
    ece_cal = reliability(clf.predict_proba(te_cases), yi)["ece"]
    ece_raw = reliability(raw.predict_proba(featurize_batch(te_cases))[:, order], yi)["ece"]
    assert ece_cal < 0.10
    assert ece_cal <= ece_raw + 0.02


def test_dead_mandates_are_caught(model):
    clf, _, te_cases, te_y = model
    p_dead = clf.predict_proba(te_cases)[:, CAUSES.index("mandate_dead")]
    is_dead = (te_y == "mandate_dead")
    # ER_mandate_* codes are fairly diagnostic — recall at a low bar should be high
    assert (p_dead[is_dead] >= 0.5).mean() > 0.75


def test_escalate_branch_is_precise(model):
    clf, _, te_cases, te_y = model
    clf.dead_threshold = 0.5
    flagged = np.array([clf.should_escalate(clf.predict_proba_one(c)) for c in te_cases])
    is_dead = (te_y == "mandate_dead")
    assert flagged.sum() > 0
    assert (flagged & is_dead).sum() / flagged.sum() >= 0.80    # precision


def test_brier_beats_predicting_the_prior(model):
    clf, _, te_cases, te_y = model
    yi = np.array([CAUSES.index(c) for c in te_y])
    prior = np.array([PRIORS["cause_mix"][c] for c in CAUSES])
    b_model = brier_multiclass(clf.predict_proba(te_cases), yi, 4)
    b_prior = brier_multiclass(np.tile(prior, (len(yi), 1)), yi, 4)
    assert b_model < b_prior


# -- persistence -------------------------------------------------------
def test_save_load_roundtrip(model, tmp_path):
    clf, _, te_cases, _ = model
    p = tmp_path / "m.pkl"
    clf.save(p)
    back = CauseClassifier.load(p)
    a = clf.predict_proba(te_cases[:10])
    b = back.predict_proba(te_cases[:10])
    assert np.allclose(a, b)
    assert back.dead_threshold == clf.dead_threshold


# -- calibration helper sanity ---------------------------------------
def test_reliability_perfect_on_calibrated_synthetic():
    rng = np.random.default_rng(0)
    conf = rng.uniform(0.5, 1.0, size=20000)
    p = np.column_stack([conf, 1 - conf, np.zeros(20000), np.zeros(20000)])
    y = (rng.random(20000) > conf).astype(int)   # class 1 with prob (1-conf)
    assert reliability(p, y)["ece"] < 0.02
