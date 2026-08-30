"""Calibrated cause classifier.

A multinomial logistic regression over observable features, wrapped in
`CalibratedClassifierCV` (isotonic) so the probabilities it reports mean what
they say — when it says 0.7, it is right about 70% of the time. The Phase 7
policy consumes `predict_proba_one`; `should_escalate` is the Phase 4 escalate
branch (stop automating a mandate that is probably dead).

Training labels come from `truth.jsonl`. That is not leakage: in production you
eventually observe whether funds arrived / the mandate was revoked, and you train
on that history. At inference `predict_proba_one(case)` sees only the observable
record.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np

from .features import FEATURE_NAMES, featurize, featurize_batch

CAUSES = ("insufficient_balance", "bank_downtime", "limit_breach", "mandate_dead")
_MODEL_PATH = Path(__file__).resolve().parent / "models" / "cause_clf.pkl"


class CauseClassifier:
    def __init__(self, model, dead_threshold: float = 0.5,
                 meta: dict | None = None):
        self._model = model                 # fitted sklearn estimator, classes_ == CAUSES order
        self.dead_threshold = float(dead_threshold)
        self.meta = meta or {}

    # -- inference --------------------------------------------------------
    def predict_proba(self, cases: list[dict]) -> np.ndarray:
        """(n, 4) calibrated distribution, columns in CAUSES order."""
        X = featurize_batch(cases)
        p = self._model.predict_proba(X)
        return self._reorder(p)

    def predict_proba_one(self, case: dict) -> dict[str, float]:
        p = self._model.predict_proba(featurize(case).reshape(1, -1))
        p = self._reorder(p)[0]
        return {c: float(p[i]) for i, c in enumerate(CAUSES)}

    def should_escalate(self, probs: dict[str, float]) -> bool:
        """The escalate branch: the mandate is probably dead — automated debit
        retries are wasted; hand off after at most one re-auth attempt."""
        return probs["mandate_dead"] >= self.dead_threshold

    def _reorder(self, p: np.ndarray) -> np.ndarray:
        classes = list(self._model.classes_)
        idx = [classes.index(c) for c in CAUSES]
        return p[:, idx]

    # -- persistence ----------------------------------------------------
    def save(self, path: str | Path | None = None) -> Path:
        path = Path(path or _MODEL_PATH)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as f:
            pickle.dump({"model": self._model, "dead_threshold": self.dead_threshold,
                         "meta": self.meta, "feature_names": list(FEATURE_NAMES)}, f)
        return path

    @classmethod
    def load(cls, path: str | Path | None = None) -> "CauseClassifier":
        path = Path(path or _MODEL_PATH)
        if not path.exists():
            raise FileNotFoundError(
                f"no trained model at {path} — run `python -m agent.train_classifier` first")
        with path.open("rb") as f:
            d = pickle.load(f)
        if d.get("feature_names") != list(FEATURE_NAMES):
            raise ValueError("feature set changed since this model was trained — retrain it")
        return cls(d["model"], d["dead_threshold"], d.get("meta"))

    @classmethod
    def default_exists(cls) -> bool:
        return _MODEL_PATH.exists()


def train(cases: list[dict], labels: list[str], *, seed: int = 0, calibrate: bool = True):
    """Fit the pipeline. Returns a fitted sklearn estimator whose `classes_` are
    a subset/permutation of CAUSES. Kept module-level so tests can build a small
    model without the CLI."""
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    X = featurize_batch(cases)
    y = np.asarray(labels)
    base = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced",
                           random_state=seed),
    )
    if not calibrate:
        return base.fit(X, y)
    clf = CalibratedClassifierCV(base, method="isotonic", cv=5)
    return clf.fit(X, y)
