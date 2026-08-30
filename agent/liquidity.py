"""Liquidity-window model.

For a case that looks like a cash-flow shortfall, predict *when* the account
will hold enough to clear the debit — as two quantiles of "days from now":

    p50   the median expectation
    p85   a safety quantile — retry here and ~85% of the time the money is there

The Phase 7 policy schedules retries against these instead of a fixed calendar.
Trained on `insufficient_balance` cases that were recoverable within the horizon;
target is days from `observed_at` to the hidden `best_retry_at`. Inference reads
only the observable case (the funding-day history the merchant already has).
"""

from __future__ import annotations

import pickle
import statistics
from datetime import timedelta
from pathlib import Path

import numpy as np

from simulator.calendar_utils import parse_dt

_MODEL_PATH = Path(__file__).resolve().parent / "models" / "liquidity.pkl"
_MONTH = 30.0

LIQ_FEATURES: tuple[str, ...] = (
    "observed_dom", "debit_day", "modal_funding_day", "mean_funding_day",
    "last_funding_day", "min_funding_day", "max_funding_day", "funding_day_std",
    "n_funding_hits", "funding_hit_rate", "days_since_last_success",
    "days_to_next_modal", "log_amount", "consecutive_successes", "cycles_observed",
    "cat::ott", "cat::sip", "cat::emi", "cat::insurance", "cat::utility",
)
_CATS = ("ott", "sip", "emi", "insurance", "utility")


def _modal(xs: list[int], fallback: int) -> float:
    if not xs:
        return float(fallback)
    try:
        return float(statistics.multimode(xs)[0])
    except statistics.StatisticsError:
        return float(round(statistics.mean(xs)))


def liquidity_features(case: dict) -> np.ndarray:
    m, h = case["mandate"], case["history"]
    observed = parse_dt(case["observed_at"])
    hits = [int(d) for d in (h.get("success_days_of_month") or [])]
    debit_day = int(m.get("debit_day", 1))
    cycles = int(h.get("cycles_observed", 0))
    dsls = h.get("days_since_last_success")
    dsls = float(dsls) if dsls is not None else 999.0

    modal = _modal(hits, debit_day)
    obs_dom = float(observed.day)
    # days until the modal funding day comes round again
    ahead = (modal - obs_dom) % _MONTH
    days_to_next_modal = ahead if ahead > 0 else _MONTH

    row = [
        obs_dom, float(debit_day), modal,
        float(statistics.mean(hits)) if hits else float(debit_day),
        float(hits[-1]) if hits else float(debit_day),
        float(min(hits)) if hits else float(debit_day),
        float(max(hits)) if hits else float(debit_day),
        float(statistics.pstdev(hits)) if len(hits) >= 2 else 0.0,
        float(len(hits)),
        len(hits) / cycles if cycles else 0.0,
        min(dsls, 400.0),
        float(days_to_next_modal),
        float(np.log1p(max(m.get("amount", 0.0), 0.0))),
        float(h.get("consecutive_successes", 0)),
        float(cycles),
        *(1.0 if m.get("category") == c else 0.0 for c in _CATS),
    ]
    return np.asarray(row, dtype=float)


def _feat_batch(cases: list[dict]) -> np.ndarray:
    return (np.vstack([liquidity_features(c) for c in cases]) if cases
            else np.empty((0, len(LIQ_FEATURES))))


class LiquidityModel:
    def __init__(self, m50, m85, horizon_days: int = 45, meta: dict | None = None):
        self._m50 = m50
        self._m85 = m85
        self.horizon_days = int(horizon_days)
        self.meta = meta or {}

    def predict_window(self, case: dict) -> dict:
        x = liquidity_features(case).reshape(1, -1)
        d50 = float(max(0.0, self._m50.predict(x)[0]))
        d85 = float(max(d50, self._m85.predict(x)[0]))
        hz = case.get("horizon_days", self.horizon_days)
        d50, d85 = min(d50, hz), min(d85, hz)
        observed = parse_dt(case["observed_at"])
        return {
            "days_p50": round(d50, 2),
            "days_p85": round(d85, 2),
            "date_p50": observed + timedelta(days=d50),
            "date_p85": observed + timedelta(days=d85),
            "within_horizon": d85 < hz,
        }

    def save(self, path: str | Path | None = None) -> Path:
        path = Path(path or _MODEL_PATH)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as f:
            pickle.dump({"m50": self._m50, "m85": self._m85,
                         "horizon_days": self.horizon_days, "meta": self.meta,
                         "features": list(LIQ_FEATURES)}, f)
        return path

    @classmethod
    def load(cls, path: str | Path | None = None) -> "LiquidityModel":
        path = Path(path or _MODEL_PATH)
        if not path.exists():
            raise FileNotFoundError(
                f"no liquidity model at {path} — run `python -m agent.train_liquidity` first")
        with path.open("rb") as f:
            d = pickle.load(f)
        if d.get("features") != list(LIQ_FEATURES):
            raise ValueError("liquidity feature set changed — retrain")
        return cls(d["m50"], d["m85"], d.get("horizon_days", 45), d.get("meta"))

    @classmethod
    def default_exists(cls) -> bool:
        return _MODEL_PATH.exists()


def train_liquidity(cases: list[dict], days_to_funding: list[float], *, seed: int = 0):
    """Fit the two quantile regressors. Module-level so tests can build a small one."""
    from sklearn.ensemble import GradientBoostingRegressor

    X = _feat_batch(cases)
    y = np.asarray(days_to_funding, dtype=float)
    common = dict(n_estimators=300, max_depth=3, learning_rate=0.05,
                  subsample=0.9, random_state=seed)
    m50 = GradientBoostingRegressor(loss="quantile", alpha=0.50, **common).fit(X, y)
    m85 = GradientBoostingRegressor(loss="quantile", alpha=0.85, **common).fit(X, y)
    return m50, m85
