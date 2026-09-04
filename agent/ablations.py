"""Ablations of `recoup` — each removes one capability so Phase 9 can price it.

  no_cause    the EV policy with NO classifier: every case is scored against the
              population cause mix (config/priors.yaml -> cause_mix). It can't
              tell a dead mandate from a cash-flow one.
  no_timing   the EV policy with NO funding-window model: every case gets a
              fixed, generic window guess. It keeps the cost logic and the
              escalate branch but retries on a calendar.

Both reuse `EVPolicy` unchanged — only the belief inputs are swapped for stubs,
which is the cleanest possible ablation.
"""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path

import yaml

from simulator.calendar_utils import parse_dt

from .classifier import CAUSES
from .ev_policy import EVPolicy

_PRIORS = yaml.safe_load((Path(__file__).resolve().parent.parent
                          / "config" / "priors.yaml").read_text())


class _ConstantClassifier:
    """Returns the population cause mix for every case."""
    def __init__(self, mix: dict[str, float]):
        s = sum(mix.get(c, 0.0) for c in CAUSES)
        self._p = {c: mix.get(c, 0.0) / s for c in CAUSES}
        self.dead_threshold = 0.5

    def predict_proba_one(self, case: dict) -> dict[str, float]:
        return dict(self._p)

    def should_escalate(self, probs) -> bool:
        return probs["mandate_dead"] >= self.dead_threshold


class _ConstantLiquidity:
    """Returns a fixed funding-window guess for every case."""
    def __init__(self, p50_days: float = 8.0, p85_days: float = 18.0):
        self.p50, self.p85 = p50_days, p85_days

    def predict_window(self, case: dict) -> dict:
        obs = parse_dt(case["observed_at"])
        return {
            "days_p50": self.p50, "days_p85": self.p85,
            "date_p50": obs + timedelta(days=self.p50),
            "date_p85": obs + timedelta(days=self.p85),
            "within_horizon": self.p85 < case.get("horizon_days", 45),
        }


def no_cause_policy() -> EVPolicy:
    p = EVPolicy(clf=_ConstantClassifier(_PRIORS["cause_mix"]))
    p.name = "no_cause"
    return p


def no_timing_policy() -> EVPolicy:
    p = EVPolicy(liq=_ConstantLiquidity())
    p.name = "no_timing"
    return p
