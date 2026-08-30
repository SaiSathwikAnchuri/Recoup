"""Cost model — the agent's *beliefs* about what each action costs and what a
live mandate is worth. Loaded from `config/costs.yaml`.

These numbers are deliberately NOT the simulator's hidden parameters: the Phase 7
policy must decide on estimates, the way a real deployment would. This module
imports nothing from `simulator/` — the no-leakage boundary (report R3).

    cm = CostModel.from_yaml()
    cm.action_cost("sms")                 # -> 0.30
    cm.ltv_estimate(3000)                 # believed value of keeping this mandate
    cm.recovery_value(3000, days=18)      # face amount, delay-discounted
    cm.missed_cycle_penalty(3000)
    cm.message_fatigue_factor(3)          # multiplier on the 3rd message in-window
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

_DEFAULT_PATH = Path(__file__).resolve().parent.parent / "config" / "costs.yaml"


@dataclass(frozen=True)
class CostModel:
    action: dict                       # action kind -> direct rupee cost
    value_per_recovery_frac: float
    delay_discount_per_day: float
    missed_cycle_penalty_frac: float
    ltv_months: int
    ltv_monthly_value_frac: float
    ltv_survival_haircut: float
    fatigue_window_days: int
    fatigue_multiplier: float
    rev_message_bump: dict          # message kind -> believed integrated revocation-prob bump
    silent_retry_bump: float
    reauth_success_belief: float

    # -- action side --------------------------------------------------------
    def action_cost(self, kind: str) -> float:
        try:
            return float(self.action[kind])
        except KeyError:
            raise KeyError(f"no cost defined for action kind {kind!r}") from None

    # -- benefit side -----------------------------------------------------
    def recovery_value(self, amount: float, days: float = 0.0) -> float:
        """Face amount of a recovery, discounted for how long it took."""
        v = amount * self.value_per_recovery_frac
        if days > 0:
            v *= (1.0 - self.delay_discount_per_day) ** days
        return v

    def missed_cycle_penalty(self, amount: float) -> float:
        """Charged when recovery lands after the next billing date."""
        return amount * self.missed_cycle_penalty_frac

    def ltv_estimate(self, amount: float) -> float:
        """Believed value of keeping this mandate alive — the multiplier on
        P(revocation) in the EV of any action."""
        return (amount * self.ltv_monthly_value_frac
                * self.ltv_months * self.ltv_survival_haircut)

    # -- messaging fatigue ----------------------------------------------
    def message_fatigue_factor(self, k_in_window: int) -> float:
        """Multiplier on the believed annoyance cost of the k-th message
        (1-based) sent inside `fatigue_window_days`."""
        return self.fatigue_multiplier ** max(0, k_in_window - 1)

    # -- revocation risk -----------------------------------------------
    def message_revocation_bump(self, kind: str) -> float:
        """Believed increase in P(revocation) from sending one message of `kind`."""
        try:
            return float(self.rev_message_bump[kind])
        except KeyError:
            raise KeyError(f"no revocation bump for message kind {kind!r}") from None

    def revocation_cost(self, kind: str, amount: float) -> float:
        """Rupee cost of the revocation risk one message adds to a *live* mandate."""
        return self.message_revocation_bump(kind) * self.ltv_estimate(amount)

    # -- loader ----------------------------------------------------------
    @classmethod
    def from_yaml(cls, path: str | Path | None = None) -> "CostModel":
        cfg = yaml.safe_load(Path(path or _DEFAULT_PATH).read_text(encoding="utf-8"))
        ltv = cfg["ltv"]
        fatigue = cfg["message_fatigue"]
        rev = cfg["revocation_belief"]
        return cls(
            action=dict(cfg["action_cost"]),
            value_per_recovery_frac=float(cfg["value_per_recovery_frac"]),
            delay_discount_per_day=float(cfg["delay_discount_per_day"]),
            missed_cycle_penalty_frac=float(cfg["missed_cycle_penalty_frac"]),
            ltv_months=int(ltv["months"]),
            ltv_monthly_value_frac=float(ltv["monthly_value_frac"]),
            ltv_survival_haircut=float(ltv["survival_haircut"]),
            fatigue_window_days=int(fatigue["window_days"]),
            fatigue_multiplier=float(fatigue["multiplier"]),
            rev_message_bump=dict(rev["message_bump"]),
            silent_retry_bump=float(rev["silent_retry_failure_bump"]),
            reauth_success_belief=float(rev["reauth_success_belief"]),
        )
