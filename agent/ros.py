"""Recovery Opportunity Score (ROS) — Recoup 2.0.

A richer decision layer *around* the Phase-7 expected-value math, not a
replacement for it. `EVPolicy` and its `EV(a)` are untouched and still drive the
`recoup` policy and every Phase 9 number. ROS keeps the same economics but

  * decomposes the score into named, separately-auditable terms, and
  * adds an explicit **retention probability** that leans on the observable
    `CustomerState` (churn risk from dunning pressure and failure history) when
    the running service has one.

    ROS(a) =  P(recovery) · recovery_value · timeliness · retention
            − action_cost
            − churn_risk_cost          (incremental revocation risk × LTV)
            − missed_cycle_cost

Every `_ev_*` / `_p_*` helper here is called straight off an `EVPolicy`
instance, so the two layers can never drift apart.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, asdict
from datetime import timedelta

from simulator.calendar_utils import add_months, ist, parse_dt

from .ev_policy import EVPolicy


@dataclass
class Candidate:
    action: str                     # retry | reauth | wait | escalate | stop
    scheduled_day: int | None
    p_success: float
    recovery_value: float
    timeliness: float               # multiplier in (0, 1]
    retention: float                # P(mandate survives this action)
    action_cost: float
    churn_risk_cost: float
    missed_cycle_cost: float
    score: float                    # the ROS
    ev: float                       # the legacy Phase-7 EV, for comparison
    confidence: float
    reason: str

    def as_dict(self) -> dict:
        d = asdict(self)
        return {k: (round(v, 2) if isinstance(v, float) else v) for k, v in d.items()}


class RecoveryOpportunity:
    """Scores every legal candidate action for one case. Reuses an `EVPolicy`
    for the cause posterior, the funding-window model and the cost beliefs."""

    def __init__(self, policy: EVPolicy | None = None):
        self.p = policy or EVPolicy()

    # ---- term helpers -------------------------------------------------
    def _timeliness(self, day: int, cycle_close_day: int) -> float:
        """Mild multiplier: a recovery loses value the longer past the billing
        date it lands. The discrete missed-cycle penalty carries the real teeth;
        this just orders two late options."""
        late = max(0, day - cycle_close_day)
        return round(max(0.80, math.exp(-late / 45.0)), 4)

    def _retention(self, kind: str, p_dead: float, churn_risk: float) -> float:
        costs = self.p.costs
        bump = (costs.message_revocation_bump(kind)
                if kind in ("reauth", "sms", "nudge") else costs.silent_retry_bump)
        # a re-auth on a likely-dead mandate risks little — there is little to lose
        exposed = 1.0 - p_dead if kind == "reauth" else 1.0
        return round(max(0.05, 1.0 - churn_risk - bump * exposed), 4)

    # ---- the scores -------------------------------------------------
    def score_candidates(self, case: dict, state, customer_state=None) -> list[Candidate]:
        p = self.p
        observed_at = state.observed_at
        hz = state.horizon_days
        amount = float(case["mandate"]["amount"])
        probs = p.clf.predict_proba_one(case)
        w = p.liq.predict_window(case)
        p_dead = probs["mandate_dead"]
        ltv = p.costs.ltv_estimate(amount)

        ny, nm = add_months(observed_at.year, observed_at.month, 1)
        cycle_close = ist(ny, nm, 1)
        close_day = (cycle_close - observed_at).days

        churn = float(getattr(customer_state, "churn_risk", 0.0) or 0.0)
        # classifier confidence = how peaked the posterior is
        conf = round(max(probs.values()), 3)

        attempts_left = state.attempt_budget - state.attempts_used
        past_retries = [parse_dt(e["at"]) for e in state.history
                        if e.get("action") == "retry" and "at" in e]
        min_day = 1
        not_funded_by = 0.0
        if past_retries:
            gap = (max(past_retries) - observed_at).total_seconds() / 86400.0
            not_funded_by, min_day = gap, int(math.ceil(gap + 1.0))

        cands: list[Candidate] = []

        # -- retry candidates, one per suggested funding day ------------
        if attempts_left > 0:
            days = p._retry_days(probs, w, observed_at, cycle_close, min_day, hz, attempts_left)
            for d in days:
                date_d = observed_at + timedelta(days=d)
                psucc = p._p_retry_success(probs, w, d, date_d, cycle_close, not_funded_by)
                val = p.costs.recovery_value(amount, days=d)
                tml = self._timeliness(d, close_day)
                ret = self._retention("retry", p_dead, churn)
                acost = p.costs.action_cost("retry")
                ccost = (1.0 - p_dead) * p.costs.silent_retry_bump * (1.0 - psucc) * ltv
                mcost = (p.costs.missed_cycle_penalty(amount) * psucc * (1.0 - p_dead)
                         if date_d > cycle_close else 0.0)
                score = psucc * val * tml * ret - acost - ccost - mcost
                ev = p._ev_retry(probs, w, d, observed_at, cycle_close, cycle_close,
                                 amount, not_funded_by)
                cands.append(Candidate(
                    "retry", d, round(psucc, 4), round(val, 2), tml, ret,
                    round(acost, 2), round(ccost, 2), round(mcost, 2),
                    round(score, 2), round(ev, 2), conf,
                    f"retry on day {d}: P(clear)={psucc:.0%}, "
                    f"{'after' if date_d > cycle_close else 'before'} the billing date"))

        # -- re-auth candidate -- only when the mandate is plausibly dead, same
        #    bar as the one-shot policy (EVParams.reauth_min_p_dead) -----------
        rd = max(1, min_day)
        if (rd <= hz and p_dead >= p.params.reauth_min_p_dead
                and not any(e.get("action") == "reauth" for e in state.history)):
            pr = p.costs.reauth_success_belief
            val = p.costs.recovery_value(amount, days=rd)
            ret = self._retention("reauth", p_dead, churn)
            acost = p.costs.action_cost("reauth")
            ccost = (1.0 - p_dead) * p.costs.message_revocation_bump("reauth") * ltv
            gain = pr * (val + p_dead * ltv) * ret       # dead mandate: re-auth also resurrects LTV
            score = gain - acost - ccost
            ev = p._ev_reauth(probs, rd, observed_at, amount)
            cands.append(Candidate(
                "reauth", rd, round(pr, 4), round(val, 2), 1.0, ret,
                round(acost, 2), round(ccost, 2), 0.0, round(score, 2), round(ev, 2), conf,
                f"re-auth link on day {rd}: P(complete)={pr:.0%}, "
                f"P(mandate dead)={p_dead:.0%}"))

        # -- wait / escalate / stop reference rows --------------------
        hold_cost = round(churn * ltv * 0.02, 2)         # a day's worth of standing churn exposure
        cands.append(Candidate("wait", None, 0.0, 0.0, 1.0, round(1.0 - churn, 4),
                               0.0, hold_cost, 0.0, round(-hold_cost, 2), 0.0, conf,
                               "hold for new information; small standing churn cost"))

        # escalation is a *fallback*, not a profit centre: a human recovers the
        # pending debit on a fraction of dead mandates automation cannot, and
        # that is all it is credited — no LTV "resurrection" bonus (the mandate
        # keeps its LTV whether or not a human touches it).
        esc_cost = p.costs.action_cost("human_escalation")
        esc_gain = p_dead * p.costs.reauth_success_belief * p.costs.recovery_value(amount, days=7) * 0.5
        cands.append(Candidate("escalate", 1, round(p.costs.reauth_success_belief, 3),
                               0.0, 1.0, 1.0, round(esc_cost, 2), 0.0, 0.0,
                               round(esc_gain - esc_cost, 2), round(esc_gain - esc_cost, 2),
                               conf, "hand to a human collections agent"))
        cands.append(Candidate("stop", None, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0,
                               0.0, 0.0, conf, "close the recovery — nothing worth doing"))

        cands.sort(key=lambda c: c.score, reverse=True)
        return cands

    def best(self, case: dict, state, customer_state=None) -> Candidate:
        return self.score_candidates(case, state, customer_state)[0]
