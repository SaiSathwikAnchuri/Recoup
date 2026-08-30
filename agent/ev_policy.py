"""Recoup — the cost-aware expected-value policy (Phase 7).

Assembles everything the earlier phases built into ONE decision. For each
candidate action it estimates, from the observable case only:

    EV(a) = SUM_c  P(cause=c) * [ P(success | a, c, t) * value(a, c)
                                  - action_cost(a)
                                  - risk(a, c) ]
            - missed_cycle_penalty * P(recovered, but after the billing date)

  * cause posterior     -> agent/classifier.py  (calibrated)
  * P(success) vs time  -> agent/liquidity.py for the funding window, plus
                           cause-specific curves for downtime / limit / dead
  * costs & LTV & risk  -> agent/costs.py       (beliefs, not the sim's truth)
  * legality            -> harness/constraints.py filters the plan anyway

It decides between three moves — line up retries across the predicted funding
window, send a re-auth request, or DO NOTHING — and picks whichever has the
highest expected value. When none clears zero it stops (or escalates a probably
-dead mandate). That stop is the point of the whole project: fewer, better-timed
actions, and none at all when none is worth it.

The plan is committed in one shot (`terminal="stop"`). A receding horizon was
tried; in this environment it only keeps the mandate exposed to the revocation
hazard longer for no information gain, so the policy commits and exits.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import timedelta

from simulator.calendar_utils import add_months, ist, parse_dt

from .classifier import CauseClassifier
from .costs import CostModel
from .liquidity import LiquidityModel

# harness/plan.py holds only the dataclasses — importing it does not pull the
# engine (which would be a cycle) or anything hidden.
from harness.plan import Plan, ScheduledAction  # noqa: E402

@dataclass
class EVParams:
    day_grid_step: int = 1
    max_exec_prob: float = 0.93        # generic tech-failure ceiling (matches priors.max_success_prob)
    downtime_p_day0: float = 0.05
    downtime_p_recovered: float = 0.80
    limit_p_before_reset: float = 0.02
    limit_p_after_reset: float = 0.90
    min_ev_to_act: float = 0.0            # absolute EV floor
    min_ev_frac_of_amount: float = 0.05   # ... or this fraction of the debit, whichever is larger:
                                          #     a recovery worth <5% of the debit in expectation is not
                                          #     worth the operational bother — do nothing / hand off.
    ib_sigma_min: float = 2.5          # narrowest funding-window bump (very confident prediction)
    ib_sigma_max: float = 12.0
    reauth_min_p_dead: float = 0.30    # re-auth only when the mandate is plausibly dead (matches the classifier escalate bar)


@dataclass
class EVPolicy:
    name: str = "recoup"
    clf: CauseClassifier = field(default_factory=CauseClassifier.load)
    liq: LiquidityModel = field(default_factory=LiquidityModel.load)
    costs: CostModel = field(default_factory=CostModel.from_yaml)
    params: EVParams = field(default_factory=EVParams)

    # ---- P(a retry on day d clears) for an insufficient-balance case -------
    # NOT a CDF: an account is funded briefly at pay-day and then spent down, so
    # P(success) PEAKS around the predicted funding day and decays either side.
    # Retrying long after the window is worse, not better.
    def _ib_success(self, w: dict, d: float, not_funded_by: float = 0.0) -> float:
        mu = w["days_p50"]
        if not_funded_by > 0.0:                        # a retry already failed by then
            mu = max(mu, not_funded_by + 2.0)
        # asymmetric: an account is definitely not funded *before* pay-day, but
        # once funded it stays usable for a few days, so decay slower afterwards.
        spread = min(max((w["days_p85"] - w["days_p50"]) / 1.5, self.params.ib_sigma_min),
                     self.params.ib_sigma_max)
        sigma = spread if d >= mu else spread * 0.7
        bump = math.exp(-0.5 * ((d - mu) / sigma) ** 2)
        return self.params.max_exec_prob * bump

    def _p_retry_success(self, probs, w, d, date_d, reset_dt, not_funded_by) -> float:
        p_ib = self._ib_success(w, d, not_funded_by)
        p_dt = self.params.downtime_p_day0 if d < 1 else self.params.downtime_p_recovered
        p_lim = (self.params.limit_p_after_reset if date_d >= reset_dt
                 else self.params.limit_p_before_reset)
        p = (probs["insufficient_balance"] * p_ib
             + probs["bank_downtime"] * p_dt
             + probs["limit_breach"] * p_lim
             + probs["mandate_dead"] * 0.0)
        return min(p, self.params.max_exec_prob)

    # ---- EV of each candidate -------------------------------------------
    def _ev_retry(self, probs, w, d, observed_at, cycle_close, reset_dt, amount,
                  not_funded_by) -> float:
        date_d = observed_at + timedelta(days=d)
        psucc = self._p_retry_success(probs, w, d, date_d, reset_dt, not_funded_by)
        value = self.costs.recovery_value(amount, days=d)
        live = 1.0 - probs["mandate_dead"]
        cost = self.costs.action_cost("retry")
        risk = live * self.costs.silent_retry_bump * (1.0 - psucc) * self.costs.ltv_estimate(amount)
        missed = 0.0
        if date_d > cycle_close:
            missed = self.costs.missed_cycle_penalty(amount) * psucc * live
        return psucc * value - cost - risk - missed

    def _ev_reauth(self, probs, d, observed_at, amount) -> float:
        pr = self.costs.reauth_success_belief
        ltv = self.costs.ltv_estimate(amount)
        value_live = self.costs.recovery_value(amount, days=d)
        cost = self.costs.action_cost("reauth")
        bump = self.costs.message_revocation_bump("reauth")
        ev_dead = pr * (value_live + ltv) - cost                # dead mandate: reauth also resurrects it
        ev_live = pr * value_live - cost - bump * ltv           # live mandate: reauth risks killing it
        return probs["mandate_dead"] * ev_dead + (1.0 - probs["mandate_dead"]) * ev_live

    # ---- the decision ------------------------------------------------
    def plan(self, case: dict, state) -> Plan:
        observed_at = state.observed_at
        hz = state.horizon_days
        amount = float(case["mandate"]["amount"])
        probs = self.clf.predict_proba_one(case)
        w = self.liq.predict_window(case)

        ny, nm = add_months(observed_at.year, observed_at.month, 1)
        cycle_close = ist(ny, nm, 1)
        reset_dt = cycle_close                                  # AutoPay cap resets on the 1st

        past_retries = [parse_dt(e["at"]) for e in state.history
                        if e.get("action") == "retry" and "at" in e]
        not_funded_by = 0.0
        min_day = 1
        if past_retries:
            gap_days = (max(past_retries) - observed_at).total_seconds() / 86400.0
            not_funded_by = gap_days
            min_day = int(math.ceil(gap_days + 1.0))            # >= 24h since the last attempt

        attempts_left = state.attempt_budget - state.attempts_used
        reauth_tried = any(e.get("action") == "reauth" for e in state.history)
        p_dead = probs["mandate_dead"]

        # rank every legal retry day by EV
        ranked = []
        if attempts_left > 0:
            d = min_day
            while d <= hz:
                ev = self._ev_retry(probs, w, d, observed_at, cycle_close, reset_dt,
                                    amount, not_funded_by)
                ranked.append((ev, d))
                d += self.params.day_grid_step
        ranked.sort(reverse=True)
        best_retry_ev = ranked[0][0] if ranked else float("-inf")

        # re-auth candidate — only worth the friction if the mandate might be
        # dead; on a very-likely-live mandate a re-auth request just annoys.
        rd = max(1, min_day)
        ev_reauth = (self._ev_reauth(probs, rd, observed_at, amount)
                     if (not reauth_tried and rd <= hz
                         and p_dead >= self.params.reauth_min_p_dead) else float("-inf"))

        # -- nothing worth doing --
        floor = max(self.params.min_ev_to_act,
                    self.params.min_ev_frac_of_amount * amount)
        if max(best_retry_ev, ev_reauth) <= floor:
            term = "escalate" if p_dead >= 0.5 else "stop"
            return Plan([], terminal=term,
                        note=f"stop: best EV {max(best_retry_ev, ev_reauth):+.0f} < "
                             f"floor {floor:.0f}; P(dead)={p_dead:.2f}")

        # -- re-auth wins --
        if ev_reauth >= best_retry_ev:
            reauth_act = ScheduledAction((observed_at + timedelta(days=rd)).replace(hour=11), "reauth")
            if p_dead >= 0.5:                                   # probably dead: one shot, then a human
                return Plan([reauth_act], terminal="escalate",
                            note=f"reauth @{rd}d EV {ev_reauth:+.0f}; P(dead)={p_dead:.2f} -> escalate")
            # uncertain: re-auth now, and still line up retries across the window
            tail = self._retry_days(probs, w, observed_at, cycle_close, min_day + 1, hz,
                                    attempts_left)
            acts = [reauth_act] + [ScheduledAction((observed_at + timedelta(days=c)).replace(hour=11), "retry")
                                   for c in tail]
            return Plan(acts, terminal="stop",
                        note=f"reauth @{rd}d + retries @{'/'.join(map(str, tail))}d  "
                             f"reauthEV {ev_reauth:+.0f}  P(dead)={p_dead:.2f}")

        # -- retry: EV says "go". Commit the whole remaining budget across the
        #    predicted funding window in one plan and exit (a receding horizon
        #    buys nothing here — the sim gives no signal back and staying engaged
        #    only exposes the mandate to the revocation hazard longer).
        days = self._retry_days(probs, w, observed_at, cycle_close, min_day, hz, attempts_left)
        close_day = (cycle_close - observed_at).days
        acts = [ScheduledAction((observed_at + timedelta(days=c)).replace(hour=11), "retry")
                for c in days]
        return Plan(acts, terminal="stop",
                    note=f"retry @{'/'.join(map(str, days))}d  bestEV {best_retry_ev:+.0f}  "
                         f"P(dead)={p_dead:.2f}  window p50/p85={w['days_p50']:.0f}/{w['days_p85']:.0f}d "
                         f"close@{close_day}d")

    # ---- structured explanation (for the audit trail, Phase 8) -----------
    def explain(self, case: dict, state=None) -> dict:
        """The decision, opened up: what the policy believed, every candidate it
        priced, what it chose and why. Pure function of the observable case."""
        from harness.engine import EngagementState

        if state is None:
            state = EngagementState(observed_at=parse_dt(case["observed_at"]),
                                    horizon_days=case["horizon_days"],
                                    attempt_budget=3)
        observed_at = state.observed_at
        hz = state.horizon_days
        amount = float(case["mandate"]["amount"])
        probs = self.clf.predict_proba_one(case)
        w = self.liq.predict_window(case)
        ny, nm = add_months(observed_at.year, observed_at.month, 1)
        cycle_close = ist(ny, nm, 1)
        close_day = (cycle_close - observed_at).days

        # price every retry day + the re-auth option + doing nothing
        cand = []
        for d in range(1, hz + 1):
            cand.append({"action": "retry", "day": d,
                         "ev": round(self._ev_retry(probs, w, d, observed_at, cycle_close,
                                                    cycle_close, amount, 0.0), 1)})
        cand.append({"action": "reauth", "day": 1,
                     "ev": round(self._ev_reauth(probs, 1, observed_at, amount), 1)})
        cand.append({"action": "stop", "day": None, "ev": 0.0})
        cand.sort(key=lambda c: c["ev"], reverse=True)

        plan = self.plan(case, state)
        return {
            "case_id": case["case_id"],
            "observed_at": case["observed_at"],
            "amount": amount,
            "failure_code": case["failure"].get("token"),
            "cause_posterior": {k: round(v, 3) for k, v in
                                sorted(probs.items(), key=lambda kv: -kv[1])},
            "funding_window": {"p50_days": w["days_p50"], "p85_days": w["days_p85"],
                               "date_p50": w["date_p50"].date().isoformat(),
                               "date_p85": w["date_p85"].date().isoformat()},
            "cycle_close_day": close_day,
            "ltv_estimate": round(self.costs.ltv_estimate(amount), 0),
            "ev_floor": round(max(self.params.min_ev_to_act,
                                  self.params.min_ev_frac_of_amount * amount), 1),
            "candidates_top": cand[:5],
            "decision": {
                "terminal": plan.terminal,
                "actions": [{"day": (a.at.date() - observed_at.date()).days, "kind": a.kind}
                            for a in plan.actions],
                "note": plan.note,
            },
        }

    def _retry_days(self, probs, w, observed_at, cycle_close, min_day, hz, budget) -> list[int]:
        """Where to place up to `budget` retries — driven by the most likely cause:

          insufficient_balance -> bracket the predicted funding window, hedge late
                                  (an account is never funded before pay-day, and
                                  the model errs early as often as late)
          bank_downtime        -> soon; outages clear in hours to a day
          limit_breach         -> just after the cap resets on the 1st
        """
        close_day = (cycle_close - observed_at).days
        top = max(("insufficient_balance", "bank_downtime", "limit_breach"),
                  key=lambda c: probs.get(c, 0.0))

        if top == "bank_downtime":
            cand = [1, 2, 4, 8]
        elif top == "limit_breach":
            cand = [close_day + 1, close_day + 2, close_day + 4]
        else:
            p50, p85 = w["days_p50"], w["days_p85"]
            spread = max(p85 - p50, 3.0)
            cand = [p50, p85, p85 + spread]

        days: list[int] = []
        for c in cand:
            di = int(round(c))
            if di < min_day or di > hz or not all(abs(di - x) >= 1 for x in days):
                continue
            days.append(di)
            if len(days) >= budget:
                break
        if not days:
            days = [min(hz, max(min_day, int(round(p85))))]
        return sorted(days)
