"""Agent-side policies (need trained models; baselines/ stays dependency-free).

Two staged deliverables, each isolating one idea so the harness can price it:

  cause_aware      Phase 4 — fixed schedule + the escalate branch (stop retrying
                   a mandate that is probably dead).
  liquidity_aware  Phase 5 — cause_aware, plus: retries are scheduled against the
                   predicted funding window instead of a fixed calendar, and a
                   limit-breach waits for the cap to reset on the 1st.

Neither prices actions against cost yet — that is the Phase 7 EV policy.
"""

from __future__ import annotations

from baselines._util import act
from harness.engine import EngagementState, Plan
from simulator.calendar_utils import add_months, ist

from .classifier import CauseClassifier
from .liquidity import LiquidityModel

_FIXED = ((1, "retry"), (3, "retry"), (4, "sms"), (7, "retry"))


class CauseAwareRetry:
    name = "cause_aware"

    def __init__(self, clf: CauseClassifier | None = None):
        self._clf = clf or CauseClassifier.load()

    def plan(self, case: dict, state: EngagementState) -> Plan:
        o = state.observed_at
        probs = self._clf.predict_proba_one(case)
        if self._clf.should_escalate(probs):
            return Plan(actions=[act(o, 2, "reauth")], terminal="escalate")
        return Plan(actions=[act(o, d, k) for d, k in _FIXED], terminal="stop")


class LiquidityAwareRetry:
    name = "liquidity_aware"

    def __init__(self, clf: CauseClassifier | None = None,
                 liq: LiquidityModel | None = None):
        self._clf = clf or CauseClassifier.load()
        self._liq = liq or LiquidityModel.load()

    def plan(self, case: dict, state: EngagementState) -> Plan:
        o = state.observed_at
        hz = state.horizon_days
        probs = self._clf.predict_proba_one(case)

        # probably dead — one re-auth, then a human
        if self._clf.should_escalate(probs):
            return Plan(actions=[act(o, 2, "reauth")], terminal="escalate")

        # limit breach — the cap resets on the 1st; retry the day after
        top = max(probs, key=probs.get)
        if top == "limit_breach" and probs["limit_breach"] >= 0.45:
            y, m = add_months(o.year, o.month, 1)
            reset_day = (ist(y, m, 1).date() - o.date()).days
            d = min(max(reset_day + 1, 1), hz)
            return Plan(actions=[act(o, d, "retry")], terminal="stop")

        # otherwise treat it as a funding-window problem
        w = self._liq.predict_window(case)
        days = sorted({max(1, round(w["days_p50"])), max(2, round(w["days_p85"]))})
        actions = [act(o, d, "retry") for d in days if 1 <= d <= hz]
        extra = round(w["days_p85"]) + 5
        if len(actions) < 3 and 1 <= extra <= hz:
            actions.append(act(o, extra, "retry"))
        if not actions:
            actions = [act(o, 3, "retry"), act(o, 10, "retry")]
        return Plan(actions=actions, terminal="stop")


class RecoupV2:
    """Recoup 2.0 — the same economics as `recoup`, but adaptive.

    Instead of committing the whole retry ladder up front, it scores every
    candidate with the Recovery Opportunity Score (`agent/ros.py`), commits the
    single best action, and asks the engine to come back (`terminal="replan"`)
    once that action's result is known. When a retry fails, the next plan is
    built from the updated engagement history — a later funding day, one fewer
    attempt, whatever messages have gone out.

    In the frozen simulator a failed retry carries no new signal, so this lands
    close to one-shot `recoup`; its payoff is operational — in the live service
    it reacts to real late-arriving `payment.failed` / funding events.
    """

    name = "recoup_v2"

    def __init__(self, ro=None):
        from .ros import RecoveryOpportunity
        self._ro = ro or RecoveryOpportunity()

    def plan(self, case: dict, state) -> Plan:
        from harness.engine import EngagementState  # noqa: F401  (type only)
        from harness.plan import ScheduledAction
        from datetime import timedelta

        p = self._ro.p
        amount = float(case["mandate"]["amount"])
        floor = max(p.params.min_ev_to_act, p.params.min_ev_frac_of_amount * amount)
        cands = self._ro.score_candidates(case, state)
        best = cands[0]
        note = (f"round {state.round}: {best.action}"
                + (f" @{best.scheduled_day}d" if best.scheduled_day else "")
                + f"  ROS {best.score:+.0f} (EV {best.ev:+.0f})  floor {floor:.0f}")

        if best.score <= floor or best.action in ("stop", "wait"):
            probs = p.clf.predict_proba_one(case)
            # escalate on the FIRST look only (parity with one-shot `recoup`); a
            # dead-end reached mid-loop, after attempts were already spent, just
            # stops rather than paying the ₹30 human-handoff again.
            term = "escalate" if (probs["mandate_dead"] >= 0.5 and state.round == 0) else "stop"
            return Plan([], terminal=term, note=note + f" -> {term}")

        if best.action == "escalate":
            return Plan([], terminal="escalate", note=note)

        o = state.observed_at
        act = ScheduledAction((o + timedelta(days=best.scheduled_day)).replace(hour=11),
                              best.action)
        if best.action == "reauth" and p.clf.predict_proba_one(case)["mandate_dead"] >= 0.5:
            return Plan([act], terminal="escalate", note=note + " -> escalate")
        return Plan([act], terminal="replan", note=note)


def load_all() -> list:
    """Agent policies whose models are trained and ready. Empty list otherwise."""
    if not CauseClassifier.default_exists():
        return []
    out = [CauseAwareRetry()]
    if LiquidityModel.default_exists():
        from .ev_policy import EVPolicy
        out.append(LiquidityAwareRetry())
        out.append(EVPolicy())
        out.append(RecoupV2())
    return out
