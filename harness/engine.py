"""The rollout engine.

A policy emits a `Plan` — an ordered list of `ScheduledAction`s plus a `terminal`
mode. The engine executes the plan against the hidden truth, drawing outcomes from
a per-case pre-sampled `World` so that every policy faces an identical realisation
of the same case (paired comparison — report R10).

  terminal="stop"      run the whole script, then give up (baselines)
  terminal="escalate"  run the script, then hand to a human (exception list)
  terminal="replan"    after a failed retry, ask the policy again (receding horizon)

Recovery and revocation race each other day by day over the engagement window.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import datetime, timedelta

import numpy as np

from simulator.calendar_utils import add_months, ist, parse_dt
from simulator.response import p_success, revocation_hazard, truth_from_record

from .constraints import Constraints, enforce
from .plan import DEBIT_HOUR, Plan, ScheduledAction  # noqa: F401  (re-exported)

# action kind -> revocation-hazard message key in priors.revocation.message_bump
MESSAGE_KIND = {
    "nudge": "predebit_notification",
    "sms": "sms_failure_framed",
    "reauth": "reauth_request",
}


# ---------------------------------------------------------------------------
# policy-facing types
# ---------------------------------------------------------------------------
@dataclass
class EngagementState:
    """Read-only-by-convention view the policy gets on each (re)plan."""
    observed_at: datetime
    horizon_days: int
    attempt_budget: int
    attempts_used: int = 0
    messages: list[tuple[int, str]] = field(default_factory=list)  # (day_index, kind)
    round: int = 0
    now: datetime | None = None
    last_retry_failed: bool = False
    history: list[dict] = field(default_factory=list)

    def horizon_end(self) -> datetime:
        return self.observed_at + timedelta(days=self.horizon_days)


# ---------------------------------------------------------------------------
# config + world
# ---------------------------------------------------------------------------
@dataclass
class HarnessConfig:
    max_rounds: int = 6              # replanning rounds before forced stop
    rev_cfg: dict = field(default_factory=dict)          # priors["revocation"]
    constraints: Constraints | None = None              # config/constraints.yaml

    def __post_init__(self):
        if self.constraints is None:
            self.constraints = Constraints.from_yaml()

    @property
    def attempt_budget(self) -> int:
        return self.constraints.max_retries


def _case_seed(case_id: str, run_seed: int) -> int:
    h = hashlib.sha256(f"{run_seed}:{case_id}".encode()).digest()
    return int.from_bytes(h[:8], "big")


class World:
    """Pre-drawn uniforms indexed by day, so policy A and policy B see the same luck."""

    def __init__(self, case_id: str, run_seed: int, horizon: int):
        rng = np.random.default_rng(_case_seed(case_id, run_seed))
        n = horizon + 3
        self.u_retry = rng.random(n)
        self.u_revoke = rng.random(n)
        self.u_reauth = rng.random(n)
        self._n = n

    def _i(self, day: int) -> int:
        return min(max(day, 0), self._n - 1)


# ---------------------------------------------------------------------------
# outcome
# ---------------------------------------------------------------------------
@dataclass
class Outcome:
    case_id: str
    policy: str
    true_cause: str
    income_pattern: str
    true_ltv: float
    amount: float
    recovered: bool
    amount_recovered: float
    recovered_on_time: bool
    via_reauth: bool
    days_to_recovery: float | None
    attempts_used: int
    messages_sent: int
    message_kinds: dict
    revoked: bool
    mandate_preserved: bool
    cycle_missed: bool
    escalated: bool
    blocked_actions: int
    stop_reason: str
    timeline: list[dict]

    def as_row(self) -> dict:
        d = dict(self.__dict__)
        d.pop("timeline")
        return d


# ---------------------------------------------------------------------------
# the rollout
# ---------------------------------------------------------------------------
def run_case(case: dict, truth_rec: dict, policy, cfg: HarnessConfig, run_seed: int) -> Outcome:
    truth = truth_from_record(truth_rec)
    observed_at = parse_dt(case["observed_at"])
    horizon = case["horizon_days"]
    horizon_end = observed_at + timedelta(days=horizon)
    world = World(case["case_id"], run_seed, horizon)
    amount = truth.amount

    ny, nm = add_months(observed_at.year, observed_at.month, 1)
    cycle_close = ist(ny, nm, 1)

    def day_of(dt: datetime) -> int:
        return (dt.date() - observed_at.date()).days

    state = EngagementState(
        observed_at=observed_at, horizon_days=horizon,
        attempt_budget=cfg.attempt_budget, now=observed_at,
    )
    timeline: list[dict] = []

    recovered = revoked = via_reauth = escalated = False
    recovered_at: datetime | None = None
    stop_reason = ""
    cursor_day = 0          # revocation has been resolved for all days < cursor_day
    last_action_day = 0
    blocked_actions = 0

    def legalize(raw_plan: Plan) -> Plan:
        """Run the policy's plan through the structural constraint filter, carrying
        forward the retries / messages already spent this engagement."""
        nonlocal blocked_actions
        prior_r = tuple(parse_dt(e["at"]) for e in state.history
                        if e.get("action") == "retry" and "at" in e)
        prior_m = tuple(observed_at + timedelta(days=d, hours=DEBIT_HOUR)
                        for d, _ in state.messages)
        legal, viol = enforce(raw_plan, observed_at=observed_at, horizon_end=horizon_end,
                              c=cfg.constraints, prior_retries=prior_r, prior_messages=prior_m)
        if raw_plan.note:
            timeline.append({"day": day_of(state.now or observed_at),
                             "action": "plan", "note": raw_plan.note})
        for v in viol:
            blocked_actions += 1
            timeline.append({"day": day_of(parse_dt(v["at"])), "action": v["action"],
                             "result": f"constraint:{v['rule']}:{v['resolution']}"})
        return legal

    def roll_revocation(upto_day: int) -> bool:
        """Resolve revocation for [cursor_day, upto_day). Returns True if it fired."""
        nonlocal cursor_day, revoked
        upto_day = min(upto_day, horizon)
        while cursor_day < upto_day:
            msgs = [m for m in state.messages if m[0] <= cursor_day]
            hz = revocation_hazard(truth.h_base, msgs, cursor_day, cfg.rev_cfg)
            if world.u_revoke[world._i(cursor_day)] < hz:
                timeline.append({"day": cursor_day, "action": "revoked",
                                 "hazard": round(hz, 4)})
                revoked = True
                cursor_day += 1
                return True
            cursor_day += 1
        return False

    plan = legalize(policy.plan(case, state))
    while True:
        state.round += 1
        actions = sorted((a for a in plan.actions if observed_at <= a.at <= horizon_end),
                         key=lambda a: a.at)
        replanned = False

        for a in actions:
            ad = day_of(a.at)
            if roll_revocation(ad):
                break
            cursor_day = max(cursor_day, ad)

            if a.kind == "retry":
                if state.attempts_used >= cfg.attempt_budget:
                    timeline.append({"day": ad, "action": "retry", "result": "blocked_no_budget"})
                    continue
                state.attempts_used += 1
                p = p_success(truth, "retry", a.at)
                ok = world.u_retry[world._i(ad)] < p
                ev = {"day": ad, "at": a.at.isoformat(), "action": "retry",
                      "p_success": round(p, 3), "result": "success" if ok else "fail"}
                timeline.append(ev)
                state.history.append(ev)
                last_action_day = ad
                if ok:
                    recovered, recovered_at = True, a.at
                    break
                state.last_retry_failed = True
                if plan.terminal == "replan":
                    state.now = a.at
                    replanned = True
                    break

            elif a.kind in ("nudge", "sms"):
                state.messages.append((ad, MESSAGE_KIND[a.kind]))
                ev = {"day": ad, "action": a.kind}
                timeline.append(ev)
                state.history.append(ev)
                last_action_day = ad

            elif a.kind == "reauth":
                state.messages.append((ad, MESSAGE_KIND["reauth"]))
                ok = world.u_reauth[world._i(ad)] < truth.reauth_success_prob
                ev = {"day": ad, "action": "reauth",
                      "p_success": round(truth.reauth_success_prob, 3),
                      "result": "success" if ok else "fail"}
                timeline.append(ev)
                state.history.append(ev)
                last_action_day = ad
                if ok:
                    recovered, recovered_at, via_reauth = True, a.at, True
                    break
                if plan.terminal == "replan":
                    state.now = a.at
                    replanned = True
                    break

        if recovered or revoked:
            break
        if replanned and state.round < cfg.max_rounds:
            plan = legalize(policy.plan(case, state))
            continue

        # plan finished without recovery / revocation
        if plan.terminal == "escalate":
            escalated, stop_reason = True, "escalate"
        elif replanned:
            stop_reason = "max_rounds"
        else:
            stop_reason = "plan_exhausted"
        break

    # a mandate left unrecovered keeps facing the (message-decayed) revocation hazard
    # for the rest of the horizon; recovering fast is what protects it
    if not recovered and not revoked:
        roll_revocation(horizon)

    days_to_recovery = ((recovered_at - observed_at).total_seconds() / 86400.0
                        if recovered_at else None)
    on_time = bool(recovered and recovered_at <= cycle_close)

    mk: dict[str, int] = {}
    for _, kind in state.messages:
        mk[kind] = mk.get(kind, 0) + 1

    return Outcome(
        case_id=case["case_id"], policy=policy.name, true_cause=truth_rec["true_cause"],
        income_pattern=truth_rec["income_pattern"],
        true_ltv=float(truth_rec.get("ltv_true", 0.0)), amount=amount,
        recovered=recovered, amount_recovered=amount if recovered else 0.0,
        recovered_on_time=on_time, via_reauth=via_reauth,
        days_to_recovery=round(days_to_recovery, 2) if days_to_recovery is not None else None,
        attempts_used=state.attempts_used, messages_sent=len(state.messages), message_kinds=mk,
        revoked=revoked, mandate_preserved=not revoked,
        cycle_missed=not on_time, escalated=escalated, blocked_actions=blocked_actions,
        stop_reason=stop_reason or ("recovered" if recovered else "revoked" if revoked else "unresolved"),
        timeline=timeline,
    )


def run_batch(cases: list[dict], truth: dict, policy, cfg: HarnessConfig,
              run_seed: int) -> list[Outcome]:
    return [run_case(c, truth[c["case_id"]], policy, cfg, run_seed) for c in cases]
