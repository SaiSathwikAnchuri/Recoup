"""The closed loop — Recoup 2.0.

Ties the pieces together into an event-driven recovery cycle:

    payment event -> idempotency -> update customer state -> decide (ROS) ->
    schedule the single next action -> execute -> observe outcome ->
    update state -> re-plan -> ... -> stop when nothing clears the EV floor

Unlike the one-shot service, the loop commits **one action at a time** and
re-decides from the updated engagement history after each result. In dry-run it
only schedules and records intent; real outcomes arrive back as webhook events
(`payment.captured`, `subscription.halted`) or, in the demo, from the simulator.
"""

from __future__ import annotations

import time
from datetime import timedelta

from agent.costs import CostModel
from harness.engine import EngagementState
from simulator.calendar_utils import parse_dt

from . import bridge, webhook
from .events import Event, EventType
from .state import CustomerState

_COSTS = CostModel.from_yaml()
_MSG_KINDS = ("sms", "nudge")


# ---------------------------------------------------------------------------
def _engagement_state(case: dict, events: list[Event]) -> EngagementState:
    """Reconstruct the engine's `EngagementState` from the event log so the
    policy re-plans against real accumulated history."""
    obs = parse_dt(case["observed_at"])
    st = EngagementState(observed_at=obs, horizon_days=int(case.get("horizon_days", 45)),
                         attempt_budget=bridge._CFG.attempt_budget, now=obs)
    hist: list[dict] = []
    msgs: list[tuple[int, str]] = []
    for e in sorted(events, key=lambda x: x.at):
        if e.type is EventType.RETRY_EXECUTED:
            at = e.payload.get("at") or obs.isoformat()
            hist.append({"action": "retry", "at": at, "result": e.payload.get("result", "fail")})
        elif e.type is EventType.REAUTH_CREATED:
            hist.append({"action": "reauth", "at": e.payload.get("at") or obs.isoformat()})
        elif e.type is EventType.MESSAGE_SENT:
            msgs.append((int(e.payload.get("day", 0)), e.payload.get("channel", "sms")))
    st.history = hist
    st.messages = msgs
    st.attempts_used = sum(1 for h in hist if h["action"] == "retry")
    st.round = st.attempts_used
    st.last_retry_failed = bool(hist and hist[-1].get("result") == "fail")
    return st


def _reward(case: dict, kind: str, result: str, before: dict | None, after: dict,
            payload: dict) -> float:
    amount = float(case["mandate"]["amount"])
    r = -_COSTS.action_cost(kind if kind in _COSTS.action else "retry")
    if result == "success" and kind in ("retry", "reauth"):
        r += _COSTS.recovery_value(amount, days=payload.get("delay_days") or 0.0)
        if payload.get("late"):
            r -= _COSTS.missed_cycle_penalty(amount)
    d_churn = float(after.get("churn_risk", 0.0)) - float((before or {}).get("churn_risk", 0.0))
    r -= d_churn * _COSTS.ltv_estimate(amount)
    return round(r, 2)


# ---------------------------------------------------------------------------
def _prior_history(store, case_id: str) -> dict | None:
    st = store.get_state(case_id)
    if not st:
        return None
    return {
        "consecutive_successes": st.get("consecutive_successes", 0),
        "last_success": st.get("last_success_date"),
        "days_since_last_success": st.get("days_since_last_success", 30),
        "success_days_of_month": st.get("historical_funding_days", []),
        "cycles_observed": max(st.get("successful_payment_count", 0), 0),
    }


def open_recovery(store, case: dict, *, source: str = "webhook",
                  dedup_key: str | None = None) -> dict:
    """Start (or continue) a recovery for `case`. Idempotent on `dedup_key`."""
    cid = case["case_id"]
    if dedup_key:
        existing = store.seen_key(dedup_key)
        if existing:
            return {"duplicate": True, "case_id": existing,
                    "case": store.get_case(existing),
                    "decision": store.latest_decision(existing),
                    "customer_state": store.get_state(existing)}
        store.remember_key(dedup_key, "webhook", cid)

    store.append_event(Event(
        EventType.PAYMENT_FAILED, customer_id=cid, mandate_id=cid, case_id=cid,
        payload={"failure_code": case["failure"].get("token"),
                 "amount": float(case["mandate"]["amount"]),
                 "age_months": case["mandate"].get("age_months"),
                 "observed_at": case["observed_at"]},
        dedup_key=f"failed:{dedup_key}" if dedup_key else None))

    events = store.events_for_customer(cid)
    cstate = CustomerState.rebuild(events, base_case=case)
    store.save_state(cstate)
    store.upsert_case(case, source=source)

    scheduled = _plan_and_schedule(store, case, cstate, first=True)
    return {"duplicate": False, "case_id": cid, "case": case,
            "decision": store.latest_decision(cid),
            "customer_state": cstate.to_dict(),
            "scheduled": scheduled}


def _plan_and_schedule(store, case: dict, cstate: CustomerState, *, first: bool = False) -> list[dict]:
    cid = case["case_id"]
    events = store.events_for_customer(cid)
    est = _engagement_state(case, events)
    decision = bridge.decide(case, customer_state=cstate, engagement=est)
    store.add_decision(cid, decision)
    store.cancel_pending_actions(cid)

    obs = parse_dt(case["observed_at"])
    acts = decision["decision"]["actions"][:1]        # closed loop: one action, then re-decide
    term = decision["decision"]["terminal"]
    out: list[dict] = []
    for a in acts:
        due = (obs + timedelta(days=a["day"])).timestamp()
        store.schedule_actions(cid, [{"kind": a["kind"], "due_at": due}])
        et = EventType.REAUTH_CREATED if a["kind"] == "reauth" else EventType.RETRY_SCHEDULED
        store.append_event(Event(et, cid, cid, cid, payload={"day": a["day"], "kind": a["kind"]}))
        out.append({"kind": a["kind"], "day": a["day"], "due_at": due})

    if not acts:
        if term == "escalate":
            store.escalate(cid, decision["decision"]["note"])
            store.append_event(Event(EventType.RECOVERY_ESCALATED, cid, cid, cid,
                                     payload={"reason": decision["decision"]["note"]}))
        else:
            store.set_case_status(cid, "closed" if not first else "open")
            store.append_event(Event(EventType.RECOVERY_STOPPED, cid, cid, cid,
                                     payload={"reason": decision["decision"]["note"]}))
    return out


def record_action_result(store, case_id: str, kind: str, result: str, **payload) -> dict:
    """Observe the outcome of one executed action, fold it into state, record the
    reward, and re-plan unless the recovery is now terminal."""
    case = store.get_case(case_id)
    if case is None:
        return {"error": "unknown case", "case_id": case_id}
    before = store.get_state(case_id)

    if kind == "retry":
        store.append_event(Event(EventType.RETRY_EXECUTED, case_id, case_id, case_id,
                                 payload={"result": result, **payload}))
    elif kind == "reauth":
        store.append_event(Event(EventType.REAUTH_COMPLETED, case_id, case_id, case_id,
                                 payload={"result": result, **payload}))
    elif kind in _MSG_KINDS:
        store.append_event(Event(EventType.MESSAGE_SENT, case_id, case_id, case_id,
                                 payload={"channel": kind, "day": payload.get("day", 0)}))

    recovered = result == "success" and kind in ("retry", "reauth")
    if recovered:
        store.append_event(Event(EventType.PAYMENT_RECOVERED, case_id, case_id, case_id,
                                 payload={"amount": float(case["mandate"]["amount"]), "via": kind,
                                          "delay_days": payload.get("delay_days"),
                                          "funding_day": payload.get("funding_day"),
                                          "recovered_date": payload.get("recovered_date")}))

    cstate = CustomerState.rebuild(store.events_for_customer(case_id), base_case=case)
    store.save_state(cstate)
    if recovered:
        store.set_case_status(case_id, "recovered")
        store.cancel_pending_actions(case_id)

    rwd = _reward(case, kind, result, before, cstate.to_dict(), payload)
    store.record_outcome(case_id, action=kind, action_at=time.time(), result=result,
                         recovered_amount=float(case["mandate"]["amount"]) if recovered else 0.0,
                         recovery_delay=payload.get("delay_days"), reward=rwd,
                         state_before=before, state_after=cstate.to_dict())

    replanned = []
    if not recovered and cstate.recovery_stage not in ("revoked", "escalated", "stopped"):
        replanned = _plan_and_schedule(store, case, cstate)

    return {"case_id": case_id, "kind": kind, "result": result, "reward": rwd,
            "recovered": recovered, "stage": cstate.recovery_stage,
            "customer_state": cstate.to_dict(),
            "next": replanned or store.pending_actions_for(case_id)}


def note_execution(store, case_id: str, kind: str) -> dict:
    """A side effect fired (dry-run or a real gateway call), but its outcome is
    not yet known. Record the intent as an event + a zero-recovery outcome row;
    the definitive result arrives later as a `payment.captured` / `failed` event."""
    case = store.get_case(case_id)
    if case is None:
        return {"error": "unknown case"}
    if kind == "retry":
        store.append_event(Event(EventType.RETRY_EXECUTED, case_id, case_id, case_id,
                                 payload={"result": "pending"}))
    elif kind == "reauth":
        store.append_event(Event(EventType.REAUTH_CREATED, case_id, case_id, case_id,
                                 payload={"result": "pending"}))
    elif kind in _MSG_KINDS:
        store.append_event(Event(EventType.MESSAGE_SENT, case_id, case_id, case_id,
                                 payload={"channel": kind}))
    cstate = CustomerState.rebuild(store.events_for_customer(case_id), base_case=case)
    store.save_state(cstate)
    store.record_outcome(case_id, action=kind, action_at=time.time(), result="executed",
                         reward=-_COSTS.action_cost(kind if kind in _COSTS.action else "retry"),
                         state_after=cstate.to_dict())
    return {"case_id": case_id, "kind": kind, "stage": cstate.recovery_stage}


def log_simulation(store, case: dict, sim: dict) -> None:
    """Demo only — replay a truth-driven 45-day simulation into the event log so
    the customer timeline and the outcomes table show a real closed loop."""
    cid = case["case_id"]
    obs = parse_dt(case["observed_at"])
    r = sim["recoup"]
    for ev in r.get("timeline", []):
        act = ev.get("action")
        day = ev.get("day", 0)
        if act == "retry":
            store.append_event(Event(EventType.RETRY_EXECUTED, cid, cid, cid,
                                     payload={"result": ev.get("result", "fail"), "day": day,
                                              "at": (obs + timedelta(days=day)).isoformat()}))
        elif act == "reauth":
            store.append_event(Event(EventType.REAUTH_CREATED, cid, cid, cid, payload={"day": day}))
            store.append_event(Event(EventType.REAUTH_COMPLETED, cid, cid, cid,
                                     payload={"result": ev.get("result", "fail"), "day": day}))
        elif act in _MSG_KINDS:
            store.append_event(Event(EventType.MESSAGE_SENT, cid, cid, cid,
                                     payload={"channel": act, "day": day}))
        elif act == "revoked":
            store.append_event(Event(EventType.MANDATE_REVOKED, cid, cid, cid, payload={"day": day}))
    if r.get("recovered"):
        store.append_event(Event(EventType.PAYMENT_RECOVERED, cid, cid, cid,
                                 payload={"amount": r.get("amount_recovered", 0.0),
                                          "via": "reauth" if r.get("via_reauth") else "retry",
                                          "delay_days": r.get("days_to_recovery")}))
    elif r.get("escalated"):
        store.append_event(Event(EventType.RECOVERY_ESCALATED, cid, cid, cid,
                                 payload={"reason": r.get("stop_reason", "escalated")}))
    cstate = CustomerState.rebuild(store.events_for_customer(cid), base_case=case)
    store.save_state(cstate)
    store.record_outcome(cid, action="simulation", action_at=time.time(),
                         result="recovered" if r.get("recovered") else
                         "revoked" if r.get("revoked") else
                         "escalated" if r.get("escalated") else "unresolved",
                         recovered_amount=r.get("amount_recovered", 0.0),
                         recovery_delay=r.get("days_to_recovery"),
                         reward=float(r.get("net_value", 0.0)),
                         state_after=cstate.to_dict())


def mark_revoked(store, case_id: str, reason: str = "mandate revoked at bank") -> dict:
    case = store.get_case(case_id)
    store.append_event(Event(EventType.MANDATE_REVOKED, case_id, case_id, case_id,
                             payload={"reason": reason}))
    store.set_case_status(case_id, "revoked")
    store.cancel_pending_actions(case_id)
    if case is not None:
        store.save_state(CustomerState.rebuild(store.events_for_customer(case_id), base_case=case))
    return {"case_id": case_id, "revoked": True}


# ---------------------------------------------------------------------------
def ingest_webhook_event(store, event: dict, *, dedup_key: str | None = None) -> dict:
    """Dispatch a raw Razorpay webhook to the right loop entry point."""
    etype = str(event.get("event", ""))
    pay = (((event.get("payload") or {}).get("payment") or {}).get("entity") or {})
    sub = (((event.get("payload") or {}).get("subscription") or {}).get("entity") or {})
    ref = sub.get("id") or pay.get("subscription_id") or pay.get("id")

    if etype in ("payment.captured",) or (etype == "subscription.charged"
                                          and pay.get("status") == "captured"):
        if ref and store.get_case(str(ref)):
            return record_action_result(store, str(ref), "retry", "success",
                                        delay_days=None)
        return {"ignored": etype, "reason": "no open recovery for this subscription"}

    if etype in ("subscription.halted", "subscription.cancelled", "mandate.revoked"):
        if ref and store.get_case(str(ref)):
            return mark_revoked(store, str(ref))
        return {"ignored": etype, "reason": "no open recovery"}

    # default: a failed / pending charge -> open or continue a recovery
    case = webhook.parse_event(event, history=_prior_history(store, str(ref)) if ref else None)
    return open_recovery(store, case, dedup_key=dedup_key)
