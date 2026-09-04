"""Recoup 2.0 — the event log and the customer state engine."""

from __future__ import annotations

from service.events import Event, EventType
from service.state import CustomerState
from service.store import Store

_CASE = {
    "case_id": "sub_1", "observed_at": "2026-03-11T11:00:00+05:30", "horizon_days": 45,
    "failure": {"token": "U30_insufficient_funds"},
    "mandate": {"amount": 3000.0, "category": "sip", "age_months": 8},
    "history": {"consecutive_successes": 4, "last_success": "2026-02-05",
                "success_days_of_month": [5, 5, 6], "cycles_observed": 6,
                "days_since_last_success": 34},
}


def _ev(t, **p):
    return Event(t, "sub_1", "sub_1", "sub_1", payload=p)


# -- event log -------------------------------------------------------------
def test_event_dedup_key_is_idempotent(tmp_path):
    s = Store(tmp_path / "e.db")
    e = Event(EventType.PAYMENT_FAILED, "c", "c", "c", dedup_key="k1")
    assert s.append_event(e) is True
    assert s.append_event(Event(EventType.PAYMENT_FAILED, "c", "c", "c", dedup_key="k1")) is False
    assert len(s.events_for_case("c")) == 1
    s.close()


def test_idempotency_key_binds_once(tmp_path):
    s = Store(tmp_path / "k.db")
    assert s.remember_key("evt_9", "webhook", "case_a") is True
    assert s.remember_key("evt_9", "webhook", "case_b") is False
    assert s.seen_key("evt_9") == "case_a"
    s.close()


def test_event_roundtrips_through_store(tmp_path):
    s = Store(tmp_path / "r.db")
    s.append_event(_ev(EventType.RETRY_EXECUTED, result="fail", day=3))
    (back,) = s.events_for_customer("sub_1")
    assert back.type is EventType.RETRY_EXECUTED and back.payload["result"] == "fail"
    s.close()


# -- state folding ------------------------------------------------------
def test_rebuild_folds_the_log():
    events = [
        _ev(EventType.PAYMENT_FAILED, failure_code="U30_insufficient_funds", amount=3000.0),
        _ev(EventType.RETRY_EXECUTED, result="fail"),
        _ev(EventType.MESSAGE_SENT, channel="sms"),
        _ev(EventType.RETRY_EXECUTED, result="fail"),
    ]
    st = CustomerState.rebuild(events, base_case=_CASE)
    assert st.failure_count == 1
    assert st.recovery_attempts == 2
    assert st.retry_results == [False, False]
    assert st.message_count == 1
    assert st.recovery_stage == "retrying"


def test_recovery_transition_and_history():
    events = [
        _ev(EventType.PAYMENT_FAILED, failure_code="U30_insufficient_funds", amount=3000.0),
        _ev(EventType.RETRY_EXECUTED, result="success"),
        _ev(EventType.PAYMENT_RECOVERED, amount=3000.0, via="retry", funding_day=7,
            recovered_date="2026-03-18"),
    ]
    st = CustomerState.rebuild(events, base_case=_CASE)
    assert st.recovery_stage == "recovered"
    assert st.recovery_successes == 1
    h = st.to_history()
    assert 7 in h["success_days_of_month"]
    assert h["last_success"] == "2026-03-18"


def test_churn_risk_rises_with_dunning_pressure():
    base = [_ev(EventType.PAYMENT_FAILED, failure_code="U30_insufficient_funds", amount=3000.0)]
    calm = CustomerState.rebuild(base, base_case=_CASE).churn_risk
    noisy = CustomerState.rebuild(
        base + [_ev(EventType.MESSAGE_SENT, channel="sms")] * 3, base_case=_CASE).churn_risk
    assert noisy > calm
    assert 0.0 <= calm <= 0.95 and 0.0 <= noisy <= 0.95


def test_state_never_carries_hidden_truth():
    import inspect
    import service.state as m
    # inspect the code, not the module docstring (which names the fields it excludes)
    body = "".join(inspect.getsource(fn) for fn in
                   (m.CustomerState.rebuild, m.CustomerState._apply, m.CustomerState.to_history))
    assert "simulator.response" not in body
    assert "true_cause" not in body and "ltv_true" not in body
    assert "truth_from_record" not in body
