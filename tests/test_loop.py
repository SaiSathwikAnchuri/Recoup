"""Recoup 2.0 — the closed loop: ingest -> decide -> schedule -> observe -> re-plan -> stop."""

from __future__ import annotations

import pytest

from agent.classifier import CauseClassifier
from agent.liquidity import LiquidityModel

pytestmark = pytest.mark.skipif(
    not (CauseClassifier.default_exists() and LiquidityModel.default_exists()),
    reason="models not trained")

from service import loop                       # noqa: E402
from service.store import Store                # noqa: E402


def _event(amount=299900, code="U30", desc="insufficient balance", sub="sub_test", pid="pay_1"):
    return {"event": "payment.failed", "payload": {"payment": {"entity": {
        "amount": amount, "error_code": code, "error_description": desc,
        "id": pid, "subscription_id": sub}}}}


@pytest.fixture
def store(tmp_path):
    s = Store(tmp_path / "loop.db")
    yield s
    s.close()


# -- ingestion + idempotency ------------------------------------------
def test_open_recovery_decides_and_schedules(store):
    r = loop.ingest_webhook_event(store, _event(), dedup_key="evt_a")
    assert r["duplicate"] is False
    assert r["case_id"] == "sub_test"
    assert r["decision"]["decision"]["terminal"] in ("stop", "escalate", "replan")
    assert "customer_state" in r
    evs = [e.type.value for e in store.events_for_case("sub_test")]
    assert "PAYMENT_FAILED" in evs


def test_duplicate_webhook_is_ignored(store):
    a = loop.ingest_webhook_event(store, _event(), dedup_key="evt_dup")
    b = loop.ingest_webhook_event(store, _event(), dedup_key="evt_dup")
    assert b["duplicate"] is True
    assert b["case_id"] == a["case_id"]
    # exactly one PAYMENT_FAILED, one case, one set of scheduled actions
    assert sum(e.type.value == "PAYMENT_FAILED" for e in store.events_for_case("sub_test")) == 1
    assert len(store.list_cases()) == 1


def test_malformed_event_does_not_crash_the_parser(store):
    r = loop.ingest_webhook_event(store, {"event": "payment.failed"}, dedup_key="evt_bad")
    assert "case_id" in r          # parser fills defaults; no exception


# -- observe outcome + re-plan --------------------------------------
def test_recovered_payment_closes_the_loop(store):
    loop.ingest_webhook_event(store, _event(sub="sub_rec"), dedup_key="e1")
    res = loop.record_action_result(store, "sub_rec", "retry", "success", delay_days=4)
    assert res["recovered"] is True
    assert res["stage"] == "recovered"
    assert store.get_case("sub_rec") and store.stats()["recovered"] == 1
    assert any(e.type.value == "PAYMENT_RECOVERED" for e in store.events_for_case("sub_rec"))
    assert store.pending_actions_for("sub_rec") == []      # nothing left scheduled


def test_failed_retry_triggers_a_replan(store):
    loop.ingest_webhook_event(store, _event(sub="sub_rp"), dedup_key="e2")
    n_decisions_before = len(store.events_for_case("sub_rp"))
    res = loop.record_action_result(store, "sub_rp", "retry", "fail")
    assert res["recovered"] is False
    # a re-plan either scheduled the next step or stopped — either way a new
    # decision + event was written
    assert len(store.events_for_case("sub_rp")) > n_decisions_before
    assert res["stage"] in ("retrying", "reauth", "stopped", "escalated")


def test_revocation_is_terminal(store):
    loop.ingest_webhook_event(store, _event(sub="sub_rev"), dedup_key="e3")
    loop.mark_revoked(store, "sub_rev")
    c = store.list_cases()
    assert [x for x in c if x["case_id"] == "sub_rev"][0]["status"] == "revoked"
    assert store.pending_actions_for("sub_rev") == []


def test_reward_is_recorded_per_action(store):
    loop.ingest_webhook_event(store, _event(sub="sub_rw"), dedup_key="e4")
    loop.record_action_result(store, "sub_rw", "retry", "success", delay_days=2)
    outs = store.outcomes_for("sub_rw")
    assert outs and outs[-1]["reward"] > 0        # a successful recovery pays off
