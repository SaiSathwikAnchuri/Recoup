"""Phase 10 — the service layer: ingestion, decision API, execution, store."""

from __future__ import annotations

import hashlib
import hmac
import json

import pytest

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from agent.classifier import CauseClassifier  # noqa: E402
from agent.liquidity import LiquidityModel  # noqa: E402
from service import webhook  # noqa: E402
from service.store import Store  # noqa: E402

_MODELS = CauseClassifier.default_exists() and LiquidityModel.default_exists()


# -- webhook parsing (no models needed) ---------------------------------
def test_signature_skipped_without_secret(monkeypatch):
    monkeypatch.delenv("RAZORPAY_WEBHOOK_SECRET", raising=False)
    assert webhook.verify_signature(b'{"x":1}', None) is True


def test_signature_enforced_with_secret(monkeypatch):
    monkeypatch.setenv("RAZORPAY_WEBHOOK_SECRET", "whsec_demo")
    body = b'{"event":"payment.failed"}'
    good = hmac.new(b"whsec_demo", body, hashlib.sha256).hexdigest()
    assert webhook.verify_signature(body, good) is True
    assert webhook.verify_signature(body, "deadbeef") is False
    assert webhook.verify_signature(body, None) is False


@pytest.mark.parametrize("code,desc,expected", [
    ("U30", "insufficient funds", "U30_insufficient_funds"),
    ("U67", "limit exceeded", "U67_limit_exceeded"),
    (None, "remitter bank offline", "U88_bank_offline"),
    (None, "mandate revoked at bank", "ER_mandate_revoked"),
    ("SOMETHING_ELSE", "", "U69_generic_decline"),
])
def test_code_mapping(code, desc, expected):
    ev = {"event": "payment.failed", "payload": {"payment": {"entity": {
        "amount": 149900, "error_code": code, "error_description": desc}}}}
    case = webhook.parse_event(ev)
    assert case["failure"]["token"] == expected
    assert case["mandate"]["amount"] == 1499.0
    assert case["horizon_days"] == 45
    assert set(case["history"]) >= {"consecutive_successes", "success_days_of_month", "cycles_observed"}


# -- store --------------------------------------------------------------
def test_store_roundtrip(tmp_path):
    s = Store(tmp_path / "t.db")
    case = {"case_id": "c1", "mandate": {"amount": 500.0}, "failure": {"token": "U30_x"}}
    s.upsert_case(case, source="demo")
    assert s.get_case("c1")["case_id"] == "c1"
    s.add_decision("c1", {"cause_posterior": {"insufficient_balance": 0.9},
                          "decision": {"terminal": "stop", "note": "n", "actions": []}})
    assert s.latest_decision("c1")["decision"]["terminal"] == "stop"
    s.schedule_actions("c1", [{"kind": "retry", "due_at": 0.0}])
    assert s.due_actions()[0]["kind"] == "retry"
    s.set_case_status("c1", "recovered")
    assert s.stats()["recovered"] == 1
    s.close()


# -- executors (dry run, no network) ----------------------------------
def test_executors_dry_run_never_call_out(monkeypatch):
    from service import executors
    monkeypatch.delenv("RECOUP_EXECUTE_MODE", raising=False)
    monkeypatch.delenv("RAZORPAY_KEY_ID", raising=False)
    case = {"case_id": "c9", "mandate": {"amount": 2500.0}}
    for kind in ("retry", "reauth", "sms", "escalate"):
        r = executors.execute(kind, case)
        assert r["mode"] == "dry_run"
        assert "intent" in r
        if kind != "escalate":
            assert r["executed"] is False


# -- the API (needs trained models) ---------------------------------
@pytest.fixture(scope="module")
def client(tmp_path_factory):
    if not _MODELS:
        pytest.skip("models not trained")
    import service.app as app_mod
    app_mod.STORE = Store(tmp_path_factory.mktemp("svc") / "svc.db")
    return TestClient(app_mod.app)


@pytest.mark.skipif(not _MODELS, reason="models not trained")
def test_healthz(client):
    h = client.get("/healthz").json()
    assert h["ok"] and h["models_ready"] and h["execute_mode"] == "dry_run"


@pytest.mark.skipif(not _MODELS, reason="models not trained")
def test_ui_and_assets_serve(client):
    idx = client.get("/")
    assert idx.status_code == 200 and b'id="root"' in idx.content
    for asset in ("/app.js", "/data.js", "/vendor/react.js", "/vendor/htm.js"):
        assert client.get(asset).status_code == 200


@pytest.mark.skipif(not _MODELS, reason="models not trained")
def test_api_results_shape(client):
    r = client.get("/api/results")
    if r.status_code == 404:
        pytest.skip("results/ not populated — run reproduce")
    d = r.json()
    assert {"headline", "scoreboard", "ablation", "sensitivity", "fairness"} <= set(d)
    assert any(row["name"] == "Recoup" for row in d["scoreboard"])


@pytest.mark.skipif(not _MODELS, reason="models not trained")
def test_demo_random_decides_and_simulates(client):
    r = client.post("/demo/random?cause=mandate_dead").json()
    d = r["decision"]
    assert abs(sum(d["cause_posterior"].values()) - 1.0) < 0.02   # values are rounded to 3dp
    assert d["narration"]
    # a dead mandate is re-auth'd or escalated, never retried
    acts = d["decision"]["actions"]
    assert (not acts) or acts[0]["kind"] == "reauth"
    sim = r["simulation"]
    assert sim["true_cause"] == "mandate_dead"
    assert {"recoup", "fixed_schedule", "net_delta"} <= set(sim)
    assert client.get(f"/cases/{r['case_id']}").json()["case"]["case_id"] == r["case_id"]


@pytest.mark.skipif(not _MODELS, reason="models not trained")
def test_demo_random_same_seed_gives_the_same_decision_every_time(client):
    """A deterministic demo must mean the SAME outcome on every replay of the same
    seed, not just the same outcome on the first call (final-audit fix: repeated
    calls used to keep appending to the same demo case_id's event log)."""
    r1 = client.post("/demo/random?cause=mandate_dead&seed=2").json()
    r2 = client.post("/demo/random?cause=mandate_dead&seed=2").json()
    r3 = client.post("/demo/random?cause=mandate_dead&seed=2").json()
    assert r1["case_id"] == r2["case_id"] == r3["case_id"]
    assert r1["decision"]["decision"] == r2["decision"]["decision"] == r3["decision"]["decision"]
    assert r1["simulation"]["net_delta"] == r2["simulation"]["net_delta"] == r3["simulation"]["net_delta"]
    tl = client.get(f"/api/cases/{r1['case_id']}/timeline").json()
    assert sum(e["type"] == "PAYMENT_FAILED" for e in tl["events"]) == 1   # not 3


@pytest.mark.skipif(not _MODELS, reason="models not trained")
def test_webhook_ingests_and_schedules(client):
    ev = {"event": "payment.failed", "payload": {"payment": {"entity": {
        "amount": 299900, "error_code": "U30", "error_description": "insufficient balance",
        "id": "pay_svc_test"}}}}
    r = client.post("/webhook", json=ev).json()
    assert r["case_id"] == "pay_svc_test"
    assert r["case"]["failure"]["token"] == "U30_insufficient_funds"
    assert r["decision"]["decision"]["terminal"] in ("stop", "escalate", "replan")
    assert all(a["kind"] in ("retry", "reauth", "sms", "nudge") for a in r["scheduled"])
    # tick executes nothing yet (all actions are days in the future)
    assert client.post("/tick").json()["executed"] == []


@pytest.mark.skipif(not _MODELS, reason="models not trained")
def test_demo_outcome_feeds_the_closed_loop(client):
    ev = {"event": "payment.failed", "payload": {"payment": {"entity": {
        "amount": 199900, "error_code": "U30", "error_description": "insufficient balance",
        "id": "pay_outcome_test"}}}}
    client.post("/webhook", json=ev)
    r = client.post("/demo/outcome/pay_outcome_test?kind=retry&result=fail").json()
    assert r["recovered"] is False
    assert r["case_id"] == "pay_outcome_test"
    r2 = client.post("/demo/outcome/pay_outcome_test?kind=retry&result=success").json()
    assert r2["recovered"] is True
    assert client.post("/demo/outcome/no_such_case?kind=retry&result=fail").status_code == 404
    assert client.post("/demo/outcome/pay_outcome_test?kind=bogus&result=fail").status_code == 400


@pytest.mark.skipif(not _MODELS, reason="models not trained")
def test_webhook_rejects_bad_signature(client, monkeypatch):
    monkeypatch.setenv("RAZORPAY_WEBHOOK_SECRET", "whsec_x")
    r = client.post("/webhook", json={"event": "payment.failed"},
                    headers={"X-Razorpay-Signature": "nope"})
    assert r.status_code == 400


# -- Recoup 2.0 endpoints --------------------------------------------
@pytest.mark.skipif(not _MODELS, reason="models not trained")
def test_duplicate_webhook_is_idempotent(client):
    ev = {"id": "evt_idem_1", "event": "payment.failed", "payload": {"payment": {"entity": {
        "amount": 149900, "error_code": "U30", "error_description": "insufficient",
        "id": "pay_idem", "subscription_id": "sub_idem"}}}}
    a = client.post("/webhook", json=ev).json()
    b = client.post("/webhook", json=ev).json()
    assert a["case_id"] == "sub_idem"
    assert b.get("duplicate") is True and b["case_id"] == "sub_idem"
    assert len([c for c in client.get("/cases").json()["cases"] if c["case_id"] == "sub_idem"]) == 1


@pytest.mark.skipif(not _MODELS, reason="models not trained")
def test_malformed_webhook_body_is_400(client):
    r = client.post("/webhook", data=b"{not json", headers={"content-type": "application/json"})
    assert r.status_code == 400


@pytest.mark.skipif(not _MODELS, reason="models not trained")
def test_timeline_and_replan(client):
    ev = {"id": "evt_tl", "event": "payment.failed", "payload": {"payment": {"entity": {
        "amount": 259900, "error_code": "U30", "error_description": "insufficient",
        "id": "pay_tl", "subscription_id": "sub_tl"}}}}
    client.post("/webhook", json=ev)
    tl = client.get("/api/cases/sub_tl/timeline").json()
    assert tl["case_id"] == "sub_tl"
    assert any(e["type"] == "PAYMENT_FAILED" for e in tl["events"])
    rp = client.post("/api/cases/sub_tl/replan").json()
    assert "decision" in rp and "customer_state" in rp


@pytest.mark.skipif(not _MODELS, reason="models not trained")
def test_metrics_and_model_health(client):
    m = client.get("/api/metrics").json()
    assert "cases_total" in m and "by_status" in m
    h = client.get("/api/models/health").json()
    assert h["classifier"]["accuracy"] and h["liquidity"]["mae_days"]
    assert h["classifier"]["status"] in ("good", "warn", "alert", "unknown")


# -- final-audit hardening ----------------------------------------------
@pytest.mark.skipif(not _MODELS, reason="models not trained")
def test_decide_rejects_malformed_body_as_400_not_500(client):
    r = client.post("/decide", data=b"{not json")
    assert r.status_code == 400
    r2 = client.post("/decide", json=["not", "an", "object"])
    assert r2.status_code == 400
    r3 = client.post("/decide", json={"case_id": "c1"})   # missing mandate/failure/history
    assert r3.status_code == 400
    assert "internal_error" not in r3.text               # a KeyError must not fall through as a 500


def test_unhandled_exception_does_not_leak_raw_exception_text(client, monkeypatch):
    import service.app as app_mod
    monkeypatch.setattr(app_mod, "monitoring", type("M", (), {
        "live_metrics": staticmethod(lambda store: (_ for _ in ()).throw(RuntimeError("s3cr3t/internal/path")))
    }))
    # the shared `client` fixture re-raises server exceptions (useful for every
    # other test); this is the one test that needs to see what a real deployed
    # server (uvicorn, not TestClient's debug passthrough) sends the caller.
    local = TestClient(app_mod.app, raise_server_exceptions=False)
    r = local.get("/api/metrics")
    assert r.status_code == 500
    assert "s3cr3t" not in r.text
    body = r.json()
    assert body["error"] == "internal_error"


@pytest.mark.skipif(not _MODELS, reason="models not trained")
def test_api_key_gate_is_off_by_default_and_opt_in(client, monkeypatch):
    monkeypatch.delenv("RECOUP_API_KEY", raising=False)
    assert client.get("/cases").status_code == 200          # default posture: open, hackathon-friendly

    monkeypatch.setenv("RECOUP_API_KEY", "test-key-123")
    try:
        assert client.get("/cases").status_code == 401                                    # no header
        assert client.get("/cases", headers={"Authorization": "Bearer wrong"}).status_code == 401
        assert client.get("/cases", headers={"Authorization": "Bearer test-key-123"}).status_code == 200
        assert client.get("/healthz").status_code == 200    # always open, even with a key set
    finally:
        monkeypatch.delenv("RECOUP_API_KEY", raising=False)


def test_claim_action_is_race_safe(tmp_path):
    from service.store import Store
    s = Store(tmp_path / "claim.db")
    s.upsert_case({"case_id": "c1", "mandate": {"amount": 100.0}, "failure": {"token": "U30_x"}},
                 source="test")
    s.schedule_actions("c1", [{"kind": "retry", "due_at": 0.0}])
    aid = s.due_actions()[0]["id"]
    assert s.claim_action(aid) is True         # first caller wins the race
    assert s.claim_action(aid) is False        # a concurrent second caller does not
    s.close()
