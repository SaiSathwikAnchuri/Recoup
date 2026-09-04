"""Recoup service — FastAPI.

  GET  /                     operator console
  GET  /healthz              readiness + execute mode
  POST /webhook              Razorpay payment.failed -> decision (+ schedule actions)
  POST /demo/random          synthesise a failed mandate -> decision + 45-day simulation
  POST /decide               decision for an arbitrary case body (no persistence)
  GET  /cases                every case seen, newest first
  GET  /cases/{id}           one case: record + decision + scheduled actions
  POST /cases/{id}/execute   run this case's due actions now
  POST /tick                 run every due action across all cases
  GET  /stats                recovered / revoked / escalated / open + recovered rupees
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from . import bridge, executors, loop, monitoring, webhook
from .store import Store

app = FastAPI(title="Recoup", version="1.0")
STORE = Store()
_WEBUI = Path(__file__).resolve().parent / "webui"
app.mount("/vendor", StaticFiles(directory=_WEBUI / "vendor"), name="vendor")


@app.get("/healthz")
def healthz():
    import os
    return {"ok": True, "models_ready": bridge.models_ready(),
            "execute_mode": os.environ.get("RECOUP_EXECUTE_MODE", "dry_run")}


@app.get("/")
def index():
    return FileResponse(_WEBUI / "index.html")


@app.get("/app.js")
def _appjs():
    return FileResponse(_WEBUI / "app.js", media_type="application/javascript")


@app.get("/data.js")
def _datajs():
    return FileResponse(_WEBUI / "data.js", media_type="application/javascript")


@app.get("/api/results")
def api_results():
    """The Overview's numbers, live from the last run if results/ is populated,
    else the baked snapshot the front-end ships with."""
    import json
    root = Path.cwd()
    h = root / "results" / "harness_42.json"
    p = root / "results" / "phase9.json"
    if not (h.exists() and p.exists()):
        raise HTTPException(404, "run `python tasks.py reproduce` to populate results/")
    hs = json.loads(h.read_text())["summaries"]
    pj = json.loads(p.read_text())

    def sbrow(name, k):
        s = hs[k]
        return {"name": name, "recovered": round(s["recovered_rs"]), "rate": s["recovery_rate"],
                "attempts": s["attempts"], "msg": s["messages_per_case"],
                "on_time": s["on_time_rate"], "preserved": s["mandates_preserved_rate"],
                "net": round(s["net_value"])}

    ab = pj["ablation"]["table"]
    return {
        "headline": {"per_case": round(ab["recoup"]["delta_vs_fixed_per_case"]),
                     "ci": [round(x) for x in ab["recoup"]["delta_ci95"]], "n": pj["n"]},
        "recoup_minus_oracle": round(pj["ablation"]["recoup_net_vs_oracle_net"]),
        "scoreboard": [sbrow(n, k) for n, k in [
            ("never act", "never_act"), ("fixed schedule", "fixed_schedule"),
            ("always nudge", "always_nudge"), ("+ cause classifier", "cause_aware"),
            ("+ funding-window timing", "liquidity_aware"), ("Recoup", "recoup")]],
        "ablation": [{"name": n.replace("_", " "), "net": round(v["delta_vs_fixed_per_case"]),
                      "gap": v["pct_of_recovery_gap_closed"], "recovery": v["recovery_rate"],
                      "on_time": v["on_time_rate"]}
                     for n, v in ab.items() if n not in ("never_act", "fixed_schedule")],
        "sensitivity": [{"name": n.replace("_", " ").replace("x1.8", "×1.8"),
                         "v": round(r["delta_net_per_case"]),
                         "lo": round(r["ci95"][0]), "hi": round(r["ci95"][1])}
                        for n, r in pj["sensitivity"].items()],
        "fairness": [{"group": g.title(), "n": v["n"], "recovery": v["recovery_rate"],
                      "vs_fixed": v["recovery_rate_vs_fixed"], "on_time": v["on_time_rate"],
                      "escalated": v["escalated_rate"],
                      "net": round(v["net_delta_per_case_vs_fixed"])}
                     for g, v in pj["fairness"]["by_income"].items()],
        "classifier": {"acc": 0.90, "ece": 0.047}, "liquidity": {"mae": 6.1, "naive": 11.8},
    }


def _require_models():
    if not bridge.models_ready():
        raise HTTPException(503, "models not trained — run `python tasks.py train`")


# ---------------------------------------------------------------------------
def _dedup_key(raw: bytes, event: dict) -> str:
    import hashlib
    pay = (((event.get("payload") or {}).get("payment") or {}).get("entity") or {})
    return (event.get("id") or pay.get("id")
            or hashlib.sha256(raw or b"").hexdigest()[:32]) + ":" + str(event.get("event", ""))


@app.post("/webhook")
async def rzp_webhook(request: Request,
                      x_razorpay_signature: str | None = Header(default=None)):
    _require_models()
    raw = await request.body()
    if not webhook.verify_signature(raw, x_razorpay_signature):
        raise HTTPException(400, "bad signature")
    try:
        event = json.loads(raw or b"{}")
    except ValueError:
        raise HTTPException(400, "malformed JSON body")
    if not isinstance(event, dict):
        raise HTTPException(400, "event must be a JSON object")

    res = loop.ingest_webhook_event(STORE, event, dedup_key=_dedup_key(raw, event))
    return res


@app.post("/demo/random")
def demo_random(cause: str | None = None, seed: int | None = None):
    _require_models()
    from service.state import CustomerState
    case, truth = bridge.random_case(seed=seed, cause=cause)

    from service.events import Event, EventType
    STORE.append_event(Event(EventType.PAYMENT_FAILED, case["case_id"], case["case_id"],
                             case["case_id"],
                             payload={"failure_code": case["failure"].get("token"),
                                      "amount": float(case["mandate"]["amount"]),
                                      "age_months": case["mandate"].get("age_months"),
                                      "observed_at": case["observed_at"]}))
    cstate = CustomerState.rebuild(STORE.events_for_customer(case["case_id"]), base_case=case)
    est = loop._engagement_state(case, STORE.events_for_customer(case["case_id"]))
    decision = bridge.decide(case, customer_state=cstate, engagement=est)
    sim = bridge.simulate(case, truth, seed=seed or 42)

    STORE.upsert_case(case, source="demo",
                      status=("recovered" if sim["recoup"]["recovered"]
                              else "revoked" if sim["recoup"]["revoked"]
                              else "escalated" if sim["recoup"]["escalated"] else "open"))
    STORE.add_decision(case["case_id"], decision)
    loop.log_simulation(STORE, case, sim)
    return {"case_id": case["case_id"], "case": case, "decision": decision,
            "customer_state": cstate.to_dict(), "simulation": sim}


@app.post("/decide")
async def decide_body(request: Request):
    _require_models()
    case = json.loads(await request.body())
    return {"decision": bridge.decide(case)}


@app.get("/cases")
def cases(limit: int = 100):
    return {"cases": STORE.list_cases(limit)}


@app.get("/cases/{case_id}")
def one_case(case_id: str):
    c = STORE.get_case(case_id)
    if not c:
        raise HTTPException(404, "unknown case")
    return {"case": c, "decision": STORE.latest_decision(case_id),
            "actions": STORE.actions_for(case_id),
            "customer_state": STORE.get_state(case_id),
            "outcomes": STORE.outcomes_for(case_id)}


_TL_PHRASE = {
    "PAYMENT_FAILED": "AutoPay debit failed",
    "RETRY_SCHEDULED": "retry scheduled",
    "RETRY_EXECUTED": "retry executed",
    "REAUTH_CREATED": "re-auth link created",
    "REAUTH_COMPLETED": "re-auth completed",
    "MESSAGE_SENT": "customer message sent",
    "PAYMENT_RECOVERED": "payment recovered",
    "MANDATE_REVOKED": "mandate revoked",
    "RECOVERY_STOPPED": "recovery stopped",
    "RECOVERY_ESCALATED": "escalated to a human",
}


@app.get("/api/cases/{case_id}/timeline")
def case_timeline(case_id: str):
    if not STORE.get_case(case_id):
        raise HTTPException(404, "unknown case")
    evs = STORE.events_for_case(case_id)
    return {"case_id": case_id, "events": [
        {"at": e.at, "type": e.type.value, "label": _TL_PHRASE.get(e.type.value, e.type.value),
         "payload": e.payload} for e in evs]}


@app.get("/api/cases/{case_id}/decision")
def case_decision(case_id: str):
    d = STORE.latest_decision(case_id)
    if not d:
        raise HTTPException(404, "no decision for this case")
    return d


@app.post("/api/cases/{case_id}/replan")
def case_replan(case_id: str):
    _require_models()
    from service.state import CustomerState
    case = STORE.get_case(case_id)
    if not case:
        raise HTTPException(404, "unknown case")
    cstate = CustomerState.rebuild(STORE.events_for_customer(case_id), base_case=case)
    scheduled = loop._plan_and_schedule(STORE, case, cstate)
    return {"case_id": case_id, "customer_state": cstate.to_dict(),
            "decision": STORE.latest_decision(case_id), "scheduled": scheduled}


@app.get("/api/metrics")
def api_metrics():
    return monitoring.live_metrics(STORE)


@app.get("/api/models/health")
def api_models_health():
    return monitoring.model_health()


def _run_due(rows: list[dict]) -> list[dict]:
    done = []
    for a in rows:
        case = STORE.get_case(a["case_id"])
        if case is None:
            STORE.mark_action(a["id"], "skipped", {"error": "case gone"})
            continue
        res = executors.execute(a["kind"], case)
        STORE.mark_action(a["id"], "done", res)
        if a["kind"] == "escalate":
            STORE.escalate(a["case_id"], "escalate action executed")
        else:
            loop.note_execution(STORE, a["case_id"], a["kind"])
        done.append({"case_id": a["case_id"], **res})
    return done


@app.post("/cases/{case_id}/execute")
def execute_case(case_id: str):
    rows = [a for a in STORE.due_actions(now=time.time() + 10**9)
            if a["case_id"] == case_id]
    return {"executed": _run_due(rows)}


@app.post("/tick")
def tick():
    return {"executed": _run_due(STORE.due_actions())}


@app.get("/stats")
def stats():
    return STORE.stats()


@app.exception_handler(Exception)
async def _err(request: Request, exc: Exception):
    return JSONResponse(status_code=500, content={"error": type(exc).__name__, "detail": str(exc)})
