"""Action executors — what actually happens when Recoup decides to act.

Two modes:
  dry_run        (default) record the intent, do nothing external. Safe everywhere.
  razorpay_test  call the Razorpay API in TEST MODE, when RAZORPAY_KEY_ID /
                 RAZORPAY_KEY_SECRET are set. Only the re-auth link is wired to a
                 real endpoint (Payment Links); retries and messages are logged
                 intent — a production build routes them to the subscriptions API
                 and an SMS / WhatsApp provider.

Never runs against live keys unless the operator sets RECOUP_EXECUTE_MODE=razorpay_test.
"""

from __future__ import annotations

import base64
import json
import os
import urllib.error
import urllib.request

_RZP_API = "https://api.razorpay.com/v1"


def _mode() -> str:
    return os.environ.get("RECOUP_EXECUTE_MODE", "dry_run")


def _rzp_auth() -> str | None:
    kid, secret = os.environ.get("RAZORPAY_KEY_ID"), os.environ.get("RAZORPAY_KEY_SECRET")
    if kid and secret and kid.startswith("rzp_test_"):        # test keys only
        return base64.b64encode(f"{kid}:{secret}".encode()).decode()
    return None


def _rzp_post(path: str, body: dict) -> dict:
    auth = _rzp_auth()
    if not auth:
        return {"ok": False, "error": "no Razorpay test key configured"}
    req = urllib.request.Request(f"{_RZP_API}{path}", data=json.dumps(body).encode(),
                                 method="POST", headers={
                                     "Authorization": f"Basic {auth}",
                                     "Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=15) as r:
            return {"ok": True, "response": json.loads(r.read())}
    except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, OSError) as e:
        return {"ok": False, "error": str(e)}


# --- the executors ---------------------------------------------------------
def execute(kind: str, case: dict, *, mode: str | None = None) -> dict:
    mode = mode or _mode()
    amount = float(case["mandate"]["amount"])
    fn = {"retry": _retry, "reauth": _reauth, "sms": _message, "nudge": _message,
          "escalate": _escalate}.get(kind, _noop)
    result = fn(case, amount, mode)
    return {"kind": kind, "mode": mode, **result}


def _retry(case, amount, mode):
    intent = f"re-attempt the ₹{amount:,.0f} AutoPay debit for {case['case_id']}"
    if mode == "razorpay_test":
        # a real build triggers the next charge on the subscription; test mode logs intent
        return {"action": "retry_debit", "intent": intent, "executed": False,
                "note": "wire to POST /subscriptions/{id} charge in production"}
    return {"action": "retry_debit", "intent": intent, "executed": False, "note": "dry run"}


def _reauth(case, amount, mode):
    intent = f"send {case['case_id']} a 1-tap link to re-authorise the mandate and clear ₹{amount:,.0f}"
    if mode == "razorpay_test":
        r = _rzp_post("/payment_links", {
            "amount": int(round(amount * 100)), "currency": "INR",
            "description": "Re-authorise your subscription and clear the pending payment",
            "notes": {"recoup_case": case["case_id"], "recoup_action": "reauth"},
            "reminder_enable": True,
        })
        return {"action": "payment_link", "intent": intent,
                "executed": r["ok"],
                "link": (r.get("response") or {}).get("short_url"),
                "error": r.get("error")}
    return {"action": "payment_link", "intent": intent, "executed": False, "note": "dry run"}


def _message(case, amount, mode):
    return {"action": "customer_message", "executed": False,
            "intent": f"notify {case['case_id']} — payment of ₹{amount:,.0f} pending",
            "note": "route to an SMS / WhatsApp provider in production"}


def _escalate(case, amount, mode):
    return {"action": "escalate", "executed": True,
            "intent": f"queue {case['case_id']} for a human collections agent"}


def _noop(case, amount, mode):
    return {"action": "noop", "executed": False}
