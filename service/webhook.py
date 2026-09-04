"""Razorpay webhook -> a Recoup `case`.

Accepts the shape Razorpay POSTs for a failed recurring charge
(`payment.failed`, `subscription.charged` with a failed payment, `subscription.pending`).
Extracts what it can from the event; fills the observable history fields from the
store's record of prior charges on that subscription, or sensible defaults.

Signature check: Razorpay signs the raw body with HMAC-SHA256 using the endpoint's
webhook secret (`X-Razorpay-Signature`). Enforced when `RAZORPAY_WEBHOOK_SECRET`
is set; skipped with a warning otherwise (local / demo).
"""

from __future__ import annotations

import hashlib
import hmac
import os
from datetime import datetime, timezone

# Razorpay / NPCI decline code -> our observable failure token.
# PLACEHOLDER mapping — reconcile with the real Razorpay Subscriptions error list (report R6).
_CODE_TO_TOKEN = {
    "BAD_REQUEST_ERROR": "U69_generic_decline",
    "GATEWAY_ERROR": "U88_bank_offline",
    "U30": "U30_insufficient_funds", "insufficient_funds": "U30_insufficient_funds",
    "U69": "U69_generic_decline",
    "U88": "U88_bank_offline",
    "U67": "U67_limit_exceeded", "payment_limit_exceeded": "U67_limit_exceeded",
    "U16": "U16_risk_declined",
    "mandate_revoked": "ER_mandate_revoked", "mandate_cancelled": "ER_mandate_revoked",
    "subscription_halted": "ER_mandate_revoked",
    "mandate_not_found": "ER_mandate_issue", "invalid_mandate": "ER_mandate_issue",
}
_REASON_HINT = {  # error_description substring -> token, checked if the code misses
    "insufficient": "U30_insufficient_funds", "balance": "U30_insufficient_funds",
    "limit": "U67_limit_exceeded", "offline": "U88_bank_offline", "down": "U88_bank_offline",
    "revoked": "ER_mandate_revoked", "cancelled": "ER_mandate_revoked",
    "not found": "ER_mandate_issue", "not active": "ER_mandate_issue", "expired": "ER_mandate_issue",
}
IST = timezone.utc  # observed_at is stored with an offset; the engine only uses the date/day-of-month


def verify_signature(raw_body: bytes, signature: str | None) -> bool:
    secret = os.environ.get("RAZORPAY_WEBHOOK_SECRET")
    if not secret:
        return True  # local / demo mode — no secret configured
    if not signature:
        return False
    expected = hmac.new(secret.encode(), raw_body, hashlib.sha256).hexdigest()
    return hmac.compare_digest(expected, signature)


def _token_from(code: str | None, description: str | None) -> str:
    if code and code in _CODE_TO_TOKEN:
        return _CODE_TO_TOKEN[code]
    text = (description or "").lower()
    for hint, tok in _REASON_HINT.items():
        if hint in text:
            return tok
    return "U69_generic_decline"


def _dig(d: dict, *path, default=None):
    for k in path:
        if not isinstance(d, dict) or k not in d:
            return default
        d = d[k]
    return d


def parse_event(event: dict, *, history: dict | None = None,
                now: datetime | None = None) -> dict:
    """Map a Razorpay webhook `event` dict to a Recoup observable case."""
    now = now or datetime.now(timezone.utc)
    payment = _dig(event, "payload", "payment", "entity", default={}) or {}
    sub = _dig(event, "payload", "subscription", "entity", default={}) or {}

    amount_paise = payment.get("amount") or sub.get("amount") or 0
    amount = round(float(amount_paise) / 100.0, 2) if amount_paise else 0.0
    code = payment.get("error_code") or payment.get("error_source") or event.get("event")
    desc = payment.get("error_description") or payment.get("error_reason")
    token = _token_from(code, desc)

    sub_id = sub.get("id") or payment.get("subscription_id") or _dig(payment, "notes", "subscription_id")
    case_id = sub_id or payment.get("id") or f"rzp_{int(now.timestamp())}"

    created = sub.get("created_at")
    age_months = 1
    if created:
        age_months = max(1, int((now.timestamp() - float(created)) / (30 * 86400)))

    h = history or {}
    return {
        "case_id": str(case_id),
        "observed_at": now.isoformat(),
        "failure": {
            "token": token,
            "npci_code": str(code) if code else "",
            "error_code": payment.get("error_code", "BAD_REQUEST_ERROR"),
            "error_reason": payment.get("error_reason", "payment_failed"),
            "description": desc or "",
        },
        "mandate": {
            "category": (payment.get("notes") or {}).get("category", "sip"),
            "amount": amount,
            "debit_day": now.day,
            "age_months": age_months,
            "authorised_ceiling": round(amount * 1.5) if amount else 0,
            "retries_used_this_cycle": int(h.get("retries_used_this_cycle", 0)),
            "created_at": datetime.fromtimestamp(float(created), timezone.utc).isoformat()
                          if created else now.isoformat(),
        },
        "history": {
            "consecutive_successes": int(h.get("consecutive_successes", 0)),
            "last_success": h.get("last_success"),
            "days_since_last_success": int(h.get("days_since_last_success", 30)),
            "success_days_of_month": list(h.get("success_days_of_month", [])),
            "cycles_observed": int(h.get("cycles_observed", max(age_months - 1, 0))),
        },
        "horizon_days": 45,
    }
