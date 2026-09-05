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

# Razorpay error_code / error_reason -> our observable failure token.
#
# R6 (reconciled): the `error_reason` values below are Razorpay's actual documented
# strings for recurring/eMandate payments, not placeholders —
#   https://razorpay.com/docs/payments/payment-gateway/rainy-day/errors/error-reasons/
#   https://razorpay.com/docs/payments/recurring-payments/emandate/errors/
# `error_code` keeps the four top-level categories Razorpay actually returns
# (BAD_REQUEST_ERROR, GATEWAY_ERROR, SERVER_ERROR) plus the synthetic NPCI-style
# "U30/U67/U88/U69" shorthand our own simulator/tests use as a compact stand-in for
# UPI decline codes (real NPCI UPI RC codes aren't surfaced in Razorpay's webhook body,
# only `error_code`/`error_reason` are — so a UPI-specific mandate would need the same
# `error_reason` mapping, not a different one).
_CODE_TO_TOKEN = {
    # -- synthetic shorthand (simulator / local tests) --
    "BAD_REQUEST_ERROR": "U69_generic_decline",
    "GATEWAY_ERROR": "U88_bank_offline",
    "SERVER_ERROR": "U88_bank_offline",
    "U30": "U30_insufficient_funds",
    "U69": "U69_generic_decline",
    "U88": "U88_bank_offline",
    "U67": "U67_limit_exceeded",
    "U16": "U16_risk_declined",
    # -- real Razorpay `error_reason` values (rainy-day + eMandate error docs) --
    "insufficient_funds": "U30_insufficient_funds",
    "bank_not_available": "U88_bank_offline",
    "bank_technical_error": "U88_bank_offline",
    "bank_cutoff_in_progress": "U88_bank_offline",
    "gateway_technical_error": "U88_bank_offline",
    "issuer_technical_error": "U88_bank_offline",
    "server_error": "U88_bank_offline",
    "transaction_limit_exceeded": "U67_limit_exceeded",
    "transaction_daily_limit_exceeded": "U67_limit_exceeded",
    "transaction_daily_count_exceeded": "U67_limit_exceeded",
    "transaction_frequency_limit_exceeded": "U67_limit_exceeded",
    "credit_limit_exceeded": "U67_limit_exceeded",
    "emi_greater_than_max_amount": "U67_limit_exceeded",
    "payment_limit_exceeded": "U67_limit_exceeded",
    "mandate_not_active": "ER_mandate_revoked",       # bank/customer cancelled the mandate
    "mandate_revoked": "ER_mandate_revoked", "mandate_cancelled": "ER_mandate_revoked",
    "subscription_halted": "ER_mandate_revoked",
    "payment_mandate_not_active": "ER_mandate_issue",  # not yet activated at the bank
    "mandate_creation_declined": "ER_mandate_issue",
    "mandate_creation_expired": "ER_mandate_issue",
    "mandate_creation_failed": "ER_mandate_issue",
    "mandate_creation_timeout": "ER_mandate_issue",
    "bank_account_invalid": "ER_mandate_issue",
    "mandate_not_found": "ER_mandate_issue", "invalid_mandate": "ER_mandate_issue",
    "card_declined": "U69_generic_decline", "payment_declined": "U69_generic_decline",
    "debit_declined": "U69_generic_decline", "payment_failed": "U69_generic_decline",
    "authentication_failed": "U69_generic_decline", "incorrect_otp": "U69_generic_decline",
    "bank_account_validation_failed": "U69_generic_decline",
    "already_declined": "U69_generic_decline", "payment_cancelled": "U69_generic_decline",
    "duplicate_request": "U69_generic_decline", "payment_timed_out": "U69_generic_decline",
    "debit_instrument_blocked": "U69_generic_decline", "debit_instrument_inactive": "U69_generic_decline",
}
_REASON_HINT = {  # error_description substring -> token, checked if the code misses
    "insufficient": "U30_insufficient_funds", "balance": "U30_insufficient_funds",
    "limit": "U67_limit_exceeded", "offline": "U88_bank_offline", "down": "U88_bank_offline",
    "revoked": "ER_mandate_revoked", "cancelled": "ER_mandate_revoked", "halted": "ER_mandate_revoked",
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


def _clean_id(raw, maxlen: int = 128) -> str | None:
    """Defensively normalize an externally-supplied identifier (Razorpay payment /
    subscription id, or anything from webhook JSON) before it becomes a SQLite key,
    a log line, or — via the audit narration — text handed to an LLM prompt: strip
    control characters and cap length. Razorpay ids are short alphanumerics, but the
    webhook body is attacker-shaped input whenever RAZORPAY_WEBHOOK_SECRET is unset
    (local/demo mode), so this never trusts the shape without checking it."""
    if raw is None:
        return None
    s = "".join(ch for ch in str(raw) if ch.isprintable())
    return s[:maxlen] or None


def recovery_ref(event: dict) -> tuple[str | None, str]:
    """For a payment-cleared-style webhook, find which open recovery it belongs to
    and which action kind actually cleared it.

    Prefers the correlation Recoup itself attached when it created a re-auth
    Payment Link (`notes.recoup_case` / `notes.recoup_action`, set in
    `service/executors.py::_reauth`) — a Payment Link's own payment is NOT
    automatically tied to the original subscription/payment id by Razorpay, so
    without this the reauth-driven recovery is invisible to `ingest_webhook_event`.
    Falls back to the subscription/payment id (an ordinary retry or subscription
    auto-charge) when no such notes are present.
    """
    payment = _dig(event, "payload", "payment", "entity", default={}) or {}
    sub = _dig(event, "payload", "subscription", "entity", default={}) or {}
    notes = payment.get("notes") or {}
    tagged_case = _clean_id(notes.get("recoup_case"))
    if tagged_case:
        kind = notes.get("recoup_action") if notes.get("recoup_action") in ("retry", "reauth") else "reauth"
        return tagged_case, kind
    ref = sub.get("id") or payment.get("subscription_id") or payment.get("id")
    return _clean_id(ref), "retry"


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
    case_id = _clean_id(sub_id or payment.get("id")) or f"rzp_{int(now.timestamp())}"

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
