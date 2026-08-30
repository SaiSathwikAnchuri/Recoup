"""Optional LLM polish for the audit narration.

The LLM's ONLY job in Recoup is prose: turn a structured decision record into a
2–3 sentence audit note that reads naturally. It never makes or influences a
decision — every number it sees was already computed by the policy.

Off by default. Enabled only with `--llm` AND an `ANTHROPIC_API_KEY` in the
environment. Every call is cached to `audit/llm_cache.json` keyed by a hash of
the input, so a batch costs one call per distinct decision shape and `reproduce`
never needs the network. No SDK dependency — a plain HTTPS POST.
"""

from __future__ import annotations

import hashlib
import json
import os
import urllib.error
import urllib.request
from pathlib import Path

_CACHE_PATH = Path(__file__).resolve().parent.parent / "audit" / "llm_cache.json"
_MODEL = "claude-haiku-4-5-20251001"
_ENDPOINT = "https://api.anthropic.com/v1/messages"

_SYSTEM = (
    "You write terse, factual audit notes for an automated payment-recovery agent. "
    "You are given a structured decision record (already computed) and its outcome. "
    "Write 2-3 sentences, past tense, no marketing language, no speculation beyond the "
    "record. State: the diagnosis and confidence, the action taken and the one-line "
    "reason, and what happened. Do not invent numbers."
)


def _load_cache() -> dict:
    try:
        return json.loads(_CACHE_PATH.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def _save_cache(cache: dict) -> None:
    _CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    _CACHE_PATH.write_text(json.dumps(cache, indent=2), encoding="utf-8")


def _key(rec: dict) -> str:
    payload = {k: rec[k] for k in ("cause_posterior", "funding_window", "cycle_close_day",
                                   "decision", "outcome") if k in rec}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]


def _call_api(rec: dict) -> str:
    api_key = os.environ["ANTHROPIC_API_KEY"]
    user = ("Decision record:\n"
            + json.dumps({k: rec[k] for k in
                          ("case_id", "amount", "failure_code", "cause_posterior",
                           "funding_window", "cycle_close_day", "ltv_estimate",
                           "candidates_top", "decision", "outcome") if k in rec}, indent=2)
            + "\n\nWrite the audit note.")
    body = json.dumps({
        "model": _MODEL, "max_tokens": 200,
        "system": _SYSTEM,
        "messages": [{"role": "user", "content": user}],
    }).encode()
    req = urllib.request.Request(_ENDPOINT, data=body, method="POST", headers={
        "content-type": "application/json",
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
    })
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = json.loads(resp.read())
    return "".join(block.get("text", "") for block in data.get("content", [])).strip()


def polish(rec: dict, *, force: bool = False) -> str:
    """Return an LLM-written narration, or fall back to the template `rec['narration']`.
    Falls back silently on any error (no key, network, rate limit)."""
    base = rec.get("narration", "")
    if not force and not os.environ.get("ANTHROPIC_API_KEY"):
        return base

    cache = _load_cache()
    k = _key(rec)
    if k in cache:
        return cache[k]
    try:
        text = _call_api(rec)
    except (KeyError, urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError):
        return base
    if not text:
        return base
    cache[k] = text
    _save_cache(cache)
    return text
