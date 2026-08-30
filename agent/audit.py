"""Audit trail — every decision the policy made, why, and what happened.

For each case: the structured decision from `EVPolicy.explain()`, joined to the
realised outcome, plus a plain-English narration. Written as JSONL (always) and
SQLite (optional). An LLM can polish the prose (`agent/llm_explain.py`) but the
template narration stands on its own and needs no network.

    python -m agent.audit --seed 42                 # write audit/audit_42.jsonl
    python -m agent.audit --seed 42 --case c0011    # print one case, full detail
    python -m agent.audit --seed 42 --sqlite --llm  # + audit/audit_42.db, + LLM prose
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path

import yaml

from harness.engine import EngagementState, HarnessConfig, run_case
from simulator.calendar_utils import parse_dt

_CAUSE_PHRASE = {
    "insufficient_balance": "the account was short of funds",
    "bank_downtime": "the customer's bank was having an outage",
    "limit_breach": "an AutoPay / transaction limit had been hit",
    "mandate_dead": "the mandate looked revoked or expired",
}
_ACTION_PHRASE = {"retry": "retry", "reauth": "re-authorisation request", "nudge": "nudge", "sms": "SMS"}


def build_record(case: dict, truth_rec: dict, policy, cfg: HarnessConfig, seed: int) -> dict:
    state = EngagementState(observed_at=parse_dt(case["observed_at"]),
                            horizon_days=case["horizon_days"],
                            attempt_budget=cfg.constraints.max_retries)
    decision = policy.explain(case, state)
    outcome = run_case(case, truth_rec, policy, cfg, seed)
    rec = {
        **decision,
        "outcome": {
            "recovered": outcome.recovered,
            "amount_recovered": outcome.amount_recovered,
            "via_reauth": outcome.via_reauth,
            "days_to_recovery": outcome.days_to_recovery,
            "recovered_on_time": outcome.recovered_on_time,
            "revoked": outcome.revoked,
            "escalated": outcome.escalated,
            "attempts_used": outcome.attempts_used,
            "messages_sent": outcome.messages_sent,
            "blocked_actions": outcome.blocked_actions,
            "stop_reason": outcome.stop_reason,
        },
        "true_cause": truth_rec["true_cause"],          # for evaluation only, not shown to the policy
    }
    rec["narration"] = narrate(rec)
    return rec


def narrate(rec: dict) -> str:
    """Deterministic plain-English audit note. Same record in -> same text out."""
    top_cause, p = next(iter(rec["cause_posterior"].items()))
    amt = f"₹{rec['amount']:,.0f}"
    d = rec["decision"]
    fw = rec["funding_window"]
    o = rec["outcome"]

    lines = []
    diag = (f"Diagnosis: most likely {top_cause.replace('_', ' ')} "
            f"({p:.0%} confidence) — {_CAUSE_PHRASE.get(top_cause, 'cause unclear')}.")
    if top_cause == "insufficient_balance":
        diag += (f" Predicted funding window {fw['p50_days']:.0f}–{fw['p85_days']:.0f} days out"
                 f" (billing date at day {rec['cycle_close_day']}).")
    elif top_cause == "limit_breach":
        diag += f" The AutoPay cap resets on day {rec['cycle_close_day'] + 1}."
    lines.append(diag)

    if not d["actions"] and d["terminal"] == "escalate":
        lines.append(f"Decision: no automated action clears the EV floor "
                     f"(₹{rec['ev_floor']:,.0f}) — handed to a human.")
    elif not d["actions"]:
        lines.append(f"Decision: nothing to do — no action has positive expected value.")
    else:
        parts = []
        for a in d["actions"]:
            parts.append(f"{_ACTION_PHRASE.get(a['kind'], a['kind'])} on day {a['day']}")
        tail = " then hand to a human" if d["terminal"] == "escalate" else ""
        best = next((c for c in rec["candidates_top"] if c["action"] != "stop"), None)
        ev = f" (best EV ≈ ₹{best['ev']:,.0f})" if best else ""
        lines.append(f"Decision: {', '.join(parts)}{tail}{ev}.")

    if o["recovered"]:
        how = "re-authorisation" if o["via_reauth"] else "retry"
        when = "on time" if o["recovered_on_time"] else f"late ({o['days_to_recovery']:.0f} days)"
        lines.append(f"Outcome: recovered {amt} by {how}, {when}; "
                     f"{o['attempts_used']} attempt(s), {o['messages_sent']} message(s).")
    elif o["revoked"]:
        lines.append(f"Outcome: the customer revoked the mandate before recovery — "
                     f"{amt} and the mandate lost.")
    elif o["escalated"]:
        lines.append(f"Outcome: unresolved, escalated to a human after "
                     f"{o['attempts_used']} attempt(s).")
    else:
        lines.append(f"Outcome: unresolved after {o['attempts_used']} attempt(s).")
    if o["blocked_actions"]:
        lines.append(f"({o['blocked_actions']} action(s) rescheduled or dropped by the constraint filter.)")
    return " ".join(lines)


# --- persistence -------------------------------------------------------------
def write_jsonl(records: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


_SCHEMA = """
CREATE TABLE IF NOT EXISTS decisions (
    case_id TEXT PRIMARY KEY, observed_at TEXT, amount REAL, failure_code TEXT,
    top_cause TEXT, top_cause_p REAL, window_p50 REAL, window_p85 REAL,
    cycle_close_day INTEGER, terminal TEXT, n_actions INTEGER, note TEXT,
    recovered INTEGER, amount_recovered REAL, via_reauth INTEGER,
    days_to_recovery REAL, on_time INTEGER, revoked INTEGER, escalated INTEGER,
    attempts_used INTEGER, messages_sent INTEGER, blocked_actions INTEGER,
    true_cause TEXT, narration TEXT, record_json TEXT
);
"""


def write_sqlite(records: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()
    con = sqlite3.connect(path)
    con.executescript(_SCHEMA)
    for r in records:
        tc, tcp = next(iter(r["cause_posterior"].items()))
        o = r["outcome"]
        con.execute(
            "INSERT INTO decisions VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (r["case_id"], r["observed_at"], r["amount"], r["failure_code"],
             tc, tcp, r["funding_window"]["p50_days"], r["funding_window"]["p85_days"],
             r["cycle_close_day"], r["decision"]["terminal"], len(r["decision"]["actions"]),
             r["decision"]["note"], int(o["recovered"]), o["amount_recovered"],
             int(o["via_reauth"]), o["days_to_recovery"], int(o["recovered_on_time"]),
             int(o["revoked"]), int(o["escalated"]), o["attempts_used"], o["messages_sent"],
             o["blocked_actions"], r["true_cause"], r["narration"], json.dumps(r)),
        )
    con.commit()
    con.close()


# --- CLI --------------------------------------------------------------------
def _load_jsonl(p: str) -> list[dict]:
    return [json.loads(x) for x in open(p)]


def main() -> None:
    ap = argparse.ArgumentParser(description="Recoup Phase 8 — decision audit trail")
    ap.add_argument("--data", default="data")
    ap.add_argument("--priors", default="config/priors.yaml")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--policy", default="recoup")
    ap.add_argument("--case", default=None, help="print one case in full and exit")
    ap.add_argument("--sqlite", action="store_true")
    ap.add_argument("--llm", action="store_true", help="polish narration with an LLM (needs ANTHROPIC_API_KEY)")
    ap.add_argument("--out", default="audit")
    args = ap.parse_args()

    cases = _load_jsonl(f"{args.data}/cases.jsonl")
    truth = {t["case_id"]: t for t in _load_jsonl(f"{args.data}/truth.jsonl")}
    priors = yaml.safe_load(open(args.priors))
    cfg = HarnessConfig(rev_cfg=priors["revocation"])

    from harness.run import _registry
    policy = _registry()[args.policy]

    if args.case:
        c = next(x for x in cases if x["case_id"] == args.case)
        rec = build_record(c, truth[args.case], policy, cfg, args.seed)
        if args.llm:
            from agent.llm_explain import polish
            rec["narration_llm"] = polish(rec)
        print(json.dumps(rec, indent=2))
        print("\n--- narration ---\n" + rec.get("narration_llm", rec["narration"]))
        return

    records = [build_record(c, truth[c["case_id"]], policy, cfg, args.seed) for c in cases]
    if args.llm:
        from agent.llm_explain import polish
        for r in records:
            r["narration_llm"] = polish(r)

    out = Path(args.out)
    write_jsonl(records, out / f"audit_{args.seed}.jsonl")
    print(f"wrote {out}/audit_{args.seed}.jsonl  ({len(records)} decisions)")
    if args.sqlite:
        write_sqlite(records, out / f"audit_{args.seed}.db")
        print(f"wrote {out}/audit_{args.seed}.db")

    esc = sum(r["decision"]["terminal"] == "escalate" for r in records)
    rec_ok = sum(r["outcome"]["recovered"] for r in records)
    print(f"  {rec_ok} recovered · {esc} escalated · "
          f"{sum(r['outcome']['blocked_actions'] for r in records)} actions filtered")


if __name__ == "__main__":
    main()
