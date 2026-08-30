"""Human-readable dump of one generated case + its hidden truth.

    python -m simulator.show c0001
    python -m simulator.show c0001 --ledger      # also print the cash ledger

Handy during development and as raw material for the demo decision trace.
"""

from __future__ import annotations

import argparse
import json

from .calendar_utils import parse_dt


def _load(path: str) -> dict:
    out = {}
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            out[r["case_id"]] = r
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="show one Recoup case")
    ap.add_argument("case_id")
    ap.add_argument("--data", default="data")
    ap.add_argument("--ledger", action="store_true")
    args = ap.parse_args()

    cases = _load(f"{args.data}/cases.jsonl")
    truth = _load(f"{args.data}/truth.jsonl")
    if args.case_id not in cases:
        raise SystemExit(f"no such case: {args.case_id}")
    c, t = cases[args.case_id], truth[args.case_id]
    m, h = c["mandate"], c["history"]

    print(f"\n  {c['case_id']}   observed {c['observed_at']}   horizon {c['horizon_days']}d")
    print("  " + "-" * 66)
    print("  OBSERVABLE (what the agent sees)")
    print(f"    failure code : {c['failure']['token']}  [{c['failure']['npci_code']}]  {c['failure']['description']}")
    print(f"    mandate      : Rs {m['amount']:.0f}  {m['category']}  debit_day={m['debit_day']}  "
          f"age={m['age_months']}mo  ceiling=Rs {m['authorised_ceiling']:.0f}")
    print(f"    history      : {h['consecutive_successes']}/{h['cycles_observed']} consecutive  "
          f"last_success={h['last_success']}  ({h['days_since_last_success']}d ago)")
    print(f"    funding days : {h['success_days_of_month']}")
    print()
    print("  HIDDEN TRUTH (harness only)")
    print(f"    true cause   : {t['true_cause']}   (cashflow_tight={t['cashflow_tight']})")
    print(f"    customer     : {t['income_pattern']}  income~Rs {t['monthly_income']:.0f}/mo  "
          f"pay_days={t['pay_days']}  h_base={t['h_base']}")
    print(f"    competing    : {t['competing_debits']}")
    print(f"    balance@obs  : Rs {t['balance_at_observed']:.0f}   opening=Rs {t['opening_balance']:.0f}")
    print(f"    day-0 P(succ): {t['day0_p_success']:.3f}")
    print(f"    recoverable  : {t['retry_recoverable_within_horizon']}   "
          f"best retry P={t['best_retry_p']:.2f} @ {t['best_retry_at']}")
    print(f"    LTV (true)   : Rs {t['ltv_true']:.0f}")
    cp = t["cause_params"]
    if cp["outages"]:
        print(f"    outages      : {cp['outages']}")
    if cp["limit_reset_dt"]:
        print(f"    limit reset  : {cp['limit_reset_dt']}")
    if t["true_cause"] == "mandate_dead":
        print(f"    reauth P     : {cp['reauth_success_prob']:.2f}")

    if args.ledger:
        print("\n  LEDGER (running balance)")
        bal = t["opening_balance"]
        print(f"    {'opening':<26} {'':>12}  Rs {bal:>12.0f}")
        for (ts, amt, kind) in t["ledger"]:
            bal += amt
            print(f"    {ts:<26} {amt:>12.0f}  Rs {bal:>12.0f}   {kind}")
    print()


if __name__ == "__main__":
    main()
