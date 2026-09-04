"""Phase 9 — the evidence: oracle ceiling, ablation ladder, prior sensitivity,
fairness slice.

    python -m experiments.phase9              # full run, writes results/phase9.json
    python -m experiments.phase9 --quick      # skip the sensitivity sweep

The trained models are held FIXED throughout (they learned the baseline priors).
The sensitivity sweep perturbs only the *world* — the honest test is whether the
policy still wins when reality differs from what the models were trained on.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import yaml

from agent.ablations import no_cause_policy, no_timing_policy
from agent.costs import CostModel
from agent.ev_policy import EVPolicy
from agent.policies import LiquidityAwareRetry
from baselines import FixedSchedule, NeverAct
from harness.engine import HarnessConfig, run_batch
from harness.metrics import attach_net_value, by_key, paired_delta, summarise
from harness.oracle import Oracle
from simulator.generate import build_batch

RESULTS = Path("results")


def _load(p):
    return [json.loads(x) for x in open(p)]


def _run(policy, cases, truth, cfg, costs, seed):
    outs = run_batch(cases, truth, policy, cfg, seed)
    attach_net_value(outs, costs)
    return outs


# ---------------------------------------------------------------------------
# A. ablation ladder + oracle
# ---------------------------------------------------------------------------
def ablation_ladder(cases, truth, cfg, costs, seed) -> dict:
    truth_by_id = truth
    ladder = [
        NeverAct(), FixedSchedule(),
        no_timing_policy(), no_cause_policy(),
        LiquidityAwareRetry(),          # = recoup without the cost/EV layer
        EVPolicy(),                     # recoup
        Oracle(truth_by_id),
    ]
    runs = {p.name: _run(p, cases, truth, cfg, costs, seed) for p in ladder}
    summ = {name: summarise(o) for name, o in runs.items()}

    base = runs["fixed_schedule"]
    # the oracle has perfect timing but optimises RECOVERY, not net value — so it
    # is the ceiling for recovery rate. recoup can (and does) exceed it on net
    # value by trading a little raw recovery for beating the billing date.
    o_rec = summ["oracle"]["recovery_rate"]
    f_rec = summ["fixed_schedule"]["recovery_rate"]
    rec_gap = o_rec - f_rec
    table = {}
    for name, o in runs.items():
        d = paired_delta(o, base, "net_value", seed=seed)
        table[name] = {
            "net_value": summ[name]["net_value"],
            "recovery_rate": summ[name]["recovery_rate"],
            "on_time_rate": summ[name]["on_time_rate"],
            "preserved_rate": summ[name]["mandates_preserved_rate"],
            "attempts": summ[name]["attempts"],
            "delta_vs_fixed_per_case": d["mean"],
            "delta_ci95": d["ci95"],
            "pct_of_recovery_gap_closed": round(100 * (summ[name]["recovery_rate"] - f_rec) / rec_gap, 1)
            if rec_gap else None,
        }
    return {"table": table, "recovery_gap_pts": round(100 * rec_gap, 1),
            "recoup_net_vs_oracle_net": round(summ["recoup"]["net_value"] - summ["oracle"]["net_value"], 0)}


# ---------------------------------------------------------------------------
# B. prior sensitivity
# ---------------------------------------------------------------------------
def _perturbations(priors: dict) -> dict[str, dict]:
    out = {}

    p = copy.deepcopy(priors)
    p["cause_mix"] = {"insufficient_balance": 0.45, "bank_downtime": 0.20,
                      "limit_breach": 0.15, "mandate_dead": 0.20}
    out["fewer_cashflow_more_dead"] = p

    p = copy.deepcopy(priors)
    for k in p["revocation"]["h_base_daily"]:
        p["revocation"]["h_base_daily"][k] *= 1.8
    p["revocation"]["fatigue_multiplier"] = 1.7
    out["churn_hazard_x1.8"] = p

    p = copy.deepcopy(priors)
    p["ltv"]["months"] = 6
    p["ltv"]["retention_factor_range"] = [0.4, 0.8]
    out["low_ltv"] = p

    p = copy.deepcopy(priors)
    p["bank_downtime"]["outage_mean_hours"] = 18.0
    p["bank_downtime"]["recurring_outage_prob"] = 0.35
    out["long_outages"] = p

    p = copy.deepcopy(priors)
    p["insufficient_balance"]["unrecoverable_frac"] = 0.25
    p["insufficient_balance"]["shortfall_frac_range"] = [0.10, 0.95]
    out["deeper_shortfalls"] = p

    return out


def sensitivity(priors: dict, n: int, seed: int, cfg: HarnessConfig, costs: CostModel) -> dict:
    rows = {}
    variants = {"baseline_priors": priors, **_perturbations(priors)}
    for name, pri in variants.items():
        c, t = build_batch(n, seed + 999, pri)          # a different draw than the training/eval batch
        truth = {x["case_id"]: x for x in t}
        rc = _run(EVPolicy(), c, truth, cfg, costs, seed)
        fx = _run(FixedSchedule(), c, truth, cfg, costs, seed)
        d = paired_delta(rc, fx, "net_value", seed=seed)
        s_rc, s_fx = summarise(rc), summarise(fx)
        rows[name] = {
            "recoup_net": s_rc["net_value"], "fixed_net": s_fx["net_value"],
            "recoup_recovery": s_rc["recovery_rate"], "fixed_recovery": s_fx["recovery_rate"],
            "delta_net_per_case": d["mean"], "ci95": d["ci95"],
            "recoup_wins": bool(d["ci95"][0] > 0),
        }
    return rows


# ---------------------------------------------------------------------------
# C. fairness
# ---------------------------------------------------------------------------
def fairness(cases, truth, cfg, costs, seed) -> dict:
    rc = _run(EVPolicy(), cases, truth, cfg, costs, seed)
    fx = _run(FixedSchedule(), cases, truth, cfg, costs, seed)
    g_rc, g_fx = by_key(rc, "income_pattern"), by_key(fx, "income_pattern")
    rc_by = {o.case_id: o for o in rc}
    fx_by = {o.case_id: o for o in fx}

    groups = {}
    for pat in g_rc:
        ids = [cid for cid, o in rc_by.items() if o.income_pattern == pat]
        rcg = [rc_by[i] for i in ids]
        fxg = [fx_by[i] for i in ids]
        d = paired_delta(rcg, fxg, "net_value", seed=seed)
        groups[pat] = {
            "n": g_rc[pat]["n"],
            "recovery_rate": g_rc[pat]["recovery_rate"],
            "on_time_rate": g_rc[pat]["on_time_rate"],
            "preserved_rate": g_rc[pat]["preserved_rate"],
            "escalated_rate": g_rc[pat]["escalated_rate"],
            "recovery_rate_vs_fixed": round(g_rc[pat]["recovery_rate"] - g_fx[pat]["recovery_rate"], 3),
            "net_delta_per_case_vs_fixed": d["mean"],
            "net_delta_ci95": d["ci95"],
        }
    rec_rates = [v["recovery_rate"] for v in groups.values()]
    esc_rates = [v["escalated_rate"] for v in groups.values()]
    return {
        "by_income": groups,
        "recovery_rate_disparity": round(max(rec_rates) - min(rec_rates), 3),
        "escalation_rate_disparity": round(max(esc_rates) - min(esc_rates), 3),
        "every_group_gains_vs_fixed": all(v["net_delta_per_case_vs_fixed"] > 0 for v in groups.values()),
        "every_group_gain_significant": all(v["net_delta_ci95"][0] > 0 for v in groups.values()),
    }


# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description="Recoup Phase 9 — evidence")
    ap.add_argument("--data", default="data")
    ap.add_argument("--priors", default="config/priors.yaml")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--quick", action="store_true", help="skip the sensitivity sweep")
    ap.add_argument("--out", default="results/phase9.json")
    args = ap.parse_args()

    cases = _load(f"{args.data}/cases.jsonl")
    truth = {t["case_id"]: t for t in _load(f"{args.data}/truth.jsonl")}
    priors = yaml.safe_load(open(args.priors))
    cfg = HarnessConfig(rev_cfg=priors["revocation"])
    costs = CostModel.from_yaml()

    report = {"seed": args.seed, "n": len(cases)}

    print("\n=== A. ablation ladder (net value, paired vs fixed_schedule) ===\n")
    report["ablation"] = ablation_ladder(cases, truth, cfg, costs, args.seed)
    print(f"  {'policy':16s} {'net value':>12} {'recov':>7} {'on-time':>8} "
          f"{'preserved':>10} {'net d/case':>11}  {'% recov gap':>11}")
    for name, r in report["ablation"]["table"].items():
        print(f"  {name:16s} {r['net_value']:>12,.0f} {r['recovery_rate']:>7.1%} "
              f"{r['on_time_rate']:>8.1%} {r['preserved_rate']:>10.1%} "
              f"{r['delta_vs_fixed_per_case']:>+11.0f}  {str(r['pct_of_recovery_gap_closed']):>11}")
    ab = report["ablation"]
    print(f"\n  oracle = perfect-timing recovery ceiling ({ab['recovery_gap_pts']:.0f} pts over fixed).")
    print(f"  recoup exceeds the oracle's NET value by Rs {ab['recoup_net_vs_oracle_net']:,.0f} "
          f"— it trades raw recovery for beating the billing date.")

    print("\n=== C. fairness by income pattern ===\n")
    report["fairness"] = fairness(cases, truth, cfg, costs, args.seed)
    for k, v in report["fairness"]["by_income"].items():
        print(f"  {k:10s}  n={v['n']:3d}  recovery {v['recovery_rate']:.1%} "
              f"({v['recovery_rate_vs_fixed']:+.1%} vs fixed)  on-time {v['on_time_rate']:.1%}  "
              f"escalated {v['escalated_rate']:.1%}  net d/case Rs {v['net_delta_per_case_vs_fixed']:+,.0f} "
              f"CI {v['net_delta_ci95']}")
    fr = report["fairness"]
    print(f"\n  recovery-rate disparity {fr['recovery_rate_disparity']:.3f} · "
          f"escalation disparity {fr['escalation_rate_disparity']:.3f} · "
          f"every group gains vs fixed: {fr['every_group_gains_vs_fixed']} "
          f"(significant: {fr['every_group_gain_significant']})")

    if not args.quick:
        print("\n=== B. prior sensitivity (recoup vs fixed_schedule, fresh worlds) ===\n")
        report["sensitivity"] = sensitivity(priors, args.n, args.seed, cfg, costs)
        for name, r in report["sensitivity"].items():
            flag = "OK" if r["recoup_wins"] else "  <-- check"
            print(f"  {name:26s}  d-net Rs {r['delta_net_per_case']:>+9.0f}/case  "
                  f"CI {r['ci95']}  {flag}")
        report["sensitivity_ordering_holds"] = all(r["recoup_wins"] for r in report["sensitivity"].values())
        print(f"\n  ordering holds under every perturbation: {report['sensitivity_ordering_holds']}")

    RESULTS.mkdir(exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2, default=str))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
