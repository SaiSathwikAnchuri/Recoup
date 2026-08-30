"""Run one or more policies over a generated batch and print the comparison.

    python -m harness.run
    python -m harness.run --data data --seed 42 --policies never_act fixed_schedule always_nudge
    python -m harness.run --trace c0142 --policy fixed_schedule

Writes results/harness_<seed>.json (full outcomes + summaries).
"""

from __future__ import annotations

import argparse
import json
import os

import yaml

from baselines import ALL as BASELINES
from .engine import HarnessConfig, run_batch, run_case
from .metrics import attach_net_value, by_key, exception_list, paired_delta, summarise


def _registry() -> dict:
    """Baselines plus any agent policies whose model is trained."""
    reg = {p.name: p for p in BASELINES}
    try:
        from agent.policies import load_all
        for p in load_all():
            reg[p.name] = p
    except Exception as e:  # missing model, import error — baselines still run
        print(f"  (agent policies unavailable: {e})")
    return reg


def _load_jsonl(path: str) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f]


def _policies(names: list[str] | None):
    reg = _registry()
    if not names:
        return list(reg.values())
    return [reg[n] for n in names]


def _fmt_rs(x) -> str:
    return f"Rs {x:,.0f}" if x is not None else "-"


def print_table(summaries: dict[str, dict]) -> None:
    cols = [
        ("net_value", "NET VALUE", _fmt_rs),
        ("recovered_rs", "RECOVERED", _fmt_rs),
        ("recovery_rate", "RATE", lambda v: f"{v:.1%}"),
        ("attempts", "ATT", str),
        ("attempts_per_1k_recovered", "ATT/1k", lambda v: f"{v:.1f}" if v else "-"),
        ("messages_per_case", "MSG/case", lambda v: f"{v:.2f}"),
        ("on_time_rate", "ON-TIME", lambda v: f"{v:.1%}"),
        ("mandates_preserved_rate", "PRESERVED", lambda v: f"{v:.1%}"),
        ("mandates_revoked", "REVOKED", str),
        ("escalated", "ESC", str),
        ("blocked_actions", "BLOCKED", str),
        ("mean_days_to_recovery", "d->REC", lambda v: f"{v:.1f}" if v else "-"),
    ]
    name_w = max(len(n) for n in summaries) + 2
    header = "POLICY".ljust(name_w) + "".join(h.rjust(11) for _, h, _ in cols)
    print(header)
    print("-" * len(header))
    for name, s in summaries.items():
        row = name.ljust(name_w) + "".join(fn(s[k]).rjust(11) for k, _, fn in cols)
        print(row)


def main() -> None:
    ap = argparse.ArgumentParser(description="Recoup harness runner")
    ap.add_argument("--data", default="data")
    ap.add_argument("--priors", default="config/priors.yaml")
    ap.add_argument("--seed", type=int, default=42, help="world seed for the rollout")
    ap.add_argument("--policies", nargs="*", default=None)
    ap.add_argument("--out", default="results")
    ap.add_argument("--trace", default=None, help="print the full timeline for one case_id and exit")
    ap.add_argument("--policy", default="fixed_schedule", help="policy to use with --trace")
    args = ap.parse_args()

    cases = _load_jsonl(f"{args.data}/cases.jsonl")
    truth = {t["case_id"]: t for t in _load_jsonl(f"{args.data}/truth.jsonl")}
    priors = yaml.safe_load(open(args.priors))
    cfg = HarnessConfig(rev_cfg=priors["revocation"])

    if args.trace:
        case = next(c for c in cases if c["case_id"] == args.trace)
        pol = _registry()[args.policy]
        o = run_case(case, truth[args.trace], pol, cfg, args.seed)
        print(json.dumps({"case_id": o.case_id, "policy": o.policy, "true_cause": o.true_cause,
                          "recovered": o.recovered, "amount_recovered": o.amount_recovered,
                          "revoked": o.revoked, "stop_reason": o.stop_reason,
                          "timeline": o.timeline}, indent=2))
        return

    policies = _policies(args.policies)
    runs = {p.name: run_batch(cases, truth, p, cfg, args.seed) for p in policies}

    from agent.costs import CostModel
    costs = CostModel.from_yaml()
    for outs in runs.values():
        attach_net_value(outs, costs)

    summaries = {name: summarise(outs) for name, outs in runs.items()}

    print(f"\nRecoup harness — {len(cases)} cases, world seed {args.seed}\n")
    print_table(summaries)

    if "fixed_schedule" in runs:
        base = runs["fixed_schedule"]
        print("\nPaired delta vs fixed_schedule  (mean per-case, bootstrap 95% CI)")
        for name, outs in runs.items():
            if name == "fixed_schedule":
                continue
            d_nv = paired_delta(outs, base, "net_value", seed=args.seed)
            d_rs = paired_delta(outs, base, "amount_recovered", seed=args.seed)
            d_mp = paired_delta(outs, base, "mandate_preserved", seed=args.seed)
            print(f"  {name:16s}  net Rs {d_nv['mean']:+9.1f}/case CI {d_nv['ci95']}"
                  f"  (total Rs {d_nv['total']:+,.0f})")
            print(f"  {'':16s}  recovered Rs {d_rs['mean']:+8.1f}/case  mandates "
                  f"{d_mp['mean']:+.3f}/case CI {d_mp['ci95']}")

    os.makedirs(args.out, exist_ok=True)
    payload = {
        "seed": args.seed,
        "n": len(cases),
        "summaries": summaries,
        "by_cause": {name: by_key(outs, "true_cause") for name, outs in runs.items()},
        "by_income": {name: by_key(outs, "income_pattern") for name, outs in runs.items()},
        "exceptions": {name: exception_list(outs) for name, outs in runs.items()},
        "outcomes": {name: [o.as_row() for o in outs] for name, outs in runs.items()},
    }
    with open(f"{args.out}/harness_{args.seed}.json", "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nwrote {args.out}/harness_{args.seed}.json")


if __name__ == "__main__":
    main()
