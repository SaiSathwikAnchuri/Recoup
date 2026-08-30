"""CLI: generate a batch of synthetic failed-mandate cases.

    python -m simulator.generate --n 400 --seed 42 --out data

Writes:
  data/cases.jsonl    one observable record per line   (all the agent sees)
  data/truth.jsonl    one hidden-truth record per line  (harness only)
  data/summary.json   batch statistics vs the priors
"""

from __future__ import annotations

import argparse
import collections
import json
import os

import numpy as np
import yaml

from .generator import CAUSES, build_truth_and_case


def build_batch(n: int, seed: int, priors: dict) -> tuple[list[dict], list[dict]]:
    master = np.random.default_rng(seed)
    cases, truths = [], []
    for i in range(1, n + 1):
        crng = np.random.default_rng(int(master.integers(2**63 - 1)))
        c, t = build_truth_and_case(i, crng, priors)
        cases.append(c)
        truths.append(t)
    return cases, truths


def summarise(cases: list[dict], truths: list[dict], priors: dict, seed: int) -> dict:
    n = len(cases)
    cc = collections.Counter(t["true_cause"] for t in truths)
    ic = collections.Counter(t["income_pattern"] for t in truths)
    day0 = [t["day0_p_success"] for t in truths]

    rec_by_cause = collections.defaultdict(lambda: [0, 0])
    for t in truths:
        rec_by_cause[t["true_cause"]][0] += int(t["retry_recoverable_within_horizon"])
        rec_by_cause[t["true_cause"]][1] += 1

    return {
        "n": n,
        "seed": seed,
        "cause_mix": {k: round(cc[k] / n, 3) for k in CAUSES},
        "cause_mix_prior": priors["cause_mix"],
        "income_mix": {k: round(ic[k] / n, 3) for k in sorted(ic)},
        "retry_recoverable_rate": round(sum(t["retry_recoverable_within_horizon"] for t in truths) / n, 3),
        "retry_recoverable_by_cause": {
            k: round(v[0] / v[1], 3) for k, v in sorted(rec_by_cause.items())
        },
        "day0_p_success_max": round(max(day0), 4),
        "day0_p_success_mean": round(sum(day0) / n, 4),
        "mean_debit_amount": round(float(np.mean([c["mandate"]["amount"] for c in cases]))),
        "mean_ledger_entries": round(float(np.mean([len(t["ledger"]) for t in truths]))),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Recoup Phase 1 — synthetic failed-mandate generator")
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--seed", type=int, default=None, help="overrides priors.seed")
    ap.add_argument("--priors", default="config/priors.yaml")
    ap.add_argument("--out", default="data")
    args = ap.parse_args()

    with open(args.priors) as f:
        priors = yaml.safe_load(f)
    seed = args.seed if args.seed is not None else priors.get("seed", 42)

    cases, truths = build_batch(args.n, seed, priors)

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "cases.jsonl"), "w") as f:
        for c in cases:
            f.write(json.dumps(c) + "\n")
    with open(os.path.join(args.out, "truth.jsonl"), "w") as f:
        for t in truths:
            f.write(json.dumps(t) + "\n")

    summary = summarise(cases, truths, priors, seed)
    with open(os.path.join(args.out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
