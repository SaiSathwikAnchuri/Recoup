"""Train + evaluate the liquidity-window model.

    python -m agent.train_liquidity --n 6000 --train-seed 1000 --eval-data data

Same seed discipline as the classifier: training cases come from a seed that is
NOT the harness seed. Writes agent/models/liquidity.pkl and
results/liquidity_report.json.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import yaml

from simulator.calendar_utils import parse_dt
from simulator.generate import build_batch

from .liquidity import LIQ_FEATURES, LiquidityModel, liquidity_features, train_liquidity

LIQ_IDX = {n: i for i, n in enumerate(LIQ_FEATURES)}


def _days_to_funding(case: dict, truth: dict) -> float:
    return (parse_dt(truth["best_retry_at"]) - parse_dt(case["observed_at"])).total_seconds() / 86400.0


def _ib_recoverable(cases, truths):
    out_c, out_y = [], []
    for c, t in zip(cases, truths):
        if t["true_cause"] == "insufficient_balance" and t["retry_recoverable_within_horizon"]:
            out_c.append(c)
            out_y.append(_days_to_funding(c, t))
    return out_c, np.asarray(out_y)


def _naive_modal_days(case: dict) -> float:
    """Baseline: next occurrence of the customer's modal historical funding day."""
    return float(liquidity_features(case)[LIQ_IDX["days_to_next_modal"]])


def _metrics(name: str, pred_d50, pred_d85, actual, horizon: int) -> dict:
    err = pred_d50 - actual
    return {
        "name": name,
        "n": int(len(actual)),
        "mae_p50": round(float(np.mean(np.abs(err))), 2),
        "median_ae_p50": round(float(np.median(np.abs(err))), 2),
        "bias_p50": round(float(np.mean(err)), 2),
        "p85_coverage": round(float(np.mean(actual <= pred_d85)), 3),
        "p85_mean_slack_days": round(float(np.mean(pred_d85 - actual)), 2),
        "within_horizon_rate": round(float(np.mean(actual <= horizon)), 3),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Recoup Phase 5 — liquidity-window model")
    ap.add_argument("--n", type=int, default=6000)
    ap.add_argument("--train-seed", type=int, default=1000)
    ap.add_argument("--split-seed", type=int, default=7)
    ap.add_argument("--priors", default="config/priors.yaml")
    ap.add_argument("--eval-data", default="data")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    priors = yaml.safe_load(Path(args.priors).read_text())
    horizon = int(priors.get("horizon_days", 45))
    cases, truths = build_batch(args.n, args.train_seed, priors)
    c_ib, y_ib = _ib_recoverable(cases, truths)

    rng = np.random.default_rng(args.split_seed)
    idx = rng.permutation(len(c_ib))
    k = int(len(idx) * 0.7)
    tr, te = idx[:k], idx[k:]
    Xtr_c = [c_ib[i] for i in tr]
    ytr = y_ib[tr]
    Xte_c = [c_ib[i] for i in te]
    yte = y_ib[te]

    m50, m85 = train_liquidity(Xtr_c, ytr, seed=args.split_seed)
    model = LiquidityModel(m50, m85, horizon_days=horizon,
                           meta={"train_n": len(tr), "train_seed": args.train_seed})
    saved = model.save()

    pw = [model.predict_window(c) for c in Xte_c]
    d50 = np.array([w["days_p50"] for w in pw])
    d85 = np.array([w["days_p85"] for w in pw])
    model_m = _metrics("liquidity_model", d50, d85, yte, horizon)

    naive_d = np.array([_naive_modal_days(c) for c in Xte_c])
    naive_m = _metrics("naive_modal_day", naive_d, naive_d, yte, horizon)
    fixed7 = np.full(len(yte), 7.0)
    fixed_m = _metrics("fixed_day7", fixed7, fixed7, yte, horizon)

    holdout = {}
    evc = Path(args.eval_data) / "cases.jsonl"
    evt = Path(args.eval_data) / "truth.jsonl"
    if evc.exists() and evt.exists():
        hc = [json.loads(l) for l in evc.read_text().splitlines()]
        ht = {t["case_id"]: t for t in (json.loads(l) for l in evt.read_text().splitlines())}
        hib_c, hib_y = _ib_recoverable(hc, [ht[c["case_id"]] for c in hc])
        hw = [model.predict_window(c) for c in hib_c]
        hd50 = np.array([w["days_p50"] for w in hw])
        hd85 = np.array([w["days_p85"] for w in hw])
        holdout = _metrics("holdout_harness_batch", hd50, hd85, hib_y, horizon)

    report = {
        "train_n": len(tr), "test_n": len(te), "train_seed": args.train_seed,
        "model_path": str(saved),
        "test": model_m,
        "baselines": {"naive_modal_day": naive_m, "fixed_day7": fixed_m},
        "holdout_harness_batch": holdout,
    }
    Path(args.out).mkdir(exist_ok=True)
    (Path(args.out) / "liquidity_report.json").write_text(json.dumps(report, indent=2))

    print(f"\nLiquidity-window model — trained on {len(tr)} IB-recoverable cases "
          f"(seed {args.train_seed}), tested on {len(te)}\n")
    print(f"  {'':18s} {'MAE':>7} {'medAE':>7} {'bias':>7} {'p85 cover':>10} {'p85 slack':>10}")
    for m in (model_m, naive_m, fixed_m):
        print(f"  {m['name']:18s} {m['mae_p50']:7.2f} {m['median_ae_p50']:7.2f} "
              f"{m['bias_p50']:7.2f} {m['p85_coverage']:10.3f} {m['p85_mean_slack_days']:10.2f}")
    if holdout:
        print(f"\n  holdout (harness batch, {holdout['n']} IB cases): "
              f"MAE {holdout['mae_p50']:.2f}, p85 coverage {holdout['p85_coverage']:.3f}")
    print(f"\nwrote {saved}")
    print(f"wrote {args.out}/liquidity_report.json")


if __name__ == "__main__":
    main()
