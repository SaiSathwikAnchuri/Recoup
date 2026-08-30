"""Train + calibrate the cause classifier, tune the escalate threshold, report.

    python -m agent.train_classifier --n 6000 --train-seed 1000 --eval-data data

Training cases are generated fresh from a DIFFERENT seed than the harness batch
(default 42), so the model is never fitted on the cases it is later scored on.
Writes agent/models/cause_clf.pkl and results/classifier_report.json.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import yaml

from simulator.generate import build_batch

from .calibration import brier_multiclass, reliability, reliability_table
from .classifier import CAUSES, CauseClassifier, train


def _split(n: int, seed: int, frac: float = 0.7):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    k = int(n * frac)
    return idx[:k], idx[k:]


def _proba_in_cause_order(model, cases: list[dict]) -> np.ndarray:
    from .features import featurize_batch
    p = model.predict_proba(featurize_batch(cases))
    order = [list(model.classes_).index(c) for c in CAUSES]
    return p[:, order]


def _tune_dead_threshold(p_dead: np.ndarray, is_dead: np.ndarray,
                         target_precision: float = 0.85) -> tuple[float, dict]:
    """Smallest threshold whose precision on 'flag as dead' clears the target."""
    best = (1.0, {"precision": 1.0, "recall": 0.0, "n_flagged": 0})
    for tau in np.round(np.arange(0.30, 0.96, 0.01), 2):
        flag = p_dead >= tau
        if flag.sum() == 0:
            continue
        prec = float((flag & is_dead).sum() / flag.sum())
        rec = float((flag & is_dead).sum() / max(is_dead.sum(), 1))
        if prec >= target_precision:
            return float(tau), {"precision": round(prec, 3), "recall": round(rec, 3),
                                "n_flagged": int(flag.sum())}
        if prec > best[1]["precision"] - 1e-9:
            best = (float(tau), {"precision": round(prec, 3), "recall": round(rec, 3),
                                 "n_flagged": int(flag.sum())})
    return best


def _confusion(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    idx = {c: i for i, c in enumerate(CAUSES)}
    m = np.zeros((4, 4), dtype=int)
    for t, p in zip(y_true, y_pred):
        m[idx[t], idx[p]] += 1
    return {CAUSES[i]: {CAUSES[j]: int(m[i, j]) for j in range(4)} for i in range(4)}


def main() -> None:
    ap = argparse.ArgumentParser(description="Recoup Phase 4 — cause classifier")
    ap.add_argument("--n", type=int, default=6000)
    ap.add_argument("--train-seed", type=int, default=1000)
    ap.add_argument("--split-seed", type=int, default=7)
    ap.add_argument("--priors", default="config/priors.yaml")
    ap.add_argument("--eval-data", default="data", help="harness batch to score as an honest holdout")
    ap.add_argument("--target-precision", type=float, default=0.85)
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    priors = yaml.safe_load(Path(args.priors).read_text())
    cases, truths = build_batch(args.n, args.train_seed, priors)
    labels = [t["true_cause"] for t in truths]

    tr, te = _split(len(cases), args.split_seed)
    Xtr = [cases[i] for i in tr]
    ytr = [labels[i] for i in tr]
    Xte = [cases[i] for i in te]
    yte = np.array([labels[i] for i in te])

    cal = train(Xtr, ytr, seed=args.split_seed, calibrate=True)
    raw = train(Xtr, ytr, seed=args.split_seed, calibrate=False)

    p_cal = _proba_in_cause_order(cal, Xte)
    p_raw = _proba_in_cause_order(raw, Xte)
    y_idx = np.array([CAUSES.index(c) for c in yte])
    pred = np.array([CAUSES[i] for i in p_cal.argmax(axis=1)])

    rel_cal = reliability(p_cal, y_idx)
    rel_raw = reliability(p_raw, y_idx)

    is_dead = (yte == "mandate_dead")
    tau, dead_stats = _tune_dead_threshold(p_cal[:, CAUSES.index("mandate_dead")],
                                           is_dead, args.target_precision)

    clf = CauseClassifier(cal, dead_threshold=tau,
                          meta={"train_n": len(tr), "train_seed": args.train_seed})
    saved = clf.save()

    # honest holdout: the real harness batch, generated from a different seed
    holdout = {}
    ev_cases = Path(args.eval_data) / "cases.jsonl"
    ev_truth = Path(args.eval_data) / "truth.jsonl"
    if ev_cases.exists() and ev_truth.exists():
        hc = [json.loads(l) for l in ev_cases.read_text().splitlines()]
        ht = {t["case_id"]: t for t in (json.loads(l) for l in ev_truth.read_text().splitlines())}
        hy = np.array([ht[c["case_id"]]["true_cause"] for c in hc])
        hp = clf.predict_proba(hc)
        hyi = np.array([CAUSES.index(c) for c in hy])
        hrel = reliability(hp, hyi)
        hpred = np.array([CAUSES[i] for i in hp.argmax(axis=1)])
        flag = hp[:, CAUSES.index("mandate_dead")] >= tau
        hd = (hy == "mandate_dead")
        holdout = {
            "n": len(hc),
            "accuracy": hrel["accuracy"],
            "ece": hrel["ece"],
            "brier": brier_multiclass(hp, hyi, 4),
            "confusion": _confusion(hy, hpred),
            "escalate_branch": {
                "threshold": tau,
                "precision": round(float((flag & hd).sum() / max(flag.sum(), 1)), 3),
                "recall": round(float((flag & hd).sum() / max(hd.sum(), 1)), 3),
                "n_flagged": int(flag.sum()),
                "n_dead": int(hd.sum()),
            },
        }

    report = {
        "train_n": len(tr), "test_n": len(te), "train_seed": args.train_seed,
        "model_path": str(saved),
        "test": {
            "accuracy": rel_cal["accuracy"],
            "prior_baseline": round(max(priors["cause_mix"].values()), 3),
            "ece_calibrated": rel_cal["ece"], "ece_uncalibrated": rel_raw["ece"],
            "mce_calibrated": rel_cal["mce"],
            "brier_calibrated": brier_multiclass(p_cal, y_idx, 4),
            "brier_uncalibrated": brier_multiclass(p_raw, y_idx, 4),
            "confusion": _confusion(yte, pred),
        },
        "escalate_branch": {"threshold": tau, **dead_stats,
                            "target_precision": args.target_precision},
        "holdout_harness_batch": holdout,
    }
    Path(args.out).mkdir(exist_ok=True)
    (Path(args.out) / "classifier_report.json").write_text(json.dumps(report, indent=2))

    print(f"\nCause classifier — trained on {len(tr)} cases (seed {args.train_seed}), "
          f"tested on {len(te)}\n")
    print(f"  accuracy            {rel_cal['accuracy']:.3f}   (prior baseline "
          f"{max(priors['cause_mix'].values()):.3f})")
    print(f"  ECE  calibrated     {rel_cal['ece']:.4f}   (uncalibrated {rel_raw['ece']:.4f})")
    print(f"  Brier calibrated    {report['test']['brier_calibrated']:.4f}   "
          f"(uncalibrated {report['test']['brier_uncalibrated']:.4f})")
    print(f"\n  reliability (calibrated, test):")
    print(reliability_table(rel_cal))
    print(f"\n  escalate branch: P(dead) >= {tau:.2f}  ->  precision {dead_stats['precision']}, "
          f"recall {dead_stats['recall']}  ({dead_stats['n_flagged']} flagged)")
    if holdout:
        h = holdout["escalate_branch"]
        print(f"\n  on the harness batch ({holdout['n']} cases, seed != train): "
              f"acc {holdout['accuracy']:.3f}, ECE {holdout['ece']:.4f}")
        print(f"  escalate branch there: precision {h['precision']}, recall {h['recall']} "
              f"({h['n_flagged']}/{h['n_dead']} dead mandates flagged)")
    print(f"\nwrote {saved}")
    print(f"wrote {args.out}/classifier_report.json")


if __name__ == "__main__":
    main()
