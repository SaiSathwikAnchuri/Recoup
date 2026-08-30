"""Calibration diagnostics — plain numpy, no sklearn, so they are cheap to test.

A classifier is *calibrated* if, among the predictions it makes with confidence
~p, a fraction ~p are correct. `reliability()` bins predictions by their
top-class confidence and reports the gap; ECE is the count-weighted mean gap.
"""

from __future__ import annotations

import numpy as np


def reliability(probs: np.ndarray, y_true: np.ndarray, n_bins: int = 10) -> dict:
    """`probs`: (n, n_classes) predicted distribution. `y_true`: (n,) class indices.
    Bins by max-class confidence. Returns per-bin (confidence, accuracy, count),
    ECE (expected calibration error) and MCE (max)."""
    probs = np.asarray(probs, dtype=float)
    y_true = np.asarray(y_true)
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    correct = (pred == y_true).astype(float)

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bins = []
    ece = mce = 0.0
    n = len(y_true)
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi) if lo > 0 else (conf >= lo) & (conf <= hi)
        c = int(m.sum())
        if not c:
            bins.append({"lo": round(lo, 2), "hi": round(hi, 2), "count": 0,
                         "confidence": None, "accuracy": None})
            continue
        bin_conf = float(conf[m].mean())
        bin_acc = float(correct[m].mean())
        gap = abs(bin_conf - bin_acc)
        ece += c / n * gap
        mce = max(mce, gap)
        bins.append({"lo": round(lo, 2), "hi": round(hi, 2), "count": c,
                     "confidence": round(bin_conf, 4), "accuracy": round(bin_acc, 4)})

    return {"bins": bins, "ece": round(ece, 4), "mce": round(mce, 4),
            "accuracy": round(float(correct.mean()), 4), "n": n}


def brier_multiclass(probs: np.ndarray, y_true: np.ndarray, n_classes: int) -> float:
    """Mean squared error between the predicted distribution and the one-hot truth."""
    probs = np.asarray(probs, dtype=float)
    onehot = np.eye(n_classes)[np.asarray(y_true)]
    return round(float(np.mean(np.sum((probs - onehot) ** 2, axis=1))), 4)


def reliability_table(rel: dict) -> str:
    lines = [f"  bin        conf     acc    count", "  " + "-" * 34]
    for b in rel["bins"]:
        if not b["count"]:
            continue
        lines.append(f"  {b['lo']:.1f}-{b['hi']:.1f}   {b['confidence']:.3f}  "
                     f"{b['accuracy']:.3f}   {b['count']:5d}")
    lines.append("  " + "-" * 34)
    lines.append(f"  ECE {rel['ece']:.4f}   MCE {rel['mce']:.4f}   acc {rel['accuracy']:.4f}   n {rel['n']}")
    return "\n".join(lines)
