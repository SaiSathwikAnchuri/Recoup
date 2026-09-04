"""Lightweight model + policy monitoring — Recoup 2.0.

No new training or metrics infra: this reads the reports the pipeline already
writes (`results/*.json`) for the offline picture, and folds the running
service's `outcomes` table for the live picture. Surfaced at `/api/models/health`
and `/api/metrics`.
"""

from __future__ import annotations

import json
from pathlib import Path

_RESULTS = Path("results")


def _load(name: str) -> dict | None:
    p = _RESULTS / name
    try:
        return json.loads(p.read_text())
    except (OSError, ValueError):
        return None


def model_health() -> dict:
    clf = _load("classifier_report.json") or {}
    liq = _load("liquidity_report.json") or {}
    h = _load("harness_42.json") or {}
    summ = (h.get("summaries") or {}).get("recoup", {})

    ct = clf.get("test", {})
    lt = liq.get("test", {})
    lb = (liq.get("baselines") or {}).get("naive_modal_day", {})
    return {
        "classifier": {
            "accuracy": ct.get("accuracy"),
            "prior_baseline": ct.get("prior_baseline"),
            "ece_calibrated": ct.get("ece_calibrated"),
            "ece_uncalibrated": ct.get("ece_uncalibrated"),
            "brier_calibrated": ct.get("brier_calibrated"),
            "escalate_precision": (clf.get("escalate_branch") or {}).get("precision"),
            "escalate_recall": (clf.get("escalate_branch") or {}).get("recall"),
            "status": _band(ct.get("ece_calibrated"), 0.06, 0.10, lower_is_better=True),
        },
        "liquidity": {
            "mae_days": lt.get("mae_p50"),
            "median_ae_days": lt.get("median_ae_p50"),
            "bias_days": lt.get("bias_p50"),
            "p85_coverage": lt.get("p85_coverage"),
            "naive_mae_days": lb.get("mae_p50"),
            "status": _band(lt.get("p85_coverage"), 0.80, 0.70, lower_is_better=False),
        },
        "policy": {
            "recovery_rate": summ.get("recovery_rate"),
            "on_time_rate": summ.get("on_time_rate"),
            "net_value_per_case": summ.get("net_value_per_case"),
            "escalated": summ.get("escalated"),
            "messages_per_case": summ.get("messages_per_case"),
            "mandates_preserved_rate": summ.get("mandates_preserved_rate"),
        },
        "generated_from": "results/*.json (offline batch, seed 42)",
    }


def _band(v, good, warn, *, lower_is_better: bool) -> str:
    if v is None:
        return "unknown"
    if lower_is_better:
        return "good" if v <= good else "warn" if v <= warn else "alert"
    return "good" if v >= good else "warn" if v >= warn else "alert"


def live_metrics(store) -> dict:
    """The running service's own numbers, from the event-sourced tables."""
    cases = store.list_cases(limit=10_000)
    all_outs = store.all_outcomes()
    outs = [o for o in all_outs if o["action"] != "simulation"]     # per-action rows
    sims = [o for o in all_outs if o["action"] == "simulation"]     # demo rollups
    by_status: dict[str, int] = {}
    for c in cases:
        by_status[c["status"]] = by_status.get(c["status"], 0) + 1

    rec = [o for o in outs if o["result"] == "success" and o["action"] in ("retry", "reauth")]
    n_actions = len(outs)
    total_reward = round(sum(o["reward"] or 0.0 for o in outs), 2)
    recovered_rs = round(sum(o["recovered_amount"] or 0.0 for o in rec), 2)
    delays = [o["recovery_delay"] for o in rec if o["recovery_delay"] is not None]

    return {
        "cases_total": len(cases),
        "by_status": by_status,
        "actions_executed": n_actions,
        "recoveries": len(rec),
        "recovered_rupees": recovered_rs,
        "total_reward": total_reward,
        "reward_per_action": round(total_reward / n_actions, 2) if n_actions else None,
        "cost_per_recovery": round((n_actions * 0.4) / len(rec), 2) if rec else None,
        "mean_recovery_delay_days": round(sum(delays) / len(delays), 1) if delays else None,
        "actions_per_recovery": round(n_actions / len(rec), 2) if rec else None,
        "stop_rate": round(by_status.get("closed", 0) / len(cases), 3) if cases else None,
        "escalation_rate": round(by_status.get("escalated", 0) / len(cases), 3) if cases else None,
        "demo_simulations": len(sims),
        "demo_net_value_total": round(sum(o["reward"] or 0.0 for o in sims), 2) if sims else None,
    }
