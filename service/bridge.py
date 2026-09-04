"""Thin wrappers over the decision engine — nothing here is new logic.

  decide(case)              -> EVPolicy.explain + plain-English narration
  random_case(seed)         -> a synthetic failed mandate + its hidden truth (demo)
  simulate(case, truth)     -> run Recoup AND fixed_schedule over the 45-day window,
                               return both timelines + outcomes for the console to replay
  due_actions(decision)     -> the plan as (kind, offset_days) the scheduler can book
"""

from __future__ import annotations

from datetime import timedelta
from functools import lru_cache
from pathlib import Path

import numpy as np
import yaml

from agent.audit import narrate
from agent.classifier import CauseClassifier
from agent.ev_policy import EVPolicy
from agent.liquidity import LiquidityModel
from baselines import FixedSchedule
from harness.engine import HarnessConfig, run_case
from harness.metrics import net_value
from simulator.calendar_utils import parse_dt
from simulator.generator import build_truth_and_case

_PRIORS = yaml.safe_load((Path("config/priors.yaml")).read_text())
_CFG = HarnessConfig(rev_cfg=_PRIORS["revocation"])


def models_ready() -> bool:
    return CauseClassifier.default_exists() and LiquidityModel.default_exists()


@lru_cache(maxsize=1)
def _policy() -> EVPolicy:
    return EVPolicy()


@lru_cache(maxsize=1)
def _costs():
    from agent.costs import CostModel
    return CostModel.from_yaml()


_BLANK_OUTCOME = {"recovered": False, "revoked": False, "escalated": False,
                  "attempts_used": 0, "messages_sent": 0, "blocked_actions": 0,
                  "via_reauth": False, "days_to_recovery": None, "recovered_on_time": False}


# ---------------------------------------------------------------------------
def decide(case: dict, *, customer_state=None, engagement=None) -> dict:
    """The decision, opened up for a human: belief, every candidate priced,
    the choice, and a one-paragraph reason.

    Recoup 2.0: pass `customer_state` (folded from the event log) and/or
    `engagement` (accumulated retry/message history) and the record also carries
    the Recovery Opportunity Score breakdown and the adaptive `recoup_v2` choice.
    With neither, this is the original one-shot `recoup` explanation."""
    pol = _policy()
    d = pol.explain(case, engagement)
    d["narration"] = narrate({**d, "outcome": _BLANK_OUTCOME}).split("Outcome:")[0].strip()

    if customer_state is not None or engagement is not None:
        from agent.ros import RecoveryOpportunity
        from agent.policies import RecoupV2
        from harness.engine import EngagementState

        st = engagement or EngagementState(
            observed_at=parse_dt(case["observed_at"]),
            horizon_days=case.get("horizon_days", 45),
            attempt_budget=_CFG.attempt_budget)
        ro = RecoveryOpportunity(pol)
        cands = ro.score_candidates(case, st, customer_state)
        d["ros_candidates"] = [c.as_dict() for c in cands[:6]]
        if customer_state is not None:
            d["customer_state"] = (customer_state.to_dict()
                                   if hasattr(customer_state, "to_dict") else customer_state)
        plan = RecoupV2(ro).plan(case, st)
        d["decision"] = {
            "terminal": plan.terminal,
            "actions": [{"day": (a.at.date() - st.observed_at.date()).days, "kind": a.kind}
                        for a in plan.actions],
            "note": plan.note,
        }
        d["policy"] = "recoup_v2"
        d["narration"] = narrate({**d, "outcome": _BLANK_OUTCOME}).split("Outcome:")[0].strip()
    return d


def due_actions(decision: dict, observed_at) -> list[dict]:
    obs = parse_dt(observed_at) if isinstance(observed_at, str) else observed_at
    out = []
    for a in decision["decision"]["actions"]:
        out.append({"kind": a["kind"],
                    "due_at": (obs + timedelta(days=a["day"])).timestamp(),
                    "day": a["day"]})
    return out


# ---------------------------------------------------------------------------
def random_case(seed: int | None = None, cause: str | None = None):
    """A synthetic failed mandate + its hidden truth. `cause` forces the scenario.
    With no cause pinned, lightly down-weights the cases where the money arrives so
    early that any schedule recovers it — the decision is more interesting when the
    funding day is past the industry retry window or the mandate is dead."""
    rng = np.random.default_rng(seed)
    fallback = None
    for _ in range(600):
        idx = int(rng.integers(1, 10_000))
        crng = np.random.default_rng(int(rng.integers(2**63 - 1)))
        case, truth = build_truth_and_case(idx, crng, _PRIORS)
        case["case_id"] = f"demo_{idx}"
        if cause is not None:
            if truth["true_cause"] == cause:
                return case, truth
            continue
        fallback = (case, truth)
        best = truth.get("best_retry_at")
        early = best and (parse_dt(best) - parse_dt(case["observed_at"])).days <= 6
        trivial = early and truth["true_cause"] not in ("mandate_dead", "limit_breach")
        if not trivial or rng.random() < 0.25:
            return case, truth
    return fallback


def simulate(case: dict, truth: dict, seed: int = 42) -> dict:
    """Run Recoup and the industry fixed schedule over the same 45-day world."""
    r = run_case(case, truth, _policy(), _CFG, seed)
    f = run_case(case, truth, FixedSchedule(), _CFG, seed)
    r.true_ltv = float(truth.get("ltv_true", 0.0))
    f.true_ltv = float(truth.get("ltv_true", 0.0))
    costs = _costs()

    def pack(o):
        return {
            "policy": o.policy, "recovered": o.recovered,
            "amount_recovered": o.amount_recovered, "via_reauth": o.via_reauth,
            "recovered_on_time": o.recovered_on_time, "days_to_recovery": o.days_to_recovery,
            "revoked": o.revoked, "escalated": o.escalated,
            "attempts_used": o.attempts_used, "messages_sent": o.messages_sent,
            "stop_reason": o.stop_reason, "timeline": o.timeline,
            "net_value": round(net_value(o, costs), 0),
        }

    return {
        "true_cause": truth["true_cause"],
        "true_best_retry_day": round(
            (parse_dt(truth["best_retry_at"]) - parse_dt(case["observed_at"])).days
        ) if truth.get("retry_recoverable_within_horizon") else None,
        "recoup": pack(r),
        "fixed_schedule": pack(f),
        "net_delta": round(net_value(r, costs) - net_value(f, costs), 0),
    }
