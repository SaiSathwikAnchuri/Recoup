"""The oracle — an upper bound on what any policy could recover on this batch.

It is allowed to read the hidden truth, so it lives in `harness/`, never `agent/`.
It is not a fair competitor; it is the ceiling every real policy is measured
against. "recoup closes X% of the oracle gap" is the honest way to read it.

  - retry_recoverable case  -> one retry at the hidden best_retry_at
  - dead mandate            -> one re-auth, then hand off
  - genuinely unrecoverable -> do nothing / escalate
"""

from __future__ import annotations

from simulator.calendar_utils import parse_dt

from .plan import Plan, ScheduledAction


class Oracle:
    name = "oracle"

    def __init__(self, truth: dict):
        self._truth = truth

    def plan(self, case: dict, state) -> Plan:
        t = self._truth[case["case_id"]]
        if t["true_cause"] == "mandate_dead":
            at = state.observed_at.replace(hour=11)
            return Plan([ScheduledAction(at, "reauth")], terminal="escalate",
                        note="oracle: dead mandate -> re-auth")
        if t["retry_recoverable_within_horizon"]:
            return Plan([ScheduledAction(parse_dt(t["best_retry_at"]), "retry")],
                        terminal="stop", note="oracle: retry at the true best moment")
        return Plan([], terminal="escalate", note="oracle: unrecoverable within horizon")
