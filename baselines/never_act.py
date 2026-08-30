from __future__ import annotations

from harness.engine import EngagementState, Plan


class NeverAct:
    """Do nothing. Recovers nothing (a UPI debit needs an explicit retry), but every
    mandate is 'preserved' — the floor for both metrics."""

    name = "never_act"

    def plan(self, case: dict, state: EngagementState) -> Plan:
        return Plan(actions=[], terminal="stop")
