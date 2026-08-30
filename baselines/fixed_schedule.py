from __future__ import annotations

from harness.engine import EngagementState, Plan
from ._util import act


class FixedSchedule:
    """Baseline A — the industry default. A calendar the merchant picked in 2023:
    retry on day +1, +3, +7, with a 'your payment failed' SMS on day +4, then lapse.
    Blind to cause, to the customer's funding cycle, and to what an attempt costs."""

    name = "fixed_schedule"

    def plan(self, case: dict, state: EngagementState) -> Plan:
        o = state.observed_at
        return Plan(
            actions=[
                act(o, 1, "retry"),
                act(o, 3, "retry"),
                act(o, 4, "sms"),
                act(o, 7, "retry"),
            ],
            terminal="stop",
        )
