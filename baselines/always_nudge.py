from __future__ import annotations

from harness.engine import EngagementState, Plan
from ._util import act


class AlwaysNudge:
    """Baseline B — treat 'more communication' as the strategy. Nudge immediately,
    retry a day later, nudge again, retry, retry. Tests whether messaging harder helps
    (it raises revocation risk without knowing if the timing is any good)."""

    name = "always_nudge"

    def plan(self, case: dict, state: EngagementState) -> Plan:
        o = state.observed_at
        return Plan(
            actions=[
                act(o, 0, "nudge"),
                act(o, 2, "retry"),
                act(o, 3, "sms"),
                act(o, 5, "retry"),
                act(o, 9, "retry"),
            ],
            terminal="stop",
        )
