"""The policy <-> harness contract: a `Plan` is an ordered list of
`ScheduledAction`s plus a `terminal` mode. Kept in its own module so the
constraint filter and the engine can both import it without a cycle.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

DEBIT_HOUR = 11
ACTION_KINDS = ("retry", "nudge", "sms", "reauth")
MESSAGE_KINDS = ("nudge", "sms", "reauth")   # everything that reaches the customer


@dataclass(frozen=True)
class ScheduledAction:
    at: datetime
    kind: str  # one of ACTION_KINDS

    def __post_init__(self):
        if self.kind not in ACTION_KINDS:
            raise ValueError(f"bad action kind: {self.kind}")


@dataclass
class Plan:
    actions: list[ScheduledAction]
    terminal: str = "stop"  # "stop" | "escalate" | "replan"
    note: str = ""          # optional one-line rationale (for the audit trail, Phase 8)
