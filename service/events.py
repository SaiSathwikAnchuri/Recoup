"""The event abstraction — Recoup 2.0.

Everything that happens to a mandate is an `Event`: the failed debit that starts
a recovery, every action Recoup schedules and executes, and every outcome the
world reports back. Events are timestamped, keyed to a customer + mandate,
persisted append-only (`service/store.py`), and folded into a `CustomerState`
(`service/state.py`).

The event log is the single source of truth for a recovery. The decision engine
never sees it directly — `state.py` derives an *observable-only* view first, so
the no-leakage boundary (report R3) still holds: no hidden simulator truth ever
reaches the policy through here.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field, asdict
from enum import Enum


class EventType(str, Enum):
    PAYMENT_FAILED = "PAYMENT_FAILED"        # a debit failed — starts / continues a recovery
    RETRY_SCHEDULED = "RETRY_SCHEDULED"
    RETRY_EXECUTED = "RETRY_EXECUTED"        # payload: {"result": "success"|"fail", "p_success": float}
    PAYMENT_RECOVERED = "PAYMENT_RECOVERED"  # payload: {"amount": float, "via": "retry"|"reauth", "delay_days": float}
    REAUTH_CREATED = "REAUTH_CREATED"        # payload: {"link": str|None}
    REAUTH_COMPLETED = "REAUTH_COMPLETED"
    MESSAGE_SENT = "MESSAGE_SENT"            # payload: {"channel": "sms"|"nudge"}
    MANDATE_REVOKED = "MANDATE_REVOKED"
    RECOVERY_STOPPED = "RECOVERY_STOPPED"    # payload: {"reason": str}
    RECOVERY_ESCALATED = "RECOVERY_ESCALATED"


# events that change nothing when replayed — safe to drop a duplicate of
_TERMINAL = {EventType.PAYMENT_RECOVERED, EventType.MANDATE_REVOKED,
             EventType.RECOVERY_STOPPED, EventType.RECOVERY_ESCALATED}


@dataclass
class Event:
    """One thing that happened. `dedup_key`, when set, makes the write idempotent:
    a second event with the same key is silently ignored by the store."""
    type: EventType
    customer_id: str
    mandate_id: str
    case_id: str
    payload: dict = field(default_factory=dict)
    at: float = field(default_factory=time.time)          # unix seconds
    event_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    dedup_key: str | None = None

    def __post_init__(self):
        if not isinstance(self.type, EventType):
            self.type = EventType(self.type)

    def to_row(self) -> dict:
        d = asdict(self)
        d["type"] = self.type.value
        return d

    @classmethod
    def from_row(cls, row: dict) -> "Event":
        return cls(
            type=EventType(row["type"]),
            customer_id=row["customer_id"], mandate_id=row["mandate_id"],
            case_id=row["case_id"], payload=row.get("payload") or {},
            at=row["at"], event_id=row["event_id"], dedup_key=row.get("dedup_key"),
        )

    @property
    def is_terminal(self) -> bool:
        return self.type in _TERMINAL
