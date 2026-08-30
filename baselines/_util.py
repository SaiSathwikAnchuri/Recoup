from __future__ import annotations

from datetime import datetime, timedelta

from harness.plan import DEBIT_HOUR, ScheduledAction


def at_offset(observed_at: datetime, days: int, hour: int = DEBIT_HOUR) -> datetime:
    slot = (observed_at + timedelta(days=days)).replace(hour=hour, minute=0, second=0, microsecond=0)
    return max(slot, observed_at)   # never schedule before the failure was observed


def act(observed_at: datetime, days: int, kind: str) -> ScheduledAction:
    return ScheduledAction(at=at_offset(observed_at, days), kind=kind)
