from __future__ import annotations

from datetime import datetime, timedelta

from harness.engine import DEBIT_HOUR, ScheduledAction


def at_offset(observed_at: datetime, days: int, hour: int = DEBIT_HOUR) -> datetime:
    return (observed_at + timedelta(days=days)).replace(hour=hour, minute=0, second=0, microsecond=0)


def act(observed_at: datetime, days: int, kind: str) -> ScheduledAction:
    return ScheduledAction(at=at_offset(observed_at, days), kind=kind)
