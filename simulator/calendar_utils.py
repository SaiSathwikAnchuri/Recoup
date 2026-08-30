"""Small date helpers. Everything is Asia/Kolkata (IST), which is where UPI lives."""

from datetime import datetime, timedelta, timezone

IST = timezone(timedelta(hours=5, minutes=30))

# Rough intraday ordering used by the cashflow model:
CREDIT_HOUR = 10        # salary / inbound credits post mid-morning
SCHED_DEBIT_HOUR = 13   # scheduled mandate debits (EMI/SIP/rent) hit early afternoon


def ist(y: int, m: int, d: int, hh: int = 0, mm: int = 0) -> datetime:
    return datetime(y, m, d, hh, mm, tzinfo=IST)


def parse_dt(s: str) -> datetime:
    return datetime.fromisoformat(s)


def iso(dt: datetime) -> str:
    return dt.isoformat()


def days_in_month(y: int, m: int) -> int:
    if m == 12:
        return 31
    return (datetime(y, m + 1, 1) - datetime(y, m, 1)).days


def months_between(start: datetime, end: datetime):
    """Yield (year, month) for every calendar month touched by [start, end]."""
    y, m = start.year, start.month
    while (y, m) <= (end.year, end.month):
        yield y, m
        m += 1
        if m == 13:
            m, y = 1, y + 1


def add_months(y: int, m: int, delta: int) -> tuple[int, int]:
    idx = y * 12 + (m - 1) + delta
    return idx // 12, idx % 12 + 1
