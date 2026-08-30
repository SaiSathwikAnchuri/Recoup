"""The constraint filter — the rules of the game, enforced by the environment.

`enforce(plan, ...)` takes a policy's Plan and returns a *legal* Plan plus a list
of violations. Illegal actions are rescheduled forward to the first legal slot,
or dropped if none fits before the horizon. It is deterministic and idempotent:
`enforce(enforce(p)) == enforce(p)` with no new violations.

Loaded from `config/constraints.yaml`. The engine applies this to every plan from
every policy, so the caps are structural, not advisory.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import yaml

from .plan import MESSAGE_KINDS, Plan, ScheduledAction

_DEFAULT_PATH = Path(__file__).resolve().parent.parent / "config" / "constraints.yaml"


@dataclass(frozen=True)
class Constraints:
    max_retries: int
    retry_min_gap_hours: float
    max_messages: int
    message_min_gap_hours: float
    quiet_start: int          # hour [0,24): quiet from here ...
    quiet_end: int            # ... to here
    respect_horizon: bool

    @classmethod
    def from_yaml(cls, path: str | Path | None = None) -> "Constraints":
        cfg = yaml.safe_load(Path(path or _DEFAULT_PATH).read_text(encoding="utf-8"))
        r, m = cfg["retry"], cfg["messaging"]
        qs, qe = m["quiet_hours"]
        return cls(
            max_retries=int(r["max_per_engagement"]),
            retry_min_gap_hours=float(r["min_gap_hours"]),
            max_messages=int(m["max_per_engagement"]),
            message_min_gap_hours=float(m["min_gap_hours"]),
            quiet_start=int(qs), quiet_end=int(qe),
            respect_horizon=bool(cfg.get("respect_horizon", True)),
        )

    # -- helpers ---------------------------------------------------------
    def in_quiet_hours(self, dt: datetime) -> bool:
        h = dt.hour
        if self.quiet_start <= self.quiet_end:
            return self.quiet_start <= h < self.quiet_end
        return h >= self.quiet_start or h < self.quiet_end

    def next_open(self, dt: datetime) -> datetime:
        """First instant at/after `dt` that is outside quiet hours."""
        if not self.in_quiet_hours(dt):
            return dt
        base = dt if dt.hour < self.quiet_end else dt + timedelta(days=1)
        return base.replace(hour=self.quiet_end, minute=0, second=0, microsecond=0)


def _v(kind: str, action: str, at: datetime, resolution: str,
       moved_to: datetime | None = None) -> dict:
    return {"rule": kind, "action": action, "at": at.isoformat(),
            "resolution": resolution,
            "moved_to": moved_to.isoformat() if moved_to else None}


def enforce(plan: Plan, *, observed_at: datetime, horizon_end: datetime,
            c: Constraints,
            prior_retries: tuple[datetime, ...] = (),
            prior_messages: tuple[datetime, ...] = ()) -> tuple[Plan, list[dict]]:
    """Return (legal_plan, violations). `prior_*` are action times already spent
    in earlier replanning rounds, so caps and gaps carry across rounds."""
    violations: list[dict] = []
    legal: list[ScheduledAction] = []
    retry_times = sorted(prior_retries)
    msg_times = sorted(prior_messages)

    def past_horizon(dt: datetime) -> bool:
        return c.respect_horizon and dt > horizon_end

    for a in sorted(plan.actions, key=lambda x: x.at):
        at = a.at

        if at < observed_at or past_horizon(at):
            violations.append(_v("out_of_window", a.kind, a.at, "dropped"))
            continue

        if a.kind == "retry":
            if len(retry_times) >= c.max_retries:
                violations.append(_v("retry_cap", a.kind, a.at, "dropped"))
                continue
            if retry_times:
                earliest = retry_times[-1] + timedelta(hours=c.retry_min_gap_hours)
                if at < earliest:
                    if past_horizon(earliest):
                        violations.append(_v("retry_min_gap", a.kind, a.at, "dropped"))
                        continue
                    violations.append(_v("retry_min_gap", a.kind, a.at, "moved", earliest))
                    at = earliest
            retry_times.append(at)
            legal.append(ScheduledAction(at, "retry"))
            continue

        # customer-facing message
        assert a.kind in MESSAGE_KINDS
        if c.in_quiet_hours(at):
            moved = c.next_open(at)
            if past_horizon(moved):
                violations.append(_v("quiet_hours", a.kind, a.at, "dropped"))
                continue
            violations.append(_v("quiet_hours", a.kind, a.at, "moved", moved))
            at = moved
        if len(msg_times) >= c.max_messages:
            violations.append(_v("message_cap", a.kind, a.at, "dropped"))
            continue
        if msg_times:
            earliest = msg_times[-1] + timedelta(hours=c.message_min_gap_hours)
            if at < earliest:
                earliest = c.next_open(earliest)
                if past_horizon(earliest):
                    violations.append(_v("message_min_gap", a.kind, a.at, "dropped"))
                    continue
                violations.append(_v("message_min_gap", a.kind, a.at, "moved", earliest))
                at = earliest
        msg_times.append(at)
        legal.append(ScheduledAction(at, a.kind))

    legal.sort(key=lambda x: x.at)
    return Plan(legal, terminal=plan.terminal, note=plan.note), violations
