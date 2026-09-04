"""Customer / mandate state engine — Recoup 2.0.

`CustomerState` is what the recovery system knows about one mandate, built by
folding its event log (`service/events.py`). It is deliberately assembled from
**observable signals only** — failed-debit codes, funding days the merchant can
see on its own ledger, actions Recoup itself took and their results. No hidden
simulator truth (`true_cause`, response functions, `ltv_true`) ever enters here;
`state.py` imports nothing from `simulator/response.py` or the truth records.

The state feeds two things:
  * `to_history()` — the observable `history` block the classifier and the
    funding-window model already consume, now backed by *accumulated* history
    instead of a single snapshot. Personalisation falls out of this for free:
    a customer with six observed funding days on the 5th gets a tighter window.
  * `churn_risk` / `recovery_stage` — cheap heuristics the planner and the
    dashboard use to decide how hard to keep trying.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta

from simulator.calendar_utils import parse_dt
from agent.costs import CostModel

from .events import Event, EventType

_COSTS = CostModel.from_yaml()


@dataclass
class CustomerState:
    customer_id: str
    mandate_id: str
    case_id: str

    mandate_age_months: int = 1
    category: str = "sip"
    amount: float = 0.0

    failure_count: int = 0                 # observed failed debits, all time
    successful_payment_count: int = 0
    consecutive_successes: int = 0

    recovery_attempts: int = 0             # retries + reauths this system has run
    recovery_successes: int = 0
    historical_failure_codes: list[str] = field(default_factory=list)
    historical_funding_days: list[int] = field(default_factory=list)   # day-of-month a debit last cleared
    retry_results: list[bool] = field(default_factory=list)
    reauth_results: list[bool] = field(default_factory=list)

    message_count: int = 0
    last_message_at: float | None = None
    last_retry_at: float | None = None
    last_success_date: str | None = None

    recovery_stage: str = "new"            # new|retrying|reauth|escalated|stopped|recovered|revoked
    observed_at: str | None = None
    horizon_days: int = 45

    # ---- derived ---------------------------------------------------------
    @property
    def estimated_ltv(self) -> float:
        return round(_COSTS.ltv_estimate(self.amount), 2)

    @property
    def churn_risk(self) -> float:
        """Observable proxy for P(this customer revokes). Rises with dunning
        pressure, a run of recent failures, and mandate age. Bounded [0, 0.95].
        Not the simulator's hidden hazard — a heuristic the planner can lean on."""
        r = 0.04
        r += 0.05 * self.message_count
        r += 0.03 * max(0, self.failure_count - self.successful_payment_count)
        r += 0.02 * len([x for x in self.retry_results[-4:] if not x])
        r += min(self.mandate_age_months, 36) / 36 * 0.06
        if "ER_mandate_revoked" in self.historical_failure_codes[-1:]:
            r += 0.4
        return round(min(r, 0.95), 4)

    @property
    def days_since_last_success(self) -> int:
        if not self.last_success_date or not self.observed_at:
            return 999
        return max(0, (parse_dt(self.observed_at).date()
                       - datetime.fromisoformat(self.last_success_date).date()).days)

    # ---- the observable history block the models consume -----------------
    def to_history(self) -> dict:
        hits = sorted({int(d) for d in self.historical_funding_days if 1 <= int(d) <= 31})
        cycles = max(self.successful_payment_count + self.failure_count, len(hits), 1)
        return {
            "consecutive_successes": self.consecutive_successes,
            "last_success": self.last_success_date,
            "days_since_last_success": self.days_since_last_success,
            "success_days_of_month": hits,
            "cycles_observed": cycles,
        }

    def to_dict(self) -> dict:
        d = asdict(self)
        d["estimated_ltv"] = self.estimated_ltv
        d["churn_risk"] = self.churn_risk
        d["days_since_last_success"] = self.days_since_last_success
        return d

    # ---- fold the event log --------------------------------------------
    @classmethod
    def rebuild(cls, events: list[Event], *, base_case: dict | None = None) -> "CustomerState":
        """Replay `events` (chronological) into a fresh state. `base_case` seeds
        the immutable facts (amount, category, age, seed history) from the case
        record that opened the recovery."""
        if not events and base_case is None:
            raise ValueError("need at least one event or a base_case")

        e0 = events[0] if events else None
        cid = (e0.customer_id if e0 else base_case["case_id"])
        mid = (e0.mandate_id if e0 else base_case["case_id"])
        case_id = (e0.case_id if e0 else base_case["case_id"])
        s = cls(customer_id=cid, mandate_id=mid, case_id=case_id)

        if base_case is not None:
            m, h = base_case.get("mandate", {}), base_case.get("history", {})
            s.amount = float(m.get("amount", 0.0))
            s.category = m.get("category", "sip")
            s.mandate_age_months = int(m.get("age_months", 1))
            s.observed_at = base_case.get("observed_at")
            s.horizon_days = int(base_case.get("horizon_days", 45))
            s.consecutive_successes = int(h.get("consecutive_successes", 0))
            s.successful_payment_count = int(h.get("consecutive_successes", 0))
            s.last_success_date = h.get("last_success")
            for d in (h.get("success_days_of_month") or []):
                s.historical_funding_days.append(int(d))

        for e in sorted(events, key=lambda x: x.at):
            s._apply(e)
        return s

    def _apply(self, e: Event) -> None:
        p = e.payload or {}
        t = e.type
        if t is EventType.PAYMENT_FAILED:
            self.failure_count += 1
            self.consecutive_successes = 0
            code = p.get("failure_code") or p.get("token")
            if code:
                self.historical_failure_codes.append(code)
            if p.get("amount"):
                self.amount = float(p["amount"])
            if p.get("age_months"):
                self.mandate_age_months = int(p["age_months"])
            if p.get("observed_at"):
                self.observed_at = p["observed_at"]
            if self.recovery_stage in ("new", "recovered"):
                self.recovery_stage = "retrying"

        elif t is EventType.RETRY_EXECUTED:
            self.recovery_attempts += 1
            self.last_retry_at = e.at
            ok = p.get("result") == "success"
            self.retry_results.append(ok)
            self.recovery_stage = "retrying"

        elif t is EventType.REAUTH_CREATED:
            self.recovery_stage = "reauth"

        elif t is EventType.REAUTH_COMPLETED:
            self.recovery_attempts += 1
            self.reauth_results.append(p.get("result") == "success")

        elif t is EventType.MESSAGE_SENT:
            self.message_count += 1
            self.last_message_at = e.at

        elif t is EventType.PAYMENT_RECOVERED:
            self.recovery_successes += 1
            self.successful_payment_count += 1
            self.consecutive_successes += 1
            self.recovery_stage = "recovered"
            if p.get("funding_day"):
                self.historical_funding_days.append(int(p["funding_day"]))
            if p.get("recovered_date"):
                self.last_success_date = p["recovered_date"]

        elif t is EventType.MANDATE_REVOKED:
            self.recovery_stage = "revoked"

        elif t is EventType.RECOVERY_ESCALATED:
            self.recovery_stage = "escalated"

        elif t is EventType.RECOVERY_STOPPED:
            self.recovery_stage = "stopped"
