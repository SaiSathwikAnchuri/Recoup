"""Agent-side policies (need the trained classifier; baselines/ stays dependency-free).

`cause_aware` is the Phase 4 deliverable: the industry fixed schedule, plus the
escalate branch. When the classifier is confident the mandate is dead it stops
burning debit retries — one re-auth request, then hand to a human. Everything
else is still the dumb calendar. It exists to measure exactly what knowing the
cause is worth, before the Phase 7 EV policy adds timing and cost.
"""

from __future__ import annotations

from harness.engine import EngagementState, Plan
from baselines._util import act

from .classifier import CauseClassifier


class CauseAwareRetry:
    name = "cause_aware"

    def __init__(self, clf: CauseClassifier | None = None):
        self._clf = clf or CauseClassifier.load()

    def plan(self, case: dict, state: EngagementState) -> Plan:
        o = state.observed_at
        probs = self._clf.predict_proba_one(case)

        if self._clf.should_escalate(probs):
            # probably dead: at most one re-auth attempt, then a human takes it
            return Plan(actions=[act(o, 2, "reauth")], terminal="escalate")

        # otherwise unchanged from fixed_schedule
        return Plan(
            actions=[act(o, 1, "retry"), act(o, 3, "retry"),
                     act(o, 4, "sms"), act(o, 7, "retry")],
            terminal="stop",
        )


def load_all() -> list:
    """Agent policies that are ready to run (model trained). Empty if not."""
    if not CauseClassifier.default_exists():
        return []
    return [CauseAwareRetry()]
