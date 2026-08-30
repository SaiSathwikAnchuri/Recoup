"""Reference policies. Built BEFORE the real policy so there is always a number to beat.

  never_act        floor — no retry ever, so recovers nothing; shows revocation baseline
  fixed_schedule   Baseline A — the industry default: retry d+1, d+3, d+7 + a failure SMS
  always_nudge     Baseline B — communicate hard: nudge, retry, nudge, retry, retry
"""

from .always_nudge import AlwaysNudge
from .fixed_schedule import FixedSchedule
from .never_act import NeverAct

ALL = [NeverAct(), FixedSchedule(), AlwaysNudge()]
