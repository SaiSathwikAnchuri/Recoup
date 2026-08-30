"""Phase 3 — cost model invariants."""

from pathlib import Path

import pytest

from agent.costs import CostModel

CM = CostModel.from_yaml()


def test_loads_from_default_yaml():
    assert CM.ltv_months == 12
    assert 0 < CM.value_per_recovery_frac <= 1


def test_all_action_costs_present_and_nonneg():
    for kind in ("retry", "nudge", "sms", "whatsapp", "reauth", "human_escalation"):
        assert CM.action_cost(kind) >= 0.0


def test_cost_ordering_matches_channel_economics():
    assert CM.action_cost("nudge") < CM.action_cost("sms") < CM.action_cost("whatsapp")
    assert CM.action_cost("human_escalation") > 10 * CM.action_cost("sms")


def test_unknown_action_raises():
    with pytest.raises(KeyError):
        CM.action_cost("carrier_pigeon")


def test_ltv_scales_linearly_and_stays_conservative():
    assert CM.ltv_estimate(2000) == pytest.approx(2 * CM.ltv_estimate(1000))
    # deliberately below the simulator's hidden mean of ~ amount * 12 * 0.85
    assert CM.ltv_estimate(1000) < 1000 * 12 * 0.85
    assert CM.ltv_estimate(1000) > 1000            # but a mandate is worth more than one debit


def test_delay_discount_is_monotone_and_gentle():
    a = CM.recovery_value(1000, days=0)
    b = CM.recovery_value(1000, days=20)
    c = CM.recovery_value(1000, days=45)
    assert a > b > c
    assert a == pytest.approx(1000)
    assert c > 0.8 * a            # gentle, not punitive


def test_missed_cycle_penalty_is_a_fraction_of_amount():
    assert 0 < CM.missed_cycle_penalty(1000) < 1000
    assert CM.missed_cycle_penalty(2000) == pytest.approx(2 * CM.missed_cycle_penalty(1000))


def test_message_fatigue_compounds():
    assert CM.message_fatigue_factor(1) == pytest.approx(1.0)
    f2, f3 = CM.message_fatigue_factor(2), CM.message_fatigue_factor(3)
    assert 1.0 < f2 < f3
    assert f3 == pytest.approx(f2 * CM.fatigue_multiplier)


def test_costs_module_does_not_import_simulator_internals():
    src = Path(CostModel.from_yaml.__globals__["__file__"]).read_text(encoding="utf-8")
    assert "simulator.response" not in src
    assert "import simulator" not in src
