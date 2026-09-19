"""M13 -- mechanical budget-parity enforcement (criterion H3, falsification question 14)."""

import pytest

from amaun.errors import BudgetParityError
from amaun.training.budget import ArmBudget, BudgetLedger


def _ledger(tolerance=0.05):
    ledger = BudgetLedger(tolerance=tolerance)
    ledger.arms["amaun"] = ArmBudget(
        arm_id="amaun", optimizer_steps=1000, tokens_seen=500_000,
        examples_seen=4000, tuning_trials=5,
    )
    return ledger


def test_matched_budgets_compare_cleanly():
    ledger = _ledger()
    ledger.arms["B2"] = ArmBudget(
        arm_id="B2", optimizer_steps=1010, tokens_seen=502_000,
        examples_seen=4020, tuning_trials=5,
    )
    verdict = ledger.check_parity("amaun", "B2")
    assert verdict["fair"] is True
    assert verdict["divergences"] == {}


def test_extra_training_blocks_the_comparison():
    """A gain bought with 2x the steps is not attributable to the method."""
    ledger = _ledger()
    ledger.arms["B2"] = ArmBudget(
        arm_id="B2", optimizer_steps=500, tokens_seen=250_000,
        examples_seen=2000, tuning_trials=5,
    )
    with pytest.raises(BudgetParityError) as info:
        ledger.check_parity("amaun", "B2")
    message = str(info.value)
    assert "optimizer_steps" in message
    assert "cannot be attributed" in message


def test_extra_tuning_effort_also_blocks_the_comparison():
    """50 tuning trials vs 1 is unfair even when every other budget matches."""
    ledger = _ledger()
    ledger.arms["B4"] = ArmBudget(
        arm_id="B4", optimizer_steps=1000, tokens_seen=500_000,
        examples_seen=4000, tuning_trials=1,
    )
    with pytest.raises(BudgetParityError) as info:
        ledger.check_parity("amaun", "B4")
    assert "tuning_trials" in str(info.value)


def test_unequal_budget_requires_a_declared_reason():
    ledger = _ledger()
    ledger.arms["B1"] = ArmBudget(arm_id="B1", optimizer_steps=100)
    with pytest.raises(BudgetParityError, match="requires an explicit reason"):
        ledger.check_parity("amaun", "B1", allow_unequal_budget=True)

    verdict = ledger.check_parity(
        "amaun", "B1", allow_unequal_budget=True,
        reason="B1 trains all backbone parameters; step counts are not comparable by design",
    )
    assert verdict["fair"] is False
    assert verdict["declared_reason"]


def test_budget_accumulates_from_real_batches():
    budget = ArmBudget(arm_id="x")
    budget.add_batch(tokens=120, examples=4)
    budget.add_steps()
    budget.add_batch(tokens=80, examples=4)
    budget.add_steps(2)
    assert budget.tokens_seen == 200
    assert budget.examples_seen == 8
    assert budget.optimizer_steps == 3


def test_ledger_persists(tmp_path):
    ledger = _ledger()
    path = ledger.save(tmp_path / "budget.json")
    import json

    payload = json.loads(path.read_text())
    assert payload["tolerance"] == 0.05
    assert payload["arms"]["amaun"]["optimizer_steps"] == 1000


def test_invalid_tolerance_is_rejected():
    with pytest.raises(ValueError):
        BudgetLedger(tolerance=1.5)
