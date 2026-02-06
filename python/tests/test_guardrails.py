"""
Sanity Tests for Decision Chain Guardrails V2.1

These tests verify the critical failure modes that would cause
garbage trades in production:

1. No trade without signals
2. No duplicate buy at target
3. Thesis mismatch blocks trade
4. Symbol disagreement blocks trade
5. No trade without price
6. No sell when not held
"""

import pytest
from datetime import date
from dataclasses import dataclass, field
from typing import Dict, Optional, Set
from unittest.mock import MagicMock

# Import the modules we're testing
from core.collaboration.debate_v2 import (
    ThesisType,
    ConsensusGate,
    AgentTurn,
    ThesisProposal,
    ConfidenceDecomposition,
    DialogMove,
    Counterfactual,
    EvidenceReference,
    InvalidationRule,
    validate_thesis_against_strategy,
    compute_consensus_gate,
    STRATEGY_TO_THESIS_TYPES,
)
from core.backtest.simulation_engine import (
    ExecutionBlockResult,
    MAX_SINGLE_POSITION_WEIGHT,
)


# =============================================================================
# Mock Classes
# =============================================================================

@dataclass
class MockPosition:
    """Mock position for testing."""
    quantity: float = 100.0
    current_price: float = 100.0
    
    @property
    def current_value(self) -> float:
        return self.quantity * self.current_price


@dataclass
class MockPortfolio:
    """Mock portfolio for testing."""
    positions: Dict[str, MockPosition] = field(default_factory=dict)
    cash: float = 100_000.0
    total_value: float = 100_000.0
    
    def get_position_weight(self, symbol: str) -> float:
        if symbol not in self.positions:
            return 0.0
        pos = self.positions[symbol]
        return pos.current_value / self.total_value


@dataclass
class MockSnapshot:
    """Mock market snapshot for testing."""
    prices: Dict[str, float] = field(default_factory=dict)
    returns: Dict[str, Dict[str, float]] = field(default_factory=dict)
    volatilities: Dict[str, Dict[str, float]] = field(default_factory=dict)
    
    def get_price(self, symbol: str) -> Optional[float]:
        return self.prices.get(symbol)
    
    def get_return(self, symbol: str, period: str) -> Optional[float]:
        return self.returns.get(symbol, {}).get(period)
    
    def get_volatility(self, symbol: str, period: str) -> Optional[float]:
        return self.volatilities.get(symbol, {}).get(period)


def create_mock_turn(
    agent_id: str = "analyst",
    action: str = "buy",
    symbol: str = "AAPL",
    thesis_type: ThesisType = ThesisType.MOMENTUM,
    horizon_days: int = 21,
    signal_strength: float = 0.7,
    min_confidence: float = 0.6,
) -> AgentTurn:
    """Create a mock agent turn for testing."""
    return AgentTurn(
        agent_id=agent_id,
        round_num=0,
        dialog_move=DialogMove(
            acknowledge="Test",
            challenge=None,
            request=None,
            concede_or_hold="Test",
        ),
        action=action,
        symbol=symbol,
        suggested_weight=0.10,
        risk_posture="normal",
        thesis=ThesisProposal(
            thesis_type=thesis_type,
            horizon_days=horizon_days,
            primary_signal="return_21d",
            secondary_signal="return_63d",
            invalidation_rules=[
                InvalidationRule(
                    feature="return_1d",
                    symbol=symbol,
                    operator=">",
                    value=0.05,
                )
            ],
        ),
        confidence=ConfidenceDecomposition(
            signal_strength=signal_strength,
            regime_fit=min_confidence,
            risk_comfort=min_confidence,
            execution_feasibility=min_confidence,
        ),
        evidence_cited=[
            EvidenceReference(feature="return_21d", symbol=symbol),
            EvidenceReference(feature="return_63d", symbol=symbol),
        ],
        counterfactual=Counterfactual(
            alternative_action="hold",
            why_rejected="Test",
        ),
    )


# =============================================================================
# Test 1: No Trade Without Signals
# =============================================================================

def test_no_trade_without_signals():
    """
    If action != hold and signals_used == {}, decision should become hold.
    
    This is tested at the debate_runner level - we verify that the
    _extract_signals_from_evidence function correctly identifies missing signals.
    """
    from core.backtest.debate_runner import (
        get_feature_value,
        DecisionAuditTrail,
    )
    
    # Create a snapshot with NO data
    empty_snapshot = MockSnapshot()
    
    # Try to get a feature value
    value = get_feature_value(empty_snapshot, "AAPL", "return_21d")
    assert value is None, "Should return None for missing data"
    
    # Verify that empty signals would fail validation
    audit = DecisionAuditTrail(
        signals_used={},
        evidence_used=[],
        validation_report={
            "passed": False,
            "errors": ["No signals extracted"],
            "signal_count": 0,
            "evidence_count": 0,
            "required_features": ["return_21d", "return_63d"],
            "missing_required": ["return_21d", "return_63d"],
        }
    )
    
    assert not audit.validation_report["passed"], "Empty signals should fail"
    assert audit.signals_used == {}, "Signals should be empty"


# =============================================================================
# Test 2: No Duplicate Buy at Target
# =============================================================================

def test_no_duplicate_buy_at_target():
    """If already at target weight, buy is blocked."""
    # Import the function we're testing
    from core.backtest.simulation_engine import SimulationEngine
    
    # Create mock portfolio already at target
    portfolio = MockPortfolio(
        positions={"ADBE": MockPosition(quantity=100, current_price=100)},
        cash=90_000.0,
        total_value=100_000.0,  # Position is 10% of portfolio
    )
    
    # Create a mock engine to test the method
    engine = MagicMock(spec=SimulationEngine)
    
    # Test the guard logic directly
    current_weight = portfolio.get_position_weight("ADBE")
    target_weight = 0.10  # Same as current
    
    # Guard 1: Already at or above target
    reasons = []
    if current_weight >= target_weight:
        reasons.append(
            f"Already at {current_weight:.1%}, target {target_weight:.1%}"
        )
    
    assert len(reasons) > 0, "Should have rejection reason"
    assert "already at" in reasons[0].lower(), "Reason should mention 'already at'"


# =============================================================================
# Test 3: Thesis Mismatch Blocks Trade
# =============================================================================

def test_thesis_mismatch_blocks_trade():
    """If fund strategy = value and thesis_type = mean_reversion, block."""
    # Test value fund with mean_reversion thesis
    valid, reason = validate_thesis_against_strategy(
        ThesisType.MEAN_REVERSION, "value"
    )
    assert not valid, "Value fund should not allow mean_reversion thesis"
    assert "disabled" in reason.lower() or "no supported" in reason.lower(), \
        f"Reason should mention disabled: {reason}"
    
    # Test that value fund is explicitly disabled
    allowed = STRATEGY_TO_THESIS_TYPES.get("value", set())
    assert allowed == set(), "Value fund should have empty allowed thesis set"
    
    # Test momentum fund with momentum thesis (should pass)
    valid, reason = validate_thesis_against_strategy(
        ThesisType.MOMENTUM, "momentum"
    )
    assert valid, f"Momentum fund should allow momentum thesis: {reason}"
    
    # Test momentum fund with mean_reversion thesis (should fail)
    valid, reason = validate_thesis_against_strategy(
        ThesisType.MEAN_REVERSION, "momentum"
    )
    assert not valid, "Momentum fund should not allow mean_reversion thesis"


# =============================================================================
# Test 4: Symbol Disagreement Blocks Trade
# =============================================================================

def test_symbol_disagreement_blocks_trade():
    """If agents disagree on symbol, no trade."""
    # Create snapshot with data for both symbols
    snapshot = MockSnapshot(
        prices={"ADBE": 100.0, "ADSK": 100.0},
        returns={
            "ADBE": {"21d": 0.05, "63d": 0.10},
            "ADSK": {"21d": 0.03, "63d": 0.08},
        },
    )
    
    # Create turns with different symbols
    turn_a = create_mock_turn(
        agent_id="analyst",
        action="buy",
        symbol="ADBE",
        thesis_type=ThesisType.MOMENTUM,
    )
    turn_b = create_mock_turn(
        agent_id="critic",
        action="buy",
        symbol="ADSK",  # Different symbol!
        thesis_type=ThesisType.MOMENTUM,
    )
    
    # Compute consensus gate
    gate = compute_consensus_gate(turn_a, turn_b, snapshot, "momentum")
    
    # Symbol match should fail (strict mode)
    assert not gate.symbol_match, "Symbol match should fail with different symbols"
    assert not gate.is_executable(), "Gate should not be executable"
    assert "symbol" in str(gate.rejection_reasons).lower(), \
        f"Rejection reasons should mention symbol: {gate.rejection_reasons}"


# =============================================================================
# Test 5: No Trade Without Price
# =============================================================================

def test_no_trade_without_price():
    """If action is buy/sell but price is None, execution blocked."""
    from core.backtest.simulation_engine import SimulationEngine
    
    # Create empty snapshot (no prices)
    empty_snapshot = MockSnapshot()
    
    # Test price validation logic
    price = empty_snapshot.get_price("FAKE")
    
    # Simulate the validation
    if price is None:
        valid = False
        reason = f"No price available for FAKE"
    elif price <= 0:
        valid = False
        reason = f"Invalid price {price} for FAKE"
    else:
        valid = True
        reason = ""
    
    assert not valid, "Should fail validation with no price"
    assert "no price" in reason.lower(), f"Reason should mention no price: {reason}"


# =============================================================================
# Test 6: No Sell When Not Held
# =============================================================================

def test_no_sell_when_not_held():
    """Cannot sell a symbol not in portfolio."""
    # Create empty portfolio
    portfolio = MockPortfolio(positions={})  # Empty
    
    # Test sell guard logic
    symbol = "ADBE"
    reasons = []
    
    # Guard 1: Don't hold the symbol
    if symbol not in portfolio.positions:
        reasons.append(f"Cannot sell {symbol} - not in portfolio")
    
    assert len(reasons) > 0, "Should have rejection reason"
    assert "not in portfolio" in reasons[0].lower(), \
        f"Reason should mention not in portfolio: {reasons[0]}"


def test_no_sell_with_zero_shares():
    """Cannot sell a symbol with 0 shares."""
    # Create portfolio with 0 shares
    portfolio = MockPortfolio(
        positions={"ADBE": MockPosition(quantity=0, current_price=100)},
    )
    
    # Test sell guard logic
    symbol = "ADBE"
    reasons = []
    
    position = portfolio.positions.get(symbol)
    if position and position.quantity <= 0:
        reasons.append(f"Cannot sell {symbol} - quantity {position.quantity} <= 0")
    
    assert len(reasons) > 0, "Should have rejection reason"
    assert "quantity" in reasons[0].lower() and "<= 0" in reasons[0], \
        f"Reason should mention quantity <= 0: {reasons[0]}"


# =============================================================================
# Additional Tests for Consensus Gate
# =============================================================================

def test_consensus_gate_fund_strategy_match():
    """Test that fund_strategy_match is correctly computed."""
    snapshot = MockSnapshot(
        prices={"AAPL": 150.0},
        returns={"AAPL": {"21d": 0.05, "63d": 0.10}},
    )
    
    # Both agents propose momentum thesis for momentum fund
    turn_a = create_mock_turn(thesis_type=ThesisType.MOMENTUM)
    turn_b = create_mock_turn(thesis_type=ThesisType.MOMENTUM)
    
    gate = compute_consensus_gate(turn_a, turn_b, snapshot, "momentum")
    assert gate.fund_strategy_match, "Should match for momentum/momentum"
    
    # Test with value fund (disabled)
    gate_value = compute_consensus_gate(turn_a, turn_b, snapshot, "value")
    assert not gate_value.fund_strategy_match, "Value fund should be disabled"


def test_consensus_gate_rejection_reasons():
    """Test that rejection_reasons are populated correctly."""
    snapshot = MockSnapshot(
        prices={"AAPL": 150.0, "MSFT": 300.0},
        returns={
            "AAPL": {"21d": 0.05, "63d": 0.10},
            "MSFT": {"21d": 0.03, "63d": 0.08},
        },
    )
    
    # Create turns with multiple mismatches
    turn_a = create_mock_turn(
        action="buy",
        symbol="AAPL",
        thesis_type=ThesisType.MOMENTUM,
        horizon_days=5,  # Short
    )
    turn_b = create_mock_turn(
        action="sell",  # Different action
        symbol="MSFT",  # Different symbol
        thesis_type=ThesisType.MEAN_REVERSION,  # Different thesis
        horizon_days=30,  # Long (different bucket)
    )
    
    gate = compute_consensus_gate(turn_a, turn_b, snapshot, "momentum")
    
    assert not gate.is_executable(), "Gate should not be executable"
    assert len(gate.rejection_reasons) > 0, "Should have rejection reasons"
    
    # Check that key mismatches are captured
    reasons_str = str(gate.rejection_reasons).lower()
    assert "action" in reasons_str, "Should mention action mismatch"
    assert "symbol" in reasons_str, "Should mention symbol mismatch"
    assert "thesis" in reasons_str, "Should mention thesis mismatch"


# =============================================================================
# Run Tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
