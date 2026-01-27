"""
PortfolioIntent - Target portfolio state, broker agnostic.

Key principles:
- Separates "what we want" from "how to get there"
- Enables backtesting and broker-agnostic execution
- Validates gross exposure and cash buffer, NOT abs-sum = 1
- Supports long-short funds
"""
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any
import uuid

from core.config.constants import CONSTANTS


class WeightBasis(Enum):
    """How weights are computed."""
    NAV = "nav"  # % of total portfolio value (recommended for v1)


@dataclass
class PositionIntent:
    """Target state for a single position."""
    symbol: str
    target_weight: float  # % of portfolio (can be negative for shorts)
    direction: str  # "long", "short", "flat"
    urgency: str = "patient"  # "immediate", "patient", "opportunistic"
    
    # Constraints
    max_slippage_bps: int = 10
    time_limit_minutes: Optional[int] = None


@dataclass
class ExitRule:
    """Exit rule for a position."""
    symbol: str
    stop_loss_pct: float
    take_profit_pct: float
    trailing_stop: bool = False
    time_stop_days: Optional[int] = None


@dataclass
class PortfolioIntent:
    """
    Target portfolio state - broker agnostic.
    
    Contains all data needed for deterministic execution.
    Includes asof_prices and portfolio_value so execution
    doesn't need to reach back into snapshot.
    """
    intent_id: str
    fund_id: str
    asof_timestamp: datetime
    
    # Valuation context (for deterministic execution)
    portfolio_value: float
    asof_prices: Dict[str, float]  # symbol -> price at decision time
    valuation_timestamp: datetime
    
    # Weight definition
    weight_basis: WeightBasis = WeightBasis.NAV
    
    # Target state
    positions: List[PositionIntent] = field(default_factory=list)
    target_cash_pct: float = 0.0
    
    # Constraints
    max_turnover: float = 1.0
    execution_window_minutes: int = 60
    
    # Policy applied
    sizing_method_used: str = ""
    policy_version: str = ""
    
    # Exit rules
    exit_rules: List[ExitRule] = field(default_factory=list)
    
    def validate_exposures(
        self,
        max_gross_exposure: float,
        min_cash_buffer: float
    ) -> Tuple[bool, Optional[str]]:
        """
        Validate gross exposure and cash buffer.
        
        FIXED: This correctly supports long-short funds.
        We validate:
        - gross <= max_gross_exposure (not abs-sum = 1)
        - cash >= min_cash_buffer
        
        For a long-short fund:
        - gross = sum(abs(w)) can be 1.5, 2.0, etc.
        - net = sum(w) can be anything
        - We just enforce the limits, not a sum constraint
        
        Args:
            max_gross_exposure: Maximum allowed gross exposure
            min_cash_buffer: Minimum required cash percentage
        
        Returns:
            Tuple of (is_valid, error_message)
        """
        # Compute gross exposure
        gross_exposure = sum(abs(p.target_weight) for p in self.positions)
        
        # Compute net exposure (signed sum)
        net_exposure = sum(p.target_weight for p in self.positions)
        
        tolerance = CONSTANTS.execution.GROSS_EXPOSURE_TOLERANCE
        
        # Check gross exposure limit
        if gross_exposure > max_gross_exposure + tolerance:
            return False, (
                f"Gross exposure {gross_exposure:.2f} exceeds limit "
                f"{max_gross_exposure:.2f}"
            )
        
        # Check cash buffer
        if self.target_cash_pct < min_cash_buffer - tolerance:
            return False, (
                f"Cash {self.target_cash_pct:.2%} below minimum "
                f"{min_cash_buffer:.2%}"
            )
        
        return True, None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON storage."""
        return {
            "intent_id": self.intent_id,
            "fund_id": self.fund_id,
            "asof_timestamp": self.asof_timestamp.isoformat(),
            "portfolio_value": self.portfolio_value,
            "asof_prices": self.asof_prices,
            "valuation_timestamp": self.valuation_timestamp.isoformat(),
            "weight_basis": self.weight_basis.value,
            "positions": [
                {
                    "symbol": p.symbol,
                    "target_weight": p.target_weight,
                    "direction": p.direction,
                    "urgency": p.urgency,
                    "max_slippage_bps": p.max_slippage_bps,
                    "time_limit_minutes": p.time_limit_minutes,
                }
                for p in self.positions
            ],
            "target_cash_pct": self.target_cash_pct,
            "max_turnover": self.max_turnover,
            "execution_window_minutes": self.execution_window_minutes,
            "sizing_method_used": self.sizing_method_used,
            "policy_version": self.policy_version,
            "exit_rules": [
                {
                    "symbol": r.symbol,
                    "stop_loss_pct": r.stop_loss_pct,
                    "take_profit_pct": r.take_profit_pct,
                    "trailing_stop": r.trailing_stop,
                    "time_stop_days": r.time_stop_days,
                }
                for r in self.exit_rules
            ],
        }
    
    @staticmethod
    def create_empty(fund_id: str, portfolio_value: float = 100000.0) -> "PortfolioIntent":
        """Create an empty intent for testing."""
        now = datetime.utcnow()
        return PortfolioIntent(
            intent_id=str(uuid.uuid4()),
            fund_id=fund_id,
            asof_timestamp=now,
            portfolio_value=portfolio_value,
            asof_prices={},
            valuation_timestamp=now,
            target_cash_pct=1.0,
        )


@dataclass
class Order:
    """Order to be sent to broker."""
    symbol: str
    quantity: float
    side: str  # "buy" or "sell"
    order_type: str = "market"
    limit_price: Optional[float] = None
    urgency: str = "patient"
    expected_price: float = 0.0
    max_slippage_bps: int = 10


class ExecutionEngine:
    """
    Converts PortfolioIntent to broker orders.
    
    Uses intent.asof_prices and intent.portfolio_value
    instead of reaching back into snapshot.
    """
    
    def execute(
        self,
        intent: PortfolioIntent,
        current_positions: Dict[str, float],  # symbol -> current weight
        policy_max_gross: float,
        policy_min_cash: float,
    ) -> Tuple[List[Order], Optional[str]]:
        """
        Convert intent to orders.
        
        Args:
            intent: Portfolio intent
            current_positions: Current position weights
            policy_max_gross: Max gross exposure from policy
            policy_min_cash: Min cash buffer from policy
        
        Returns:
            Tuple of (orders, error)
        """
        # Validate exposures first
        valid, error = intent.validate_exposures(policy_max_gross, policy_min_cash)
        if not valid:
            return [], error
        
        orders = []
        for pos_intent in intent.positions:
            current_weight = current_positions.get(pos_intent.symbol, 0.0)
            delta_weight = pos_intent.target_weight - current_weight
            
            if abs(delta_weight) < CONSTANTS.execution.MIN_WEIGHT_DELTA:
                continue
            
            # Get price from intent (not external lookup)
            price = intent.asof_prices.get(pos_intent.symbol)
            if price is None or price <= 0:
                continue
            
            # Calculate quantity
            delta_value = delta_weight * intent.portfolio_value
            quantity = abs(delta_value / price)
            
            # Determine side
            side = "buy" if delta_weight > 0 else "sell"
            
            orders.append(Order(
                symbol=pos_intent.symbol,
                quantity=quantity,
                side=side,
                order_type="market",  # v1: market orders only
                urgency=pos_intent.urgency,
                expected_price=price,
                max_slippage_bps=pos_intent.max_slippage_bps,
            ))
        
        return orders, None
