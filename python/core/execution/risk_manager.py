"""
Risk Manager - Multi-stage risk checking with explicit violations.

Key principles:
- 3-stage checking: pre-trade (intent), order-level, post-fill
- RiskCheckResult with scale_factor and detailed violations
- Quantity-based go_flat (no stale prices)
- Uses FundRiskStateRepo for cooldown tracking
- Vol-based position sizing for strategy-aware controls
"""
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, TYPE_CHECKING
import numpy as np

from core.config.constants import CONSTANTS
from core.execution.risk_repo import FundRiskState, FundRiskStateRepo

if TYPE_CHECKING:
    from core.funds.fund import Fund, FundPortfolio
    from core.execution.intent import PortfolioIntent, Order


@dataclass
class RiskViolation:
    """A specific risk rule violation."""
    rule_name: str
    symbol: Optional[str]  # None for portfolio-level rules
    limit: float
    actual: float
    severity: str  # "warning", "scale", "veto"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "rule_name": self.rule_name,
            "symbol": self.symbol,
            "limit": self.limit,
            "actual": self.actual,
            "severity": self.severity,
        }


@dataclass
class RiskCheckResult:
    """
    Result of risk check with explicit details.
    
    Status can be:
    - "approved": Trade as proposed
    - "scaled": Trade with reduced sizes
    - "vetoed": No trade allowed
    """
    status: str  # "approved", "scaled", "vetoed"
    scale_factor: float  # 1.0 for approved, <1.0 for scaled, 0.0 for vetoed
    per_symbol_scales: Dict[str, float] = field(default_factory=dict)
    violations: List[RiskViolation] = field(default_factory=list)
    applied_rules: List[str] = field(default_factory=list)
    reason: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "scale_factor": self.scale_factor,
            "per_symbol_scales": self.per_symbol_scales,
            "violations": [v.to_dict() for v in self.violations],
            "applied_rules": self.applied_rules,
            "reason": self.reason,
        }


@dataclass
class CloseOrder:
    """
    Quantity-based close order - no price needed.
    
    This is the fix for stale prices in go_flat.
    Uses current quantity for market close.
    """
    symbol: str
    quantity: float  # Positive = shares to close
    side: str  # "sell" for longs, "buy" for shorts (to close)
    order_type: str = "market"
    urgency: str = "immediate"
    max_slippage_bps: int = CONSTANTS.risk.EMERGENCY_CLOSE_SLIPPAGE_BPS


@dataclass
class PostFillAction:
    """Result of post-fill risk check."""
    action: str  # "none", "halt", "go_flat"
    reason: Optional[str] = None
    close_orders: List[CloseOrder] = field(default_factory=list)


@dataclass
class Fill:
    """A filled order."""
    symbol: str
    quantity: float
    price: float
    side: str
    timestamp: datetime = field(default_factory=datetime.utcnow)


class RiskManager:
    """
    Multi-stage risk checking with repository pattern.
    
    Stages:
    1. check_intent: Pre-trade validation of PortfolioIntent
    2. check_orders: Validation of concrete orders before sending
    3. check_post_fill: Post-execution validation, may trigger go_flat
    """
    
    def __init__(self, risk_state_repo: FundRiskStateRepo):
        """
        Initialize with risk state repository.
        
        Args:
            risk_state_repo: Repository for fund risk state (cooldowns, P&L)
        """
        self.risk_state_repo = risk_state_repo
    
    def compute_vol_based_position_size(
        self,
        base_weight: float,
        asset_vol: float,
        target_portfolio_vol: float,
        max_position_cap: float,
    ) -> float:
        """
        Compute vol-adjusted position size (risk parity).
        
        Position size scales inversely with asset volatility:
        size = min(cap, base_weight * (target_vol / asset_vol))
        
        Args:
            base_weight: Base position weight (e.g., 0.10)
            asset_vol: Asset 21d volatility (annualized)
            target_portfolio_vol: Target portfolio vol (e.g., 0.15)
            max_position_cap: Hard cap (e.g., 0.15)
            
        Returns:
            Vol-adjusted position size
        """
        if asset_vol <= 0:
            return max_position_cap
        
        # Risk parity scaling
        vol_adjusted = base_weight * (target_portfolio_vol / asset_vol)
        
        # Apply cap
        return min(vol_adjusted, max_position_cap)
    
    def compute_strategy_limits(
        self,
        strategy: str,
        realized_vol: float,
    ) -> Dict[str, float]:
        """
        Compute strategy-specific risk limits based on realized vol.
        
        Different strategies need different risk controls:
        - Momentum: moderate drawdown tolerance
        - Mean reversion: tighter stops, shorter horizon
        - Value: higher drawdown tolerance
        
        Args:
            strategy: Strategy name
            realized_vol: Realized portfolio volatility
            
        Returns:
            Dict with computed limits
        """
        # Base multipliers by strategy
        if strategy == "momentum":
            daily_loss_mult = 2.5
            weekly_dd_mult = 3.5
            stop_mult = 2.5
        elif strategy == "mean_reversion":
            daily_loss_mult = 2.0
            weekly_dd_mult = 3.0
            stop_mult = 2.0
        elif strategy in ["value", "quality_ls"]:
            daily_loss_mult = 2.5
            weekly_dd_mult = 4.0
            stop_mult = 3.0
        elif strategy == "low_vol":
            daily_loss_mult = 2.0
            weekly_dd_mult = 3.0
            stop_mult = 2.0
        else:
            daily_loss_mult = 2.5
            weekly_dd_mult = 3.5
            stop_mult = 2.5
        
        # Compute limits from realized vol
        expected_daily_vol = realized_vol / np.sqrt(252)
        
        return {
            "max_daily_loss_pct": daily_loss_mult * expected_daily_vol,
            "max_weekly_drawdown_pct": weekly_dd_mult * expected_daily_vol,
            "stop_loss_pct": stop_mult * expected_daily_vol,
        }
    
    def check_intent(
        self,
        intent: "PortfolioIntent",
        fund: "Fund",
        current_portfolio: "FundPortfolio"
    ) -> RiskCheckResult:
        """
        Stage 1: Pre-trade validation of PortfolioIntent.
        
        Checks:
        - Cooldown status
        - Position size limits
        - Gross exposure limits
        - Cash buffer requirements
        """
        applied_rules: List[str] = []
        violations: List[RiskViolation] = []
        
        # Check cooldown first
        risk_state = self.risk_state_repo.get(fund.fund_id)
        if risk_state and risk_state.is_in_cooldown():
            return RiskCheckResult(
                status="vetoed",
                scale_factor=0.0,
                violations=[RiskViolation(
                    rule_name="cooldown",
                    symbol=None,
                    limit=0,
                    actual=0,
                    severity="veto"
                )],
                applied_rules=["cooldown"],
                reason=f"In cooldown until {risk_state.risk_off_until}"
            )
        applied_rules.append("cooldown")
        
        # Check exposure limits
        valid, error = intent.validate_exposures(
            fund.policy.max_gross_exposure,
            fund.policy.min_cash_buffer
        )
        if not valid:
            violations.append(RiskViolation(
                rule_name="exposure_limit",
                symbol=None,
                limit=fund.policy.max_gross_exposure,
                actual=sum(abs(p.target_weight) for p in intent.positions),
                severity="veto"
            ))
            return RiskCheckResult(
                status="vetoed",
                scale_factor=0.0,
                violations=violations,
                applied_rules=applied_rules + ["exposure_limit"],
                reason=error
            )
        applied_rules.append("exposure_limit")
        
        # Check individual position limits
        scale_factor = 1.0
        per_symbol_scales: Dict[str, float] = {}
        
        for pos in intent.positions:
            if abs(pos.target_weight) > fund.risk_limits.max_position_pct:
                violations.append(RiskViolation(
                    rule_name="position_size",
                    symbol=pos.symbol,
                    limit=fund.risk_limits.max_position_pct,
                    actual=abs(pos.target_weight),
                    severity="scale"
                ))
                # Calculate scale to bring within limit
                symbol_scale = fund.risk_limits.max_position_pct / abs(pos.target_weight)
                per_symbol_scales[pos.symbol] = symbol_scale
                scale_factor = min(scale_factor, symbol_scale)
        applied_rules.append("position_size")
        
        # Check max positions
        if len(intent.positions) > fund.policy.max_positions:
            violations.append(RiskViolation(
                rule_name="max_positions",
                symbol=None,
                limit=fund.policy.max_positions,
                actual=len(intent.positions),
                severity="warning"  # Warning only, not scaling
            ))
        applied_rules.append("max_positions")
        
        if violations and scale_factor < 1.0:
            return RiskCheckResult(
                status="scaled",
                scale_factor=scale_factor,
                per_symbol_scales=per_symbol_scales,
                violations=violations,
                applied_rules=applied_rules,
                reason="Position sizes scaled to comply with limits"
            )
        
        if violations:
            return RiskCheckResult(
                status="approved",
                scale_factor=1.0,
                violations=violations,  # Warnings only
                applied_rules=applied_rules,
                reason=None
            )
        
        return RiskCheckResult(
            status="approved",
            scale_factor=1.0,
            violations=[],
            applied_rules=applied_rules,
            reason=None
        )
    
    def check_orders(
        self,
        orders: List["Order"],
        fund: "Fund",
        current_portfolio: "FundPortfolio"
    ) -> RiskCheckResult:
        """
        Stage 2: Validation of concrete orders before sending.
        
        Orders can violate limits due to rounding or slippage.
        """
        violations: List[RiskViolation] = []
        applied_rules: List[str] = []
        
        portfolio_value = current_portfolio.total_value
        if portfolio_value <= 0:
            return RiskCheckResult(
                status="vetoed",
                scale_factor=0.0,
                violations=[RiskViolation(
                    rule_name="portfolio_value",
                    symbol=None,
                    limit=0,
                    actual=portfolio_value,
                    severity="veto"
                )],
                applied_rules=["portfolio_value"],
                reason="Portfolio value is zero or negative"
            )
        
        for order in orders:
            # Calculate projected weight from order
            projected_value = order.quantity * order.expected_price
            projected_weight = projected_value / portfolio_value
            
            tolerance = 1.0 + CONSTANTS.risk.ORDER_SIZE_TOLERANCE
            max_allowed = fund.risk_limits.max_position_pct * tolerance
            
            if projected_weight > max_allowed:
                violations.append(RiskViolation(
                    rule_name="order_size_exceeds_limit",
                    symbol=order.symbol,
                    limit=fund.risk_limits.max_position_pct,
                    actual=projected_weight,
                    severity="veto"
                ))
        applied_rules.append("order_size_check")
        
        if any(v.severity == "veto" for v in violations):
            return RiskCheckResult(
                status="vetoed",
                scale_factor=0.0,
                violations=violations,
                applied_rules=applied_rules,
                reason="Order validation failed"
            )
        
        return RiskCheckResult(
            status="approved",
            scale_factor=1.0,
            violations=violations,
            applied_rules=applied_rules,
            reason=None
        )
    
    def check_post_fill(
        self,
        fills: List[Fill],
        fund: "Fund",
        new_portfolio: "FundPortfolio"
    ) -> PostFillAction:
        """
        Stage 3: Post-execution validation.
        
        Checks circuit breakers and may trigger go_flat.
        Returns quantity-based close orders (no stale prices).
        """
        # Update P&L tracking
        self._update_pnl_tracking(fund, new_portfolio)
        
        # Check circuit breakers
        risk_state = self.risk_state_repo.get(fund.fund_id)
        if risk_state and self._circuit_breaker_tripped(fund, risk_state):
            self._enter_cooldown(fund, "circuit_breaker")
            
            if fund.policy.go_flat_on_circuit_breaker:
                # Generate quantity-based close orders (no prices needed)
                close_orders = self._generate_close_orders(new_portfolio)
                return PostFillAction(
                    action="go_flat",
                    reason="Circuit breaker tripped - closing all positions",
                    close_orders=close_orders
                )
            else:
                return PostFillAction(
                    action="halt",
                    reason="Circuit breaker tripped - halting trading",
                    close_orders=[]
                )
        
        return PostFillAction(action="none", reason=None, close_orders=[])
    
    def _update_pnl_tracking(
        self,
        fund: "Fund",
        portfolio: "FundPortfolio"
    ) -> None:
        """Update P&L tracking for circuit breakers."""
        risk_state = self.risk_state_repo.get(fund.fund_id)
        
        if risk_state is None:
            risk_state = FundRiskState(
                fund_id=fund.fund_id,
                peak_nav=portfolio.total_value
            )
        
        current_nav = portfolio.total_value
        
        # Update peak NAV
        if current_nav > risk_state.peak_nav:
            risk_state.peak_nav = current_nav
        
        # Calculate drawdown from peak
        if risk_state.peak_nav > 0:
            drawdown = (risk_state.peak_nav - current_nav) / risk_state.peak_nav
            risk_state.current_weekly_drawdown_pct = drawdown
        
        self.risk_state_repo.upsert(risk_state)
    
    def _circuit_breaker_tripped(
        self,
        fund: "Fund",
        risk_state: FundRiskState
    ) -> bool:
        """Check if any circuit breaker has been tripped."""
        # Daily loss limit
        if abs(risk_state.current_daily_pnl_pct) > fund.risk_limits.max_daily_loss_pct:
            return True
        
        # Weekly drawdown limit
        if risk_state.current_weekly_drawdown_pct > fund.risk_limits.max_weekly_drawdown_pct:
            return True
        
        return False
    
    def _enter_cooldown(self, fund: "Fund", reason: str) -> None:
        """Enter cooldown mode."""
        cooldown_until = datetime.utcnow() + timedelta(
            days=fund.risk_limits.breach_cooldown_days
        )
        
        risk_state = self.risk_state_repo.get(fund.fund_id)
        if risk_state is None:
            risk_state = FundRiskState(fund_id=fund.fund_id)
        
        risk_state.risk_off_until = cooldown_until
        risk_state.last_breach_reason = reason
        risk_state.last_breach_time = datetime.utcnow()
        
        self.risk_state_repo.upsert(risk_state)
    
    def _generate_close_orders(
        self,
        portfolio: "FundPortfolio"
    ) -> List[CloseOrder]:
        """
        Generate quantity-based close orders.
        
        FIXED: No price calculation needed - market orders at current quantity.
        This avoids stale price issues in go_flat.
        """
        close_orders: List[CloseOrder] = []
        
        for symbol, position in portfolio.positions.items():
            if position.quantity == 0:
                continue
            
            # Close longs by selling, close shorts by buying
            if position.quantity > 0:
                close_orders.append(CloseOrder(
                    symbol=symbol,
                    quantity=position.quantity,
                    side="sell",
                    order_type="market",
                    urgency="immediate",
                    max_slippage_bps=CONSTANTS.risk.EMERGENCY_CLOSE_SLIPPAGE_BPS
                ))
            else:
                close_orders.append(CloseOrder(
                    symbol=symbol,
                    quantity=abs(position.quantity),
                    side="buy",
                    order_type="market",
                    urgency="immediate",
                    max_slippage_bps=CONSTANTS.risk.EMERGENCY_CLOSE_SLIPPAGE_BPS
                ))
        
        return close_orders
    
    def get_risk_state(self, fund_id: str) -> Optional[FundRiskState]:
        """Get current risk state for a fund."""
        return self.risk_state_repo.get(fund_id)
    
    def clear_cooldown(self, fund_id: str) -> None:
        """Clear cooldown for a fund (for testing)."""
        risk_state = self.risk_state_repo.get(fund_id)
        if risk_state:
            risk_state.risk_off_until = None
            self.risk_state_repo.upsert(risk_state)
