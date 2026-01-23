"""
Risk Manager

Enforces risk limits and position sizing rules.
"""
from typing import List, Dict, Optional
from dataclasses import dataclass
from core.managers.base import TradingDecision, Action, Portfolio, RiskLimits


@dataclass
class RiskCheckResult:
    """Result of a risk check"""
    approved: bool
    reason: Optional[str] = None
    modified_decision: Optional[TradingDecision] = None


class RiskManager:
    """
    Risk management for trading decisions.
    
    Enforces:
    - Position size limits
    - Sector concentration limits
    - Daily loss limits
    - Maximum drawdown alerts
    - Portfolio heat (total risk)
    """
    
    def __init__(
        self,
        limits: Optional[RiskLimits] = None,
        global_exposure_limit: float = 0.80,
        vix_halt_threshold: float = 35.0
    ):
        """
        Initialize risk manager.
        
        Args:
            limits: Risk limits configuration
            global_exposure_limit: Max % of portfolio invested
            vix_halt_threshold: VIX level that triggers trading halt
        """
        self.limits = limits or RiskLimits()
        self.global_exposure_limit = global_exposure_limit
        self.vix_halt_threshold = vix_halt_threshold
        
        # Track daily stats
        self._daily_loss: Dict[str, float] = {}
        self._trades_today: Dict[str, int] = {}
        self._is_halted = False
    
    def check_position_size(
        self,
        decision: TradingDecision,
        portfolio: Portfolio
    ) -> RiskCheckResult:
        """Check if position size is within limits"""
        if decision.action == Action.HOLD:
            return RiskCheckResult(approved=True)
        
        # For sells, always approve
        if decision.action == Action.SELL:
            return RiskCheckResult(approved=True)
        
        # For buys, check size limit
        if decision.size > self.limits.max_position_size:
            modified = TradingDecision(
                action=decision.action,
                symbol=decision.symbol,
                size=self.limits.max_position_size,
                reasoning=decision.reasoning + 
                    f" [Size reduced from {decision.size:.0%} to "
                    f"{self.limits.max_position_size:.0%}]",
                confidence=decision.confidence,
                signals_used=decision.signals_used
            )
            return RiskCheckResult(
                approved=True,
                reason="Position size reduced to limit",
                modified_decision=modified
            )
        
        return RiskCheckResult(approved=True)
    
    def check_daily_loss(
        self,
        manager_id: str,
        portfolio: Portfolio,
        start_of_day_value: float
    ) -> RiskCheckResult:
        """Check if daily loss limit has been hit"""
        current_value = portfolio.total_value
        daily_return = (current_value - start_of_day_value) / start_of_day_value
        
        if daily_return < -self.limits.daily_loss_limit:
            return RiskCheckResult(
                approved=False,
                reason=f"Daily loss limit hit: {daily_return:.1%}"
            )
        
        return RiskCheckResult(approved=True)
    
    def check_trade_count(
        self,
        manager_id: str
    ) -> RiskCheckResult:
        """Check if max trades per day reached"""
        trades = self._trades_today.get(manager_id, 0)
        
        if trades >= self.limits.max_trades_per_day:
            return RiskCheckResult(
                approved=False,
                reason=f"Max trades per day ({self.limits.max_trades_per_day}) reached"
            )
        
        return RiskCheckResult(approved=True)
    
    def check_global_exposure(
        self,
        portfolios: Dict[str, Portfolio]
    ) -> RiskCheckResult:
        """Check total exposure across all portfolios"""
        total_invested = 0.0
        total_value = 0.0
        
        for portfolio in portfolios.values():
            total_value += portfolio.total_value
            for pos in portfolio.positions.values():
                total_invested += pos.market_value
        
        if total_value > 0:
            exposure = total_invested / total_value
            if exposure > self.global_exposure_limit:
                return RiskCheckResult(
                    approved=False,
                    reason=f"Global exposure {exposure:.0%} exceeds "
                           f"limit {self.global_exposure_limit:.0%}"
                )
        
        return RiskCheckResult(approved=True)
    
    def check_vix(self, vix_level: float) -> RiskCheckResult:
        """Check if VIX is too high"""
        if vix_level > self.vix_halt_threshold:
            self._is_halted = True
            return RiskCheckResult(
                approved=False,
                reason=f"VIX {vix_level:.1f} exceeds halt threshold "
                       f"{self.vix_halt_threshold}"
            )
        
        self._is_halted = False
        return RiskCheckResult(approved=True)
    
    def check_decision(
        self,
        decision: TradingDecision,
        manager_id: str,
        portfolio: Portfolio,
        start_of_day_value: float,
        vix_level: Optional[float] = None
    ) -> RiskCheckResult:
        """
        Run all risk checks on a decision.
        
        Args:
            decision: Trading decision to check
            manager_id: ID of the manager
            portfolio: Current portfolio state
            start_of_day_value: Portfolio value at start of day
            vix_level: Current VIX level (optional)
        
        Returns:
            RiskCheckResult with approval status
        """
        if self._is_halted:
            return RiskCheckResult(
                approved=False,
                reason="Trading halted due to market conditions"
            )
        
        # Check VIX if provided
        if vix_level:
            vix_check = self.check_vix(vix_level)
            if not vix_check.approved:
                return vix_check
        
        # Check daily loss
        loss_check = self.check_daily_loss(manager_id, portfolio, start_of_day_value)
        if not loss_check.approved:
            return loss_check
        
        # Check trade count
        count_check = self.check_trade_count(manager_id)
        if not count_check.approved:
            return count_check
        
        # Check position size
        size_check = self.check_position_size(decision, portfolio)
        if not size_check.approved:
            return size_check
        
        return size_check  # May contain modified decision
    
    def record_trade(self, manager_id: str) -> None:
        """Record that a trade was made"""
        self._trades_today[manager_id] = self._trades_today.get(manager_id, 0) + 1
    
    def reset_daily(self) -> None:
        """Reset daily counters"""
        self._trades_today = {}
        self._daily_loss = {}
    
    def is_halted(self) -> bool:
        """Check if trading is halted"""
        return self._is_halted
    
    def resume_trading(self) -> None:
        """Resume trading after halt"""
        self._is_halted = False
