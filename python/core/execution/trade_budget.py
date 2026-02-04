"""
Trade Budget Control Plane

Enforces trade frequency limits BEFORE LLM proposals.
Makes violations impossible through deterministic gating.

Key principles:
- Check budget BEFORE debate (not after)
- Rebalance cadence as code (not LLM's choice)
- Hysteresis to prevent micro-adjustments
- Count rebalance events (not individual orders)
"""

from dataclasses import dataclass
from datetime import date, timedelta
from typing import Optional, Dict, Any, List
import logging

logger = logging.getLogger(__name__)


@dataclass
class TradeBudget:
    """
    Authoritative trade budget - gates action space BEFORE proposals.
    
    Makes trade frequency violations impossible by:
    1. Checking budget before LLM debate
    2. Forcing action space to {hold, sell_only} when exhausted
    3. Hard validator rejection as final gate
    """
    fund_id: str
    current_date: date
    portfolio_value: float
    
    # Weekly budget
    trades_this_week: int = 0
    max_trades_per_week: int = 3
    
    # Rebalance schedule (deterministic, not vibes)
    rebalance_cadence: str = "weekly"  # "daily", "weekly", "monthly"
    last_rebalance_date: Optional[date] = None
    next_rebalance_date: Optional[date] = None
    
    # Hysteresis (prevent spam micro-adjustments)
    min_weight_delta: float = 0.02  # 2% minimum change to trade
    min_order_notional: float = 1000.0  # $1,000 minimum order size
    
    # Cooldown (circuit breaker override)
    cooldown_until: Optional[date] = None
    
    def __post_init__(self):
        """Compute next rebalance date if not provided."""
        if self.next_rebalance_date is None and self.last_rebalance_date:
            self.next_rebalance_date = self._compute_next_rebalance()
    
    def can_buy(self) -> bool:
        """
        Can we buy new positions?
        
        Checked BEFORE LLM proposal to gate action space.
        
        Returns:
            True if buys are authorized
        """
        # Cooldown override (circuit breaker)
        if self.cooldown_until and self.current_date < self.cooldown_until:
            logger.debug(
                f"[{self.fund_id}] Buys DENIED - cooldown until "
                f"{self.cooldown_until}"
            )
            return False
        
        # Budget exhausted
        if self.trades_this_week >= self.max_trades_per_week:
            logger.debug(
                f"[{self.fund_id}] Buys DENIED - budget exhausted "
                f"({self.trades_this_week}/{self.max_trades_per_week})"
            )
            return False
        
        # Not a rebalance day
        if not self._is_rebalance_day():
            logger.debug(
                f"[{self.fund_id}] Buys DENIED - not rebalance day "
                f"(next: {self.next_rebalance_date})"
            )
            return False
        
        return True
    
    def can_sell(self) -> bool:
        """
        Can we sell positions?
        
        Always True - even in cooldown, we can exit positions.
        
        Returns:
            True (always)
        """
        return True
    
    def can_rebalance(self) -> bool:
        """
        Is today a rebalance day?
        
        Alias for _is_rebalance_day() for external use.
        
        Returns:
            True if today is a rebalance day
        """
        return self._is_rebalance_day()
    
    def _is_rebalance_day(self) -> bool:
        """
        Is today a rebalance day? (deterministic, not LLM's choice)
        
        Returns:
            True if rebalance is allowed today
        """
        if self.rebalance_cadence == "daily":
            return True
        
        if self.last_rebalance_date is None:
            # First rebalance ever
            return True
        
        days_since_last = (self.current_date - self.last_rebalance_date).days
        
        if self.rebalance_cadence == "weekly":
            # Rebalance every Monday (or 5 trading days)
            return days_since_last >= 5
        
        if self.rebalance_cadence == "monthly":
            # Rebalance every ~20 trading days
            return days_since_last >= 20
        
        if self.rebalance_cadence == "quarterly":
            # Rebalance every ~60 trading days
            return days_since_last >= 60
        
        logger.warning(
            f"Unknown rebalance cadence: {self.rebalance_cadence}, "
            f"defaulting to daily"
        )
        return True
    
    def _compute_next_rebalance(self) -> date:
        """Compute next rebalance date based on cadence."""
        if self.last_rebalance_date is None:
            return self.current_date
        
        if self.rebalance_cadence == "daily":
            return self.current_date + timedelta(days=1)
        elif self.rebalance_cadence == "weekly":
            return self.last_rebalance_date + timedelta(days=7)
        elif self.rebalance_cadence == "monthly":
            return self.last_rebalance_date + timedelta(days=30)
        elif self.rebalance_cadence == "quarterly":
            return self.last_rebalance_date + timedelta(days=90)
        else:
            return self.current_date + timedelta(days=1)
    
    def should_trade(
        self,
        current_weight: float,
        target_weight: float,
        price: float
    ) -> bool:
        """
        Apply hysteresis - only trade if change is meaningful.
        
        Prevents spam micro-adjustments that waste money on commissions.
        
        Args:
            current_weight: Current portfolio weight (0-1)
            target_weight: Desired portfolio weight (0-1)
            price: Current asset price
            
        Returns:
            True if trade should execute
        """
        delta = abs(target_weight - current_weight)
        
        # Change too small to matter
        if delta < self.min_weight_delta:
            logger.debug(
                f"Trade skipped - delta {delta:.3f} < min "
                f"{self.min_weight_delta:.3f}"
            )
            return False
        
        # Check minimum notional value
        notional = delta * self.portfolio_value
        if notional < self.min_order_notional:
            logger.debug(
                f"Trade skipped - notional ${notional:.0f} < min "
                f"${self.min_order_notional:.0f}"
            )
            return False
        
        return True
    
    def consume_trade_event(self) -> None:
        """
        Consume one trade from budget.
        
        Call this when a rebalance event happens (even if multiple orders).
        """
        self.trades_this_week += 1
        self.last_rebalance_date = self.current_date
        self.next_rebalance_date = self._compute_next_rebalance()
        
        logger.info(
            f"[{self.fund_id}] Trade event consumed: "
            f"{self.trades_this_week}/{self.max_trades_per_week} used"
        )
    
    def reset_weekly_counter(self) -> None:
        """
        Reset weekly trade counter.
        
        Call this at the start of each trading week.
        """
        logger.info(
            f"[{self.fund_id}] Weekly budget reset: "
            f"{self.trades_this_week} trades last week"
        )
        self.trades_this_week = 0
    
    def to_context_string(self) -> str:
        """
        Format budget state for LLM context.
        
        This tells the LLM what actions are authorized.
        
        Returns:
            Formatted string for prompt
        """
        can_buy = self.can_buy()
        can_rebalance = self._is_rebalance_day()
        
        context = f"""TRADE BUDGET:
- Trades this week: {self.trades_this_week}/{self.max_trades_per_week}
- Buy authority: {"AUTHORIZED" if can_buy else "DENIED"}
- Sell authority: AUTHORIZED (always allowed)
- Rebalance window: {"OPEN" if can_rebalance else f"CLOSED until {self.next_rebalance_date}"}
"""
        
        if self.cooldown_until:
            context += f"- Cooldown: ACTIVE until {self.cooldown_until}\n"
        else:
            context += "- Cooldown: NONE\n"
        
        context += f"""
CONSTRAINTS:
- Minimum weight change: {self.min_weight_delta:.1%}
- Minimum order size: ${self.min_order_notional:,.0f}
"""
        
        if not can_buy:
            context += "\nIMPORTANT: You may only HOLD or SELL positions today.\n"
        
        return context
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize to dictionary for logging.
        
        Returns:
            Dictionary representation
        """
        return {
            "fund_id": self.fund_id,
            "current_date": self.current_date.isoformat(),
            "trades_this_week": self.trades_this_week,
            "max_trades_per_week": self.max_trades_per_week,
            "can_buy": self.can_buy(),
            "can_sell": self.can_sell(),
            "can_rebalance": self.can_rebalance(),
            "rebalance_cadence": self.rebalance_cadence,
            "last_rebalance_date": (
                self.last_rebalance_date.isoformat()
                if self.last_rebalance_date else None
            ),
            "next_rebalance_date": (
                self.next_rebalance_date.isoformat()
                if self.next_rebalance_date else None
            ),
            "cooldown_until": (
                self.cooldown_until.isoformat()
                if self.cooldown_until else None
            ),
            "min_weight_delta": self.min_weight_delta,
            "min_order_notional": self.min_order_notional,
        }
    
    @classmethod
    def from_fund_policy(
        cls,
        fund_id: str,
        current_date: date,
        portfolio_value: float,
        policy: "FundPolicy",
        trades_this_week: int = 0,
        last_rebalance_date: Optional[date] = None,
        cooldown_until: Optional[date] = None,
    ) -> "TradeBudget":
        """
        Create TradeBudget from FundPolicy.
        
        Args:
            fund_id: Fund identifier
            current_date: Current simulation date
            portfolio_value: Current portfolio value
            policy: FundPolicy with rebalance_cadence and limits
            trades_this_week: Current weekly trade count
            last_rebalance_date: Last rebalance date
            cooldown_until: Cooldown end date (if any)
            
        Returns:
            TradeBudget instance
        """
        return cls(
            fund_id=fund_id,
            current_date=current_date,
            portfolio_value=portfolio_value,
            trades_this_week=trades_this_week,
            max_trades_per_week=3,  # Standard limit
            rebalance_cadence=policy.rebalance_cadence,
            last_rebalance_date=last_rebalance_date,
            min_weight_delta=0.02,  # 2% hysteresis
            min_order_notional=1000.0,  # $1k minimum
            cooldown_until=cooldown_until,
        )


def count_rebalance_events(
    orders: List["Order"],
    start_date: date,
    end_date: date
) -> int:
    """
    Count rebalance events (trading days with activity), not individual orders.
    
    A rebalance event = any day with ≥1 order.
    If you buy 3 stocks and sell 2 on the same day → 1 event.
    
    Args:
        orders: List of orders to count
        start_date: Start of period
        end_date: End of period
        
    Returns:
        Number of unique trading days with orders
    """
    trading_days_with_activity = set()
    
    for order in orders:
        order_date = getattr(order, 'date', None) or getattr(order, 'created_at', None)
        if order_date and start_date <= order_date.date() <= end_date:
            trading_days_with_activity.add(order_date.date())
    
    return len(trading_days_with_activity)
