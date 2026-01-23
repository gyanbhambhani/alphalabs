"""
Base Manager Class

Abstract base class for all portfolio managers (LLM and Quant).
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from typing import Dict, List, Optional, Literal
from enum import Enum


class Action(Enum):
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"


@dataclass
class TradingDecision:
    """A single trading decision from a manager"""
    action: Action
    symbol: str
    size: float  # Percentage of portfolio (0-1)
    reasoning: str
    confidence: float = 0.5  # 0-1
    signals_used: Dict[str, float] = field(default_factory=dict)


@dataclass
class Position:
    """Current position in a security"""
    symbol: str
    quantity: float
    avg_entry_price: float
    current_price: float
    
    @property
    def market_value(self) -> float:
        return self.quantity * self.current_price
    
    @property
    def unrealized_pnl(self) -> float:
        return (self.current_price - self.avg_entry_price) * self.quantity
    
    @property
    def unrealized_pnl_pct(self) -> float:
        if self.avg_entry_price == 0:
            return 0.0
        return (self.current_price - self.avg_entry_price) / self.avg_entry_price


@dataclass
class Portfolio:
    """Portfolio state for a manager"""
    cash_balance: float
    positions: Dict[str, Position] = field(default_factory=dict)
    
    @property
    def total_value(self) -> float:
        positions_value = sum(p.market_value for p in self.positions.values())
        return self.cash_balance + positions_value
    
    @property
    def invested_pct(self) -> float:
        if self.total_value == 0:
            return 0.0
        positions_value = sum(p.market_value for p in self.positions.values())
        return positions_value / self.total_value


@dataclass
class StrategySignals:
    """All strategy signals from the toolbox"""
    momentum: Dict[str, float]  # symbol -> score
    mean_reversion: Dict[str, float]  # symbol -> score
    technical: Dict[str, dict]  # symbol -> indicators
    ml_prediction: Dict[str, float]  # symbol -> predicted return
    volatility_regime: str
    semantic_search: dict  # Semantic search results


@dataclass
class ManagerContext:
    """Full context provided to a manager for decision-making"""
    timestamp: datetime
    portfolio: Portfolio
    market_data: Dict[str, float]  # symbol -> current price
    signals: StrategySignals
    
    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp.isoformat(),
            "portfolio": {
                "cash_balance": self.portfolio.cash_balance,
                "total_value": self.portfolio.total_value,
                "positions": {
                    symbol: {
                        "quantity": pos.quantity,
                        "avg_entry_price": pos.avg_entry_price,
                        "current_price": pos.current_price,
                        "unrealized_pnl": pos.unrealized_pnl
                    }
                    for symbol, pos in self.portfolio.positions.items()
                }
            },
            "market_data": self.market_data,
            "signals": {
                "momentum": self.signals.momentum,
                "mean_reversion": self.signals.mean_reversion,
                "volatility_regime": self.signals.volatility_regime,
                "ml_prediction": self.signals.ml_prediction,
                "semantic_search": self.signals.semantic_search
            }
        }


@dataclass
class RiskLimits:
    """Risk limits for a manager"""
    max_position_size: float = 0.20  # 20% max in single stock
    max_sector_exposure: float = 0.40  # 40% max in single sector
    max_drawdown: float = 0.15  # -15% triggers review
    daily_loss_limit: float = 0.05  # -5% halts trading
    max_trades_per_day: int = 10


class BaseManager(ABC):
    """
    Abstract base class for portfolio managers.
    
    All managers (LLM and Quant) inherit from this class.
    """
    
    def __init__(
        self,
        manager_id: str,
        name: str,
        manager_type: Literal["llm", "quant"],
        initial_capital: float = 25000.0,
        risk_limits: Optional[RiskLimits] = None
    ):
        self.manager_id = manager_id
        self.name = name
        self.manager_type = manager_type
        self.risk_limits = risk_limits or RiskLimits()
        
        # Initialize portfolio
        self.portfolio = Portfolio(
            cash_balance=initial_capital,
            positions={}
        )
        
        # Track daily stats
        self.daily_trades = 0
        self.daily_pnl = 0.0
        self.start_of_day_value = initial_capital
    
    @abstractmethod
    async def make_decisions(
        self, 
        context: ManagerContext
    ) -> List[TradingDecision]:
        """
        Make trading decisions based on current context.
        
        This is the main method that each manager type implements differently.
        
        Args:
            context: Full trading context including portfolio, signals, market data
        
        Returns:
            List of TradingDecision objects
        """
        pass
    
    def apply_risk_limits(
        self, 
        decisions: List[TradingDecision],
        context: ManagerContext
    ) -> List[TradingDecision]:
        """
        Apply risk limits to filter/modify decisions.
        
        Args:
            decisions: Raw decisions from the manager
            context: Current trading context
        
        Returns:
            Filtered decisions that pass risk checks
        """
        approved_decisions = []
        
        for decision in decisions:
            # Skip holds
            if decision.action == Action.HOLD:
                continue
            
            # Check daily trade limit
            if self.daily_trades >= self.risk_limits.max_trades_per_day:
                continue
            
            # Check position size limit
            if decision.action == Action.BUY:
                if decision.size > self.risk_limits.max_position_size:
                    decision.size = self.risk_limits.max_position_size
                    decision.reasoning += (
                        f" [Size reduced to {self.risk_limits.max_position_size:.0%}"
                        f" due to position limit]"
                    )
            
            # Check daily loss limit
            daily_return = (
                (self.portfolio.total_value - self.start_of_day_value) 
                / self.start_of_day_value
            )
            if daily_return < -self.risk_limits.daily_loss_limit:
                continue  # Skip new trades if daily loss limit hit
            
            approved_decisions.append(decision)
        
        return approved_decisions
    
    def update_position(
        self,
        symbol: str,
        quantity_change: float,
        price: float
    ):
        """Update position after a trade"""
        if symbol in self.portfolio.positions:
            pos = self.portfolio.positions[symbol]
            if quantity_change > 0:  # Buy
                # Update average entry price
                total_cost = (
                    pos.quantity * pos.avg_entry_price 
                    + quantity_change * price
                )
                new_quantity = pos.quantity + quantity_change
                pos.avg_entry_price = total_cost / new_quantity
                pos.quantity = new_quantity
            else:  # Sell
                pos.quantity += quantity_change  # quantity_change is negative
                if pos.quantity <= 0:
                    del self.portfolio.positions[symbol]
            
            if symbol in self.portfolio.positions:
                self.portfolio.positions[symbol].current_price = price
        else:
            if quantity_change > 0:  # New position
                self.portfolio.positions[symbol] = Position(
                    symbol=symbol,
                    quantity=quantity_change,
                    avg_entry_price=price,
                    current_price=price
                )
    
    def reset_daily_stats(self):
        """Reset daily tracking stats"""
        self.daily_trades = 0
        self.daily_pnl = 0.0
        self.start_of_day_value = self.portfolio.total_value
    
    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"id={self.manager_id}, "
            f"name={self.name}, "
            f"value=${self.portfolio.total_value:,.2f})"
        )
