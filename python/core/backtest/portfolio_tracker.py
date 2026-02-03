"""
Portfolio Tracker for Backtesting.

Tracks fund portfolios, positions, and performance metrics over time.
"""

from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Dict, List, Optional
import numpy as np
import pandas as pd


@dataclass
class BacktestPosition:
    """A position held in a backtest portfolio."""
    symbol: str
    quantity: float
    avg_entry_price: float
    current_price: float = 0.0
    opened_at: Optional[date] = None
    
    @property
    def current_value(self) -> float:
        return self.quantity * self.current_price
    
    @property
    def cost_basis(self) -> float:
        return self.quantity * self.avg_entry_price
    
    @property
    def unrealized_pnl(self) -> float:
        return self.current_value - self.cost_basis
    
    @property
    def unrealized_return(self) -> float:
        if self.cost_basis == 0:
            return 0.0
        return self.unrealized_pnl / self.cost_basis


@dataclass
class BacktestTrade:
    """A trade executed in backtest."""
    trade_id: str
    fund_id: str
    symbol: str
    side: str  # "buy" or "sell"
    quantity: float
    price: float
    commission: float
    timestamp: datetime
    reasoning: str = ""
    
    @property
    def total_cost(self) -> float:
        """Total cost including commission."""
        if self.side == "buy":
            return self.quantity * self.price + self.commission
        else:
            return -(self.quantity * self.price - self.commission)


@dataclass
class BacktestPortfolio:
    """
    Tracks a fund's portfolio state during backtesting.
    
    Each fund starts with $100,000 and tracks all positions,
    trades, and performance metrics.
    """
    fund_id: str
    initial_cash: float = 100_000.0
    cash: float = field(default=100_000.0)
    positions: Dict[str, BacktestPosition] = field(default_factory=dict)
    
    # Performance tracking
    total_value_history: List[float] = field(default_factory=list)
    date_history: List[date] = field(default_factory=list)
    peak_value: float = 100_000.0
    max_drawdown: float = 0.0
    
    # Trade tracking
    all_trades: List[BacktestTrade] = field(default_factory=list)
    total_commissions: float = 0.0
    
    def __post_init__(self):
        """Initialize cash to initial_cash if not set."""
        if self.cash == 100_000.0 and self.initial_cash != 100_000.0:
            self.cash = self.initial_cash
    
    @property
    def total_value(self) -> float:
        """Total portfolio value (cash + positions)."""
        position_value = sum(p.current_value for p in self.positions.values())
        return self.cash + position_value
    
    @property
    def invested_value(self) -> float:
        """Total value invested in positions."""
        return sum(p.current_value for p in self.positions.values())
    
    @property
    def invested_pct(self) -> float:
        """Percentage of portfolio invested."""
        total = self.total_value
        if total == 0:
            return 0.0
        return self.invested_value / total
    
    @property
    def daily_return(self) -> float:
        """Return since last snapshot."""
        if len(self.total_value_history) < 2:
            return 0.0
        prev = self.total_value_history[-1]
        if prev == 0:
            return 0.0
        return (self.total_value - prev) / prev
    
    @property
    def cumulative_return(self) -> float:
        """Return since inception."""
        if self.initial_cash == 0:
            return 0.0
        return (self.total_value - self.initial_cash) / self.initial_cash
    
    @property
    def sharpe_ratio(self) -> float:
        """
        Rolling Sharpe ratio (annualized).
        
        Uses last 252 days if available.
        """
        if len(self.total_value_history) < 20:
            return 0.0
        
        values = pd.Series(self.total_value_history)
        returns = values.pct_change().dropna()
        
        if len(returns) < 20:
            return 0.0
        
        # Use last 252 days max
        recent = returns.iloc[-252:] if len(returns) > 252 else returns
        
        mean_ret = recent.mean()
        std_ret = recent.std()
        
        if std_ret == 0:
            return 0.0
        
        # Annualize
        return (mean_ret / std_ret) * np.sqrt(252)
    
    def update_prices(self, prices: Dict[str, float]) -> None:
        """
        Update all position prices (mark-to-market).
        
        Args:
            prices: Dict of symbol -> current price
        """
        for symbol, position in self.positions.items():
            if symbol in prices:
                position.current_price = prices[symbol]
    
    def record_snapshot(self, current_date: date) -> None:
        """
        Record daily snapshot for performance tracking.
        
        Call this after update_prices() each day.
        """
        current_value = self.total_value
        self.total_value_history.append(current_value)
        self.date_history.append(current_date)
        
        # Update peak and drawdown
        if current_value > self.peak_value:
            self.peak_value = current_value
        
        if self.peak_value > 0:
            drawdown = (self.peak_value - current_value) / self.peak_value
            if drawdown > self.max_drawdown:
                self.max_drawdown = drawdown
    
    def execute_buy(
        self,
        symbol: str,
        quantity: float,
        price: float,
        commission: float,
        timestamp: datetime,
        reasoning: str = "",
    ) -> BacktestTrade:
        """
        Execute a buy order.
        
        Returns:
            The trade that was executed
        """
        total_cost = quantity * price + commission
        
        if total_cost > self.cash:
            raise ValueError(
                f"Insufficient cash: need ${total_cost:.2f}, have ${self.cash:.2f}"
            )
        
        # Update cash
        self.cash -= total_cost
        self.total_commissions += commission
        
        # Update or create position
        if symbol in self.positions:
            pos = self.positions[symbol]
            # Average in
            total_qty = pos.quantity + quantity
            total_cost_basis = pos.cost_basis + (quantity * price)
            pos.avg_entry_price = total_cost_basis / total_qty
            pos.quantity = total_qty
            pos.current_price = price
        else:
            self.positions[symbol] = BacktestPosition(
                symbol=symbol,
                quantity=quantity,
                avg_entry_price=price,
                current_price=price,
                opened_at=timestamp.date() if isinstance(timestamp, datetime) else timestamp,
            )
        
        # Record trade
        trade = BacktestTrade(
            trade_id=f"{self.fund_id}_{symbol}_{timestamp.isoformat()}",
            fund_id=self.fund_id,
            symbol=symbol,
            side="buy",
            quantity=quantity,
            price=price,
            commission=commission,
            timestamp=timestamp,
            reasoning=reasoning,
        )
        self.all_trades.append(trade)
        
        return trade
    
    def execute_sell(
        self,
        symbol: str,
        quantity: float,
        price: float,
        commission: float,
        timestamp: datetime,
        reasoning: str = "",
    ) -> BacktestTrade:
        """
        Execute a sell order.
        
        Returns:
            The trade that was executed
        """
        if symbol not in self.positions:
            raise ValueError(f"No position in {symbol} to sell")
        
        pos = self.positions[symbol]
        if quantity > pos.quantity:
            raise ValueError(
                f"Cannot sell {quantity} {symbol}, only have {pos.quantity}"
            )
        
        # Calculate proceeds
        proceeds = quantity * price - commission
        
        # Update cash
        self.cash += proceeds
        self.total_commissions += commission
        
        # Update position
        pos.quantity -= quantity
        pos.current_price = price
        
        # Remove position if fully closed
        if pos.quantity <= 0:
            del self.positions[symbol]
        
        # Record trade
        trade = BacktestTrade(
            trade_id=f"{self.fund_id}_{symbol}_{timestamp.isoformat()}",
            fund_id=self.fund_id,
            symbol=symbol,
            side="sell",
            quantity=quantity,
            price=price,
            commission=commission,
            timestamp=timestamp,
            reasoning=reasoning,
        )
        self.all_trades.append(trade)
        
        return trade
    
    def get_position_weight(self, symbol: str) -> float:
        """Get position weight as fraction of total portfolio."""
        if symbol not in self.positions:
            return 0.0
        
        total = self.total_value
        if total == 0:
            return 0.0
        
        return self.positions[symbol].current_value / total
    
    def get_allocation(self) -> Dict[str, float]:
        """Get current allocation as weights."""
        total = self.total_value
        if total == 0:
            return {"cash": 1.0}
        
        allocation = {"cash": self.cash / total}
        for symbol, pos in self.positions.items():
            allocation[symbol] = pos.current_value / total
        
        return allocation
    
    def summary(self) -> Dict:
        """Get portfolio summary."""
        return {
            "fund_id": self.fund_id,
            "total_value": self.total_value,
            "cash": self.cash,
            "invested_pct": self.invested_pct,
            "cumulative_return": self.cumulative_return,
            "max_drawdown": self.max_drawdown,
            "sharpe_ratio": self.sharpe_ratio,
            "n_positions": len(self.positions),
            "n_trades": len(self.all_trades),
            "total_commissions": self.total_commissions,
        }
    
    def to_snapshot_dict(self) -> Dict:
        """Convert to dict for database storage."""
        return {
            "fund_id": self.fund_id,
            "cash": self.cash,
            "positions": {
                sym: {
                    "quantity": pos.quantity,
                    "avg_entry_price": pos.avg_entry_price,
                    "current_price": pos.current_price,
                    "current_value": pos.current_value,
                }
                for sym, pos in self.positions.items()
            },
            "total_value": self.total_value,
            "daily_return": self.daily_return,
            "cumulative_return": self.cumulative_return,
            "max_drawdown": self.max_drawdown,
        }
