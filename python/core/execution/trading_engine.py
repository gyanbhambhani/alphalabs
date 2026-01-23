"""
Trading Engine

Orchestrates the trading cycle for all managers.
"""
import asyncio
from datetime import datetime, time
from typing import Dict, List, Optional
from dataclasses import dataclass

from core.managers.base import (
    BaseManager, TradingDecision, ManagerContext, 
    Portfolio, Position, StrategySignals, Action
)
from core.managers.quant_bot import QuantBot
from core.managers.llm_manager import (
    LLMManager, create_gpt4_manager, 
    create_claude_manager, create_gemini_manager
)
from core.execution.alpaca_client import AlpacaClient
from core.execution.risk_manager import RiskManager


@dataclass
class TradeResult:
    """Result of a trade execution"""
    manager_id: str
    decision: TradingDecision
    success: bool
    order_id: Optional[str] = None
    filled_price: Optional[float] = None
    error: Optional[str] = None


class TradingEngine:
    """
    Main trading engine that coordinates all managers.
    
    Responsibilities:
    - Run trading cycles at intervals
    - Collect strategy signals
    - Distribute context to managers
    - Execute approved trades
    - Track performance
    """
    
    def __init__(
        self,
        alpaca_client: AlpacaClient,
        risk_manager: RiskManager,
        initial_capital_per_manager: float = 25000.0
    ):
        """
        Initialize trading engine.
        
        Args:
            alpaca_client: Alpaca trading client
            risk_manager: Risk manager
            initial_capital_per_manager: Starting capital per manager
        """
        self.alpaca = alpaca_client
        self.risk_manager = risk_manager
        self.initial_capital = initial_capital_per_manager
        
        # Initialize managers
        self.managers: Dict[str, BaseManager] = {}
        self._initialize_managers()
        
        # Track start of day values
        self._start_of_day_values: Dict[str, float] = {}
    
    def _initialize_managers(self) -> None:
        """Create all 4 portfolio managers"""
        # GPT-4 Manager
        self.managers["gpt4"] = create_gpt4_manager(
            initial_capital=self.initial_capital
        )
        
        # Claude Manager
        self.managers["claude"] = create_claude_manager(
            initial_capital=self.initial_capital
        )
        
        # Gemini Manager
        self.managers["gemini"] = create_gemini_manager(
            initial_capital=self.initial_capital
        )
        
        # Quant Bot (baseline)
        self.managers["quant"] = QuantBot(
            initial_capital=self.initial_capital
        )
    
    def _build_context(
        self,
        manager: BaseManager,
        market_data: Dict[str, float],
        signals: StrategySignals
    ) -> ManagerContext:
        """Build trading context for a manager"""
        return ManagerContext(
            timestamp=datetime.utcnow(),
            portfolio=manager.portfolio,
            market_data=market_data,
            signals=signals
        )
    
    async def _execute_decision(
        self,
        manager: BaseManager,
        decision: TradingDecision,
        current_prices: Dict[str, float]
    ) -> TradeResult:
        """Execute a single trading decision"""
        symbol = decision.symbol
        
        if symbol not in current_prices:
            return TradeResult(
                manager_id=manager.manager_id,
                decision=decision,
                success=False,
                error=f"No price available for {symbol}"
            )
        
        price = current_prices[symbol]
        
        # Calculate quantity
        if decision.action == Action.BUY:
            # Calculate shares based on position size
            position_value = manager.portfolio.total_value * decision.size
            quantity = position_value / price
            
            # Check if we have enough cash
            if position_value > manager.portfolio.cash_balance:
                # Reduce to available cash
                quantity = manager.portfolio.cash_balance * 0.95 / price
            
            if quantity < 1:
                return TradeResult(
                    manager_id=manager.manager_id,
                    decision=decision,
                    success=False,
                    error="Insufficient funds for minimum position"
                )
            
            side = "buy"
        
        elif decision.action == Action.SELL:
            # Sell existing position
            if symbol not in manager.portfolio.positions:
                return TradeResult(
                    manager_id=manager.manager_id,
                    decision=decision,
                    success=False,
                    error=f"No position in {symbol} to sell"
                )
            
            position = manager.portfolio.positions[symbol]
            quantity = position.quantity * decision.size
            side = "sell"
        
        else:
            return TradeResult(
                manager_id=manager.manager_id,
                decision=decision,
                success=True  # Hold is always successful
            )
        
        # Submit order
        order = self.alpaca.submit_market_order(
            symbol=symbol,
            quantity=quantity,
            side=side
        )
        
        if order is None:
            return TradeResult(
                manager_id=manager.manager_id,
                decision=decision,
                success=False,
                error="Order submission failed"
            )
        
        # Update manager's portfolio
        if side == "buy":
            manager.update_position(symbol, quantity, price)
            manager.portfolio.cash_balance -= quantity * price
        else:
            manager.update_position(symbol, -quantity, price)
            manager.portfolio.cash_balance += quantity * price
        
        # Record trade
        self.risk_manager.record_trade(manager.manager_id)
        
        return TradeResult(
            manager_id=manager.manager_id,
            decision=decision,
            success=True,
            order_id=order.id,
            filled_price=price
        )
    
    async def run_cycle(
        self,
        market_data: Dict[str, float],
        signals: StrategySignals
    ) -> List[TradeResult]:
        """
        Run one trading cycle for all managers.
        
        Args:
            market_data: Current prices for all symbols
            signals: Strategy signals from toolbox
        
        Returns:
            List of TradeResult for all executed trades
        """
        results = []
        
        # Update positions with current prices
        for manager in self.managers.values():
            for symbol, pos in manager.portfolio.positions.items():
                if symbol in market_data:
                    pos.current_price = market_data[symbol]
        
        # Get decisions from each manager
        for manager_id, manager in self.managers.items():
            # Build context
            context = self._build_context(manager, market_data, signals)
            
            # Get decisions
            decisions = await manager.make_decisions(context)
            
            # Check and execute each decision
            for decision in decisions:
                # Risk check
                start_value = self._start_of_day_values.get(
                    manager_id, 
                    manager.portfolio.total_value
                )
                
                risk_result = self.risk_manager.check_decision(
                    decision=decision,
                    manager_id=manager_id,
                    portfolio=manager.portfolio,
                    start_of_day_value=start_value
                )
                
                if not risk_result.approved:
                    results.append(TradeResult(
                        manager_id=manager_id,
                        decision=decision,
                        success=False,
                        error=risk_result.reason
                    ))
                    continue
                
                # Use modified decision if risk manager adjusted it
                final_decision = risk_result.modified_decision or decision
                
                # Execute
                result = await self._execute_decision(
                    manager, final_decision, market_data
                )
                results.append(result)
        
        return results
    
    def start_of_day(self) -> None:
        """Called at start of trading day"""
        self.risk_manager.reset_daily()
        
        for manager_id, manager in self.managers.items():
            self._start_of_day_values[manager_id] = manager.portfolio.total_value
            manager.reset_daily_stats()
    
    def get_leaderboard(self) -> List[Dict]:
        """Get current leaderboard sorted by portfolio value"""
        entries = []
        
        for manager_id, manager in self.managers.items():
            entries.append({
                "manager_id": manager_id,
                "name": manager.name,
                "type": manager.manager_type,
                "total_value": manager.portfolio.total_value,
                "cash": manager.portfolio.cash_balance,
                "positions": len(manager.portfolio.positions),
                "invested_pct": manager.portfolio.invested_pct
            })
        
        # Sort by total value
        entries.sort(key=lambda x: x["total_value"], reverse=True)
        
        # Add ranks
        for i, entry in enumerate(entries):
            entry["rank"] = i + 1
        
        return entries
    
    def get_manager_portfolio(self, manager_id: str) -> Optional[Dict]:
        """Get portfolio details for a manager"""
        if manager_id not in self.managers:
            return None
        
        manager = self.managers[manager_id]
        
        return {
            "manager_id": manager_id,
            "name": manager.name,
            "type": manager.manager_type,
            "cash_balance": manager.portfolio.cash_balance,
            "total_value": manager.portfolio.total_value,
            "positions": [
                {
                    "symbol": symbol,
                    "quantity": pos.quantity,
                    "avg_entry_price": pos.avg_entry_price,
                    "current_price": pos.current_price,
                    "market_value": pos.market_value,
                    "unrealized_pnl": pos.unrealized_pnl
                }
                for symbol, pos in manager.portfolio.positions.items()
            ]
        }
