"""
Trading Loop Scheduler

Orchestrates daily trading cycles:
1. Fetch market data
2. Calculate strategy signals
3. Query managers for decisions
4. Execute trades
5. Update database
6. Calculate performance
"""
import asyncio
from datetime import datetime, time as dt_time, date
from typing import Dict, List, Optional
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, desc
from decimal import Decimal

from data.ingest import DataIngestion
from core.strategies.momentum import calculate_momentum_signals
from core.strategies.mean_reversion import calculate_mean_reversion_signals
from core.strategies.technical import calculate_all_technical_indicators
from core.strategies.volatility import detect_volatility_regime
from core.semantic.search import SemanticSearchEngine
from core.execution.trading_engine import TradingEngine
from core.execution.alpaca_client import AlpacaClient
from core.execution.risk_manager import RiskManager
from core.execution.performance import PerformanceTracker
from core.managers.base import StrategySignals
from db.database import get_async_session
from db.models import Manager, Portfolio, Trade, DailySnapshot
from app.config import get_settings


class TradingScheduler:
    """
    Main trading scheduler that runs daily cycles.
    
    Workflow:
    1. Start of day: Reset stats, fetch market data
    2. Calculate signals: All strategy toolbox signals
    3. Get decisions: Query each manager
    4. Execute: Submit orders, update portfolios
    5. End of day: Calculate performance, update database
    """
    
    def __init__(self):
        """Initialize scheduler"""
        settings = get_settings()
        
        # Components
        self.settings = settings
        self.data_ingestion = DataIngestion(years_of_history=10)
        self.semantic_engine = SemanticSearchEngine(
            persist_directory=str(settings.chroma_persist_directory),
            in_memory=False
        )
        
        # Trading components
        self.alpaca = AlpacaClient(
            api_key=settings.alpaca_api_key,
            secret_key=settings.alpaca_secret_key,
            paper=settings.alpaca_paper
        )
        self.risk_manager = RiskManager()
        self.trading_engine = TradingEngine(
            alpaca_client=self.alpaca,
            risk_manager=self.risk_manager,
            initial_capital_per_manager=settings.initial_capital
        )
        self.performance_tracker = PerformanceTracker(
            risk_free_rate=0.05
        )
        
        # State
        self._is_running = False
        self._last_cycle_time: Optional[datetime] = None
    
    def calculate_signals(
        self,
        price_data: Dict
    ) -> StrategySignals:
        """
        Calculate all strategy signals from the toolbox.
        
        Args:
            price_data: Dict of symbol -> StockData
        
        Returns:
            StrategySignals with all calculated signals
        """
        print("  Calculating strategy signals...")
        
        # Prepare price series
        close_prices = {
            symbol: data.close 
            for symbol, data in price_data.items()
        }
        
        # 1. Momentum signals
        momentum_signals = calculate_momentum_signals(close_prices)
        momentum_dict = {
            s.symbol: s.score 
            for s in momentum_signals
        }
        
        # 2. Mean reversion signals
        mean_rev_signals = calculate_mean_reversion_signals(close_prices)
        mean_rev_dict = {
            s.symbol: s.score 
            for s in mean_rev_signals
        }
        
        # 3. Technical indicators
        technical_dict = {}
        for symbol, data in price_data.items():
            if len(data.close) >= 200:
                indicators = calculate_all_technical_indicators(
                    data.close,
                    data.high,
                    data.low,
                    data.data.get('volume')
                )
                technical_dict[symbol] = {
                    "rsi": indicators.rsi,
                    "macd": {
                        "macd": indicators.macd.macd,
                        "signal": indicators.macd.signal,
                        "histogram": indicators.macd.histogram
                    },
                    "sma20": indicators.sma20,
                    "sma50": indicators.sma50,
                    "sma200": indicators.sma200,
                    "atr": indicators.atr
                }
        
        # 4. Volatility regime
        if "SPY" in price_data:
            spy_data = price_data["SPY"]
            regime = detect_volatility_regime(
                spy_data.close,
                spy_data.high,
                spy_data.low
            )
            vol_regime = f"{regime.volatility.value}_{regime.trend.value}"
        else:
            vol_regime = "normal_vol_ranging"
        
        # 5. ML predictions (placeholder - simple momentum-based for now)
        ml_predictions = {}
        for symbol, score in momentum_dict.items():
            # Simple model: predict 5-day return based on momentum
            predicted_return = score * 0.02  # Scale to realistic returns
            ml_predictions[symbol] = predicted_return
        
        # 6. Semantic search
        semantic_result = {}
        if "SPY" in price_data:
            try:
                spy_data = price_data["SPY"]
                search_result = self.semantic_engine.search(
                    close=spy_data.close,
                    high=spy_data.high,
                    low=spy_data.low,
                    volume=spy_data.data.get('volume'),
                    top_k=50
                )
                semantic_result = {
                    "avg_5d_return": search_result.avg_5d_return,
                    "avg_20d_return": search_result.avg_20d_return,
                    "positive_5d_rate": search_result.positive_5d_rate,
                    "interpretation": search_result.interpretation
                }
            except Exception as e:
                print(f"    Semantic search error: {e}")
                semantic_result = {
                    "avg_5d_return": 0.0,
                    "avg_20d_return": 0.0,
                    "positive_5d_rate": 0.5,
                    "interpretation": "Semantic search unavailable"
                }
        
        return StrategySignals(
            momentum=momentum_dict,
            mean_reversion=mean_rev_dict,
            technical=technical_dict,
            ml_prediction=ml_predictions,
            volatility_regime=vol_regime,
            semantic_search=semantic_result
        )
    
    async def update_database(
        self,
        db: AsyncSession,
        trade_results: List
    ) -> None:
        """
        Update database with trade results and performance.
        
        Args:
            db: Database session
            trade_results: List of TradeResult from trading cycle
        """
        print("  Updating database...")
        
        # Update portfolios and positions
        for manager_id, manager in self.trading_engine.managers.items():
            # Get or create portfolio
            result = await db.execute(
                select(Portfolio).where(Portfolio.manager_id == manager_id)
            )
            portfolio = result.scalar_one_or_none()
            
            if portfolio:
                portfolio.cash_balance = Decimal(
                    str(manager.portfolio.cash_balance)
                )
                portfolio.total_value = Decimal(
                    str(manager.portfolio.total_value)
                )
                portfolio.updated_at = datetime.utcnow()
        
        # Store trades
        for result in trade_results:
            if result.success and result.filled_price:
                trade = Trade(
                    manager_id=result.manager_id,
                    symbol=result.decision.symbol,
                    side=result.decision.action.value,
                    quantity=Decimal(str(result.decision.size)),
                    price=Decimal(str(result.filled_price)),
                    reasoning=result.decision.reasoning,
                    signals_used=result.decision.signals_used,
                    alpaca_order_id=result.order_id,
                    executed_at=datetime.utcnow()
                )
                db.add(trade)
        
        await db.commit()
        print("    ✓ Portfolios and trades updated")
    
    async def calculate_and_store_performance(
        self,
        db: AsyncSession
    ) -> None:
        """Calculate daily performance snapshots"""
        print("  Calculating performance...")
        
        today = date.today()
        
        for manager_id, manager in self.trading_engine.managers.items():
            # Get historical snapshots
            result = await db.execute(
                select(DailySnapshot)
                .where(DailySnapshot.manager_id == manager_id)
                .order_by(DailySnapshot.date)
            )
            snapshots = result.scalars().all()
            
            # Build portfolio value history
            portfolio_values = [
                float(s.portfolio_value) for s in snapshots
            ]
            portfolio_values.append(manager.portfolio.total_value)
            
            # Get trades for win rate
            trades_result = await db.execute(
                select(Trade).where(Trade.manager_id == manager_id)
            )
            all_trades = trades_result.scalars().all()
            
            # Calculate metrics
            metrics = self.performance_tracker.calculate_metrics(
                manager_id=manager_id,
                portfolio_values=portfolio_values,
                trades=[],  # Would need P&L per trade
                initial_value=self.settings.initial_capital
            )
            
            # Create or update snapshot
            existing = await db.execute(
                select(DailySnapshot)
                .where(
                    DailySnapshot.manager_id == manager_id,
                    DailySnapshot.date == today
                )
            )
            snapshot = existing.scalar_one_or_none()
            
            if snapshot:
                # Update existing
                snapshot.portfolio_value = Decimal(
                    str(manager.portfolio.total_value)
                )
                snapshot.daily_return = Decimal(str(metrics.total_return))
                snapshot.cumulative_return = Decimal(str(metrics.total_return))
                snapshot.sharpe_ratio = Decimal(str(metrics.sharpe_ratio))
                snapshot.volatility = Decimal(str(metrics.volatility))
                snapshot.max_drawdown = Decimal(str(metrics.max_drawdown))
                snapshot.win_rate = Decimal(str(metrics.win_rate))
                snapshot.total_trades = len(all_trades)
            else:
                # Create new
                snapshot = DailySnapshot(
                    manager_id=manager_id,
                    date=today,
                    portfolio_value=Decimal(
                        str(manager.portfolio.total_value)
                    ),
                    daily_return=Decimal(str(metrics.total_return)),
                    cumulative_return=Decimal(str(metrics.total_return)),
                    sharpe_ratio=Decimal(str(metrics.sharpe_ratio)),
                    volatility=Decimal(str(metrics.volatility)),
                    max_drawdown=Decimal(str(metrics.max_drawdown)),
                    win_rate=Decimal(str(metrics.win_rate)),
                    total_trades=len(all_trades)
                )
                db.add(snapshot)
        
        await db.commit()
        print("    ✓ Performance snapshots updated")
    
    async def run_trading_cycle(self) -> Dict:
        """
        Execute one complete trading cycle.
        
        Returns:
            Summary dict of cycle results
        """
        print("\n" + "=" * 60)
        print(f"Trading Cycle: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 60)
        
        cycle_start = datetime.now()
        
        # Step 1: Start of day
        print("\n[1/6] Start of day...")
        self.trading_engine.start_of_day()
        print("  ✓ Daily stats reset")
        
        # Step 2: Fetch market data
        print("\n[2/6] Fetching market data...")
        price_data = self.data_ingestion.fetch_all()
        latest_prices = self.data_ingestion.get_latest_prices()
        print(f"  ✓ Fetched data for {len(price_data)} symbols")
        
        # Step 3: Calculate signals
        print("\n[3/6] Calculating strategy signals...")
        signals = self.calculate_signals(price_data)
        print(f"  ✓ Signals calculated")
        print(f"    Momentum: {len(signals.momentum)} symbols")
        print(f"    Mean reversion: {len(signals.mean_reversion)} symbols")
        print(f"    Regime: {signals.volatility_regime}")
        
        # Step 4: Run trading cycle
        print("\n[4/6] Running trading cycle...")
        trade_results = await self.trading_engine.run_cycle(
            latest_prices,
            signals
        )
        
        successful = sum(1 for r in trade_results if r.success)
        print(f"  ✓ Executed {successful}/{len(trade_results)} trades")
        
        # Step 5: Update database
        print("\n[5/6] Updating database...")
        async for db in get_async_session():
            try:
                await self.update_database(db, trade_results)
                await self.calculate_and_store_performance(db)
                break
            finally:
                await db.close()
        
        # Step 6: Summary
        print("\n[6/6] Cycle summary...")
        leaderboard = self.trading_engine.get_leaderboard()
        
        print("\n  Leaderboard:")
        for entry in leaderboard:
            print(
                f"    {entry['rank']}. {entry['name']}: "
                f"${entry['total_value']:,.2f}"
            )
        
        cycle_duration = (datetime.now() - cycle_start).total_seconds()
        
        print("\n" + "=" * 60)
        print(f"✓ Cycle complete in {cycle_duration:.1f}s")
        print("=" * 60)
        
        self._last_cycle_time = datetime.now()
        
        return {
            "timestamp": cycle_start.isoformat(),
            "duration_seconds": cycle_duration,
            "trades_executed": successful,
            "trades_attempted": len(trade_results),
            "leaderboard": leaderboard
        }
    
    async def run_once(self) -> Dict:
        """Run a single trading cycle"""
        return await self.run_trading_cycle()
    
    def should_run_cycle(self, now: datetime) -> bool:
        """
        Determine if we should run a cycle now.
        
        Basic logic: Run once per day during market hours.
        """
        # Check if market hours (9:30 AM - 4:00 PM ET, simplified)
        market_open = dt_time(9, 30)
        market_close = dt_time(16, 0)
        
        current_time = now.time()
        
        if current_time < market_open or current_time > market_close:
            return False
        
        # Check if already ran today
        if self._last_cycle_time:
            if self._last_cycle_time.date() == now.date():
                return False
        
        return True
    
    async def run_continuous(
        self,
        check_interval: int = 300  # 5 minutes
    ) -> None:
        """
        Run continuous trading loop.
        
        Checks every `check_interval` seconds if it's time to trade.
        """
        print("Starting continuous trading scheduler...")
        print(f"Check interval: {check_interval}s")
        print()
        
        self._is_running = True
        
        while self._is_running:
            now = datetime.now()
            
            if self.should_run_cycle(now):
                try:
                    await self.run_trading_cycle()
                except Exception as e:
                    print(f"Error in trading cycle: {e}")
                    import traceback
                    traceback.print_exc()
            
            # Wait before next check
            await asyncio.sleep(check_interval)
    
    def stop(self) -> None:
        """Stop the continuous scheduler"""
        print("Stopping scheduler...")
        self._is_running = False


async def main():
    """Run scheduler"""
    scheduler = TradingScheduler()
    
    # Run once for testing
    await scheduler.run_once()


if __name__ == "__main__":
    asyncio.run(main())
