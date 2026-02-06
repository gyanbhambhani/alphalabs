"""
Simulation Engine for AI Fund Time Machine.

Orchestrates the backtest simulation with:
- Daily ticks through 2000-2025
- Time dilation for fast visualization
- Multi-fund parallel execution
- Trade limits (3 per week per fund)
- Event streaming to frontend
- Database persistence for training data
- Trade outcome tracking for evaluation metrics
"""

import asyncio
import uuid
from datetime import date, datetime
from typing import AsyncGenerator, Dict, List, Optional, Tuple, TYPE_CHECKING
from dataclasses import dataclass, field
import logging
import time

from core.backtest.data_loader import HistoricalDataLoader
from core.backtest.snapshot_builder import PointInTimeSnapshotBuilder
from core.backtest.portfolio_tracker import BacktestPortfolio, BacktestTrade
from core.backtest.debate_runner import (
    DailyDebateRunner,
    CollaborativeDebateRunner,
    TradingDecision,
    get_debate_runner,
)
from core.backtest.events import (
    BacktestEvent,
    DayTickEvent,
    DebateStartEvent,
    DebateMessageEvent,
    DecisionEvent,
    TradeExecutedEvent,
    PortfolioUpdateEvent,
    LeaderboardEvent,
    FundRanking,
    SimulationStartEvent,
    SimulationEndEvent,
    ErrorEvent,
)
from core.data.snapshot import GlobalMarketSnapshot
from core.execution.trade_budget import TradeBudget

# Import evaluation metrics for trade outcome tracking
from core.evals.metrics import OutcomeTracker, compute_fund_metrics, TradeResult, FundMetrics

if TYPE_CHECKING:
    from core.backtest.persistence import BacktestPersistence

logger = logging.getLogger(__name__)


# =============================================================================
# Execution Guard Constants
# =============================================================================

MAX_SINGLE_POSITION_WEIGHT = 0.20  # 20% max concentration

# Minimum holding periods by strategy (prevents churn)
MIN_HOLDING_DAYS: Dict[str, int] = {
    "momentum": 10,
    "trend_macro": 10,
    "mean_reversion": 2,
    "volatility": 5,
    "value": 20,
    "quality_ls": 20,
    "event_driven": 5,
}


@dataclass
class ExecutionBlockResult:
    """Result of execution guard checks."""
    can_execute: bool
    block_reasons: List[str] = field(default_factory=list)


@dataclass
class FundConfig:
    """Configuration for a fund in the simulation."""
    fund_id: str
    name: str
    thesis: str
    initial_cash: float = 100_000.0


@dataclass
class FundDayResult:
    """Result of a fund's trading day."""
    fund_id: str
    decision: TradingDecision
    trades: List[BacktestTrade] = field(default_factory=list)
    new_positions: int = 0


class SimulationEngine:
    """
    Main simulation engine for backtesting AI funds.
    
    Features:
    - Processes 6,300 trading days (2000-2025)
    - Time dilation: compress 25 years into minutes
    - Multiple funds running in parallel
    - TradeBudget enforced BEFORE LLM debates (not after)
    - Streams events to frontend via SSE
    """
    
    # Commission model
    COMMISSION_PER_SHARE = 0.01
    MIN_COMMISSION_PCT = 0.001  # 0.1%
    
    def __init__(
        self,
        funds: List[FundConfig],
        data_loader: HistoricalDataLoader,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
        speed_multiplier: float = 100.0,
        initial_cash: float = 100_000.0,
        persistence: Optional["BacktestPersistence"] = None,
        debate_version: str = "v2",
    ):
        """
        Initialize the simulation engine.
        
        Args:
            funds: List of fund configurations
            data_loader: Historical data loader
            start_date: Simulation start (defaults to 2000-01-03)
            end_date: Simulation end (defaults to 2025-01-01)
            speed_multiplier: Time dilation factor (100 = 100x speed)
            initial_cash: Starting cash per fund
            persistence: Optional persistence layer for saving training data
            debate_version: "v1" for DailyDebateRunner, "v2" for CollaborativeDebateRunner
        """
        self.funds = funds
        self.data_loader = data_loader
        self.snapshot_builder = PointInTimeSnapshotBuilder(data_loader)
        self.debate_version = debate_version
        self.debate_runner = get_debate_runner(version=debate_version)
        self.persistence = persistence
        
        # Date range - default to 2001-01-02 to ensure 1 year of historical data
        # This gives V2 debate system enough return data (21d, 63d returns)
        trading_days = data_loader.trading_days
        default_start = date(2001, 1, 2)  # Start 2001, not 2000
        self.start_date = start_date or default_start
        self.end_date = end_date or trading_days[-1].date()
        
        # Get trading days in range
        self.trading_days = data_loader.get_trading_days_range(
            self.start_date, self.end_date
        )
        
        # Initialize portfolios
        self.portfolios: Dict[str, BacktestPortfolio] = {
            f.fund_id: BacktestPortfolio(
                fund_id=f.fund_id,
                initial_cash=f.initial_cash or initial_cash,
                cash=f.initial_cash or initial_cash,
            )
            for f in funds
        }
        
        # Trade budgets (replaced old rolling window approach)
        self.trade_budgets: Dict[str, TradeBudget] = {}
        self._last_budget_reset: Optional[date] = None
        self._fund_orders: Dict[str, List] = {
            f.fund_id: [] for f in funds
        }
        
        # Simulation state
        self.speed = speed_multiplier
        self.paused = False
        self.stopped = False
        self.current_day_idx = 0
        
        # Event queue
        self._event_queue: asyncio.Queue = asyncio.Queue()
        
        # Run ID for this simulation
        self.run_id = str(uuid.uuid4())[:8]
        
        # Benchmark tracking (SPY)
        self.benchmark_start_value: Optional[float] = None
        
        # Tracking for persistence
        self._total_trades = 0
        self._total_decisions = 0
        self._trades_today: Dict[str, int] = {f.fund_id: 0 for f in funds}
        
        # Outcome tracking for evaluation metrics
        self.outcome_trackers: Dict[str, OutcomeTracker] = {
            f.fund_id: OutcomeTracker()
            for f in funds
        }
        
        # Track trade results for computing fund metrics
        self._trade_results: Dict[str, List[TradeResult]] = {
            f.fund_id: []
            for f in funds
        }
        self._trade_values: Dict[str, List[float]] = {
            f.fund_id: []
            for f in funds
        }
        
        # Track trade counts per day for each fund
        self.trade_counts: Dict[str, List[int]] = {
            f.fund_id: []
            for f in funds
        }
        
        # Store computed fund metrics
        self.fund_metrics: Dict[str, FundMetrics] = {}
        
        # Track last trade date per fund per symbol (for churn prevention)
        self._last_trade_date: Dict[str, Dict[str, date]] = {}
    
    @property
    def progress(self) -> float:
        """Current progress as percentage."""
        if not self.trading_days:
            return 0.0
        return self.current_day_idx / len(self.trading_days)
    
    @property
    def current_date(self) -> Optional[date]:
        """Current simulation date."""
        if self.current_day_idx < len(self.trading_days):
            return self.trading_days[self.current_day_idx]
        return None
    
    def _get_or_create_budget(
        self,
        fund_id: str,
        current_date: date,
        fund_thesis: str,
    ) -> TradeBudget:
        """
        Get or create TradeBudget for fund.
        
        Handles weekly reset logic.
        """
        portfolio = self.portfolios[fund_id]
        
        # Check if we need to reset weekly counter
        if self._last_budget_reset is None:
            self._last_budget_reset = current_date
        
        days_since_reset = (current_date - self._last_budget_reset).days
        if days_since_reset >= 7:
            # Reset all budgets
            for budget in self.trade_budgets.values():
                budget.reset_weekly_counter()
            self._last_budget_reset = current_date
        
        # Get or create budget
        if fund_id not in self.trade_budgets:
            # Determine rebalance cadence from thesis
            rebalance_cadence = self._infer_rebalance_cadence(fund_thesis)
            
            self.trade_budgets[fund_id] = TradeBudget(
                fund_id=fund_id,
                current_date=current_date,
                portfolio_value=portfolio.total_value,
                trades_this_week=0,
                max_trades_per_week=3,
                rebalance_cadence=rebalance_cadence,
                last_rebalance_date=None,
                min_weight_delta=0.02,
                min_order_notional=1000.0,
            )
        else:
            # Update budget state
            budget = self.trade_budgets[fund_id]
            budget.current_date = current_date
            budget.portfolio_value = portfolio.total_value
        
        return self.trade_budgets[fund_id]
    
    def _infer_rebalance_cadence(self, fund_thesis: str) -> str:
        """
        Get default rebalance cadence for fund type.
        
        Uses strategy-specific cadences for sensible defaults.
        """
        thesis_lower = fund_thesis.lower()
        
        cadence_map = {
            "momentum": "weekly",
            "trend_macro": "weekly",
            "mean_reversion": "daily",
            "value": "monthly",
            "quality_ls": "monthly",
            "event_driven": "weekly",
            "volatility": "weekly",
        }
        
        for key, cadence in cadence_map.items():
            if key in thesis_lower:
                return cadence
        
        return "weekly"  # Conservative default
    
    def _validate_price_for_trade(
        self,
        symbol: str,
        snapshot: GlobalMarketSnapshot,
    ) -> Tuple[bool, str]:
        """
        Ensure we have a valid price before trading.
        
        Args:
            symbol: Stock symbol
            snapshot: Market data snapshot
            
        Returns:
            Tuple of (is_valid, reason_if_invalid)
        """
        price = snapshot.get_price(symbol)
        
        if price is None:
            return False, f"No price available for {symbol}"
        
        if price <= 0:
            return False, f"Invalid price {price} for {symbol}"
        
        return True, ""
    
    def _should_execute_buy(
        self,
        symbol: str,
        target_weight: float,
        portfolio: BacktestPortfolio,
        budget: TradeBudget,
        price: float,
    ) -> ExecutionBlockResult:
        """
        Check if buy should execute given current holdings.
        
        Guards:
        1. Already at or above target weight
        2. Delta too small (hysteresis)
        3. Would exceed max concentration
        4. Quantity check (ensure we're buying positive shares)
        
        Args:
            symbol: Stock symbol
            target_weight: Target portfolio weight
            portfolio: Current portfolio state
            budget: Trade budget constraints
            price: Current price
            
        Returns:
            ExecutionBlockResult with can_execute and block_reasons
        """
        reasons: List[str] = []
        current_weight = portfolio.get_position_weight(symbol)
        
        # Guard 1: Already at or above target
        if current_weight >= target_weight:
            reasons.append(
                f"Already at {current_weight:.1%}, target {target_weight:.1%}"
            )
        
        # Guard 2: Delta too small (hysteresis via TradeBudget)
        if budget and not budget.should_trade(current_weight, target_weight, price):
            delta = target_weight - current_weight
            reasons.append(f"Delta {delta:.1%} below threshold")
        
        # Guard 3: Max concentration
        if target_weight > MAX_SINGLE_POSITION_WEIGHT:
            reasons.append(
                f"Target {target_weight:.1%} exceeds max "
                f"{MAX_SINGLE_POSITION_WEIGHT:.1%}"
            )
        
        # Guard 4: Quantity sanity check
        if price > 0:
            target_value = portfolio.total_value * target_weight
            current_value = portfolio.total_value * current_weight
            buy_value = target_value - current_value
            buy_qty = int(buy_value / price)
            if buy_qty <= 0:
                reasons.append(f"Computed buy quantity {buy_qty} <= 0")
        
        return ExecutionBlockResult(
            can_execute=len(reasons) == 0,
            block_reasons=reasons,
        )
    
    def _should_execute_sell(
        self,
        symbol: str,
        target_weight: float,
        portfolio: BacktestPortfolio,
        price: float,
    ) -> ExecutionBlockResult:
        """
        Check if sell should execute.
        
        Guards:
        1. Don't hold the symbol
        2. Current shares <= 0
        3. Already at or below target
        4. Sell quantity sanity (don't sell more than we have)
        
        Args:
            symbol: Stock symbol
            target_weight: Target portfolio weight
            portfolio: Current portfolio state
            price: Current price
            
        Returns:
            ExecutionBlockResult with can_execute and block_reasons
        """
        reasons: List[str] = []
        
        # Guard 1: Don't hold the symbol
        if symbol not in portfolio.positions:
            reasons.append(f"Cannot sell {symbol} - not in portfolio")
            return ExecutionBlockResult(can_execute=False, block_reasons=reasons)
        
        position = portfolio.positions[symbol]
        current_weight = portfolio.get_position_weight(symbol)
        
        # Guard 2: Current shares <= 0
        if position.quantity <= 0:
            reasons.append(
                f"Cannot sell {symbol} - quantity {position.quantity} <= 0"
            )
        
        # Guard 3: Already at or below target
        if current_weight <= target_weight:
            reasons.append(
                f"Already at {current_weight:.1%}, target {target_weight:.1%}"
            )
        
        # Guard 4: Sell quantity sanity
        if price > 0 and position.quantity > 0:
            target_value = portfolio.total_value * target_weight
            current_value = position.quantity * price
            sell_value = current_value - target_value
            sell_qty = int(sell_value / price)
            if sell_qty > position.quantity:
                reasons.append(
                    f"Sell qty {sell_qty} > held qty {position.quantity}"
                )
        
        return ExecutionBlockResult(
            can_execute=len(reasons) == 0,
            block_reasons=reasons,
        )
    
    def _check_churn_guard(
        self,
        fund_id: str,
        fund_strategy: str,
        symbol: str,
        action: str,
        current_date: date,
    ) -> Tuple[bool, str]:
        """
        Check if trade violates minimum holding period.
        
        Only applies to sells/reduces - you can always buy more.
        
        Args:
            fund_id: Fund identifier
            fund_strategy: Fund's strategy
            symbol: Stock symbol
            action: Trade action
            current_date: Current simulation date
            
        Returns:
            Tuple of (can_trade, reason_if_blocked)
        """
        if action == "buy":
            return True, ""  # Buys always allowed (other guards handle duplicates)
        
        # Get last trade date for this symbol in this fund
        fund_trades = self._last_trade_date.get(fund_id, {})
        last_date = fund_trades.get(symbol)
        
        if last_date is None:
            return True, ""  # Never traded, can sell
        
        # Get min holding period for strategy
        strategy_key = fund_strategy.lower().split()[0]
        min_days = MIN_HOLDING_DAYS.get(strategy_key, 5)
        
        days_held = (current_date - last_date).days
        if days_held < min_days:
            return False, (
                f"Churn guard: {symbol} bought {days_held}d ago, "
                f"min hold {min_days}d for {strategy_key}"
            )
        
        return True, ""
    
    def _record_trade(
        self,
        fund_id: str,
        symbol: str,
        action: str,
        current_date: date,
    ) -> None:
        """
        Record trade date for churn prevention.
        
        Args:
            fund_id: Fund identifier
            symbol: Stock symbol
            action: Trade action
            current_date: Current simulation date
        """
        if action == "buy":
            if fund_id not in self._last_trade_date:
                self._last_trade_date[fund_id] = {}
            self._last_trade_date[fund_id][symbol] = current_date
    
    def set_speed(self, multiplier: float) -> None:
        """Change simulation speed."""
        self.speed = max(1.0, multiplier)
    
    def pause(self) -> None:
        """Pause the simulation."""
        self.paused = True
    
    def resume(self) -> None:
        """Resume the simulation."""
        self.paused = False
    
    def stop(self) -> None:
        """Stop the simulation."""
        self.stopped = True
        self.paused = False
    
    async def emit_event(self, event: BacktestEvent) -> None:
        """Emit an event to the queue."""
        await self._event_queue.put(event)
    
    async def events(self) -> AsyncGenerator[BacktestEvent, None]:
        """Async generator for consuming events."""
        while not self.stopped or not self._event_queue.empty():
            try:
                event = await asyncio.wait_for(
                    self._event_queue.get(),
                    timeout=0.1
                )
                yield event
            except asyncio.TimeoutError:
                continue
    
    async def run(self) -> None:
        """
        Run the full simulation.
        
        Processes each trading day, runs fund debates, executes trades,
        and emits events for frontend visualization.
        """
        start_time = time.time()
        
        # Create persistence record
        if self.persistence:
            try:
                self.persistence.create_run(
                    run_id=self.run_id,
                    fund_ids=[f.fund_id for f in self.funds],
                    start_date=self.start_date,
                    end_date=self.end_date,
                    initial_cash=self.portfolios[self.funds[0].fund_id].initial_cash,
                    config={
                        "funds": [
                            {"id": f.fund_id, "name": f.name, "thesis": f.thesis}
                            for f in self.funds
                        ],
                        "speed": self.speed,
                    },
                    universe=self.data_loader.universe,
                )
                logger.info(f"Created persistence record for run {self.run_id}")
            except Exception as e:
                logger.error(f"Failed to create persistence record: {e}")
        
        # Emit start event
        await self.emit_event(SimulationStartEvent(
            timestamp=datetime.utcnow(),
            start_date=self.start_date,
            end_date=self.end_date,
            total_days=len(self.trading_days),
            funds=[{"id": f.fund_id, "name": f.name} for f in self.funds],
            universe=self.data_loader.universe,
            initial_cash=self.portfolios[self.funds[0].fund_id].initial_cash,
        ))
        
        logger.info(
            f"Starting simulation: {self.start_date} to {self.end_date} "
            f"({len(self.trading_days)} days, {len(self.funds)} funds)"
        )
        
        try:
            for day_idx, current_date in enumerate(self.trading_days):
                self.current_day_idx = day_idx
                
                # Check for stop/pause
                if self.stopped:
                    break
                
                while self.paused:
                    await asyncio.sleep(0.1)
                    if self.stopped:
                        break
                
                # Process this trading day
                await self._process_day(current_date, day_idx)
                
                # Time dilation delay
                delay = 1.0 / self.speed
                if delay > 0.001:  # Skip tiny delays
                    await asyncio.sleep(delay)
            
            # Simulation complete
            elapsed = time.time() - start_time
            
            # Compute fund metrics for each fund
            for fund in self.funds:
                portfolio = self.portfolios[fund.fund_id]
                trade_results = self._trade_results[fund.fund_id]
                trade_values = self._trade_values[fund.fund_id]
                portfolio_values = portfolio.total_value_history
                
                metrics = compute_fund_metrics(
                    fund_id=fund.fund_id,
                    period_start=self.start_date,
                    period_end=self.current_date or self.end_date,
                    results=trade_results,
                    trade_values=trade_values,
                    portfolio_value=portfolio.total_value,
                    portfolio_values=portfolio_values,
                )
                self.fund_metrics[fund.fund_id] = metrics
                
                logger.info(
                    f"Fund {fund.fund_id} metrics: "
                    f"hit_rate={metrics.hit_rate:.1%}, "
                    f"brier={metrics.brier_score:.3f}, "
                    f"n_trades={metrics.n_trades}"
                )
                
                # Save metrics to persistence
                if self.persistence:
                    try:
                        from dataclasses import asdict
                        self.persistence.save_fund_metrics(
                            run_id=self.run_id,
                            fund_id=fund.fund_id,
                            metrics=asdict(metrics),
                        )
                    except Exception as e:
                        logger.error(f"Failed to save fund metrics: {e}")
            
            # Build final rankings
            final_rankings = self._build_leaderboard(self.current_date)
            
            # Complete persistence record
            if self.persistence:
                try:
                    self.persistence.complete_run(
                        run_id=self.run_id,
                        total_trades=self._total_trades,
                        total_decisions=self._total_decisions,
                        elapsed_seconds=elapsed,
                    )
                    logger.info(
                        f"Persisted run {self.run_id}: "
                        f"{self._total_trades} trades, {self._total_decisions} decisions"
                    )
                except Exception as e:
                    logger.error(f"Failed to complete persistence record: {e}")
            
            await self.emit_event(SimulationEndEvent(
                timestamp=datetime.utcnow(),
                elapsed_seconds=elapsed,
                total_days_simulated=self.current_day_idx + 1,
                final_rankings=final_rankings,
            ))
            
            logger.info(
                f"Simulation complete: {elapsed:.1f}s, "
                f"{self.current_day_idx + 1} days"
            )
            
        except Exception as e:
            logger.error(f"Simulation error: {e}")
            
            # Mark run as failed
            if self.persistence:
                try:
                    self.persistence.fail_run(self.run_id, str(e))
                except Exception as pe:
                    logger.error(f"Failed to mark run as failed: {pe}")
            
            await self.emit_event(ErrorEvent(
                timestamp=datetime.utcnow(),
                error_type="simulation_error",
                message=str(e),
                recoverable=False,
            ))
            raise
    
    async def _process_day(self, current_date: date, day_idx: int) -> None:
        """Process a single trading day."""
        # Build point-in-time snapshot (NO FUTURE DATA)
        snapshot = self.snapshot_builder.build_snapshot(current_date)
        
        # Get benchmark value (SPY)
        spy_price = snapshot.get_price("SPY") or 0
        if self.benchmark_start_value is None and spy_price > 0:
            self.benchmark_start_value = spy_price
        
        benchmark_return = 0.0
        if self.benchmark_start_value and spy_price:
            benchmark_return = (spy_price - self.benchmark_start_value) / self.benchmark_start_value
        
        # Emit day tick
        await self.emit_event(DayTickEvent(
            timestamp=datetime.utcnow(),
            simulation_date=current_date,
            day_index=day_idx,
            total_days=len(self.trading_days),
            progress_pct=day_idx / len(self.trading_days) * 100,
            benchmark_value=spy_price,
            benchmark_return=benchmark_return,
            prices=dict(list(snapshot.prices.items())[:20]),  # Top 20 for SSE
        ))
        
        # Update all portfolio prices first
        for fund in self.funds:
            portfolio = self.portfolios[fund.fund_id]
            portfolio.update_prices(snapshot.prices)
        
        # Run funds SEQUENTIALLY (not parallel) for better visualization
        # Each fund gets its turn, with a pause between for readability
        results = []
        for fund in self.funds:
            result = await self._run_fund_day(fund, snapshot, current_date)
            results.append(result)
            
            # Small delay between funds for readability (scaled by speed)
            fund_delay = 0.5 / max(self.speed / 10, 1)
            if fund_delay > 0.01:
                await asyncio.sleep(fund_delay)
        
        # Update trade counts
        for fund, result in zip(self.funds, results):
            self.trade_counts[fund.fund_id].append(result.new_positions)
        
        # Record portfolio snapshots and emit updates
        for fund in self.funds:
            portfolio = self.portfolios[fund.fund_id]
            portfolio.record_snapshot(current_date)
            
            # Save snapshot to persistence
            if self.persistence:
                try:
                    self.persistence.save_portfolio_snapshot(
                        run_id=self.run_id,
                        fund_id=fund.fund_id,
                        snapshot_date=current_date,
                        cash=portfolio.cash,
                        positions={
                            sym: {
                                "quantity": pos.quantity,
                                "avg_entry_price": pos.avg_entry_price,
                                "current_price": pos.current_price,
                                "current_value": pos.current_value,
                            }
                            for sym, pos in portfolio.positions.items()
                        },
                        total_value=portfolio.total_value,
                        daily_return=portfolio.daily_return,
                        cumulative_return=portfolio.cumulative_return,
                        max_drawdown=portfolio.max_drawdown,
                        sharpe_ratio=portfolio.sharpe_ratio,
                        n_trades_today=self._trades_today.get(fund.fund_id, 0),
                    )
                except Exception as e:
                    logger.error(f"Failed to save snapshot: {e}")
            
            await self.emit_event(PortfolioUpdateEvent(
                timestamp=datetime.utcnow(),
                fund_id=fund.fund_id,
                fund_name=fund.name,
                simulation_date=current_date,
                total_value=portfolio.total_value,
                cash=portfolio.cash,
                invested_pct=portfolio.invested_pct,
                daily_return=portfolio.daily_return,
                cumulative_return=portfolio.cumulative_return,
                max_drawdown=portfolio.max_drawdown,
                sharpe_ratio=portfolio.sharpe_ratio,
                positions={
                    sym: {
                        "quantity": pos.quantity,
                        "current_price": pos.current_price,
                        "current_value": pos.current_value,
                        "unrealized_return": pos.unrealized_return,
                    }
                    for sym, pos in portfolio.positions.items()
                },
            ))
        
        # Reset trades today counter
        self._trades_today = {f.fund_id: 0 for f in self.funds}
        
        # Emit leaderboard update every day for better tracking
        rankings = self._build_leaderboard(current_date)
        await self.emit_event(LeaderboardEvent(
            timestamp=datetime.utcnow(),
            simulation_date=current_date,
            rankings=rankings,
            benchmark_return=benchmark_return,
        ))
    
    async def _run_fund_day(
        self,
        fund: FundConfig,
        snapshot: GlobalMarketSnapshot,
        current_date: date,
    ) -> FundDayResult:
        """Run a single fund's trading day with TradeBudget enforcement."""
        portfolio = self.portfolios[fund.fund_id]
        
        # Get/create trade budget BEFORE debate
        budget = self._get_or_create_budget(
            fund.fund_id,
            current_date,
            fund.thesis,
        )
        
        # Emit debate start
        await self.emit_event(DebateStartEvent(
            timestamp=datetime.utcnow(),
            fund_id=fund.fund_id,
            fund_name=fund.name,
            simulation_date=current_date,
        ))
        
        # Run AI debate with budget constraints
        decision = await self.debate_runner.run_debate(
            fund_id=fund.fund_id,
            fund_name=fund.name,
            fund_thesis=fund.thesis,
            portfolio=portfolio,
            snapshot=snapshot,
            simulation_date=current_date,
            trade_budget=budget,  # Pass budget to debate
        )
        
        # Emit debate messages
        for msg in decision.debate_transcript:
            await self.emit_event(DebateMessageEvent(
                timestamp=msg.timestamp,
                fund_id=fund.fund_id,
                phase=msg.phase,
                model=msg.model,
                content=msg.content,
                tokens_used=msg.tokens_used,
            ))
        
        # Generate decision ID for tracking
        decision_id = f"{self.run_id}_{fund.fund_id}_{current_date.isoformat()}"
        
        # Determine status based on action
        # hold = finalized (no execution needed)
        # buy/sell = will be executed next
        status = "finalized" if decision.action == "hold" else "sent_to_broker"
        
        # Emit decision
        await self.emit_event(DecisionEvent(
            timestamp=datetime.utcnow(),
            fund_id=fund.fund_id,
            fund_name=fund.name,
            simulation_date=current_date,
            action=decision.action,
            symbol=decision.symbol,
            target_weight=decision.target_weight,
            reasoning=decision.reasoning,
            confidence=decision.confidence,
            signals_used=decision.signals_used,
            status=status,
            decision_id=decision_id,
        ))
        
        # Save decision to persistence
        self._total_decisions += 1
        if self.persistence:
            try:
                self.persistence.save_decision(
                    run_id=self.run_id,
                    fund_id=fund.fund_id,
                    decision_date=current_date,
                    action=decision.action,
                    symbol=decision.symbol,
                    target_weight=decision.target_weight,
                    confidence=decision.confidence,
                    reasoning=decision.reasoning,
                    debate_transcript=[
                        {
                            "phase": msg.phase,
                            "model": msg.model,
                            "content": msg.content,  # Store full content
                            "tokens": msg.tokens_used,
                            "timestamp": msg.timestamp.isoformat(),
                        }
                        for msg in decision.debate_transcript
                    ] if decision.debate_transcript else None,
                    signals_snapshot=decision.signals_used,
                    models_used=decision.models_used,
                    tokens_used=decision.total_tokens,
                )
            except Exception as e:
                logger.error(f"Failed to save decision: {e}")
        
        # Execute trades (budget was already checked)
        trades = []
        new_positions = 0
        
        if decision.action != "hold":
            # Validate decision against budget one more time (safety gate)
            if decision.action == "buy" and not budget.can_buy():
                logger.warning(
                    f"[{fund.fund_id}] Buy decision rejected by budget gate "
                    f"- this should not happen!"
                )
            else:
                trade = await self._execute_decision(
                    fund, portfolio, decision, snapshot, current_date
                )
                if trade:
                    trades.append(trade)
                    self._fund_orders[fund.fund_id].append(trade)
                    
                    # Consume budget on successful trade
                    if decision.action == "buy":
                        budget.consume_trade_event()
                        new_positions = 1
        
        return FundDayResult(
            fund_id=fund.fund_id,
            decision=decision,
            trades=trades,
            new_positions=new_positions,
        )
    
    async def _execute_decision(
        self,
        fund: FundConfig,
        portfolio: BacktestPortfolio,
        decision: TradingDecision,
        snapshot: GlobalMarketSnapshot,
        current_date: date,
    ) -> Optional[BacktestTrade]:
        """
        Execute a trading decision with execution guards.
        
        Guards applied:
        1. Price validation (must exist and be positive)
        2. Buy guards (duplicate, concentration, quantity)
        3. Sell guards (held, quantity)
        4. Churn guard (minimum holding period)
        """
        if not decision.symbol:
            return None
        
        # GUARD 1: Price validation
        price_valid, price_reason = self._validate_price_for_trade(
            decision.symbol, snapshot
        )
        if not price_valid:
            logger.warning(f"[{fund.fund_id}] {price_reason}")
            return None
        
        price = snapshot.get_price(decision.symbol)
        target_weight = decision.target_weight or 0.1
        
        # Get budget for buy guards
        budget = self.trade_budgets.get(fund.fund_id)
        
        try:
            if decision.action == "buy":
                # GUARD 2: Buy-side guards
                buy_check = self._should_execute_buy(
                    decision.symbol, target_weight, portfolio, budget, price
                )
                if not buy_check.can_execute:
                    logger.info(
                        f"[{fund.fund_id}] Buy blocked: {buy_check.block_reasons}"
                    )
                    return None
                
                # Calculate quantity from target weight
                target_value = portfolio.total_value * target_weight
                current_weight = portfolio.get_position_weight(decision.symbol)
                current_value = portfolio.total_value * current_weight
                buy_value = target_value - current_value
                quantity = int(buy_value / price)
                
                if quantity <= 0:
                    return None
                
                # Calculate commission
                commission = max(
                    quantity * self.COMMISSION_PER_SHARE,
                    quantity * price * self.MIN_COMMISSION_PCT
                )
                
                # Check if we have enough cash
                total_cost = quantity * price + commission
                if total_cost > portfolio.cash:
                    # Reduce quantity to fit
                    quantity = int((portfolio.cash - commission) / price)
                    if quantity <= 0:
                        return None
                    commission = max(
                        quantity * self.COMMISSION_PER_SHARE,
                        quantity * price * self.MIN_COMMISSION_PCT
                    )
                
                trade = portfolio.execute_buy(
                    symbol=decision.symbol,
                    quantity=quantity,
                    price=price,
                    commission=commission,
                    timestamp=datetime.combine(current_date, datetime.min.time()),
                    reasoning=decision.reasoning,
                )
                
                # Record trade entry for outcome tracking
                decision_id = f"{self.run_id}_{fund.fund_id}_{current_date.isoformat()}"
                predicted_direction = "up"  # Buy implies expecting price to go up
                self.outcome_trackers[fund.fund_id].record_entry(
                    decision_id=decision_id,
                    fund_id=fund.fund_id,
                    symbol=decision.symbol,
                    direction="long",
                    entry_price=price,
                    entry_weight=decision.target_weight or 0.1,
                    predicted_direction=predicted_direction,
                    predicted_confidence=decision.confidence,
                )
                
                # Track trade value for turnover calculation
                self._trade_values[fund.fund_id].append(quantity * price)
                
                # Save trade to persistence
                self._total_trades += 1
                self._trades_today[fund.fund_id] = (
                    self._trades_today.get(fund.fund_id, 0) + 1
                )
                if self.persistence:
                    try:
                        self.persistence.save_trade(
                            run_id=self.run_id,
                            fund_id=fund.fund_id,
                            trade_date=current_date,
                            symbol=decision.symbol,
                            side="buy",
                            quantity=quantity,
                            price=price,
                            commission=commission,
                            reasoning=decision.reasoning,
                            confidence=decision.confidence,
                        )
                    except Exception as e:
                        logger.error(f"Failed to save trade: {e}")
                
                await self.emit_event(TradeExecutedEvent(
                    timestamp=datetime.utcnow(),
                    fund_id=fund.fund_id,
                    fund_name=fund.name,
                    symbol=decision.symbol,
                    side="buy",
                    quantity=quantity,
                    price=price,
                    commission=commission,
                    total_cost=quantity * price + commission,
                    reasoning=decision.reasoning,
                    decision_id=decision_id,
                ))
                
                # Record trade for churn prevention
                self._record_trade(fund.fund_id, decision.symbol, "buy", current_date)
                
                return trade
                
            elif decision.action == "sell":
                # GUARD 3: Sell-side guards
                sell_check = self._should_execute_sell(
                    decision.symbol, 0.0, portfolio, price  # target_weight=0 for full sell
                )
                if not sell_check.can_execute:
                    logger.info(
                        f"[{fund.fund_id}] Sell blocked: {sell_check.block_reasons}"
                    )
                    return None
                
                # GUARD 4: Churn guard (minimum holding period)
                churn_ok, churn_reason = self._check_churn_guard(
                    fund.fund_id, fund.thesis, decision.symbol, "sell", current_date
                )
                if not churn_ok:
                    logger.info(f"[{fund.fund_id}] {churn_reason}")
                    return None
                
                pos = portfolio.positions[decision.symbol]
                
                # Check minimum holding period (prevent panic selling)
                if not pos.can_sell(current_date, min_hold_days=3):
                    days_held = pos.days_held(current_date)
                    logger.info(
                        f"[{fund.fund_id}] Sell blocked: {decision.symbol} "
                        f"held only {days_held} days (min 3), "
                        f"loss {pos.unrealized_return:.1%} (need >15% for override)"
                    )
                    return None
                
                quantity = pos.quantity
                
                commission = max(
                    quantity * self.COMMISSION_PER_SHARE,
                    quantity * price * self.MIN_COMMISSION_PCT
                )
                
                trade = portfolio.execute_sell(
                    symbol=decision.symbol,
                    quantity=quantity,
                    price=price,
                    commission=commission,
                    timestamp=datetime.combine(current_date, datetime.min.time()),
                    reasoning=decision.reasoning,
                )
                
                # Generate decision_id for sell
                sell_decision_id = f"{self.run_id}_{fund.fund_id}_{current_date.isoformat()}"
                
                # Record trade exit for outcome tracking
                slippage_bps = commission / (quantity * price) * 10000  # Basis points
                trade_result = self.outcome_trackers[fund.fund_id].record_exit(
                    fund_id=fund.fund_id,
                    symbol=decision.symbol,
                    exit_price=price,
                    exit_reason=decision.reasoning,
                    slippage_bps=slippage_bps,
                )
                
                if trade_result:
                    self._trade_results[fund.fund_id].append(trade_result)
                
                # Track trade value for turnover calculation
                self._trade_values[fund.fund_id].append(quantity * price)
                
                # Save trade to persistence
                self._total_trades += 1
                self._trades_today[fund.fund_id] = (
                    self._trades_today.get(fund.fund_id, 0) + 1
                )
                if self.persistence:
                    try:
                        self.persistence.save_trade(
                            run_id=self.run_id,
                            fund_id=fund.fund_id,
                            trade_date=current_date,
                            symbol=decision.symbol,
                            side="sell",
                            quantity=quantity,
                            price=price,
                            commission=commission,
                            reasoning=decision.reasoning,
                            confidence=decision.confidence,
                        )
                    except Exception as e:
                        logger.error(f"Failed to save trade: {e}")
                
                await self.emit_event(TradeExecutedEvent(
                    timestamp=datetime.utcnow(),
                    fund_id=fund.fund_id,
                    fund_name=fund.name,
                    symbol=decision.symbol,
                    side="sell",
                    quantity=quantity,
                    price=price,
                    commission=commission,
                    total_cost=-(quantity * price - commission),
                    reasoning=decision.reasoning,
                    decision_id=sell_decision_id,
                ))
                
                return trade
                
        except Exception as e:
            logger.error(f"Error executing trade: {e}")
            await self.emit_event(ErrorEvent(
                timestamp=datetime.utcnow(),
                error_type="trade_error",
                message=str(e),
                fund_id=fund.fund_id,
                recoverable=True,
            ))
        
        return None
    
    def _build_leaderboard(self, current_date: date) -> List[FundRanking]:
        """Build current fund rankings."""
        rankings = []
        
        for fund in self.funds:
            portfolio = self.portfolios[fund.fund_id]
            rankings.append(FundRanking(
                fund_id=fund.fund_id,
                fund_name=fund.name,
                rank=0,  # Will be set below
                total_value=portfolio.total_value,
                cumulative_return=portfolio.cumulative_return,
                sharpe_ratio=portfolio.sharpe_ratio,
                max_drawdown=portfolio.max_drawdown,
            ))
        
        # Sort by cumulative return
        rankings.sort(key=lambda r: r.cumulative_return, reverse=True)
        
        # Assign ranks
        for i, r in enumerate(rankings):
            r.rank = i + 1
        
        return rankings


# Convenience function to create and run a simulation
async def run_backtest(
    funds: List[FundConfig],
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
    speed: float = 1000.0,
    initial_cash: float = 100_000.0,
    cache_dir: str = "data/backtest_cache",
) -> SimulationEngine:
    """
    Create and run a backtest simulation.
    
    Args:
        funds: List of fund configurations
        start_date: Simulation start date
        end_date: Simulation end date
        speed: Time dilation multiplier
        initial_cash: Starting cash per fund
        cache_dir: Directory for data cache
        
    Returns:
        SimulationEngine instance (with results)
    """
    # Load historical data
    loader = HistoricalDataLoader(cache_dir=cache_dir)
    loader.fetch_and_cache_all()
    
    # Create and run engine
    engine = SimulationEngine(
        funds=funds,
        data_loader=loader,
        start_date=start_date,
        end_date=end_date,
        speed_multiplier=speed,
        initial_cash=initial_cash,
    )
    
    await engine.run()
    return engine
