"""
Backtesting module for AI Fund Time Machine.

Simulates AI funds trading from 2000-2025 with:
- Point-in-time data (no look-ahead bias)
- Daily decision points
- Time dilation for fast visualization
- Full audit trail of all decisions
"""

from core.backtest.data_loader import HistoricalDataLoader, BACKTEST_UNIVERSE
from core.backtest.snapshot_builder import PointInTimeSnapshotBuilder
from core.backtest.portfolio_tracker import (
    BacktestPortfolio,
    BacktestPosition,
    BacktestTrade,
)
from core.backtest.events import (
    BacktestEvent,
    EventType,
    DayTickEvent,
    DebateStartEvent,
    DebateMessageEvent,
    DecisionEvent,
    TradeExecutedEvent,
    PortfolioUpdateEvent,
    LeaderboardEvent,
    FundRanking,
    MilestoneEvent,
    SimulationStartEvent,
    SimulationEndEvent,
    ErrorEvent,
)
from core.backtest.debate_runner import (
    DailyDebateRunner,
    TradingDecision,
    DebateMessage,
)
from core.backtest.simulation_engine import (
    SimulationEngine,
    FundConfig,
    FundDayResult,
    run_backtest,
)

__all__ = [
    # Data
    "HistoricalDataLoader",
    "BACKTEST_UNIVERSE",
    "PointInTimeSnapshotBuilder",
    # Portfolio
    "BacktestPortfolio",
    "BacktestPosition",
    "BacktestTrade",
    # Debate
    "DailyDebateRunner",
    "TradingDecision",
    "DebateMessage",
    # Simulation
    "SimulationEngine",
    "FundConfig",
    "FundDayResult",
    "run_backtest",
    # Events
    "BacktestEvent",
    "EventType",
    "DayTickEvent",
    "DebateStartEvent",
    "DebateMessageEvent",
    "DecisionEvent",
    "TradeExecutedEvent",
    "PortfolioUpdateEvent",
    "LeaderboardEvent",
    "FundRanking",
    "MilestoneEvent",
    "SimulationStartEvent",
    "SimulationEndEvent",
    "ErrorEvent",
]
