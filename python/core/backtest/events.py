"""
Backtest Event Types for SSE Streaming.

Events emitted during backtest simulation for frontend visualization.
"""

from dataclasses import dataclass, field, asdict
from datetime import date, datetime
from typing import Any, Dict, List, Optional
from enum import Enum
import json


class EventType(str, Enum):
    """Types of events emitted during backtest."""
    DAY_TICK = "day_tick"
    DEBATE_START = "debate_start"
    DEBATE_MESSAGE = "debate_message"
    DECISION = "decision"
    TRADE_EXECUTED = "trade_executed"
    PORTFOLIO_UPDATE = "portfolio_update"
    LEADERBOARD = "leaderboard"
    MILESTONE = "milestone"
    SIMULATION_START = "simulation_start"
    SIMULATION_END = "simulation_end"
    ERROR = "error"


@dataclass
class BacktestEvent:
    """Base class for backtest events."""
    event_type: EventType = field(default=EventType.DAY_TICK)
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    def to_sse(self) -> str:
        """Convert to Server-Sent Events format."""
        data = self.to_dict()
        return f"event: {self.event_type.value}\ndata: {json.dumps(data)}\n\n"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        d = asdict(self)
        d["event_type"] = self.event_type.value
        d["timestamp"] = self.timestamp.isoformat()
        return d


@dataclass
class DayTickEvent(BacktestEvent):
    """Emitted at the start of each trading day."""
    event_type: EventType = field(default=EventType.DAY_TICK)
    simulation_date: Optional[date] = None
    day_index: int = 0
    total_days: int = 0
    progress_pct: float = 0.0
    benchmark_value: float = 0.0  # SPY value
    benchmark_return: float = 0.0
    prices: Dict[str, float] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        d = super().to_dict()
        if self.simulation_date:
            d["simulation_date"] = self.simulation_date.isoformat()
        return d


@dataclass
class DebateStartEvent(BacktestEvent):
    """Emitted when a fund starts its debate cycle."""
    event_type: EventType = field(default=EventType.DEBATE_START)
    fund_id: str = ""
    fund_name: str = ""
    simulation_date: Optional[date] = None
    
    def to_dict(self) -> Dict[str, Any]:
        d = super().to_dict()
        if self.simulation_date:
            d["simulation_date"] = self.simulation_date.isoformat()
        return d


@dataclass
class DebateMessageEvent(BacktestEvent):
    """Emitted for each message in the debate."""
    event_type: EventType = field(default=EventType.DEBATE_MESSAGE)
    fund_id: str = ""
    phase: str = ""  # "analyze", "propose", "decide"
    model: str = ""  # "gemini", "gpt", "claude"
    content: str = ""
    tokens_used: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        d = super().to_dict()
        # Truncate content for SSE if too long
        if len(self.content) > 2000:
            d["content"] = self.content[:2000] + "..."
            d["truncated"] = True
        return d


@dataclass
class DecisionEvent(BacktestEvent):
    """Emitted when a fund makes a trading decision."""
    event_type: EventType = field(default=EventType.DECISION)
    fund_id: str = ""
    fund_name: str = ""
    simulation_date: Optional[date] = None
    action: str = ""  # "buy", "sell", "hold"
    symbol: Optional[str] = None
    quantity: Optional[float] = None
    target_weight: Optional[float] = None
    reasoning: str = ""
    confidence: float = 0.0
    signals_used: Dict[str, float] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        d = super().to_dict()
        if self.simulation_date:
            d["simulation_date"] = self.simulation_date.isoformat()
        return d


@dataclass
class TradeExecutedEvent(BacktestEvent):
    """Emitted when a trade is executed."""
    event_type: EventType = field(default=EventType.TRADE_EXECUTED)
    fund_id: str = ""
    fund_name: str = ""
    symbol: str = ""
    side: str = ""  # "buy" or "sell"
    quantity: float = 0.0
    price: float = 0.0
    commission: float = 0.0
    total_cost: float = 0.0
    reasoning: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        d = super().to_dict()
        return d


@dataclass
class PortfolioUpdateEvent(BacktestEvent):
    """Emitted after each day to update portfolio state."""
    event_type: EventType = field(default=EventType.PORTFOLIO_UPDATE)
    fund_id: str = ""
    fund_name: str = ""
    simulation_date: Optional[date] = None
    total_value: float = 0.0
    cash: float = 0.0
    invested_pct: float = 0.0
    daily_return: float = 0.0
    cumulative_return: float = 0.0
    max_drawdown: float = 0.0
    sharpe_ratio: float = 0.0
    positions: Dict[str, Dict] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        d = super().to_dict()
        if self.simulation_date:
            d["simulation_date"] = self.simulation_date.isoformat()
        return d


@dataclass
class FundRanking:
    """A single fund's ranking info."""
    fund_id: str
    fund_name: str
    rank: int
    total_value: float
    cumulative_return: float
    sharpe_ratio: float
    max_drawdown: float


@dataclass
class LeaderboardEvent(BacktestEvent):
    """Emitted to update the fund leaderboard."""
    event_type: EventType = field(default=EventType.LEADERBOARD)
    simulation_date: Optional[date] = None
    rankings: List[FundRanking] = field(default_factory=list)
    benchmark_return: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        d = super().to_dict()
        if self.simulation_date:
            d["simulation_date"] = self.simulation_date.isoformat()
        d["rankings"] = [
            {
                "fund_id": r.fund_id,
                "fund_name": r.fund_name,
                "rank": r.rank,
                "total_value": r.total_value,
                "cumulative_return": r.cumulative_return,
                "sharpe_ratio": r.sharpe_ratio,
                "max_drawdown": r.max_drawdown,
            }
            for r in self.rankings
        ]
        return d


@dataclass
class MilestoneEvent(BacktestEvent):
    """Emitted for notable events (new highs, drawdowns, market crashes)."""
    event_type: EventType = field(default=EventType.MILESTONE)
    milestone_type: str = ""  # "new_high", "drawdown", "market_crash", etc.
    fund_id: Optional[str] = None
    fund_name: Optional[str] = None
    description: str = ""
    value: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        d = super().to_dict()
        return d


@dataclass
class SimulationStartEvent(BacktestEvent):
    """Emitted when simulation starts."""
    event_type: EventType = field(default=EventType.SIMULATION_START)
    start_date: Optional[date] = None
    end_date: Optional[date] = None
    total_days: int = 0
    funds: List[Dict] = field(default_factory=list)
    universe: List[str] = field(default_factory=list)
    initial_cash: float = 100_000.0
    
    def to_dict(self) -> Dict[str, Any]:
        d = super().to_dict()
        if self.start_date:
            d["start_date"] = self.start_date.isoformat()
        if self.end_date:
            d["end_date"] = self.end_date.isoformat()
        return d


@dataclass
class SimulationEndEvent(BacktestEvent):
    """Emitted when simulation completes."""
    event_type: EventType = field(default=EventType.SIMULATION_END)
    elapsed_seconds: float = 0.0
    total_days_simulated: int = 0
    final_rankings: List[FundRanking] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        d = super().to_dict()
        d["final_rankings"] = [
            {
                "fund_id": r.fund_id,
                "fund_name": r.fund_name,
                "rank": r.rank,
                "total_value": r.total_value,
                "cumulative_return": r.cumulative_return,
                "sharpe_ratio": r.sharpe_ratio,
                "max_drawdown": r.max_drawdown,
            }
            for r in self.final_rankings
        ]
        return d


@dataclass
class ErrorEvent(BacktestEvent):
    """Emitted when an error occurs."""
    event_type: EventType = field(default=EventType.ERROR)
    error_type: str = ""
    message: str = ""
    fund_id: Optional[str] = None
    recoverable: bool = True
