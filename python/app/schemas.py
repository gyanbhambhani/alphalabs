from datetime import datetime, date
from decimal import Decimal
from typing import Optional, Literal
from pydantic import BaseModel, Field


# Manager schemas
class ManagerBase(BaseModel):
    name: str
    type: Literal["llm", "quant"]
    provider: Optional[Literal["openai", "anthropic", "google"]] = None
    description: Optional[str] = None


class ManagerCreate(ManagerBase):
    id: str


class ManagerResponse(ManagerBase):
    id: str
    is_active: bool
    
    class Config:
        from_attributes = True


# Portfolio schemas
class PortfolioResponse(BaseModel):
    manager_id: str = Field(alias="managerId")
    cash_balance: float = Field(alias="cashBalance")
    total_value: float = Field(alias="totalValue")
    updated_at: datetime = Field(alias="updatedAt")
    
    class Config:
        from_attributes = True
        populate_by_name = True


# Position schemas
class PositionResponse(BaseModel):
    id: int
    manager_id: str = Field(alias="managerId")
    symbol: str
    quantity: float
    avg_entry_price: float = Field(alias="avgEntryPrice")
    current_price: Optional[float] = Field(alias="currentPrice")
    unrealized_pnl: Optional[float] = Field(alias="unrealizedPnl")
    opened_at: datetime = Field(alias="openedAt")
    
    class Config:
        from_attributes = True
        populate_by_name = True


# Trade schemas
class TradeCreate(BaseModel):
    manager_id: str
    symbol: str
    side: Literal["buy", "sell"]
    quantity: float
    price: float
    reasoning: Optional[str] = None
    signals_used: Optional[dict] = None


class TradeResponse(BaseModel):
    id: int
    manager_id: str = Field(alias="managerId")
    symbol: str
    side: Literal["buy", "sell"]
    quantity: float
    price: float
    reasoning: Optional[str] = None
    signals_used: Optional[dict] = Field(alias="signalsUsed", default=None)
    executed_at: Optional[datetime] = Field(alias="executedAt")
    
    class Config:
        from_attributes = True
        populate_by_name = True


# Daily snapshot schemas
class DailySnapshotResponse(BaseModel):
    id: int
    manager_id: str = Field(alias="managerId")
    date: date
    portfolio_value: float = Field(alias="portfolioValue")
    daily_return: Optional[float] = Field(alias="dailyReturn")
    cumulative_return: Optional[float] = Field(alias="cumulativeReturn")
    sharpe_ratio: Optional[float] = Field(alias="sharpeRatio")
    
    class Config:
        from_attributes = True
        populate_by_name = True


# Strategy signal schemas
class MomentumSignal(BaseModel):
    symbol: str
    score: float  # -1 to +1


class MeanReversionSignal(BaseModel):
    symbol: str
    score: float  # -1 to +1


class TechnicalIndicators(BaseModel):
    symbol: str
    rsi: float
    macd: dict  # {macd, signal, histogram}
    sma20: float
    sma50: float
    sma200: float
    atr: float


class MLPrediction(BaseModel):
    symbol: str
    predicted_return: float = Field(alias="predictedReturn")
    confidence: float
    
    class Config:
        populate_by_name = True


class SimilarPeriod(BaseModel):
    date: str
    similarity: float
    return_5d: float = Field(alias="return5d")
    return_20d: float = Field(alias="return20d")
    
    class Config:
        populate_by_name = True


class SemanticSearchResult(BaseModel):
    similar_periods: list[SimilarPeriod] = Field(alias="similarPeriods")
    avg_5d_return: float = Field(alias="avg5dReturn")
    avg_20d_return: float = Field(alias="avg20dReturn")
    positive_5d_rate: float = Field(alias="positive5dRate")
    interpretation: str
    
    class Config:
        populate_by_name = True


class StrategySignals(BaseModel):
    momentum: list[MomentumSignal]
    mean_reversion: list[MeanReversionSignal] = Field(alias="meanReversion")
    technical: list[TechnicalIndicators]
    ml_prediction: list[MLPrediction] = Field(alias="mlPrediction")
    volatility_regime: str = Field(alias="volatilityRegime")
    semantic_search: SemanticSearchResult = Field(alias="semanticSearch")
    timestamp: datetime
    
    class Config:
        populate_by_name = True


# Leaderboard schemas
class LeaderboardEntry(BaseModel):
    rank: int
    manager: ManagerResponse
    portfolio: PortfolioResponse
    sharpe_ratio: float = Field(alias="sharpeRatio")
    total_return: float = Field(alias="totalReturn")
    volatility: float
    max_drawdown: float = Field(alias="maxDrawdown")
    total_trades: int = Field(alias="totalTrades")
    win_rate: float = Field(alias="winRate")
    
    class Config:
        populate_by_name = True


# Trading decision schemas
class TradingDecision(BaseModel):
    action: Literal["buy", "sell", "hold"]
    symbol: str
    size: float  # Percentage of portfolio (0-1)
    reasoning: str


class ManagerDecisions(BaseModel):
    manager_id: str
    decisions: list[TradingDecision]
    timestamp: datetime
