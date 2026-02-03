"""
Pydantic schemas for LangChain structured outputs.

These schemas replace manual JSON parsing with validated Pydantic models.
LangChain's with_structured_output() uses these to ensure type-safe responses.
"""

from typing import List, Optional, Literal
from pydantic import BaseModel, Field


class MarketAnalysis(BaseModel):
    """
    Structured output for Phase 1: ANALYZE.
    
    Gemini analyzes market conditions and identifies opportunities.
    """
    opportunities: List[str] = Field(
        description="Key trading opportunities identified based on price movements"
    )
    risk_factors: List[str] = Field(
        description="Risk factors to consider before trading"
    )
    recommended_symbols: List[str] = Field(
        description="Symbols that align with the fund's thesis"
    )
    market_regime: str = Field(
        description="Current market regime (bullish, bearish, neutral, volatile)"
    )
    summary: str = Field(
        description="Concise summary of market analysis (200 words max)"
    )


class TradeProposal(BaseModel):
    """
    Structured output for Phase 2: PROPOSE.
    
    GPT proposes specific trades based on analysis.
    """
    action: Literal["buy", "sell", "hold"] = Field(
        description="Proposed action: buy, sell, or hold"
    )
    symbol: Optional[str] = Field(
        default=None,
        description="Symbol to trade (null for hold)"
    )
    target_weight: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=0.20,
        description="Target portfolio weight (0.0 to 0.20, max 20%)"
    )
    reasoning: str = Field(
        description="Detailed reasoning for the trade proposal"
    )
    risk_assessment: str = Field(
        description="Assessment of risks for this specific trade"
    )


class TradingDecisionOutput(BaseModel):
    """
    Structured output for Phase 3: DECIDE.
    
    GPT makes the final trading decision with confidence score.
    Replaces manual JSON parsing in debate_runner._parse_decision().
    """
    action: Literal["buy", "sell", "hold"] = Field(
        description="Final action: buy, sell, or hold"
    )
    symbol: Optional[str] = Field(
        default=None,
        description="Symbol to trade (null for hold)"
    )
    target_weight: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=0.20,
        description="Target portfolio weight (0.0 to 0.20)"
    )
    reasoning: str = Field(
        description="Brief explanation for the decision"
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Confidence level (0.0 to 1.0)"
    )


class RiskConfirmation(BaseModel):
    """
    Structured output for Phase 4: CONFIRM (Claude).
    
    Claude reviews major trades (>5% of portfolio) for risk.
    """
    approved: bool = Field(
        description="Whether the trade is approved"
    )
    reason: str = Field(
        description="Explanation for approval or rejection"
    )
    risk_score: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Risk score (0.0 = low risk, 1.0 = high risk)"
    )


class TradeDetail(BaseModel):
    """Individual trade within an enhanced decision."""
    action: Literal["buy", "sell"] = Field(
        description="Trade action"
    )
    symbol: str = Field(
        description="Symbol to trade"
    )
    size: float = Field(
        ge=0.0,
        le=0.20,
        description="Position size as portfolio weight"
    )
    reasoning: str = Field(
        description="Trade-specific reasoning"
    )
    historical_precedent: Optional[str] = Field(
        default=None,
        description="Historical date reference for similar trade"
    )
    expected_holding_period: Optional[str] = Field(
        default=None,
        description="Expected holding period (e.g., '3-6 months')"
    )
    stop_loss: Optional[float] = Field(
        default=None,
        description="Stop loss as negative percentage (e.g., -0.08 for 8%)"
    )
    target_return: Optional[float] = Field(
        default=None,
        description="Target return as positive percentage (e.g., 0.25 for 25%)"
    )


class EnhancedTradingDecision(BaseModel):
    """
    Enhanced trading decision with full context for llm_manager.
    
    Replaces manual JSON parsing in llm_manager._parse_response().
    """
    thesis: str = Field(
        description="Complete investment thesis (500+ words)"
    )
    conviction: float = Field(
        ge=0.0,
        le=1.0,
        description="Overall conviction level"
    )
    market_regime: str = Field(
        description="Current market regime assessment"
    )
    geopolitical_factors: List[str] = Field(
        default_factory=list,
        description="Geopolitical factors affecting the decision"
    )
    trades: List[TradeDetail] = Field(
        default_factory=list,
        description="List of specific trades to execute"
    )
    risks: List[str] = Field(
        default_factory=list,
        description="Key risks identified"
    )
    market_outlook: str = Field(
        description="Overall market assessment"
    )


class ResearchAnalysis(BaseModel):
    """
    Structured output for streaming analyzer AI synthesis.
    """
    interpretation: str = Field(
        description="2-3 sentence interpretation answering user's question"
    )
    key_insights: List[str] = Field(
        description="Actionable insights from the analysis"
    )
    sentiment: Literal["bullish", "bearish", "neutral"] = Field(
        description="Overall sentiment based on data"
    )
