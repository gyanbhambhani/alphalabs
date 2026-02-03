"""
LangChain orchestration module for AI trading system.

Provides:
- Structured output schemas for LLM responses
- LangChain tools for market data and portfolio operations
- Chains for debate orchestration
- Agents for autonomous trading decisions
"""

from core.langchain.schemas import (
    MarketAnalysis,
    TradeProposal,
    TradingDecisionOutput,
    RiskConfirmation,
    EnhancedTradingDecision,
    TradeDetail,
    ResearchAnalysis,
)

from core.langchain.chains import (
    AnalyzeChain,
    ProposeChain,
    DecideChain,
    ConfirmChain,
    DebateSequence,
)

from core.langchain.agents import (
    TradingAgent,
    ResearchAgent,
    StreamingLLM,
    create_trading_agent,
    create_research_agent,
)

from core.langchain.tools import (
    MARKET_DATA_TOOLS,
    PORTFOLIO_TOOLS,
    RESEARCH_TOOLS,
    ALL_TOOLS,
)

__all__ = [
    # Schemas
    "MarketAnalysis",
    "TradeProposal",
    "TradingDecisionOutput",
    "RiskConfirmation",
    "EnhancedTradingDecision",
    "TradeDetail",
    "ResearchAnalysis",
    # Chains
    "AnalyzeChain",
    "ProposeChain",
    "DecideChain",
    "ConfirmChain",
    "DebateSequence",
    # Agents
    "TradingAgent",
    "ResearchAgent",
    "StreamingLLM",
    "create_trading_agent",
    "create_research_agent",
    # Tools
    "MARKET_DATA_TOOLS",
    "PORTFOLIO_TOOLS",
    "RESEARCH_TOOLS",
    "ALL_TOOLS",
]
