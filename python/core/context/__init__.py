"""
Market Context Engine

Provides deep historical context, geopolitical analysis, and narrative generation
for conscious trading decisions.
"""
from core.context.market_context import (
    MarketContextProvider,
    DeepContext,
    HistoricalPeriodAnalysis,
    MarketRegime
)
from core.context.external_data import (
    ExternalDataProvider,
    ExternalContext,
    MarketSentiment,
    EconomicContext,
    SentimentLevel
)

__all__ = [
    "MarketContextProvider",
    "DeepContext",
    "HistoricalPeriodAnalysis",
    "MarketRegime",
    "ExternalDataProvider",
    "ExternalContext",
    "MarketSentiment",
    "EconomicContext",
    "SentimentLevel"
]
