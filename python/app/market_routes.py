"""Market context and sentiment API endpoints"""
from fastapi import APIRouter, HTTPException

router = APIRouter(prefix="/api/lab", tags=["market"])


@router.get("/context/{symbol}")
async def get_market_context(symbol: str):
    """
    Get deep market context for a symbol.

    Returns historical analysis, similar periods, and recommendations.
    """
    try:
        from core.context.market_context import MarketContextProvider
        provider = MarketContextProvider(persist_directory="./chroma_data")
        context = provider.get_deep_context(symbol.upper())

        return {
            "symbol": context.symbol,
            "current_date": context.current_date,
            "regime": context.current_regime.value,
            "volatility": context.current_volatility,
            "momentum_1m": context.current_momentum_1m,
            "momentum_3m": context.current_momentum_3m,
            "recommendation": context.recommended_stance,
            "confidence": context.confidence_score,
            "interpretation": context.market_interpretation,
            "avg_forward_return_1m": context.avg_forward_return_1m,
            "avg_forward_return_3m": context.avg_forward_return_3m,
            "positive_outcome_rate": context.positive_outcome_rate,
            "worst_case_drawdown": context.worst_case_drawdown,
            "key_risks": context.key_risks,
            "similar_periods": [
                {
                    "date": p.date,
                    "similarity": p.similarity,
                    "regime": p.regime.value,
                    "narrative": p.narrative,
                    "geopolitical_context": p.geopolitical_context,
                    "forward_return_1m": p.forward_outcome.return_1m,
                    "forward_return_3m": p.forward_outcome.return_3m
                }
                for p in context.similar_periods[:10]
            ]
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get context: {str(e)}"
        )


@router.get("/sentiment")
async def get_market_sentiment():
    """Get current market sentiment and external context"""
    try:
        from core.context.external_data import ExternalDataProvider
        provider = ExternalDataProvider()
        context = provider.get_full_context()

        return {
            "timestamp": context.timestamp.isoformat(),
            "sentiment": {
                "level": context.sentiment.sentiment_level.value,
                "vix": context.sentiment.vix_level,
                "vix_percentile": context.sentiment.vix_percentile,
                "breadth": context.sentiment.breadth,
                "interpretation": context.sentiment.interpretation
            },
            "economic": {
                "ten_year_yield": context.economic.ten_year_yield,
                "yield_curve_spread": context.economic.yield_curve_spread,
                "recession_risk": context.economic.is_recession_risk,
                "interpretation": context.economic.interpretation
            },
            "narrative": context.market_narrative,
            "geopolitical": context.geopolitical_summary
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get sentiment: {str(e)}"
        )
