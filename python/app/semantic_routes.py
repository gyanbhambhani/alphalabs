"""Semantic vector search API endpoints"""
from pydantic import BaseModel
from fastapi import APIRouter, HTTPException

router = APIRouter(prefix="/api/lab", tags=["semantic"])


class SemanticSearchQuery(BaseModel):
    """Semantic search query request"""
    query: str
    top_k: int = 20
    search_mode: str = "current"  # "current" or "similar_to_query"


@router.post("/semantic-search/{symbol}")
async def semantic_vector_search(symbol: str, request: SemanticSearchQuery):
    """
    True semantic vector similarity search.

    Finds historical periods with similar market conditions using vector embeddings.

    Search modes:
    - "current": Find periods similar to current market conditions
    - "similar_to_query": Parse query to understand target conditions

    Returns similarity-ranked results with forward outcomes.
    """
    import yfinance as yf
    import pandas as pd
    import numpy as np
    from core.semantic.encoder import MarketStateEncoder
    from core.semantic.vector_db import VectorDatabase

    try:
        symbol = symbol.upper()

        # Get the vector database for this symbol
        db = VectorDatabase(
            persist_directory="./chroma_data",
            symbol=symbol
        )

        count = db.get_count()
        if count == 0:
            return {
                "results": [],
                "interpretation": f"No embeddings found for {symbol}. "
                    "Please generate embeddings first.",
                "avg_forward_return": 0.0,
                "positive_rate": 0.5
            }

        # Fetch recent price data to encode current state
        ticker = yf.Ticker(symbol)
        hist = ticker.history(period="2y")

        if hist.empty or len(hist) < 252:
            return {
                "results": [],
                "interpretation": f"Insufficient price data for {symbol}",
                "avg_forward_return": 0.0,
                "positive_rate": 0.5
            }

        # Normalize column names
        hist.columns = [c.lower() for c in hist.columns]

        # Encode current market state
        encoder = MarketStateEncoder()
        current_state = encoder.encode(
            date=str(hist.index[-1].date()),
            close=hist['close'],
            high=hist.get('high'),
            low=hist.get('low'),
            volume=hist.get('volume')
        )

        # Parse query for additional context
        query_lower = request.query.lower()
        interpretation_parts = []

        # Detect query intent
        if any(w in query_lower for w in ["crash", "selloff", "panic"]):
            interpretation_parts.append("Looking for crash/selloff periods")
        elif any(w in query_lower for w in ["rally", "bull", "surge"]):
            interpretation_parts.append("Looking for rally/bullish periods")
        elif any(w in query_lower for w in ["volatile", "volatility"]):
            interpretation_parts.append("Looking for high volatility periods")
        elif any(w in query_lower for w in ["similar", "like now", "current"]):
            interpretation_parts.append(
                "Finding periods similar to current conditions"
            )
        else:
            interpretation_parts.append(
                f"Semantic search for: {request.query}"
            )

        # Search by vector similarity
        search_results = db.search(
            query_vector=current_state.vector,
            top_k=request.top_k + 21  # Extra to filter recent
        )

        # Filter out recent dates and calculate forward outcomes
        from datetime import datetime, timedelta
        cutoff_date = (datetime.now() - timedelta(days=21)).strftime('%Y-%m-%d')

        results = []
        forward_returns = []

        for result in search_results:
            if result.date >= cutoff_date:
                continue

            # Use forward returns from stored metadata (pre-calculated during embedding)
            forward_5d = result.metadata.get("forward_5d_return")
            forward_10d = result.metadata.get("forward_10d_return")
            forward_1m = result.metadata.get("forward_1m_return")
            forward_3m = result.metadata.get("forward_3m_return")

            # Use 1-month forward return for statistics (more meaningful)
            if forward_1m is not None:
                forward_returns.append(forward_1m)
            elif forward_10d is not None:
                forward_returns.append(forward_10d)

            results.append({
                "date": result.date,
                "similarity": round(result.similarity, 4),
                "metadata": {
                    "date": result.metadata.get("date", result.date),
                    "return_1m": round(
                        result.metadata.get("return_1m", 0), 4
                    ),
                    "return_3m": round(
                        result.metadata.get("return_3m", 0), 4
                    ),
                    "volatility_21d": round(
                        result.metadata.get("volatility_21d", 0), 4
                    ),
                    "price": round(result.metadata.get("price", 0), 2),
                },
                "forward_return_5d": (
                    round(forward_5d, 4) if forward_5d is not None else None
                ),
                "forward_return_10d": (
                    round(forward_10d, 4) if forward_10d is not None else None
                ),
                "forward_return_1m": (
                    round(forward_1m, 4) if forward_1m is not None else None
                ),
                "forward_return_3m": (
                    round(forward_3m, 4) if forward_3m is not None else None
                ),
            })

            if len(results) >= request.top_k:
                break

        # Calculate summary statistics
        avg_forward = (
            np.mean(forward_returns) if forward_returns else 0.0
        )
        positive_rate = (
            sum(1 for r in forward_returns if r > 0) / len(forward_returns)
            if forward_returns else 0.5
        )

        # Build interpretation
        current_vol = current_state.metadata.get('volatility_21d', 0)
        current_mom = current_state.metadata.get('return_1m', 0)

        vol_desc = (
            "low" if current_vol < 0.15 else
            "high" if current_vol > 0.25 else "normal"
        )
        trend_desc = (
            "bullish" if current_mom > 0.02 else
            "bearish" if current_mom < -0.02 else "neutral"
        )

        interpretation_parts.append(
            f"Current: {vol_desc} volatility ({current_vol:.1%}), "
            f"{trend_desc} momentum ({current_mom:+.1%} monthly)"
        )
        interpretation_parts.append(
            f"Found {len(results)} similar periods. "
            f"Avg 1-month forward return: {avg_forward:+.1%} "
            f"({positive_rate:.0%} positive)"
        )

        return {
            "results": results,
            "interpretation": ". ".join(interpretation_parts),
            "avg_forward_return": round(avg_forward, 4),
            "positive_rate": round(positive_rate, 4),
            "current_state": {
                "date": current_state.date,
                "volatility_21d": round(current_vol, 4),
                "return_1m": round(current_mom, 4),
            }
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Semantic search failed: {str(e)}"
        )
