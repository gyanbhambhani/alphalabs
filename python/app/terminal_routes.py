"""AI Stock Terminal API endpoints (Stream-First Architecture)

BYOK (Bring Your Own Key): Pass X-OpenAI-API-Key header for user-provided keys.
Keys are used per-request only, never stored.
"""
from typing import Optional, List as TypingList
from pydantic import BaseModel, Field
from fastapi import APIRouter, Depends, HTTPException, Query, Request
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from starlette.responses import StreamingResponse
import json as json_module

from db import get_db, Stock

router = APIRouter(prefix="/api/search", tags=["terminal"])


class SearchSessionRequest(BaseModel):
    """Request to create a new search session"""
    query: str
    symbols: TypingList[str]


class SearchSessionResponse(BaseModel):
    """Response with session info"""
    session_id: str = Field(serialization_alias="sessionId")
    cached: bool
    cached_chunks: Optional[TypingList[dict]] = Field(
        serialization_alias="cachedChunks", default=None
    )

    model_config = {"populate_by_name": True}


@router.post("/session", response_model=SearchSessionResponse)
async def create_search_session(
    request: SearchSessionRequest,
    http_request: Request,
):
    """
    Create a search analysis session.

    Returns either:
    - A session_id for streaming (if not cached)
    - Cached results (if identical query was run recently)

    Usage:
    1. POST here to get session_id
    2. If cached=true, display cachedChunks directly
    3. If cached=false, connect to GET /api/search/analyze-stream?session_id=...
    """
    from core.ai.session_manager import get_session_manager

    session_manager = get_session_manager()

    # Validate symbols (at least one, max 5)
    symbols = [s.upper().strip() for s in request.symbols if s.strip()]
    if not symbols:
        raise HTTPException(status_code=400, detail="At least one symbol required")
    if len(symbols) > 5:
        symbols = symbols[:5]  # Limit to 5

    # Validate query
    query = request.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    if len(query) > 500:
        raise HTTPException(status_code=400, detail="Query too long (max 500 chars)")

    # BYOK: User-provided API key (never stored, used per-request only)
    byok_key = http_request.headers.get("X-OpenAI-API-Key", "").strip() or None

    try:
        session, cached_chunks = await session_manager.create_session(
            query=query,
            symbols=symbols,
            openai_api_key=byok_key,
        )

        if cached_chunks:
            # Return cached results immediately
            return SearchSessionResponse(
                session_id="",
                cached=True,
                cached_chunks=cached_chunks
            )

        if session:
            return SearchSessionResponse(
                session_id=session.id,
                cached=False,
                cached_chunks=None
            )

        # Shouldn't happen, but handle gracefully
        raise HTTPException(status_code=500, detail="Failed to create session")

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Session creation failed: {str(e)}"
        )


@router.get("/analyze-stream")
async def stream_analysis(
    session_id: str = Query(..., description="Session ID from POST /api/search/session"),
):
    """
    Stream analysis results via Server-Sent Events (SSE).

    Connect with EventSource after getting session_id from POST /api/search/session.

    Event types:
    - text: Plain text message
    - chart: Chart specification to render
    - table: Table specification to render
    - error: Error message
    - complete: Stream finished

    Example frontend usage:
    ```javascript
    const eventSource = new EventSource(`/api/search/analyze-stream?session_id=${sessionId}`);
    eventSource.onmessage = (event) => {
        const chunk = JSON.parse(event.data);
        // Handle chunk.type: 'text', 'chart', 'table', 'error', 'complete'
    };
    ```
    """
    from core.ai.session_manager import get_session_manager
    from core.ai.streaming_analyzer import StreamingQuantAnalyzer
    from app.config import get_settings

    settings = get_settings()
    session_manager = get_session_manager()

    # Get session
    session = await session_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found or expired")

    async def generate_events():
        """Generate SSE events from analyzer stream."""
        collected_chunks = []

        # BYOK: Use user key from session if provided, else server config
        api_key = session_manager.get_byok_key(session_id) or settings.openai_api_key
        if not api_key:
            error_data = json_module.dumps({
                "type": "error",
                "content": {
                    "message": (
                        "No API key. Add OPENAI_API_KEY to .env or pass "
                        "X-OpenAI-API-Key header (BYOK)."
                    )
                },
                "metadata": {}
            })
            yield f"data: {error_data}\n\n"
            return

        try:
            analyzer = StreamingQuantAnalyzer(openai_api_key=api_key)

            async for chunk in analyzer.analyze_stream(session):
                # Format as SSE
                data = json_module.dumps(chunk.model_dump())
                yield f"data: {data}\n\n"

                # Collect for caching
                collected_chunks.append(chunk)

            # Cache results for future identical queries
            if collected_chunks:
                await session_manager.cache_result(session, collected_chunks)

        except Exception as e:
            # Send error event
            error_data = json_module.dumps({
                "type": "error",
                "content": {"message": str(e)},
                "metadata": {}
            })
            yield f"data: {error_data}\n\n"

    return StreamingResponse(
        generate_events(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # Disable nginx buffering
            "Access-Control-Allow-Origin": "*",
        }
    )


@router.get("/symbols")
async def search_symbols(
    q: str = Query("", description="Search query"),
    limit: int = Query(10, ge=1, le=50),
    db: AsyncSession = Depends(get_db)
):
    """
    Search for stock symbols for autocomplete.

    Returns matching symbols sorted by relevance.
    """
    if not q or len(q) < 1:
        # Return popular symbols
        popular = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "SPY"]
        return {"symbols": popular[:limit]}

    try:
        search_pattern = f"%{q.upper()}%"

        result = await db.execute(
            select(Stock)
            .where(
                (Stock.symbol.ilike(search_pattern)) |
                (Stock.name.ilike(search_pattern))
            )
            .order_by(Stock.symbol)
            .limit(limit)
        )
        stocks = result.scalars().all()

        return {
            "symbols": [
                {
                    "symbol": s.symbol,
                    "name": s.name,
                    "sector": s.sector
                }
                for s in stocks
            ]
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Symbol search failed: {str(e)}"
        )


# =============================================================================
# Real-time data (Terminal ticker tape)
# =============================================================================

_terminal_router = APIRouter(prefix="/api/terminal", tags=["terminal"])


@_terminal_router.get("/live")
async def get_live_prices(
    symbols: str = Query(
        "SPY,AAPL,MSFT,GOOGL,AMZN,NVDA,META,TSLA",
        description="Comma-separated symbols",
    ),
):
    """
    Get live (or ~15min delayed) prices for ticker tape.

    Uses Yahoo Finance free tier. No API key required.
    """
    import yfinance as yf

    sym_list = [s.strip().upper() for s in symbols.split(",") if s.strip()][:20]
    if not sym_list:
        return {"prices": [], "timestamp": None}

    result = []
    for symbol in sym_list:
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.fast_info
            prev = getattr(info, "previous_close", None) or info.last_price
            change = (
                (info.last_price - prev) / prev if prev and prev != 0 else 0
            )
            result.append({
                "symbol": symbol,
                "price": round(info.last_price, 2),
                "change": round(change, 4),
                "change_pct": round(change * 100, 2),
            })
        except Exception:
            continue

    from datetime import datetime
    return {
        "prices": result,
        "timestamp": datetime.utcnow().isoformat(),
        "source": "yahoo_finance",
        "delay": "~15 min (free tier)",
    }
