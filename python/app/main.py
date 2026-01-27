from datetime import datetime
from fastapi import FastAPI, Depends, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, desc
from typing import Optional

from app.config import get_settings
from app.schemas import (
    ManagerResponse, PortfolioResponse, PositionResponse,
    TradeResponse, DailySnapshotResponse, StrategySignals,
    LeaderboardEntry, MomentumSignal, MeanReversionSignal,
    SemanticSearchResult, SimilarPeriod, TechnicalIndicators,
    MLPrediction, EmbeddingResponse, EmbeddingMetadata,
    EmbeddingSearchQuery, EmbeddingSearchResult,
    EmbeddingsStatsResponse, EmbeddingsListResponse,
    StockResponse, StocksListResponse
)
from db import get_db, Manager, Portfolio, Position, Trade, DailySnapshot, Stock
from app.query_parser import parse_query
from core.semantic.vector_db import VectorDatabase

settings = get_settings()

app = FastAPI(
    title=settings.app_name,
    description="AI Trading Lab - Where AI Portfolio Managers Compete",
    version="0.1.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify actual origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    return {
        "name": settings.app_name,
        "status": "running",
        "timestamp": datetime.utcnow().isoformat()
    }


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "timestamp": datetime.utcnow().isoformat(),
        "service": "ai-trading-lab-api"
    }


# Manager endpoints
@app.get("/api/managers", response_model=list[ManagerResponse])
async def get_managers(db: AsyncSession = Depends(get_db)):
    result = await db.execute(
        select(Manager).where(Manager.is_active == True)
    )
    managers = result.scalars().all()
    return managers


@app.get("/api/managers/{manager_id}", response_model=ManagerResponse)
async def get_manager(manager_id: str, db: AsyncSession = Depends(get_db)):
    result = await db.execute(
        select(Manager).where(Manager.id == manager_id)
    )
    manager = result.scalar_one_or_none()
    if not manager:
        raise HTTPException(status_code=404, detail="Manager not found")
    return manager


# Portfolio endpoints
@app.get("/api/portfolios", response_model=list[PortfolioResponse])
async def get_portfolios(db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(Portfolio))
    portfolios = result.scalars().all()
    return portfolios


@app.get("/api/portfolios/{manager_id}", response_model=PortfolioResponse)
async def get_portfolio(manager_id: str, db: AsyncSession = Depends(get_db)):
    result = await db.execute(
        select(Portfolio).where(Portfolio.manager_id == manager_id)
    )
    portfolio = result.scalar_one_or_none()
    if not portfolio:
        raise HTTPException(status_code=404, detail="Portfolio not found")
    return portfolio


# Position endpoints
@app.get("/api/positions/{manager_id}", response_model=list[PositionResponse])
async def get_positions(manager_id: str, db: AsyncSession = Depends(get_db)):
    result = await db.execute(
        select(Position).where(Position.manager_id == manager_id)
    )
    positions = result.scalars().all()
    return positions


# Trade endpoints
@app.get("/api/trades", response_model=list[TradeResponse])
async def get_trades(
    manager_id: Optional[str] = Query(None),
    limit: int = Query(50, ge=1, le=500),
    db: AsyncSession = Depends(get_db)
):
    query = select(Trade).order_by(desc(Trade.executed_at)).limit(limit)
    if manager_id:
        query = query.where(Trade.manager_id == manager_id)
    result = await db.execute(query)
    trades = result.scalars().all()
    return trades


# Performance endpoints
@app.get(
    "/api/performance/{manager_id}", 
    response_model=list[DailySnapshotResponse]
)
async def get_performance(
    manager_id: str,
    days: int = Query(30, ge=1, le=365),
    db: AsyncSession = Depends(get_db)
):
    result = await db.execute(
        select(DailySnapshot)
        .where(DailySnapshot.manager_id == manager_id)
        .order_by(desc(DailySnapshot.date))
        .limit(days)
    )
    snapshots = result.scalars().all()
    return snapshots


# Strategy signals endpoint - REAL DATA from yfinance
@app.get("/api/signals", response_model=StrategySignals)
async def get_signals(force_refresh: bool = False):
    """
    Get current strategy signals computed from live market data.
    
    Uses yfinance to fetch real prices and computes:
    - Momentum signals (12M return, skip 1M)
    - Mean reversion signals (Bollinger Z-score)
    - Technical indicators (RSI, MACD, SMAs, ATR)
    - Volatility regime detection
    
    Results are cached for 5 minutes to avoid rate limits.
    """
    try:
        from core.strategies.signals_service import get_cached_signals
        
        # Get real signals (cached for 5 min)
        result = await get_cached_signals(force_refresh=force_refresh)
        
        # Convert to API response format
        momentum_signals = [
            MomentumSignal(symbol=s["symbol"], score=s["score"])
            for s in result.momentum
        ]
        
        mean_reversion_signals = [
            MeanReversionSignal(symbol=s["symbol"], score=s["score"])
            for s in result.mean_reversion
        ]
        
        technical_indicators = [
            TechnicalIndicators(
                symbol=t["symbol"],
                rsi=t["rsi"],
                macd=t["macd"],
                sma20=t["sma20"],
                sma50=t["sma50"],
                sma200=t["sma200"],
                atr=t["atr"]
            )
            for t in result.technical
        ]
        
        # Build interpretation based on regime
        regime = result.volatility_regime
        vol_pct = f"{result.realized_vol * 100:.1f}%"
        
        if "low_vol" in regime and "up" in regime:
            interpretation = (
                f"Low volatility ({vol_pct} realized) with upward trend. "
                "Momentum strategies tend to outperform in this regime. "
                "Focus on winners with strong 12-month returns."
            )
        elif "high_vol" in regime:
            interpretation = (
                f"High volatility regime ({vol_pct} realized). "
                "Mean reversion strategies may work better. "
                "Look for oversold names with high positive z-scores."
            )
        elif "down" in regime:
            interpretation = (
                f"Downward trend detected ({vol_pct} vol). "
                "Consider defensive positioning or reduced exposure. "
                "Watch for reversal signals in oversold names."
            )
        else:
            interpretation = (
                f"Normal volatility ({vol_pct} realized), ranging market. "
                "Mixed signals - consider balanced approach across strategies."
            )
        
        return StrategySignals(
            momentum=momentum_signals,
            meanReversion=mean_reversion_signals,
            technical=technical_indicators,
            mlPrediction=[],  # No ML predictions yet
            volatilityRegime=regime,
            semanticSearch=SemanticSearchResult(
                similarPeriods=[],  # TODO: Hook up to embeddings search
                avg5dReturn=0.0,
                avg20dReturn=0.0,
                positive5dRate=0.0,
                interpretation=interpretation
            ),
            timestamp=result.timestamp,
            dataFreshness=result.data_freshness
        )
        
    except Exception as e:
        # Fallback to minimal response on error
        print(f"Signal generation error: {e}")
        return StrategySignals(
            momentum=[],
            meanReversion=[],
            technical=[],
            mlPrediction=[],
            volatilityRegime="unknown",
            semanticSearch=SemanticSearchResult(
                similarPeriods=[],
                avg5dReturn=0.0,
                avg20dReturn=0.0,
                positive5dRate=0.0,
                interpretation=f"Error fetching market data: {str(e)}"
            ),
            timestamp=datetime.utcnow(),
            dataFreshness="error"
        )


# Leaderboard endpoint
@app.get("/api/leaderboard", response_model=list[LeaderboardEntry])
async def get_leaderboard(db: AsyncSession = Depends(get_db)):
    """Get ranked leaderboard by Sharpe ratio"""
    # Get all managers with their latest snapshots
    managers_result = await db.execute(
        select(Manager).where(Manager.is_active == True)
    )
    managers = managers_result.scalars().all()
    
    entries = []
    for manager in managers:
        # Get portfolio
        portfolio_result = await db.execute(
            select(Portfolio).where(Portfolio.manager_id == manager.id)
        )
        portfolio = portfolio_result.scalar_one_or_none()
        
        # Get latest snapshot
        snapshot_result = await db.execute(
            select(DailySnapshot)
            .where(DailySnapshot.manager_id == manager.id)
            .order_by(desc(DailySnapshot.date))
            .limit(1)
        )
        snapshot = snapshot_result.scalar_one_or_none()
        
        # Get trade count
        trades_result = await db.execute(
            select(Trade).where(Trade.manager_id == manager.id)
        )
        trades = trades_result.scalars().all()
        
        if portfolio and snapshot:
            entries.append({
                "manager": manager,
                "portfolio": portfolio,
                "sharpe_ratio": float(snapshot.sharpe_ratio or 0),
                "total_return": float(snapshot.cumulative_return or 0),
                "volatility": float(snapshot.volatility or 0),
                "max_drawdown": float(snapshot.max_drawdown or 0),
                "total_trades": len(trades),
                "win_rate": float(snapshot.win_rate or 0),
            })
    
    # Sort by Sharpe ratio
    entries.sort(key=lambda x: x["sharpe_ratio"], reverse=True)
    
    # Add ranks
    for i, entry in enumerate(entries):
        entry["rank"] = i + 1
    
    return entries


# Trading cycle trigger
@app.post("/api/trading/cycle")
async def trigger_trading_cycle():
    """Manually trigger a trading cycle"""
    try:
        from core.execution.scheduler import TradingScheduler
        
        scheduler = TradingScheduler()
        result = await scheduler.run_once()
        
        return {
            "success": True,
            "message": "Trading cycle completed",
            "result": result
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Trading cycle failed: {str(e)}"
        )


# Embeddings endpoints
@app.get("/api/embeddings/stats", response_model=EmbeddingsStatsResponse)
async def get_embeddings_stats():
    """Get summary statistics about the embeddings database"""
    try:
        db = VectorDatabase(persist_directory="./chroma_data")
        
        total_count = db.get_count()
        
        if total_count == 0:
            return EmbeddingsStatsResponse(
                totalCount=0,
                dateRange=("", ""),
                avgReturn1m=0.0,
                avgVolatility21d=0.0
            )
        
        # Get date range
        date_range = db.get_date_range()
        
        # Get all data to calculate averages
        all_data = db.collection.get(include=['metadatas'])
        
        # Calculate averages
        returns_1m = [
            m.get('return_1m', 0) 
            for m in all_data['metadatas']
        ]
        vols_21d = [
            m.get('volatility_21d', 0) 
            for m in all_data['metadatas']
        ]
        
        avg_return = sum(returns_1m) / len(returns_1m) if returns_1m else 0
        avg_vol = sum(vols_21d) / len(vols_21d) if vols_21d else 0
        
        return EmbeddingsStatsResponse(
            totalCount=total_count,
            dateRange=date_range,
            avgReturn1m=avg_return,
            avgVolatility21d=avg_vol
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get stats: {str(e)}"
        )


@app.get("/api/embeddings", response_model=EmbeddingsListResponse)
async def get_embeddings(
    page: int = Query(1, ge=1),
    per_page: int = Query(50, ge=1, le=500),
    sort_by: str = Query("date", 
                         pattern="^(date|return_1m|volatility_21d|price)$"),
    order: str = Query("desc", pattern="^(asc|desc)$")
):
    """List all embeddings with pagination and sorting"""
    try:
        db = VectorDatabase(persist_directory="./chroma_data")
        
        # Get all data
        all_data = db.collection.get(include=['metadatas'])
        
        if not all_data['ids']:
            return EmbeddingsListResponse(
                embeddings=[],
                total=0,
                page=page,
                perPage=per_page
            )
        
        # Create list of embeddings with metadata
        embeddings_list = []
        for i, doc_id in enumerate(all_data['ids']):
            metadata = all_data['metadatas'][i]
            
            embeddings_list.append({
                'id': doc_id,
                'metadata': metadata,
                'sort_key': metadata.get(sort_by, doc_id)
            })
        
        # Sort
        reverse = (order == "desc")
        embeddings_list.sort(key=lambda x: x['sort_key'], reverse=reverse)
        
        # Paginate
        total = len(embeddings_list)
        start_idx = (page - 1) * per_page
        end_idx = start_idx + per_page
        page_data = embeddings_list[start_idx:end_idx]
        
        # Convert to response format
        embeddings = [
            EmbeddingResponse(
                id=item['id'],
                metadata=EmbeddingMetadata(**item['metadata'])
            )
            for item in page_data
        ]
        
        return EmbeddingsListResponse(
            embeddings=embeddings,
            total=total,
            page=page,
            perPage=per_page
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get embeddings: {str(e)}"
        )


@app.post("/api/embeddings/search")
async def search_embeddings(query: EmbeddingSearchQuery):
    """Natural language semantic search of market embeddings"""
    try:
        db = VectorDatabase(persist_directory="./chroma_data")
        
        # Parse the natural language query
        where_filter, interpretation = parse_query(query.query)
        
        # Get matching results
        if where_filter:
            # Query with filters
            results = db.collection.get(
                where=where_filter,
                limit=query.top_k,
                include=['metadatas']
            )
        else:
            # Get all results (no specific filters)
            all_data = db.collection.get(include=['metadatas'])
            
            # Limit to top_k
            results = {
                'ids': all_data['ids'][:query.top_k],
                'metadatas': all_data['metadatas'][:query.top_k]
            }
        
        # Convert to response format
        search_results = []
        for i, doc_id in enumerate(results['ids']):
            metadata = results['metadatas'][i]
            
            search_results.append(
                EmbeddingSearchResult(
                    id=doc_id,
                    metadata=EmbeddingMetadata(**metadata),
                    similarity=1.0,  # Metadata search, all matches equally
                    queryInterpretation=interpretation
                )
            )
        
        return {
            "results": search_results,
            "interpretation": interpretation,
            "total": len(search_results)
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Search failed: {str(e)}"
        )


# Stock endpoints
@app.get("/api/stocks", response_model=StocksListResponse)
async def get_stocks(
    sector: Optional[str] = Query(None),
    has_embeddings: Optional[bool] = Query(None),
    search: Optional[str] = Query(None),
    db: AsyncSession = Depends(get_db)
):
    """
    Get list of stocks with optional filtering.
    
    Query params:
        sector: Filter by sector
        has_embeddings: Filter by embedding availability
        search: Search by symbol or name
    """
    try:
        # Build query
        query = select(Stock)
        
        if sector:
            query = query.where(Stock.sector == sector)
        
        if has_embeddings is not None:
            query = query.where(Stock.has_embeddings == has_embeddings)
        
        if search:
            search_pattern = f"%{search.upper()}%"
            query = query.where(
                (Stock.symbol.ilike(search_pattern)) | 
                (Stock.name.ilike(search_pattern))
            )
        
        query = query.order_by(Stock.symbol)
        
        result = await db.execute(query)
        stocks = result.scalars().all()
        
        # Get unique sectors
        sectors_result = await db.execute(
            select(Stock.sector).distinct().where(Stock.sector.isnot(None))
        )
        sectors = sorted([s for s in sectors_result.scalars().all() if s])
        
        return StocksListResponse(
            stocks=[StockResponse.from_orm(s) for s in stocks],
            total=len(stocks),
            sectors=sectors
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get stocks: {str(e)}"
        )


@app.get("/api/stocks/{symbol}", response_model=StockResponse)
async def get_stock(symbol: str, db: AsyncSession = Depends(get_db)):
    """Get details for a specific stock"""
    try:
        result = await db.execute(
            select(Stock).where(Stock.symbol == symbol.upper())
        )
        stock = result.scalar_one_or_none()
        
        if not stock:
            raise HTTPException(status_code=404, detail="Stock not found")
        
        return StockResponse.from_orm(stock)
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get stock: {str(e)}"
        )


# Update embeddings endpoints to support symbol parameter
@app.get(
    "/api/embeddings/stats/{symbol}", 
    response_model=EmbeddingsStatsResponse
)
async def get_embeddings_stats_for_symbol(symbol: str):
    """Get embedding statistics for a specific stock"""
    try:
        db = VectorDatabase(
            persist_directory="./chroma_data",
            symbol=symbol.upper()
        )
        
        total_count = db.get_count()
        
        if total_count == 0:
            return EmbeddingsStatsResponse(
                totalCount=0,
                dateRange=("", ""),
                avgReturn1m=0.0,
                avgVolatility21d=0.0
            )
        
        date_range = db.get_date_range()
        
        # Get all data to calculate averages
        all_data = db.collection.get(include=['metadatas'])
        
        returns_1m = [m.get('return_1m', 0) for m in all_data['metadatas']]
        vols_21d = [
            m.get('volatility_21d', 0) 
            for m in all_data['metadatas']
        ]
        
        avg_return = sum(returns_1m) / len(returns_1m) if returns_1m else 0
        avg_vol = sum(vols_21d) / len(vols_21d) if vols_21d else 0
        
        return EmbeddingsStatsResponse(
            totalCount=total_count,
            dateRange=date_range,
            avgReturn1m=avg_return,
            avgVolatility21d=avg_vol
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get stats: {str(e)}"
        )


@app.get(
    "/api/embeddings/{symbol}", 
    response_model=EmbeddingsListResponse
)
async def get_embeddings_for_symbol(
    symbol: str,
    page: int = Query(1, ge=1),
    per_page: int = Query(50, ge=1, le=500),
    sort_by: str = Query("date", 
                         pattern="^(date|return_1m|volatility_21d|price)$"),
    order: str = Query("desc", pattern="^(asc|desc)$")
):
    """List embeddings for a specific stock"""
    try:
        db = VectorDatabase(
            persist_directory="./chroma_data",
            symbol=symbol.upper()
        )
        
        # Get all data
        all_data = db.collection.get(include=['metadatas'])
        
        if not all_data['ids']:
            return EmbeddingsListResponse(
                embeddings=[],
                total=0,
                page=page,
                perPage=per_page
            )
        
        # Create list with sort keys
        embeddings_list = []
        for i, doc_id in enumerate(all_data['ids']):
            metadata = all_data['metadatas'][i]
            
            embeddings_list.append({
                'id': doc_id,
                'metadata': metadata,
                'sort_key': metadata.get(sort_by, doc_id)
            })
        
        # Sort
        reverse = (order == "desc")
        embeddings_list.sort(key=lambda x: x['sort_key'], reverse=reverse)
        
        # Paginate
        total = len(embeddings_list)
        start_idx = (page - 1) * per_page
        end_idx = start_idx + per_page
        page_data = embeddings_list[start_idx:end_idx]
        
        # Convert to response
        embeddings = [
            EmbeddingResponse(
                id=item['id'],
                metadata=EmbeddingMetadata(**item['metadata'])
            )
            for item in page_data
        ]
        
        return EmbeddingsListResponse(
            embeddings=embeddings,
            total=total,
            page=page,
            perPage=per_page
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get embeddings: {str(e)}"
        )


@app.post("/api/embeddings/search/{symbol}")
async def search_embeddings_for_symbol(
    symbol: str, 
    query: EmbeddingSearchQuery
):
    """Search embeddings for a specific stock"""
    try:
        db = VectorDatabase(
            persist_directory="./chroma_data",
            symbol=symbol.upper()
        )
        
        # Parse query
        where_filter, interpretation = parse_query(query.query)
        
        # Get matching results
        if where_filter:
            results = db.collection.get(
                where=where_filter,
                limit=query.top_k,
                include=['metadatas']
            )
        else:
            all_data = db.collection.get(include=['metadatas'])
            results = {
                'ids': all_data['ids'][:query.top_k],
                'metadatas': all_data['metadatas'][:query.top_k]
            }
        
        # Convert to response
        search_results = []
        for i, doc_id in enumerate(results['ids']):
            metadata = results['metadatas'][i]
            
            search_results.append(
                EmbeddingSearchResult(
                    id=doc_id,
                    metadata=EmbeddingMetadata(**metadata),
                    similarity=1.0,
                    queryInterpretation=interpretation
                )
            )
        
        return {
            "results": search_results,
            "interpretation": interpretation,
            "total": len(search_results)
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Search failed: {str(e)}"
        )


# =============================================================================
# Trading Lab Chat Endpoints
# =============================================================================

from pydantic import BaseModel
from typing import List as TypingList


class ChatQuery(BaseModel):
    """Chat query request"""
    query: str
    conversation_id: Optional[str] = None


class ChatMessageResponse(BaseModel):
    """Chat message response"""
    message: str
    query_type: str
    data: Optional[dict] = None
    suggestions: TypingList[str] = []
    sources: TypingList[str] = []


# Store chat instances per conversation
_chat_instances: dict = {}


def get_chat_instance(conversation_id: str = "default"):
    """Get or create chat instance for a conversation"""
    if conversation_id not in _chat_instances:
        from app.chat import TradingLabChat
        _chat_instances[conversation_id] = TradingLabChat(
            persist_directory="./chroma_data"
        )
    return _chat_instances[conversation_id]


@app.post("/api/lab/chat", response_model=ChatMessageResponse)
async def lab_chat(query: ChatQuery):
    """
    Handle conversational queries about markets.
    
    Examples:
    - "What happened after every Fed rate hike?"
    - "Find periods similar to current conditions"
    - "Compare AAPL and MSFT"
    - "Generate a research report on NVDA"
    """
    try:
        conversation_id = query.conversation_id or "default"
        chat = get_chat_instance(conversation_id)
        
        response = await chat.handle_query(query.query)
        
        return ChatMessageResponse(
            message=response.message,
            query_type=response.query_type.value,
            data=response.data,
            suggestions=response.suggestions,
            sources=response.sources
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Chat error: {str(e)}"
        )


@app.get("/api/lab/suggestions")
async def get_suggestions():
    """Get example queries for the chat"""
    chat = get_chat_instance()
    return {
        "suggestions": chat.example_queries
    }


@app.get("/api/lab/history/{conversation_id}")
async def get_chat_history(conversation_id: str):
    """Get conversation history"""
    chat = get_chat_instance(conversation_id)
    return {
        "history": chat.get_conversation_history()
    }


@app.delete("/api/lab/history/{conversation_id}")
async def clear_chat_history(conversation_id: str):
    """Clear conversation history"""
    chat = get_chat_instance(conversation_id)
    chat.clear_history()
    return {"status": "cleared"}


@app.get("/api/lab/context/{symbol}")
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


@app.get("/api/lab/sentiment")
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


# =============================================================================
# Research Query Engine Endpoints
# =============================================================================

class ResearchQueryRequest(BaseModel):
    """Research query request"""
    query_type: str = "stock_analysis"
    symbols: TypingList[str]
    topic: Optional[str] = None
    time_period: str = "1y"
    compare_to: Optional[TypingList[str]] = None


class ResearchReportResponse(BaseModel):
    """Research report response"""
    title: str
    report_type: str
    generated_at: str
    symbols: TypingList[str]
    executive_summary: str
    sections: TypingList[dict]
    recommendation: str
    confidence: float
    key_risks: TypingList[str]
    sources: TypingList[str]
    markdown: str


@app.post("/api/lab/research", response_model=ResearchReportResponse)
async def generate_research_report(request: ResearchQueryRequest):
    """
    Generate an AI-powered research report.
    
    Report types:
    - stock_analysis: Deep dive into a single stock
    - market_outlook: Overall market conditions
    - historical_comparison: Compare current to historical periods
    - risk_assessment: Focus on risk factors
    - trade_idea: Actionable trade setup
    """
    try:
        from core.research.engine import ResearchEngine, ResearchQuery, ReportType
        
        engine = ResearchEngine(persist_directory="./chroma_data")
        
        # Map string to enum
        report_type_map = {
            "stock_analysis": ReportType.STOCK_ANALYSIS,
            "market_outlook": ReportType.MARKET_OUTLOOK,
            "historical_comparison": ReportType.HISTORICAL_COMPARISON,
            "risk_assessment": ReportType.RISK_ASSESSMENT,
            "trade_idea": ReportType.TRADE_IDEA
        }
        
        query = ResearchQuery(
            query_type=report_type_map.get(
                request.query_type, 
                ReportType.STOCK_ANALYSIS
            ),
            symbols=request.symbols,
            topic=request.topic,
            time_period=request.time_period,
            compare_to=request.compare_to
        )
        
        report = await engine.generate_report(query)
        
        return ResearchReportResponse(
            title=report.title,
            report_type=report.report_type.value,
            generated_at=report.generated_at.isoformat(),
            symbols=report.symbols,
            executive_summary=report.executive_summary,
            sections=[
                {"title": s.title, "content": s.content, "data": s.data}
                for s in report.sections
            ],
            recommendation=report.recommendation,
            confidence=report.confidence,
            key_risks=report.key_risks,
            sources=report.sources,
            markdown=report.to_markdown()
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Research generation failed: {str(e)}"
        )


@app.get("/api/lab/research/{symbol}")
async def get_stock_research(
    symbol: str,
    report_type: str = Query(
        default="stock_analysis",
        pattern="^(stock_analysis|risk_assessment|trade_idea)$"
    )
):
    """
    Quick research report for a single symbol.
    
    Shortcut endpoint that generates the specified report type.
    """
    try:
        from core.research.engine import ResearchEngine, ResearchQuery, ReportType
        
        engine = ResearchEngine(persist_directory="./chroma_data")
        
        type_map = {
            "stock_analysis": ReportType.STOCK_ANALYSIS,
            "risk_assessment": ReportType.RISK_ASSESSMENT,
            "trade_idea": ReportType.TRADE_IDEA
        }
        
        query = ResearchQuery(
            query_type=type_map.get(report_type, ReportType.STOCK_ANALYSIS),
            symbols=[symbol.upper()]
        )
        
        report = await engine.generate_report(query)
        
        return {
            "title": report.title,
            "report_type": report.report_type.value,
            "symbol": symbol.upper(),
            "executive_summary": report.executive_summary,
            "recommendation": report.recommendation,
            "confidence": report.confidence,
            "key_risks": report.key_risks,
            "markdown": report.to_markdown()
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Research generation failed: {str(e)}"
        )


@app.get("/api/lab/research/market/outlook")
async def get_market_outlook():
    """
    Get current market outlook report.
    
    Analyzes major indices and market sentiment.
    """
    try:
        from core.research.engine import ResearchEngine, ResearchQuery, ReportType
        
        engine = ResearchEngine(persist_directory="./chroma_data")
        
        query = ResearchQuery(
            query_type=ReportType.MARKET_OUTLOOK,
            symbols=["SPY", "QQQ", "IWM"]
        )
        
        report = await engine.generate_report(query)
        
        return {
            "title": report.title,
            "executive_summary": report.executive_summary,
            "recommendation": report.recommendation,
            "confidence": report.confidence,
            "key_risks": report.key_risks,
            "markdown": report.to_markdown()
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Market outlook generation failed: {str(e)}"
        )


# =============================================================================
# Collaborative Funds Endpoints
# =============================================================================

from pydantic import Field


class FundResponse(BaseModel):
    """Fund summary response"""
    fund_id: str = Field(serialization_alias="fundId")
    name: str
    strategy: str
    description: Optional[str] = None
    total_value: float = Field(serialization_alias="totalValue")
    cash_balance: float = Field(serialization_alias="cashBalance")
    gross_exposure: float = Field(serialization_alias="grossExposure")
    net_exposure: float = Field(serialization_alias="netExposure")
    n_positions: int = Field(serialization_alias="nPositions")
    is_active: bool = Field(serialization_alias="isActive")
    
    model_config = {"from_attributes": True, "populate_by_name": True}


class FundDetailResponse(FundResponse):
    """Detailed fund response including thesis and policy"""
    thesis: Optional[dict] = None
    policy: Optional[dict] = None
    risk_limits: Optional[dict] = Field(serialization_alias="riskLimits", default=None)


class FundPositionResponse(BaseModel):
    """Fund position response"""
    symbol: str
    quantity: float
    avg_entry_price: float = Field(serialization_alias="avgEntryPrice")
    current_price: float = Field(serialization_alias="currentPrice")
    market_value: float = Field(serialization_alias="marketValue")
    unrealized_pnl: float = Field(serialization_alias="unrealizedPnl")
    weight_pct: float = Field(serialization_alias="weightPct")
    
    model_config = {"populate_by_name": True}


class DecisionResponse(BaseModel):
    """Decision record response"""
    decision_id: str = Field(serialization_alias="decisionId")
    fund_id: str = Field(serialization_alias="fundId")
    asof_timestamp: str = Field(serialization_alias="asofTimestamp")
    decision_type: str = Field(serialization_alias="decisionType")
    status: str
    no_trade_reason: Optional[str] = Field(
        serialization_alias="noTradeReason", default=None
    )
    universe_hash: Optional[str] = Field(
        serialization_alias="universeHash", default=None
    )
    inputs_hash: Optional[str] = Field(
        serialization_alias="inputsHash", default=None
    )
    predicted_directions: Optional[dict] = Field(
        serialization_alias="predictedDirections", default=None
    )
    expected_return: Optional[float] = Field(
        serialization_alias="expectedReturn", default=None
    )
    
    model_config = {"populate_by_name": True}


class DebateResponse(BaseModel):
    """Debate transcript summary response"""
    transcript_id: str = Field(serialization_alias="transcriptId")
    fund_id: str = Field(serialization_alias="fundId")
    started_at: str = Field(serialization_alias="startedAt")
    completed_at: Optional[str] = Field(
        serialization_alias="completedAt", default=None
    )
    num_proposals: int = Field(serialization_alias="numProposals")
    num_critiques: int = Field(serialization_alias="numCritiques")
    final_consensus_level: float = Field(serialization_alias="finalConsensusLevel")
    
    model_config = {"populate_by_name": True}


class FundLeaderboardEntry(BaseModel):
    """Fund leaderboard entry"""
    rank: int
    fund_id: str = Field(serialization_alias="fundId")
    name: str
    strategy: str
    total_value: float = Field(serialization_alias="totalValue")
    gross_exposure: float = Field(serialization_alias="grossExposure")
    is_active: bool = Field(serialization_alias="isActive")
    
    model_config = {"populate_by_name": True}


@app.get("/api/funds", response_model=list[FundResponse])
async def get_funds(db: AsyncSession = Depends(get_db)):
    """Get all funds"""
    from db.models import FundModel
    
    result = await db.execute(select(FundModel))
    funds = result.scalars().all()
    
    return [
        FundResponse(
            fund_id=f.id,
            name=f.name,
            strategy=f.strategy,
            description=f.description,
            total_value=float(f.total_value or 0),
            cash_balance=float(f.cash_balance or 0),
            gross_exposure=0.0,  # Computed on demand
            net_exposure=0.0,
            n_positions=0,
            is_active=f.is_active,
        )
        for f in funds
    ]


@app.get("/api/funds/{fund_id}", response_model=FundDetailResponse)
async def get_fund(fund_id: str, db: AsyncSession = Depends(get_db)):
    """Get fund details"""
    from db.models import FundModel, FundPosition
    
    result = await db.execute(
        select(FundModel).where(FundModel.id == fund_id)
    )
    fund = result.scalar_one_or_none()
    if not fund:
        raise HTTPException(status_code=404, detail="Fund not found")
    
    # Get positions
    pos_result = await db.execute(
        select(FundPosition).where(FundPosition.fund_id == fund_id)
    )
    positions = pos_result.scalars().all()
    
    total_value = float(fund.total_value or 0)
    gross_exposure = 0.0
    net_exposure = 0.0
    
    if total_value > 0:
        for pos in positions:
            market_value = float(pos.quantity or 0) * float(pos.current_price or 0)
            weight = market_value / total_value
            gross_exposure += abs(weight)
            net_exposure += weight
    
    return FundDetailResponse(
        fund_id=fund.id,
        name=fund.name,
        strategy=fund.strategy,
        description=fund.description,
        total_value=total_value,
        cash_balance=float(fund.cash_balance or 0),
        gross_exposure=gross_exposure,
        net_exposure=net_exposure,
        n_positions=len(positions),
        is_active=fund.is_active,
        thesis=fund.thesis_json,
        policy=fund.policy_json,
        risk_limits=fund.risk_limits_json,
    )


@app.get("/api/funds/{fund_id}/positions", response_model=list[FundPositionResponse])
async def get_fund_positions(fund_id: str, db: AsyncSession = Depends(get_db)):
    """Get positions for a fund"""
    from db.models import FundModel, FundPosition
    
    # Get fund for total value
    fund_result = await db.execute(
        select(FundModel).where(FundModel.id == fund_id)
    )
    fund = fund_result.scalar_one_or_none()
    if not fund:
        raise HTTPException(status_code=404, detail="Fund not found")
    
    total_value = float(fund.total_value or 1)
    
    result = await db.execute(
        select(FundPosition).where(FundPosition.fund_id == fund_id)
    )
    positions = result.scalars().all()
    
    return [
        FundPositionResponse(
            symbol=pos.symbol,
            quantity=float(pos.quantity or 0),
            avg_entry_price=float(pos.avg_entry_price or 0),
            current_price=float(pos.current_price or 0),
            market_value=float(pos.quantity or 0) * float(pos.current_price or 0),
            unrealized_pnl=float(pos.unrealized_pnl or 0),
            weight_pct=(
                (float(pos.quantity or 0) * float(pos.current_price or 0))
                / total_value * 100
            ),
        )
        for pos in positions
    ]


@app.get("/api/funds/{fund_id}/decisions", response_model=list[DecisionResponse])
async def get_fund_decisions(
    fund_id: str,
    limit: int = Query(50, ge=1, le=500),
    db: AsyncSession = Depends(get_db)
):
    """Get recent decisions for a fund"""
    from db.models import DecisionRecordModel
    
    result = await db.execute(
        select(DecisionRecordModel)
        .where(DecisionRecordModel.fund_id == fund_id)
        .order_by(desc(DecisionRecordModel.asof_timestamp))
        .limit(limit)
    )
    decisions = result.scalars().all()
    
    return [
        DecisionResponse(
            decision_id=d.id,
            fund_id=d.fund_id,
            asof_timestamp=d.asof_timestamp.isoformat(),
            decision_type=d.decision_type,
            status=d.status,
            no_trade_reason=d.no_trade_reason,
            universe_hash=d.universe_hash,
            inputs_hash=d.inputs_hash,
            predicted_directions=d.predicted_directions_json,
            expected_return=d.expected_return,
        )
        for d in decisions
    ]


@app.get("/api/decisions/{decision_id}")
async def get_decision_detail(decision_id: str, db: AsyncSession = Depends(get_db)):
    """Get full decision record with intent and risk result"""
    from db.models import DecisionRecordModel
    
    result = await db.execute(
        select(DecisionRecordModel).where(DecisionRecordModel.id == decision_id)
    )
    decision = result.scalar_one_or_none()
    if not decision:
        raise HTTPException(status_code=404, detail="Decision not found")
    
    return {
        "decision_id": decision.id,
        "fund_id": decision.fund_id,
        "snapshot_id": decision.snapshot_id,
        "asof_timestamp": decision.asof_timestamp.isoformat(),
        "idempotency_key": decision.idempotency_key,
        "run_context": decision.run_context,
        "decision_type": decision.decision_type,
        "no_trade_reason": decision.no_trade_reason,
        "status": decision.status,
        "status_history": decision.status_history_json,
        "intent": decision.intent_json,
        "risk_result": decision.risk_result_json,
        "snapshot_quality": decision.snapshot_quality_json,
        "universe_result": decision.universe_result_json,
        "universe_hash": decision.universe_hash,
        "inputs_hash": decision.inputs_hash,
        "model_versions": decision.model_versions_json,
        "prompt_hashes": decision.prompt_hashes_json,
        "predicted_directions": decision.predicted_directions_json,
        "expected_return": decision.expected_return,
        "expected_holding_days": decision.expected_holding_days,
    }


@app.get("/api/decisions/{decision_id}/debate", response_model=DebateResponse)
async def get_decision_debate(decision_id: str, db: AsyncSession = Depends(get_db)):
    """Get debate transcript for a decision (drill-down)"""
    from db.models import DebateTranscriptModel
    
    result = await db.execute(
        select(DebateTranscriptModel)
        .where(DebateTranscriptModel.decision_id == decision_id)
    )
    transcript = result.scalar_one_or_none()
    if not transcript:
        raise HTTPException(status_code=404, detail="Debate transcript not found")
    
    return DebateResponse(
        transcript_id=transcript.id,
        fund_id=transcript.fund_id,
        started_at=transcript.started_at.isoformat(),
        completed_at=(
            transcript.completed_at.isoformat() if transcript.completed_at else None
        ),
        num_proposals=transcript.num_proposals,
        num_critiques=transcript.num_critiques,
        final_consensus_level=transcript.final_consensus_level or 0.0,
    )


@app.get("/api/debates/{transcript_id}")
async def get_debate_detail(transcript_id: str, db: AsyncSession = Depends(get_db)):
    """Get full debate transcript with all messages"""
    from db.models import DebateTranscriptModel
    
    result = await db.execute(
        select(DebateTranscriptModel).where(DebateTranscriptModel.id == transcript_id)
    )
    transcript = result.scalar_one_or_none()
    if not transcript:
        raise HTTPException(status_code=404, detail="Debate transcript not found")
    
    return {
        "transcript_id": transcript.id,
        "fund_id": transcript.fund_id,
        "snapshot_id": transcript.snapshot_id,
        "started_at": transcript.started_at.isoformat(),
        "completed_at": (
            transcript.completed_at.isoformat() if transcript.completed_at else None
        ),
        "messages": transcript.messages_json or [],
        "num_proposals": transcript.num_proposals,
        "num_critiques": transcript.num_critiques,
        "final_consensus_level": transcript.final_consensus_level,
        "total_input_tokens": transcript.total_input_tokens,
        "total_output_tokens": transcript.total_output_tokens,
    }


@app.get("/api/funds/leaderboard", response_model=list[FundLeaderboardEntry])
async def get_funds_leaderboard(db: AsyncSession = Depends(get_db)):
    """Get fund leaderboard sorted by total value"""
    from db.models import FundModel
    
    result = await db.execute(
        select(FundModel).order_by(desc(FundModel.total_value))
    )
    funds = result.scalars().all()
    
    entries = []
    for i, fund in enumerate(funds):
        entries.append(FundLeaderboardEntry(
            rank=i + 1,
            fund_id=fund.id,
            name=fund.name,
            strategy=fund.strategy,
            total_value=float(fund.total_value or 0),
            gross_exposure=0.0,  # Would need to compute from positions
            is_active=fund.is_active,
        ))
    
    return entries


@app.post("/api/funds/trading/cycle")
async def trigger_fund_trading_cycle():
    """Manually trigger a trading cycle for all funds"""
    # Placeholder - would instantiate FundTradingEngine and run
    return {
        "success": True,
        "message": "Fund trading cycle not yet implemented",
        "note": "This will run debates for all active funds"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
