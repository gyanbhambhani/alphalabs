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


# Strategy signals endpoint (mock for now)
@app.get("/api/signals", response_model=StrategySignals)
async def get_signals():
    """Get current strategy signals from the toolbox"""
    # Mock signals - will be replaced with actual signal generation
    return StrategySignals(
        momentum=[
            MomentumSignal(symbol="NVDA", score=0.85),
            MomentumSignal(symbol="MSFT", score=0.62),
            MomentumSignal(symbol="AAPL", score=0.45),
            MomentumSignal(symbol="TSLA", score=-0.15),
        ],
        meanReversion=[
            MeanReversionSignal(symbol="TSLA", score=0.72),
            MeanReversionSignal(symbol="AMD", score=0.58),
            MeanReversionSignal(symbol="NVDA", score=-0.45),
        ],
        technical=[
            TechnicalIndicators(
                symbol="NVDA",
                rsi=68,
                macd={"macd": 2.5, "signal": 2.1, "histogram": 0.4},
                sma20=138,
                sma50=132,
                sma200=115,
                atr=4.2
            )
        ],
        mlPrediction=[
            MLPrediction(symbol="NVDA", predictedReturn=0.023, confidence=0.72),
            MLPrediction(symbol="MSFT", predictedReturn=0.015, confidence=0.68),
        ],
        volatilityRegime="low_vol_trending_up",
        semanticSearch=SemanticSearchResult(
            similarPeriods=[
                SimilarPeriod(
                    date="2023-11-15", 
                    similarity=0.92, 
                    return5d=0.032, 
                    return20d=0.078
                ),
                SimilarPeriod(
                    date="2021-03-22", 
                    similarity=0.88, 
                    return5d=0.025, 
                    return20d=0.065
                ),
            ],
            avg5dReturn=0.0182,
            avg20dReturn=0.0556,
            positive5dRate=0.72,
            interpretation=(
                "Current market conditions resemble low-volatility tech rallies. "
                "Historically, similar periods led to continued gains."
            )
        ),
        timestamp=datetime.utcnow()
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


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
