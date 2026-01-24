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
    MLPrediction
)
from db import get_db, Manager, Portfolio, Position, Trade, DailySnapshot

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


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
