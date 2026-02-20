"""Portfolio, Position, Trade, and Performance API endpoints"""
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, desc

from db import get_db, Portfolio, Position, Trade, DailySnapshot
from app.schemas import (
    PortfolioResponse, PositionResponse, TradeResponse, DailySnapshotResponse
)

router = APIRouter(prefix="/api", tags=["portfolios"])


# Portfolio endpoints
@router.get("/portfolios", response_model=list[PortfolioResponse])
async def get_portfolios(db: AsyncSession = Depends(get_db)):
    """Get all portfolios"""
    result = await db.execute(select(Portfolio))
    portfolios = result.scalars().all()
    return portfolios


@router.get("/portfolios/{manager_id}", response_model=PortfolioResponse)
async def get_portfolio(manager_id: str, db: AsyncSession = Depends(get_db)):
    """Get portfolio for a specific manager"""
    result = await db.execute(
        select(Portfolio).where(Portfolio.manager_id == manager_id)
    )
    portfolio = result.scalar_one_or_none()
    if not portfolio:
        raise HTTPException(status_code=404, detail="Portfolio not found")
    return portfolio


# Position endpoints
@router.get("/positions/{manager_id}", response_model=list[PositionResponse])
async def get_positions(manager_id: str, db: AsyncSession = Depends(get_db)):
    """Get all positions for a manager"""
    result = await db.execute(
        select(Position).where(Position.manager_id == manager_id)
    )
    positions = result.scalars().all()
    return positions


# Trade endpoints
@router.get("/trades", response_model=list[TradeResponse])
async def get_trades(
    manager_id: Optional[str] = Query(None),
    limit: int = Query(50, ge=1, le=500),
    db: AsyncSession = Depends(get_db)
):
    """Get recent trades, optionally filtered by manager"""
    query = select(Trade).order_by(desc(Trade.executed_at)).limit(limit)
    if manager_id:
        query = query.where(Trade.manager_id == manager_id)
    result = await db.execute(query)
    trades = result.scalars().all()
    return trades


# Performance endpoints
@router.get(
    "/performance/{manager_id}",
    response_model=list[DailySnapshotResponse]
)
async def get_performance(
    manager_id: str,
    days: int = Query(30, ge=1, le=365),
    db: AsyncSession = Depends(get_db)
):
    """Get performance history for a manager"""
    result = await db.execute(
        select(DailySnapshot)
        .where(DailySnapshot.manager_id == manager_id)
        .order_by(desc(DailySnapshot.date))
        .limit(days)
    )
    snapshots = result.scalars().all()
    return snapshots
