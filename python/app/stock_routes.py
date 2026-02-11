"""Stock API endpoints"""
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from db import get_db, Stock
from app.schemas import StockResponse, StocksListResponse

router = APIRouter(prefix="/api/stocks", tags=["stocks"])


@router.get("", response_model=StocksListResponse)
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


@router.get("/{symbol}", response_model=StockResponse)
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
