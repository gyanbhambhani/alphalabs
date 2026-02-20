"""Strategy signals and leaderboard API endpoints"""
from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, desc

from db import get_db, Manager, Portfolio, DailySnapshot, Trade
from app.schemas import (
    StrategySignals, MomentumSignal, MeanReversionSignal,
    SemanticSearchResult, TechnicalIndicators, LeaderboardEntry
)

router = APIRouter(prefix="/api", tags=["signals"])


@router.get("/signals", response_model=StrategySignals)
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


@router.get("/leaderboard", response_model=list[LeaderboardEntry])
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


@router.post("/trading/cycle")
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
