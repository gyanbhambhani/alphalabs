"""
Real-time Strategy Signals Service

Fetches live market data and computes real signals
from all strategy modules.
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
import yfinance as yf
import asyncio
from concurrent.futures import ThreadPoolExecutor

from core.strategies.momentum import (
    calculate_momentum_signals,
    MomentumSignal as MomentumSignalInternal
)
from core.strategies.mean_reversion import (
    calculate_mean_reversion_signals,
    calculate_rsi,
    MeanReversionSignal as MeanReversionSignalInternal
)
from core.strategies.technical import (
    calculate_technical_indicators,
    TechnicalIndicators as TechnicalIndicatorsInternal
)
from core.strategies.volatility import (
    detect_volatility_regime,
    VolatilityRegime
)


# Default universe for signals (can be customized)
DEFAULT_SIGNAL_UNIVERSE = [
    # Mega-cap tech
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA",
    # Semiconductors
    "AMD", "INTC", "AVGO", "QCOM",
    # Financials
    "JPM", "BAC", "GS", "V", "MA",
    # Healthcare
    "UNH", "JNJ", "PFE", "ABBV",
    # Consumer
    "WMT", "COST", "HD", "MCD",
    # Energy
    "XOM", "CVX",
    # Market ETFs
    "SPY", "QQQ", "IWM",
]


@dataclass
class SignalResult:
    """Aggregated signal results from all strategies"""
    momentum: List[Dict]
    mean_reversion: List[Dict]
    technical: List[Dict]
    volatility_regime: str
    realized_vol: float
    market_trend: str
    timestamp: datetime
    data_freshness: str


def _fetch_price_data(
    symbols: List[str], 
    period: str = "1y"
) -> Dict[str, pd.DataFrame]:
    """
    Fetch price data for all symbols using yfinance.
    
    Returns dict of symbol -> DataFrame with OHLCV data
    """
    data = {}
    
    # Use yfinance download for batch efficiency
    try:
        tickers = yf.Tickers(" ".join(symbols))
        
        for symbol in symbols:
            try:
                ticker = tickers.tickers.get(symbol)
                if ticker is None:
                    continue
                    
                df = ticker.history(period=period)
                if df.empty or len(df) < 50:
                    continue
                
                # Standardize column names
                df.columns = df.columns.str.lower()
                df = df[['open', 'high', 'low', 'close', 'volume']].copy()
                df = df.dropna()
                
                if len(df) >= 50:
                    data[symbol] = df
                    
            except Exception as e:
                print(f"Error fetching {symbol}: {e}")
                continue
                
    except Exception as e:
        print(f"Batch fetch error: {e}")
        # Fallback to individual fetches
        for symbol in symbols:
            try:
                ticker = yf.Ticker(symbol)
                df = ticker.history(period=period)
                if not df.empty and len(df) >= 50:
                    df.columns = df.columns.str.lower()
                    df = df[['open', 'high', 'low', 'close', 'volume']].copy()
                    df = df.dropna()
                    if len(df) >= 50:
                        data[symbol] = df
            except:
                continue
    
    return data


def compute_all_signals(
    symbols: Optional[List[str]] = None,
    period: str = "2y"
) -> SignalResult:
    """
    Compute all strategy signals using real market data.
    
    Args:
        symbols: List of symbols to analyze (uses default if None)
        period: Historical period for data fetch (default 2y for momentum)
        
    Returns:
        SignalResult with all computed signals
    """
    if symbols is None:
        symbols = DEFAULT_SIGNAL_UNIVERSE
    
    # Fetch market data - need 2y for proper momentum calculation
    price_data = _fetch_price_data(symbols, period)
    
    if not price_data:
        raise ValueError("No price data available - check network connection")
    
    # Get close prices for momentum/mean reversion
    close_prices = {sym: df['close'] for sym, df in price_data.items()}
    
    # 1. Calculate Momentum Signals (use 250 days = ~12 months trading days)
    momentum_signals = []
    try:
        mom_results = calculate_momentum_signals(close_prices, lookback=250, skip=21)
        for sig in mom_results[:15]:  # Top 15
            momentum_signals.append({
                "symbol": sig.symbol,
                "score": round(sig.score, 3),
                "rawReturn": round(sig.raw_return, 4)
            })
    except Exception as e:
        print(f"Momentum calculation error: {e}")
    
    # 2. Calculate Mean Reversion Signals
    mean_reversion_signals = []
    try:
        mr_results = calculate_mean_reversion_signals(close_prices, window=20, num_std=2.0)
        for sig in mr_results[:15]:  # Top 15
            mean_reversion_signals.append({
                "symbol": sig.symbol,
                "score": round(sig.score, 3),
                "zScore": round(sig.z_score, 2),
                "currentPrice": round(sig.current_price, 2),
                "meanPrice": round(sig.mean_price, 2)
            })
    except Exception as e:
        print(f"Mean reversion calculation error: {e}")
    
    # 3. Calculate Technical Indicators
    technical_indicators = []
    try:
        for symbol, df in list(price_data.items())[:10]:  # Top 10
            tech = calculate_technical_indicators(
                symbol=symbol,
                close=df['close'],
                high=df['high'],
                low=df['low'],
                volume=df['volume']
            )
            technical_indicators.append({
                "symbol": tech.symbol,
                "rsi": round(tech.rsi, 1),
                "macd": {
                    "macd": round(tech.macd.macd, 2),
                    "signal": round(tech.macd.signal, 2),
                    "histogram": round(tech.macd.histogram, 2)
                },
                "sma20": round(tech.sma_20, 2),
                "sma50": round(tech.sma_50, 2),
                "sma200": round(tech.sma_200, 2),
                "atr": round(tech.atr, 2),
                "currentPrice": round(tech.current_price, 2)
            })
    except Exception as e:
        print(f"Technical indicator error: {e}")
    
    # 4. Detect Volatility Regime (use SPY as market proxy)
    volatility_regime = "normal_vol_ranging"
    realized_vol = 0.0
    market_trend = "ranging"
    
    try:
        if "SPY" in price_data:
            spy_prices = price_data["SPY"]['close']
            regime = detect_volatility_regime(spy_prices)
            volatility_regime = regime.full_name
            realized_vol = round(regime.realized_vol, 4)
            market_trend = regime.trend.value
    except Exception as e:
        print(f"Volatility regime error: {e}")
    
    # Determine data freshness
    latest_date = max(
        df.index[-1] for df in price_data.values()
    ) if price_data else datetime.now()
    
    days_old = (datetime.now() - latest_date.to_pydatetime().replace(tzinfo=None)).days
    if days_old <= 1:
        freshness = "live"
    elif days_old <= 3:
        freshness = "recent"
    else:
        freshness = f"{days_old} days old"
    
    return SignalResult(
        momentum=momentum_signals,
        mean_reversion=mean_reversion_signals,
        technical=technical_indicators,
        volatility_regime=volatility_regime,
        realized_vol=realized_vol,
        market_trend=market_trend,
        timestamp=datetime.utcnow(),
        data_freshness=freshness
    )


async def compute_signals_async(
    symbols: Optional[List[str]] = None,
    period: str = "1y"
) -> SignalResult:
    """
    Async wrapper for signal computation.
    Runs the blocking yfinance calls in a thread pool.
    """
    loop = asyncio.get_event_loop()
    with ThreadPoolExecutor() as executor:
        result = await loop.run_in_executor(
            executor,
            compute_all_signals,
            symbols,
            period
        )
    return result


# Simple cache to avoid hammering yfinance
_signal_cache: Optional[Tuple[datetime, SignalResult]] = None
_cache_ttl = timedelta(minutes=5)


async def get_cached_signals(
    symbols: Optional[List[str]] = None,
    period: str = "1y",
    force_refresh: bool = False
) -> SignalResult:
    """
    Get signals with caching to avoid rate limits.
    
    Cache TTL is 5 minutes for real-time feel without abuse.
    """
    global _signal_cache
    
    now = datetime.utcnow()
    
    if not force_refresh and _signal_cache is not None:
        cache_time, cached_result = _signal_cache
        if now - cache_time < _cache_ttl:
            return cached_result
    
    # Compute fresh signals
    result = await compute_signals_async(symbols, period)
    _signal_cache = (now, result)
    
    return result
