"""
Technical Analysis Indicators

RSI, MACD, Moving Averages, ATR, and other technical indicators.
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass


@dataclass
class MACDResult:
    macd: float
    signal: float
    histogram: float


@dataclass 
class TechnicalIndicators:
    symbol: str
    rsi: float
    macd: MACDResult
    sma_20: float
    sma_50: float
    sma_200: float
    ema_12: float
    ema_26: float
    atr: float
    current_price: float
    volume_sma_20: float


def calculate_sma(prices: pd.Series, window: int) -> float:
    """Calculate Simple Moving Average"""
    if len(prices) < window:
        return prices.mean() if len(prices) > 0 else 0.0
    return float(prices.iloc[-window:].mean())


def calculate_ema(prices: pd.Series, span: int) -> float:
    """Calculate Exponential Moving Average"""
    if len(prices) < span:
        return prices.mean() if len(prices) > 0 else 0.0
    ema = prices.ewm(span=span, adjust=False).mean()
    return float(ema.iloc[-1])


def calculate_rsi(prices: pd.Series, period: int = 14) -> float:
    """
    Calculate Relative Strength Index (RSI).
    
    Returns value between 0 and 100.
    RSI > 70 = overbought, RSI < 30 = oversold
    """
    if len(prices) < period + 1:
        return 50.0
    
    delta = prices.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    
    current_avg_gain = avg_gain.iloc[-1]
    current_avg_loss = avg_loss.iloc[-1]
    
    if current_avg_loss == 0:
        return 100.0 if current_avg_gain > 0 else 50.0
    
    rs = current_avg_gain / current_avg_loss
    rsi = 100 - (100 / (1 + rs))
    
    return float(rsi)


def calculate_macd(
    prices: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9
) -> MACDResult:
    """
    Calculate MACD (Moving Average Convergence Divergence).
    
    Args:
        prices: Price series
        fast: Fast EMA period (default 12)
        slow: Slow EMA period (default 26)
        signal: Signal line period (default 9)
    
    Returns:
        MACDResult with macd, signal, and histogram values
    """
    if len(prices) < slow:
        return MACDResult(macd=0.0, signal=0.0, histogram=0.0)
    
    ema_fast = prices.ewm(span=fast, adjust=False).mean()
    ema_slow = prices.ewm(span=slow, adjust=False).mean()
    
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    
    return MACDResult(
        macd=float(macd_line.iloc[-1]),
        signal=float(signal_line.iloc[-1]),
        histogram=float(histogram.iloc[-1])
    )


def calculate_atr(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14
) -> float:
    """
    Calculate Average True Range (ATR).
    
    Measures volatility as average of true ranges.
    """
    if len(high) < period + 1:
        return 0.0
    
    # True Range components
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = true_range.rolling(window=period).mean()
    
    return float(atr.iloc[-1]) if not pd.isna(atr.iloc[-1]) else 0.0


def calculate_technical_indicators(
    symbol: str,
    close: pd.Series,
    high: Optional[pd.Series] = None,
    low: Optional[pd.Series] = None,
    volume: Optional[pd.Series] = None,
) -> TechnicalIndicators:
    """
    Calculate all technical indicators for a symbol.
    
    Args:
        symbol: Stock symbol
        close: Close prices
        high: High prices (optional, for ATR)
        low: Low prices (optional, for ATR)
        volume: Volume data (optional)
    
    Returns:
        TechnicalIndicators with all calculated values
    """
    # Use close for high/low if not provided
    if high is None:
        high = close
    if low is None:
        low = close
    
    current_price = float(close.iloc[-1]) if len(close) > 0 else 0.0
    
    return TechnicalIndicators(
        symbol=symbol,
        rsi=calculate_rsi(close),
        macd=calculate_macd(close),
        sma_20=calculate_sma(close, 20),
        sma_50=calculate_sma(close, 50),
        sma_200=calculate_sma(close, 200),
        ema_12=calculate_ema(close, 12),
        ema_26=calculate_ema(close, 26),
        atr=calculate_atr(high, low, close),
        current_price=current_price,
        volume_sma_20=calculate_sma(volume, 20) if volume is not None else 0.0
    )


def calculate_all_technical_indicators(
    price_data: Dict[str, Dict[str, pd.Series]]
) -> List[TechnicalIndicators]:
    """
    Calculate technical indicators for all symbols.
    
    Args:
        price_data: Dict of symbol -> {'close': Series, 'high': Series, ...}
    
    Returns:
        List of TechnicalIndicators
    """
    indicators = []
    
    for symbol, data in price_data.items():
        close = data.get('close', pd.Series())
        high = data.get('high')
        low = data.get('low')
        volume = data.get('volume')
        
        if len(close) == 0:
            continue
        
        ind = calculate_technical_indicators(
            symbol=symbol,
            close=close,
            high=high,
            low=low,
            volume=volume
        )
        indicators.append(ind)
    
    return indicators


def is_golden_cross(sma_50: float, sma_200: float, prev_sma_50: float, prev_sma_200: float) -> bool:
    """Check if golden cross (50-day SMA crosses above 200-day SMA)"""
    return prev_sma_50 <= prev_sma_200 and sma_50 > sma_200


def is_death_cross(sma_50: float, sma_200: float, prev_sma_50: float, prev_sma_200: float) -> bool:
    """Check if death cross (50-day SMA crosses below 200-day SMA)"""
    return prev_sma_50 >= prev_sma_200 and sma_50 < sma_200


def get_trend_direction(price: float, sma_50: float, sma_200: float) -> str:
    """
    Determine trend direction based on price and moving averages.
    
    Returns: 'bullish', 'bearish', or 'neutral'
    """
    if price > sma_50 > sma_200:
        return 'bullish'
    elif price < sma_50 < sma_200:
        return 'bearish'
    else:
        return 'neutral'
