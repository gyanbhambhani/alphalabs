"""
Mean Reversion Strategy Signals

Based on Bollinger Bands and Z-score calculations.
Positive signal = oversold (buy), Negative signal = overbought (sell).
"""
import pandas as pd
import numpy as np
from typing import Dict, List
from dataclasses import dataclass


@dataclass
class MeanReversionSignal:
    symbol: str
    score: float  # -1 to +1 (positive = oversold)
    z_score: float
    current_price: float
    mean_price: float
    upper_band: float
    lower_band: float


def calculate_mean_reversion_signal(
    prices: pd.Series,
    window: int = 20,
    num_std: float = 2.0,
) -> MeanReversionSignal:
    """
    Calculate mean reversion signal using Bollinger Bands.
    
    Args:
        prices: Price series (most recent last)
        window: Rolling window for mean/std calculation
        num_std: Number of standard deviations for bands
    
    Returns:
        MeanReversionSignal with score from -1 to +1
        Positive = oversold (price below lower band)
        Negative = overbought (price above upper band)
    """
    if len(prices) < window:
        return MeanReversionSignal(
            symbol="",
            score=0.0,
            z_score=0.0,
            current_price=prices.iloc[-1] if len(prices) > 0 else 0,
            mean_price=0,
            upper_band=0,
            lower_band=0
        )
    
    # Calculate rolling statistics
    rolling_mean = prices.rolling(window=window).mean()
    rolling_std = prices.rolling(window=window).std()
    
    current_price = prices.iloc[-1]
    mean = rolling_mean.iloc[-1]
    std = rolling_std.iloc[-1]
    
    if std == 0:
        z_score = 0.0
    else:
        z_score = (current_price - mean) / std
    
    upper_band = mean + num_std * std
    lower_band = mean - num_std * std
    
    # Score: negative z-score / num_std, capped at -1 to +1
    # Flip sign so positive = oversold (buy signal)
    score = np.clip(-z_score / num_std, -1, 1)
    
    return MeanReversionSignal(
        symbol="",
        score=float(score),
        z_score=float(z_score),
        current_price=float(current_price),
        mean_price=float(mean),
        upper_band=float(upper_band),
        lower_band=float(lower_band)
    )


def calculate_mean_reversion_signals(
    price_data: Dict[str, pd.Series],
    window: int = 20,
    num_std: float = 2.0,
) -> List[MeanReversionSignal]:
    """
    Calculate mean reversion signals for multiple securities.
    
    Args:
        price_data: Dictionary of symbol -> price series
        window: Rolling window for mean/std calculation
        num_std: Number of standard deviations for bands
    
    Returns:
        List of MeanReversionSignal sorted by score (descending)
    """
    signals = []
    
    for symbol, prices in price_data.items():
        signal = calculate_mean_reversion_signal(prices, window, num_std)
        signal.symbol = symbol
        signals.append(signal)
    
    # Sort by score descending (most oversold first)
    signals.sort(key=lambda x: x.score, reverse=True)
    
    return signals


def calculate_rsi(
    prices: pd.Series,
    period: int = 14
) -> float:
    """
    Calculate Relative Strength Index.
    
    Args:
        prices: Price series
        period: RSI period
    
    Returns:
        RSI value (0-100)
    """
    if len(prices) < period + 1:
        return 50.0
    
    delta = prices.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    
    if avg_loss.iloc[-1] == 0:
        return 100.0
    
    rs = avg_gain.iloc[-1] / avg_loss.iloc[-1]
    rsi = 100 - (100 / (1 + rs))
    
    return float(rsi)


def rsi_to_signal(rsi: float) -> float:
    """
    Convert RSI to mean reversion signal.
    
    RSI < 30 = oversold = positive signal
    RSI > 70 = overbought = negative signal
    """
    if rsi < 30:
        return (30 - rsi) / 30  # 0 to 1
    elif rsi > 70:
        return -(rsi - 70) / 30  # -1 to 0
    else:
        return 0.0
