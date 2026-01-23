"""
Momentum Strategy Signals

12-month momentum with 1-month skip (standard momentum factor).
Signals range from -1 to +1.
"""
import pandas as pd
import numpy as np
from typing import Dict, List
from dataclasses import dataclass


@dataclass
class MomentumSignal:
    symbol: str
    score: float  # -1 to +1
    raw_return: float
    lookback_days: int
    skip_days: int


def calculate_momentum_signal(
    prices: pd.Series,
    lookback: int = 252,  # ~12 months
    skip: int = 21,  # ~1 month
) -> float:
    """
    Calculate momentum signal for a single security.
    
    Args:
        prices: Price series (most recent last)
        lookback: Number of days to look back
        skip: Number of recent days to skip (avoid short-term reversal)
    
    Returns:
        Momentum score from -1 to +1
    """
    if len(prices) < lookback:
        return 0.0
    
    # Calculate return from lookback to skip
    start_price = prices.iloc[-lookback]
    end_price = prices.iloc[-skip] if skip > 0 else prices.iloc[-1]
    
    if start_price <= 0:
        return 0.0
    
    raw_return = (end_price / start_price) - 1
    
    # Normalize to -1 to +1 range
    # Assuming typical annual return range of -50% to +100%
    normalized = np.clip(raw_return / 0.5, -1, 1)
    
    return float(normalized)


def calculate_momentum_signals(
    price_data: Dict[str, pd.Series],
    lookback: int = 252,
    skip: int = 21,
) -> List[MomentumSignal]:
    """
    Calculate momentum signals for multiple securities.
    
    Args:
        price_data: Dictionary of symbol -> price series
        lookback: Number of days to look back
        skip: Number of recent days to skip
    
    Returns:
        List of MomentumSignal sorted by score (descending)
    """
    signals = []
    
    for symbol, prices in price_data.items():
        if len(prices) < lookback:
            continue
        
        start_price = prices.iloc[-lookback]
        end_price = prices.iloc[-skip] if skip > 0 else prices.iloc[-1]
        
        if start_price <= 0:
            continue
        
        raw_return = (end_price / start_price) - 1
        score = np.clip(raw_return / 0.5, -1, 1)
        
        signals.append(MomentumSignal(
            symbol=symbol,
            score=float(score),
            raw_return=float(raw_return),
            lookback_days=lookback,
            skip_days=skip
        ))
    
    # Sort by score descending
    signals.sort(key=lambda x: x.score, reverse=True)
    
    return signals


def get_momentum_quintiles(
    signals: List[MomentumSignal]
) -> Dict[str, List[str]]:
    """
    Divide signals into quintiles for long-short strategies.
    
    Returns:
        Dict with 'long' (top quintile) and 'short' (bottom quintile) lists
    """
    if len(signals) < 5:
        return {"long": [], "short": []}
    
    n = len(signals)
    quintile_size = n // 5
    
    return {
        "long": [s.symbol for s in signals[:quintile_size]],
        "short": [s.symbol for s in signals[-quintile_size:]],
        "q1": [s.symbol for s in signals[:quintile_size]],
        "q2": [s.symbol for s in signals[quintile_size:2*quintile_size]],
        "q3": [s.symbol for s in signals[2*quintile_size:3*quintile_size]],
        "q4": [s.symbol for s in signals[3*quintile_size:4*quintile_size]],
        "q5": [s.symbol for s in signals[4*quintile_size:]],
    }
