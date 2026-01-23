"""
Volatility Regime Detection

Classifies current market conditions into volatility regimes
for strategy selection.
"""
import pandas as pd
import numpy as np
from typing import Dict, Tuple
from dataclasses import dataclass
from enum import Enum


class VolatilityLevel(Enum):
    LOW = "low_vol"
    NORMAL = "normal_vol"
    HIGH = "high_vol"


class TrendDirection(Enum):
    UP = "trending_up"
    DOWN = "trending_down"
    RANGING = "ranging"


@dataclass
class VolatilityRegime:
    level: VolatilityLevel
    trend: TrendDirection
    realized_vol: float  # Annualized
    vix: float | None
    regime_name: str
    
    @property
    def full_name(self) -> str:
        return f"{self.level.value}_{self.trend.value}"


def calculate_realized_volatility(
    prices: pd.Series,
    window: int = 20,
    annualization_factor: int = 252
) -> float:
    """
    Calculate realized (historical) volatility.
    
    Args:
        prices: Price series
        window: Rolling window for calculation
        annualization_factor: Trading days per year
    
    Returns:
        Annualized volatility as a decimal (e.g., 0.20 for 20%)
    """
    if len(prices) < window + 1:
        return 0.0
    
    # Calculate log returns
    returns = np.log(prices / prices.shift(1))
    
    # Rolling standard deviation
    rolling_std = returns.rolling(window=window).std()
    
    # Annualize
    current_vol = rolling_std.iloc[-1]
    if pd.isna(current_vol):
        return 0.0
    
    annualized_vol = current_vol * np.sqrt(annualization_factor)
    
    return float(annualized_vol)


def detect_trend(
    prices: pd.Series,
    short_window: int = 50,
    long_window: int = 200
) -> TrendDirection:
    """
    Detect trend direction using moving average crossover.
    
    Args:
        prices: Price series
        short_window: Short-term MA period
        long_window: Long-term MA period
    
    Returns:
        TrendDirection enum
    """
    if len(prices) < long_window:
        return TrendDirection.RANGING
    
    current_price = prices.iloc[-1]
    sma_short = prices.iloc[-short_window:].mean()
    sma_long = prices.iloc[-long_window:].mean()
    
    if current_price > sma_short > sma_long:
        return TrendDirection.UP
    elif current_price < sma_short < sma_long:
        return TrendDirection.DOWN
    else:
        return TrendDirection.RANGING


def classify_volatility_level(
    realized_vol: float,
    low_threshold: float = 0.15,
    high_threshold: float = 0.25
) -> VolatilityLevel:
    """
    Classify volatility into low/normal/high.
    
    Args:
        realized_vol: Annualized realized volatility
        low_threshold: Below this = low vol
        high_threshold: Above this = high vol
    
    Returns:
        VolatilityLevel enum
    """
    if realized_vol < low_threshold:
        return VolatilityLevel.LOW
    elif realized_vol > high_threshold:
        return VolatilityLevel.HIGH
    else:
        return VolatilityLevel.NORMAL


def detect_volatility_regime(
    prices: pd.Series,
    vix: float | None = None,
    vol_window: int = 20,
    trend_short_window: int = 50,
    trend_long_window: int = 200,
) -> VolatilityRegime:
    """
    Detect current volatility regime.
    
    Args:
        prices: Price series (e.g., SPY or index)
        vix: VIX value if available
        vol_window: Window for volatility calculation
        trend_short_window: Short-term trend MA
        trend_long_window: Long-term trend MA
    
    Returns:
        VolatilityRegime with full classification
    """
    realized_vol = calculate_realized_volatility(prices, vol_window)
    
    # Use VIX if available and reasonable
    if vix is not None and vix > 0:
        effective_vol = vix / 100  # VIX is in percentage points
    else:
        effective_vol = realized_vol
    
    vol_level = classify_volatility_level(effective_vol)
    trend = detect_trend(prices, trend_short_window, trend_long_window)
    
    regime_name = f"{vol_level.value}_{trend.value}"
    
    return VolatilityRegime(
        level=vol_level,
        trend=trend,
        realized_vol=realized_vol,
        vix=vix,
        regime_name=regime_name
    )


def get_regime_strategy_weights(regime: VolatilityRegime) -> Dict[str, float]:
    """
    Get recommended strategy weights based on regime.
    
    Returns weights for different strategies that sum to 1.
    """
    # Default weights
    weights = {
        "momentum": 0.25,
        "mean_reversion": 0.25,
        "ml_prediction": 0.25,
        "semantic_search": 0.25
    }
    
    # Adjust based on regime
    if regime.level == VolatilityLevel.LOW:
        if regime.trend == TrendDirection.UP:
            # Low vol trending up - favor momentum
            weights = {
                "momentum": 0.40,
                "mean_reversion": 0.15,
                "ml_prediction": 0.20,
                "semantic_search": 0.25
            }
        elif regime.trend == TrendDirection.DOWN:
            # Low vol trending down - cautious
            weights = {
                "momentum": 0.30,
                "mean_reversion": 0.20,
                "ml_prediction": 0.20,
                "semantic_search": 0.30
            }
    
    elif regime.level == VolatilityLevel.HIGH:
        # High vol - favor mean reversion and reduce momentum
        weights = {
            "momentum": 0.15,
            "mean_reversion": 0.35,
            "ml_prediction": 0.20,
            "semantic_search": 0.30
        }
    
    elif regime.trend == TrendDirection.RANGING:
        # Ranging market - favor mean reversion
        weights = {
            "momentum": 0.20,
            "mean_reversion": 0.35,
            "ml_prediction": 0.20,
            "semantic_search": 0.25
        }
    
    return weights
