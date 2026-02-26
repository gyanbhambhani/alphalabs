"""
Data-First Candidate Screening for Trading Strategies

This module implements robust stock screening with strategy-specific scoring.
Screens the stock universe before AI debate to:
1. Filter out low-quality/risky candidates (penny stocks, extreme volatility, gaps)
2. Score remaining stocks based on strategy-specific criteria
3. Return top K candidates for AI agents to evaluate

Strategies supported:
- Momentum: Ranks by 63d + 21d returns with volatility penalty
- Mean Reversion: Finds oversold stocks with positive longer-term trends
- Volatility: Identifies vol spikes (5d vs 21d ratio)
"""

import logging
from typing import Optional, List, Dict, Any
from dataclasses import dataclass

from core.data.snapshot import GlobalMarketSnapshot
from core.backtest.portfolio_tracker import BacktestPortfolio

logger = logging.getLogger(__name__)


def get_feature_value(
    snapshot: GlobalMarketSnapshot,
    symbol: str,
    feature: str
) -> Optional[float]:
    """
    Canonical feature lookup - one function, no fragile if-elses.
    
    Args:
        snapshot: Market data snapshot
        symbol: Stock symbol
        feature: Feature name (e.g., "return_21d", "volatility_5d", "price")
    
    Returns:
        Feature value or None if not available
    
    Examples:
        >>> get_feature_value(snapshot, "AAPL", "price")
        150.25
        >>> get_feature_value(snapshot, "AAPL", "return_21d")
        0.0342
        >>> get_feature_value(snapshot, "AAPL", "volatility_5d")
        0.18
    """
    if feature == "price":
        return snapshot.get_price(symbol)
    
    if feature.startswith("return_"):
        # return_21d -> period "21d"
        period = feature[7:]  # Strip "return_"
        return snapshot.get_return(symbol, period)
    
    if feature.startswith("volatility_"):
        # volatility_21d -> period "21d"
        period = feature[11:]  # Strip "volatility_"
        return snapshot.get_volatility(symbol, period)
    
    return None


@dataclass
class DecisionAuditTrail:
    """
    Full audit trail for a trading decision.
    
    Contains:
    - signals_used: Actual feature values extracted
    - evidence_used: Full citation details with agent attribution
    - validation_report: Pass/fail with detailed errors
    """
    signals_used: Dict[str, float]  # {symbol_feature: value}
    evidence_used: List[Dict[str, Any]]  # [{symbol, feature, agent, value, ...}]
    validation_report: Dict[str, Any]  # {passed, errors, missing_required, ...}


@dataclass
class ScreenedCandidate:
    """
    A pre-screened candidate stock with its score.
    
    Attributes:
        symbol: Stock ticker
        score: Strategy-specific score (higher = better)
        signals: Dictionary of signals used in scoring
        thesis_fit: Brief explanation of why this candidate fits the strategy
        reasoning: Detailed scoring breakdown
    """
    symbol: str
    score: float
    signals: Dict[str, float]
    thesis_fit: str
    reasoning: str


def screen_universe_for_strategy(
    snapshot: GlobalMarketSnapshot,
    strategy: str,
    portfolio: BacktestPortfolio,
    top_k: int = 5,
) -> List[ScreenedCandidate]:
    """
    Screen the entire universe and return top candidates for a strategy.
    
    This is DATA-FIRST: we look at actual numbers, not LLM guesses.
    
    ROBUSTNESS FILTERS applied:
    1. Price floor ($5) - no penny stocks
    2. Volatility cap (300% annualized) - skip broken data
    3. Gap filter (25% daily move) - skip extreme overnight moves
    4. Liquidity proxy - require sufficient history (vol_21d exists)
    
    Args:
        snapshot: Market data with all symbols
        strategy: Fund strategy (momentum, mean_reversion, volatility)
        portfolio: Current portfolio (to exclude existing positions)
        top_k: Number of candidates to return
        
    Returns:
        List of ScreenedCandidate sorted by score (best first)
    """
    candidates: List[ScreenedCandidate] = []
    
    # Get all symbols with data
    symbols = list(snapshot.prices.keys())
    
    # =========================================================================
    # ROBUSTNESS FILTERS - Pre-filter to avoid garbage trades
    # =========================================================================
    filtered_symbols = []
    for symbol in symbols:
        price = snapshot.get_price(symbol)
        if price is None:
            continue
        
        # 1. Price floor - no penny stocks (high spreads, manipulation risk)
        if price < 5.0:
            continue
        
        # 2. Liquidity proxy - require sufficient history
        vol_21d = snapshot.get_volatility(symbol, "21d")
        if vol_21d is None:
            continue  # Not enough data to trade safely
        
        # 3. Extreme volatility filter - skip broken data or distressed stocks
        # 300% annualized vol = ~19% daily, almost certainly data issue or delisting
        if vol_21d > 3.0:
            continue
        
        # 4. Gap filter - skip extreme overnight moves (likely news/earnings gap)
        ret_1d = snapshot.get_return(symbol, "1d")
        if ret_1d is not None and abs(ret_1d) > 0.25:  # 25% daily move
            continue
        
        filtered_symbols.append(symbol)
    
    # Use filtered symbols for screening
    symbols = filtered_symbols
    
    # Determine which signals to use based on strategy
    strategy_lower = strategy.lower()
    
    if "momentum" in strategy_lower or "trend" in strategy_lower:
        # Momentum: look for strong positive returns over 21d and 63d
        thesis_fit = "momentum"
        for symbol in symbols:
            ret_21d = snapshot.get_return(symbol, "21d")
            ret_63d = snapshot.get_return(symbol, "63d")
            ret_5d = snapshot.get_return(symbol, "5d")
            vol_21d = snapshot.get_volatility(symbol, "21d")
            price = snapshot.get_price(symbol)
            
            if ret_21d is None or ret_63d is None or price is None:
                continue
            
            # Skip if already held
            if symbol in portfolio.positions:
                continue
            
            # Momentum score: positive returns, skip recent month effect
            # Higher 63d return + positive 21d = strong momentum
            score = (ret_63d * 0.6) + (ret_21d * 0.4)
            
            # Penalize high volatility slightly
            if vol_21d and vol_21d > 0.4:
                score *= 0.8
            
            if score > 0.02:  # Only consider positive momentum
                candidates.append(ScreenedCandidate(
                    symbol=symbol,
                    score=score,
                    signals={
                        "return_21d": ret_21d,
                        "return_63d": ret_63d,
                        "return_5d": ret_5d or 0,
                        "volatility_21d": vol_21d or 0,
                        "price": price,
                    },
                    thesis_fit=thesis_fit,
                    reasoning=(
                        f"Strong momentum: 63d={ret_63d:+.1%}, 21d={ret_21d:+.1%}"
                    ),
                ))
    
    elif "mean_reversion" in strategy_lower or "oversold" in strategy_lower:
        # Mean reversion: look for oversold stocks (negative short-term returns)
        thesis_fit = "mean_reversion"
        for symbol in symbols:
            ret_1d = snapshot.get_return(symbol, "1d")
            ret_5d = snapshot.get_return(symbol, "5d")
            ret_21d = snapshot.get_return(symbol, "21d")
            vol_21d = snapshot.get_volatility(symbol, "21d")
            price = snapshot.get_price(symbol)
            
            if ret_1d is None or ret_5d is None or price is None:
                continue
            
            # Skip if already held
            if symbol in portfolio.positions:
                continue
            
            # Mean reversion score: negative short-term returns = oversold
            # More negative = higher score (potential bounce)
            score = -(ret_1d * 0.4 + ret_5d * 0.6)
            
            # Only consider if actually oversold
            if ret_5d < -0.02 or ret_1d < -0.01:
                # Bonus if longer-term is positive (healthy stock, temporary dip)
                if ret_21d and ret_21d > 0:
                    score *= 1.2
                
                candidates.append(ScreenedCandidate(
                    symbol=symbol,
                    score=score,
                    signals={
                        "return_1d": ret_1d,
                        "return_5d": ret_5d,
                        "return_21d": ret_21d or 0,
                        "volatility_21d": vol_21d or 0,
                        "price": price,
                    },
                    thesis_fit=thesis_fit,
                    reasoning=(
                        f"Oversold bounce: 1d={ret_1d:+.1%}, 5d={ret_5d:+.1%}"
                    ),
                ))
    
    elif "volatility" in strategy_lower:
        # Volatility: look for vol spikes or regime changes
        thesis_fit = "volatility"
        for symbol in symbols:
            vol_5d = snapshot.get_volatility(symbol, "5d")
            vol_21d = snapshot.get_volatility(symbol, "21d")
            ret_5d = snapshot.get_return(symbol, "5d")
            price = snapshot.get_price(symbol)
            
            if vol_5d is None or vol_21d is None or price is None:
                continue
            
            # Skip if already held
            if symbol in portfolio.positions:
                continue
            
            # Vol spike score: short-term vol much higher than longer-term
            vol_ratio = vol_5d / vol_21d if vol_21d > 0 else 1.0
            score = vol_ratio - 1.0  # How much vol has spiked
            
            if vol_ratio > 1.3:  # Significant vol spike
                candidates.append(ScreenedCandidate(
                    symbol=symbol,
                    score=score,
                    signals={
                        "volatility_5d": vol_5d,
                        "volatility_21d": vol_21d,
                        "return_5d": ret_5d or 0,
                        "price": price,
                    },
                    thesis_fit=thesis_fit,
                    reasoning=(
                        f"Vol spike: 5d={vol_5d:.1%} vs 21d={vol_21d:.1%} "
                        f"(ratio={vol_ratio:.1f}x)"
                    ),
                ))
    
    else:
        # Unknown strategy - return empty
        signals_log(
            f"Unknown strategy '{strategy}' - no screening available",
            level=logging.WARNING
        )
        return []
    
    # Sort by score (highest first) and return top_k
    candidates.sort(key=lambda c: c.score, reverse=True)
    
    signals_log(
        f"Screened {len(symbols)} stocks, found {len(candidates)} candidates "
        f"for {thesis_fit}"
    )
    if candidates:
        top_3 = candidates[:3]
        signals_log(
            f"Top 3: {', '.join(f'{c.symbol}({c.score:.2f})' for c in top_3)}"
        )
    
    return candidates[:top_k]


