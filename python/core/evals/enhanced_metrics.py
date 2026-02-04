"""
Enhanced Evaluation Metrics

Adds production-grade metrics on top of existing trade-level metrics:
- Alpha vs SPY (benchmark-relative)
- Information ratio
- Turnover cost drag
- Regime-split performance
- PnL attribution
"""

from dataclasses import dataclass
from datetime import date
from typing import Dict, List, Optional
import numpy as np
import logging

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkMetrics:
    """Benchmark-relative performance metrics."""
    strategy_return: float
    benchmark_return: float
    alpha: float  # Excess return
    beta: float  # Market sensitivity
    tracking_error: float  # Std of excess returns
    information_ratio: float  # Alpha / tracking error
    correlation: float  # Correlation with benchmark


@dataclass
class CostMetrics:
    """Transaction cost metrics."""
    gross_return: float
    total_commissions: float
    total_slippage_bps: float
    net_return: float  # After costs
    turnover_annualized: float
    cost_drag_bps: float  # Total cost as bps of AUM


@dataclass
class RegimeMetrics:
    """Performance by market regime."""
    regime: str  # "bull_low_vol", "bear_high_vol", etc.
    n_decisions: int
    avg_return: float
    sharpe_ratio: float
    win_rate: float
    max_drawdown: float


@dataclass
class AttributionMetrics:
    """PnL attribution by fund/sector."""
    pnl_by_fund: Dict[str, float]
    pnl_by_sector: Dict[str, float]
    return_by_fund: Dict[str, float]
    best_fund: str
    worst_fund: str


@dataclass
class EnhancedBacktestMetrics:
    """
    Complete backtest evaluation metrics.
    
    Combines trade-level metrics with benchmark-relative,
    cost-adjusted, and regime-aware analysis.
    """
    run_id: str
    start_date: date
    end_date: date
    
    # Benchmark-relative
    benchmark_metrics: BenchmarkMetrics
    
    # Cost accounting
    cost_metrics: CostMetrics
    
    # Regime analysis
    performance_by_regime: List[RegimeMetrics]
    
    # Attribution
    attribution: AttributionMetrics
    
    # Quality metrics
    best_month: float
    worst_month: float
    max_consecutive_losses: int
    recovery_time_days: int  # Days to recover from max drawdown


def compute_alpha(
    strategy_returns: List[float],
    benchmark_returns: List[float],
) -> float:
    """
    Compute annualized alpha (excess return vs benchmark).
    
    Args:
        strategy_returns: Daily strategy returns
        benchmark_returns: Daily benchmark returns
        
    Returns:
        Annualized alpha
    """
    if not strategy_returns or not benchmark_returns:
        return 0.0
    
    # Compute cumulative returns
    strategy_cum = np.prod([1 + r for r in strategy_returns]) - 1
    benchmark_cum = np.prod([1 + r for r in benchmark_returns]) - 1
    
    # Annualize
    n_days = len(strategy_returns)
    strategy_annual = (1 + strategy_cum) ** (252 / n_days) - 1
    benchmark_annual = (1 + benchmark_cum) ** (252 / n_days) - 1
    
    return strategy_annual - benchmark_annual


def compute_beta(
    strategy_returns: List[float],
    benchmark_returns: List[float],
) -> float:
    """
    Compute beta (market sensitivity).
    
    Beta = Cov(strategy, benchmark) / Var(benchmark)
    
    Args:
        strategy_returns: Daily strategy returns
        benchmark_returns: Daily benchmark returns
        
    Returns:
        Beta coefficient
    """
    if len(strategy_returns) < 2 or len(benchmark_returns) < 2:
        return 0.0
    
    # Ensure same length
    min_len = min(len(strategy_returns), len(benchmark_returns))
    strat = np.array(strategy_returns[:min_len])
    bench = np.array(benchmark_returns[:min_len])
    
    # Compute beta
    cov = np.cov(strat, bench)[0, 1]
    var_bench = np.var(bench)
    
    if var_bench > 0:
        return cov / var_bench
    
    return 0.0


def compute_tracking_error(
    strategy_returns: List[float],
    benchmark_returns: List[float],
) -> float:
    """
    Compute tracking error (annualized std of excess returns).
    
    Args:
        strategy_returns: Daily strategy returns
        benchmark_returns: Daily benchmark returns
        
    Returns:
        Annualized tracking error
    """
    if not strategy_returns or not benchmark_returns:
        return 0.0
    
    # Compute excess returns
    min_len = min(len(strategy_returns), len(benchmark_returns))
    excess = [
        strategy_returns[i] - benchmark_returns[i]
        for i in range(min_len)
    ]
    
    # Annualize std
    return float(np.std(excess) * np.sqrt(252))


def compute_information_ratio(
    alpha: float,
    tracking_error: float,
) -> float:
    """
    Compute information ratio (alpha / tracking error).
    
    Args:
        alpha: Annualized alpha
        tracking_error: Annualized tracking error
        
    Returns:
        Information ratio
    """
    if tracking_error > 0:
        return alpha / tracking_error
    return 0.0


def compute_benchmark_metrics(
    strategy_returns: List[float],
    benchmark_returns: List[float],
) -> BenchmarkMetrics:
    """
    Compute all benchmark-relative metrics.
    
    Args:
        strategy_returns: Daily strategy returns
        benchmark_returns: Daily benchmark returns (SPY)
        
    Returns:
        BenchmarkMetrics dataclass
    """
    # Cumulative returns
    strategy_cum = np.prod([1 + r for r in strategy_returns]) - 1 if strategy_returns else 0.0
    benchmark_cum = np.prod([1 + r for r in benchmark_returns]) - 1 if benchmark_returns else 0.0
    
    # Alpha and beta
    alpha = compute_alpha(strategy_returns, benchmark_returns)
    beta = compute_beta(strategy_returns, benchmark_returns)
    
    # Tracking error and IR
    tracking_error = compute_tracking_error(strategy_returns, benchmark_returns)
    information_ratio = compute_information_ratio(alpha, tracking_error)
    
    # Correlation
    correlation = 0.0
    if len(strategy_returns) >= 2 and len(benchmark_returns) >= 2:
        min_len = min(len(strategy_returns), len(benchmark_returns))
        correlation = float(np.corrcoef(
            strategy_returns[:min_len],
            benchmark_returns[:min_len]
        )[0, 1])
    
    return BenchmarkMetrics(
        strategy_return=strategy_cum,
        benchmark_return=benchmark_cum,
        alpha=alpha,
        beta=beta,
        tracking_error=tracking_error,
        information_ratio=information_ratio,
        correlation=correlation,
    )


def compute_cost_metrics(
    gross_return: float,
    total_commissions: float,
    total_slippage_bps: float,
    avg_portfolio_value: float,
    total_trade_value: float,
    n_days: int,
) -> CostMetrics:
    """
    Compute transaction cost metrics.
    
    Args:
        gross_return: Return before costs
        total_commissions: Total commissions paid ($)
        total_slippage_bps: Total slippage (bps)
        avg_portfolio_value: Average portfolio value
        total_trade_value: Total value traded (sum of |trade_value|)
        n_days: Number of trading days
        
    Returns:
        CostMetrics dataclass
    """
    # Commission drag
    commission_drag_bps = (total_commissions / avg_portfolio_value) * 10000 if avg_portfolio_value > 0 else 0
    
    # Total cost drag
    cost_drag_bps = commission_drag_bps + total_slippage_bps
    
    # Net return
    net_return = gross_return - (cost_drag_bps / 10000)
    
    # Turnover
    turnover_daily = total_trade_value / avg_portfolio_value / n_days if avg_portfolio_value > 0 and n_days > 0 else 0
    turnover_annualized = turnover_daily * 252
    
    return CostMetrics(
        gross_return=gross_return,
        total_commissions=total_commissions,
        total_slippage_bps=total_slippage_bps,
        net_return=net_return,
        turnover_annualized=turnover_annualized,
        cost_drag_bps=cost_drag_bps,
    )


def detect_regime(
    spy_return_21d: Optional[float],
    spy_vol_21d: Optional[float],
) -> str:
    """
    Detect market regime based on SPY.
    
    Args:
        spy_return_21d: SPY 21-day return
        spy_vol_21d: SPY 21-day volatility
        
    Returns:
        Regime label: "bull_low_vol", "bull_high_vol", "bear_low_vol", "bear_high_vol"
    """
    if spy_return_21d is None or spy_vol_21d is None:
        return "unknown"
    
    trend = "bull" if spy_return_21d > 0 else "bear"
    vol_level = "high_vol" if spy_vol_21d > 0.20 else "low_vol"
    
    return f"{trend}_{vol_level}"


def compute_regime_performance(
    decisions: List[Dict],
    regime_labels: List[str],
    outcomes: List[float],
) -> List[RegimeMetrics]:
    """
    Compute performance by market regime.
    
    Args:
        decisions: List of decision dicts
        regime_labels: Regime label for each decision
        outcomes: Outcome for each decision
        
    Returns:
        List of RegimeMetrics by regime
    """
    # Group by regime
    regime_groups: Dict[str, List[float]] = {}
    
    for regime, outcome in zip(regime_labels, outcomes):
        if regime not in regime_groups:
            regime_groups[regime] = []
        regime_groups[regime].append(outcome)
    
    # Compute metrics per regime
    results = []
    for regime, returns in regime_groups.items():
        if not returns:
            continue
        
        n_decisions = len(returns)
        avg_return = float(np.mean(returns))
        sharpe = float(np.mean(returns) / np.std(returns) * np.sqrt(252)) if np.std(returns) > 0 else 0.0
        win_rate = sum(1 for r in returns if r > 0) / n_decisions
        
        # Max drawdown
        cumulative = [np.prod([1 + r for r in returns[:i+1]]) for i in range(len(returns))]
        max_dd = 0.0
        peak = cumulative[0] if cumulative else 1.0
        for val in cumulative:
            if val > peak:
                peak = val
            elif peak > 0:
                dd = (peak - val) / peak
                max_dd = max(max_dd, dd)
        
        results.append(RegimeMetrics(
            regime=regime,
            n_decisions=n_decisions,
            avg_return=avg_return,
            sharpe_ratio=sharpe,
            win_rate=win_rate,
            max_drawdown=max_dd,
        ))
    
    return results


def compute_attribution(
    pnl_by_fund: Dict[str, float],
    pnl_by_sector: Dict[str, float],
    portfolio_values: Dict[str, List[float]],
) -> AttributionMetrics:
    """
    Compute PnL attribution by fund and sector.
    
    Args:
        pnl_by_fund: Dict of fund_id -> total PnL
        pnl_by_sector: Dict of sector -> total PnL
        portfolio_values: Dict of fund_id -> [values over time]
        
    Returns:
        AttributionMetrics
    """
    # Compute returns by fund
    return_by_fund = {}
    for fund_id, values in portfolio_values.items():
        if len(values) >= 2:
            ret = (values[-1] - values[0]) / values[0]
            return_by_fund[fund_id] = ret
    
    # Find best/worst
    best_fund = max(pnl_by_fund.keys(), key=lambda f: pnl_by_fund[f]) if pnl_by_fund else "N/A"
    worst_fund = min(pnl_by_fund.keys(), key=lambda f: pnl_by_fund[f]) if pnl_by_fund else "N/A"
    
    return AttributionMetrics(
        pnl_by_fund=pnl_by_fund,
        pnl_by_sector=pnl_by_sector,
        return_by_fund=return_by_fund,
        best_fund=best_fund,
        worst_fund=worst_fund,
    )
