"""
Evaluation Metrics - Trade-level outcomes for v1.

Key principles:
- Trade-level outcomes first, Sharpe later
- Store outcomes for later analysis
- Minimal metrics: hit_rate, Brier score, turnover, avg_slippage
- Defer complex metrics like Sharpe (low sample comedy show)
"""
from dataclasses import dataclass
from datetime import date, datetime
from typing import List, Optional


@dataclass
class TradeResult:
    """Result of a single trade for evaluation."""
    symbol: str
    direction: str  # "long" or "short"
    predicted_direction: str  # "up" or "down"
    predicted_confidence: float  # 0-1
    actual_return: float  # realized return
    was_correct: bool  # did direction match?
    holding_days: int
    slippage_bps: float


@dataclass
class FundMetrics:
    """
    Minimal v1 fund metrics.
    
    Focus on trade-level outcomes. Defer Sharpe and complex risk metrics.
    """
    fund_id: str
    period_start: date
    period_end: date
    
    # Trade counts
    n_trades: int
    n_winning_trades: int
    n_losing_trades: int
    
    # Return metrics
    avg_return: float
    median_return: float
    max_return: float
    min_return: float
    
    # Directional accuracy
    hit_rate: float  # % of trades with correct direction
    
    # Calibration (Brier score)
    brier_score: float  # Lower is better, 0 = perfect
    
    # Activity metrics
    turnover: float  # Total turnover over period
    avg_holding_days: float
    
    # Execution quality
    avg_slippage_bps: float
    
    # Risk (simple for v1)
    max_drawdown: float  # Peak to trough


def compute_hit_rate(results: List[TradeResult]) -> float:
    """
    Compute directional hit rate.
    
    % of trades where predicted direction matched actual.
    """
    if not results:
        return 0.0
    
    correct = sum(1 for r in results if r.was_correct)
    return correct / len(results)


def compute_brier_score(results: List[TradeResult]) -> float:
    """
    Compute Brier score for calibration.
    
    Brier score = mean((confidence - actual)^2)
    where actual = 1 if correct, 0 if wrong.
    
    Lower is better:
    - 0.0 = perfect calibration
    - 0.25 = random guessing
    - 1.0 = always wrong with full confidence
    """
    if not results:
        return 0.0
    
    total = 0.0
    for r in results:
        actual = 1.0 if r.was_correct else 0.0
        total += (r.predicted_confidence - actual) ** 2
    
    return total / len(results)


def compute_turnover(
    trade_values: List[float],
    portfolio_value: float,
    days: int
) -> float:
    """
    Compute annualized turnover.
    
    Turnover = (total trade value / avg portfolio value) * (252 / days)
    """
    if portfolio_value <= 0 or days <= 0:
        return 0.0
    
    total_traded = sum(abs(v) for v in trade_values)
    daily_turnover = total_traded / portfolio_value / days
    annualized = daily_turnover * 252
    
    return annualized


def compute_fund_metrics(
    fund_id: str,
    period_start: date,
    period_end: date,
    results: List[TradeResult],
    trade_values: List[float],
    portfolio_value: float,
    portfolio_values: List[float],  # For drawdown
) -> FundMetrics:
    """
    Compute all fund metrics for a period.
    
    Args:
        fund_id: Fund identifier
        period_start: Start of evaluation period
        period_end: End of evaluation period
        results: List of trade results
        trade_values: List of trade values (for turnover)
        portfolio_value: Current portfolio value
        portfolio_values: Time series of portfolio values (for drawdown)
    
    Returns:
        FundMetrics with all computed values
    """
    if not results:
        return FundMetrics(
            fund_id=fund_id,
            period_start=period_start,
            period_end=period_end,
            n_trades=0,
            n_winning_trades=0,
            n_losing_trades=0,
            avg_return=0.0,
            median_return=0.0,
            max_return=0.0,
            min_return=0.0,
            hit_rate=0.0,
            brier_score=0.0,
            turnover=0.0,
            avg_holding_days=0.0,
            avg_slippage_bps=0.0,
            max_drawdown=0.0,
        )
    
    # Basic counts
    n_trades = len(results)
    returns = [r.actual_return for r in results]
    n_winning = sum(1 for r in returns if r > 0)
    n_losing = sum(1 for r in returns if r < 0)
    
    # Return stats
    sorted_returns = sorted(returns)
    avg_return = sum(returns) / len(returns)
    median_return = sorted_returns[len(sorted_returns) // 2]
    max_return = max(returns)
    min_return = min(returns)
    
    # Hit rate
    hit_rate = compute_hit_rate(results)
    
    # Brier score
    brier = compute_brier_score(results)
    
    # Turnover
    days = (period_end - period_start).days or 1
    turnover = compute_turnover(trade_values, portfolio_value, days)
    
    # Holding period
    holding_days = [r.holding_days for r in results if r.holding_days > 0]
    avg_holding = sum(holding_days) / len(holding_days) if holding_days else 0.0
    
    # Slippage
    slippages = [r.slippage_bps for r in results]
    avg_slippage = sum(slippages) / len(slippages) if slippages else 0.0
    
    # Max drawdown
    max_dd = compute_max_drawdown(portfolio_values)
    
    return FundMetrics(
        fund_id=fund_id,
        period_start=period_start,
        period_end=period_end,
        n_trades=n_trades,
        n_winning_trades=n_winning,
        n_losing_trades=n_losing,
        avg_return=avg_return,
        median_return=median_return,
        max_return=max_return,
        min_return=min_return,
        hit_rate=hit_rate,
        brier_score=brier,
        turnover=turnover,
        avg_holding_days=avg_holding,
        avg_slippage_bps=avg_slippage,
        max_drawdown=max_dd,
    )


def compute_max_drawdown(values: List[float]) -> float:
    """
    Compute maximum drawdown from portfolio value series.
    
    Max drawdown = max((peak - trough) / peak)
    """
    if not values or len(values) < 2:
        return 0.0
    
    max_dd = 0.0
    peak = values[0]
    
    for v in values:
        if v > peak:
            peak = v
        elif peak > 0:
            dd = (peak - v) / peak
            max_dd = max(max_dd, dd)
    
    return max_dd


class OutcomeTracker:
    """
    Track trade outcomes for later evaluation.
    
    Records entry, tracks position, then records exit.
    """
    
    def __init__(self):
        self._open_trades: dict = {}  # symbol -> entry info
    
    def record_entry(
        self,
        decision_id: str,
        fund_id: str,
        symbol: str,
        direction: str,
        entry_price: float,
        entry_weight: float,
        predicted_direction: str,
        predicted_confidence: float,
    ) -> None:
        """Record a trade entry."""
        key = f"{fund_id}:{symbol}"
        self._open_trades[key] = {
            "decision_id": decision_id,
            "fund_id": fund_id,
            "symbol": symbol,
            "direction": direction,
            "entry_price": entry_price,
            "entry_weight": entry_weight,
            "entry_timestamp": datetime.utcnow(),
            "predicted_direction": predicted_direction,
            "predicted_confidence": predicted_confidence,
        }
    
    def record_exit(
        self,
        fund_id: str,
        symbol: str,
        exit_price: float,
        exit_reason: str,
        slippage_bps: float = 0.0,
    ) -> Optional[TradeResult]:
        """
        Record a trade exit and return the result.
        
        Returns None if no matching entry found.
        """
        key = f"{fund_id}:{symbol}"
        entry = self._open_trades.pop(key, None)
        
        if entry is None:
            return None
        
        # Calculate return
        if entry["direction"] == "long":
            actual_return = (exit_price - entry["entry_price"]) / entry["entry_price"]
        else:
            actual_return = (entry["entry_price"] - exit_price) / entry["entry_price"]
        
        # Check direction correctness
        predicted = entry["predicted_direction"]
        actual_direction = "up" if actual_return > 0 else "down"
        was_correct = (
            (predicted == "up" and actual_return > 0) or
            (predicted == "down" and actual_return < 0)
        )
        
        # Calculate holding period
        entry_ts = entry["entry_timestamp"]
        holding_days = (datetime.utcnow() - entry_ts).days or 1
        
        return TradeResult(
            symbol=symbol,
            direction=entry["direction"],
            predicted_direction=predicted,
            predicted_confidence=entry["predicted_confidence"],
            actual_return=actual_return,
            was_correct=was_correct,
            holding_days=holding_days,
            slippage_bps=slippage_bps,
        )
    
    def get_open_trades(self) -> List[dict]:
        """Get all open trades."""
        return list(self._open_trades.values())
