"""
Performance Tracking and Metrics

Calculates Sharpe ratio, volatility, max drawdown, and other metrics.
"""
import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from decimal import Decimal


@dataclass
class PerformanceMetrics:
    """Complete performance metrics for a manager"""
    manager_id: str
    total_return: float
    annualized_return: float
    volatility: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    win_rate: float
    total_trades: int
    profitable_trades: int
    avg_win: float
    avg_loss: float
    profit_factor: float
    current_drawdown: float
    days_tracked: int


class PerformanceTracker:
    """
    Track and calculate performance metrics for portfolio managers.
    
    Calculates:
    - Returns (daily, cumulative, annualized)
    - Risk metrics (volatility, Sharpe, Sortino, max drawdown)
    - Trade statistics (win rate, profit factor, etc.)
    """
    
    def __init__(self, risk_free_rate: float = 0.05):
        """
        Initialize performance tracker.
        
        Args:
            risk_free_rate: Annual risk-free rate (default 5%)
        """
        self.risk_free_rate = risk_free_rate
        self.daily_rf_rate = risk_free_rate / 252  # Daily risk-free rate
    
    def calculate_returns(
        self,
        portfolio_values: List[float]
    ) -> Tuple[List[float], float]:
        """
        Calculate daily and cumulative returns.
        
        Args:
            portfolio_values: List of portfolio values over time
        
        Returns:
            Tuple of (daily_returns, cumulative_return)
        """
        if len(portfolio_values) < 2:
            return [], 0.0
        
        # Daily returns
        daily_returns = []
        for i in range(1, len(portfolio_values)):
            ret = (
                (portfolio_values[i] - portfolio_values[i-1]) 
                / portfolio_values[i-1]
            )
            daily_returns.append(ret)
        
        # Cumulative return
        cum_return = (
            (portfolio_values[-1] - portfolio_values[0]) 
            / portfolio_values[0]
        )
        
        return daily_returns, cum_return
    
    def calculate_volatility(
        self,
        returns: List[float],
        annualize: bool = True
    ) -> float:
        """
        Calculate volatility (standard deviation of returns).
        
        Args:
            returns: List of returns
            annualize: If True, annualize volatility
        
        Returns:
            Volatility (annualized if annualize=True)
        """
        if len(returns) < 2:
            return 0.0
        
        vol = np.std(returns, ddof=1)
        
        if annualize:
            vol = vol * np.sqrt(252)
        
        return float(vol)
    
    def calculate_sharpe_ratio(
        self,
        returns: List[float],
        risk_free_rate: Optional[float] = None
    ) -> float:
        """
        Calculate Sharpe ratio.
        
        Sharpe = (Return - RiskFreeRate) / Volatility
        
        Args:
            returns: List of daily returns
            risk_free_rate: Annual risk-free rate (uses instance default if None)
        
        Returns:
            Annualized Sharpe ratio
        """
        if len(returns) < 2:
            return 0.0
        
        rf_rate = risk_free_rate if risk_free_rate is not None else self.risk_free_rate
        daily_rf = rf_rate / 252
        
        # Calculate excess returns
        excess_returns = [r - daily_rf for r in returns]
        
        mean_excess = np.mean(excess_returns)
        std_excess = np.std(excess_returns, ddof=1)
        
        if std_excess == 0:
            return 0.0
        
        # Annualize
        sharpe = (mean_excess / std_excess) * np.sqrt(252)
        
        return float(sharpe)
    
    def calculate_sortino_ratio(
        self,
        returns: List[float],
        risk_free_rate: Optional[float] = None,
        target_return: float = 0.0
    ) -> float:
        """
        Calculate Sortino ratio (Sharpe but only using downside volatility).
        
        Args:
            returns: List of daily returns
            risk_free_rate: Annual risk-free rate
            target_return: Target return (default 0)
        
        Returns:
            Annualized Sortino ratio
        """
        if len(returns) < 2:
            return 0.0
        
        rf_rate = risk_free_rate if risk_free_rate is not None else self.risk_free_rate
        daily_rf = rf_rate / 252
        
        # Calculate excess returns
        excess_returns = [r - daily_rf for r in returns]
        mean_excess = np.mean(excess_returns)
        
        # Downside deviation (only negative returns)
        downside_returns = [r for r in excess_returns if r < target_return]
        
        if len(downside_returns) < 2:
            return 0.0
        
        downside_std = np.std(downside_returns, ddof=1)
        
        if downside_std == 0:
            return 0.0
        
        # Annualize
        sortino = (mean_excess / downside_std) * np.sqrt(252)
        
        return float(sortino)
    
    def calculate_max_drawdown(
        self,
        portfolio_values: List[float]
    ) -> Tuple[float, float]:
        """
        Calculate maximum drawdown and current drawdown.
        
        Args:
            portfolio_values: List of portfolio values
        
        Returns:
            Tuple of (max_drawdown, current_drawdown)
        """
        if len(portfolio_values) < 2:
            return 0.0, 0.0
        
        peak = portfolio_values[0]
        max_dd = 0.0
        
        for value in portfolio_values:
            if value > peak:
                peak = value
            
            dd = (value - peak) / peak
            if dd < max_dd:
                max_dd = dd
        
        # Current drawdown
        current_dd = (portfolio_values[-1] - peak) / peak
        
        return float(max_dd), float(current_dd)
    
    def calculate_trade_statistics(
        self,
        trades: List[Dict]
    ) -> Dict[str, float]:
        """
        Calculate trade-level statistics.
        
        Args:
            trades: List of trade dicts with 'pnl' key
        
        Returns:
            Dict of trade statistics
        """
        if not trades:
            return {
                "total_trades": 0,
                "profitable_trades": 0,
                "win_rate": 0.0,
                "avg_win": 0.0,
                "avg_loss": 0.0,
                "profit_factor": 0.0
            }
        
        pnls = [t.get('pnl', 0) for t in trades]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p < 0]
        
        total_trades = len(trades)
        profitable_trades = len(wins)
        win_rate = profitable_trades / total_trades if total_trades > 0 else 0
        
        avg_win = np.mean(wins) if wins else 0.0
        avg_loss = np.mean(losses) if losses else 0.0
        
        total_wins = sum(wins)
        total_losses = abs(sum(losses))
        profit_factor = total_wins / total_losses if total_losses > 0 else 0.0
        
        return {
            "total_trades": total_trades,
            "profitable_trades": profitable_trades,
            "win_rate": float(win_rate),
            "avg_win": float(avg_win),
            "avg_loss": float(avg_loss),
            "profit_factor": float(profit_factor)
        }
    
    def calculate_metrics(
        self,
        manager_id: str,
        portfolio_values: List[float],
        trades: Optional[List[Dict]] = None,
        initial_value: Optional[float] = None
    ) -> PerformanceMetrics:
        """
        Calculate complete performance metrics.
        
        Args:
            manager_id: Manager ID
            portfolio_values: Historical portfolio values
            trades: List of completed trades
            initial_value: Initial portfolio value
        
        Returns:
            PerformanceMetrics object
        """
        if not portfolio_values:
            return PerformanceMetrics(
                manager_id=manager_id,
                total_return=0.0,
                annualized_return=0.0,
                volatility=0.0,
                sharpe_ratio=0.0,
                sortino_ratio=0.0,
                max_drawdown=0.0,
                win_rate=0.0,
                total_trades=0,
                profitable_trades=0,
                avg_win=0.0,
                avg_loss=0.0,
                profit_factor=0.0,
                current_drawdown=0.0,
                days_tracked=0
            )
        
        # Returns
        daily_returns, cum_return = self.calculate_returns(portfolio_values)
        
        # Annualized return
        days = len(portfolio_values) - 1
        if days > 0 and initial_value:
            ann_return = (
                (portfolio_values[-1] / initial_value) ** (252 / days) - 1
            )
        else:
            ann_return = 0.0
        
        # Risk metrics
        volatility = self.calculate_volatility(daily_returns)
        sharpe = self.calculate_sharpe_ratio(daily_returns)
        sortino = self.calculate_sortino_ratio(daily_returns)
        max_dd, current_dd = self.calculate_max_drawdown(portfolio_values)
        
        # Trade statistics
        trade_stats = self.calculate_trade_statistics(trades or [])
        
        return PerformanceMetrics(
            manager_id=manager_id,
            total_return=float(cum_return),
            annualized_return=float(ann_return),
            volatility=volatility,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            max_drawdown=max_dd,
            win_rate=trade_stats['win_rate'],
            total_trades=trade_stats['total_trades'],
            profitable_trades=trade_stats['profitable_trades'],
            avg_win=trade_stats['avg_win'],
            avg_loss=trade_stats['avg_loss'],
            profit_factor=trade_stats['profit_factor'],
            current_drawdown=current_dd,
            days_tracked=len(portfolio_values)
        )
    
    def compare_managers(
        self,
        metrics_list: List[PerformanceMetrics],
        sort_by: str = "sharpe_ratio"
    ) -> List[PerformanceMetrics]:
        """
        Compare and rank managers by a metric.
        
        Args:
            metrics_list: List of PerformanceMetrics
            sort_by: Metric to sort by
        
        Returns:
            Sorted list of metrics
        """
        return sorted(
            metrics_list,
            key=lambda m: getattr(m, sort_by, 0),
            reverse=True
        )
    
    def to_dict(self, metrics: PerformanceMetrics) -> Dict:
        """Convert metrics to dictionary for database storage"""
        return {
            "manager_id": metrics.manager_id,
            "total_return": Decimal(str(metrics.total_return)),
            "annualized_return": Decimal(str(metrics.annualized_return)),
            "volatility": Decimal(str(metrics.volatility)),
            "sharpe_ratio": Decimal(str(metrics.sharpe_ratio)),
            "sortino_ratio": Decimal(str(metrics.sortino_ratio)),
            "max_drawdown": Decimal(str(metrics.max_drawdown)),
            "win_rate": Decimal(str(metrics.win_rate)),
            "total_trades": metrics.total_trades,
            "profitable_trades": metrics.profitable_trades,
            "avg_win": Decimal(str(metrics.avg_win)),
            "avg_loss": Decimal(str(metrics.avg_loss)),
            "profit_factor": Decimal(str(metrics.profit_factor)),
            "current_drawdown": Decimal(str(metrics.current_drawdown))
        }
