"""
Point-in-Time Snapshot Builder for Backtesting.

Builds GlobalMarketSnapshot using ONLY data available at that moment.
Strict enforcement of no look-ahead bias.
"""

from datetime import date, datetime, time
from typing import Dict, List, Optional
import logging

from core.data.snapshot import (
    GlobalMarketSnapshot,
    DataQuality,
    EarningsEvent,
    MacroRelease,
)
from core.backtest.data_loader import HistoricalDataLoader

logger = logging.getLogger(__name__)


class PointInTimeSnapshotBuilder:
    """
    Builds GlobalMarketSnapshot with ONLY historical data.
    
    Key principle: NO FUTURE DATA LEAKAGE.
    All calculations use data <= asof_date.
    """
    
    # Return periods to calculate (in trading days)
    RETURN_PERIODS = {
        "1d": 1,
        "5d": 5,
        "21d": 21,   # ~1 month
        "63d": 63,   # ~3 months
    }
    
    # Volatility windows (in trading days)
    VOLATILITY_WINDOWS = {
        "5d": 5,
        "21d": 21,
    }
    
    def __init__(
        self,
        data_loader: HistoricalDataLoader,
        universe: Optional[List[str]] = None,
    ):
        """
        Initialize the snapshot builder.
        
        Args:
            data_loader: HistoricalDataLoader instance
            universe: Optional custom universe (defaults to loader's universe)
        """
        self.loader = data_loader
        self.universe = universe or data_loader.universe
    
    def build_snapshot(
        self,
        asof_date: date,
        include_volatility: bool = True,
    ) -> GlobalMarketSnapshot:
        """
        Build a GlobalMarketSnapshot using ONLY historical data.
        
        CRITICAL: This method NEVER accesses data after asof_date.
        All prices, returns, and volatility are calculated from
        data that existed on or before asof_date.
        
        Args:
            asof_date: The "current" date in simulation
            include_volatility: Whether to include volatility calculations
            
        Returns:
            GlobalMarketSnapshot with point-in-time data
        """
        # Get prices for all symbols as of this date
        prices = self._build_prices(asof_date)
        
        # Calculate returns using ONLY historical data
        returns = self._build_returns(asof_date, list(prices.keys()))
        
        # Calculate volatility using ONLY historical data
        volatility = {}
        if include_volatility:
            volatility = self._build_volatility(asof_date, list(prices.keys()))
        
        # Build quality metrics
        quality = self._build_quality(prices, returns, volatility)
        
        # Create snapshot
        snapshot = GlobalMarketSnapshot(
            snapshot_id=f"backtest_{asof_date.isoformat()}",
            asof_timestamp=datetime.combine(asof_date, time(16, 0)),  # Market close
            prices=prices,
            returns=returns,
            volatility=volatility,
            correlations={},  # Skip correlations for performance
            upcoming_earnings=[],  # No future events in backtest
            recent_macro_releases=[],
            news_summaries=[],
            quality=quality,
            coverage_symbols=list(prices.keys()),
            data_sources=["yfinance_backtest_cache"],
        )
        
        return snapshot
    
    def _build_prices(self, asof_date: date) -> Dict[str, float]:
        """
        Get closing prices for all symbols as of asof_date.
        
        Uses the most recent available price if asof_date is not a trading day.
        """
        prices = {}
        
        for symbol in self.universe:
            price = self.loader.get_price_asof(symbol, asof_date)
            if price is not None:
                prices[symbol] = price
        
        return prices
    
    def _build_returns(
        self,
        asof_date: date,
        symbols: List[str],
    ) -> Dict[str, Dict[str, float]]:
        """
        Calculate returns for all periods using ONLY historical data.
        
        CRITICAL: All lookback calculations use data <= asof_date.
        """
        returns = {}
        
        for symbol in symbols:
            symbol_returns = {}
            
            for period_name, days in self.RETURN_PERIODS.items():
                ret = self.loader.calc_return(symbol, asof_date, days)
                if ret is not None:
                    symbol_returns[period_name] = ret
            
            if symbol_returns:
                returns[symbol] = symbol_returns
        
        return returns
    
    def _build_volatility(
        self,
        asof_date: date,
        symbols: List[str],
    ) -> Dict[str, Dict[str, float]]:
        """
        Calculate volatility for all windows using ONLY historical data.
        
        CRITICAL: All rolling calculations use data <= asof_date.
        """
        volatility = {}
        
        for symbol in symbols:
            symbol_vol = {}
            
            for window_name, window in self.VOLATILITY_WINDOWS.items():
                vol = self.loader.calc_volatility(symbol, asof_date, window)
                if vol is not None:
                    symbol_vol[window_name] = vol
            
            if symbol_vol:
                volatility[symbol] = symbol_vol
        
        return volatility
    
    def _build_quality(
        self,
        prices: Dict[str, float],
        returns: Dict[str, Dict[str, float]],
        volatility: Dict[str, Dict[str, float]],
    ) -> DataQuality:
        """Build data quality metrics."""
        # Calculate coverage ratio
        coverage = len(prices) / len(self.universe) if self.universe else 0.0
        
        # Track missing fields per symbol
        missing_fields = {}
        for symbol in self.universe:
            missing = []
            if symbol not in prices:
                missing.append("price")
            if symbol not in returns:
                missing.append("returns")
            elif not returns.get(symbol, {}).get("1d"):
                missing.append("returns.1d")
            if volatility and symbol not in volatility:
                missing.append("volatility")
            
            if missing:
                missing_fields[symbol] = missing
        
        warnings = []
        if coverage < 0.9:
            warnings.append(f"Low coverage: {coverage:.1%}")
        
        return DataQuality(
            coverage_ratio=coverage,
            staleness_seconds={},  # Not applicable for backtest
            missing_fields=missing_fields,
            warnings=warnings,
        )
    
    def build_snapshots_range(
        self,
        start_date: date,
        end_date: date,
        include_volatility: bool = True,
    ) -> Dict[date, GlobalMarketSnapshot]:
        """
        Build snapshots for a range of dates.
        
        Useful for pre-computing snapshots for faster backtesting.
        
        Args:
            start_date: Start of range
            end_date: End of range
            include_volatility: Whether to include volatility
            
        Returns:
            Dict of date -> GlobalMarketSnapshot
        """
        trading_days = self.loader.get_trading_days_range(start_date, end_date)
        snapshots = {}
        
        logger.info(f"Building {len(trading_days)} snapshots...")
        
        for i, day in enumerate(trading_days):
            if i % 100 == 0:
                logger.info(f"Progress: {i}/{len(trading_days)} snapshots")
            
            snapshot = self.build_snapshot(day, include_volatility)
            snapshots[day] = snapshot
        
        logger.info(f"Built {len(snapshots)} snapshots")
        return snapshots


# Convenience function
def build_backtest_snapshot(
    loader: HistoricalDataLoader,
    asof_date: date,
) -> GlobalMarketSnapshot:
    """
    Quick helper to build a single backtest snapshot.
    
    Args:
        loader: HistoricalDataLoader instance
        asof_date: The simulation date
        
    Returns:
        GlobalMarketSnapshot with point-in-time data
    """
    builder = PointInTimeSnapshotBuilder(loader)
    return builder.build_snapshot(asof_date)
