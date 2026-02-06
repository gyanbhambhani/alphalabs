"""
Historical Data Loader for Backtesting.

Fetches and caches 25 years of daily OHLCV data (2000-2025).
Enforces strict point-in-time access - NO FUTURE DATA.

Can optionally use ChromaDB embeddings data as a fallback for price info.
"""

import os
import pandas as pd
import numpy as np
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import yfinance as yf
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

# Try to import ChromaDB for checking existing data
try:
    from core.semantic.vector_db import VectorDatabase
    CHROMA_AVAILABLE = True
except ImportError:
    CHROMA_AVAILABLE = False

# Legacy small universe (for quick testing)
LEGACY_UNIVERSE = [
    # Tech
    "AAPL", "MSFT", "INTC", "CSCO", "ORCL", "IBM",
    # Finance
    "JPM", "BAC", "WFC", "GS", "MS", "C", "AXP",
    # Consumer
    "WMT", "KO", "PEP", "MCD", "PG", "JNJ", "DIS",
    # Industrial
    "GE", "CAT", "MMM", "BA", "HON", "UPS",
    # Energy
    "XOM", "CVX", "SLB",
    # Healthcare
    "PFE", "MRK", "AMGN", "UNH",
    # Benchmark
    "SPY",
]


def load_sp500_universe() -> List[str]:
    """
    Load S&P 500 symbols from the CSV file.
    Falls back to legacy universe if file not found.
    """
    # Try multiple possible paths
    possible_paths = [
        Path(__file__).parent.parent.parent / "data" / "sp500_list.csv",
        Path("data/sp500_list.csv"),
        Path("python/data/sp500_list.csv"),
    ]
    
    for csv_path in possible_paths:
        if csv_path.exists():
            try:
                df = pd.read_csv(csv_path)
                symbols = df['symbol'].tolist()
                # Always include SPY for benchmark
                if 'SPY' not in symbols:
                    symbols.append('SPY')
                logger.info(f"Loaded {len(symbols)} S&P 500 symbols from {csv_path}")
                return symbols
            except Exception as e:
                logger.warning(f"Error loading S&P 500 list from {csv_path}: {e}")
    
    logger.warning("S&P 500 list not found, using legacy universe")
    return LEGACY_UNIVERSE


# Default universe - full S&P 500
BACKTEST_UNIVERSE = load_sp500_universe()


@dataclass
class CachedOHLCV:
    """Container for cached OHLCV data."""
    symbol: str
    data: pd.DataFrame  # Columns: open, high, low, close, volume
    start_date: date
    end_date: date
    
    @property
    def trading_days(self) -> int:
        return len(self.data)


class HistoricalDataLoader:
    """
    Loads and caches historical OHLCV data with strict point-in-time access.
    
    Key principles:
    - All data pre-fetched and cached to disk (parquet)
    - get_data_asof() ONLY returns data <= asof_date (no future leakage)
    - Data is immutable once cached
    """
    
    DEFAULT_START = date(2000, 1, 1)
    DEFAULT_END = date(2025, 1, 1)
    
    def __init__(
        self,
        cache_dir: Optional[str] = None,
        universe: Optional[List[str]] = None,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
    ):
        """
        Initialize the data loader.
        
        Args:
            cache_dir: Directory for parquet cache files
            universe: List of symbols to load (defaults to BACKTEST_UNIVERSE)
            start_date: Start date for data (defaults to 2000-01-01)
            end_date: End date for data (defaults to 2025-01-01)
        """
        self.cache_dir = Path(cache_dir or "data/backtest_cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.universe = universe or BACKTEST_UNIVERSE
        self.start_date = start_date or self.DEFAULT_START
        self.end_date = end_date or self.DEFAULT_END
        
        # In-memory cache (loaded from disk)
        self._data: Dict[str, pd.DataFrame] = {}
        self._loaded = False
        
        # Trading calendar (populated on first load)
        self._trading_days: Optional[pd.DatetimeIndex] = None
    
    @property
    def trading_days(self) -> pd.DatetimeIndex:
        """Get list of all trading days in the backtest period."""
        if self._trading_days is None:
            self._ensure_loaded()
            # Use SPY as reference for trading days
            if "SPY" in self._data:
                self._trading_days = self._data["SPY"].index
            else:
                # Fallback: use first available symbol
                first_sym = next(iter(self._data.keys()))
                self._trading_days = self._data[first_sym].index
        return self._trading_days
    
    def _cache_path(self, symbol: str) -> Path:
        """Get cache file path for a symbol."""
        return self.cache_dir / f"{symbol}.parquet"
    
    def _is_cached(self, symbol: str) -> bool:
        """Check if symbol data is cached on disk."""
        return self._cache_path(symbol).exists()
    
    def _load_from_cache(self, symbol: str) -> Optional[pd.DataFrame]:
        """Load symbol data from disk cache."""
        cache_path = self._cache_path(symbol)
        if not cache_path.exists():
            return None
        
        try:
            df = pd.read_parquet(cache_path)
            df.index = pd.to_datetime(df.index)
            return df
        except Exception as e:
            logger.error(f"Error loading cache for {symbol}: {e}")
            return None
    
    def _save_to_cache(self, symbol: str, df: pd.DataFrame) -> None:
        """Save symbol data to disk cache."""
        cache_path = self._cache_path(symbol)
        try:
            df.to_parquet(cache_path)
            logger.info(f"Cached {symbol} to {cache_path}")
        except Exception as e:
            logger.error(f"Error caching {symbol}: {e}")
    
    def _fetch_from_yfinance(self, symbol: str) -> Optional[pd.DataFrame]:
        """Fetch historical data from yfinance."""
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(
                start=self.start_date.isoformat(),
                end=self.end_date.isoformat(),
            )
            
            if df.empty:
                logger.warning(f"No data returned for {symbol}")
                return None
            
            # Standardize column names
            df.columns = df.columns.str.lower()
            
            # Keep only OHLCV
            required_cols = ['open', 'high', 'low', 'close', 'volume']
            for col in required_cols:
                if col not in df.columns:
                    logger.warning(f"Missing column {col} for {symbol}")
                    return None
            
            df = df[required_cols].copy()
            df = df.dropna()
            
            # Ensure datetime index
            df.index = pd.to_datetime(df.index).tz_localize(None)
            
            return df
            
        except Exception as e:
            logger.error(f"Error fetching {symbol}: {e}")
            return None
    
    def _fetch_batch_from_yfinance(
        self, 
        symbols: List[str], 
        batch_size: int = 50
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch multiple symbols in batches using yfinance's batch download.
        Much faster than fetching one at a time.
        """
        results = {}
        
        for i in range(0, len(symbols), batch_size):
            batch = symbols[i:i + batch_size]
            batch_str = " ".join(batch)
            
            print(f"  Fetching batch {i//batch_size + 1}: {len(batch)} symbols...")
            
            try:
                # Use yfinance download for batch fetching
                df = yf.download(
                    batch_str,
                    start=self.start_date.isoformat(),
                    end=self.end_date.isoformat(),
                    progress=False,
                    threads=True,
                )
                
                if df.empty:
                    continue
                
                # Handle single vs multi-symbol response
                if len(batch) == 1:
                    # Single symbol - df has simple columns
                    symbol = batch[0]
                    df.columns = df.columns.str.lower()
                    if 'close' in df.columns:
                        df = df[['open', 'high', 'low', 'close', 'volume']].copy()
                        df = df.dropna()
                        df.index = pd.to_datetime(df.index).tz_localize(None)
                        results[symbol] = df
                else:
                    # Multi-symbol - df has MultiIndex columns (Price, Symbol)
                    for symbol in batch:
                        try:
                            if symbol in df.columns.get_level_values(1):
                                sym_df = df.xs(symbol, level=1, axis=1).copy()
                                sym_df.columns = sym_df.columns.str.lower()
                                if 'close' in sym_df.columns:
                                    sym_df = sym_df[
                                        ['open', 'high', 'low', 'close', 'volume']
                                    ].copy()
                                    sym_df = sym_df.dropna()
                                    sym_df.index = pd.to_datetime(
                                        sym_df.index
                                    ).tz_localize(None)
                                    if len(sym_df) > 100:  # Min 100 days
                                        results[symbol] = sym_df
                        except Exception as e:
                            logger.debug(f"Error extracting {symbol}: {e}")
                            
            except Exception as e:
                logger.error(f"Batch fetch error: {e}")
        
        return results
    
    def fetch_and_cache_all(
        self, 
        force_refresh: bool = False,
        use_batch: bool = True,
    ) -> Dict[str, int]:
        """
        Fetch all symbols and cache to disk.
        
        Args:
            force_refresh: If True, re-fetch even if cached
            use_batch: If True, use batch downloading (much faster)
            
        Returns:
            Dict of symbol -> number of trading days fetched
        """
        results = {}
        symbols_to_fetch = []
        
        # Check cache first
        for symbol in self.universe:
            if not force_refresh and self._is_cached(symbol):
                df = self._load_from_cache(symbol)
                if df is not None:
                    results[symbol] = len(df)
                    continue
            symbols_to_fetch.append(symbol)
        
        if not symbols_to_fetch:
            print(f"All {len(results)} symbols loaded from cache")
            return results
        
        print(f"Fetching {len(symbols_to_fetch)} symbols...")
        
        if use_batch:
            # Batch fetch (much faster)
            batch_results = self._fetch_batch_from_yfinance(symbols_to_fetch)
            for symbol, df in batch_results.items():
                if df is not None and not df.empty:
                    self._save_to_cache(symbol, df)
                    results[symbol] = len(df)
                    
            # Report results
            fetched = len(batch_results)
            failed = len(symbols_to_fetch) - fetched
            print(f"  Fetched: {fetched}, Failed: {failed}")
        else:
            # One-by-one fetch (slower but more reliable)
            for symbol in symbols_to_fetch:
                logger.info(f"Fetching {symbol} from yfinance...")
                df = self._fetch_from_yfinance(symbol)
                
                if df is not None and not df.empty:
                    self._save_to_cache(symbol, df)
                    results[symbol] = len(df)
                    logger.info(f"{symbol}: fetched {len(df)} days")
                else:
                    logger.warning(f"{symbol}: fetch failed")
        
        return results
    
    def _ensure_loaded(self) -> None:
        """
        Load all cached data into memory.
        
        IMPORTANT: This only loads from cache - it does NOT fetch from yfinance.
        Call fetch_and_cache_all() first to populate the cache.
        """
        if self._loaded:
            return
        
        for symbol in self.universe:
            if symbol in self._data:
                continue
            
            df = self._load_from_cache(symbol)
            if df is not None:
                self._data[symbol] = df
            else:
                # Don't auto-fetch - just log that it's missing
                logger.debug(f"{symbol} not in cache (run fetch_and_cache_all first)")
        
        self._loaded = True
        logger.info(f"Loaded {len(self._data)}/{len(self.universe)} symbols from cache")
    
    def get_data_asof(
        self,
        symbol: str,
        asof_date: date,
    ) -> Optional[pd.DataFrame]:
        """
        Get OHLCV data for a symbol up to (and including) asof_date.
        
        CRITICAL: This is the ONLY way to access historical data.
        Returns ONLY data that existed at asof_date - NO FUTURE DATA.
        
        Args:
            symbol: Stock symbol
            asof_date: The "current" date in simulation (no data after this)
            
        Returns:
            DataFrame with OHLCV data up to asof_date, or None if not available
        """
        self._ensure_loaded()
        
        if symbol not in self._data:
            return None
        
        df = self._data[symbol]
        
        # Convert asof_date to datetime for comparison
        asof_dt = pd.Timestamp(asof_date)
        
        # CRITICAL: Filter to only include data <= asof_date
        mask = df.index <= asof_dt
        filtered = df[mask]
        
        if filtered.empty:
            return None
        
        return filtered.copy()
    
    def get_price_asof(
        self,
        symbol: str,
        asof_date: date,
    ) -> Optional[float]:
        """
        Get closing price for a symbol on asof_date.
        
        If asof_date is not a trading day, returns the most recent price.
        
        Args:
            symbol: Stock symbol
            asof_date: The date to get price for
            
        Returns:
            Closing price or None if not available
        """
        df = self.get_data_asof(symbol, asof_date)
        if df is None or df.empty:
            return None
        
        return float(df['close'].iloc[-1])
    
    def get_prices_asof(
        self,
        asof_date: date,
        symbols: Optional[List[str]] = None,
    ) -> Dict[str, float]:
        """
        Get closing prices for all symbols on asof_date.
        
        Args:
            asof_date: The date to get prices for
            symbols: Optional list of symbols (defaults to universe)
            
        Returns:
            Dict of symbol -> price
        """
        symbols = symbols or self.universe
        prices = {}
        
        for symbol in symbols:
            price = self.get_price_asof(symbol, asof_date)
            if price is not None:
                prices[symbol] = price
        
        return prices
    
    def calc_return(
        self,
        symbol: str,
        asof_date: date,
        lookback_days: int,
    ) -> Optional[float]:
        """
        Calculate return over lookback period ending at asof_date.
        
        Uses ONLY historical data - no future leakage.
        
        Args:
            symbol: Stock symbol
            asof_date: End date for calculation
            lookback_days: Number of trading days to look back
            
        Returns:
            Return as decimal (e.g., 0.05 for 5%), or None if insufficient data
        """
        df = self.get_data_asof(symbol, asof_date)
        if df is None or len(df) < lookback_days + 1:
            return None
        
        current_price = df['close'].iloc[-1]
        past_price = df['close'].iloc[-(lookback_days + 1)]
        
        return (current_price - past_price) / past_price
    
    def calc_volatility(
        self,
        symbol: str,
        asof_date: date,
        window: int,
    ) -> Optional[float]:
        """
        Calculate rolling volatility (annualized std of returns).
        
        Uses ONLY historical data - no future leakage.
        
        Args:
            symbol: Stock symbol
            asof_date: End date for calculation
            window: Number of trading days for rolling window
            
        Returns:
            Annualized volatility, or None if insufficient data
        """
        df = self.get_data_asof(symbol, asof_date)
        if df is None or len(df) < window + 1:
            return None
        
        # Calculate daily returns
        returns = df['close'].pct_change().dropna()
        
        if len(returns) < window:
            return None
        
        # Rolling std, annualized
        rolling_std = returns.iloc[-window:].std()
        annualized_vol = rolling_std * np.sqrt(252)
        
        return float(annualized_vol)
    
    def get_available_symbols(self, asof_date: date) -> List[str]:
        """
        Get list of symbols that have data available on asof_date.
        
        Args:
            asof_date: The date to check availability
            
        Returns:
            List of symbols with data on or before asof_date
        """
        self._ensure_loaded()
        
        available = []
        for symbol in self.universe:
            if symbol in self._data:
                df = self._data[symbol]
                if df.index[0].date() <= asof_date:
                    available.append(symbol)
        
        return available
    
    def get_trading_days_range(
        self,
        start: date,
        end: date,
    ) -> List[date]:
        """
        Get list of trading days between start and end dates.
        
        Args:
            start: Start date (inclusive)
            end: End date (inclusive)
            
        Returns:
            List of trading days
        """
        all_days = self.trading_days
        mask = (all_days >= pd.Timestamp(start)) & (all_days <= pd.Timestamp(end))
        return [d.date() for d in all_days[mask]]
    
    def cache_status(self) -> Dict:
        """
        Check cache status WITHOUT loading data or fetching from yfinance.
        
        Use this for status checks to avoid triggering downloads.
        Also checks ChromaDB for any existing embeddings data.
        """
        cached_symbols = []
        for symbol in self.universe:
            if self._is_cached(symbol):
                cached_symbols.append(symbol)
        
        # Try to get date range from SPY if cached
        start_date = None
        end_date = None
        trading_days = 0
        
        if "SPY" in cached_symbols:
            df = self._load_from_cache("SPY")
            if df is not None and not df.empty:
                start_date = str(df.index[0].date())
                end_date = str(df.index[-1].date())
                trading_days = len(df)
        
        # Check ChromaDB for existing embeddings
        chroma_symbols = []
        chroma_date_range = None
        if CHROMA_AVAILABLE:
            try:
                chroma_symbols = VectorDatabase.get_all_symbols("./chroma_data")
                if chroma_symbols:
                    # Check SPY embeddings for date range
                    if "SPY" in chroma_symbols:
                        db = VectorDatabase(
                            persist_directory="./chroma_data", 
                            symbol="SPY"
                        )
                        chroma_date_range = db.get_date_range()
            except Exception as e:
                logger.debug(f"ChromaDB check failed: {e}")
        
        return {
            "symbols_cached": len(cached_symbols),
            "total_symbols": len(self.universe),
            "trading_days": trading_days,
            "start_date": start_date,
            "end_date": end_date,
            "cache_dir": str(self.cache_dir),
            "ready": len(cached_symbols) >= len(self.universe) * 0.8,  # 80% threshold
            "chroma_symbols": len(chroma_symbols),
            "chroma_date_range": chroma_date_range,
        }
    
    def summary(self) -> Dict:
        """Get summary statistics of loaded data."""
        # If not loaded yet, just return cache status
        if not self._loaded:
            status = self.cache_status()
            return {
                "symbols_loaded": status["symbols_cached"],
                "total_symbols": status["total_symbols"],
                "trading_days": status["trading_days"],
                "start_date": status["start_date"],
                "end_date": status["end_date"],
                "cache_dir": status["cache_dir"],
            }
        
        return {
            "symbols_loaded": len(self._data),
            "total_symbols": len(self.universe),
            "trading_days": len(self.trading_days),
            "start_date": str(self.trading_days[0].date()) if len(self.trading_days) > 0 else None,
            "end_date": str(self.trading_days[-1].date()) if len(self.trading_days) > 0 else None,
            "cache_dir": str(self.cache_dir),
        }
