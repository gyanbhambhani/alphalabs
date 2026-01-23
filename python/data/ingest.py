"""
Data Ingestion Pipeline

Fetches historical market data from yfinance and Alpaca.
Supports batch loading for semantic search indexing.
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import yfinance as yf
from dataclasses import dataclass


@dataclass
class StockData:
    """Container for stock OHLCV data"""
    symbol: str
    data: pd.DataFrame  # Columns: open, high, low, close, volume
    start_date: str
    end_date: str
    
    @property
    def close(self) -> pd.Series:
        return self.data['close']
    
    @property
    def high(self) -> pd.Series:
        return self.data['high']
    
    @property
    def low(self) -> pd.Series:
        return self.data['low']
    
    @property
    def volume(self) -> pd.Series:
        return self.data['volume']


def fetch_stock_data(
    symbol: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    period: str = "10y"
) -> Optional[StockData]:
    """
    Fetch historical stock data from yfinance.
    
    Args:
        symbol: Stock ticker symbol
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        period: If no dates, use this period (1d, 5d, 1mo, 3mo, 6mo, 1y, 5y, 10y, max)
    
    Returns:
        StockData or None if fetch fails
    """
    try:
        ticker = yf.Ticker(symbol)
        
        if start_date and end_date:
            df = ticker.history(start=start_date, end=end_date)
        else:
            df = ticker.history(period=period)
        
        if df.empty:
            print(f"No data returned for {symbol}")
            return None
        
        # Standardize column names
        df.columns = df.columns.str.lower()
        
        # Keep only OHLCV
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        for col in required_cols:
            if col not in df.columns:
                print(f"Missing column {col} for {symbol}")
                return None
        
        df = df[required_cols].copy()
        
        # Remove any NaN rows
        df = df.dropna()
        
        return StockData(
            symbol=symbol,
            data=df,
            start_date=str(df.index[0].date()),
            end_date=str(df.index[-1].date())
        )
    
    except Exception as e:
        print(f"Error fetching {symbol}: {e}")
        return None


def fetch_multiple_stocks(
    symbols: List[str],
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    period: str = "10y"
) -> Dict[str, StockData]:
    """
    Fetch data for multiple stocks.
    
    Args:
        symbols: List of ticker symbols
        start_date: Start date
        end_date: End date
        period: Period if no dates specified
    
    Returns:
        Dictionary of symbol -> StockData
    """
    results = {}
    
    for symbol in symbols:
        data = fetch_stock_data(symbol, start_date, end_date, period)
        if data:
            results[symbol] = data
    
    return results


class DataIngestion:
    """
    High-level data ingestion manager.
    
    Handles fetching, caching, and updating market data.
    """
    
    DEFAULT_UNIVERSE = [
        # Tech
        "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA",
        "META", "TSLA", "AMD", "INTC", "NFLX",
        # Market indices (via ETFs)
        "SPY", "QQQ", "IWM",
        # Sectors
        "XLF", "XLE", "XLK", "XLV", "XLI"
    ]
    
    def __init__(
        self,
        universe: Optional[List[str]] = None,
        years_of_history: int = 10
    ):
        """
        Initialize data ingestion.
        
        Args:
            universe: List of symbols to track
            years_of_history: How many years of data to fetch
        """
        self.universe = universe or self.DEFAULT_UNIVERSE
        self.years_of_history = years_of_history
        self._data_cache: Dict[str, StockData] = {}
    
    def fetch_all(
        self,
        force_refresh: bool = False
    ) -> Dict[str, StockData]:
        """
        Fetch data for entire universe.
        
        Args:
            force_refresh: If True, ignore cache
        
        Returns:
            Dictionary of symbol -> StockData
        """
        if not force_refresh and self._data_cache:
            return self._data_cache
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=self.years_of_history * 365)
        
        self._data_cache = fetch_multiple_stocks(
            self.universe,
            start_date=start_date.strftime('%Y-%m-%d'),
            end_date=end_date.strftime('%Y-%m-%d')
        )
        
        print(f"Fetched data for {len(self._data_cache)}/{len(self.universe)} symbols")
        
        return self._data_cache
    
    def get_latest_prices(self) -> Dict[str, float]:
        """Get latest prices for all symbols"""
        if not self._data_cache:
            self.fetch_all()
        
        return {
            symbol: float(data.close.iloc[-1])
            for symbol, data in self._data_cache.items()
        }
    
    def get_market_data(self, symbol: str = "SPY") -> Optional[pd.DataFrame]:
        """Get market index data for relative calculations"""
        if not self._data_cache:
            self.fetch_all()
        
        if symbol in self._data_cache:
            return self._data_cache[symbol].data
        
        return None
    
    def get_close_prices(self) -> Dict[str, pd.Series]:
        """Get close price series for all symbols"""
        if not self._data_cache:
            self.fetch_all()
        
        return {
            symbol: data.close
            for symbol, data in self._data_cache.items()
        }
    
    def get_price_matrix(self) -> pd.DataFrame:
        """
        Get aligned price matrix for all symbols.
        
        Returns DataFrame with symbols as columns, dates as index.
        """
        if not self._data_cache:
            self.fetch_all()
        
        prices = {}
        for symbol, data in self._data_cache.items():
            prices[symbol] = data.close
        
        return pd.DataFrame(prices)
    
    def update_latest(self) -> Dict[str, float]:
        """
        Fetch only the latest day's data (for real-time updates).
        
        Returns latest prices.
        """
        latest = {}
        
        for symbol in self.universe:
            try:
                ticker = yf.Ticker(symbol)
                info = ticker.fast_info
                latest[symbol] = info.last_price
            except Exception as e:
                print(f"Error getting latest for {symbol}: {e}")
        
        return latest


def prepare_data_for_indexing(
    data: StockData
) -> pd.DataFrame:
    """
    Prepare stock data for semantic search indexing.
    
    Args:
        data: StockData object
    
    Returns:
        DataFrame ready for indexing
    """
    df = data.data.copy()
    
    # Ensure datetime index
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    
    # Sort by date
    df = df.sort_index()
    
    # Remove any duplicates
    df = df[~df.index.duplicated(keep='first')]
    
    return df
