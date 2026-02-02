"""
Data Normalizer

Single source of truth for DataFrame normalization in the embeddings pipeline.
Handles all the quirks of yfinance output formats consistently.
"""
import logging
from typing import List, Optional, Set

import pandas as pd

logger = logging.getLogger(__name__)


class NormalizationError(Exception):
    """Raised when DataFrame cannot be normalized to required format."""
    
    def __init__(self, symbol: str, message: str, missing_columns: List[str] = None):
        self.symbol = symbol
        self.message = message
        self.missing_columns = missing_columns or []
        super().__init__(f"{symbol}: {message}")


class DataNormalizer:
    """
    Normalizes stock DataFrames to a consistent format.
    
    Handles:
    - MultiIndex columns from yfinance (e.g., ('Close', 'AAPL'))
    - Case variations (Close vs close)
    - Duplicate columns after flattening
    - Missing optional columns
    
    Output format:
    - All columns lowercase: close, high, low, volume, open
    - DatetimeIndex for rows
    - Single-level column index
    """
    
    REQUIRED_COLUMNS: Set[str] = {'close', 'high', 'low'}
    OPTIONAL_COLUMNS: Set[str] = {'volume', 'open'}
    ALL_COLUMNS: Set[str] = REQUIRED_COLUMNS | OPTIONAL_COLUMNS
    
    # Common column name mappings
    COLUMN_ALIASES = {
        'adj close': 'close',
        'adjusted close': 'close',
        'adj_close': 'close',
    }
    
    def normalize(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """
        Normalize a DataFrame to standard format.
        
        Args:
            df: Raw DataFrame from yfinance or other source
            symbol: Stock symbol (for error messages)
            
        Returns:
            Normalized DataFrame with lowercase column names
            
        Raises:
            NormalizationError: If required columns are missing
        """
        if df is None or len(df) == 0:
            raise NormalizationError(symbol, "DataFrame is empty or None")
        
        # Work on a copy to avoid modifying original
        df = df.copy()
        
        # Step 1: Flatten MultiIndex columns
        df = self._flatten_multiindex(df, symbol)
        
        # Step 2: Lowercase all column names
        df = self._lowercase_columns(df)
        
        # Step 3: Apply column aliases
        df = self._apply_aliases(df)
        
        # Step 4: Remove duplicate columns
        df = self._deduplicate_columns(df, symbol)
        
        # Step 5: Validate required columns exist
        self._validate_required_columns(df, symbol)
        
        # Step 6: Ensure proper types
        df = self._ensure_types(df, symbol)
        
        logger.debug(f"{symbol}: Normalized to columns {list(df.columns)}")
        return df
    
    def _flatten_multiindex(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Flatten MultiIndex columns to single level."""
        if isinstance(df.columns, pd.MultiIndex):
            # yfinance returns columns like ('Close', 'AAPL')
            # Take the first level (the column name, not the ticker)
            original_cols = list(df.columns)
            df.columns = df.columns.get_level_values(0)
            logger.debug(
                f"{symbol}: Flattened MultiIndex columns "
                f"{original_cols[:3]}... -> {list(df.columns)[:3]}..."
            )
        return df
    
    def _lowercase_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert all column names to lowercase."""
        df.columns = [str(c).lower().strip() for c in df.columns]
        return df
    
    def _apply_aliases(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply column name aliases for common variations."""
        rename_map = {}
        for col in df.columns:
            if col in self.COLUMN_ALIASES:
                rename_map[col] = self.COLUMN_ALIASES[col]
        
        if rename_map:
            df = df.rename(columns=rename_map)
            logger.debug(f"Applied column aliases: {rename_map}")
        return df
    
    def _deduplicate_columns(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Remove duplicate columns, keeping first occurrence."""
        if df.columns.duplicated().any():
            duplicates = df.columns[df.columns.duplicated()].tolist()
            logger.warning(
                f"{symbol}: Removing duplicate columns: {duplicates}"
            )
            df = df.loc[:, ~df.columns.duplicated()]
        return df
    
    def _validate_required_columns(self, df: pd.DataFrame, symbol: str) -> None:
        """Validate that all required columns are present."""
        missing = self.REQUIRED_COLUMNS - set(df.columns)
        if missing:
            raise NormalizationError(
                symbol,
                f"Missing required columns: {sorted(missing)}",
                missing_columns=sorted(missing)
            )
    
    def _ensure_types(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Ensure numeric columns have proper types."""
        for col in self.ALL_COLUMNS:
            if col in df.columns:
                # Convert to numeric, coercing errors to NaN
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Ensure index is DatetimeIndex
        if not isinstance(df.index, pd.DatetimeIndex):
            try:
                df.index = pd.to_datetime(df.index)
            except Exception as e:
                logger.warning(f"{symbol}: Could not convert index to datetime: {e}")
        
        return df
    
    def get_series(
        self,
        df: pd.DataFrame,
        column: str,
        fallback_column: Optional[str] = None
    ) -> Optional[pd.Series]:
        """
        Safely extract a column as a Series.
        
        Args:
            df: Normalized DataFrame
            column: Column name to extract
            fallback_column: Alternative column if primary not found
            
        Returns:
            Series or None if column doesn't exist
        """
        if column in df.columns:
            series = df[column]
            # Handle case where column selection returns DataFrame
            if isinstance(series, pd.DataFrame):
                series = series.iloc[:, 0]
            return series
        
        if fallback_column and fallback_column in df.columns:
            series = df[fallback_column]
            if isinstance(series, pd.DataFrame):
                series = series.iloc[:, 0]
            return series
        
        return None
    
    def extract_ohlcv(self, df: pd.DataFrame) -> dict:
        """
        Extract OHLCV data as a dictionary of Series.
        
        Args:
            df: Normalized DataFrame
            
        Returns:
            Dict with keys: close, high, low, volume, open (volume/open may be None)
        """
        return {
            'close': self.get_series(df, 'close'),
            'high': self.get_series(df, 'high', 'close'),
            'low': self.get_series(df, 'low', 'close'),
            'volume': self.get_series(df, 'volume'),
            'open': self.get_series(df, 'open'),
        }
