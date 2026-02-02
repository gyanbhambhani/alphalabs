"""
Data Cache

Local file-based cache for downloaded stock data to avoid redundant API calls.
"""
import hashlib
import json
import logging
import pickle
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, Any

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """Metadata for a cached item."""
    symbol: str
    start_date: str
    end_date: str
    cached_at: float
    row_count: int
    file_path: str
    
    def is_expired(self, max_age_hours: float = 24.0) -> bool:
        """Check if cache entry has expired."""
        age_hours = (time.time() - self.cached_at) / 3600
        return age_hours > max_age_hours
    
    def covers_range(self, start_date: str, end_date: str) -> bool:
        """Check if this cache entry covers the requested date range."""
        return self.start_date <= start_date and self.end_date >= end_date


class DataCache:
    """
    File-based cache for stock data.
    
    Features:
    - Pickle-based storage for DataFrames
    - Metadata index for fast lookups
    - Automatic expiration
    - Thread-safe file operations
    """
    
    METADATA_FILE = "cache_index.json"
    
    def __init__(
        self,
        cache_dir: str = "./.cache/embeddings",
        max_age_hours: float = 24.0,
        enabled: bool = True
    ):
        """
        Initialize the cache.
        
        Args:
            cache_dir: Directory to store cached files
            max_age_hours: Maximum age before cache entry expires
            enabled: If False, all cache operations are no-ops
        """
        self.cache_dir = Path(cache_dir)
        self.max_age_hours = max_age_hours
        self.enabled = enabled
        self._metadata: Dict[str, CacheEntry] = {}
        
        if self.enabled:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            self._load_metadata()
            logger.debug(
                f"DataCache initialized at {self.cache_dir} "
                f"with {len(self._metadata)} entries"
            )
    
    def _load_metadata(self) -> None:
        """Load cache metadata from disk."""
        metadata_path = self.cache_dir / self.METADATA_FILE
        if metadata_path.exists():
            try:
                with open(metadata_path, 'r') as f:
                    data = json.load(f)
                    self._metadata = {
                        k: CacheEntry(**v) for k, v in data.items()
                    }
            except Exception as e:
                logger.warning(f"Failed to load cache metadata: {e}")
                self._metadata = {}
    
    def _save_metadata(self) -> None:
        """Save cache metadata to disk."""
        metadata_path = self.cache_dir / self.METADATA_FILE
        try:
            data = {
                k: {
                    'symbol': v.symbol,
                    'start_date': v.start_date,
                    'end_date': v.end_date,
                    'cached_at': v.cached_at,
                    'row_count': v.row_count,
                    'file_path': v.file_path,
                }
                for k, v in self._metadata.items()
            }
            with open(metadata_path, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save cache metadata: {e}")
    
    def _get_cache_key(self, symbol: str, start_date: str, end_date: str) -> str:
        """Generate cache key for a query."""
        return f"{symbol}_{start_date}_{end_date}"
    
    def _get_file_path(self, symbol: str, start_date: str, end_date: str) -> Path:
        """Generate file path for cached data."""
        key = self._get_cache_key(symbol, start_date, end_date)
        hash_suffix = hashlib.md5(key.encode()).hexdigest()[:8]
        return self.cache_dir / f"{symbol}_{hash_suffix}.pkl"
    
    def get(
        self,
        symbol: str,
        start_date: str,
        end_date: str
    ) -> Optional[pd.DataFrame]:
        """
        Get cached data if available and fresh.
        
        Args:
            symbol: Stock symbol
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            
        Returns:
            Cached DataFrame or None if not available/expired
        """
        if not self.enabled:
            return None
        
        key = self._get_cache_key(symbol, start_date, end_date)
        entry = self._metadata.get(key)
        
        if entry is None:
            logger.debug(f"Cache miss for {symbol} ({start_date} to {end_date})")
            return None
        
        # Check expiration
        if entry.is_expired(self.max_age_hours):
            logger.debug(f"Cache expired for {symbol}")
            self.invalidate(symbol, start_date, end_date)
            return None
        
        # Check if cached range covers requested range
        if not entry.covers_range(start_date, end_date):
            logger.debug(f"Cache range insufficient for {symbol}")
            return None
        
        # Load from disk
        try:
            file_path = Path(entry.file_path)
            if file_path.exists():
                with open(file_path, 'rb') as f:
                    df = pickle.load(f)
                logger.debug(f"Cache hit for {symbol} ({len(df)} rows)")
                return df
            else:
                logger.debug(f"Cache file missing for {symbol}")
                self._remove_entry(key)
                return None
        except Exception as e:
            logger.warning(f"Failed to load cached data for {symbol}: {e}")
            self._remove_entry(key)
            return None
    
    def put(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        data: pd.DataFrame
    ) -> bool:
        """
        Cache downloaded data.
        
        Args:
            symbol: Stock symbol
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            data: DataFrame to cache
            
        Returns:
            True if successfully cached
        """
        if not self.enabled:
            return False
        
        if data is None or len(data) == 0:
            return False
        
        key = self._get_cache_key(symbol, start_date, end_date)
        file_path = self._get_file_path(symbol, start_date, end_date)
        
        try:
            # Save DataFrame
            with open(file_path, 'wb') as f:
                pickle.dump(data, f)
            
            # Update metadata
            self._metadata[key] = CacheEntry(
                symbol=symbol,
                start_date=start_date,
                end_date=end_date,
                cached_at=time.time(),
                row_count=len(data),
                file_path=str(file_path)
            )
            self._save_metadata()
            
            logger.debug(f"Cached {symbol} ({len(data)} rows)")
            return True
            
        except Exception as e:
            logger.warning(f"Failed to cache data for {symbol}: {e}")
            return False
    
    def invalidate(
        self,
        symbol: str,
        start_date: str = None,
        end_date: str = None
    ) -> bool:
        """
        Remove cached data for a symbol.
        
        Args:
            symbol: Stock symbol
            start_date: Optional start date (if None, removes all for symbol)
            end_date: Optional end date
            
        Returns:
            True if something was removed
        """
        if not self.enabled:
            return False
        
        removed = False
        
        if start_date and end_date:
            # Remove specific entry
            key = self._get_cache_key(symbol, start_date, end_date)
            if key in self._metadata:
                removed = self._remove_entry(key)
        else:
            # Remove all entries for symbol
            keys_to_remove = [
                k for k, v in self._metadata.items()
                if v.symbol == symbol
            ]
            for key in keys_to_remove:
                self._remove_entry(key)
                removed = True
        
        if removed:
            self._save_metadata()
        
        return removed
    
    def _remove_entry(self, key: str) -> bool:
        """Remove a cache entry and its file."""
        entry = self._metadata.pop(key, None)
        if entry:
            try:
                file_path = Path(entry.file_path)
                if file_path.exists():
                    file_path.unlink()
            except Exception as e:
                logger.warning(f"Failed to remove cache file: {e}")
            return True
        return False
    
    def clear(self) -> int:
        """
        Clear all cached data.
        
        Returns:
            Number of entries cleared
        """
        if not self.enabled:
            return 0
        
        count = len(self._metadata)
        
        for key in list(self._metadata.keys()):
            self._remove_entry(key)
        
        self._metadata = {}
        self._save_metadata()
        
        logger.info(f"Cleared {count} cache entries")
        return count
    
    def cleanup_expired(self) -> int:
        """
        Remove expired cache entries.
        
        Returns:
            Number of entries removed
        """
        if not self.enabled:
            return 0
        
        expired_keys = [
            k for k, v in self._metadata.items()
            if v.is_expired(self.max_age_hours)
        ]
        
        for key in expired_keys:
            self._remove_entry(key)
        
        if expired_keys:
            self._save_metadata()
            logger.info(f"Cleaned up {len(expired_keys)} expired cache entries")
        
        return len(expired_keys)
    
    def stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        if not self.enabled:
            return {'enabled': False}
        
        total_size = 0
        for entry in self._metadata.values():
            try:
                path = Path(entry.file_path)
                if path.exists():
                    total_size += path.stat().st_size
            except Exception:
                pass
        
        return {
            'enabled': True,
            'entry_count': len(self._metadata),
            'total_size_mb': total_size / (1024 * 1024),
            'cache_dir': str(self.cache_dir),
            'max_age_hours': self.max_age_hours,
        }
