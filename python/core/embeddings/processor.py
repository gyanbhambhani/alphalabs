"""
Stock Processor

Unified stock processing for embedding generation.
Consolidates the previously duplicated functions into a single class.
"""
import logging
import time
import warnings
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Optional, List, Tuple, Any

import pandas as pd
import yfinance as yf

from core.embeddings.config import EmbeddingPipelineConfig, DEFAULT_CONFIG
from core.embeddings.normalizer import DataNormalizer, NormalizationError
from core.embeddings.validator import (
    DataValidator,
    ValidationResult,
    calculate_forward_returns
)
from core.embeddings.rate_limiter import RateLimiter
from core.embeddings.cache import DataCache
from core.embeddings.metrics import LineageMetadata
from core.semantic.encoder import MarketStateEncoder, MarketState
from core.semantic.vector_db import VectorDatabase

# Suppress yfinance warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', message='.*YF.download.*')

logger = logging.getLogger(__name__)


class ProcessingStrategy(Enum):
    """Strategy for processing a stock."""
    INCREMENTAL = "incremental"  # Check existing, add new dates only
    FULL_REFRESH = "full"  # Delete and regenerate all
    FROM_DATAFRAME = "from_df"  # Use provided DataFrame (no download)


@dataclass
class ProcessingResult:
    """Result of processing a single stock."""
    symbol: str
    success: bool
    embeddings_added: int
    status: str  # 'new', 'updated', 'current', 'failed'
    last_date: Optional[str] = None
    error: Optional[str] = None
    error_type: Optional[str] = None
    duration_ms: int = 0
    validation_warnings: List[str] = field(default_factory=list)
    
    def __bool__(self) -> bool:
        return self.success


class StockProcessor:
    """
    Unified processor for generating stock embeddings.
    
    Features:
    - Incremental updates (only add new dates)
    - Full refresh capability
    - Rate-limited API calls
    - Data validation
    - Caching support
    - Idempotent operations
    """
    
    def __init__(
        self,
        config: EmbeddingPipelineConfig = None,
        encoder: MarketStateEncoder = None,
        rate_limiter: RateLimiter = None,
        cache: DataCache = None,
        dry_run: bool = False
    ):
        """
        Initialize the processor.
        
        Args:
            config: Pipeline configuration
            encoder: MarketStateEncoder instance (created if None)
            rate_limiter: Rate limiter for API calls
            cache: Data cache for downloaded data
            dry_run: If True, don't actually write to DB
        """
        self.config = config or DEFAULT_CONFIG
        self.encoder = encoder or MarketStateEncoder()
        self.rate_limiter = rate_limiter or RateLimiter(
            calls_per_second=self.config.rate_limit_calls_per_second
        )
        self.cache = cache
        self.dry_run = dry_run
        
        # Initialize helpers
        self.normalizer = DataNormalizer()
        self.validator = DataValidator(self.config)
        
        # Track lineage
        self._lineage = LineageMetadata.create(
            pipeline_version=self.config.pipeline_version,
            encoder_version=getattr(self.encoder, 'VERSION', '1.0.0'),
            lookback_days=self.config.indicator_lookback_days,
            config=self.config
        )
        
        logger.debug(f"StockProcessor initialized (dry_run={dry_run})")
    
    def process(
        self,
        symbol: str,
        strategy: ProcessingStrategy = ProcessingStrategy.INCREMENTAL,
        data: Optional[pd.DataFrame] = None,
        persist_directory: Optional[str] = None
    ) -> ProcessingResult:
        """
        Process a single stock - the main entry point.
        
        Args:
            symbol: Stock ticker symbol
            strategy: Processing strategy to use
            data: Pre-downloaded DataFrame (for FROM_DATAFRAME strategy)
            persist_directory: Override ChromaDB directory
            
        Returns:
            ProcessingResult with status and statistics
        """
        start_time = time.time()
        persist_dir = persist_directory or self.config.chroma_dir
        
        try:
            # Initialize vector DB
            vector_db = VectorDatabase(
                persist_directory=persist_dir,
                symbol=symbol
            )
            
            # Determine last embedding date
            last_embedding_date = None
            existing_count = 0
            
            if strategy == ProcessingStrategy.FULL_REFRESH:
                # Delete existing embeddings
                existing_count = vector_db.get_count()
                if existing_count > 0 and not self.dry_run:
                    vector_db.delete_all()
                    logger.info(f"{symbol}: Deleted {existing_count} existing embeddings")
            elif strategy == ProcessingStrategy.INCREMENTAL:
                # Check existing embeddings
                existing_count = vector_db.get_count()
                if existing_count > 0:
                    try:
                        _, last_embedding_date = vector_db.get_date_range()
                    except Exception as e:
                        logger.warning(f"{symbol}: Could not get date range: {e}")
            
            # Get current trading date
            current_date = self._get_current_trading_date()
            
            # Check if already up to date
            if (
                strategy == ProcessingStrategy.INCREMENTAL
                and last_embedding_date
                and last_embedding_date >= current_date
            ):
                duration_ms = int((time.time() - start_time) * 1000)
                return ProcessingResult(
                    symbol=symbol,
                    success=True,
                    embeddings_added=0,
                    status='current',
                    last_date=last_embedding_date,
                    duration_ms=duration_ms
                )
            
            # Get data (download or use provided)
            if strategy == ProcessingStrategy.FROM_DATAFRAME:
                if data is None:
                    raise ValueError("FROM_DATAFRAME strategy requires data parameter")
                df = data
            else:
                df = self._fetch_data(symbol, last_embedding_date, current_date)
            
            if df is None or len(df) < self.config.indicator_lookback_days:
                # Not enough data, but might still be "current"
                if existing_count > 0 and last_embedding_date:
                    duration_ms = int((time.time() - start_time) * 1000)
                    return ProcessingResult(
                        symbol=symbol,
                        success=True,
                        embeddings_added=0,
                        status='current',
                        last_date=last_embedding_date,
                        duration_ms=duration_ms
                    )
                raise ValueError(
                    f"Insufficient data: {len(df) if df is not None else 0} rows"
                )
            
            # Normalize data
            df = self.normalizer.normalize(df, symbol)
            
            # Validate data
            validation_warnings = []
            if self.config.enable_validation:
                validation = self.validator.validate(df, symbol)
                validation_warnings = validation.warnings
                if not validation.is_valid:
                    raise ValueError(f"Validation failed: {validation.issues}")
            
            # Get latest date in downloaded data
            date_strings = [d.strftime('%Y-%m-%d') for d in df.index]
            latest_data_date = date_strings[-1] if date_strings else None
            
            # Check if data is older than what we have
            if (
                last_embedding_date
                and latest_data_date
                and latest_data_date <= last_embedding_date
            ):
                duration_ms = int((time.time() - start_time) * 1000)
                return ProcessingResult(
                    symbol=symbol,
                    success=True,
                    embeddings_added=0,
                    status='current',
                    last_date=last_embedding_date,
                    duration_ms=duration_ms,
                    validation_warnings=validation_warnings
                )
            
            # Generate embeddings
            states = self._generate_embeddings(
                df=df,
                symbol=symbol,
                date_strings=date_strings,
                last_embedding_date=last_embedding_date,
                strategy=strategy
            )
            
            if not states:
                duration_ms = int((time.time() - start_time) * 1000)
                return ProcessingResult(
                    symbol=symbol,
                    success=True,
                    embeddings_added=0,
                    status='current',
                    last_date=last_embedding_date or latest_data_date,
                    duration_ms=duration_ms,
                    validation_warnings=validation_warnings
                )
            
            # Store embeddings (idempotent)
            if not self.dry_run:
                added_count = self._add_embeddings_idempotent(vector_db, states)
            else:
                added_count = len(states)
                logger.info(f"{symbol}: Dry run - would add {added_count} embeddings")
            
            new_last_date = states[-1].date
            status = 'updated' if last_embedding_date else 'new'
            duration_ms = int((time.time() - start_time) * 1000)
            
            return ProcessingResult(
                symbol=symbol,
                success=True,
                embeddings_added=added_count,
                status=status,
                last_date=new_last_date,
                duration_ms=duration_ms,
                validation_warnings=validation_warnings
            )
            
        except NormalizationError as e:
            duration_ms = int((time.time() - start_time) * 1000)
            logger.warning(f"{symbol}: Normalization error - {e}")
            return ProcessingResult(
                symbol=symbol,
                success=False,
                embeddings_added=0,
                status='failed',
                error=str(e),
                error_type='NormalizationError',
                duration_ms=duration_ms
            )
        except Exception as e:
            duration_ms = int((time.time() - start_time) * 1000)
            error_type = type(e).__name__
            logger.error(f"{symbol}: {error_type} - {e}")
            return ProcessingResult(
                symbol=symbol,
                success=False,
                embeddings_added=0,
                status='failed',
                error=str(e)[:200],
                error_type=error_type,
                duration_ms=duration_ms
            )
    
    def _get_current_trading_date(self) -> str:
        """Get current trading date (adjusted for weekends)."""
        today = datetime.now()
        
        if today.weekday() == 5:  # Saturday
            today = today - timedelta(days=1)
        elif today.weekday() == 6:  # Sunday
            today = today - timedelta(days=2)
        
        return today.strftime('%Y-%m-%d')
    
    def _fetch_data(
        self,
        symbol: str,
        last_embedding_date: Optional[str],
        current_date: str
    ) -> Optional[pd.DataFrame]:
        """Fetch data with caching and rate limiting."""
        
        # Calculate date range
        if last_embedding_date:
            # Incremental: need lookback days + new days
            start_date = (
                datetime.strptime(last_embedding_date, '%Y-%m-%d')
                - timedelta(days=self.config.indicator_lookback_days + 10)
            ).strftime('%Y-%m-%d')
        else:
            # Full: from 2020
            start_date = "2020-01-01"
        
        # Try cache first
        if self.cache:
            cached = self.cache.get(symbol, start_date, current_date)
            if cached is not None:
                logger.debug(f"{symbol}: Using cached data")
                return cached
        
        # Rate limit API call
        self.rate_limiter.acquire()
        
        # Download from yfinance
        try:
            df = yf.download(
                symbol,
                start=start_date,
                progress=False,
                threads=False
            )
            
            if df is not None and len(df) > 0:
                # Cache the result
                if self.cache:
                    actual_start = df.index[0].strftime('%Y-%m-%d')
                    actual_end = df.index[-1].strftime('%Y-%m-%d')
                    self.cache.put(symbol, actual_start, actual_end, df)
                
                return df
            
            return None
            
        except Exception as e:
            logger.warning(f"{symbol}: Download failed - {e}")
            return None
    
    def _generate_embeddings(
        self,
        df: pd.DataFrame,
        symbol: str,
        date_strings: List[str],
        last_embedding_date: Optional[str],
        strategy: ProcessingStrategy
    ) -> List[MarketState]:
        """Generate embeddings for new dates."""
        
        # Extract OHLCV data
        ohlcv = self.normalizer.extract_ohlcv(df)
        close = ohlcv['close']
        high = ohlcv['high']
        low = ohlcv['low']
        volume = ohlcv['volume']
        
        # Determine start index
        if last_embedding_date and strategy != ProcessingStrategy.FULL_REFRESH:
            start_idx = None
            for i, d in enumerate(date_strings):
                if d > last_embedding_date and i >= self.config.indicator_lookback_days:
                    start_idx = i
                    break
            
            if start_idx is None:
                return []
        else:
            start_idx = self.config.indicator_lookback_days
        
        # Generate embeddings
        states = []
        for i in range(start_idx, len(df)):
            date_str = date_strings[i]
            
            # Skip if already have this date
            if last_embedding_date and date_str <= last_embedding_date:
                continue
            
            # Calculate window indices
            window_start = i - self.config.indicator_lookback_days
            window_end = i + 1
            
            # Encode market state
            market_state = self.encoder.encode(
                date=date_str,
                close=close.iloc[window_start:window_end],
                high=high.iloc[window_start:window_end] if high is not None else None,
                low=low.iloc[window_start:window_end] if low is not None else None,
                volume=volume.iloc[window_start:window_end] if volume is not None else None
            )
            
            # Add forward returns
            forward_returns = calculate_forward_returns(df, i)
            market_state.metadata.update(forward_returns)
            
            # Add symbol and lineage
            market_state.metadata['symbol'] = symbol
            if self.config.enable_lineage:
                market_state.metadata.update(self._lineage.to_dict())
            
            states.append(market_state)
        
        return states
    
    def _add_embeddings_idempotent(
        self,
        vector_db: VectorDatabase,
        states: List[MarketState]
    ) -> int:
        """
        Add embeddings idempotently - only add dates not already in DB.
        
        Args:
            vector_db: VectorDatabase instance
            states: List of MarketState objects to add
            
        Returns:
            Number of embeddings actually added
        """
        # Get existing dates
        try:
            existing_data = vector_db.collection.get(include=[])
            existing_dates = set(existing_data['ids']) if existing_data['ids'] else set()
        except Exception:
            existing_dates = set()
        
        # Filter to new states only
        new_states = [s for s in states if s.date not in existing_dates]
        
        if new_states:
            vector_db.add_batch(new_states, batch_size=self.config.batch_size)
            logger.debug(
                f"Added {len(new_states)} new embeddings "
                f"(skipped {len(states) - len(new_states)} existing)"
            )
        
        return len(new_states)
    
    def process_sync(
        self,
        symbol: str,
        strategy: ProcessingStrategy = ProcessingStrategy.INCREMENTAL,
        persist_directory: Optional[str] = None
    ) -> ProcessingResult:
        """
        Synchronous version for use with thread pools.
        
        This is a convenience wrapper for asyncio.to_thread() usage.
        """
        return self.process(
            symbol=symbol,
            strategy=strategy,
            persist_directory=persist_directory
        )
