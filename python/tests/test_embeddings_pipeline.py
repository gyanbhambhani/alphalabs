"""
Unit Tests for Embeddings Pipeline

Tests for:
- DataNormalizer
- DataValidator  
- RateLimiter
- DataCache
- StockProcessor
- CheckpointManager
"""
import pytest
import time
import tempfile
import json
from pathlib import Path
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock

import pandas as pd
import numpy as np

# Add parent to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.embeddings.config import EmbeddingPipelineConfig
from core.embeddings.normalizer import DataNormalizer, NormalizationError
from core.embeddings.validator import (
    DataValidator,
    ValidationResult,
    calculate_forward_return,
    calculate_forward_returns
)
from core.embeddings.rate_limiter import RateLimiter, RetryWithBackoff
from core.embeddings.cache import DataCache
from core.embeddings.metrics import PipelineMetrics, SectorMetrics, LineageMetadata


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def sample_df():
    """Create a sample stock DataFrame."""
    dates = pd.date_range(start='2023-01-01', periods=400, freq='B')
    np.random.seed(42)
    
    close = 100 + np.cumsum(np.random.randn(400) * 2)
    high = close + np.random.rand(400) * 2
    low = close - np.random.rand(400) * 2
    volume = np.random.randint(1000000, 10000000, 400)
    
    return pd.DataFrame({
        'Close': close,
        'High': high,
        'Low': low,
        'Volume': volume
    }, index=dates)


@pytest.fixture
def multiindex_df(sample_df):
    """Create a DataFrame with MultiIndex columns (yfinance style)."""
    df = sample_df.copy()
    df.columns = pd.MultiIndex.from_tuples([
        ('Close', 'AAPL'),
        ('High', 'AAPL'),
        ('Low', 'AAPL'),
        ('Volume', 'AAPL')
    ])
    return df


@pytest.fixture
def config():
    """Create a test configuration."""
    return EmbeddingPipelineConfig(
        indicator_lookback_days=50,
        min_embedding_start_days=50,
        max_workers=2,
        batch_size=100,
        rate_limit_calls_per_second=100.0,
        max_nan_ratio=0.1,
        max_daily_return=0.5
    )


@pytest.fixture
def temp_cache_dir():
    """Create a temporary directory for cache testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


# =============================================================================
# DataNormalizer Tests
# =============================================================================

class TestDataNormalizer:
    """Tests for DataNormalizer."""
    
    def test_normalize_standard_df(self, sample_df):
        """Test normalizing a standard DataFrame."""
        normalizer = DataNormalizer()
        result = normalizer.normalize(sample_df, 'AAPL')
        
        assert 'close' in result.columns
        assert 'high' in result.columns
        assert 'low' in result.columns
        assert 'volume' in result.columns
        assert len(result) == len(sample_df)
    
    def test_normalize_multiindex(self, multiindex_df):
        """Test normalizing MultiIndex columns."""
        normalizer = DataNormalizer()
        result = normalizer.normalize(multiindex_df, 'AAPL')
        
        assert not isinstance(result.columns, pd.MultiIndex)
        assert 'close' in result.columns
    
    def test_normalize_missing_columns(self):
        """Test that missing required columns raise error."""
        normalizer = DataNormalizer()
        df = pd.DataFrame({'price': [1, 2, 3]})
        
        with pytest.raises(NormalizationError) as exc_info:
            normalizer.normalize(df, 'AAPL')
        
        assert 'close' in str(exc_info.value).lower()
    
    def test_normalize_empty_df(self):
        """Test that empty DataFrame raises error."""
        normalizer = DataNormalizer()
        df = pd.DataFrame()
        
        with pytest.raises(NormalizationError):
            normalizer.normalize(df, 'AAPL')
    
    def test_get_series(self, sample_df):
        """Test extracting series from DataFrame."""
        normalizer = DataNormalizer()
        df = normalizer.normalize(sample_df, 'AAPL')
        
        close = normalizer.get_series(df, 'close')
        assert isinstance(close, pd.Series)
        assert len(close) == len(df)
        
        # Test fallback
        missing = normalizer.get_series(df, 'nonexistent', 'close')
        assert isinstance(missing, pd.Series)
    
    def test_extract_ohlcv(self, sample_df):
        """Test extracting OHLCV data."""
        normalizer = DataNormalizer()
        df = normalizer.normalize(sample_df, 'AAPL')
        
        ohlcv = normalizer.extract_ohlcv(df)
        
        assert 'close' in ohlcv
        assert 'high' in ohlcv
        assert 'low' in ohlcv
        assert 'volume' in ohlcv
        assert ohlcv['close'] is not None


# =============================================================================
# DataValidator Tests
# =============================================================================

class TestDataValidator:
    """Tests for DataValidator."""
    
    def test_validate_good_data(self, sample_df, config):
        """Test validation passes for good data."""
        normalizer = DataNormalizer()
        df = normalizer.normalize(sample_df, 'AAPL')
        
        validator = DataValidator(config)
        result = validator.validate(df, 'AAPL')
        
        assert result.is_valid
        assert len(result.issues) == 0
    
    def test_validate_insufficient_data(self, config):
        """Test validation fails for insufficient data."""
        df = pd.DataFrame({
            'close': [1, 2, 3],
            'high': [1.1, 2.1, 3.1],
            'low': [0.9, 1.9, 2.9]
        })
        
        validator = DataValidator(config)
        result = validator.validate(df, 'AAPL', min_rows=100)
        
        assert not result.is_valid
        assert any('Insufficient' in issue for issue in result.issues)
    
    def test_validate_nan_values(self, sample_df, config):
        """Test validation detects NaN values."""
        normalizer = DataNormalizer()
        df = normalizer.normalize(sample_df, 'AAPL')
        
        # Add too many NaN values
        df.loc[df.index[:50], 'close'] = np.nan
        
        validator = DataValidator(config)
        result = validator.validate(df, 'AAPL')
        
        assert not result.is_valid or len(result.warnings) > 0
    
    def test_validate_zero_prices(self, sample_df, config):
        """Test validation detects zero prices."""
        normalizer = DataNormalizer()
        df = normalizer.normalize(sample_df, 'AAPL')
        
        # Add zero prices
        df.loc[df.index[100], 'close'] = 0
        
        validator = DataValidator(config)
        result = validator.validate(df, 'AAPL')
        
        assert not result.is_valid
        assert any('zero' in issue.lower() for issue in result.issues)
    
    def test_calculate_forward_return(self):
        """Test forward return calculation."""
        assert calculate_forward_return(100, 110) == pytest.approx(0.1)
        assert calculate_forward_return(100, 90) == pytest.approx(-0.1)
        assert calculate_forward_return(0, 100) is None
        assert calculate_forward_return(100, 0) is None
        assert calculate_forward_return(np.nan, 100) is None


# =============================================================================
# RateLimiter Tests
# =============================================================================

class TestRateLimiter:
    """Tests for RateLimiter."""
    
    def test_rate_limiting(self):
        """Test that rate limiter actually limits."""
        limiter = RateLimiter(calls_per_second=10.0, burst_size=2)
        
        # First burst should be fast
        start = time.time()
        limiter.acquire()
        limiter.acquire()
        burst_time = time.time() - start
        
        # Burst should be quick
        assert burst_time < 0.1
        
        # Next call should wait
        start = time.time()
        limiter.acquire()
        wait_time = time.time() - start
        
        # Should have waited ~0.1s
        assert wait_time >= 0.05
    
    def test_try_acquire(self):
        """Test non-blocking acquire."""
        limiter = RateLimiter(calls_per_second=1.0, burst_size=1)
        
        # First should succeed
        assert limiter.try_acquire()
        
        # Second should fail (no tokens)
        assert not limiter.try_acquire()
    
    def test_decorator(self):
        """Test rate limiter as decorator."""
        limiter = RateLimiter(calls_per_second=100.0, burst_size=10)
        
        @limiter
        def my_function():
            return "result"
        
        result = my_function()
        assert result == "result"


class TestRetryWithBackoff:
    """Tests for RetryWithBackoff."""
    
    def test_successful_call(self):
        """Test that successful calls don't retry."""
        call_count = 0
        
        @RetryWithBackoff(max_retries=3)
        def successful_func():
            nonlocal call_count
            call_count += 1
            return "success"
        
        result = successful_func()
        assert result == "success"
        assert call_count == 1
    
    def test_retry_on_failure(self):
        """Test that failures are retried."""
        call_count = 0
        
        @RetryWithBackoff(max_retries=3, base_delay=0.01)
        def failing_then_succeeding():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ValueError("Not yet")
            return "success"
        
        result = failing_then_succeeding()
        assert result == "success"
        assert call_count == 3
    
    def test_max_retries_exceeded(self):
        """Test that max retries raises exception."""
        @RetryWithBackoff(max_retries=2, base_delay=0.01)
        def always_fails():
            raise ValueError("Always fails")
        
        with pytest.raises(ValueError):
            always_fails()


# =============================================================================
# DataCache Tests
# =============================================================================

class TestDataCache:
    """Tests for DataCache."""
    
    def test_put_and_get(self, sample_df, temp_cache_dir):
        """Test storing and retrieving data."""
        cache = DataCache(cache_dir=temp_cache_dir)
        
        cache.put('AAPL', '2023-01-01', '2023-12-31', sample_df)
        
        result = cache.get('AAPL', '2023-01-01', '2023-12-31')
        
        assert result is not None
        assert len(result) == len(sample_df)
    
    def test_cache_miss(self, temp_cache_dir):
        """Test cache miss returns None."""
        cache = DataCache(cache_dir=temp_cache_dir)
        
        result = cache.get('AAPL', '2023-01-01', '2023-12-31')
        
        assert result is None
    
    def test_invalidate(self, sample_df, temp_cache_dir):
        """Test cache invalidation."""
        cache = DataCache(cache_dir=temp_cache_dir)
        
        cache.put('AAPL', '2023-01-01', '2023-12-31', sample_df)
        cache.invalidate('AAPL', '2023-01-01', '2023-12-31')
        
        result = cache.get('AAPL', '2023-01-01', '2023-12-31')
        
        assert result is None
    
    def test_disabled_cache(self, sample_df, temp_cache_dir):
        """Test that disabled cache is no-op."""
        cache = DataCache(cache_dir=temp_cache_dir, enabled=False)
        
        assert not cache.put('AAPL', '2023-01-01', '2023-12-31', sample_df)
        assert cache.get('AAPL', '2023-01-01', '2023-12-31') is None
    
    def test_stats(self, sample_df, temp_cache_dir):
        """Test cache statistics."""
        cache = DataCache(cache_dir=temp_cache_dir)
        
        cache.put('AAPL', '2023-01-01', '2023-12-31', sample_df)
        
        stats = cache.stats()
        
        assert stats['enabled']
        assert stats['entry_count'] == 1


# =============================================================================
# PipelineMetrics Tests
# =============================================================================

class TestPipelineMetrics:
    """Tests for PipelineMetrics."""
    
    def test_record_success(self):
        """Test recording successful processing."""
        metrics = PipelineMetrics()
        
        metrics.record_success('AAPL', 'new', 100, 500)
        
        assert metrics.stocks_processed == 1
        assert metrics.embeddings_created == 100
        assert metrics.status_counts['new'] == 1
    
    def test_record_failure(self):
        """Test recording failed processing."""
        metrics = PipelineMetrics()
        
        metrics.record_failure('AAPL', 'ValueError', 'Test error')
        
        assert metrics.stocks_processed == 1
        assert metrics.status_counts['failed'] == 1
        assert 'ValueError' in metrics.failures_by_type
    
    def test_sector_metrics(self):
        """Test per-sector metrics."""
        metrics = PipelineMetrics()
        
        sector = metrics.start_sector('Technology', ['AAPL', 'MSFT', 'GOOGL'])
        
        assert sector.total == 3
        assert metrics.sector_metrics['Technology'] == sector
    
    def test_summary(self):
        """Test metrics summary generation."""
        metrics = PipelineMetrics()
        
        metrics.record_success('AAPL', 'new', 100, 500)
        metrics.record_success('MSFT', 'current', 0, 50)
        metrics.record_failure('GOOGL', 'ValueError', 'Error')
        
        summary = metrics.summary()
        
        assert summary['stocks_processed'] == 3
        assert summary['embeddings_created'] == 100
        assert summary['status_breakdown']['new'] == 1
        assert summary['status_breakdown']['current'] == 1
        assert summary['status_breakdown']['failed'] == 1


class TestSectorMetrics:
    """Tests for SectorMetrics."""
    
    def test_record_result(self):
        """Test recording sector results."""
        sector = SectorMetrics(name='Technology', symbols=['AAPL', 'MSFT'])
        
        sector.record_result('new', 100, 500)
        sector.record_result('current', 0, 50)
        
        assert sector.completed == 2
        assert sector.new == 1
        assert sector.current == 1
        assert sector.embeddings == 100
        assert len(sector.processing_times_ms) == 2


class TestLineageMetadata:
    """Tests for LineageMetadata."""
    
    def test_create(self):
        """Test lineage metadata creation."""
        lineage = LineageMetadata.create(
            pipeline_version='2.0.0',
            encoder_version='1.0.0',
            lookback_days=300
        )
        
        assert lineage.pipeline_version == '2.0.0'
        assert lineage.encoder_version == '1.0.0'
        assert lineage.lookback_days == 300
        assert lineage.generated_at is not None
    
    def test_to_dict(self):
        """Test converting to dictionary."""
        lineage = LineageMetadata.create(
            pipeline_version='2.0.0',
            encoder_version='1.0.0',
            lookback_days=300
        )
        
        data = lineage.to_dict()
        
        assert data['pipeline_version'] == '2.0.0'
        assert 'generated_at' in data


# =============================================================================
# Configuration Tests
# =============================================================================

class TestEmbeddingPipelineConfig:
    """Tests for EmbeddingPipelineConfig."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = EmbeddingPipelineConfig()
        
        assert config.indicator_lookback_days == 300
        assert config.min_embedding_start_days == 252
        assert config.max_workers == 8
    
    def test_from_env(self, monkeypatch):
        """Test configuration from environment."""
        monkeypatch.setenv('PIPELINE_ENV', 'dev')
        
        config = EmbeddingPipelineConfig.from_env()
        
        # Dev config has different defaults
        assert config.max_workers == 4
    
    def test_to_dict(self):
        """Test configuration serialization."""
        config = EmbeddingPipelineConfig()
        
        data = config.to_dict()
        
        assert 'indicator_lookback_days' in data
        assert 'max_workers' in data


# =============================================================================
# Run tests
# =============================================================================

if __name__ == '__main__':
    pytest.main([__file__, '-v'])
