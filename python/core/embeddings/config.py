"""
Embeddings Pipeline Configuration

Centralized configuration for the embeddings generation pipeline.
Supports environment-based overrides for dev/staging/prod.
"""
import os
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Any

logger = logging.getLogger(__name__)


def _env_int(key: str, default: int) -> int:
    """Get integer from environment variable with default."""
    value = os.getenv(key)
    if value is not None:
        try:
            return int(value)
        except ValueError:
            logger.warning(f"Invalid int for {key}: {value}, using default {default}")
    return default


def _env_float(key: str, default: float) -> float:
    """Get float from environment variable with default."""
    value = os.getenv(key)
    if value is not None:
        try:
            return float(value)
        except ValueError:
            logger.warning(f"Invalid float for {key}: {value}, using default {default}")
    return default


def _env_str(key: str, default: str) -> str:
    """Get string from environment variable with default."""
    return os.getenv(key, default)


def _env_bool(key: str, default: bool) -> bool:
    """Get boolean from environment variable with default."""
    value = os.getenv(key)
    if value is not None:
        return value.lower() in ('true', '1', 'yes', 'on')
    return default


@dataclass(frozen=True)
class EmbeddingPipelineConfig:
    """
    Configuration for the embeddings generation pipeline.
    
    All values can be overridden via environment variables prefixed with
    EMBEDDING_PIPELINE_ (e.g., EMBEDDING_PIPELINE_MAX_WORKERS=16).
    """
    
    # Processing parameters
    indicator_lookback_days: int = 300  # Days needed for 200-day MA etc.
    min_embedding_start_days: int = 252  # ~1 trading year before generating
    max_workers: int = 8  # Concurrent async workers
    batch_size: int = 1000  # Embeddings per ChromaDB batch
    
    # Rate limiting (for yfinance API)
    rate_limit_calls_per_second: float = 10.0
    rate_limit_burst: int = 5  # Allow short bursts
    
    # Retry configuration
    max_retries: int = 3
    retry_delay_base: float = 1.0  # Exponential backoff base
    retry_delay_max: float = 30.0  # Max delay between retries
    
    # Paths (relative to python/ directory)
    chroma_dir: str = "./chroma_data"
    cache_dir: str = "./.cache/embeddings"
    state_file: str = "./data/embedding_state.json"
    checkpoint_file: str = "./data/embedding_checkpoint.json"
    log_dir: str = "./logs"
    
    # Data validation thresholds
    max_nan_ratio: float = 0.05  # Max 5% NaN values allowed
    max_daily_return: float = 0.50  # Flag returns > 50%
    min_volume_days: int = 3  # Detect trading halts
    max_price_gap_days: int = 5  # Max calendar days between prices
    
    # Feature flags
    enable_cache: bool = True
    enable_validation: bool = True
    enable_metrics: bool = True
    enable_lineage: bool = True
    
    # Lineage tracking
    pipeline_version: str = "2.0.0"
    
    @classmethod
    def from_env(cls, env: Optional[str] = None) -> "EmbeddingPipelineConfig":
        """
        Create configuration from environment variables.
        
        Args:
            env: Environment name (dev/staging/prod). If None, reads from
                 PIPELINE_ENV environment variable, defaulting to 'dev'.
                 
        Returns:
            Configured EmbeddingPipelineConfig instance
        """
        env = env or os.getenv("PIPELINE_ENV", "dev")
        
        # Environment-specific defaults
        env_configs: Dict[str, Dict[str, Any]] = {
            "dev": {
                "max_workers": 4,
                "enable_cache": True,
                "chroma_dir": "./chroma_data_dev",
            },
            "staging": {
                "max_workers": 8,
                "enable_cache": True,
                "chroma_dir": "./chroma_data_staging",
            },
            "prod": {
                "max_workers": 16,
                "enable_cache": True,
                "chroma_dir": "./chroma_data",
                "rate_limit_calls_per_second": 5.0,  # More conservative in prod
            },
        }
        
        # Start with environment defaults
        config_dict = env_configs.get(env, {})
        
        # Override with explicit environment variables
        config_dict.update({
            "indicator_lookback_days": _env_int(
                "EMBEDDING_PIPELINE_LOOKBACK_DAYS",
                config_dict.get("indicator_lookback_days", 300)
            ),
            "min_embedding_start_days": _env_int(
                "EMBEDDING_PIPELINE_MIN_START_DAYS",
                config_dict.get("min_embedding_start_days", 252)
            ),
            "max_workers": _env_int(
                "EMBEDDING_PIPELINE_MAX_WORKERS",
                config_dict.get("max_workers", 8)
            ),
            "batch_size": _env_int(
                "EMBEDDING_PIPELINE_BATCH_SIZE",
                config_dict.get("batch_size", 1000)
            ),
            "rate_limit_calls_per_second": _env_float(
                "EMBEDDING_PIPELINE_RATE_LIMIT",
                config_dict.get("rate_limit_calls_per_second", 10.0)
            ),
            "chroma_dir": _env_str(
                "EMBEDDING_PIPELINE_CHROMA_DIR",
                config_dict.get("chroma_dir", "./chroma_data")
            ),
            "cache_dir": _env_str(
                "EMBEDDING_PIPELINE_CACHE_DIR",
                config_dict.get("cache_dir", "./.cache/embeddings")
            ),
            "state_file": _env_str(
                "EMBEDDING_PIPELINE_STATE_FILE",
                config_dict.get("state_file", "./data/embedding_state.json")
            ),
            "enable_cache": _env_bool(
                "EMBEDDING_PIPELINE_ENABLE_CACHE",
                config_dict.get("enable_cache", True)
            ),
            "enable_validation": _env_bool(
                "EMBEDDING_PIPELINE_ENABLE_VALIDATION",
                config_dict.get("enable_validation", True)
            ),
        })
        
        logger.info(f"Loaded config for environment: {env}")
        return cls(**config_dict)
    
    def get_chroma_path(self) -> Path:
        """Get absolute path to ChromaDB directory."""
        return Path(self.chroma_dir).resolve()
    
    def get_cache_path(self) -> Path:
        """Get absolute path to cache directory."""
        return Path(self.cache_dir).resolve()
    
    def get_state_path(self) -> Path:
        """Get absolute path to state file."""
        return Path(self.state_file).resolve()
    
    def get_checkpoint_path(self) -> Path:
        """Get absolute path to checkpoint file."""
        return Path(self.checkpoint_file).resolve()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary for logging/serialization."""
        return {
            "indicator_lookback_days": self.indicator_lookback_days,
            "min_embedding_start_days": self.min_embedding_start_days,
            "max_workers": self.max_workers,
            "batch_size": self.batch_size,
            "rate_limit_calls_per_second": self.rate_limit_calls_per_second,
            "chroma_dir": self.chroma_dir,
            "cache_dir": self.cache_dir,
            "state_file": self.state_file,
            "enable_cache": self.enable_cache,
            "enable_validation": self.enable_validation,
            "enable_metrics": self.enable_metrics,
            "pipeline_version": self.pipeline_version,
        }


# Default configuration instance
DEFAULT_CONFIG = EmbeddingPipelineConfig()
