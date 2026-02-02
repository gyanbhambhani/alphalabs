"""
Embeddings Pipeline Module

Provides infrastructure for generating and managing market state embeddings
for S&P 500 stocks with support for:
- Incremental updates
- Rate limiting
- Data validation
- Progress checkpointing
- Metrics collection
"""
import logging
import sys
from pathlib import Path
from typing import Optional

__version__ = "2.0.0"

# Module-level logger
logger = logging.getLogger(__name__)


def setup_logging(
    level: int = logging.INFO,
    log_file: Optional[str] = None,
    format_string: Optional[str] = None
) -> logging.Logger:
    """
    Configure logging for the embeddings pipeline.
    
    Args:
        level: Logging level (default: INFO)
        log_file: Optional file path for log output
        format_string: Custom format string (uses default if None)
        
    Returns:
        Configured root logger for the embeddings module
    """
    if format_string is None:
        format_string = (
            "%(asctime)s - %(name)s - %(levelname)s - "
            "[%(filename)s:%(lineno)d] - %(message)s"
        )
    
    formatter = logging.Formatter(format_string)
    
    # Get the embeddings module logger
    embeddings_logger = logging.getLogger("core.embeddings")
    embeddings_logger.setLevel(level)
    
    # Remove existing handlers to avoid duplicates
    embeddings_logger.handlers.clear()
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    embeddings_logger.addHandler(console_handler)
    
    # File handler (optional)
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        embeddings_logger.addHandler(file_handler)
        logger.info(f"Logging to file: {log_file}")
    
    return embeddings_logger


# Lazy imports to avoid circular dependencies
# These are imported when accessed from the module
def __getattr__(name):
    """Lazy import of submodules."""
    if name == "EmbeddingPipelineConfig":
        from core.embeddings.config import EmbeddingPipelineConfig
        return EmbeddingPipelineConfig
    elif name == "DataNormalizer":
        from core.embeddings.normalizer import DataNormalizer
        return DataNormalizer
    elif name == "NormalizationError":
        from core.embeddings.normalizer import NormalizationError
        return NormalizationError
    elif name == "DataValidator":
        from core.embeddings.validator import DataValidator
        return DataValidator
    elif name == "ValidationResult":
        from core.embeddings.validator import ValidationResult
        return ValidationResult
    elif name == "RateLimiter":
        from core.embeddings.rate_limiter import RateLimiter
        return RateLimiter
    elif name == "DataCache":
        from core.embeddings.cache import DataCache
        return DataCache
    elif name == "PipelineMetrics":
        from core.embeddings.metrics import PipelineMetrics
        return PipelineMetrics
    elif name == "SectorMetrics":
        from core.embeddings.metrics import SectorMetrics
        return SectorMetrics
    elif name == "LineageMetadata":
        from core.embeddings.metrics import LineageMetadata
        return LineageMetadata
    elif name == "StockProcessor":
        from core.embeddings.processor import StockProcessor
        return StockProcessor
    elif name == "ProcessingStrategy":
        from core.embeddings.processor import ProcessingStrategy
        return ProcessingStrategy
    elif name == "ProcessingResult":
        from core.embeddings.processor import ProcessingResult
        return ProcessingResult
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    # Version
    "__version__",
    # Logging
    "setup_logging",
    "logger",
    # Config
    "EmbeddingPipelineConfig",
    # Data handling
    "DataNormalizer",
    "NormalizationError",
    "DataValidator",
    "ValidationResult",
    # Infrastructure
    "RateLimiter",
    "DataCache",
    "PipelineMetrics",
    # Processing
    "StockProcessor",
    "ProcessingStrategy",
    "ProcessingResult",
]
