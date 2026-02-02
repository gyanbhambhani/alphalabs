"""
Pipeline Metrics

Observability and metrics collection for the embeddings pipeline.
"""
import json
import logging
import statistics
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)


@dataclass
class SectorMetrics:
    """Metrics for a single sector."""
    name: str
    symbols: List[str] = field(default_factory=list)
    completed: int = 0
    current: int = 0  # Already up to date
    updated: int = 0  # Incremental update
    new: int = 0  # Fresh generation
    failed: int = 0
    embeddings: int = 0
    processing_times_ms: List[int] = field(default_factory=list)
    start_time: float = field(default_factory=time.time)
    
    @property
    def total(self) -> int:
        return len(self.symbols)
    
    @property
    def success(self) -> int:
        return self.current + self.updated + self.new
    
    @property
    def pct_complete(self) -> float:
        return (self.completed / self.total * 100) if self.total > 0 else 0
    
    @property
    def elapsed_seconds(self) -> float:
        return time.time() - self.start_time
    
    @property
    def avg_processing_ms(self) -> float:
        if not self.processing_times_ms:
            return 0
        return statistics.mean(self.processing_times_ms)
    
    def record_result(
        self,
        status: str,
        embeddings_count: int,
        duration_ms: int
    ) -> None:
        """Record a processing result."""
        self.completed += 1
        self.embeddings += embeddings_count
        self.processing_times_ms.append(duration_ms)
        
        if status == 'current':
            self.current += 1
        elif status == 'updated':
            self.updated += 1
        elif status == 'new':
            self.new += 1
        elif status == 'failed':
            self.failed += 1


@dataclass
class PipelineMetrics:
    """
    Comprehensive metrics for the embedding pipeline.
    
    Tracks:
    - Overall progress and timing
    - Per-sector breakdown
    - Failure categorization
    - Processing performance
    """
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    
    # Counters
    stocks_processed: int = 0
    embeddings_created: int = 0
    
    # Status breakdown
    status_counts: Dict[str, int] = field(default_factory=lambda: {
        'current': 0,
        'updated': 0,
        'new': 0,
        'failed': 0
    })
    
    # Failures by error type
    failures_by_type: Dict[str, List[str]] = field(default_factory=dict)
    
    # Processing times
    processing_times_ms: List[int] = field(default_factory=list)
    
    # Per-sector metrics
    sector_metrics: Dict[str, SectorMetrics] = field(default_factory=dict)
    
    # Lineage information
    pipeline_version: str = ""
    encoder_version: str = ""
    config_hash: str = ""
    
    def start_sector(self, sector_name: str, symbols: List[str]) -> SectorMetrics:
        """Initialize metrics for a sector."""
        metrics = SectorMetrics(name=sector_name, symbols=symbols)
        self.sector_metrics[sector_name] = metrics
        return metrics
    
    def record_success(
        self,
        symbol: str,
        status: str,
        count: int,
        duration_ms: int,
        sector: Optional[str] = None
    ) -> None:
        """Record a successful stock processing."""
        self.stocks_processed += 1
        self.embeddings_created += count
        self.processing_times_ms.append(duration_ms)
        
        if status in self.status_counts:
            self.status_counts[status] += 1
        
        if sector and sector in self.sector_metrics:
            self.sector_metrics[sector].record_result(status, count, duration_ms)
    
    def record_failure(
        self,
        symbol: str,
        error_type: str,
        error_message: str,
        duration_ms: int = 0,
        sector: Optional[str] = None
    ) -> None:
        """Record a failed stock processing."""
        self.stocks_processed += 1
        self.status_counts['failed'] += 1
        
        if error_type not in self.failures_by_type:
            self.failures_by_type[error_type] = []
        self.failures_by_type[error_type].append(f"{symbol}: {error_message}")
        
        if sector and sector in self.sector_metrics:
            self.sector_metrics[sector].record_result('failed', 0, duration_ms)
    
    def finish(self) -> None:
        """Mark the pipeline run as complete."""
        self.end_time = time.time()
    
    @property
    def elapsed_seconds(self) -> float:
        end = self.end_time or time.time()
        return end - self.start_time
    
    @property
    def failure_rate(self) -> float:
        if self.stocks_processed == 0:
            return 0
        return self.status_counts['failed'] / self.stocks_processed
    
    @property
    def avg_processing_ms(self) -> float:
        if not self.processing_times_ms:
            return 0
        return statistics.mean(self.processing_times_ms)
    
    @property
    def throughput_per_minute(self) -> float:
        elapsed_minutes = self.elapsed_seconds / 60
        if elapsed_minutes == 0:
            return 0
        return self.stocks_processed / elapsed_minutes
    
    def summary(self) -> Dict[str, Any]:
        """Generate a summary dictionary of all metrics."""
        return {
            'elapsed_seconds': round(self.elapsed_seconds, 1),
            'stocks_processed': self.stocks_processed,
            'embeddings_created': self.embeddings_created,
            'throughput_per_minute': round(self.throughput_per_minute, 1),
            'status_breakdown': self.status_counts.copy(),
            'failure_rate': round(self.failure_rate, 3),
            'failures_by_type': {k: len(v) for k, v in self.failures_by_type.items()},
            'avg_processing_ms': round(self.avg_processing_ms, 1),
            'sector_count': len(self.sector_metrics),
            'pipeline_version': self.pipeline_version,
        }
    
    def print_summary(self) -> None:
        """Print a formatted summary to the console."""
        summary = self.summary()
        
        print()
        print("=" * 80)
        print("Pipeline Metrics Summary")
        print("=" * 80)
        print()
        
        print(f"  Duration:     {summary['elapsed_seconds']:.1f} seconds")
        print(f"  Throughput:   {summary['throughput_per_minute']:.1f} stocks/minute")
        print()
        
        print("  Status Breakdown:")
        print(f"    • Current (skipped):  {summary['status_breakdown']['current']:4d}")
        print(f"    • Updated:            {summary['status_breakdown']['updated']:4d}")
        print(f"    • New:                {summary['status_breakdown']['new']:4d}")
        print(f"    • Failed:             {summary['status_breakdown']['failed']:4d}")
        print()
        
        print(f"  Embeddings Created: {summary['embeddings_created']:,}")
        print(f"  Avg Processing:     {summary['avg_processing_ms']:.0f} ms/stock")
        print()
        
        if self.failures_by_type:
            print("  Failures by Type:")
            for error_type, symbols in self.failures_by_type.items():
                print(f"    • {error_type}: {len(symbols)}")
            print()
        
        if self.sector_metrics:
            print("  Sector Summary:")
            for name, metrics in sorted(self.sector_metrics.items()):
                status_icon = "✓" if metrics.failed == 0 else "⚠"
                print(
                    f"    {status_icon} {name[:25]:<25} "
                    f"⏭{metrics.current:2d} ↑{metrics.updated:2d} "
                    f"✓{metrics.new:2d} ✗{metrics.failed:2d} | "
                    f"+{metrics.embeddings:,} emb"
                )
            print()
        
        print("=" * 80)
    
    def to_json(self) -> str:
        """Export metrics as JSON string."""
        data = {
            'timestamp': datetime.now().isoformat(),
            'summary': self.summary(),
            'failures': {
                error_type: symbols[:10]  # Limit to 10 examples per type
                for error_type, symbols in self.failures_by_type.items()
            },
            'sectors': {
                name: {
                    'total': m.total,
                    'completed': m.completed,
                    'current': m.current,
                    'updated': m.updated,
                    'new': m.new,
                    'failed': m.failed,
                    'embeddings': m.embeddings,
                    'elapsed_seconds': m.elapsed_seconds,
                }
                for name, m in self.sector_metrics.items()
            }
        }
        return json.dumps(data, indent=2)
    
    def save(self, path: str) -> None:
        """Save metrics to a JSON file."""
        try:
            file_path = Path(path)
            file_path.parent.mkdir(parents=True, exist_ok=True)
            with open(file_path, 'w') as f:
                f.write(self.to_json())
            logger.info(f"Metrics saved to {path}")
        except Exception as e:
            logger.warning(f"Failed to save metrics: {e}")


@dataclass
class LineageMetadata:
    """Lineage information for embedding tracking."""
    pipeline_version: str
    encoder_version: str
    generated_at: str
    lookback_days: int
    config_hash: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'pipeline_version': self.pipeline_version,
            'encoder_version': self.encoder_version,
            'generated_at': self.generated_at,
            'lookback_days': self.lookback_days,
            'config_hash': self.config_hash,
        }
    
    @classmethod
    def create(
        cls,
        pipeline_version: str,
        encoder_version: str,
        lookback_days: int,
        config: Optional[Any] = None
    ) -> "LineageMetadata":
        """Create lineage metadata for current run."""
        import hashlib
        
        config_hash = ""
        if config:
            try:
                config_str = str(config.to_dict() if hasattr(config, 'to_dict') else config)
                config_hash = hashlib.md5(config_str.encode()).hexdigest()[:8]
            except Exception:
                pass
        
        return cls(
            pipeline_version=pipeline_version,
            encoder_version=encoder_version,
            generated_at=datetime.now().isoformat(),
            lookback_days=lookback_days,
            config_hash=config_hash
        )
