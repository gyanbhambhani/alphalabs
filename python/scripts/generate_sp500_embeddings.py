"""
Generate Embeddings for All S&P 500 Stocks

Refactored version using the core.embeddings module for:
- Unified processing with StockProcessor
- Rate limiting and caching
- Data validation
- Comprehensive metrics
- Progress checkpointing

Usage:
    python scripts/generate_sp500_embeddings.py
    python scripts/generate_sp500_embeddings.py --workers 16
    python scripts/generate_sp500_embeddings.py --mode full --workers 8
    python scripts/generate_sp500_embeddings.py --dry-run
"""
import sys
import asyncio
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Set

import pandas as pd

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.embeddings.config import EmbeddingPipelineConfig
from core.embeddings.processor import StockProcessor, ProcessingStrategy, ProcessingResult
from core.embeddings.rate_limiter import RateLimiter
from core.embeddings.cache import DataCache
from core.embeddings.metrics import PipelineMetrics, SectorMetrics
from core.semantic.encoder import MarketStateEncoder
from core.semantic.vector_db import VectorDatabase
from scripts.fetch_sp500_list import fetch_sp500_list, clean_sp500_data, load_sp500_list

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class CheckpointManager:
    """Manages progress checkpoints for crash recovery."""
    
    def __init__(self, checkpoint_file: str):
        self.checkpoint_file = Path(checkpoint_file)
        self._completed: Set[str] = set()
        self._load()
    
    def _load(self) -> None:
        """Load checkpoint from file."""
        if self.checkpoint_file.exists():
            try:
                with open(self.checkpoint_file, 'r') as f:
                    data = json.load(f)
                    self._completed = set(data.get('completed', []))
                    logger.info(f"Loaded checkpoint: {len(self._completed)} completed")
            except Exception as e:
                logger.warning(f"Failed to load checkpoint: {e}")
    
    def save(self) -> None:
        """Save checkpoint to file."""
        try:
            self.checkpoint_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.checkpoint_file, 'w') as f:
                json.dump({
                    'completed': list(self._completed),
                    'updated_at': datetime.now().isoformat()
                }, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save checkpoint: {e}")
    
    def mark_completed(self, symbol: str) -> None:
        """Mark a symbol as completed."""
        self._completed.add(symbol)
    
    def is_completed(self, symbol: str) -> bool:
        """Check if symbol is already completed."""
        return symbol in self._completed
    
    def get_pending(self, symbols: List[str]) -> List[str]:
        """Filter to symbols not yet completed."""
        return [s for s in symbols if s not in self._completed]
    
    def clear(self) -> None:
        """Clear all checkpoints."""
        self._completed.clear()
        if self.checkpoint_file.exists():
            self.checkpoint_file.unlink()


class StateManager:
    """Backward-compatible state file management."""
    
    def __init__(self, state_file: str):
        self.state_file = Path(state_file)
        self._state = self._load()
    
    def _load(self) -> Dict:
        """Load state from file."""
        if self.state_file.exists():
            try:
                with open(self.state_file, 'r') as f:
                    return json.load(f)
            except Exception:
                pass
        return {"stocks": {}, "last_run": None}
    
    def save(self) -> None:
        """Save state to file."""
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        self._state["last_run"] = datetime.now().isoformat()
        with open(self.state_file, 'w') as f:
            json.dump(self._state, f, indent=2)
    
    def update_stock(
        self,
        symbol: str,
        last_date: str,
        count: int,
        status: str
    ) -> None:
        """Update state for a stock."""
        self._state["stocks"][symbol] = {
            "last_embedding_date": last_date,
            "embedding_count": count,
            "last_updated": datetime.now().isoformat(),
            "status": status
        }
    
    @property
    def last_run(self) -> Optional[str]:
        return self._state.get("last_run")
    
    @property
    def tracked_count(self) -> int:
        return len(self._state.get("stocks", {}))


def group_stocks_by_sector(sp500_df: pd.DataFrame) -> Dict[str, List[str]]:
    """Group stocks by their GICS sector."""
    sectors = {}
    
    # Find sector column
    sector_col = None
    for col in ['sector', 'GICS Sector', 'gics_sector']:
        if col in sp500_df.columns:
            sector_col = col
            break
    
    if sector_col is None:
        return {"All Stocks": sp500_df['symbol'].tolist()}
    
    for sector in sp500_df[sector_col].unique():
        symbols = sp500_df[sp500_df[sector_col] == sector]['symbol'].tolist()
        sectors[sector] = symbols
    
    return dict(sorted(sectors.items()))


async def process_stock_async(
    processor: StockProcessor,
    symbol: str,
    strategy: ProcessingStrategy,
    semaphore: asyncio.Semaphore,
    sector_metrics: Optional[SectorMetrics] = None
) -> ProcessingResult:
    """Process a single stock asynchronously."""
    async with semaphore:
        # Run sync processor in thread pool
        result = await asyncio.to_thread(
            processor.process_sync,
            symbol,
            strategy
        )
        
        # Update sector metrics
        if sector_metrics:
            if result.success:
                if result.status == 'current':
                    sector_metrics.current += 1
                elif result.status == 'updated':
                    sector_metrics.updated += 1
                elif result.status == 'new':
                    sector_metrics.new += 1
                sector_metrics.embeddings += result.embeddings_added
            else:
                sector_metrics.failed += 1
            sector_metrics.completed += 1
            sector_metrics.processing_times_ms.append(result.duration_ms)
        
        return result


async def run_pipeline(
    sp500_df: pd.DataFrame,
    config: EmbeddingPipelineConfig,
    strategy: ProcessingStrategy,
    max_workers: int,
    dry_run: bool = False,
    use_checkpoint: bool = True
) -> PipelineMetrics:
    """Run the embedding generation pipeline."""
    
    # Initialize components
    encoder = MarketStateEncoder()
    rate_limiter = RateLimiter(calls_per_second=config.rate_limit_calls_per_second)
    cache = DataCache(
        cache_dir=config.cache_dir,
        enabled=config.enable_cache
    ) if config.enable_cache else None
    
    processor = StockProcessor(
        config=config,
        encoder=encoder,
        rate_limiter=rate_limiter,
        cache=cache,
        dry_run=dry_run
    )
    
    checkpoint = CheckpointManager(config.checkpoint_file) if use_checkpoint else None
    state = StateManager(config.state_file)
    metrics = PipelineMetrics(pipeline_version=config.pipeline_version)
    
    # Group by sector
    sectors = group_stocks_by_sector(sp500_df)
    
    print(f"\n  Found {len(sectors)} sectors:")
    for sector, symbols in sectors.items():
        print(f"    • {sector}: {len(symbols)} stocks")
    print()
    
    # Create semaphore for concurrency control
    semaphore = asyncio.Semaphore(max_workers)
    
    # Progress display task
    async def display_progress():
        while True:
            await asyncio.sleep(2)
            
            total_done = sum(m.completed for m in metrics.sector_metrics.values())
            total_all = sum(m.total for m in metrics.sector_metrics.values())
            total_emb = sum(m.embeddings for m in metrics.sector_metrics.values())
            total_current = sum(m.current for m in metrics.sector_metrics.values())
            total_updated = sum(m.updated for m in metrics.sector_metrics.values())
            total_new = sum(m.new for m in metrics.sector_metrics.values())
            total_failed = sum(m.failed for m in metrics.sector_metrics.values())
            
            if total_all > 0:
                pct = total_done / total_all * 100
                elapsed = metrics.elapsed_seconds
                rate = total_done / elapsed if elapsed > 0 else 0
                eta = (total_all - total_done) / rate if rate > 0 else 0
                
                print(
                    f"\r  [{total_done}/{total_all}] {pct:5.1f}% | "
                    f"⏭{total_current} ↑{total_updated} ✓{total_new} ✗{total_failed} | "
                    f"+{total_emb:,} emb | ETA: {eta:.0f}s    ",
                    end="", flush=True
                )
            
            if total_done >= total_all and total_all > 0:
                break
    
    # Start progress display
    progress_task = asyncio.create_task(display_progress())
    
    try:
        # Process all sectors concurrently
        async def process_sector(sector_name: str, symbols: List[str]):
            # Filter out already-completed symbols
            if checkpoint and strategy != ProcessingStrategy.FULL_REFRESH:
                pending = checkpoint.get_pending(symbols)
                if len(pending) < len(symbols):
                    logger.info(
                        f"{sector_name}: Resuming, {len(symbols) - len(pending)} "
                        f"already complete"
                    )
                symbols = pending
            
            sector_metrics = metrics.start_sector(sector_name, symbols)
            
            tasks = [
                process_stock_async(
                    processor, sym, strategy, semaphore, sector_metrics
                )
                for sym in symbols
            ]
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process results
            for result in results:
                if isinstance(result, Exception):
                    metrics.record_failure(
                        symbol='unknown',
                        error_type=type(result).__name__,
                        error_message=str(result),
                        sector=sector_name
                    )
                    continue
                
                if result.success:
                    metrics.record_success(
                        symbol=result.symbol,
                        status=result.status,
                        count=result.embeddings_added,
                        duration_ms=result.duration_ms,
                        sector=sector_name
                    )
                    
                    if result.last_date:
                        state.update_stock(
                            result.symbol,
                            result.last_date,
                            result.embeddings_added,
                            result.status
                        )
                    
                    if checkpoint:
                        checkpoint.mark_completed(result.symbol)
                else:
                    metrics.record_failure(
                        symbol=result.symbol,
                        error_type=result.error_type or 'Unknown',
                        error_message=result.error or 'No error message',
                        duration_ms=result.duration_ms,
                        sector=sector_name
                    )
            
            # Save checkpoint periodically
            if checkpoint:
                checkpoint.save()
            
            return results
        
        sector_tasks = [
            process_sector(name, symbols)
            for name, symbols in sectors.items()
        ]
        
        await asyncio.gather(*sector_tasks)
        
    finally:
        progress_task.cancel()
        try:
            await progress_task
        except asyncio.CancelledError:
            pass
    
    print()  # New line after progress
    
    # Save final state
    state.save()
    if checkpoint:
        checkpoint.save()
    
    metrics.finish()
    return metrics


async def main(
    mode: str = "sync",
    force_full: bool = False,
    max_workers: int = 8,
    dry_run: bool = False,
    env: str = None
):
    """Main entry point for the embedding pipeline."""
    
    # Load configuration
    config = EmbeddingPipelineConfig.from_env(env)
    
    # Determine strategy
    strategy = (
        ProcessingStrategy.FULL_REFRESH if force_full
        else ProcessingStrategy.INCREMENTAL
    )
    
    # Print header
    print("=" * 80)
    print("S&P 500 Embedding Generator v2.0")
    print("=" * 80)
    print(f"  Mode: {'FULL REGENERATION' if force_full else mode.upper()}")
    print(f"  Workers: {max_workers} async workers")
    print(f"  Dry Run: {dry_run}")
    print(f"  Config: {env or 'default'}")
    print(f"  ChromaDB: {config.chroma_dir}")
    print()
    
    # Load state info
    state = StateManager(config.state_file)
    if state.last_run:
        print(f"  Last run: {state.last_run[:19]}")
        print(f"  Tracking: {state.tracked_count} stocks")
    print("=" * 80)
    print()
    
    # Step 1: Load S&P 500 list
    print("[1/4] Loading S&P 500 stock list...")
    sp500_path = Path(__file__).parent.parent / "data" / "sp500_list.csv"
    
    if sp500_path.exists():
        print("  Loading from saved file...")
        sp500_df = load_sp500_list(str(sp500_path))
    else:
        print("  Fetching from Wikipedia...")
        sp500_df = fetch_sp500_list()
        sp500_df = clean_sp500_data(sp500_df)
        sp500_df.to_csv(sp500_path, index=False)
    
    symbols = sp500_df['symbol'].tolist()
    sectors = group_stocks_by_sector(sp500_df)
    print(f"  ✓ Loaded {len(symbols)} stocks in {len(sectors)} sectors")
    print()
    
    # Step 2: Check existing embeddings
    print("[2/4] Analyzing existing embeddings...")
    existing_symbols = VectorDatabase.get_all_symbols(config.chroma_dir)
    print(f"  Found {len(existing_symbols)} stocks with existing embeddings")
    
    # Bulk delete for full regeneration
    if force_full and existing_symbols and not dry_run:
        print(f"  Deleting all {len(existing_symbols)} collections...", end=" ", flush=True)
        import chromadb
        client = chromadb.PersistentClient(path=config.chroma_dir)
        deleted = 0
        for sym in existing_symbols:
            try:
                client.delete_collection(name=f"market_states_{sym}")
                deleted += 1
            except Exception:
                pass
        print(f"✓ Deleted {deleted}")
        existing_symbols = []
    
    new_count = len([s for s in symbols if s not in existing_symbols])
    existing_count = len([s for s in symbols if s in existing_symbols])
    print(f"  • {new_count} stocks need initial generation")
    print(f"  • {existing_count} stocks may need updates")
    print()
    
    # Filter by mode
    if mode == "new_only":
        sp500_df = sp500_df[~sp500_df['symbol'].isin(existing_symbols)]
        print(f"  Mode 'new_only': Processing {len(sp500_df)} new stocks")
    else:
        print(f"  Mode 'sync': Processing all {len(symbols)} stocks")
    print()
    
    if len(sp500_df) == 0:
        print("No stocks to process!")
        return
    
    # Step 3: Initialize encoder
    print("[3/4] Initializing encoder...")
    print("  ✓ Encoder ready")
    print()
    
    # Step 4: Run pipeline
    print(f"[4/4] Processing {len(sp500_df)} stocks with {max_workers} workers...")
    
    metrics = await run_pipeline(
        sp500_df=sp500_df,
        config=config,
        strategy=strategy,
        max_workers=max_workers,
        dry_run=dry_run,
        use_checkpoint=not force_full
    )
    
    # Print summary
    metrics.print_summary()
    
    # Save metrics
    if config.enable_metrics:
        metrics_path = Path(config.log_dir) / "pipeline_metrics.json"
        metrics.save(str(metrics_path))
    
    # Final stats
    all_symbols = VectorDatabase.get_all_symbols(config.chroma_dir)
    print(f"Database now contains {len(all_symbols)} stocks with embeddings")
    print()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Generate/sync S&P 500 stock embeddings with async workers"
    )
    parser.add_argument(
        "--mode",
        choices=["sync", "new_only", "full"],
        default="sync",
        help=(
            "sync: update all stocks to current date (default), "
            "new_only: only process stocks without embeddings, "
            "full: delete and regenerate all embeddings from scratch"
        )
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force full regeneration (same as --mode full)"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Number of concurrent async workers (default: 8)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without making changes"
    )
    parser.add_argument(
        "--env",
        choices=["dev", "staging", "prod"],
        default=None,
        help="Environment for configuration (default: from PIPELINE_ENV)"
    )
    
    args = parser.parse_args()
    
    force_full = args.force or (args.mode == "full")
    
    asyncio.run(main(
        mode=args.mode,
        force_full=force_full,
        max_workers=args.workers,
        dry_run=args.dry_run,
        env=args.env
    ))
