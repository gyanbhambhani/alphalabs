#!/usr/bin/env python3.11
"""
Seed S&P 500 Historical Data

Downloads and caches 25 years of daily OHLCV data for all S&P 500 stocks.
Uses batch downloading for efficiency (~5-10 minutes for full S&P 500).

Usage:
    python3.11 scripts/seed_sp500_data.py
    
Options:
    --force     Force re-download even if cached
    --quick     Only download a subset (50 stocks) for quick testing
"""

import sys
import time
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.backtest.data_loader import (
    HistoricalDataLoader, 
    load_sp500_universe,
    LEGACY_UNIVERSE,
)


def main():
    print("=" * 70)
    print("S&P 500 Historical Data Seeder")
    print("=" * 70)
    print()
    
    # Parse args
    force_refresh = "--force" in sys.argv
    quick_mode = "--quick" in sys.argv
    
    # Load universe
    if quick_mode:
        universe = LEGACY_UNIVERSE
        print(f"Quick mode: Using {len(universe)} stocks")
    else:
        universe = load_sp500_universe()
        print(f"Full mode: Using {len(universe)} S&P 500 stocks")
    
    print()
    
    # Initialize loader
    loader = HistoricalDataLoader(
        cache_dir="data/backtest_cache",
        universe=universe,
    )
    
    print(f"Cache directory: {loader.cache_dir}")
    print(f"Date range: {loader.start_date} to {loader.end_date}")
    print()
    
    # Check existing cache
    cached_count = sum(1 for s in universe if loader._is_cached(s))
    print(f"Already cached: {cached_count}/{len(universe)} symbols")
    
    if force_refresh:
        print("Force refresh enabled - will re-download all")
    print()
    
    # Fetch and cache
    print("[1/2] Downloading historical data...")
    start_time = time.time()
    
    results = loader.fetch_and_cache_all(force_refresh=force_refresh)
    
    elapsed = time.time() - start_time
    print(f"  Completed in {elapsed:.1f} seconds")
    print()
    
    # Summary
    print("[2/2] Summary:")
    print(f"  Total symbols: {len(universe)}")
    print(f"  Successfully cached: {len(results)}")
    print(f"  Failed: {len(universe) - len(results)}")
    
    if results:
        # Stats on data
        total_days = sum(results.values())
        avg_days = total_days / len(results)
        print(f"  Total trading days: {total_days:,}")
        print(f"  Average days per symbol: {avg_days:.0f}")
    
    print()
    print("=" * 70)
    print("✓ Data seeding complete!")
    print()
    
    # Show sample
    if results:
        print("Sample symbols cached:")
        for i, (symbol, days) in enumerate(list(results.items())[:10]):
            print(f"  {symbol}: {days} days")
        if len(results) > 10:
            print(f"  ... and {len(results) - 10} more")
    
    print()
    print("You can now run backtests with the full S&P 500 universe!")
    print("=" * 70)


if __name__ == "__main__":
    main()
