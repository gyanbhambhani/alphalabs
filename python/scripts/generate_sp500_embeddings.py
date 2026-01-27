"""
Generate Embeddings for All S&P 500 Stocks

Fetches maximum historical data for all S&P 500 constituents
and generates market state embeddings for semantic search.

Supports incremental updates - checks last embedding date and
fetches only new data to current trading date.
"""
import sys
import asyncio
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import time

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.ingest import fetch_stock_data
from core.semantic.encoder import MarketStateEncoder
from core.semantic.vector_db import VectorDatabase
from scripts.fetch_sp500_list import (
    fetch_sp500_list, 
    clean_sp500_data, 
    load_sp500_list
)


# Days of historical data needed for encoder to compute indicators (200-day MA, etc.)
INDICATOR_LOOKBACK_DAYS = 300

# Minimum days of data before we start generating embeddings for new stocks
MIN_EMBEDDING_START_DAYS = 252


def get_last_embedding_date(
    symbol: str,
    persist_directory: str = "./chroma_data"
) -> Optional[str]:
    """
    Get the last date for which we have an embedding for a stock.
    
    Args:
        symbol: Stock ticker
        persist_directory: Where embeddings are stored
        
    Returns:
        Last date string (YYYY-MM-DD) or None if no embeddings exist
    """
    try:
        vector_db = VectorDatabase(
            persist_directory=persist_directory,
            symbol=symbol
        )
        
        if vector_db.get_count() == 0:
            return None
        
        _, last_date = vector_db.get_date_range()
        return last_date if last_date else None
    except Exception:
        return None


def get_current_trading_date() -> str:
    """
    Get the current trading date (today, or last Friday if weekend).
    
    Returns:
        Date string (YYYY-MM-DD)
    """
    today = datetime.now()
    
    # If weekend, go back to Friday
    if today.weekday() == 5:  # Saturday
        today = today - timedelta(days=1)
    elif today.weekday() == 6:  # Sunday
        today = today - timedelta(days=2)
    
    return today.strftime('%Y-%m-%d')


async def generate_embeddings_for_stock(
    symbol: str,
    encoder: MarketStateEncoder,
    persist_directory: str = "./chroma_data",
    force_full: bool = False
) -> Tuple[str, bool, int, str]:
    """
    Generate embeddings for a single stock (supports incremental updates).
    
    Args:
        symbol: Stock ticker
        encoder: MarketStateEncoder instance
        persist_directory: Where to store embeddings
        force_full: If True, regenerate all embeddings from scratch
        
    Returns:
        (symbol, success, count, status) tuple where status is:
        - 'new': Fresh embeddings generated
        - 'updated': Incremental update performed
        - 'current': Already up to date
        - 'failed': Error occurred
    """
    try:
        print(f"  Processing {symbol}...", end=" ")
        
        # Initialize vector database for this symbol
        vector_db = VectorDatabase(
            persist_directory=persist_directory,
            symbol=symbol
        )
        
        # Check existing embeddings
        existing_count = vector_db.get_count()
        last_embedding_date = None
        
        if existing_count > 0 and not force_full:
            _, last_embedding_date = vector_db.get_date_range()
        
        # Determine what data we need
        current_date = get_current_trading_date()
        
        if last_embedding_date:
            # Check if we're already up to date
            if last_embedding_date >= current_date:
                print(f"✓ Already current ({existing_count} embeddings, "
                      f"last: {last_embedding_date})")
                return (symbol, True, 0, 'current')
            
            # Incremental update: need lookback data + new dates
            # Fetch from (last_date - lookback) to ensure proper indicator calculation
            lookback_start = (
                datetime.strptime(last_embedding_date, '%Y-%m-%d') 
                - timedelta(days=INDICATOR_LOOKBACK_DAYS + 50)  # Extra buffer for trading days
            ).strftime('%Y-%m-%d')
            
            print(f"[updating from {last_embedding_date}]...", end=" ")
            stock_data = fetch_stock_data(
                symbol, 
                start_date=lookback_start,
                end_date=current_date
            )
        else:
            # Full fetch for new stock
            stock_data = fetch_stock_data(symbol, period="max")
        
        # For updates, we need enough data for indicator calculation
        # For new stocks, we need enough for both indicators AND to start generating embeddings
        min_required = INDICATOR_LOOKBACK_DAYS if last_embedding_date else MIN_EMBEDDING_START_DAYS
        
        if not stock_data or len(stock_data.data) < min_required:
            data_len = len(stock_data.data) if stock_data else 0
            print(f"✗ Insufficient data ({data_len} days, need {min_required})")
            return (symbol, False, 0, 'failed')
        
        # Prepare data
        df = stock_data.data.copy()
        
        # Determine which dates to generate embeddings for
        if last_embedding_date:
            # Find the index where we need to start generating new embeddings
            dates_in_df = [str(d.date()) for d in df.index]
            
            try:
                last_idx = dates_in_df.index(last_embedding_date)
                start_idx = last_idx + 1  # Start from day after last embedding
            except ValueError:
                # Last embedding date not in data, find closest date after
                start_idx = None
                for i, d in enumerate(dates_in_df):
                    if d > last_embedding_date:
                        start_idx = i
                        break
                
                if start_idx is None:
                    print("✓ No new dates to process")
                    return (symbol, True, 0, 'current')
            
            # Ensure we have enough lookback from start_idx for indicator calculation
            if start_idx < INDICATOR_LOOKBACK_DAYS:
                # Adjust start_idx to ensure we have enough lookback
                start_idx = max(start_idx, INDICATOR_LOOKBACK_DAYS)
                if start_idx >= len(df):
                    print("✗ Insufficient lookback data for update")
                    return (symbol, False, 0, 'failed')
        else:
            # New stock: start after minimum lookback for embeddings
            start_idx = MIN_EMBEDDING_START_DAYS
        
        # Generate new states
        states = []
        for i in range(start_idx, len(df)):
            window = df.iloc[:i+1]
            date_str = str(window.index[-1].date())
            
            # Skip if we already have this date (safety check)
            if last_embedding_date and date_str <= last_embedding_date:
                continue
            
            state = encoder.encode(
                date=date_str,
                close=window['close'],
                high=window.get('high'),
                low=window.get('low'),
                volume=window.get('volume')
            )
            
            # Add symbol to metadata
            state.metadata['symbol'] = symbol
            states.append(state)
        
        if not states:
            print("✓ No new states to add (already current)")
            return (symbol, True, 0, 'current')
        
        # Store in ChromaDB
        count = vector_db.add_batch(states, batch_size=500)
        
        if last_embedding_date:
            new_last = states[-1].date if states else last_embedding_date
            print(f"✓ +{count} new embeddings ({last_embedding_date} → {new_last})")
            return (symbol, True, count, 'updated')
        else:
            print(f"✓ {count} embeddings "
                  f"({stock_data.start_date} to {stock_data.end_date})")
            return (symbol, True, count, 'new')
        
    except Exception as e:
        print(f"✗ Error: {str(e)[:60]}")
        return (symbol, False, 0, 'failed')


async def update_all_embeddings(
    symbols: List[str],
    encoder: MarketStateEncoder,
    persist_directory: str = "./chroma_data",
    force_full: bool = False
) -> Dict[str, List]:
    """
    Update embeddings for all symbols (incremental where possible).
    
    Args:
        symbols: List of stock symbols to process
        encoder: MarketStateEncoder instance
        persist_directory: Where to store embeddings
        force_full: If True, regenerate all from scratch
        
    Returns:
        Dictionary with categorized results
    """
    start_time = time.time()
    results = {
        'new': [],      # Newly generated (no prior embeddings)
        'updated': [],  # Incrementally updated
        'current': [],  # Already up to date
        'failed': []    # Errors
    }
    
    total_new_embeddings = 0
    
    for i, symbol in enumerate(symbols, 1):
        print(f"  [{i}/{len(symbols)}] ", end="")
        
        symbol, success, count, status = await generate_embeddings_for_stock(
            symbol,
            encoder,
            persist_directory=persist_directory,
            force_full=force_full
        )
        
        results[status].append((symbol, count))
        if status in ('new', 'updated'):
            total_new_embeddings += count
        
        # Progress update every 25 stocks
        if i % 25 == 0:
            elapsed = time.time() - start_time
            rate = i / elapsed if elapsed > 0 else 1
            remaining = (len(symbols) - i) / rate
            print(f"\n  Progress: {i}/{len(symbols)} | "
                  f"New: {len(results['new'])}, "
                  f"Updated: {len(results['updated'])}, "
                  f"Current: {len(results['current'])}, "
                  f"Failed: {len(results['failed'])} | "
                  f"ETA: {remaining/60:.1f} min\n")
    
    results['total_new_embeddings'] = total_new_embeddings
    results['elapsed_time'] = time.time() - start_time
    
    return results


def print_summary(results: Dict, symbols_count: int):
    """Print a summary of the embedding generation results."""
    print()
    print("=" * 80)
    print("✓ Embedding sync complete!")
    print("=" * 80)
    print()
    
    elapsed = results.get('elapsed_time', 0)
    print(f"  Time elapsed: {elapsed/60:.1f} minutes")
    print()
    
    print("  Results:")
    print(f"    • New stocks:     {len(results['new']):4d} stocks")
    print(f"    • Updated:        {len(results['updated']):4d} stocks")
    print(f"    • Already current:{len(results['current']):4d} stocks")
    print(f"    • Failed:         {len(results['failed']):4d} stocks")
    print()
    print(f"  Total new embeddings generated: "
          f"{results.get('total_new_embeddings', 0):,}")
    print()
    
    # Show failed stocks
    if results['failed']:
        print("  Failed stocks:")
        for symbol, _ in results['failed']:
            print(f"    • {symbol}")
        print()
    
    # Show updated stocks with counts
    if results['updated']:
        print("  Updated stocks (with new embedding counts):")
        for symbol, count in sorted(results['updated'], key=lambda x: -x[1])[:10]:
            print(f"    • {symbol}: +{count}")
        if len(results['updated']) > 10:
            print(f"    ... and {len(results['updated']) - 10} more")
        print()


async def main(mode: str = "sync", force_full: bool = False):
    """
    Generate/update embeddings for all S&P 500 stocks.
    
    Args:
        mode: 
            - 'sync': Check all stocks and update those needing new data (default)
            - 'new_only': Only process stocks with no embeddings
        force_full: If True, regenerate all embeddings from scratch
    """
    
    print("=" * 80)
    print("S&P 500 Embedding Sync")
    print(f"Mode: {mode.upper()}" + (" (FORCE FULL)" if force_full else ""))
    print(f"Current trading date: {get_current_trading_date()}")
    print("=" * 80)
    print()
    
    # Step 1: Get S&P 500 list
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
    print(f"  ✓ Loaded {len(symbols)} stocks")
    print()
    
    # Step 2: Check existing collections
    print("[2/4] Analyzing existing embeddings...")
    existing_symbols = VectorDatabase.get_all_symbols("./chroma_data")
    print(f"  Found {len(existing_symbols)} stocks with existing embeddings")
    
    new_symbols = [s for s in symbols if s not in existing_symbols]
    existing_in_list = [s for s in symbols if s in existing_symbols]
    
    print(f"  • {len(new_symbols)} stocks need initial generation")
    print(f"  • {len(existing_in_list)} stocks may need updates")
    print()
    
    # Determine which symbols to process based on mode
    if mode == "new_only":
        symbols_to_process = new_symbols
        print(f"  Mode 'new_only': Processing {len(symbols_to_process)} new stocks")
    else:  # sync mode
        symbols_to_process = symbols  # Process all to check for updates
        print(f"  Mode 'sync': Checking all {len(symbols_to_process)} stocks")
    print()
    
    if not symbols_to_process:
        print("No stocks to process!")
        print()
        return
    
    # Step 3: Initialize encoder
    print("[3/4] Initializing encoder...")
    encoder = MarketStateEncoder()
    print("  ✓ Encoder ready")
    print()
    
    # Step 4: Process stocks
    print(f"[4/4] Processing {len(symbols_to_process)} stocks...")
    if mode == "sync":
        print("  (Checking each stock for updates - this may take a while)")
    print()
    
    results = await update_all_embeddings(
        symbols_to_process,
        encoder,
        persist_directory="./chroma_data",
        force_full=force_full
    )
    
    # Print summary
    print_summary(results, len(symbols_to_process))
    
    # Final database stats
    all_symbols = VectorDatabase.get_all_symbols("./chroma_data")
    print(f"Database now contains {len(all_symbols)} stocks with embeddings")
    print()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Generate/sync S&P 500 stock embeddings"
    )
    parser.add_argument(
        "--mode",
        choices=["sync", "new_only"],
        default="sync",
        help="sync: update all stocks to current date (default), "
             "new_only: only process stocks without embeddings"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force full regeneration of all embeddings"
    )
    
    args = parser.parse_args()
    
    asyncio.run(main(mode=args.mode, force_full=args.force))
