"""
Generate Embeddings for All S&P 500 Stocks

Fetches maximum historical data for all S&P 500 constituents
and generates market state embeddings for semantic search.
"""
import sys
import asyncio
from pathlib import Path
from datetime import datetime
import pandas as pd
from typing import Dict, List
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


async def generate_embeddings_for_stock(
    symbol: str,
    encoder: MarketStateEncoder,
    persist_directory: str = "./chroma_data",
    period: str = "max"
) -> tuple[str, bool, int]:
    """
    Generate embeddings for a single stock.
    
    Args:
        symbol: Stock ticker
        encoder: MarketStateEncoder instance
        persist_directory: Where to store embeddings
        period: How much history to fetch (max = all available)
        
    Returns:
        (symbol, success, count) tuple
    """
    try:
        print(f"  Processing {symbol}...", end=" ")
        
        # Fetch stock data with maximum history
        stock_data = fetch_stock_data(symbol, period=period)
        
        if not stock_data or len(stock_data.data) < 300:
            print(f"✗ Insufficient data ({len(stock_data.data) if stock_data else 0} days)")
            return (symbol, False, 0)
        
        # Initialize vector database for this symbol
        vector_db = VectorDatabase(
            persist_directory=persist_directory,
            symbol=symbol
        )
        
        # Check if already exists
        existing_count = vector_db.get_count()
        if existing_count > 0:
            print(f"⚠️  Already has {existing_count} embeddings (skipping)")
            return (symbol, True, existing_count)
        
        # Prepare data
        df = stock_data.data.copy()
        
        # Generate states (need 252 days lookback)
        min_lookback = 252
        if len(df) < min_lookback:
            print(f"✗ Too few days ({len(df)} < {min_lookback})")
            return (symbol, False, 0)
        
        states = []
        for i in range(min_lookback, len(df)):
            window = df.iloc[:i+1]
            date_str = str(window.index[-1].date())
            
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
        
        # Store in ChromaDB
        count = vector_db.add_batch(states, batch_size=500)
        
        print(f"✓ {count} embeddings ({stock_data.start_date} to {stock_data.end_date})")
        return (symbol, True, count)
        
    except Exception as e:
        print(f"✗ Error: {str(e)[:50]}")
        return (symbol, False, 0)


async def main():
    """Generate embeddings for all S&P 500 stocks"""
    
    print("=" * 80)
    print("S&P 500 Bulk Embedding Generation")
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
    print("[2/4] Checking existing embeddings...")
    existing_symbols = VectorDatabase.get_all_symbols("./chroma_data")
    print(f"  Found {len(existing_symbols)} stocks with existing embeddings")
    
    # Filter to only new stocks
    symbols_to_process = [s for s in symbols if s not in existing_symbols]
    print(f"  Will process {len(symbols_to_process)} new stocks")
    print()
    
    if not symbols_to_process:
        print("All stocks already have embeddings!")
        print()
        return
    
    # Step 3: Initialize encoder
    print("[3/4] Initializing encoder...")
    encoder = MarketStateEncoder()
    print("  ✓ Encoder ready")
    print()
    
    # Step 4: Process all stocks
    print(f"[4/4] Processing {len(symbols_to_process)} stocks...")
    print(f"  (This may take several hours for all S&P 500 stocks)")
    print()
    
    start_time = time.time()
    results = []
    successful = 0
    failed = 0
    total_embeddings = 0
    
    for i, symbol in enumerate(symbols_to_process, 1):
        print(f"  [{i}/{len(symbols_to_process)}] ", end="")
        
        result = await generate_embeddings_for_stock(
            symbol, 
            encoder, 
            persist_directory="./chroma_data",
            period="max"
        )
        
        results.append(result)
        
        if result[1]:  # success
            successful += 1
            total_embeddings += result[2]
        else:
            failed += 1
        
        # Progress update every 10 stocks
        if i % 10 == 0:
            elapsed = time.time() - start_time
            rate = i / elapsed
            remaining = (len(symbols_to_process) - i) / rate
            print(f"\n  Progress: {i}/{len(symbols_to_process)} "
                  f"({successful} success, {failed} failed) "
                  f"| ETA: {remaining/60:.1f} min\n")
    
    # Summary
    elapsed = time.time() - start_time
    print()
    print("=" * 80)
    print("✓ Bulk embedding generation complete!")
    print(f"  Time elapsed: {elapsed/60:.1f} minutes")
    print(f"  Successful: {successful}/{len(symbols_to_process)}")
    print(f"  Failed: {failed}/{len(symbols_to_process)}")
    print(f"  Total embeddings generated: {total_embeddings:,}")
    print("=" * 80)
    print()
    
    # Show failed stocks
    if failed > 0:
        print("Failed stocks:")
        for symbol, success, _ in results:
            if not success:
                print(f"  • {symbol}")
        print()
    
    # Final stats
    all_symbols = VectorDatabase.get_all_symbols("./chroma_data")
    print(f"Database now contains {len(all_symbols)} stocks with embeddings")
    print()


if __name__ == "__main__":
    asyncio.run(main())
