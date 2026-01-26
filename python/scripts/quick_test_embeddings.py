#!/usr/bin/env python3.11
"""
Quick Test: Generate embeddings for top 10 most liquid S&P 500 stocks

This is a faster way to test the system before doing the full 503 stocks.
Takes ~10-20 minutes instead of 2-6 hours.
"""
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import the main generation script
from scripts.generate_sp500_embeddings import (
    generate_embeddings_for_stock,
    MarketStateEncoder,
    VectorDatabase,
    load_sp500_list,
    fetch_sp500_list,
    clean_sp500_data
)
import time
import asyncio


# Top 10 most liquid stocks by market cap and volume
TOP_10_STOCKS = [
    'AAPL',   # Apple
    'MSFT',   # Microsoft
    'GOOGL',  # Alphabet
    'AMZN',   # Amazon
    'NVDA',   # NVIDIA
    'META',   # Meta
    'TSLA',   # Tesla
    'BRK-B',  # Berkshire Hathaway
    'JPM',    # JPMorgan Chase
    'V',      # Visa
]


async def main():
    """Generate embeddings for top 10 stocks only"""
    
    print("=" * 80)
    print("Quick Test: Top 10 Stocks Embedding Generation")
    print("=" * 80)
    print()
    print(f"Generating embeddings for: {', '.join(TOP_10_STOCKS)}")
    print()
    
    # Initialize encoder
    print("[1/2] Initializing encoder...")
    encoder = MarketStateEncoder()
    print("  ✓ Encoder ready")
    print()
    
    # Process stocks
    print(f"[2/2] Processing {len(TOP_10_STOCKS)} stocks...")
    print()
    
    start_time = time.time()
    results = []
    successful = 0
    failed = 0
    total_embeddings = 0
    
    for i, symbol in enumerate(TOP_10_STOCKS, 1):
        print(f"  [{i}/{len(TOP_10_STOCKS)}] ", end="")
        
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
    
    # Summary
    elapsed = time.time() - start_time
    print()
    print("=" * 80)
    print("✓ Quick test complete!")
    print(f"  Time elapsed: {elapsed/60:.1f} minutes")
    print(f"  Successful: {successful}/{len(TOP_10_STOCKS)}")
    print(f"  Failed: {failed}/{len(TOP_10_STOCKS)}")
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
    
    print("Next steps:")
    print("  1. Run populate_stocks_table.py to update metadata")
    print("  2. Start backend and frontend")
    print("  3. Test the stock selector with these 10 stocks")
    print()
    print("To generate all 503 stocks:")
    print("  python3.11 scripts/generate_sp500_embeddings.py")
    print()


if __name__ == "__main__":
    asyncio.run(main())
