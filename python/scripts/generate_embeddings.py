"""
Generate Historical Market Embeddings

Loads 10 years of historical data and encodes market states into
ChromaDB for semantic search.
"""
import sys
import asyncio
from pathlib import Path
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.ingest import DataIngestion
from core.semantic.encoder import MarketStateEncoder
from core.semantic.vector_db import VectorDatabase


async def main():
    """Generate and store historical market embeddings"""
    
    print("=" * 60)
    print("Historical Market Embedding Generation")
    print("=" * 60)
    print()
    
    # Step 1: Initialize components
    print("[1/4] Initializing components...")
    ingestion = DataIngestion(years_of_history=10)
    encoder = MarketStateEncoder()
    vector_db = VectorDatabase(
        persist_directory="./chroma_data",
        in_memory=False
    )
    
    # Check if data already exists
    existing_count = vector_db.get_count()
    if existing_count > 0:
        print(f"  ⚠️  Found {existing_count} existing embeddings in database")
        response = input("  Delete and regenerate? (y/N): ")
        if response.lower() == 'y':
            print("  Deleting existing data...")
            vector_db.delete_all()
        else:
            print("  Keeping existing data. Exiting.")
            return
    
    print("  ✓ Components initialized")
    print()
    
    # Step 2: Fetch historical data
    print("[2/4] Fetching 10 years of historical data...")
    print(f"  Universe: {len(ingestion.universe)} symbols")
    data_dict = ingestion.fetch_all()
    print(f"  ✓ Fetched data for {len(data_dict)} symbols")
    print()
    
    # Step 3: Generate embeddings for market index (SPY)
    print("[3/4] Generating embeddings for market index (SPY)...")
    
    if "SPY" not in data_dict:
        print("  ✗ SPY not found in data. Cannot proceed.")
        return
    
    spy_data = data_dict["SPY"]
    print(f"  Date range: {spy_data.start_date} to {spy_data.end_date}")
    print(f"  Trading days: {len(spy_data.data)}")
    
    # Prepare data
    df = spy_data.data.copy()
    df.columns = df.columns.str.lower()
    
    # Generate states (need 252 days lookback)
    min_lookback = 252
    states = []
    
    print(f"  Encoding market states (starting from day {min_lookback})...")
    for i in range(min_lookback, len(df)):
        if i % 100 == 0:
            progress = (i - min_lookback) / (len(df) - min_lookback) * 100
            print(f"    Progress: {progress:.1f}%", end='\r')
        
        window = df.iloc[:i+1]
        date_str = str(window.index[-1].date())
        
        # Calculate forward returns for metadata
        forward_returns = {}
        for days, label in [(5, '5d'), (10, '10d'), (21, '1m'), (63, '3m')]:
            if i + days < len(df):
                start_price = df.iloc[i]['close']
                end_price = df.iloc[i + days]['close']
                forward_returns[f'forward_{label}_return'] = (
                    (end_price - start_price) / start_price
                )
        
        state = encoder.encode(
            date=date_str,
            close=window['close'],
            high=window.get('high'),
            low=window.get('low'),
            volume=window.get('volume')
        )
        
        # Add forward returns to metadata
        state.metadata.update(forward_returns)
        states.append(state)
    
    print()
    print(f"  ✓ Generated {len(states)} market state embeddings")
    print()
    
    # Step 4: Store in ChromaDB
    print("[4/4] Storing embeddings in ChromaDB...")
    count = vector_db.add_batch(states, batch_size=500)
    print(f"  ✓ Stored {count} embeddings")
    print()
    
    # Verify
    date_range = vector_db.get_date_range()
    total_count = vector_db.get_count()
    print("=" * 60)
    print("✓ Embedding generation complete!")
    print(f"  Total embeddings: {total_count}")
    print(f"  Date range: {date_range[0]} to {date_range[1]}")
    print(f"  Storage: ./chroma_data")
    print("=" * 60)
    print()
    print("You can now use semantic search to find similar market periods!")
    print()


if __name__ == "__main__":
    asyncio.run(main())
