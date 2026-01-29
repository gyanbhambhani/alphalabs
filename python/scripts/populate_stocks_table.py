"""
Populate Stock Metadata Table (Fast Version)

Loads S&P 500 stock metadata into PostgreSQL.
Just checks which ChromaDB collections exist - doesn't query them.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import chromadb
from db import get_sync_session, Stock
from scripts.fetch_sp500_list import (
    load_sp500_list,
    fetch_sp500_list,
    clean_sp500_data
)


def get_symbols_with_embeddings(persist_directory: str = "./chroma_data") -> set:
    """
    Just get the SET of symbols that have collections.
    Fast - only lists collection names, doesn't query data.
    """
    symbols = set()
    
    try:
        print("  Connecting to ChromaDB...", flush=True)
        client = chromadb.PersistentClient(path=persist_directory)
        
        print("  Listing collections...", flush=True)
        collections = client.list_collections()
        
        for col in collections:
            if col.name.startswith("market_states_"):
                symbol = col.name.replace("market_states_", "")
                symbols.add(symbol)
        
        print(f"  ✓ Found {len(symbols)} stocks with embeddings")
        
    except Exception as e:
        print(f"  Warning: ChromaDB error: {e}")
    
    return symbols


def populate_stocks_table():
    """Populate stocks table with S&P 500 data"""
    
    print("=" * 70)
    print("Populate Stock Metadata Table (Fast)")
    print("=" * 70)
    print()
    
    # Load S&P 500 list
    print("[1/3] Loading S&P 500 data...")
    sp500_path = Path(__file__).parent.parent / "data" / "sp500_list.csv"
    
    if sp500_path.exists():
        sp500_df = load_sp500_list(str(sp500_path))
    else:
        print("  Fetching from Wikipedia...")
        sp500_df = fetch_sp500_list()
        sp500_df = clean_sp500_data(sp500_df)
        sp500_path.parent.mkdir(exist_ok=True)
        sp500_df.to_csv(sp500_path, index=False)
    
    print(f"  ✓ Loaded {len(sp500_df)} stocks")
    print()
    
    # Get symbols with embeddings (fast - just collection names)
    print("[2/3] Checking for embeddings...")
    symbols_with_embeddings = get_symbols_with_embeddings("./chroma_data")
    print()
    
    # Populate database
    print("[3/3] Populating database...")
    
    added = 0
    updated = 0
    
    with get_sync_session() as session:
        try:
            for _, row in sp500_df.iterrows():
                symbol = row['symbol']
                has_embeddings = symbol in symbols_with_embeddings
                
                # Check if stock exists
                stock = session.query(Stock).filter(
                    Stock.symbol == symbol
                ).first()
                
                if stock:
                    # Update existing
                    stock.name = row['name']
                    stock.sector = row['sector']
                    stock.sub_industry = row['sub_industry']
                    stock.headquarters = row['headquarters']
                    stock.date_added = row['date_added']
                    stock.cik = row['cik']
                    stock.founded = row['founded']
                    stock.has_embeddings = has_embeddings
                    # Skip count/date range - too slow
                    updated += 1
                else:
                    # Create new
                    stock = Stock(
                        symbol=symbol,
                        name=row['name'],
                        sector=row['sector'],
                        sub_industry=row['sub_industry'],
                        headquarters=row['headquarters'],
                        date_added=row['date_added'],
                        cik=row['cik'],
                        founded=row['founded'],
                        has_embeddings=has_embeddings
                    )
                    session.add(stock)
                    added += 1
                
                if (added + updated) % 100 == 0:
                    print(f"  Progress: {added + updated}/{len(sp500_df)}")
            
            session.commit()
            print(f"  ✓ Added {added} new stocks")
            print(f"  ✓ Updated {updated} existing stocks")
            
        except Exception as e:
            session.rollback()
            print(f"  ✗ Error: {e}")
            raise
    
    print()
    print("=" * 70)
    print("✓ Stock metadata table populated!")
    print("=" * 70)
    print()


if __name__ == "__main__":
    populate_stocks_table()
