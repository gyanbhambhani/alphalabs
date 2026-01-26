"""
Populate Stock Metadata Table

Loads S&P 500 stock metadata into the PostgreSQL database
and syncs with ChromaDB embeddings status.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from db import get_sync_session, Stock
from core.semantic.vector_db import VectorDatabase
from scripts.fetch_sp500_list import load_sp500_list, fetch_sp500_list, clean_sp500_data


def populate_stocks_table():
    """Populate stocks table with S&P 500 data"""
    
    print("=" * 70)
    print("Populate Stock Metadata Table")
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
    
    # Check ChromaDB for embeddings
    print("[2/3] Checking ChromaDB for embeddings...")
    symbols_with_embeddings = VectorDatabase.get_all_symbols("./chroma_data")
    print(f"  ✓ Found {len(symbols_with_embeddings)} stocks with embeddings")
    print()
    
    # Populate database
    print("[3/3] Populating database...")
    
    added = 0
    updated = 0
    
    with get_sync_session() as session:
        try:
            for _, row in sp500_df.iterrows():
                symbol = row['symbol']
                
                # Check if stock exists
                stock = session.query(Stock).filter(
                    Stock.symbol == symbol
                ).first()
                
                has_embeddings = symbol in symbols_with_embeddings
                
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
                    
                    if has_embeddings:
                        # Get embedding stats
                        vdb = VectorDatabase(symbol=symbol)
                        stock.embeddings_count = vdb.get_count()
                        date_range = vdb.get_date_range()
                        stock.embeddings_date_range_start = date_range[0]
                        stock.embeddings_date_range_end = date_range[1]
                    
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
                    
                    if has_embeddings:
                        vdb = VectorDatabase(symbol=symbol)
                        stock.embeddings_count = vdb.get_count()
                        date_range = vdb.get_date_range()
                        stock.embeddings_date_range_start = date_range[0]
                        stock.embeddings_date_range_end = date_range[1]
                    
                    session.add(stock)
                    added += 1
                
                if (added + updated) % 50 == 0:
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
    print("✓ Stock metadata table populated successfully!")
    print("=" * 70)
    print()


if __name__ == "__main__":
    populate_stocks_table()
