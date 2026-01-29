"""
Verify ChromaDB Contents (Fast - names only)
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import chromadb


def main():
    print("=" * 60)
    print("ChromaDB Verification (Fast)")
    print("=" * 60)
    print()
    
    print("Connecting...", flush=True)
    client = chromadb.PersistentClient(path="./chroma_data")
    
    print("Listing collections...", flush=True)
    collections = client.list_collections()
    
    # Just count names - NO data queries
    default_col = None
    symbols = []
    
    for col in collections:
        if col.name == "market_states":
            default_col = col.name
        elif col.name.startswith("market_states_"):
            symbols.append(col.name.replace("market_states_", ""))
    
    print()
    print(f"Total collections: {len(collections)}")
    print(f"Default collection: {'Yes' if default_col else 'No'}")
    print(f"Per-symbol collections: {len(symbols)}")
    print()
    
    # Show some symbol names
    symbols.sort()
    print(f"Sample symbols (first 20): {', '.join(symbols[:20])}")
    print(f"Sample symbols (last 20): {', '.join(symbols[-20:])}")
    print()
    
    # Quick count from just ONE collection to verify data exists
    if symbols:
        test_symbol = "AAPL" if "AAPL" in symbols else symbols[0]
        print(f"Testing {test_symbol} collection...", flush=True)
        try:
            test_col = client.get_collection(f"market_states_{test_symbol}")
            count = test_col.count()
            print(f"  {test_symbol} has {count:,} embeddings")
            
            # Quick sample
            sample = test_col.get(limit=1, include=['metadatas'])
            if sample['ids']:
                meta = sample['metadatas'][0]
                print(f"  Sample: date={meta.get('date')}, "
                      f"price=${meta.get('price', 0):.2f}")
        except Exception as e:
            print(f"  Error: {e}")
    
    print()
    print("=" * 60)
    if len(symbols) > 0:
        print(f"SUCCESS: {len(symbols)} stocks have embeddings in ChromaDB")
    else:
        print("WARNING: No per-symbol collections found")
    print("=" * 60)


if __name__ == "__main__":
    main()
