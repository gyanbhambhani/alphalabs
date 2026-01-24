"""
Verify ChromaDB Contents

Quick script to check what's stored in the vector database.
"""
import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.semantic.vector_db import VectorDatabase


def main():
    print("🔍 Checking ChromaDB...")
    print("=" * 60)
    
    # Initialize DB
    db = VectorDatabase()
    
    # Get collection stats
    collection = db.collection
    count = collection.count()
    
    print(f"\n✅ Collection: {collection.name}")
    print(f"✅ Total embeddings: {count:,}")
    
    if count > 0:
        # Get more samples to show actual data
        results = collection.get(limit=20, include=['embeddings', 'metadatas'])
        
        print(f"\n📊 Stored Data Points (showing 20 of {count:,}):")
        print("-" * 80)
        
        # Table header
        print(f"{'Date':<12} {'1W Ret':<8} {'1M Ret':<8} {'3M Ret':<8} "
              f"{'21D Vol':<8} {'RSI':<8} {'Regime':<10}")
        print("-" * 80)
        
        for id, metadata in zip(results['ids'], results['metadatas']):
            date = metadata.get('date', 'N/A')
            ret_1w = metadata.get('return_1w', 0)
            ret_1m = metadata.get('return_1m', 0)
            ret_3m = metadata.get('return_3m', 0)
            vol_21d = metadata.get('volatility_21d', 0)
            rsi = metadata.get('rsi', 0)
            regime = metadata.get('regime', 'N/A')
            
            print(f"{date:<12} {ret_1w:>6.2%}  {ret_1m:>6.2%}  {ret_3m:>6.2%}  "
                  f"{vol_21d:>6.2%}  {rsi:>6.1f}  {regime:<10}")
        
        # Show date range
        all_ids = collection.get(include=[])['ids']
        dates = sorted(all_ids)
        
        print("\n" + "=" * 80)
        print(f"📅 Date Range: {dates[0]} to {dates[-1]}")
        print(f"📊 Total Trading Days: {len(dates):,}")
        
        # Check embedding dimension
        sample_embedding = results['embeddings'][0]
        print(f"🔢 Embedding Dimension: {len(sample_embedding)}")
        
        # Show available metadata fields
        sample_metadata = results['metadatas'][0]
        print(f"\n📋 Metadata Fields ({len(sample_metadata)} total):")
        for key in sorted(sample_metadata.keys())[:15]:  # Show first 15
            value = sample_metadata[key]
            if isinstance(value, float):
                print(f"   • {key}: {value:.4f}")
            else:
                print(f"   • {key}: {value}")
        
        if len(sample_metadata) > 15:
            print(f"   ... and {len(sample_metadata) - 15} more fields")
        
        print("\n" + "=" * 80)
        print("✅ ChromaDB is working correctly!")
        print("\n💡 You can now:")
        print("   1. Start the backend API")
        print("   2. Query semantic market memory")
        print("   3. Run trading simulations")
        
    else:
        print("\n⚠️  No embeddings found!")
        print("   Run: python3.11 scripts/generate_embeddings.py")
    
    print()


if __name__ == "__main__":
    main()
