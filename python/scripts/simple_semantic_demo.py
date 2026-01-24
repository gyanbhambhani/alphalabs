"""
Simple Semantic Search Demo

Demonstrates ChromaDB semantic search by directly querying
similar market states and their forward returns.
"""
import sys
from pathlib import Path
from datetime import datetime, timedelta

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.ingest import fetch_stock_data
from core.semantic.encoder import MarketStateEncoder
from core.semantic.vector_db import VectorDatabase


def main():
    print("🔍 Semantic Market Memory - Simple Demo")
    print("=" * 70)
    print()
    
    # Step 1: Fetch current market data
    print("[1/3] Fetching current market data...")
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)
    
    spy_data = fetch_stock_data('SPY', 
                                start_date=start_date.strftime('%Y-%m-%d'),
                                end_date=end_date.strftime('%Y-%m-%d'))
    
    if not spy_data:
        print("  ❌ Failed to fetch market data")
        return
    
    print(f"  ✓ Current price: ${spy_data.data['close'].iloc[-1]:.2f}")
    print()
    
    # Step 2: Initialize components
    print("[2/3] Initializing Semantic Search...")
    
    encoder = MarketStateEncoder()
    vector_db = VectorDatabase(persist_directory="./chroma_data")
    
    total_states = vector_db.get_count()
    print(f"  ✓ Loaded {total_states:,} historical market states")
    print()
    
    # Step 3: Encode and search
    print("[3/3] Finding Similar Market Conditions...")
    
    latest_date = spy_data.data.index[-1].strftime('%Y-%m-%d')
    
    current_state = encoder.encode(
        date=latest_date,
        close=spy_data.data['close'],
        high=spy_data.data['high'],
        low=spy_data.data['low'],
        volume=spy_data.data['volume']
    )
    
    print(f"\n  Today's Market ({latest_date}):")
    print(f"    • 1M Return: {current_state.metadata.get('return_1m', 0):>7.2%}")
    print(f"    • 3M Return: {current_state.metadata.get('return_3m', 0):>7.2%}")
    print(f"    • Volatility: {current_state.metadata.get('volatility_21d', 0):>7.2%}")
    print()
    
    # Direct vector search
    search_results = vector_db.search(current_state.vector, top_k=10)
    
    print("  🔍 Top 10 Most Similar Historical Periods:")
    print("  " + "-" * 66)
    print(f"  {'Date':<12} {'Similarity':<12} {'Then +1M':<12} {'Future +1M':<12}")
    print("  " + "-" * 66)
    
    has_forward_returns = False
    for result in search_results:
        similarity_pct = result.similarity * 100
        then_return = result.metadata.get('return_1m', 0)
        future_return = result.metadata.get('forward_1m_return', None)
        
        if future_return is not None:
            has_forward_returns = True
            outcome = '📈' if future_return > 0 else '📉'
            print(f"  {result.date:<12} {similarity_pct:>6.1f}%      "
                  f"{then_return:>6.2%}        {future_return:>6.2%} {outcome}")
        else:
            print(f"  {result.date:<12} {similarity_pct:>6.1f}%      "
                  f"{then_return:>6.2%}        N/A")
    
    print("  " + "-" * 66)
    print()
    
    if has_forward_returns:
        # Calculate statistics
        forward_returns = [r.metadata.get('forward_1m_return', 0) 
                          for r in search_results 
                          if r.metadata.get('forward_1m_return') is not None]
        
        if forward_returns:
            avg = sum(forward_returns) / len(forward_returns)
            positive = sum(1 for r in forward_returns if r > 0)
            
            print("  📊 Historical Pattern Analysis:")
            print(f"    • Average outcome after 1 month: {avg:>7.2%}")
            print(f"    • Positive outcomes: {positive}/{len(forward_returns)} "
                  f"({positive/len(forward_returns)*100:.0f}%)")
            print()
            
            if avg > 0.02:
                signal = "🟢 BULLISH"
            elif avg < -0.02:
                signal = "🔴 BEARISH"
            else:
                signal = "⚪ NEUTRAL"
            
            print(f"  💡 Signal: {signal}")
            print()
    else:
        print("  ⚠️  No forward return data found in embeddings")
        print("     Run generate_embeddings.py to add forward returns")
        print()
    
    print("=" * 70)
    print("✅ Semantic search complete!")
    print()


if __name__ == "__main__":
    main()
