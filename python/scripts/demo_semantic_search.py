"""
Test Semantic Search

Demonstrates the ChromaDB semantic search feature - finding similar
historical market conditions and their outcomes.
"""
import sys
from pathlib import Path
from datetime import datetime, timedelta

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.ingest import fetch_stock_data
from core.semantic.encoder import MarketStateEncoder
from core.semantic.search import SemanticSearchEngine
from core.semantic.vector_db import VectorDatabase


def main():
    print("🔍 AI Trading Lab - Semantic Market Memory Demo")
    print("=" * 70)
    print()
    
    # Step 1: Fetch current market data
    print("[1/4] Fetching current market data...")
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)  # Get 1 year for encoding
    
    spy_data = fetch_stock_data('SPY', 
                                start_date=start_date.strftime('%Y-%m-%d'),
                                end_date=end_date.strftime('%Y-%m-%d'))
    
    if not spy_data:
        print("  ❌ Failed to fetch market data")
        return
    
    print(f"  ✓ Fetched {len(spy_data.data)} days of SPY data")
    print(f"  ✓ Current price: ${spy_data.data['close'].iloc[-1]:.2f}")
    print()
    
    # Step 2: Initialize semantic search
    print("[2/4] Initializing Semantic Search Engine...")
    
    encoder = MarketStateEncoder()
    vector_db = VectorDatabase(persist_directory="./chroma_data")
    search_engine = SemanticSearchEngine(encoder, vector_db)
    
    total_states = vector_db.get_count()
    print(f"  ✓ ChromaDB loaded with {total_states:,} historical market states")
    print()
    
    # Step 3: Encode current market state
    print("[3/4] Encoding Current Market State...")
    
    # Get the latest date
    latest_date = spy_data.data.index[-1].strftime('%Y-%m-%d')
    
    current_state = encoder.encode(
        date=latest_date,
        close=spy_data.data['close'],
        high=spy_data.data['high'],
        low=spy_data.data['low'],
        volume=spy_data.data['volume']
    )
    
    print(f"  📅 Date: {current_state.date}")
    print(f"  💰 Price: ${current_state.metadata.get('price', 0):.2f}")
    print()
    print("  Market Characteristics:")
    print(f"    • 1 Week Return:  {current_state.metadata.get('return_1w', 0):>7.2%}")
    print(f"    • 1 Month Return: {current_state.metadata.get('return_1m', 0):>7.2%}")
    print(f"    • 3 Month Return: {current_state.metadata.get('return_3m', 0):>7.2%}")
    print(f"    • 6 Month Return: {current_state.metadata.get('return_6m', 0):>7.2%}")
    print(f"    • 5D Volatility:  {current_state.metadata.get('volatility_5d', 0):>7.2%}")
    print(f"    • 21D Volatility: {current_state.metadata.get('volatility_21d', 0):>7.2%}")
    print()
    
    # Step 4: Search for similar periods (THE MAGIC!)
    print("[4/4] 🔍 Searching for Similar Historical Periods...")
    print()
    
    # Search using price series
    results = search_engine.search(
        close=spy_data.data['close'],
        high=spy_data.data['high'],
        low=spy_data.data['low'],
        volume=spy_data.data['volume'],
        top_k=10,
        exclude_recent_days=0  # Don't exclude any - we want to see all results
    )
    
    print("  📊 Top 10 Most Similar Market Conditions:")
    print("  " + "-" * 66)
    print(f"  {'Date':<12} {'Similarity':<12} {'1M Ret':<10} {'Future 1M':<12}")
    print("  " + "-" * 66)
    
    for result in results.similar_periods[:10]:
        similarity_pct = result.similarity * 100
        hist_return = result.metadata.get('return_1m', 0)
        forward_ret = result.metadata.get('forward_1m_return', 0)
        
        # Color code based on outcome
        outcome_arrow = '📈' if forward_ret > 0 else '📉'
        
        print(f"  {result.date:<12} {similarity_pct:>6.1f}%      "
              f"{hist_return:>6.2%}     {forward_ret:>6.2%} {outcome_arrow}")
    
    print("  " + "-" * 66)
    print()
    
    # Check if we got results
    if not results.similar_periods:
        print("  ⚠️  No similar periods found!")
        print("  This might mean:")
        print("    • The ChromaDB embeddings don't have forward returns")
        print("    • The exclude_recent_days filter removed all results")
        print()
        return
    
    # Summary statistics
    forward_returns = [r.metadata.get('forward_1m_return', 0) 
                      for r in results.similar_periods]
    avg_forward = sum(forward_returns) / len(forward_returns) if forward_returns else 0
    positive_count = sum(1 for r in forward_returns if r > 0)
    
    print("  📈 Historical Outcome Analysis:")
    print(f"    • Average 1M forward return: {avg_forward:>7.2%}")
    if forward_returns:
        print(f"    • Positive outcomes: {positive_count}/{len(forward_returns)} "
              f"({positive_count/len(forward_returns)*100:.0f}%)")
    else:
        print(f"    • No forward return data available")
    print()
    
    print("  💡 AI Interpretation:")
    print(f"  {results.interpretation}")
    print()
    
    print("=" * 70)
    print("✅ Semantic Search Complete!")
    print()
    print("This is how AI managers make decisions:")
    print("  1. Encode current market conditions into a 512-dim vector")
    print("  2. Find the most similar historical periods using cosine similarity")
    print("  3. Analyze what happened after those similar periods")
    print("  4. Use those patterns to inform trading decisions")
    print()


if __name__ == "__main__":
    main()
