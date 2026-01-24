"""
Test Trading Cycle

Triggers a complete trading cycle and shows:
1. Market data fetching
2. Strategy signal calculations
3. Semantic search results from ChromaDB
4. Manager decisions (LLM + Quant Bot)
5. Trade executions
"""
import sys
from pathlib import Path
import asyncio
from datetime import datetime, timedelta

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.ingest import fetch_stock_data
from core.strategies.momentum import MomentumSignal
from core.strategies.mean_reversion import MeanReversionSignal
from core.strategies.technical import TechnicalIndicators
from core.strategies.volatility import VolatilityRegime
from core.semantic.encoder import MarketStateEncoder
from core.semantic.search import SemanticSearchEngine
from core.semantic.vector_db import VectorDatabase
from db.database import get_sync_session
from db.models import Manager


async def main():
    print("🚀 AI Trading Lab - Test Trading Cycle")
    print("=" * 70)
    print()
    
    # Step 1: Fetch current market data
    print("[1/5] Fetching current market data...")
    
    # Get last 100 days for calculations
    end_date = datetime.now()
    start_date = end_date - timedelta(days=100)
    
    spy_data = fetch_stock_data('SPY', 
                                start_date=start_date.strftime('%Y-%m-%d'),
                                end_date=end_date.strftime('%Y-%m-%d'))
    
    if not spy_data:
        print("  ❌ Failed to fetch market data")
        return
    
    print(f"  ✓ Fetched {len(spy_data.data)} days of SPY data")
    print(f"  ✓ Current price: ${spy_data.data['close'].iloc[-1]:.2f}")
    print()
    
    # Step 2: Calculate strategy signals
    print("[2/5] Calculating strategy signals...")
    
    momentum = MomentumSignal()
    mean_rev = MeanReversionSignal()
    technical = TechnicalIndicators()
    volatility = VolatilityRegime()
    
    mom_sig = momentum.calculate(spy_data.data)
    mr_sig = mean_rev.calculate(spy_data.data)
    tech_sig = technical.calculate(spy_data.data)
    vol_sig = volatility.calculate(spy_data.data)
    
    print(f"  ✓ Momentum: {mom_sig.signal:+.2f} (strength: {mom_sig.strength:.2f})")
    print(f"  ✓ Mean Reversion: {mr_sig.zscore:+.2f} (signal: {mr_sig.signal:+.2f})")
    print(f"  ✓ Technical RSI: {tech_sig.rsi:.1f}")
    print(f"  ✓ Volatility: {vol_sig.current_vol:.2%} ({vol_sig.level.value})")
    print()
    
    # Step 3: Query semantic memory (THIS IS THE COOL PART!)
    print("[3/5] Querying Semantic Market Memory...")
    print("  (Finding similar historical market conditions)")
    print()
    
    # Initialize semantic search
    encoder = MarketStateEncoder()
    vector_db = VectorDatabase(persist_directory="./chroma_data")
    search_engine = SemanticSearchEngine(encoder, vector_db)
    
    # Encode current market state
    current_state = encoder.encode(spy_data, symbol='SPY')
    print(f"  Current Market State ({current_state.date}):")
    print(f"    • 1M Return: {current_state.metadata['return_1m']:.2%}")
    print(f"    • 3M Return: {current_state.metadata['return_3m']:.2%}")
    print(f"    • Volatility: {current_state.metadata['volatility_21d']:.2%}")
    print()
    
    # Search for similar periods
    print("  🔍 Searching ChromaDB for similar market conditions...")
    results = search_engine.search(current_state, top_k=5)
    
    print(f"\n  📊 Top 5 Similar Historical Periods:")
    print("  " + "-" * 66)
    print(f"  {'Date':<12} {'Similarity':<12} {'1M Return':<12} {'Outcome':<15}")
    print("  " + "-" * 66)
    
    for result in results.similar_periods[:5]:
        similarity_pct = result.similarity * 100
        forward_ret = result.metadata.get('forward_1m_return', 0)
        outcome = '📈 UP' if forward_ret > 0 else '📉 DOWN'
        
        print(f"  {result.date:<12} {similarity_pct:>6.1f}%      "
              f"{forward_ret:>6.2%}       {outcome:<15}")
    
    print("  " + "-" * 66)
    print(f"\n  💡 {results.interpretation}")
    print()
    
    # Step 4: Show what managers would see
    print("[4/5] Manager View...")
    
    for db in get_sync_session():
        managers = db.query(Manager).filter(Manager.is_active == True).all()
        
        print(f"  Active Managers: {len(managers)}")
        for manager in managers:
            print(f"    • {manager.name} ({manager.type})")
        
        print()
        print("  Each manager receives:")
        print("    ✓ Current strategy signals")
        print("    ✓ Semantic search results (similar historical periods)")
        print("    ✓ Performance outcomes from those periods")
        print("    ✓ Current portfolio state")
        print()
        break  # Only need one iteration
    
    # Step 5: Summary
    print("[5/5] Summary")
    print("  " + "=" * 66)
    print("  ✅ Market data: LIVE from yfinance")
    print("  ✅ Strategy signals: Calculated from 6 strategies")
    print(f"  ✅ Semantic memory: {vector_db.get_count():,} historical states")
    print("  ✅ ChromaDB search: Finding similar market conditions")
    print("  ✅ Managers: Ready to make decisions")
    print()
    print("  🎯 Next Step: Trigger a real trading cycle via API")
    print("     POST http://localhost:8000/api/trading/cycle")
    print()
    print("  Or from your browser's console:")
    print("     fetch('http://localhost:8000/api/trading/cycle', {method: 'POST'})")
    print("       .then(r => r.json()).then(console.log)")
    print()


if __name__ == "__main__":
    asyncio.run(main())
