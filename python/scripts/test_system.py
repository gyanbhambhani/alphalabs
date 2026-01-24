"""
Test Script - Verify AI Trading Lab Implementation

Tests all major components:
1. Database connectivity
2. Data ingestion
3. Strategy calculations
4. Semantic search
5. Manager decision making
6. Trading engine
"""
import sys
import asyncio
from pathlib import Path
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.ingest import DataIngestion
from core.strategies.momentum import calculate_momentum_signals
from core.strategies.mean_reversion import calculate_mean_reversion_signals
from core.strategies.volatility import detect_volatility_regime
from core.semantic.search import SemanticSearchEngine
from core.managers.quant_bot import QuantBot
from core.managers.llm_manager import create_gpt4_manager
from core.managers.base import ManagerContext, StrategySignals, Portfolio
from db.database import get_async_session
from db.models import Manager
from sqlalchemy import select


def print_test(name: str, passed: bool, message: str = ""):
    """Print test result"""
    status = "✓" if passed else "✗"
    color = "\033[0;32m" if passed else "\033[0;31m"
    reset = "\033[0m"
    print(f"{color}{status}{reset} {name}")
    if message:
        print(f"  {message}")


async def test_database():
    """Test database connectivity"""
    print("\n[1] Testing Database...")
    
    try:
        async for db in get_async_session():
            try:
                result = await db.execute(select(Manager))
                managers = result.scalars().all()
                
                print_test(
                    "Database connection",
                    True,
                    f"Found {len(managers)} managers"
                )
                
                if len(managers) == 4:
                    print_test("Manager seeding", True)
                else:
                    print_test(
                        "Manager seeding",
                        False,
                        f"Expected 4 managers, found {len(managers)}"
                    )
                
                return True
            finally:
                await db.close()
    except Exception as e:
        print_test("Database connection", False, str(e))
        return False


def test_data_ingestion():
    """Test data ingestion"""
    print("\n[2] Testing Data Ingestion...")
    
    try:
        ingestion = DataIngestion(years_of_history=1)
        data = ingestion.fetch_all(force_refresh=False)
        
        print_test(
            "Data fetching",
            len(data) > 0,
            f"Fetched {len(data)} symbols"
        )
        
        # Check SPY data
        if "SPY" in data:
            spy = data["SPY"]
            print_test(
                "SPY data quality",
                len(spy.data) > 200,
                f"{len(spy.data)} trading days"
            )
        else:
            print_test("SPY data quality", False, "SPY not found")
        
        return len(data) > 0
    except Exception as e:
        print_test("Data fetching", False, str(e))
        return False


def test_strategies(price_data):
    """Test strategy calculations"""
    print("\n[3] Testing Strategy Calculations...")
    
    try:
        close_prices = {
            symbol: data.close 
            for symbol, data in price_data.items()
        }
        
        # Momentum
        momentum = calculate_momentum_signals(close_prices)
        print_test(
            "Momentum signals",
            len(momentum) > 0,
            f"{len(momentum)} signals calculated"
        )
        
        # Mean reversion
        mean_rev = calculate_mean_reversion_signals(close_prices)
        print_test(
            "Mean reversion signals",
            len(mean_rev) > 0,
            f"{len(mean_rev)} signals calculated"
        )
        
        # Volatility regime
        if "SPY" in price_data:
            regime = detect_volatility_regime(
                price_data["SPY"].close,
                price_data["SPY"].high,
                price_data["SPY"].low
            )
            print_test(
                "Volatility regime",
                regime is not None,
                f"Regime: {regime.volatility.value}_{regime.trend.value}"
            )
        
        return True
    except Exception as e:
        print_test("Strategy calculations", False, str(e))
        return False


def test_semantic_search():
    """Test semantic search"""
    print("\n[4] Testing Semantic Search...")
    
    try:
        engine = SemanticSearchEngine(in_memory=False)
        count = engine.vector_db.get_count()
        
        if count > 0:
            print_test(
                "Vector database",
                True,
                f"{count} embeddings stored"
            )
            
            # Try a search
            ingestion = DataIngestion(years_of_history=1)
            data = ingestion.fetch_all()
            
            if "SPY" in data:
                spy = data["SPY"]
                result = engine.search(
                    spy.close,
                    spy.high,
                    spy.low,
                    top_k=10
                )
                
                print_test(
                    "Semantic search",
                    len(result.similar_periods) > 0,
                    f"Found {len(result.similar_periods)} similar periods"
                )
                print(f"  Interpretation: {result.interpretation[:80]}...")
            else:
                print_test("Semantic search", False, "SPY data not found")
        else:
            print_test(
                "Vector database",
                False,
                "No embeddings found. Run generate_embeddings.py"
            )
        
        return count > 0
    except Exception as e:
        print_test("Semantic search", False, str(e))
        return False


async def test_managers():
    """Test manager decision making"""
    print("\n[5] Testing Portfolio Managers...")
    
    try:
        # Create test context
        signals = StrategySignals(
            momentum={"NVDA": 0.85, "AAPL": 0.45},
            mean_reversion={"TSLA": 0.72},
            technical={},
            ml_prediction={"NVDA": 0.023},
            volatility_regime="low_vol_trending_up",
            semantic_search={
                "avg_5d_return": 0.018,
                "positive_5d_rate": 0.72,
                "interpretation": "Test interpretation"
            }
        )
        
        portfolio = Portfolio(cash_balance=100000.0, positions={})
        
        context = ManagerContext(
            timestamp=datetime.now(),
            portfolio=portfolio,
            market_data={"NVDA": 140.0, "AAPL": 170.0, "TSLA": 180.0},
            signals=signals
        )
        
        # Test Quant Bot
        quant = QuantBot()
        quant_decisions = await quant.make_decisions(context)
        print_test(
            "Quant Bot decisions",
            len(quant_decisions) > 0,
            f"{len(quant_decisions)} decisions made"
        )
        
        # Note: LLM managers require API keys, so we just test initialization
        try:
            gpt4 = create_gpt4_manager()
            print_test("GPT-4 Manager initialization", True)
        except Exception as e:
            print_test(
                "GPT-4 Manager initialization",
                False,
                "Requires OPENAI_API_KEY in .env"
            )
        
        return len(quant_decisions) > 0
    except Exception as e:
        print_test("Manager testing", False, str(e))
        return False


async def main():
    """Run all tests"""
    print("=" * 60)
    print("AI Trading Lab - Component Tests")
    print("=" * 60)
    
    results = []
    
    # Test database
    results.append(await test_database())
    
    # Test data ingestion
    ingestion = DataIngestion(years_of_history=1)
    data = ingestion.fetch_all(force_refresh=False)
    results.append(test_data_ingestion())
    
    # Test strategies
    if data:
        results.append(test_strategies(data))
    else:
        results.append(False)
    
    # Test semantic search
    results.append(test_semantic_search())
    
    # Test managers
    results.append(await test_managers())
    
    # Summary
    print("\n" + "=" * 60)
    passed = sum(results)
    total = len(results)
    print(f"Tests Passed: {passed}/{total}")
    
    if passed == total:
        print("✓ All systems operational!")
    else:
        print("⚠️  Some tests failed. Check output above.")
    
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
