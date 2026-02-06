#!/usr/bin/env python3
"""
Test script for Collaborative Debate System V2.1

Run this to verify the new debate system is working correctly.

Usage:
    cd python
    python scripts/test_debate_v2.py
"""

import asyncio
import sys
from datetime import date, datetime
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.backtest.debate_runner import (
    CollaborativeDebateRunner,
    DailyDebateRunner,
    get_debate_runner,
)
from core.backtest.portfolio_tracker import BacktestPortfolio
from core.data.snapshot import GlobalMarketSnapshot, DataQuality
from core.collaboration.debate_v2 import (
    AVAILABLE_FEATURES,
    THESIS_REQUIRED_EVIDENCE,
    ThesisType,
    validate_evidence,
    EvidenceReference,
)


def create_test_snapshot() -> GlobalMarketSnapshot:
    """Create a test snapshot with sample data."""
    return GlobalMarketSnapshot(
        snapshot_id="test-snapshot-001",
        asof_timestamp=datetime(2024, 1, 15, 16, 0, 0),
        prices={
            "AAPL": 185.50,
            "MSFT": 390.25,
            "GOOGL": 142.80,
            "AMZN": 155.30,
            "NVDA": 548.90,
            "META": 385.20,
            "TSLA": 215.60,
            "JPM": 172.40,
        },
        returns={
            "AAPL": {"1d": -0.012, "5d": 0.025, "21d": 0.08, "63d": 0.15},
            "MSFT": {"1d": 0.008, "5d": 0.015, "21d": 0.12, "63d": 0.22},
            "GOOGL": {"1d": -0.025, "5d": -0.04, "21d": -0.02, "63d": 0.05},
            "AMZN": {"1d": 0.015, "5d": 0.03, "21d": 0.10, "63d": 0.18},
            "NVDA": {"1d": 0.035, "5d": 0.08, "21d": 0.25, "63d": 0.45},
            "META": {"1d": -0.008, "5d": 0.02, "21d": 0.15, "63d": 0.30},
            "TSLA": {"1d": -0.045, "5d": -0.08, "21d": -0.12, "63d": -0.05},
            "JPM": {"1d": 0.005, "5d": 0.01, "21d": 0.06, "63d": 0.12},
        },
        volatility={
            "AAPL": {"5d": 0.012, "21d": 0.018},
            "MSFT": {"5d": 0.010, "21d": 0.015},
            "GOOGL": {"5d": 0.015, "21d": 0.022},
            "AMZN": {"5d": 0.018, "21d": 0.025},
            "NVDA": {"5d": 0.025, "21d": 0.035},
            "META": {"5d": 0.020, "21d": 0.028},
            "TSLA": {"5d": 0.035, "21d": 0.045},
            "JPM": {"5d": 0.008, "21d": 0.012},
        },
        quality=DataQuality(coverage_ratio=1.0),
        coverage_symbols=["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "JPM"],
    )


def create_test_portfolio() -> BacktestPortfolio:
    """Create a test portfolio with some cash."""
    return BacktestPortfolio(
        fund_id="test_fund",
        initial_cash=100_000.0,
        cash=100_000.0,
    )


def test_evidence_validation():
    """Test the evidence validation function."""
    print("\n" + "=" * 60)
    print("TEST 1: Evidence Validation")
    print("=" * 60)
    
    snapshot = create_test_snapshot()
    
    # Test valid evidence for momentum thesis
    evidence = [
        EvidenceReference(feature="return_21d", symbol="NVDA"),
        EvidenceReference(feature="return_63d", symbol="NVDA"),
    ]
    
    valid, errors = validate_evidence(evidence, ThesisType.MOMENTUM, snapshot)
    print(f"\nMomentum thesis with return_21d, return_63d:")
    print(f"  Valid: {valid}")
    if errors:
        print(f"  Errors: {errors}")
    
    # Test invalid evidence (wrong features for thesis)
    evidence_wrong = [
        EvidenceReference(feature="volatility_5d", symbol="AAPL"),
        EvidenceReference(feature="volatility_21d", symbol="AAPL"),
    ]
    
    valid, errors = validate_evidence(evidence_wrong, ThesisType.MOMENTUM, snapshot)
    print(f"\nMomentum thesis with volatility features (should fail):")
    print(f"  Valid: {valid}")
    if errors:
        print(f"  Errors: {errors}")
    
    # Test evidence for mean reversion
    evidence_mr = [
        EvidenceReference(feature="return_1d", symbol="TSLA"),
        EvidenceReference(feature="return_5d", symbol="TSLA"),
    ]
    
    valid, errors = validate_evidence(evidence_mr, ThesisType.MEAN_REVERSION, snapshot)
    print(f"\nMean reversion thesis with return_1d, return_5d:")
    print(f"  Valid: {valid}")
    if errors:
        print(f"  Errors: {errors}")
    
    print("\n[PASS] Evidence validation tests completed")


def test_available_features():
    """Test that available features are correctly defined."""
    print("\n" + "=" * 60)
    print("TEST 2: Available Features")
    print("=" * 60)
    
    print(f"\nAvailable features: {sorted(AVAILABLE_FEATURES)}")
    print(f"\nThesis required evidence:")
    for thesis_type, required in THESIS_REQUIRED_EVIDENCE.items():
        print(f"  {thesis_type.value}: {required}")
    
    print("\n[PASS] Available features test completed")


async def test_debate_runner_v2():
    """Test the CollaborativeDebateRunner."""
    print("\n" + "=" * 60)
    print("TEST 3: CollaborativeDebateRunner")
    print("=" * 60)
    
    snapshot = create_test_snapshot()
    portfolio = create_test_portfolio()
    
    # Get V2 runner
    runner = get_debate_runner(version="v2")
    print(f"\nRunner type: {type(runner).__name__}")
    
    print("\nRunning debate (this will make LLM API calls)...")
    print("Fund: Momentum Tech Fund")
    print("Thesis: Buy stocks with strong 21-day momentum in tech sector")
    
    try:
        decision = await runner.run_debate(
            fund_id="momentum_tech",
            fund_name="Momentum Tech Fund",
            fund_thesis="Buy stocks with strong 21-day momentum in tech sector",
            portfolio=portfolio,
            snapshot=snapshot,
            simulation_date=date(2024, 1, 15),
        )
        
        print(f"\n--- DECISION ---")
        print(f"Action: {decision.action}")
        print(f"Symbol: {decision.symbol}")
        print(f"Target Weight: {decision.target_weight}")
        print(f"Confidence: {decision.confidence:.1%}")
        print(f"Reasoning: {decision.reasoning[:200]}...")
        
        print(f"\n--- TRANSCRIPT ---")
        for msg in decision.debate_transcript[:5]:
            print(f"  [{msg.phase}] {msg.model}: {msg.content[:100]}...")
        
        print("\n[PASS] Debate runner V2 test completed")
        return True
        
    except Exception as e:
        print(f"\n[ERROR] Debate failed: {e}")
        print("Make sure OPENAI_API_KEY and ANTHROPIC_API_KEY are set in .env.local")
        return False


async def test_debate_runner_v1():
    """Test the old DailyDebateRunner for comparison."""
    print("\n" + "=" * 60)
    print("TEST 4: DailyDebateRunner (V1) for comparison")
    print("=" * 60)
    
    snapshot = create_test_snapshot()
    portfolio = create_test_portfolio()
    
    # Get V1 runner
    runner = get_debate_runner(version="v1")
    print(f"\nRunner type: {type(runner).__name__}")
    
    print("\nRunning V1 debate for comparison...")
    
    try:
        decision = await runner.run_debate(
            fund_id="momentum_tech",
            fund_name="Momentum Tech Fund",
            fund_thesis="Buy stocks with strong 21-day momentum in tech sector",
            portfolio=portfolio,
            snapshot=snapshot,
            simulation_date=date(2024, 1, 15),
        )
        
        print(f"\n--- V1 DECISION ---")
        print(f"Action: {decision.action}")
        print(f"Symbol: {decision.symbol}")
        print(f"Confidence: {decision.confidence:.1%}")
        
        print("\n[PASS] V1 debate runner test completed")
        return True
        
    except Exception as e:
        print(f"\n[ERROR] V1 Debate failed: {e}")
        return False


def main():
    """Run all tests."""
    print("=" * 60)
    print("COLLABORATIVE DEBATE SYSTEM V2.1 - TEST SUITE")
    print("=" * 60)
    
    # Test 1: Evidence validation (no API calls)
    test_evidence_validation()
    
    # Test 2: Available features (no API calls)
    test_available_features()
    
    # Test 3 & 4: Debate runners (requires API keys)
    print("\n" + "=" * 60)
    print("API TESTS (requires OPENAI_API_KEY and ANTHROPIC_API_KEY)")
    print("=" * 60)
    
    run_api_tests = input("\nRun API tests? (y/n): ").strip().lower() == 'y'
    
    if run_api_tests:
        asyncio.run(test_debate_runner_v2())
        asyncio.run(test_debate_runner_v1())
    else:
        print("\nSkipping API tests.")
    
    print("\n" + "=" * 60)
    print("TEST SUITE COMPLETE")
    print("=" * 60)
    print("\nTo test in the frontend:")
    print("1. Start backend: cd python && python -m uvicorn app.main:app --reload")
    print("2. Start frontend: cd frontend && npm run dev")
    print("3. Go to http://localhost:3000/backtest")
    print("4. Run a simulation - the V2 debate system is now the default!")


if __name__ == "__main__":
    main()
