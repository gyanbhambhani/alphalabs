"""
Debate Runner Module - Multi-Agent Trading Decision System

This module provides AI-powered debate systems for making trading decisions.
Multiple debate implementations are available, each with different architectures.

Architecture:
- debate_common.py: Shared dataclasses, LLM initialization, parsing utilities
- debate_runner_v1.py: Legacy 3-phase debate (ANALYZE -> PROPOSE -> DECIDE)
- debate_runner_v2.py: Collaborative investment committee with pre-screening
- screening.py: Data-first candidate screening with strategy-specific scoring

Usage:
    from core.backtest.debate_runner import get_debate_runner
    
    # Get V1 runner (legacy)
    runner_v1 = get_debate_runner(version="v1")
    
    # Get V2 runner (recommended)
    runner_v2 = get_debate_runner(version="v2", 
                                   analyst_model="gpt-4o-mini",
                                   critic_model="claude-3-haiku-20240307")
    
    # Run debate
    decision = await runner.run_debate(
        fund_id="fund1",
        fund_name="Momentum Fund",
        fund_thesis="...",
        portfolio=portfolio,
        snapshot=snapshot,
        simulation_date=date.today()
    )

Debate Systems:

V1 - DailyDebateRunner (DEPRECATED):
    3-phase sequential debate:
    1. ANALYZE: GPT analyzes market conditions
    2. PROPOSE: GPT proposes specific trades
    3. DECIDE: Claude makes final decision
    
    Limitations:
    - No pre-screening of candidates
    - Sequential phases without collaboration
    - Limited signal validation
    
V2 - CollaborativeDebateRunner (RECOMMENDED):
    Investment committee with data-first approach:
    1. SCREEN: Score all stocks by strategy signals
    2. PRESENT: Show top candidates to both agents
    3. DEBATE: Multi-round collaborative discussion
    4. DECIDE: Converge on single stock with validated confidence
    
    Advantages:
    - Pre-screened candidates with actual signals
    - Collaborative debate with evidence validation
    - Confidence decomposition with min thresholds
    - Integrated experience memory (can be disabled for backtesting)
"""

from typing import Union

# Import debate runners
from core.backtest.debate_runner_v1 import DailyDebateRunner
from core.backtest.debate_runner_v2 import CollaborativeDebateRunner

# Import shared utilities for convenience
from core.backtest.debate_common import (
    DebateMessage,
    TradingDecision,
    get_llm,
    extract_json,
)


def get_debate_runner(
    version: str = "v2",
    **kwargs,
) -> Union[DailyDebateRunner, CollaborativeDebateRunner]:
    """
    Factory function to get the appropriate debate runner.
    
    Args:
        version: "v1" for DailyDebateRunner (legacy), "v2" for CollaborativeDebateRunner (recommended)
        **kwargs: Arguments to pass to runner constructor
        
    Returns:
        Debate runner instance
        
    Examples:
        # V1 runner (legacy)
        runner = get_debate_runner(
            version="v1",
            analyze_model="gpt-4o-mini",
            propose_model="gpt-4o-mini",
            decide_model="claude-3-5-sonnet-20241022"
        )
        
        # V2 runner (recommended)
        runner = get_debate_runner(
            version="v2",
            analyst_model="gpt-4o-mini",
            critic_model="claude-3-haiku-20240307",
            disable_experience_store=True  # For backtesting
        )
    """
    if version == "v1":
        return DailyDebateRunner(**kwargs)
    elif version == "v2":
        return CollaborativeDebateRunner(**kwargs)
    else:
        raise ValueError(f"Unknown debate runner version: {version}. Use 'v1' or 'v2'.")


__all__ = [
    "get_debate_runner",
    "DailyDebateRunner",
    "CollaborativeDebateRunner",
    "DebateMessage",
    "TradingDecision",
    "get_llm",
    "extract_json",
]
