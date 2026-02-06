"""
Backtest Replay API Routes

Provides observability into AI decision-making:
- List decisions with filters
- Get full decision bundle
- Replay decisions with overrides
- Diff decisions

This enables step-through debugging of the AI fund system.
"""

from fastapi import APIRouter, HTTPException, Query
from typing import Optional, Dict, Any
from datetime import date

from core.backtest.replay import DecisionReplayService
from core.backtest.persistence import BacktestPersistence

router = APIRouter(prefix="/api/backtest/replay", tags=["backtest-replay"])


# Initialize persistence (singleton)
_persistence: Optional[BacktestPersistence] = None


def get_persistence() -> BacktestPersistence:
    """Get or create persistence instance."""
    global _persistence
    if _persistence is None:
        _persistence = BacktestPersistence()
    return _persistence


def get_replay_service() -> DecisionReplayService:
    """Get replay service instance."""
    persistence = get_persistence()
    from core.backtest.replay import create_replay_service
    return create_replay_service(persistence)


@router.get("/decisions")
async def list_decisions(
    run_id: str,
    fund_id: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    action: Optional[str] = Query(None, pattern="^(buy|sell|hold)$"),
    limit: int = Query(100, ge=1, le=1000),
) -> Dict[str, Any]:
    """
    List decisions with filters.
    
    Query params:
    - run_id: Backtest run ID (required)
    - fund_id: Filter by fund (optional)
    - start_date: Start date YYYY-MM-DD (optional)
    - end_date: End date YYYY-MM-DD (optional)
    - action: Filter by action (buy/sell/hold)
    - limit: Max results (default 100)
    
    Returns:
        List of decision summaries
    """
    service = get_replay_service()
    
    # Parse date range
    date_range = None
    if start_date and end_date:
        date_range = (
            date.fromisoformat(start_date),
            date.fromisoformat(end_date)
        )
    
    decisions = service.list_decisions(
        run_id=run_id,
        fund_id=fund_id,
        date_range=date_range,
        action=action,
        limit=limit,
    )
    
    return {
        "run_id": run_id,
        "count": len(decisions),
        "decisions": decisions,
    }


@router.get("/decisions/{decision_id}")
async def get_decision_bundle(
    decision_id: str,
) -> Dict[str, Any]:
    """
    Get complete decision bundle.
    
    Returns full state:
    - Portfolio before/after
    - All candidates
    - Agent debate transcript
    - Validation results
    - Execution details
    - Outcomes
    
    Path params:
    - decision_id: Decision ID
    
    Returns:
        Complete DecisionBundle
    """
    service = get_replay_service()
    
    bundle = service.get_decision_bundle(decision_id)
    
    if not bundle:
        raise HTTPException(status_code=404, detail="Decision not found")
    
    return bundle.to_dict()


@router.post("/decisions/{decision_id}/replay")
async def replay_decision(
    decision_id: str,
    overrides: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Replay a decision with optional parameter overrides.
    
    Useful for testing: "What if we used lower temperature?"
    
    Path params:
    - decision_id: Decision to replay
    
    Body:
    - overrides: Dict of parameters to override
      - temperature: float
      - top_k_candidates: int
      - min_confidence: float
    
    Returns:
        Original and replayed results
    """
    service = get_replay_service()
    
    result = service.replay_decision(decision_id, overrides)
    
    if "error" in result:
        raise HTTPException(status_code=404, detail=result["error"])
    
    return result


@router.get("/decisions/{decision_id}/diff/{other_decision_id}")
async def diff_decisions(
    decision_id: str,
    other_decision_id: str,
) -> Dict[str, Any]:
    """
    Compare two decisions to see what changed.
    
    Useful for: "Why did we buy on Day 10 but not Day 11?"
    
    Path params:
    - decision_id: First decision
    - other_decision_id: Second decision
    
    Returns:
        Differences between decisions
    """
    service = get_replay_service()
    
    result = service.diff_decisions(decision_id, other_decision_id)
    
    if "error" in result:
        raise HTTPException(status_code=404, detail=result["error"])
    
    return result
