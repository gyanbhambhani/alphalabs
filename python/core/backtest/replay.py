"""
Decision Replay Bundle

Step-through debugging for AI decisions.

Provides complete visibility into:
- State before decision (portfolio, risk, budget)
- All candidates considered
- Agent debate transcript
- Validation results
- Risk checks
- Execution details
- Outcomes (post-hoc)

Enables:
- Replay any decision with different parameters
- Diff two decisions to see what changed
- Audit trail for regulatory/debugging
"""

from dataclasses import dataclass, asdict
from datetime import date
from typing import Dict, List, Optional, Any
import json
import logging

from core.backtest.persistence import BacktestPersistence
from db.models import (
    BacktestDecisionRecord,
    BacktestDecisionCandidate,
    BacktestTradeRecord,
    BacktestPortfolioSnapshotRecord,
)

logger = logging.getLogger(__name__)


@dataclass
class DecisionBundle:
    """
    Complete decision state for replay/debugging.
    
    Everything needed to understand and replay a decision.
    """
    # Core identifiers
    decision_id: str
    run_id: str
    fund_id: str
    decision_date: date
    simulation_day: int
    
    # State before decision
    portfolio_before: Dict[str, Any]
    risk_state: Dict[str, Any]
    trade_budget: Dict[str, Any]
    
    # Candidates
    candidate_set: List[Dict[str, Any]]
    selected_asset_id: Optional[str]
    
    # Agent debate
    agent_messages: List[Dict[str, Any]]
    models_used: Dict[str, str]
    tokens_used: int
    
    # Decision output
    action: str
    symbol: Optional[str]
    target_weight: Optional[float]
    confidence: float
    reasoning: str
    
    # Validation
    validation_result: Optional[Dict[str, Any]] = None
    risk_check_result: Optional[Dict[str, Any]] = None
    
    # Execution
    orders_created: List[Dict[str, Any]] = None
    orders_filled: List[Dict[str, Any]] = None
    
    # Outcome (filled post-hoc)
    outcome_1d: Optional[float] = None
    outcome_5d: Optional[float] = None
    outcome_21d: Optional[float] = None
    realized_pnl: Optional[float] = None
    
    # State after decision
    portfolio_after: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)
    
    def to_json(self, indent: int = 2) -> str:
        """Convert to formatted JSON string."""
        return json.dumps(self.to_dict(), indent=indent, default=str)


class DecisionReplayService:
    """
    Service for retrieving and replaying decisions.
    
    Provides observability into AI decision-making.
    """
    
    def __init__(self, persistence: BacktestPersistence):
        """
        Initialize replay service.
        
        Args:
            persistence: Persistence layer for DB access
        """
        self.persistence = persistence
    
    def get_decision_bundle(
        self,
        decision_id: str,
    ) -> Optional[DecisionBundle]:
        """
        Get complete decision bundle for a decision.
        
        Args:
            decision_id: Decision ID to retrieve
            
        Returns:
            DecisionBundle with full state, or None if not found
        """
        with self.persistence.get_session() as session:
            # Get decision record
            decision = session.query(BacktestDecisionRecord).filter_by(
                id=decision_id
            ).first()
            
            if not decision:
                logger.warning(f"Decision {decision_id} not found")
                return None
            
            # Get candidates
            candidates = session.query(BacktestDecisionCandidate).filter_by(
                decision_id=decision_id
            ).all()
            
            # Get portfolio snapshot before
            portfolio_before = session.query(BacktestPortfolioSnapshotRecord).filter_by(
                run_id=decision.run_id,
                fund_id=decision.fund_id,
                snapshot_date=decision.decision_date,
            ).first()
            
            # Get trades executed
            trades = session.query(BacktestTradeRecord).filter_by(
                run_id=decision.run_id,
                fund_id=decision.fund_id,
                trade_date=decision.decision_date,
            ).all()
            
            # Build candidate set
            candidate_set = []
            selected_asset_id = None
            
            for cand in candidates:
                candidate_dict = {
                    "symbol": cand.symbol,
                    "sector": cand.sector,
                    "selected": cand.selected,
                    "features": cand.features,
                    "scores": cand.scores,
                    "target_weight": cand.target_weight,
                    "outcome_21d": cand.outcome_21d,
                }
                candidate_set.append(candidate_dict)
                
                if cand.selected:
                    selected_asset_id = cand.symbol
            
            # Build portfolio state
            portfolio_state = {}
            if portfolio_before:
                portfolio_state = {
                    "total_value": float(portfolio_before.total_value),
                    "cash": float(portfolio_before.cash),
                    "invested_pct": float(portfolio_before.invested_pct),
                    "positions": portfolio_before.positions,
                    "cumulative_return": float(portfolio_before.cumulative_return or 0),
                    "max_drawdown": float(portfolio_before.max_drawdown or 0),
                }
            
            # Build trade info
            orders_filled = []
            for trade in trades:
                if trade.symbol == decision.symbol:
                    orders_filled.append({
                        "symbol": trade.symbol,
                        "direction": trade.direction,
                        "quantity": float(trade.quantity),
                        "price": float(trade.price),
                        "commission": float(trade.commission),
                        "value": float(trade.value),
                    })
            
            # Build bundle
            bundle = DecisionBundle(
                decision_id=decision.id,
                run_id=decision.run_id,
                fund_id=decision.fund_id,
                decision_date=decision.decision_date,
                simulation_day=0,  # TODO: Calculate day number
                portfolio_before=portfolio_state,
                risk_state={},  # TODO: Add risk state
                trade_budget={},  # TODO: Add budget state
                candidate_set=candidate_set,
                selected_asset_id=selected_asset_id,
                agent_messages=decision.debate_transcript or [],
                models_used=decision.models_used or {},
                tokens_used=decision.tokens_used,
                action=decision.action,
                symbol=decision.symbol,
                target_weight=decision.target_weight,
                confidence=decision.confidence or 0.0,
                reasoning=decision.reasoning or "",
                orders_filled=orders_filled,
            )
            
            return bundle
    
    def list_decisions(
        self,
        run_id: str,
        fund_id: Optional[str] = None,
        date_range: Optional[tuple] = None,
        action: Optional[str] = None,
        rejected: Optional[bool] = None,
        limit: int = 100,
    ) -> List[Dict]:
        """
        List decisions with filters.
        
        Args:
            run_id: Backtest run ID
            fund_id: Filter by fund
            date_range: (start_date, end_date) tuple
            action: Filter by action ("buy", "sell", "hold")
            rejected: Filter by rejected status
            limit: Maximum results
            
        Returns:
            List of decision summaries
        """
        with self.persistence.get_session() as session:
            query = session.query(BacktestDecisionRecord).filter_by(
                run_id=run_id
            )
            
            if fund_id:
                query = query.filter_by(fund_id=fund_id)
            
            if action:
                query = query.filter_by(action=action)
            
            if date_range:
                start_date, end_date = date_range
                query = query.filter(
                    BacktestDecisionRecord.decision_date >= start_date,
                    BacktestDecisionRecord.decision_date <= end_date,
                )
            
            # TODO: Add rejected filter (needs validation_result field)
            
            query = query.limit(limit)
            decisions = query.all()
            
            # Build summaries
            summaries = []
            for decision in decisions:
                summaries.append({
                    "decision_id": decision.id,
                    "fund_id": decision.fund_id,
                    "date": decision.decision_date.isoformat(),
                    "action": decision.action,
                    "symbol": decision.symbol,
                    "target_weight": decision.target_weight,
                    "confidence": decision.confidence,
                    "tokens_used": decision.tokens_used,
                })
            
            return summaries
    
    def replay_decision(
        self,
        decision_id: str,
        overrides: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        """
        Replay a decision with optional parameter overrides.
        
        Useful for testing: "What if we used lower temperature?"
        
        Args:
            decision_id: Decision to replay
            overrides: Dict of parameters to override
                - temperature: float
                - top_k_candidates: int
                - min_confidence: float
                
        Returns:
            Dict with replay results
        """
        bundle = self.get_decision_bundle(decision_id)
        
        if not bundle:
            return {"error": "Decision not found"}
        
        # TODO: Actually re-run the debate with overrides
        # For now, return the original bundle
        
        return {
            "original": bundle.to_dict(),
            "overrides": overrides or {},
            "replay_result": "Not yet implemented",
        }
    
    def diff_decisions(
        self,
        decision_id_1: str,
        decision_id_2: str,
    ) -> Dict[str, Any]:
        """
        Compare two decisions to see what changed.
        
        Useful for: "Why did we buy on Day 10 but not Day 11?"
        
        Args:
            decision_id_1: First decision
            decision_id_2: Second decision
            
        Returns:
            Dict with differences
        """
        bundle1 = self.get_decision_bundle(decision_id_1)
        bundle2 = self.get_decision_bundle(decision_id_2)
        
        if not bundle1 or not bundle2:
            return {"error": "One or both decisions not found"}
        
        # Compute differences
        diffs = {
            "action_changed": bundle1.action != bundle2.action,
            "symbol_changed": bundle1.symbol != bundle2.symbol,
            "confidence_delta": bundle2.confidence - bundle1.confidence,
            "tokens_delta": bundle2.tokens_used - bundle1.tokens_used,
            "portfolio_value_delta": (
                bundle2.portfolio_before.get("total_value", 0) -
                bundle1.portfolio_before.get("total_value", 0)
            ),
        }
        
        return {
            "decision_1": bundle1.to_dict(),
            "decision_2": bundle2.to_dict(),
            "differences": diffs,
        }


def create_replay_service(
    persistence: BacktestPersistence,
) -> DecisionReplayService:
    """
    Factory function to create replay service.
    
    Args:
        persistence: Persistence instance
        
    Returns:
        Configured DecisionReplayService
    """
    return DecisionReplayService(persistence)
