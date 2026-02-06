"""
Experience Memory Store for Collaborative Debate System V2.1

Stores and retrieves past trading decisions for:
- Similar episode retrieval (with regime gates)
- Outcome tracking (for accountability briefs)
- Learning from history (without RL complexity)

Key features:
- Time decay on old records
- Regime similarity filtering
- Anti-example retrieval (similar state, opposite outcome)
"""

import json
import logging
import sqlite3
import uuid
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from core.collaboration.debate_v2 import (
    AgentTurn,
    ExperienceRecord,
    ThesisType,
    retrieve_with_diversity,
)
from core.data.snapshot import GlobalMarketSnapshot

logger = logging.getLogger(__name__)


class ExperienceStore:
    """
    SQLite-backed experience store for debate learning.
    
    Stores decisions with state features and outcomes for similarity
    retrieval and accountability briefs.
    """
    
    def __init__(self, db_path: str = "data/experience.db"):
        """
        Initialize experience store.
        
        Args:
            db_path: Path to SQLite database
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()
    
    def _init_db(self) -> None:
        """Initialize database schema."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS experience_records (
                    record_id TEXT PRIMARY KEY,
                    record_date TEXT NOT NULL,
                    fund_id TEXT NOT NULL,
                    state_features TEXT NOT NULL,
                    regime_tags TEXT NOT NULL,
                    action TEXT NOT NULL,
                    symbol TEXT,
                    thesis_type TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    rationale_summary TEXT,
                    counterfactual TEXT,
                    outcome_1d REAL,
                    outcome_5d REAL,
                    outcome_21d REAL,
                    max_drawdown REAL,
                    invalidation_triggered INTEGER,
                    created_at TEXT NOT NULL
                )
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_experience_fund_date 
                ON experience_records(fund_id, record_date)
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_experience_date 
                ON experience_records(record_date)
            """)
            
            conn.commit()
    
    def record(
        self,
        snapshot: GlobalMarketSnapshot,
        fund_id: str,
        action: str,
        symbol: Optional[str],
        thesis_type: ThesisType,
        confidence: float,
        conversation: List[AgentTurn],
    ) -> str:
        """
        Record a decision for future retrieval.
        
        Args:
            snapshot: Market snapshot at decision time
            fund_id: Fund making the decision
            action: Action taken (buy/sell/hold)
            symbol: Symbol traded (if any)
            thesis_type: Type of thesis
            confidence: Overall confidence
            conversation: Full conversation history
            
        Returns:
            Record ID
        """
        record_id = str(uuid.uuid4())[:12]
        features = self._extract_feature_vector(snapshot)
        regime = self._detect_regime(snapshot)
        
        # Summarize conversation
        rationale = self._summarize_conversation(conversation)
        counterfactual = self._extract_counterfactuals(conversation)
        
        record_date = snapshot.asof_timestamp.date().isoformat()
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO experience_records (
                    record_id, record_date, fund_id, state_features, regime_tags,
                    action, symbol, thesis_type, confidence, rationale_summary,
                    counterfactual, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                record_id,
                record_date,
                fund_id,
                json.dumps(features),
                json.dumps(regime),
                action,
                symbol,
                thesis_type.value,
                confidence,
                rationale,
                counterfactual,
                datetime.utcnow().isoformat(),
            ))
            conn.commit()
        
        logger.info(f"Recorded experience {record_id} for {fund_id}")
        return record_id
    
    def retrieve_similar(
        self,
        snapshot: GlobalMarketSnapshot,
        fund_id: Optional[str] = None,
        k: int = 3,
        max_age_days: int = 252,
    ) -> Dict[str, List[ExperienceRecord]]:
        """
        Retrieve similar past episodes with diversity.
        
        Args:
            snapshot: Current market snapshot
            fund_id: Optional fund filter
            k: Number of similar records
            max_age_days: Maximum age of records
            
        Returns:
            Dict with similar, positive_precedent, negative_precedent
        """
        current_features = self._extract_feature_vector(snapshot)
        current_regime = self._detect_regime(snapshot)
        current_date = snapshot.asof_timestamp.date()
        
        # Load all records
        all_records = self._load_all(fund_id=fund_id, with_outcomes=True)
        
        # Use the retrieval function from debate_v2
        return retrieve_with_diversity(
            current_features=current_features,
            current_regime=current_regime,
            current_date=current_date,
            all_records=all_records,
            k=k,
            max_age_days=max_age_days,
        )
    
    def update_outcomes(
        self,
        record_id: str,
        outcome_1d: Optional[float] = None,
        outcome_5d: Optional[float] = None,
        outcome_21d: Optional[float] = None,
        max_drawdown: Optional[float] = None,
        invalidation_triggered: Optional[bool] = None,
    ) -> None:
        """
        Update outcomes after holding period.
        
        Called by simulation engine after N days.
        
        Args:
            record_id: Record to update
            outcome_1d: 1-day return
            outcome_5d: 5-day return
            outcome_21d: 21-day return
            max_drawdown: Maximum drawdown during period
            invalidation_triggered: Whether invalidation rules triggered
        """
        updates = []
        params = []
        
        if outcome_1d is not None:
            updates.append("outcome_1d = ?")
            params.append(outcome_1d)
        if outcome_5d is not None:
            updates.append("outcome_5d = ?")
            params.append(outcome_5d)
        if outcome_21d is not None:
            updates.append("outcome_21d = ?")
            params.append(outcome_21d)
        if max_drawdown is not None:
            updates.append("max_drawdown = ?")
            params.append(max_drawdown)
        if invalidation_triggered is not None:
            updates.append("invalidation_triggered = ?")
            params.append(1 if invalidation_triggered else 0)
        
        if not updates:
            return
        
        params.append(record_id)
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                f"UPDATE experience_records SET {', '.join(updates)} "
                f"WHERE record_id = ?",
                params
            )
            conn.commit()
        
        logger.debug(f"Updated outcomes for {record_id}")
    
    def get_latest_decision(
        self,
        fund_id: str,
        before_date: date,
    ) -> Optional[Dict[str, Any]]:
        """
        Get the most recent decision for a fund before a date.
        
        Used for accountability briefs.
        
        Args:
            fund_id: Fund ID
            before_date: Get decisions before this date
            
        Returns:
            Decision dict or None
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("""
                SELECT * FROM experience_records
                WHERE fund_id = ? AND record_date < ?
                ORDER BY record_date DESC
                LIMIT 1
            """, (fund_id, before_date.isoformat()))
            
            row = cursor.fetchone()
            if not row:
                return None
            
            return {
                "id": row["record_id"],
                "decision_date": date.fromisoformat(row["record_date"]),
                "action": row["action"],
                "symbol": row["symbol"],
                "thesis_type": row["thesis_type"],
                "confidence": row["confidence"],
                "rationale": row["rationale_summary"],
                "outcome_1d": row["outcome_1d"],
                "outcome_5d": row["outcome_5d"],
                "outcome_21d": row["outcome_21d"],
                "invalidation_triggered": bool(row["invalidation_triggered"])
                    if row["invalidation_triggered"] is not None else None,
            }
    
    def get_outcome(
        self,
        record_id: str,
        horizon: str = "5d",
    ) -> Optional[float]:
        """
        Get outcome for a specific record and horizon.
        
        Args:
            record_id: Record ID
            horizon: "1d", "5d", or "21d"
            
        Returns:
            Outcome value or None
        """
        column = f"outcome_{horizon}"
        if column not in ("outcome_1d", "outcome_5d", "outcome_21d"):
            return None
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                f"SELECT {column} FROM experience_records WHERE record_id = ?",
                (record_id,)
            )
            row = cursor.fetchone()
            return row[0] if row else None
    
    def check_invalidation_triggered(self, record_id: str) -> Optional[bool]:
        """
        Check if invalidation was triggered for a record.
        
        Args:
            record_id: Record ID
            
        Returns:
            True/False or None if not set
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "SELECT invalidation_triggered FROM experience_records "
                "WHERE record_id = ?",
                (record_id,)
            )
            row = cursor.fetchone()
            if not row or row[0] is None:
                return None
            return bool(row[0])
    
    def get_recent_accuracy(
        self,
        fund_id: str,
        lookback_days: int = 20,
    ) -> float:
        """
        Get recent accuracy (hit rate) for a fund.
        
        Args:
            fund_id: Fund ID
            lookback_days: Number of days to look back
            
        Returns:
            Accuracy as fraction (0-1)
        """
        cutoff = date.today().isoformat()
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT outcome_5d FROM experience_records
                WHERE fund_id = ? 
                AND record_date >= date(?, '-' || ? || ' days')
                AND outcome_5d IS NOT NULL
                AND action != 'hold'
            """, (fund_id, cutoff, lookback_days))
            
            outcomes = [row[0] for row in cursor.fetchall()]
        
        if not outcomes:
            return 0.5  # Default to 50% if no data
        
        # Count wins (positive outcomes)
        wins = sum(1 for o in outcomes if o > 0)
        return wins / len(outcomes)
    
    def _load_all(
        self,
        fund_id: Optional[str] = None,
        with_outcomes: bool = False,
    ) -> List[ExperienceRecord]:
        """
        Load all experience records.
        
        Args:
            fund_id: Optional fund filter
            with_outcomes: Only return records with outcomes
            
        Returns:
            List of ExperienceRecord
        """
        query = "SELECT * FROM experience_records"
        params: List[Any] = []
        conditions = []
        
        if fund_id:
            conditions.append("fund_id = ?")
            params.append(fund_id)
        
        if with_outcomes:
            conditions.append("outcome_5d IS NOT NULL")
        
        if conditions:
            query += " WHERE " + " AND ".join(conditions)
        
        records: List[ExperienceRecord] = []
        
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(query, params)
            
            for row in cursor.fetchall():
                records.append(ExperienceRecord(
                    record_id=row["record_id"],
                    record_date=date.fromisoformat(row["record_date"]),
                    fund_id=row["fund_id"],
                    state_features=json.loads(row["state_features"]),
                    regime_tags=json.loads(row["regime_tags"]),
                    action=row["action"],
                    symbol=row["symbol"],
                    thesis_type=row["thesis_type"],
                    confidence=row["confidence"],
                    rationale_summary=row["rationale_summary"] or "",
                    counterfactual=row["counterfactual"] or "",
                    outcome_1d=row["outcome_1d"],
                    outcome_5d=row["outcome_5d"],
                    outcome_21d=row["outcome_21d"],
                    max_drawdown=row["max_drawdown"],
                    invalidation_triggered=bool(row["invalidation_triggered"])
                        if row["invalidation_triggered"] is not None else None,
                ))
        
        return records
    
    def _extract_feature_vector(
        self,
        snapshot: GlobalMarketSnapshot,
    ) -> Dict[str, float]:
        """
        Extract feature vector from snapshot for similarity search.
        
        Uses cross-sectional averages of returns and volatility.
        """
        features: Dict[str, float] = {}
        
        # Average returns across symbols
        for period in ["1d", "5d", "21d", "63d"]:
            returns = [
                snapshot.get_return(sym, period)
                for sym in snapshot.coverage_symbols
            ]
            valid_returns = [r for r in returns if r is not None]
            if valid_returns:
                features[f"avg_return_{period}"] = sum(valid_returns) / len(
                    valid_returns
                )
        
        # Average volatility
        for period in ["5d", "21d"]:
            vols = [
                snapshot.get_volatility(sym, period)
                for sym in snapshot.coverage_symbols
            ]
            valid_vols = [v for v in vols if v is not None]
            if valid_vols:
                features[f"avg_volatility_{period}"] = sum(valid_vols) / len(
                    valid_vols
                )
        
        return features
    
    def _detect_regime(
        self,
        snapshot: GlobalMarketSnapshot,
    ) -> Dict[str, str]:
        """
        Detect market regime from snapshot.
        
        Returns tags for vol regime, trend regime, etc.
        """
        regime: Dict[str, str] = {}
        
        # Vol regime based on average 21d vol
        vols = [
            snapshot.get_volatility(sym, "21d")
            for sym in snapshot.coverage_symbols
        ]
        valid_vols = [v for v in vols if v is not None]
        
        if valid_vols:
            avg_vol = sum(valid_vols) / len(valid_vols)
            if avg_vol < 0.015:
                regime["vol"] = "low"
            elif avg_vol > 0.025:
                regime["vol"] = "high"
            else:
                regime["vol"] = "normal"
        
        # Trend regime based on average 21d returns
        returns = [
            snapshot.get_return(sym, "21d")
            for sym in snapshot.coverage_symbols
        ]
        valid_returns = [r for r in returns if r is not None]
        
        if valid_returns:
            avg_return = sum(valid_returns) / len(valid_returns)
            if avg_return > 0.03:
                regime["trend"] = "up"
            elif avg_return < -0.03:
                regime["trend"] = "down"
            else:
                regime["trend"] = "sideways"
        
        return regime
    
    def _summarize_conversation(
        self,
        conversation: List[AgentTurn],
    ) -> str:
        """
        Create a brief summary of the conversation.
        
        Args:
            conversation: List of agent turns
            
        Returns:
            Summary string
        """
        if not conversation:
            return ""
        
        # Get final positions
        final_turns = conversation[-2:] if len(conversation) >= 2 else conversation
        
        summaries = []
        for turn in final_turns:
            summary = (
                f"{turn.agent_id}: {turn.action} "
                f"{turn.symbol or ''} "
                f"({turn.thesis.thesis_type.value}, "
                f"conf={turn.confidence.overall():.0%})"
            )
            summaries.append(summary)
        
        return " | ".join(summaries)
    
    def _extract_counterfactuals(
        self,
        conversation: List[AgentTurn],
    ) -> str:
        """
        Extract counterfactuals from conversation.
        
        Args:
            conversation: List of agent turns
            
        Returns:
            Counterfactual summary
        """
        if not conversation:
            return ""
        
        counterfactuals = []
        for turn in conversation:
            if turn.counterfactual:
                cf = (
                    f"{turn.agent_id}: considered {turn.counterfactual.alternative_action}, "
                    f"rejected because {turn.counterfactual.why_rejected}"
                )
                counterfactuals.append(cf)
        
        return " | ".join(counterfactuals[:2])  # Limit to 2


# =============================================================================
# Accountability Brief Generator
# =============================================================================

def generate_accountability_brief(
    fund_id: str,
    current_date: date,
    experience_store: ExperienceStore,
) -> str:
    """
    Generate accountability brief from ACTUAL logged metrics.
    
    No LLM involvement - pure data from experience store.
    
    Args:
        fund_id: Fund ID
        current_date: Current simulation date
        experience_store: Experience store instance
        
    Returns:
        Accountability brief string
    """
    prior = experience_store.get_latest_decision(fund_id, before_date=current_date)
    
    if not prior:
        return "No prior decision to review."
    
    days_since = (current_date - prior["decision_date"]).days
    
    brief = f"""ACCOUNTABILITY BRIEF - {fund_id} - {current_date}

PRIOR DECISION ({prior["decision_date"]}):
- Action: {prior["action"]} {prior.get("symbol", "")}
- Thesis: {prior.get("thesis_type", "unknown")}
- Confidence: {prior.get("confidence", 0):.0%}
"""
    
    if days_since >= 1 and prior.get("symbol"):
        outcome_1d = prior.get("outcome_1d")
        outcome_5d = prior.get("outcome_5d") if days_since >= 5 else None
        
        if outcome_1d is not None:
            brief += f"""
REALIZED OUTCOME:
- 1-day return: {outcome_1d:+.2%}
"""
        
        if outcome_5d is not None:
            brief += f"- 5-day return: {outcome_5d:+.2%}\n"
        
        # Check invalidation
        invalidation = prior.get("invalidation_triggered")
        if invalidation is not None:
            brief += f"- Invalidation triggered: {'YES' if invalidation else 'No'}\n"
        
        # Calibration note
        recent_accuracy = experience_store.get_recent_accuracy(
            fund_id, lookback_days=20
        )
        if recent_accuracy < 0.4:
            brief += (
                f"\nCALIBRATION WARNING: Recent accuracy {recent_accuracy:.0%} "
                f"- consider scaling down confidence.\n"
            )
    
    return brief
