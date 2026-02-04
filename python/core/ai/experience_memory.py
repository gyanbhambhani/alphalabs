"""
Experience Memory - Retrieval-based learning

Stores and retrieves past trades for contextual learning.

Before each decision, the system can query:
- "Show me similar past trades"
- "What happened when we were in this state before?"
- "What's the win rate for this signal?"

This enables:
1. Experience replay (RL-style)
2. Contextual bandit learning
3. LLM can see past outcomes as context
"""

from typing import List, Optional, Dict, Tuple
from datetime import date
import numpy as np
import logging
from dataclasses import dataclass

from sqlalchemy import select

from db.models import ExperienceRecord
from core.backtest.persistence import BacktestPersistence

logger = logging.getLogger(__name__)


@dataclass
class SimilarTrade:
    """A similar past trade retrieved from memory."""
    decision_id: str
    symbol: str
    action: str
    weight: float
    regime: str
    outcome_21d: float
    outcome_5d: Optional[float]
    alpha_vs_spy: Optional[float]
    win: bool
    similarity_score: float  # Cosine similarity
    decision_date: date


@dataclass
class AggregateStats:
    """Aggregate statistics from similar trades."""
    count: int
    win_rate: float
    median_return_21d: float
    mean_return_21d: float
    median_alpha: float
    volatility: float


class ExperienceMemory:
    """
    Stores and retrieves past trades for contextual learning.
    
    Uses cosine similarity on normalized feature vectors.
    """
    
    def __init__(self, persistence: BacktestPersistence):
        """
        Initialize experience memory.
        
        Args:
            persistence: Persistence layer for DB access
        """
        self.persistence = persistence
    
    def store_experience(
        self,
        decision_id: str,
        run_id: str,
        fund_id: str,
        feature_vector: List[float],
        action: str,
        symbol: Optional[str] = None,
        weight: Optional[float] = None,
        regime: Optional[str] = None,
        decision_date: Optional[date] = None,
    ) -> str:
        """
        Store a trade in experience memory.
        
        Args:
            decision_id: Decision ID
            run_id: Run ID
            fund_id: Fund ID
            feature_vector: Normalized feature vector
            action: "buy", "sell", "hold"
            symbol: Asset symbol
            weight: Position weight
            regime: Market regime
            decision_date: Date of decision
            
        Returns:
            Experience record ID
        """
        return self.persistence.save_experience_record(
            decision_id=decision_id,
            run_id=run_id,
            fund_id=fund_id,
            feature_vector=feature_vector,
            action=action,
            symbol=symbol,
            weight=weight,
            regime=regime,
            decision_date=decision_date,
        )
    
    def retrieve_similar(
        self,
        feature_vector: List[float],
        k: int = 5,
        fund_id: Optional[str] = None,
        action_filter: Optional[str] = None,
        regime_filter: Optional[str] = None,
        min_outcome_data: bool = True,
    ) -> List[SimilarTrade]:
        """
        Retrieve k most similar past trades.
        
        Uses cosine similarity on feature vectors.
        
        Args:
            feature_vector: Current state's feature vector
            k: Number of similar trades to retrieve
            fund_id: Filter by fund (optional)
            action_filter: Filter by action (optional)
            regime_filter: Filter by regime (optional)
            min_outcome_data: Only return trades with outcomes filled in
            
        Returns:
            List of similar trades, sorted by similarity
        """
        # Normalize query vector
        query_norm = self._normalize_vector(feature_vector)
        
        # Fetch experiences from DB
        with self.persistence.get_session() as session:
            query = select(ExperienceRecord)
            
            # Apply filters
            if fund_id:
                query = query.where(ExperienceRecord.fund_id == fund_id)
            if action_filter:
                query = query.where(ExperienceRecord.action == action_filter)
            if regime_filter:
                query = query.where(ExperienceRecord.regime == regime_filter)
            if min_outcome_data:
                query = query.where(ExperienceRecord.outcome_21d.isnot(None))
            
            experiences = session.execute(query).scalars().all()
        
        if not experiences:
            logger.debug("No experiences found matching filters")
            return []
        
        # Compute similarities
        similarities = []
        for exp in experiences:
            stored_vector = exp.feature_vector
            if not stored_vector:
                continue
            
            # Compute cosine similarity
            similarity = self._cosine_similarity(query_norm, stored_vector)
            
            similarities.append((similarity, exp))
        
        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x[0], reverse=True)
        
        # Return top k
        results = []
        for sim_score, exp in similarities[:k]:
            results.append(SimilarTrade(
                decision_id=exp.decision_id,
                symbol=exp.symbol or "N/A",
                action=exp.action,
                weight=exp.weight or 0.0,
                regime=exp.regime or "unknown",
                outcome_21d=exp.outcome_21d or 0.0,
                outcome_5d=exp.outcome_5d,
                alpha_vs_spy=exp.alpha_vs_spy,
                win=exp.win or False,
                similarity_score=sim_score,
                decision_date=exp.decision_date,
            ))
        
        return results
    
    def get_aggregate_stats(
        self,
        similar_trades: List[SimilarTrade],
    ) -> Optional[AggregateStats]:
        """
        Compute aggregate statistics from similar trades.
        
        Args:
            similar_trades: List of similar trades
            
        Returns:
            Aggregate statistics
        """
        if not similar_trades:
            return None
        
        # Extract outcomes
        returns_21d = [t.outcome_21d for t in similar_trades if t.outcome_21d is not None]
        alphas = [t.alpha_vs_spy for t in similar_trades if t.alpha_vs_spy is not None]
        wins = [t.win for t in similar_trades]
        
        if not returns_21d:
            return None
        
        return AggregateStats(
            count=len(similar_trades),
            win_rate=sum(wins) / len(wins) if wins else 0.0,
            median_return_21d=float(np.median(returns_21d)),
            mean_return_21d=float(np.mean(returns_21d)),
            median_alpha=float(np.median(alphas)) if alphas else 0.0,
            volatility=float(np.std(returns_21d)),
        )
    
    def format_for_llm_context(
        self,
        similar_trades: List[SimilarTrade],
        aggregate_stats: Optional[AggregateStats] = None,
    ) -> str:
        """
        Format similar trades as LLM context string.
        
        Args:
            similar_trades: Retrieved similar trades
            aggregate_stats: Aggregate statistics
            
        Returns:
            Formatted string for LLM prompt
        """
        if not similar_trades:
            return "EXPERIENCE MEMORY: No similar past trades found.\n"
        
        context = "EXPERIENCE MEMORY (similar past trades):\n\n"
        
        for i, trade in enumerate(similar_trades, 1):
            context += (
                f"{i}. {trade.symbol} - {trade.action.upper()} "
                f"{trade.weight:.1%} (similarity: {trade.similarity_score:.2f})\n"
                f"   Regime: {trade.regime}\n"
                f"   Outcome 21d: {trade.outcome_21d:+.1%}"
            )
            if trade.alpha_vs_spy is not None:
                context += f" (alpha: {trade.alpha_vs_spy:+.1%})"
            context += f" - {'WIN' if trade.win else 'LOSS'}\n"
        
        if aggregate_stats:
            context += f"\nAGGREGATE STATS ({aggregate_stats.count} similar trades):\n"
            context += f"- Win rate: {aggregate_stats.win_rate:.0%}\n"
            context += f"- Median 21d return: {aggregate_stats.median_return_21d:+.1%}\n"
            context += f"- Median alpha: {aggregate_stats.median_alpha:+.1%}\n"
            context += f"- Volatility: {aggregate_stats.volatility:.1%}\n"
        
        context += "\n"
        return context
    
    def _normalize_vector(self, vector: List[float]) -> np.ndarray:
        """Normalize vector to unit length."""
        arr = np.array(vector, dtype=float)
        norm = np.linalg.norm(arr)
        if norm > 0:
            return arr / norm
        return arr
    
    def _cosine_similarity(
        self,
        vec1: np.ndarray,
        vec2: List[float],
    ) -> float:
        """
        Compute cosine similarity between two vectors.
        
        Args:
            vec1: First vector (normalized)
            vec2: Second vector (list)
            
        Returns:
            Similarity score (0-1)
        """
        vec2_arr = np.array(vec2, dtype=float)
        vec2_norm = np.linalg.norm(vec2_arr)
        
        if vec2_norm == 0:
            return 0.0
        
        vec2_unit = vec2_arr / vec2_norm
        
        # Cosine similarity
        similarity = np.dot(vec1, vec2_unit)
        
        # Clamp to [0, 1] (handle floating point errors)
        return float(np.clip(similarity, 0, 1))
    
    def count_experiences(
        self,
        fund_id: Optional[str] = None,
        run_id: Optional[str] = None,
    ) -> int:
        """
        Count stored experiences.
        
        Args:
            fund_id: Filter by fund
            run_id: Filter by run
            
        Returns:
            Number of experiences
        """
        with self.persistence.get_session() as session:
            query = select(ExperienceRecord)
            
            if fund_id:
                query = query.where(ExperienceRecord.fund_id == fund_id)
            if run_id:
                query = query.where(ExperienceRecord.run_id == run_id)
            
            count = len(session.execute(query).scalars().all())
        
        return count


def create_experience_memory(
    persistence: BacktestPersistence,
) -> ExperienceMemory:
    """
    Factory function to create experience memory.
    
    Args:
        persistence: Persistence instance
        
    Returns:
        Configured ExperienceMemory
    """
    return ExperienceMemory(persistence)
