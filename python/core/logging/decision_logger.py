"""
Decision Logger

Logs every aspect of trading decisions for analysis and learning.
Provides full "consciousness" tracking of AI decision-making.
"""
import json
import uuid
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum

from sqlalchemy import (
    Column, String, Integer, Float, Boolean, DateTime,
    Text, JSON, DECIMAL, Index, ForeignKey
)
from sqlalchemy.orm import relationship

from db.models import Base


class DecisionStatus(Enum):
    """Status of a trading decision"""
    PENDING = "pending"
    EXECUTED = "executed"
    PARTIAL = "partial"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


@dataclass
class HistoricalMatch:
    """A historical period that influenced the decision"""
    date: str
    similarity: float
    regime: str
    forward_return_1m: float
    forward_return_3m: float
    narrative: str
    geopolitical_context: str


@dataclass
class DecisionLog:
    """Complete log of a trading decision"""
    decision_id: str
    manager_id: str
    timestamp: datetime
    
    # Trade details
    symbol: str
    action: str  # buy, sell, hold
    size: float
    price: float
    
    # The thesis (narrative)
    thesis: str
    conviction: float
    
    # Historical context
    historical_matches: List[HistoricalMatch]
    top_match_date: str
    top_match_similarity: float
    
    # Quantitative context
    sharpe_ratio_expected: float
    sortino_ratio_expected: float
    optimal_weight: float
    portfolio_weight_actual: float
    
    # Market context
    market_regime: str
    volatility_current: float
    momentum_1m: float
    momentum_3m: float
    
    # Geopolitical context
    geopolitical_factors: List[str]
    
    # Risk assessment
    expected_return: float
    max_drawdown_expected: float
    stop_loss: Optional[float]
    target_return: Optional[float]
    
    # Signals that influenced decision
    signals_used: Dict[str, float] = field(default_factory=dict)
    
    # Status
    status: DecisionStatus = DecisionStatus.PENDING
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage"""
        result = asdict(self)
        result['status'] = self.status.value
        result['timestamp'] = self.timestamp.isoformat()
        result['historical_matches'] = [
            asdict(m) for m in self.historical_matches
        ]
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DecisionLog':
        """Create from dictionary"""
        data['status'] = DecisionStatus(data['status'])
        data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        data['historical_matches'] = [
            HistoricalMatch(**m) for m in data['historical_matches']
        ]
        return cls(**data)


@dataclass
class DecisionOutcome:
    """Actual outcome of a decision"""
    decision_id: str
    
    # Actual results
    actual_return: float
    holding_period_days: int
    exit_price: float
    exit_date: datetime
    exit_reason: str
    
    # Comparison to expectation
    return_vs_expected: float
    was_correct_direction: bool
    
    # Market conditions at exit
    exit_regime: str
    exit_volatility: float


@dataclass
class PerformanceNarrative:
    """Post-trade analysis narrative"""
    decision_id: str
    
    # The story
    narrative: str
    
    # Key learnings
    what_worked: str
    what_failed: str
    key_difference_from_precedent: str
    
    # Scores
    prediction_accuracy: float
    regime_prediction_correct: bool
    
    # Recommendations
    would_do_differently: str


# SQLAlchemy model for database storage
class DecisionLogModel(Base):
    """Database model for decision logs"""
    __tablename__ = "decision_logs"
    
    id = Column(String(50), primary_key=True)
    manager_id = Column(String(50), ForeignKey("managers.id"), nullable=False)
    timestamp = Column(DateTime, nullable=False, default=datetime.utcnow)
    
    # Trade details
    symbol = Column(String(10), nullable=False)
    action = Column(String(10), nullable=False)
    size = Column(DECIMAL(10, 4), nullable=False)
    price = Column(DECIMAL(18, 8), nullable=True)
    
    # Thesis
    thesis = Column(Text, nullable=True)
    conviction = Column(DECIMAL(5, 4), nullable=True)
    
    # Historical context
    top_match_date = Column(String(20), nullable=True)
    top_match_similarity = Column(DECIMAL(5, 4), nullable=True)
    historical_matches = Column(JSON, nullable=True)
    
    # Quantitative
    sharpe_expected = Column(DECIMAL(8, 4), nullable=True)
    sortino_expected = Column(DECIMAL(8, 4), nullable=True)
    optimal_weight = Column(DECIMAL(5, 4), nullable=True)
    
    # Market context
    market_regime = Column(String(50), nullable=True)
    volatility = Column(DECIMAL(8, 6), nullable=True)
    momentum_1m = Column(DECIMAL(8, 6), nullable=True)
    momentum_3m = Column(DECIMAL(8, 6), nullable=True)
    
    # Geopolitical
    geopolitical_factors = Column(JSON, nullable=True)
    
    # Risk
    expected_return = Column(DECIMAL(8, 6), nullable=True)
    max_drawdown_expected = Column(DECIMAL(8, 6), nullable=True)
    stop_loss = Column(DECIMAL(8, 6), nullable=True)
    target_return = Column(DECIMAL(8, 6), nullable=True)
    
    # Signals
    signals_used = Column(JSON, nullable=True)
    
    # Status and outcome
    status = Column(String(20), default="pending")
    actual_return = Column(DECIMAL(8, 6), nullable=True)
    holding_period_days = Column(Integer, nullable=True)
    exit_reason = Column(Text, nullable=True)
    
    # Performance narrative
    performance_narrative = Column(Text, nullable=True)
    
    __table_args__ = (
        Index("idx_decision_logs_manager", "manager_id"),
        Index("idx_decision_logs_timestamp", "timestamp"),
        Index("idx_decision_logs_symbol", "symbol"),
    )


class DecisionLogger:
    """
    Comprehensive decision logging system.
    
    Logs every aspect of trading decisions for:
    - Transparency and auditability
    - Post-trade analysis
    - Strategy improvement
    - LLM learning and feedback
    """
    
    def __init__(self, db_session_factory=None):
        """
        Initialize the logger.
        
        Args:
            db_session_factory: Factory for database sessions
        """
        self.db_session_factory = db_session_factory
        self._memory_logs: Dict[str, DecisionLog] = {}
        self._outcomes: Dict[str, DecisionOutcome] = {}
    
    def log_decision(
        self,
        manager_id: str,
        symbol: str,
        action: str,
        size: float,
        price: float,
        thesis: str,
        conviction: float,
        historical_matches: List[HistoricalMatch],
        market_regime: str,
        volatility: float,
        momentum_1m: float,
        momentum_3m: float,
        sharpe_expected: float,
        sortino_expected: float,
        optimal_weight: float,
        expected_return: float,
        max_drawdown_expected: float,
        geopolitical_factors: List[str],
        signals_used: Dict[str, float],
        stop_loss: Optional[float] = None,
        target_return: Optional[float] = None,
    ) -> str:
        """
        Log a complete trading decision.
        
        Returns:
            decision_id for tracking
        """
        decision_id = f"dec_{uuid.uuid4().hex[:12]}"
        
        log = DecisionLog(
            decision_id=decision_id,
            manager_id=manager_id,
            timestamp=datetime.utcnow(),
            symbol=symbol,
            action=action,
            size=size,
            price=price,
            thesis=thesis,
            conviction=conviction,
            historical_matches=historical_matches,
            top_match_date=historical_matches[0].date if historical_matches else "",
            top_match_similarity=(
                historical_matches[0].similarity if historical_matches else 0
            ),
            sharpe_ratio_expected=sharpe_expected,
            sortino_ratio_expected=sortino_expected,
            optimal_weight=optimal_weight,
            portfolio_weight_actual=size,
            market_regime=market_regime,
            volatility_current=volatility,
            momentum_1m=momentum_1m,
            momentum_3m=momentum_3m,
            geopolitical_factors=geopolitical_factors,
            expected_return=expected_return,
            max_drawdown_expected=max_drawdown_expected,
            stop_loss=stop_loss,
            target_return=target_return,
            signals_used=signals_used,
            status=DecisionStatus.PENDING
        )
        
        # Store in memory
        self._memory_logs[decision_id] = log
        
        # Store in database if available
        if self.db_session_factory:
            self._save_to_db(log)
        
        return decision_id
    
    def _save_to_db(self, log: DecisionLog) -> None:
        """Save decision log to database"""
        try:
            with self.db_session_factory() as session:
                db_log = DecisionLogModel(
                    id=log.decision_id,
                    manager_id=log.manager_id,
                    timestamp=log.timestamp,
                    symbol=log.symbol,
                    action=log.action,
                    size=log.size,
                    price=log.price,
                    thesis=log.thesis,
                    conviction=log.conviction,
                    top_match_date=log.top_match_date,
                    top_match_similarity=log.top_match_similarity,
                    historical_matches=[
                        asdict(m) for m in log.historical_matches
                    ],
                    sharpe_expected=log.sharpe_ratio_expected,
                    sortino_expected=log.sortino_ratio_expected,
                    optimal_weight=log.optimal_weight,
                    market_regime=log.market_regime,
                    volatility=log.volatility_current,
                    momentum_1m=log.momentum_1m,
                    momentum_3m=log.momentum_3m,
                    geopolitical_factors=log.geopolitical_factors,
                    expected_return=log.expected_return,
                    max_drawdown_expected=log.max_drawdown_expected,
                    stop_loss=log.stop_loss,
                    target_return=log.target_return,
                    signals_used=log.signals_used,
                    status=log.status.value
                )
                session.add(db_log)
                session.commit()
        except Exception as e:
            print(f"Error saving decision log to database: {e}")
    
    def update_status(
        self,
        decision_id: str,
        status: DecisionStatus
    ) -> None:
        """Update decision status"""
        if decision_id in self._memory_logs:
            self._memory_logs[decision_id].status = status
        
        if self.db_session_factory:
            try:
                with self.db_session_factory() as session:
                    log = session.query(DecisionLogModel).filter(
                        DecisionLogModel.id == decision_id
                    ).first()
                    if log:
                        log.status = status.value
                        session.commit()
            except Exception as e:
                print(f"Error updating decision status: {e}")
    
    def record_outcome(
        self,
        decision_id: str,
        actual_return: float,
        holding_period_days: int,
        exit_price: float,
        exit_reason: str,
        exit_regime: str,
        exit_volatility: float
    ) -> DecisionOutcome:
        """
        Record the actual outcome of a decision.
        """
        log = self._memory_logs.get(decision_id)
        if not log:
            raise ValueError(f"Decision {decision_id} not found")
        
        outcome = DecisionOutcome(
            decision_id=decision_id,
            actual_return=actual_return,
            holding_period_days=holding_period_days,
            exit_price=exit_price,
            exit_date=datetime.utcnow(),
            exit_reason=exit_reason,
            return_vs_expected=actual_return - log.expected_return,
            was_correct_direction=(
                (actual_return > 0 and log.action == "buy") or
                (actual_return < 0 and log.action == "sell")
            ),
            exit_regime=exit_regime,
            exit_volatility=exit_volatility
        )
        
        self._outcomes[decision_id] = outcome
        
        # Update database
        if self.db_session_factory:
            try:
                with self.db_session_factory() as session:
                    db_log = session.query(DecisionLogModel).filter(
                        DecisionLogModel.id == decision_id
                    ).first()
                    if db_log:
                        db_log.actual_return = actual_return
                        db_log.holding_period_days = holding_period_days
                        db_log.exit_reason = exit_reason
                        db_log.status = DecisionStatus.EXECUTED.value
                        session.commit()
            except Exception as e:
                print(f"Error recording outcome: {e}")
        
        return outcome
    
    def generate_performance_narrative(
        self,
        decision_id: str
    ) -> PerformanceNarrative:
        """
        Generate a post-trade analysis narrative.
        
        Explains what happened vs expectations and key learnings.
        """
        log = self._memory_logs.get(decision_id)
        outcome = self._outcomes.get(decision_id)
        
        if not log:
            raise ValueError(f"Decision {decision_id} not found")
        if not outcome:
            raise ValueError(f"Outcome for {decision_id} not recorded")
        
        # Generate narrative
        direction_correct = "correct" if outcome.was_correct_direction else "incorrect"
        magnitude_diff = outcome.return_vs_expected
        
        if outcome.actual_return > 0:
            result_desc = f"gain of {outcome.actual_return:+.1%}"
        else:
            result_desc = f"loss of {outcome.actual_return:.1%}"
        
        top_match = log.historical_matches[0] if log.historical_matches else None
        
        narrative = f"""
Trade Analysis for {log.symbol} ({log.action.upper()})

## Summary
The trade resulted in a {result_desc} over {outcome.holding_period_days} days.
Direction prediction was {direction_correct}.
Expected return was {log.expected_return:+.1%}, actual was {outcome.actual_return:+.1%}.

## Historical Context Used
"""
        
        if top_match:
            narrative += f"""
You thought this was similar to {top_match.date} 
({top_match.geopolitical_context}, similarity: {top_match.similarity:.1%}).
That period saw {top_match.forward_return_1m:+.1%} over the next month.
"""
        
        # What worked / what failed
        if outcome.was_correct_direction:
            what_worked = (
                f"Direction call was correct. "
                f"Historical pattern matching identified the right regime."
            )
            if magnitude_diff > 0.02:
                what_failed = (
                    f"Magnitude was underestimated by {magnitude_diff:.1%}. "
                    "Could have sized larger."
                )
            elif magnitude_diff < -0.02:
                what_failed = (
                    f"Magnitude was overestimated by {abs(magnitude_diff):.1%}. "
                    "Expectations were too aggressive."
                )
            else:
                what_failed = "Execution was close to plan."
        else:
            what_worked = "Risk management kept losses contained." if (
                outcome.actual_return > -0.05
            ) else "Limited what worked in this trade."
            what_failed = (
                f"Direction was wrong. Expected {log.expected_return:+.1%}, "
                f"got {outcome.actual_return:+.1%}."
            )
        
        # Key difference from precedent
        key_diff = "Market conditions evolved differently than the historical match."
        if top_match:
            if outcome.exit_regime != log.market_regime:
                key_diff = (
                    f"Regime shifted from {log.market_regime} to "
                    f"{outcome.exit_regime} during holding period."
                )
        
        # Would do differently
        if outcome.was_correct_direction and magnitude_diff > 0:
            would_change = "Would increase position size given correct read."
        elif not outcome.was_correct_direction:
            would_change = (
                "Would have required additional confirmation signals "
                "before entry."
            )
        else:
            would_change = "Trade execution was satisfactory."
        
        # Prediction accuracy score
        accuracy = 1.0 - min(1.0, abs(magnitude_diff) / 0.1)
        if not outcome.was_correct_direction:
            accuracy *= 0.5
        
        perf_narrative = PerformanceNarrative(
            decision_id=decision_id,
            narrative=narrative,
            what_worked=what_worked,
            what_failed=what_failed,
            key_difference_from_precedent=key_diff,
            prediction_accuracy=accuracy,
            regime_prediction_correct=(outcome.exit_regime == log.market_regime),
            would_do_differently=would_change
        )
        
        # Save to database
        if self.db_session_factory:
            try:
                with self.db_session_factory() as session:
                    db_log = session.query(DecisionLogModel).filter(
                        DecisionLogModel.id == decision_id
                    ).first()
                    if db_log:
                        db_log.performance_narrative = narrative
                        session.commit()
            except Exception as e:
                print(f"Error saving performance narrative: {e}")
        
        return perf_narrative
    
    def get_decision(self, decision_id: str) -> Optional[DecisionLog]:
        """Get a decision log by ID"""
        return self._memory_logs.get(decision_id)
    
    def get_manager_decisions(
        self,
        manager_id: str,
        limit: int = 50
    ) -> List[DecisionLog]:
        """Get recent decisions for a manager"""
        decisions = [
            log for log in self._memory_logs.values()
            if log.manager_id == manager_id
        ]
        decisions.sort(key=lambda x: x.timestamp, reverse=True)
        return decisions[:limit]
    
    def get_decision_summary(self, decision_id: str) -> Dict[str, Any]:
        """Get a summary of a decision for display"""
        log = self._memory_logs.get(decision_id)
        if not log:
            return {}
        
        outcome = self._outcomes.get(decision_id)
        
        summary = {
            "decision_id": decision_id,
            "manager_id": log.manager_id,
            "timestamp": log.timestamp.isoformat(),
            "symbol": log.symbol,
            "action": log.action,
            "size": log.size,
            "conviction": log.conviction,
            "thesis_preview": log.thesis[:200] + "..." if len(log.thesis) > 200 else log.thesis,
            "market_regime": log.market_regime,
            "expected_return": log.expected_return,
            "top_historical_match": log.top_match_date,
            "status": log.status.value
        }
        
        if outcome:
            summary["actual_return"] = outcome.actual_return
            summary["was_correct"] = outcome.was_correct_direction
            summary["holding_days"] = outcome.holding_period_days
        
        return summary
    
    def to_prompt_feedback(
        self,
        manager_id: str,
        last_n: int = 5
    ) -> str:
        """
        Generate feedback text for LLM prompt about recent decisions.
        """
        decisions = self.get_manager_decisions(manager_id, limit=last_n)
        
        if not decisions:
            return "No recent trading history to review."
        
        feedback = "## Recent Decision Performance\n\n"
        
        for log in decisions:
            outcome = self._outcomes.get(log.decision_id)
            
            feedback += f"### {log.symbol} ({log.action.upper()}) - {log.timestamp.strftime('%Y-%m-%d')}\n"
            feedback += f"- Conviction: {log.conviction:.0%}\n"
            feedback += f"- Expected: {log.expected_return:+.1%}\n"
            
            if outcome:
                result = "correct" if outcome.was_correct_direction else "wrong"
                feedback += f"- Actual: {outcome.actual_return:+.1%} ({result} direction)\n"
                feedback += f"- Holding period: {outcome.holding_period_days} days\n"
            else:
                feedback += f"- Status: {log.status.value}\n"
            
            feedback += "\n"
        
        # Calculate overall stats
        completed = [
            log for log in decisions
            if log.decision_id in self._outcomes
        ]
        
        if completed:
            correct = sum(
                1 for log in completed
                if self._outcomes[log.decision_id].was_correct_direction
            )
            accuracy = correct / len(completed)
            avg_return = sum(
                self._outcomes[log.decision_id].actual_return
                for log in completed
            ) / len(completed)
            
            feedback += f"**Overall: {accuracy:.0%} correct direction, "
            feedback += f"{avg_return:+.1%} avg return**\n"
        
        return feedback
