"""
Decision State Machine - Lifecycle management for trading decisions.

Key principles:
- Explicit states and transitions (no ambiguity)
- Terminal states cannot transition out
- Idempotency via deterministic keys
- inputs_hash for reproducibility tracking (includes PM prompt)
"""
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any, Set
import hashlib
import uuid


class DecisionStatus(Enum):
    """Decision lifecycle status."""
    # Pre-execution states
    CREATED = "created"
    DEBATED = "debated"
    RISK_VETOED = "risk_vetoed"  # Terminal
    FINALIZED = "finalized"
    
    # Execution states
    SENT_TO_BROKER = "sent_to_broker"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"  # Terminal
    
    # Error states
    CANCELED = "canceled"  # Terminal
    ERRORED = "errored"  # Terminal


# Terminal states - cannot transition out
TERMINAL_STATES: Set[DecisionStatus] = {
    DecisionStatus.RISK_VETOED,
    DecisionStatus.FILLED,
    DecisionStatus.CANCELED,
    DecisionStatus.ERRORED,
}


class DecisionType(Enum):
    """Type of decision made."""
    TRADE = "trade"
    NO_TRADE = "no_trade"


class NoTradeReason(Enum):
    """
    Explicit reason for no-trade decisions.
    
    Saves hours when debugging "why are we not trading?"
    """
    SNAPSHOT_INVALID = "snapshot_invalid"
    RISK_VETO = "risk_veto"
    DISAGREEMENT = "disagreement"
    BASELINE_FAILED = "baseline_failed"
    COOLDOWN = "cooldown"
    UNIVERSE_EMPTY = "universe_empty"
    NO_OPPORTUNITIES = "no_opportunities"


class RunContext(Enum):
    """Execution context."""
    BACKTEST = "backtest"
    PAPER = "paper"
    LIVE = "live"


@dataclass
class StatusTransition:
    """Record of a status transition."""
    from_status: DecisionStatus
    to_status: DecisionStatus
    timestamp: datetime
    reason: str


def compute_idempotency_key(
    fund_id: str,
    run_context: RunContext,
    decision_window_start: datetime
) -> str:
    """
    Compute deterministic idempotency key.
    
    Prevents duplicate execution of the same decision.
    Includes run_context and decision_window for stable bucketing.
    
    Args:
        fund_id: Fund identifier
        run_context: backtest/paper/live
        decision_window_start: Start of decision window (rounded timestamp)
    
    Returns:
        Idempotency key
    """
    # Round to minute for stable bucketing
    window_str = decision_window_start.strftime("%Y%m%d%H%M")
    payload = f"{fund_id}:{run_context.value}:{window_str}"
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def compute_inputs_hash(
    snapshot_id: str,
    universe_hash: str,
    fund_policy_version: str,
    fund_thesis_version: str,
    pm_prompt_hash: str
) -> str:
    """
    Hash of all decision inputs.
    
    If two decisions have same inputs_hash but different outputs,
    you know the non-determinism came from the model.
    
    INCLUDES pm_prompt_hash because PM instructions affect output.
    
    Args:
        snapshot_id: Snapshot identifier
        universe_hash: Hash of resolved universe symbols
        fund_policy_version: Policy version string
        fund_thesis_version: Thesis version string
        pm_prompt_hash: Hash of PM prompt template
    
    Returns:
        Inputs hash
    """
    payload = ":".join([
        snapshot_id,
        universe_hash,
        fund_policy_version,
        fund_thesis_version,
        pm_prompt_hash  # PM prompt changes affect output
    ])
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


@dataclass
class DecisionRecord:
    """
    Structured decision with lifecycle tracking.
    
    This is the machine-readable audit trail.
    Separate from DebateTranscript which is for humans.
    """
    decision_id: str
    fund_id: str
    snapshot_id: str
    asof_timestamp: datetime
    
    # Idempotency
    idempotency_key: str
    run_context: RunContext
    decision_window_start: datetime
    
    # Decision outcome
    decision_type: DecisionType
    no_trade_reason: Optional[NoTradeReason] = None
    
    # Lifecycle
    status: DecisionStatus = DecisionStatus.CREATED
    status_history: List[StatusTransition] = field(default_factory=list)
    
    # Intent (if trade)
    intent_json: Optional[Dict[str, Any]] = None
    risk_result_json: Optional[Dict[str, Any]] = None
    
    # Context for audit
    snapshot_quality_json: Dict[str, Any] = field(default_factory=dict)
    universe_result_json: Dict[str, Any] = field(default_factory=dict)
    
    # Reproducibility hashes
    universe_hash: Optional[str] = None
    inputs_hash: Optional[str] = None
    
    # Model tracking (anti-overfitting)
    model_versions: Dict[str, str] = field(default_factory=dict)
    prompt_hashes: Dict[str, str] = field(default_factory=dict)
    
    # Predictions (for eval)
    predicted_directions: Dict[str, str] = field(default_factory=dict)  # symbol -> "up"/"down"
    expected_holding_days: Optional[int] = None
    expected_return: Optional[float] = None
    
    # Timestamps
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON storage."""
        return {
            "decision_id": self.decision_id,
            "fund_id": self.fund_id,
            "snapshot_id": self.snapshot_id,
            "asof_timestamp": self.asof_timestamp.isoformat(),
            "idempotency_key": self.idempotency_key,
            "run_context": self.run_context.value,
            "decision_window_start": self.decision_window_start.isoformat(),
            "decision_type": self.decision_type.value,
            "no_trade_reason": self.no_trade_reason.value if self.no_trade_reason else None,
            "status": self.status.value,
            "status_history": [
                {
                    "from_status": t.from_status.value,
                    "to_status": t.to_status.value,
                    "timestamp": t.timestamp.isoformat(),
                    "reason": t.reason,
                }
                for t in self.status_history
            ],
            "intent_json": self.intent_json,
            "risk_result_json": self.risk_result_json,
            "snapshot_quality_json": self.snapshot_quality_json,
            "universe_result_json": self.universe_result_json,
            "universe_hash": self.universe_hash,
            "inputs_hash": self.inputs_hash,
            "model_versions": self.model_versions,
            "prompt_hashes": self.prompt_hashes,
            "predicted_directions": self.predicted_directions,
            "expected_holding_days": self.expected_holding_days,
            "expected_return": self.expected_return,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }


class DecisionStateMachine:
    """
    Manages decision lifecycle transitions.
    
    Key principles:
    - Explicit valid transitions
    - Terminal states cannot transition out
    - Bug fix: captures old status BEFORE update
    """
    
    VALID_TRANSITIONS: Dict[DecisionStatus, Set[DecisionStatus]] = {
        DecisionStatus.CREATED: {
            DecisionStatus.DEBATED,
            DecisionStatus.ERRORED,
        },
        DecisionStatus.DEBATED: {
            DecisionStatus.RISK_VETOED,
            DecisionStatus.FINALIZED,
            DecisionStatus.ERRORED,
        },
        DecisionStatus.FINALIZED: {
            DecisionStatus.SENT_TO_BROKER,
            DecisionStatus.CANCELED,
            DecisionStatus.ERRORED,
        },
        DecisionStatus.SENT_TO_BROKER: {
            DecisionStatus.FILLED,
            DecisionStatus.PARTIALLY_FILLED,
            DecisionStatus.CANCELED,
            DecisionStatus.ERRORED,
        },
        DecisionStatus.PARTIALLY_FILLED: {
            DecisionStatus.FILLED,
            DecisionStatus.CANCELED,
            DecisionStatus.ERRORED,
        },
        # Terminal states have no transitions
        DecisionStatus.RISK_VETOED: set(),
        DecisionStatus.FILLED: set(),
        DecisionStatus.CANCELED: set(),
        DecisionStatus.ERRORED: set(),
    }
    
    def can_transition(
        self,
        from_status: DecisionStatus,
        to_status: DecisionStatus
    ) -> bool:
        """Check if transition is valid."""
        if from_status in TERMINAL_STATES:
            return False
        valid = self.VALID_TRANSITIONS.get(from_status, set())
        return to_status in valid
    
    def transition(
        self,
        record: DecisionRecord,
        new_status: DecisionStatus,
        reason: str = ""
    ) -> DecisionRecord:
        """
        Transition a decision to a new status.
        
        Bug fix: Captures old status BEFORE updating.
        
        Args:
            record: Decision record to transition
            new_status: Target status
            reason: Reason for transition
        
        Returns:
            Updated record
        
        Raises:
            ValueError: If transition is invalid
        """
        # FIXED: Capture old status BEFORE update
        old_status = record.status
        
        if not self.can_transition(old_status, new_status):
            raise ValueError(
                f"Invalid transition: {old_status.value} -> {new_status.value}"
            )
        
        # Update status
        record.status = new_status
        record.updated_at = datetime.utcnow()
        
        # Record transition
        record.status_history.append(StatusTransition(
            from_status=old_status,
            to_status=new_status,
            timestamp=datetime.utcnow(),
            reason=reason,
        ))
        
        return record
    
    def create_decision(
        self,
        fund_id: str,
        snapshot_id: str,
        run_context: RunContext,
        decision_window_start: datetime,
        asof_timestamp: Optional[datetime] = None,
    ) -> DecisionRecord:
        """
        Create a new decision record.
        
        Args:
            fund_id: Fund identifier
            snapshot_id: Snapshot identifier
            run_context: backtest/paper/live
            decision_window_start: Start of decision window
            asof_timestamp: Decision timestamp (defaults to now)
        
        Returns:
            New decision record
        """
        now = asof_timestamp or datetime.utcnow()
        
        return DecisionRecord(
            decision_id=str(uuid.uuid4()),
            fund_id=fund_id,
            snapshot_id=snapshot_id,
            asof_timestamp=now,
            idempotency_key=compute_idempotency_key(
                fund_id, run_context, decision_window_start
            ),
            run_context=run_context,
            decision_window_start=decision_window_start,
            decision_type=DecisionType.NO_TRADE,  # Default, updated later
            status=DecisionStatus.CREATED,
        )
    
    def mark_no_trade(
        self,
        record: DecisionRecord,
        reason: NoTradeReason
    ) -> DecisionRecord:
        """
        Mark decision as no-trade.
        
        Args:
            record: Decision record
            reason: Why no trade
        
        Returns:
            Updated record
        """
        record.decision_type = DecisionType.NO_TRADE
        record.no_trade_reason = reason
        record.updated_at = datetime.utcnow()
        return record
    
    def mark_trade(
        self,
        record: DecisionRecord,
        intent_json: Dict[str, Any],
        predicted_directions: Dict[str, str],
        expected_holding_days: Optional[int] = None,
        expected_return: Optional[float] = None,
    ) -> DecisionRecord:
        """
        Mark decision as trade.
        
        Args:
            record: Decision record
            intent_json: Portfolio intent as JSON
            predicted_directions: symbol -> direction predictions
            expected_holding_days: Expected holding period
            expected_return: Expected return
        
        Returns:
            Updated record
        """
        record.decision_type = DecisionType.TRADE
        record.no_trade_reason = None
        record.intent_json = intent_json
        record.predicted_directions = predicted_directions
        record.expected_holding_days = expected_holding_days
        record.expected_return = expected_return
        record.updated_at = datetime.utcnow()
        return record
