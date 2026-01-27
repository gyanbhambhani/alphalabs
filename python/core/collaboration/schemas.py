"""
Structured JSON schemas for debate outputs.

All AI outputs are strict JSON for:
- Replayability
- Evals
- Audits

Every output includes version, fund_id, asof_timestamp, and snapshot_id.
"""
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Any


@dataclass
class TradeCandidate:
    """A proposed trade candidate."""
    symbol: str
    direction: str  # "long", "short", "flat"
    target_weight: float  # % of portfolio
    expected_horizon_days: int
    rationale: str
    key_features_used: List[str]  # Must be subset of snapshot.available_features()
    confidence: float  # 0-1
    failure_modes: List[str]  # What could go wrong
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "direction": self.direction,
            "target_weight": self.target_weight,
            "expected_horizon_days": self.expected_horizon_days,
            "rationale": self.rationale,
            "key_features_used": self.key_features_used,
            "confidence": self.confidence,
            "failure_modes": self.failure_modes,
        }


@dataclass
class ProposalOutput:
    """
    Output from propose() phase.
    
    Each participant proposes trade candidates.
    """
    # Metadata
    version: str
    fund_id: str
    asof_timestamp: datetime
    snapshot_id: str
    participant_id: str
    model_name: str
    model_version: str
    prompt_hash: str
    
    # Content
    candidates: List[TradeCandidate]
    market_view: str  # Overall market assessment
    key_drivers: List[str]  # Main drivers of recommendations
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": self.version,
            "fund_id": self.fund_id,
            "asof_timestamp": self.asof_timestamp.isoformat(),
            "snapshot_id": self.snapshot_id,
            "participant_id": self.participant_id,
            "model_name": self.model_name,
            "model_version": self.model_version,
            "prompt_hash": self.prompt_hash,
            "candidates": [c.to_dict() for c in self.candidates],
            "market_view": self.market_view,
            "key_drivers": self.key_drivers,
        }


@dataclass
class CritiqueItem:
    """A specific critique of a proposal."""
    target_symbol: str
    issue_type: str  # "data_concern", "risk", "counter_thesis", "missing_info"
    description: str
    severity: str  # "minor", "major", "critical"
    counter_argument: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "target_symbol": self.target_symbol,
            "issue_type": self.issue_type,
            "description": self.description,
            "severity": self.severity,
            "counter_argument": self.counter_argument,
        }


@dataclass
class CritiqueOutput:
    """
    Output from critique() phase.
    
    Each participant critiques proposals.
    """
    # Metadata
    version: str
    fund_id: str
    asof_timestamp: datetime
    snapshot_id: str
    participant_id: str
    model_name: str
    model_version: str
    prompt_hash: str
    critiqued_proposal_id: str
    
    # Content
    critiques: List[CritiqueItem]
    missing_data_flags: List[str]  # Data we wish we had
    risk_flags: List[str]  # Potential risks identified
    overall_assessment: str  # "support", "neutral", "oppose"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": self.version,
            "fund_id": self.fund_id,
            "asof_timestamp": self.asof_timestamp.isoformat(),
            "snapshot_id": self.snapshot_id,
            "participant_id": self.participant_id,
            "model_name": self.model_name,
            "model_version": self.model_version,
            "prompt_hash": self.prompt_hash,
            "critiqued_proposal_id": self.critiqued_proposal_id,
            "critiques": [c.to_dict() for c in self.critiques],
            "missing_data_flags": self.missing_data_flags,
            "risk_flags": self.risk_flags,
            "overall_assessment": self.overall_assessment,
        }


@dataclass
class MergedPlan:
    """A merged trade plan from synthesis."""
    symbol: str
    direction: str
    target_weight: float
    consensus_level: float  # How much agreement (0-1)
    supporting_participants: List[str]
    opposing_participants: List[str]
    key_reasoning: str
    unresolved_concerns: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "direction": self.direction,
            "target_weight": self.target_weight,
            "consensus_level": self.consensus_level,
            "supporting_participants": self.supporting_participants,
            "opposing_participants": self.opposing_participants,
            "key_reasoning": self.key_reasoning,
            "unresolved_concerns": self.unresolved_concerns,
        }


@dataclass
class SynthesisOutput:
    """
    Output from synthesize() phase.
    
    Merges proposals and critiques into candidate plans.
    """
    # Metadata
    version: str
    fund_id: str
    asof_timestamp: datetime
    snapshot_id: str
    model_name: str
    model_version: str
    prompt_hash: str
    
    # Content
    merged_plans: List[MergedPlan]
    consensus_level: float  # Overall consensus (0-1)
    key_agreements: List[str]
    key_disagreements: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": self.version,
            "fund_id": self.fund_id,
            "asof_timestamp": self.asof_timestamp.isoformat(),
            "snapshot_id": self.snapshot_id,
            "model_name": self.model_name,
            "model_version": self.model_version,
            "prompt_hash": self.prompt_hash,
            "merged_plans": [p.to_dict() for p in self.merged_plans],
            "consensus_level": self.consensus_level,
            "key_agreements": self.key_agreements,
            "key_disagreements": self.key_disagreements,
        }


@dataclass
class FinalPosition:
    """A final position from PM."""
    symbol: str
    target_weight: float
    direction: str
    stop_loss_pct: float
    take_profit_pct: float
    expected_holding_days: int
    exit_rationale: str
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "target_weight": self.target_weight,
            "direction": self.direction,
            "stop_loss_pct": self.stop_loss_pct,
            "take_profit_pct": self.take_profit_pct,
            "expected_holding_days": self.expected_holding_days,
            "exit_rationale": self.exit_rationale,
        }


@dataclass
class FinalizeOutput:
    """
    Output from finalize() phase (PM decision).
    
    The PM makes the final executable decision.
    """
    # Metadata
    version: str
    fund_id: str
    asof_timestamp: datetime
    snapshot_id: str
    model_name: str
    model_version: str
    prompt_hash: str
    
    # Content
    decision: str  # "trade", "no_trade"
    positions: List[FinalPosition]
    target_cash_pct: float
    sizing_method_used: str
    policy_version: str
    
    # Reasoning
    key_reasoning: str
    risk_notes: List[str]
    expected_return: Optional[float] = None
    expected_holding_days: Optional[int] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": self.version,
            "fund_id": self.fund_id,
            "asof_timestamp": self.asof_timestamp.isoformat(),
            "snapshot_id": self.snapshot_id,
            "model_name": self.model_name,
            "model_version": self.model_version,
            "prompt_hash": self.prompt_hash,
            "decision": self.decision,
            "positions": [p.to_dict() for p in self.positions],
            "target_cash_pct": self.target_cash_pct,
            "sizing_method_used": self.sizing_method_used,
            "policy_version": self.policy_version,
            "key_reasoning": self.key_reasoning,
            "risk_notes": self.risk_notes,
            "expected_return": self.expected_return,
            "expected_holding_days": self.expected_holding_days,
        }
