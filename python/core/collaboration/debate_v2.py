"""
Collaborative AI Debate System V2.1

This module contains all dataclasses and types for the investment committee
debate system with:
- Machine-validated evidence citations
- Deterministic consensus gates
- Structured invalidation rules
- Risk manager sizing (separate from debate)

Key principles:
- Agents cite features by key only (no values in output)
- Server validates and joins actual values
- Execution depends ONLY on deterministic gates, not LLM alignment scores
"""

from dataclasses import dataclass, field
from datetime import date
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple, Any
import math

from core.data.snapshot import GlobalMarketSnapshot


# =============================================================================
# Available Features (from PointInTimeSnapshotBuilder)
# =============================================================================

AVAILABLE_FEATURES: Set[str] = {
    # Returns (from RETURN_PERIODS)
    "return_1d",
    "return_5d",
    "return_21d",
    "return_63d",
    
    # Volatility (from VOLATILITY_WINDOWS)
    "volatility_5d",
    "volatility_21d",
    
    # Price
    "price",
}


# =============================================================================
# ThesisType Enum (Locked - No Future Types Exposed)
# =============================================================================

class ThesisType(str, Enum):
    """
    Only thesis types the engine can evaluate.
    No future types exposed to agents.
    """
    MOMENTUM = "momentum"
    MEAN_REVERSION = "mean_reversion"
    VOLATILITY = "volatility"
    # VALUE and QUALITY removed - not available until we have fundamentals


# =============================================================================
# Required Evidence Per Thesis Type
# =============================================================================

THESIS_REQUIRED_EVIDENCE: Dict[ThesisType, Set[str]] = {
    ThesisType.MOMENTUM: {"return_21d", "return_63d"},
    ThesisType.MEAN_REVERSION: {"return_1d", "return_5d"},
    ThesisType.VOLATILITY: {"volatility_5d", "volatility_21d"},
}


# =============================================================================
# Fund Strategy to Thesis Type Mapping
# =============================================================================

# Mapping from fund strategy to allowed thesis types
# Empty set = BLOCKED (fund cannot trade until thesis type exists)
STRATEGY_TO_THESIS_TYPES: Dict[str, Set[ThesisType]] = {
    "momentum": {ThesisType.MOMENTUM},
    "trend_macro": {ThesisType.MOMENTUM},
    "mean_reversion": {ThesisType.MEAN_REVERSION},
    "volatility": {ThesisType.VOLATILITY},
    # BLOCKED until we implement these thesis types
    "value": set(),
    "quality_ls": set(),
    "event_driven": set(),
}


def validate_thesis_against_strategy(
    thesis_type: ThesisType,
    fund_strategy: str
) -> Tuple[bool, str]:
    """
    Check if thesis type is allowed for fund strategy.
    
    Args:
        thesis_type: The thesis type proposed by the agent
        fund_strategy: The fund's strategy string
        
    Returns:
        Tuple of (is_valid, reason_if_invalid)
    """
    # Normalize: "Mean Reversion Fund" -> "mean_reversion"
    strategy_key = fund_strategy.lower().replace(" ", "_")
    for suffix in ["_fund", "_sleeve"]:
        if strategy_key.endswith(suffix):
            strategy_key = strategy_key[:-len(suffix)]
    
    # Fallback: first word
    if strategy_key not in STRATEGY_TO_THESIS_TYPES:
        strategy_key = fund_strategy.lower().split()[0]
    
    allowed = STRATEGY_TO_THESIS_TYPES.get(strategy_key)
    
    if allowed is None:
        return False, f"Unknown fund strategy '{fund_strategy}'"
    
    if not allowed:
        return False, (
            f"Fund strategy '{fund_strategy}' disabled - no supported thesis types"
        )
    
    if thesis_type not in allowed:
        return False, (
            f"Thesis '{thesis_type.value}' not allowed for '{fund_strategy}'. "
            f"Allowed: {[t.value for t in allowed]}"
        )
    
    return True, ""

# Interpretation templates per thesis (for documentation/UI)
THESIS_INTERPRETATIONS: Dict[ThesisType, Dict[str, str]] = {
    ThesisType.MEAN_REVERSION: {
        "return_1d": "negative_suggests_bounce",
        "return_5d": "negative_suggests_bounce",
        "volatility_21d": "elevated_supports_reversion",
    },
    ThesisType.MOMENTUM: {
        "return_21d": "positive_suggests_trend",
        "return_63d": "positive_confirms_trend",
        "volatility_21d": "stable_supports_trend",
    },
    ThesisType.VOLATILITY: {
        "volatility_5d": "spike_triggers_derisk",
        "volatility_21d": "elevated_triggers_derisk",
    },
}


# =============================================================================
# InvalidationRule (Structured - No Free Text)
# =============================================================================

@dataclass
class InvalidationRule:
    """
    Machine-evaluable invalidation condition.
    
    When evaluate() returns True, the thesis is BROKEN and position should
    be reconsidered.
    """
    feature: str   # Must be in AVAILABLE_FEATURES
    symbol: str    # Symbol to check
    operator: str  # ">", "<", ">=", "<=", "=="
    value: float   # Threshold
    
    def evaluate(self, snapshot: GlobalMarketSnapshot) -> bool:
        """
        Returns True if invalidation TRIGGERED (thesis broken).
        
        Args:
            snapshot: Current market snapshot
            
        Returns:
            True if the invalidation condition is met
        """
        actual: Optional[float] = None
        
        if self.feature.startswith("return_"):
            period = self.feature.split("_")[1]
            actual = snapshot.get_return(self.symbol, period)
        elif self.feature.startswith("volatility_"):
            period = self.feature.split("_")[1]
            actual = snapshot.get_volatility(self.symbol, period)
        elif self.feature == "price":
            actual = snapshot.get_price(self.symbol)
        
        if actual is None:
            return False
        
        ops = {
            ">": lambda a, b: a > b,
            "<": lambda a, b: a < b,
            ">=": lambda a, b: a >= b,
            "<=": lambda a, b: a <= b,
            "==": lambda a, b: abs(a - b) < 0.001,
        }
        
        op_func = ops.get(self.operator)
        if op_func is None:
            return False
            
        return op_func(actual, self.value)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "feature": self.feature,
            "symbol": self.symbol,
            "operator": self.operator,
            "value": self.value,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "InvalidationRule":
        """Create from dictionary."""
        return cls(
            feature=data["feature"],
            symbol=data["symbol"],
            operator=data["operator"],
            value=float(data["value"]),
        )


# =============================================================================
# EvidenceReference (No Values - Server Joins)
# =============================================================================

@dataclass
class EvidenceReference:
    """
    Agent cites feature by key only. Server attaches value.
    
    NO value field - server joins this after validation.
    NO interpretation field - determined by thesis_type.
    """
    feature: str  # Must be in AVAILABLE_FEATURES
    symbol: str   # Symbol being referenced
    
    def to_dict(self) -> Dict[str, str]:
        """Convert to dictionary for JSON serialization."""
        return {
            "feature": self.feature,
            "symbol": self.symbol,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EvidenceReference":
        """Create from dictionary."""
        return cls(
            feature=data["feature"],
            symbol=data["symbol"],
        )


# =============================================================================
# ThesisProposal (Uses InvalidationRule, Not Strings)
# =============================================================================

@dataclass
class ThesisProposal:
    """
    Structured thesis - all fields machine-readable.
    
    Uses InvalidationRule list, NOT List[str] for invalidation conditions.
    """
    thesis_type: ThesisType          # Enum, not string
    horizon_days: int                 # 1, 5, 21, etc.
    primary_signal: str               # Feature key from AVAILABLE_FEATURES
    secondary_signal: Optional[str]   # Optional second feature
    invalidation_rules: List[InvalidationRule]  # NOT List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "thesis_type": self.thesis_type.value,
            "horizon_days": self.horizon_days,
            "primary_signal": self.primary_signal,
            "secondary_signal": self.secondary_signal,
            "invalidation_rules": [r.to_dict() for r in self.invalidation_rules],
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ThesisProposal":
        """Create from dictionary."""
        return cls(
            thesis_type=ThesisType(data["thesis_type"]),
            horizon_days=int(data["horizon_days"]),
            primary_signal=data["primary_signal"],
            secondary_signal=data.get("secondary_signal"),
            invalidation_rules=[
                InvalidationRule.from_dict(r) 
                for r in data.get("invalidation_rules", [])
            ],
        )


# =============================================================================
# ConfidenceDecomposition
# =============================================================================

@dataclass
class ConfidenceDecomposition:
    """
    Decomposed confidence for better calibration tracking.
    
    Each component is 0-1. One weak component pulls down overall confidence.
    """
    signal_strength: float       # 0-1: how strong is the signal
    regime_fit: float            # 0-1: does regime support thesis
    risk_comfort: float          # 0-1: risk-adjusted comfort
    execution_feasibility: float # 0-1: can we execute
    
    def min_component(self) -> float:
        """Return the minimum component value."""
        return min(
            self.signal_strength,
            self.regime_fit,
            self.risk_comfort,
            self.execution_feasibility
        )
    
    def overall(self) -> float:
        """
        Geometric mean - one weak component pulls down overall.
        
        Returns:
            Overall confidence as geometric mean of components
        """
        product = (
            self.signal_strength *
            self.regime_fit *
            self.risk_comfort *
            self.execution_feasibility
        )
        return product ** 0.25
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary for JSON serialization."""
        return {
            "signal_strength": self.signal_strength,
            "regime_fit": self.regime_fit,
            "risk_comfort": self.risk_comfort,
            "execution_feasibility": self.execution_feasibility,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ConfidenceDecomposition":
        """Create from dictionary."""
        return cls(
            signal_strength=float(data.get("signal_strength", 0.5)),
            regime_fit=float(data.get("regime_fit", 0.5)),
            risk_comfort=float(data.get("risk_comfort", 0.5)),
            execution_feasibility=float(data.get("execution_feasibility", 0.5)),
        )


# =============================================================================
# DialogMove (Forces Actual Conversation)
# =============================================================================

@dataclass
class DialogMove:
    """
    Forces actual conversation, not JSON blobs.
    
    Each agent must explicitly engage with colleague's position.
    """
    acknowledge: str            # Restate colleague's claim
    challenge: Optional[str]    # One specific disagreement (or None)
    request: Optional[str]      # Ask for missing evidence (or None)
    concede_or_hold: str        # What changed vs stayed firm
    
    def to_dict(self) -> Dict[str, Optional[str]]:
        """Convert to dictionary for JSON serialization."""
        return {
            "acknowledge": self.acknowledge,
            "challenge": self.challenge,
            "request": self.request,
            "concede_or_hold": self.concede_or_hold,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DialogMove":
        """Create from dictionary."""
        return cls(
            acknowledge=data.get("acknowledge", ""),
            challenge=data.get("challenge"),
            request=data.get("request"),
            concede_or_hold=data.get("concede_or_hold", ""),
        )


# =============================================================================
# Counterfactual (Required: What Else Was Considered)
# =============================================================================

@dataclass
class Counterfactual:
    """
    Required: what else was considered.
    
    Forces agents to think about alternatives.
    """
    alternative_action: str  # "hold", "sell", etc.
    why_rejected: str        # Brief reason
    
    def to_dict(self) -> Dict[str, str]:
        """Convert to dictionary for JSON serialization."""
        return {
            "alternative_action": self.alternative_action,
            "why_rejected": self.why_rejected,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Counterfactual":
        """Create from dictionary."""
        return cls(
            alternative_action=data.get("alternative_action", "hold"),
            why_rejected=data.get("why_rejected", ""),
        )


# =============================================================================
# AgentTurn (Canonical - No Values in Evidence)
# =============================================================================

@dataclass
class AgentTurn:
    """
    Complete agent output per round.
    
    Key principle: evidence_cited contains NO VALUES - just references.
    Server joins actual values after validation.
    """
    agent_id: str
    round_num: int
    
    # Dialog (forces conversation)
    dialog_move: DialogMove
    
    # Proposal
    action: str                 # "buy", "sell", "hold"
    symbol: Optional[str]       # Required if action != "hold"
    suggested_weight: float     # 0-1, suggestion only
    risk_posture: str           # "normal", "defensive", "aggressive"
    
    # Thesis
    thesis: ThesisProposal
    
    # Confidence
    confidence: ConfidenceDecomposition
    
    # Evidence (NO VALUES - just references)
    evidence_cited: List[EvidenceReference]
    
    # Counterfactual (required)
    counterfactual: Counterfactual
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "agent_id": self.agent_id,
            "round_num": self.round_num,
            "dialog_move": self.dialog_move.to_dict(),
            "action": self.action,
            "symbol": self.symbol,
            "suggested_weight": self.suggested_weight,
            "risk_posture": self.risk_posture,
            "thesis": self.thesis.to_dict(),
            "confidence": self.confidence.to_dict(),
            "evidence_cited": [e.to_dict() for e in self.evidence_cited],
            "counterfactual": self.counterfactual.to_dict(),
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AgentTurn":
        """Create from dictionary."""
        return cls(
            agent_id=data["agent_id"],
            round_num=int(data["round_num"]),
            dialog_move=DialogMove.from_dict(data.get("dialog_move", {})),
            action=data["action"],
            symbol=data.get("symbol"),
            suggested_weight=float(data.get("suggested_weight", 0.0)),
            risk_posture=data.get("risk_posture", "normal"),
            thesis=ThesisProposal.from_dict(data.get("thesis", {})),
            confidence=ConfidenceDecomposition.from_dict(
                data.get("confidence", {})
            ),
            evidence_cited=[
                EvidenceReference.from_dict(e) 
                for e in data.get("evidence_cited", [])
            ],
            counterfactual=Counterfactual.from_dict(
                data.get("counterfactual", {})
            ),
        )


# =============================================================================
# ConsensusGate (Deterministic - Hard Gates)
# =============================================================================

@dataclass
class ConsensusGate:
    """
    Deterministic checks for AGENT AGREEMENT quality.
    
    Does NOT include execution concerns (price, portfolio state).
    Those are checked separately in simulation_engine.
    
    All gates must pass for execution. LLM alignment score is for UX only.
    
    V2.1.2 - Strict gates:
    - symbol_match: STRICT - must match for non-hold
    - min_confidence_met: >= 0.5 (restored)
    - min_edge_met: >= 0.6 (restored)
    - fund_strategy_match: NEW - thesis must match fund mandate
    """
    # Hard requirements
    action_match: bool = False
    symbol_match: bool = False              # STRICT: must match for non-hold
    thesis_type_match: bool = False
    fund_strategy_match: bool = False       # NEW: thesis matches fund mandate
    horizon_within_tolerance: bool = False  # Bucket-based (short/medium/long)
    invalidation_compatible: bool = False   # At least one has rules
    min_confidence_met: bool = False        # All components >= 0.5 (restored)
    evidence_validated: bool = False        # All refs exist + relevant
    min_edge_met: bool = False              # signal_strength >= 0.6 (restored)
    
    # Rejection reasons for debugging
    rejection_reasons: List[str] = field(default_factory=list)
    
    def is_executable(self) -> bool:
        """
        ALL agreement checks must pass for execution.
        
        Returns:
            True if all gates pass
        """
        return all([
            self.action_match,
            self.symbol_match,
            self.thesis_type_match,
            self.fund_strategy_match,
            self.horizon_within_tolerance,
            self.invalidation_compatible,
            self.min_confidence_met,
            self.evidence_validated,
            self.min_edge_met,
        ])
    
    def failed_gates(self) -> List[str]:
        """
        Return names of failed gates for debugging.
        
        Returns:
            List of gate names that failed
        """
        gates = {
            "action_match": self.action_match,
            "symbol_match": self.symbol_match,
            "thesis_type_match": self.thesis_type_match,
            "fund_strategy_match": self.fund_strategy_match,
            "horizon_within_tolerance": self.horizon_within_tolerance,
            "invalidation_compatible": self.invalidation_compatible,
            "min_confidence_met": self.min_confidence_met,
            "evidence_validated": self.evidence_validated,
            "min_edge_met": self.min_edge_met,
        }
        return [name for name, passed in gates.items() if not passed]
    
    def passed_gates(self) -> List[str]:
        """
        Return names of passed gates.
        
        Returns:
            List of gate names that passed
        """
        gates = {
            "action_match": self.action_match,
            "symbol_match": self.symbol_match,
            "thesis_type_match": self.thesis_type_match,
            "fund_strategy_match": self.fund_strategy_match,
            "horizon_within_tolerance": self.horizon_within_tolerance,
            "invalidation_compatible": self.invalidation_compatible,
            "min_confidence_met": self.min_confidence_met,
            "evidence_validated": self.evidence_validated,
            "min_edge_met": self.min_edge_met,
        }
        return [name for name, passed in gates.items() if passed]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "action_match": self.action_match,
            "symbol_match": self.symbol_match,
            "thesis_type_match": self.thesis_type_match,
            "fund_strategy_match": self.fund_strategy_match,
            "horizon_within_tolerance": self.horizon_within_tolerance,
            "invalidation_compatible": self.invalidation_compatible,
            "min_confidence_met": self.min_confidence_met,
            "evidence_validated": self.evidence_validated,
            "min_edge_met": self.min_edge_met,
            "is_executable": self.is_executable(),
            "rejection_reasons": self.rejection_reasons,
        }


# =============================================================================
# ConsensusBoard (Hybrid - Deterministic + LLM Summary)
# =============================================================================

@dataclass
class ConsensusBoard:
    """
    Full consensus state. Gate is deterministic, rest is for UX.
    
    Execution depends ONLY on the deterministic gate.
    LLM-generated fields are for UI display only.
    """
    # Deterministic gate (execution depends on this)
    gate: ConsensusGate = field(default_factory=ConsensusGate)
    
    # Agreed items (computed deterministically)
    agreed_action: Optional[str] = None
    agreed_symbol: Optional[str] = None
    agreed_thesis_type: Optional[ThesisType] = None
    agreed_horizon: Optional[int] = None
    
    # LLM-generated (for UX only, not execution)
    alignment_score: float = 0.0        # 0-1, moderator's assessment
    open_questions: List[str] = field(default_factory=list)
    concessions_made: List[str] = field(default_factory=list)
    disputed_topics: List[str] = field(default_factory=list)
    dispute_summary: str = ""           # LLM narrative for UI
    
    def can_execute(self) -> bool:
        """
        Execution depends ONLY on deterministic gate.
        
        Returns:
            True if the deterministic gate passes
        """
        return self.gate.is_executable()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "gate": self.gate.to_dict(),
            "agreed_action": self.agreed_action,
            "agreed_symbol": self.agreed_symbol,
            "agreed_thesis_type": (
                self.agreed_thesis_type.value 
                if self.agreed_thesis_type else None
            ),
            "agreed_horizon": self.agreed_horizon,
            "alignment_score": self.alignment_score,
            "open_questions": self.open_questions,
            "concessions_made": self.concessions_made,
            "disputed_topics": self.disputed_topics,
            "dispute_summary": self.dispute_summary,
            "can_execute": self.can_execute(),
        }


# =============================================================================
# RiskManagerDecision (After Consensus, Not Inside Debate)
# =============================================================================

@dataclass
class RiskManagerDecision:
    """
    Final sizing decision - separate from debate.
    
    Debate outputs suggestions, risk manager applies constraints.
    """
    approved: bool
    final_weight: float
    constraints_hit: List[str] = field(default_factory=list)
    
    @staticmethod
    def compute(
        action: str,
        symbol: Optional[str],
        suggested_weight: float,
        risk_posture: str,
        portfolio_cash: float,
        portfolio_total_value: float,
        current_position_weight: float = 0.0,
        max_position_pct: float = 0.15,
        max_turnover_daily: float = 0.20,
    ) -> "RiskManagerDecision":
        """
        Deterministic sizing based on constraints.
        
        Args:
            action: "buy", "sell", or "hold"
            symbol: Symbol being traded
            suggested_weight: Agent's suggested weight (0-1)
            risk_posture: "normal", "defensive", or "aggressive"
            portfolio_cash: Available cash
            portfolio_total_value: Total portfolio value
            current_position_weight: Current weight of this position
            max_position_pct: Maximum position size
            max_turnover_daily: Maximum daily turnover
            
        Returns:
            RiskManagerDecision with final sizing
        """
        constraints_hit: List[str] = []
        final = suggested_weight
        
        # 1. Cap at max position
        if final > max_position_pct:
            constraints_hit.append(f"max_position: {max_position_pct:.0%}")
            final = max_position_pct
        
        # 2. Scale for defensive posture
        if risk_posture == "defensive":
            constraints_hit.append("defensive_posture: 50% scale")
            final *= 0.5
        
        # 3. Turnover constraint
        turnover = abs(final - current_position_weight)
        if turnover > max_turnover_daily:
            constraints_hit.append(f"turnover: max {max_turnover_daily:.0%}")
            direction = 1 if final > current_position_weight else -1
            final = current_position_weight + (max_turnover_daily * direction)
        
        # 4. Cash constraint (for buys)
        if action == "buy" and portfolio_total_value > 0:
            max_from_cash = (portfolio_cash / portfolio_total_value) * 0.9
            if final > max_from_cash:
                constraints_hit.append(f"cash: max {max_from_cash:.0%}")
                final = max_from_cash
        
        return RiskManagerDecision(
            approved=final >= 0.01,
            final_weight=max(0.0, final),
            constraints_hit=constraints_hit,
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "approved": self.approved,
            "final_weight": self.final_weight,
            "constraints_hit": self.constraints_hit,
        }


# =============================================================================
# ExperienceRecord (For Replay/Learning)
# =============================================================================

@dataclass
class ExperienceRecord:
    """
    Record for replay/learning.
    
    Stores state, decision, and outcomes for similarity retrieval.
    """
    record_id: str
    record_date: date
    fund_id: str
    
    # State at decision time
    state_features: Dict[str, float] = field(default_factory=dict)
    regime_tags: Dict[str, str] = field(default_factory=dict)
    
    # Decision made
    action: str = "hold"
    symbol: Optional[str] = None
    thesis_type: str = ""
    confidence: float = 0.0
    rationale_summary: str = ""
    counterfactual: str = ""
    
    # Outcomes (filled post-hoc)
    outcome_1d: Optional[float] = None
    outcome_5d: Optional[float] = None
    outcome_21d: Optional[float] = None
    max_drawdown: Optional[float] = None
    invalidation_triggered: Optional[bool] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "record_id": self.record_id,
            "record_date": self.record_date.isoformat(),
            "fund_id": self.fund_id,
            "state_features": self.state_features,
            "regime_tags": self.regime_tags,
            "action": self.action,
            "symbol": self.symbol,
            "thesis_type": self.thesis_type,
            "confidence": self.confidence,
            "rationale_summary": self.rationale_summary,
            "counterfactual": self.counterfactual,
            "outcome_1d": self.outcome_1d,
            "outcome_5d": self.outcome_5d,
            "outcome_21d": self.outcome_21d,
            "max_drawdown": self.max_drawdown,
            "invalidation_triggered": self.invalidation_triggered,
        }


# =============================================================================
# DebateOutput (What Debate Produces)
# =============================================================================

@dataclass
class DebateOutput:
    """
    What debate produces - suggestions, not final sizing.
    
    Risk manager takes this and applies portfolio constraints.
    """
    action: str
    symbol: Optional[str]
    thesis: ThesisProposal
    suggested_weight: float
    risk_posture: str
    conviction_level: float
    consensus_board: ConsensusBoard
    conversation: List[AgentTurn] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "action": self.action,
            "symbol": self.symbol,
            "thesis": self.thesis.to_dict(),
            "suggested_weight": self.suggested_weight,
            "risk_posture": self.risk_posture,
            "conviction_level": self.conviction_level,
            "consensus_board": self.consensus_board.to_dict(),
            "conversation": [t.to_dict() for t in self.conversation],
        }


# =============================================================================
# Evidence Validation (With Relevance Check)
# =============================================================================

def validate_evidence(
    evidence: List[EvidenceReference],
    thesis_type: ThesisType,
    snapshot: GlobalMarketSnapshot,
) -> Tuple[bool, List[str]]:
    """
    Validate evidence citations.
    
    Checks:
    1. Feature exists in AVAILABLE_FEATURES
    2. Symbol has data in snapshot
    3. At least one feature from required set for thesis_type
    4. Minimum 2 evidence items
    
    Args:
        evidence: List of evidence references from agent
        thesis_type: The thesis type being proposed
        snapshot: Current market snapshot
        
    Returns:
        Tuple of (is_valid, list_of_errors)
    """
    errors: List[str] = []
    
    # Check minimum count
    if len(evidence) < 2:
        errors.append(f"Need at least 2 evidence items, got {len(evidence)}")
    
    # Check each reference
    cited_features: Set[str] = set()
    
    for ref in evidence:
        # Check feature is in available set
        if ref.feature not in AVAILABLE_FEATURES:
            errors.append(f"Feature '{ref.feature}' not available")
            continue
        
        # Check data exists in snapshot
        value: Optional[float] = None
        
        if ref.feature.startswith("return_"):
            period = ref.feature.split("_")[1]
            value = snapshot.get_return(ref.symbol, period)
        elif ref.feature.startswith("volatility_"):
            period = ref.feature.split("_")[1]
            value = snapshot.get_volatility(ref.symbol, period)
        elif ref.feature == "price":
            value = snapshot.get_price(ref.symbol)
        
        if value is None:
            errors.append(f"No data for {ref.feature} on {ref.symbol}")
        else:
            cited_features.add(ref.feature)
    
    # Check relevance to thesis
    required = THESIS_REQUIRED_EVIDENCE.get(thesis_type, set())
    if required and not cited_features.intersection(required):
        errors.append(
            f"Thesis '{thesis_type.value}' requires evidence from {required}, "
            f"but only cited {cited_features}"
        )
    
    return len(errors) == 0, errors


# =============================================================================
# Deterministic Gate Computation
# =============================================================================

def compute_consensus_gate(
    turn_a: AgentTurn,
    turn_b: AgentTurn,
    snapshot: GlobalMarketSnapshot,
    fund_strategy: str = "",
) -> ConsensusGate:
    """
    Compute all gate checks deterministically with strict thresholds.
    
    No LLM involvement - pure boolean logic based on agent outputs.
    
    Args:
        turn_a: First agent's turn
        turn_b: Second agent's turn
        snapshot: Current market snapshot
        fund_strategy: Fund's strategy for thesis validation
        
    Returns:
        ConsensusGate with all checks computed
    """
    rejection_reasons: List[str] = []
    
    # Action match
    action_match = turn_a.action == turn_b.action
    if not action_match:
        rejection_reasons.append(
            f"Action mismatch: {turn_a.action} vs {turn_b.action}"
        )
    
    # Symbol match - STRICT for non-hold
    if turn_a.action == "hold" and turn_b.action == "hold":
        symbol_match = True  # Both holding, no symbol needed
    else:
        symbol_match = turn_a.symbol == turn_b.symbol
        if not symbol_match:
            rejection_reasons.append(
                f"Symbol mismatch: {turn_a.symbol} vs {turn_b.symbol}"
            )
    
    # Thesis type match
    thesis_type_match = (
        turn_a.thesis.thesis_type == turn_b.thesis.thesis_type
    )
    if not thesis_type_match:
        rejection_reasons.append(
            f"Thesis mismatch: {turn_a.thesis.thesis_type.value} vs "
            f"{turn_b.thesis.thesis_type.value}"
        )
    
    # Fund strategy match - NEW
    fund_strategy_match = True
    if fund_strategy:
        valid_a, reason_a = validate_thesis_against_strategy(
            turn_a.thesis.thesis_type, fund_strategy
        )
        valid_b, reason_b = validate_thesis_against_strategy(
            turn_b.thesis.thesis_type, fund_strategy
        )
        fund_strategy_match = valid_a and valid_b
        if not fund_strategy_match:
            if not valid_a:
                rejection_reasons.append(f"Analyst: {reason_a}")
            if not valid_b:
                rejection_reasons.append(f"Critic: {reason_b}")
    
    # Horizon within tolerance - use buckets
    def horizon_bucket(days: int) -> str:
        if days <= 5:
            return "short"
        elif days <= 21:
            return "medium"
        else:
            return "long"
    
    bucket_a = horizon_bucket(turn_a.thesis.horizon_days)
    bucket_b = horizon_bucket(turn_b.thesis.horizon_days)
    horizon_within_tolerance = bucket_a == bucket_b
    if not horizon_within_tolerance:
        rejection_reasons.append(
            f"Horizon mismatch: {turn_a.thesis.horizon_days}d ({bucket_a}) vs "
            f"{turn_b.thesis.horizon_days}d ({bucket_b})"
        )
    
    # Invalidation compatible
    a_has_rules = len(turn_a.thesis.invalidation_rules) > 0
    b_has_rules = len(turn_b.thesis.invalidation_rules) > 0
    
    if a_has_rules and b_has_rules:
        a_features = {
            r.feature.split("_")[0] 
            for r in turn_a.thesis.invalidation_rules
        }
        b_features = {
            r.feature.split("_")[0] 
            for r in turn_b.thesis.invalidation_rules
        }
        invalidation_compatible = len(a_features.intersection(b_features)) > 0
    else:
        invalidation_compatible = True
    
    # Min confidence (all components >= 0.5) - RESTORED
    min_conf_a = turn_a.confidence.min_component()
    min_conf_b = turn_b.confidence.min_component()
    min_confidence_met = min_conf_a >= 0.5 and min_conf_b >= 0.5
    if not min_confidence_met:
        rejection_reasons.append(
            f"Confidence too low: {min_conf_a:.2f}, {min_conf_b:.2f} (need >= 0.5)"
        )
    
    # Evidence validated
    valid_a, errors_a = validate_evidence(
        turn_a.evidence_cited,
        turn_a.thesis.thesis_type,
        snapshot
    )
    valid_b, errors_b = validate_evidence(
        turn_b.evidence_cited,
        turn_b.thesis.thesis_type,
        snapshot
    )
    evidence_validated = valid_a and valid_b
    if not evidence_validated:
        if errors_a:
            rejection_reasons.append(f"Analyst evidence: {errors_a}")
        if errors_b:
            rejection_reasons.append(f"Critic evidence: {errors_b}")
    
    # Min edge (signal_strength >= 0.6) - RESTORED
    min_edge_met = (
        turn_a.confidence.signal_strength >= 0.6 and
        turn_b.confidence.signal_strength >= 0.6
    )
    if not min_edge_met:
        rejection_reasons.append(
            f"Edge too low: {turn_a.confidence.signal_strength:.2f}, "
            f"{turn_b.confidence.signal_strength:.2f} (need >= 0.6)"
        )
    
    return ConsensusGate(
        action_match=action_match,
        symbol_match=symbol_match,
        thesis_type_match=thesis_type_match,
        fund_strategy_match=fund_strategy_match,
        horizon_within_tolerance=horizon_within_tolerance,
        invalidation_compatible=invalidation_compatible,
        min_confidence_met=min_confidence_met,
        evidence_validated=evidence_validated,
        min_edge_met=min_edge_met,
        rejection_reasons=rejection_reasons,
    )


# =============================================================================
# Update Consensus Board from Agent Turns
# =============================================================================

def update_consensus_board(
    board: ConsensusBoard,
    turn_a: AgentTurn,
    turn_b: AgentTurn,
    snapshot: GlobalMarketSnapshot,
    fund_strategy: str = "",
) -> ConsensusBoard:
    """
    Update consensus board deterministically from agent turns.
    
    Computes the gate and agreed items. LLM fields are left for
    moderator to fill.
    
    Args:
        board: Current consensus board
        turn_a: First agent's turn
        turn_b: Second agent's turn
        snapshot: Current market snapshot
        fund_strategy: Fund's strategy for thesis validation
        
    Returns:
        Updated ConsensusBoard
    """
    # Compute deterministic gate
    gate = compute_consensus_gate(turn_a, turn_b, snapshot, fund_strategy)
    
    # Compute agreed items
    agreed_action = turn_a.action if gate.action_match else None
    
    # Symbol selection: pick higher confidence symbol when agents disagree
    if gate.action_match and turn_a.action != "hold":
        if turn_a.symbol == turn_b.symbol:
            agreed_symbol = turn_a.symbol
        else:
            # Different symbols - pick the one with higher confidence
            conf_a = turn_a.confidence.overall()
            conf_b = turn_b.confidence.overall()
            agreed_symbol = turn_a.symbol if conf_a >= conf_b else turn_b.symbol
    elif gate.action_match and turn_a.action == "hold":
        agreed_symbol = None
    else:
        agreed_symbol = None
    
    agreed_thesis_type = (
        turn_a.thesis.thesis_type if gate.thesis_type_match else None
    )
    
    # Average horizon if within tolerance
    agreed_horizon = None
    if gate.horizon_within_tolerance:
        agreed_horizon = (
            turn_a.thesis.horizon_days + turn_b.thesis.horizon_days
        ) // 2
    
    # Compute disputed topics
    disputed: List[str] = []
    if not gate.action_match:
        disputed.append(f"action: {turn_a.action} vs {turn_b.action}")
    if not gate.symbol_match:
        disputed.append(f"symbol: {turn_a.symbol} vs {turn_b.symbol}")
    if not gate.thesis_type_match:
        disputed.append(
            f"thesis: {turn_a.thesis.thesis_type.value} vs "
            f"{turn_b.thesis.thesis_type.value}"
        )
    if not gate.horizon_within_tolerance:
        disputed.append(
            f"horizon: {turn_a.thesis.horizon_days}d vs "
            f"{turn_b.thesis.horizon_days}d"
        )
    
    # Weight difference
    weight_diff = abs(turn_a.suggested_weight - turn_b.suggested_weight)
    if weight_diff > 0.05:
        disputed.append(
            f"sizing: {turn_a.suggested_weight:.0%} vs "
            f"{turn_b.suggested_weight:.0%}"
        )
    
    return ConsensusBoard(
        gate=gate,
        agreed_action=agreed_action,
        agreed_symbol=agreed_symbol,
        agreed_thesis_type=agreed_thesis_type,
        agreed_horizon=agreed_horizon,
        alignment_score=board.alignment_score,  # Keep existing LLM score
        open_questions=board.open_questions,
        concessions_made=board.concessions_made,
        disputed_topics=disputed,
        dispute_summary=board.dispute_summary,
    )


# =============================================================================
# Conservative Sizing Rule
# =============================================================================

def compute_conservative_weight(
    turn_a: AgentTurn,
    turn_b: AgentTurn,
) -> float:
    """
    Conservative sizing: take minimum unless explicit agreement.
    
    Args:
        turn_a: First agent's turn
        turn_b: Second agent's turn
        
    Returns:
        Conservative weight to use
    """
    weight_a = turn_a.suggested_weight
    weight_b = turn_b.suggested_weight
    
    # If weights differ by more than 5%, take the minimum
    if abs(weight_a - weight_b) > 0.05:
        return min(weight_a, weight_b)
    
    # If risk_comfort differs significantly, scale down
    risk_diff = abs(
        turn_a.confidence.risk_comfort - turn_b.confidence.risk_comfort
    )
    if risk_diff > 0.2:
        return min(weight_a, weight_b) * 0.8
    
    # Otherwise, average
    return (weight_a + weight_b) / 2


# =============================================================================
# Helper: Cosine Similarity for Experience Retrieval
# =============================================================================

def cosine_similarity(
    features_a: Dict[str, float],
    features_b: Dict[str, float],
) -> float:
    """
    Compute cosine similarity between two feature dictionaries.
    
    Args:
        features_a: First feature vector
        features_b: Second feature vector
        
    Returns:
        Cosine similarity (0-1)
    """
    # Get common keys
    common_keys = set(features_a.keys()) & set(features_b.keys())
    
    if not common_keys:
        return 0.0
    
    # Compute dot product and magnitudes
    dot_product = sum(
        features_a[k] * features_b[k] for k in common_keys
    )
    
    mag_a = math.sqrt(sum(features_a[k] ** 2 for k in common_keys))
    mag_b = math.sqrt(sum(features_b[k] ** 2 for k in common_keys))
    
    if mag_a == 0 or mag_b == 0:
        return 0.0
    
    return dot_product / (mag_a * mag_b)


# =============================================================================
# Experience Retrieval with Diversity
# =============================================================================

def retrieve_with_diversity(
    current_features: Dict[str, float],
    current_regime: Dict[str, str],
    current_date: date,
    all_records: List[ExperienceRecord],
    k: int = 3,
    max_age_days: int = 252,
    min_regime_similarity: float = 0.5,
) -> Dict[str, List[ExperienceRecord]]:
    """
    Retrieve similar + anti-similar with time decay and regime gates.
    
    Args:
        current_features: Current state features
        current_regime: Current regime tags
        current_date: Current date
        all_records: All experience records
        k: Number of similar records to return
        max_age_days: Maximum age of records to consider
        min_regime_similarity: Minimum regime similarity threshold
        
    Returns:
        Dict with "similar", "positive_precedent", "negative_precedent"
    """
    scored: List[Tuple[ExperienceRecord, float, float]] = []
    
    for record in all_records:
        # Time decay
        age_days = (current_date - record.record_date).days
        if age_days > max_age_days:
            continue
        if age_days < 0:
            continue  # Future record (shouldn't happen)
            
        # 50% decay at max age
        time_weight = 1.0 - (age_days / max_age_days) * 0.5
        
        # Feature similarity
        feat_sim = cosine_similarity(current_features, record.state_features)
        
        # Regime similarity
        if current_regime:
            regime_matches = sum(
                1 for key, val in current_regime.items()
                if record.regime_tags.get(key) == val
            )
            regime_sim = regime_matches / len(current_regime)
        else:
            regime_sim = 1.0
        
        # Skip if regime too different
        if regime_sim < min_regime_similarity:
            continue
        
        # Combined score
        score = feat_sim * 0.5 + regime_sim * 0.3 + time_weight * 0.2
        outcome = record.outcome_5d or 0.0
        
        scored.append((record, score, outcome))
    
    # Sort by similarity score
    scored.sort(key=lambda x: x[1], reverse=True)
    
    # Get results
    similar = [r for r, _, _ in scored[:k]]
    
    # Anti-examples: similar features but opposite outcome
    positive_precedent = [
        r for r, s, o in scored 
        if o > 0.02 and s > 0.5
    ][:1]
    
    negative_precedent = [
        r for r, s, o in scored 
        if o < -0.02 and s > 0.5
    ][:1]
    
    return {
        "similar": similar,
        "positive_precedent": positive_precedent,
        "negative_precedent": negative_precedent,
    }
