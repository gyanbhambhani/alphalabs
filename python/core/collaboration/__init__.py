"""
Collaboration module - AI debate engine and participants.
"""
from core.collaboration.schemas import (
    ProposalOutput,
    CritiqueOutput,
    SynthesisOutput,
    FinalizeOutput,
)
from core.collaboration.participant import (
    AIParticipant,
    InvalidFeatureError,
)
from core.collaboration.debate import (
    DebateEngine,
    DebateTranscript,
    DebatePhase,
)

# V2.1 Collaborative Debate System
from core.collaboration.debate_v2 import (
    AVAILABLE_FEATURES,
    THESIS_REQUIRED_EVIDENCE,
    THESIS_INTERPRETATIONS,
    ThesisType,
    InvalidationRule,
    EvidenceReference,
    ThesisProposal,
    ConfidenceDecomposition,
    DialogMove,
    Counterfactual,
    AgentTurn,
    ConsensusGate,
    ConsensusBoard,
    RiskManagerDecision,
    ExperienceRecord,
    DebateOutput,
    validate_evidence,
    compute_consensus_gate,
    update_consensus_board,
    compute_conservative_weight,
    cosine_similarity,
    retrieve_with_diversity,
)

__all__ = [
    # Schemas
    "ProposalOutput",
    "CritiqueOutput",
    "SynthesisOutput",
    "FinalizeOutput",
    # Participant
    "AIParticipant",
    "InvalidFeatureError",
    # Debate
    "DebateEngine",
    "DebateTranscript",
    "DebatePhase",
    # V2.1 Types
    "AVAILABLE_FEATURES",
    "THESIS_REQUIRED_EVIDENCE",
    "THESIS_INTERPRETATIONS",
    "ThesisType",
    "InvalidationRule",
    "EvidenceReference",
    "ThesisProposal",
    "ConfidenceDecomposition",
    "DialogMove",
    "Counterfactual",
    "AgentTurn",
    "ConsensusGate",
    "ConsensusBoard",
    "RiskManagerDecision",
    "ExperienceRecord",
    "DebateOutput",
    # V2.1 Functions
    "validate_evidence",
    "compute_consensus_gate",
    "update_consensus_board",
    "compute_conservative_weight",
    "cosine_similarity",
    "retrieve_with_diversity",
]
