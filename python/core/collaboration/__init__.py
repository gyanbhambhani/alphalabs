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
]
