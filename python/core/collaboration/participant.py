"""
AI Participant - Refactored from LLMManager.

Key changes:
- No longer owns a portfolio
- Participates in debates
- Dynamic feature validation via snapshot.available_features()
"""
from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional, TYPE_CHECKING
import hashlib
import json

from core.collaboration.schemas import (
    ProposalOutput,
    CritiqueOutput,
    TradeCandidate,
    CritiqueItem,
)
from core.data.snapshot import SNAPSHOT_INSTRUCTION

if TYPE_CHECKING:
    from core.data.snapshot import GlobalMarketSnapshot
    from core.funds.fund import FundThesis


class InvalidFeatureError(Exception):
    """Raised when an AI references a feature not in snapshot."""
    pass


@dataclass
class ParticipantConfig:
    """Configuration for an AI participant."""
    participant_id: str
    model_provider: str  # "openai", "anthropic", "google"
    model_name: str
    prompt_template: str
    temperature: float = 0.3
    
    @property
    def prompt_hash(self) -> str:
        """Hash of prompt template for tracking."""
        return hashlib.sha256(self.prompt_template.encode()).hexdigest()[:12]


class AIParticipant:
    """
    AI participant in fund debates.
    
    Refactored from LLMManager:
    - No portfolio ownership
    - Focused on debate participation
    - Dynamic feature validation
    """
    
    def __init__(self, config: ParticipantConfig):
        """
        Initialize participant.
        
        Args:
            config: Participant configuration
        """
        self.config = config
        self._client = None  # Lazy loaded
    
    def propose(
        self,
        snapshot: "GlobalMarketSnapshot",
        thesis: "FundThesis",
        universe_symbols: List[str],
    ) -> ProposalOutput:
        """
        Propose trade candidates.
        
        Args:
            snapshot: Market snapshot (only source of data)
            thesis: Fund thesis for context
            universe_symbols: Tradable symbols for this fund
        
        Returns:
            ProposalOutput with candidates
        """
        # Build prompt
        prompt = self._build_proposal_prompt(snapshot, thesis, universe_symbols)
        
        # Call model
        response = self._call_model(prompt)
        
        # Parse and validate response
        candidates = self._parse_proposal_response(response, snapshot)
        
        return ProposalOutput(
            version="1.0",
            fund_id=thesis.name,
            asof_timestamp=snapshot.asof_timestamp,
            snapshot_id=snapshot.snapshot_id,
            participant_id=self.config.participant_id,
            model_name=self.config.model_name,
            model_version=self._get_model_version(),
            prompt_hash=self.config.prompt_hash,
            candidates=candidates,
            market_view=response.get("market_view", ""),
            key_drivers=response.get("key_drivers", []),
        )
    
    def critique(
        self,
        proposal: ProposalOutput,
        snapshot: "GlobalMarketSnapshot",
        thesis: "FundThesis",
    ) -> CritiqueOutput:
        """
        Critique a proposal.
        
        Args:
            proposal: Proposal to critique
            snapshot: Market snapshot
            thesis: Fund thesis
        
        Returns:
            CritiqueOutput with critiques
        """
        prompt = self._build_critique_prompt(proposal, snapshot, thesis)
        response = self._call_model(prompt)
        critiques = self._parse_critique_response(response)
        
        return CritiqueOutput(
            version="1.0",
            fund_id=thesis.name,
            asof_timestamp=snapshot.asof_timestamp,
            snapshot_id=snapshot.snapshot_id,
            participant_id=self.config.participant_id,
            model_name=self.config.model_name,
            model_version=self._get_model_version(),
            prompt_hash=self.config.prompt_hash,
            critiqued_proposal_id=proposal.participant_id,
            critiques=critiques,
            missing_data_flags=response.get("missing_data_flags", []),
            risk_flags=response.get("risk_flags", []),
            overall_assessment=response.get("overall_assessment", "neutral"),
        )
    
    def validate_features_used(
        self,
        features: List[str],
        snapshot: "GlobalMarketSnapshot"
    ) -> None:
        """
        Validate that all referenced features exist in snapshot.
        
        Uses snapshot.available_features() for dynamic validation.
        
        Args:
            features: Features referenced by the model
            snapshot: Snapshot to validate against
        
        Raises:
            InvalidFeatureError: If any feature is not available
        """
        available = snapshot.available_features()
        
        for f in features:
            if f not in available:
                raise InvalidFeatureError(
                    f"Feature '{f}' not available in snapshot. "
                    f"Available: {sorted(available)}"
                )
    
    def _build_proposal_prompt(
        self,
        snapshot: "GlobalMarketSnapshot",
        thesis: "FundThesis",
        universe_symbols: List[str],
    ) -> str:
        """Build prompt for proposal phase."""
        # Serialize snapshot data for prompt
        snapshot_data = self._serialize_snapshot_for_prompt(snapshot, universe_symbols)
        
        return f"""
{SNAPSHOT_INSTRUCTION}

You are an AI participant in a {thesis.name} fund debate.

FUND THESIS:
- Strategy: {thesis.strategy}
- Description: {thesis.description}
- Horizon: {thesis.horizon_days[0]}-{thesis.horizon_days[1]} days
- Edge: {thesis.edge}

TRADABLE UNIVERSE:
{json.dumps(universe_symbols, indent=2)}

MARKET DATA (ONLY use this data):
{snapshot_data}

TASK:
Propose 0-5 trade candidates. For each candidate, provide:
- symbol: The stock symbol
- direction: "long", "short", or "flat"
- target_weight: Recommended portfolio weight (0.01-0.15)
- expected_horizon_days: Expected holding period
- rationale: Why this trade
- key_features_used: Which data you used (must be from available features)
- confidence: Your confidence (0-1)
- failure_modes: What could go wrong

Also provide:
- market_view: Your overall market assessment
- key_drivers: Main drivers of your recommendations

Respond in JSON format.
"""
    
    def _build_critique_prompt(
        self,
        proposal: ProposalOutput,
        snapshot: "GlobalMarketSnapshot",
        thesis: "FundThesis",
    ) -> str:
        """Build prompt for critique phase."""
        snapshot_data = self._serialize_snapshot_for_prompt(
            snapshot,
            [c.symbol for c in proposal.candidates]
        )
        
        return f"""
{SNAPSHOT_INSTRUCTION}

You are critiquing a proposal for the {thesis.name} fund.

PROPOSAL TO CRITIQUE:
{json.dumps(proposal.to_dict(), indent=2)}

MARKET DATA (for verification):
{snapshot_data}

TASK:
Critique each trade candidate. For each critique:
- target_symbol: Which symbol
- issue_type: "data_concern", "risk", "counter_thesis", "missing_info"
- description: What's the issue
- severity: "minor", "major", "critical"
- counter_argument: Alternative perspective (optional)

Also provide:
- missing_data_flags: Data we wish we had
- risk_flags: Risks identified
- overall_assessment: "support", "neutral", or "oppose"

Respond in JSON format.
"""
    
    def _serialize_snapshot_for_prompt(
        self,
        snapshot: "GlobalMarketSnapshot",
        symbols: List[str]
    ) -> str:
        """Serialize relevant snapshot data for prompt."""
        data = {
            "snapshot_id": snapshot.snapshot_id,
            "asof_timestamp": snapshot.asof_timestamp.isoformat(),
            "available_features": list(snapshot.available_features()),
            "prices": {s: snapshot.prices.get(s) for s in symbols if s in snapshot.prices},
            "returns": {
                s: snapshot.returns.get(s, {})
                for s in symbols if s in snapshot.returns
            },
            "volatility": {
                s: snapshot.volatility.get(s, {})
                for s in symbols if s in snapshot.volatility
            },
            "upcoming_earnings": [
                {
                    "symbol": e.symbol,
                    "date": e.date,
                    "time": e.time,
                }
                for e in snapshot.upcoming_earnings
                if e.symbol in symbols
            ],
            "news_summaries": [
                {
                    "headline": n.headline,
                    "symbols": n.symbols,
                    "sentiment": n.sentiment,
                }
                for n in snapshot.news_summaries
                if any(s in symbols for s in n.symbols)
            ][:5],  # Limit to 5 news items
        }
        return json.dumps(data, indent=2, default=str)
    
    def _call_model(self, prompt: str) -> dict:
        """
        Call the language model.
        
        For v1, this is a placeholder. In production, this would
        call the appropriate provider's API.
        """
        # Placeholder - in production, would call OpenAI/Anthropic/etc
        return {
            "candidates": [],
            "market_view": "Market view not generated (placeholder)",
            "key_drivers": [],
            "critiques": [],
            "missing_data_flags": [],
            "risk_flags": [],
            "overall_assessment": "neutral",
        }
    
    def _parse_proposal_response(
        self,
        response: dict,
        snapshot: "GlobalMarketSnapshot"
    ) -> List[TradeCandidate]:
        """Parse and validate proposal response."""
        candidates = []
        
        for c in response.get("candidates", []):
            # Validate features
            features = c.get("key_features_used", [])
            self.validate_features_used(features, snapshot)
            
            candidates.append(TradeCandidate(
                symbol=c.get("symbol", ""),
                direction=c.get("direction", "flat"),
                target_weight=c.get("target_weight", 0.0),
                expected_horizon_days=c.get("expected_horizon_days", 1),
                rationale=c.get("rationale", ""),
                key_features_used=features,
                confidence=c.get("confidence", 0.5),
                failure_modes=c.get("failure_modes", []),
            ))
        
        return candidates
    
    def _parse_critique_response(self, response: dict) -> List[CritiqueItem]:
        """Parse critique response."""
        critiques = []
        
        for c in response.get("critiques", []):
            critiques.append(CritiqueItem(
                target_symbol=c.get("target_symbol", ""),
                issue_type=c.get("issue_type", "risk"),
                description=c.get("description", ""),
                severity=c.get("severity", "minor"),
                counter_argument=c.get("counter_argument"),
            ))
        
        return critiques
    
    def _get_model_version(self) -> str:
        """Get model version string."""
        # For v1, return model name as version
        # In production, could query provider for actual version
        return self.config.model_name
