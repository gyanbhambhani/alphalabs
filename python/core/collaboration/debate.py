"""
Debate Engine - Orchestrates the propose/critique/synthesize/risk_check/finalize pipeline.

Key principles:
- Deterministic pipeline (no voting by vibes)
- PM Finalizer makes the call
- Risk Manager can veto
- Full audit trail
"""
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any, TYPE_CHECKING
import uuid
import hashlib

from core.collaboration.schemas import (
    ProposalOutput,
    CritiqueOutput,
    SynthesisOutput,
    FinalizeOutput,
    MergedPlan,
    FinalPosition,
)
from core.collaboration.participant import AIParticipant, ParticipantConfig
from core.config.constants import CONSTANTS
from core.data.snapshot import SNAPSHOT_INSTRUCTION

if TYPE_CHECKING:
    from core.data.snapshot import GlobalMarketSnapshot
    from core.funds.fund import Fund
    from core.execution.risk_manager import RiskManager, RiskCheckResult
    from core.execution.intent import PortfolioIntent


class DebatePhase(Enum):
    """Phases of the debate."""
    PROPOSE = "propose"
    CRITIQUE = "critique"
    SYNTHESIZE = "synthesize"
    RISK_CHECK = "risk_check"
    FINALIZE = "finalize"


@dataclass
class DebateMessage:
    """A single message in the debate."""
    phase: DebatePhase
    participant_id: str
    timestamp: datetime
    content: Dict[str, Any]  # JSON-serializable output
    model_name: str
    model_version: str
    prompt_hash: str


@dataclass
class DebateTranscript:
    """
    Raw debate transcript for humans and debugging.
    
    Separate from DecisionRecord which is for machines and backtests.
    """
    transcript_id: str
    fund_id: str
    snapshot_id: str
    started_at: datetime
    completed_at: Optional[datetime] = None
    
    messages: List[DebateMessage] = field(default_factory=list)
    
    # Summary stats
    num_proposals: int = 0
    num_critiques: int = 0
    final_consensus_level: float = 0.0
    
    # Token usage tracking
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    
    def add_message(self, message: DebateMessage) -> None:
        """Add a message to the transcript."""
        self.messages.append(message)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "transcript_id": self.transcript_id,
            "fund_id": self.fund_id,
            "snapshot_id": self.snapshot_id,
            "started_at": self.started_at.isoformat(),
            "completed_at": (
                self.completed_at.isoformat() if self.completed_at else None
            ),
            "messages": [
                {
                    "phase": m.phase.value,
                    "participant_id": m.participant_id,
                    "timestamp": m.timestamp.isoformat(),
                    "content": m.content,
                    "model_name": m.model_name,
                    "model_version": m.model_version,
                    "prompt_hash": m.prompt_hash,
                }
                for m in self.messages
            ],
            "num_proposals": self.num_proposals,
            "num_critiques": self.num_critiques,
            "final_consensus_level": self.final_consensus_level,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
        }


@dataclass
class DebateResult:
    """Result of running a full debate."""
    success: bool
    decision: str  # "trade", "no_trade"
    intent: Optional["PortfolioIntent"] = None
    risk_result: Optional["RiskCheckResult"] = None
    transcript: Optional[DebateTranscript] = None
    synthesis: Optional[SynthesisOutput] = None
    final_output: Optional[FinalizeOutput] = None
    error: Optional[str] = None


class DebateEngine:
    """
    Orchestrates the full debate pipeline.
    
    Pipeline: propose -> critique -> synthesize -> risk_check -> finalize
    """
    
    def __init__(
        self,
        participants: List[AIParticipant],
        synthesizer_config: ParticipantConfig,
        risk_manager: "RiskManager",
    ):
        """
        Initialize debate engine.
        
        Args:
            participants: AI participants for debate
            synthesizer_config: Config for synthesis model
            risk_manager: Risk manager for validation
        """
        self.participants = participants
        self.synthesizer_config = synthesizer_config
        self.risk_manager = risk_manager
    
    def run_debate(
        self,
        fund: "Fund",
        snapshot: "GlobalMarketSnapshot",
        universe_symbols: List[str],
    ) -> DebateResult:
        """
        Run a full debate cycle.
        
        Args:
            fund: The fund running the debate
            snapshot: Market snapshot
            universe_symbols: Tradable symbols
        
        Returns:
            DebateResult with intent or no-trade decision
        """
        transcript = DebateTranscript(
            transcript_id=str(uuid.uuid4()),
            fund_id=fund.fund_id,
            snapshot_id=snapshot.snapshot_id,
            started_at=datetime.utcnow(),
        )
        
        try:
            # Phase 1: Propose
            proposals = self._run_propose_phase(
                fund, snapshot, universe_symbols, transcript
            )
            
            if not proposals:
                return DebateResult(
                    success=True,
                    decision="no_trade",
                    transcript=transcript,
                    error="No proposals generated"
                )
            
            # Phase 2: Critique
            critiques = self._run_critique_phase(
                proposals, snapshot, fund.thesis, transcript
            )
            
            # Phase 3: Synthesize
            synthesis = self._run_synthesize_phase(
                proposals, critiques, snapshot, fund, transcript
            )
            
            # Check consensus threshold
            if synthesis.consensus_level < CONSTANTS.debate.CONSENSUS_THRESHOLD:
                return DebateResult(
                    success=True,
                    decision="no_trade",
                    transcript=transcript,
                    synthesis=synthesis,
                    error="Insufficient consensus"
                )
            
            # Phase 4: Risk Check (pre-finalize)
            # Create preliminary intent from synthesis
            prelim_intent = self._synthesis_to_intent(synthesis, fund, snapshot)
            risk_result = self.risk_manager.check_intent(
                prelim_intent, fund, fund.portfolio
            )
            
            if risk_result.status == "vetoed":
                transcript.completed_at = datetime.utcnow()
                return DebateResult(
                    success=True,
                    decision="no_trade",
                    transcript=transcript,
                    synthesis=synthesis,
                    risk_result=risk_result,
                    error="Risk veto"
                )
            
            # Phase 5: Finalize (PM decision)
            final_output = self._run_finalize_phase(
                synthesis, risk_result, fund, snapshot, transcript
            )
            
            if final_output.decision == "no_trade":
                transcript.completed_at = datetime.utcnow()
                return DebateResult(
                    success=True,
                    decision="no_trade",
                    transcript=transcript,
                    synthesis=synthesis,
                    final_output=final_output,
                )
            
            # Create final intent
            final_intent = self._finalize_to_intent(final_output, fund, snapshot)
            
            transcript.completed_at = datetime.utcnow()
            transcript.final_consensus_level = synthesis.consensus_level
            
            return DebateResult(
                success=True,
                decision="trade",
                intent=final_intent,
                risk_result=risk_result,
                transcript=transcript,
                synthesis=synthesis,
                final_output=final_output,
            )
            
        except Exception as e:
            transcript.completed_at = datetime.utcnow()
            return DebateResult(
                success=False,
                decision="no_trade",
                transcript=transcript,
                error=str(e),
            )
    
    def _run_propose_phase(
        self,
        fund: "Fund",
        snapshot: "GlobalMarketSnapshot",
        universe_symbols: List[str],
        transcript: DebateTranscript,
    ) -> List[ProposalOutput]:
        """Run propose phase with all participants."""
        proposals: List[ProposalOutput] = []
        
        for participant in self.participants:
            proposal = participant.propose(snapshot, fund.thesis, universe_symbols)
            
            transcript.add_message(DebateMessage(
                phase=DebatePhase.PROPOSE,
                participant_id=participant.config.participant_id,
                timestamp=datetime.utcnow(),
                content=proposal.to_dict(),
                model_name=participant.config.model_name,
                model_version=proposal.model_version,
                prompt_hash=proposal.prompt_hash,
            ))
            
            proposals.append(proposal)
        
        transcript.num_proposals = len(proposals)
        return proposals
    
    def _run_critique_phase(
        self,
        proposals: List[ProposalOutput],
        snapshot: "GlobalMarketSnapshot",
        thesis: "FundThesis",
        transcript: DebateTranscript,
    ) -> List[CritiqueOutput]:
        """Run critique phase - each participant critiques others."""
        critiques: List[CritiqueOutput] = []
        
        for proposal in proposals:
            for participant in self.participants:
                # Don't critique yourself
                if participant.config.participant_id == proposal.participant_id:
                    continue
                
                critique = participant.critique(proposal, snapshot, thesis)
                
                transcript.add_message(DebateMessage(
                    phase=DebatePhase.CRITIQUE,
                    participant_id=participant.config.participant_id,
                    timestamp=datetime.utcnow(),
                    content=critique.to_dict(),
                    model_name=participant.config.model_name,
                    model_version=critique.model_version,
                    prompt_hash=critique.prompt_hash,
                ))
                
                critiques.append(critique)
        
        transcript.num_critiques = len(critiques)
        return critiques
    
    def _run_synthesize_phase(
        self,
        proposals: List[ProposalOutput],
        critiques: List[CritiqueOutput],
        snapshot: "GlobalMarketSnapshot",
        fund: "Fund",
        transcript: DebateTranscript,
    ) -> SynthesisOutput:
        """Synthesize proposals and critiques into merged plans."""
        # Collect all candidates across proposals
        all_candidates: Dict[str, List[Any]] = {}  # symbol -> list of (proposal, candidate)
        
        for proposal in proposals:
            for candidate in proposal.candidates:
                if candidate.symbol not in all_candidates:
                    all_candidates[candidate.symbol] = []
                all_candidates[candidate.symbol].append((proposal, candidate))
        
        # Collect critiques by symbol
        critiques_by_symbol: Dict[str, List[CritiqueOutput]] = {}
        for critique in critiques:
            for item in critique.critiques:
                if item.target_symbol not in critiques_by_symbol:
                    critiques_by_symbol[item.target_symbol] = []
                critiques_by_symbol[item.target_symbol].append(critique)
        
        # Merge plans
        merged_plans: List[MergedPlan] = []
        
        for symbol, candidates in all_candidates.items():
            # Calculate consensus
            directions = [c.direction for _, c in candidates]
            dominant_direction = max(set(directions), key=directions.count)
            consensus_level = directions.count(dominant_direction) / len(directions)
            
            # Find supporters and opposers
            supporters = [
                p.participant_id for p, c in candidates
                if c.direction == dominant_direction
            ]
            opposers = [
                p.participant_id for p, c in candidates
                if c.direction != dominant_direction
            ]
            
            # Average target weight among supporters
            supporter_weights = [
                c.target_weight for p, c in candidates
                if c.direction == dominant_direction
            ]
            avg_weight = sum(supporter_weights) / len(supporter_weights) if supporter_weights else 0  # noqa
            
            # Gather unresolved concerns
            concerns = []
            for critique in critiques_by_symbol.get(symbol, []):
                for item in critique.critiques:
                    if item.target_symbol == symbol and item.severity in ["major", "critical"]:  # noqa
                        concerns.append(item.description)
            
            # Combine reasoning
            reasoning = "; ".join(
                c.rationale[:100] for _, c in candidates
                if c.direction == dominant_direction
            )[:300]
            
            merged_plans.append(MergedPlan(
                symbol=symbol,
                direction=dominant_direction,
                target_weight=avg_weight,
                consensus_level=consensus_level,
                supporting_participants=supporters,
                opposing_participants=opposers,
                key_reasoning=reasoning,
                unresolved_concerns=concerns[:3],
            ))
        
        # Calculate overall consensus
        overall_consensus = (
            sum(p.consensus_level for p in merged_plans) / len(merged_plans)
            if merged_plans else 0.0
        )
        
        synthesis = SynthesisOutput(
            version="1.0",
            fund_id=fund.fund_id,
            asof_timestamp=snapshot.asof_timestamp,
            snapshot_id=snapshot.snapshot_id,
            model_name=self.synthesizer_config.model_name,
            model_version=self.synthesizer_config.model_name,
            prompt_hash=self.synthesizer_config.prompt_hash,
            merged_plans=merged_plans,
            consensus_level=overall_consensus,
            key_agreements=[],
            key_disagreements=[],
        )
        
        transcript.add_message(DebateMessage(
            phase=DebatePhase.SYNTHESIZE,
            participant_id="synthesizer",
            timestamp=datetime.utcnow(),
            content=synthesis.to_dict(),
            model_name=self.synthesizer_config.model_name,
            model_version=self.synthesizer_config.model_name,
            prompt_hash=self.synthesizer_config.prompt_hash,
        ))
        
        return synthesis
    
    def _run_finalize_phase(
        self,
        synthesis: SynthesisOutput,
        risk_result: "RiskCheckResult",
        fund: "Fund",
        snapshot: "GlobalMarketSnapshot",
        transcript: DebateTranscript,
    ) -> FinalizeOutput:
        """PM Finalizer makes the final decision."""
        # Apply risk scaling to synthesis
        scale_factor = risk_result.scale_factor
        
        # Filter plans that pass threshold
        viable_plans = [
            p for p in synthesis.merged_plans
            if p.consensus_level >= CONSTANTS.debate.CONSENSUS_THRESHOLD
        ]
        
        if not viable_plans:
            return FinalizeOutput(
                version="1.0",
                fund_id=fund.fund_id,
                asof_timestamp=snapshot.asof_timestamp,
                snapshot_id=snapshot.snapshot_id,
                model_name=fund.pm_config.model_name,
                model_version=fund.pm_config.model_name,
                prompt_hash=fund.pm_config.prompt_hash,
                decision="no_trade",
                positions=[],
                target_cash_pct=1.0,
                sizing_method_used=fund.policy.sizing_method,
                policy_version=fund.policy.version,
                key_reasoning="No viable plans after filtering",
                risk_notes=["No consensus above threshold"],
            )
        
        # Create final positions
        positions: List[FinalPosition] = []
        for plan in viable_plans[:fund.policy.max_positions]:
            scaled_weight = plan.target_weight * scale_factor
            
            positions.append(FinalPosition(
                symbol=plan.symbol,
                target_weight=scaled_weight,
                direction=plan.direction,
                stop_loss_pct=fund.policy.default_stop_loss_pct,
                take_profit_pct=fund.policy.default_take_profit_pct,
                expected_holding_days=(
                    fund.thesis.horizon_days[0] + fund.thesis.horizon_days[1]
                ) // 2,
                exit_rationale=f"Default policy exits: SL={fund.policy.default_stop_loss_pct:.0%}, TP={fund.policy.default_take_profit_pct:.0%}",  # noqa
            ))
        
        total_weight = sum(abs(p.target_weight) for p in positions)
        target_cash = max(fund.policy.min_cash_buffer, 1.0 - total_weight)
        
        final_output = FinalizeOutput(
            version="1.0",
            fund_id=fund.fund_id,
            asof_timestamp=snapshot.asof_timestamp,
            snapshot_id=snapshot.snapshot_id,
            model_name=fund.pm_config.model_name,
            model_version=fund.pm_config.model_name,
            prompt_hash=fund.pm_config.prompt_hash,
            decision="trade",
            positions=positions,
            target_cash_pct=target_cash,
            sizing_method_used=fund.policy.sizing_method,
            policy_version=fund.policy.version,
            key_reasoning=f"Trading {len(positions)} positions with {scale_factor:.0%} scaling",  # noqa
            risk_notes=[v.rule_name for v in risk_result.violations],
        )
        
        transcript.add_message(DebateMessage(
            phase=DebatePhase.FINALIZE,
            participant_id="pm_finalizer",
            timestamp=datetime.utcnow(),
            content=final_output.to_dict(),
            model_name=fund.pm_config.model_name,
            model_version=fund.pm_config.model_name,
            prompt_hash=fund.pm_config.prompt_hash,
        ))
        
        return final_output
    
    def _synthesis_to_intent(
        self,
        synthesis: SynthesisOutput,
        fund: "Fund",
        snapshot: "GlobalMarketSnapshot",
    ) -> "PortfolioIntent":
        """Convert synthesis to preliminary intent for risk check."""
        from core.execution.intent import PortfolioIntent, PositionIntent
        
        positions = [
            PositionIntent(
                symbol=plan.symbol,
                target_weight=plan.target_weight,
                direction=plan.direction,
            )
            for plan in synthesis.merged_plans
        ]
        
        total_weight = sum(abs(p.target_weight) for p in positions)
        
        return PortfolioIntent(
            intent_id=f"prelim_{snapshot.snapshot_id}",
            fund_id=fund.fund_id,
            asof_timestamp=snapshot.asof_timestamp,
            portfolio_value=fund.portfolio.total_value,
            asof_prices=snapshot.prices.copy(),
            valuation_timestamp=snapshot.asof_timestamp,
            positions=positions,
            target_cash_pct=max(fund.policy.min_cash_buffer, 1.0 - total_weight),
            max_turnover=fund.policy.max_turnover_daily,
            execution_window_minutes=60,
            sizing_method_used=fund.policy.sizing_method,
            policy_version=fund.policy.version,
        )
    
    def _finalize_to_intent(
        self,
        final_output: FinalizeOutput,
        fund: "Fund",
        snapshot: "GlobalMarketSnapshot",
    ) -> "PortfolioIntent":
        """Convert finalize output to portfolio intent."""
        from core.execution.intent import PortfolioIntent, PositionIntent, ExitRule
        
        positions = [
            PositionIntent(
                symbol=pos.symbol,
                target_weight=pos.target_weight,
                direction=pos.direction,
            )
            for pos in final_output.positions
        ]
        
        exit_rules = [
            ExitRule(
                symbol=pos.symbol,
                stop_loss_pct=pos.stop_loss_pct,
                take_profit_pct=pos.take_profit_pct,
            )
            for pos in final_output.positions
        ]
        
        return PortfolioIntent(
            intent_id=f"final_{snapshot.snapshot_id}",
            fund_id=fund.fund_id,
            asof_timestamp=snapshot.asof_timestamp,
            portfolio_value=fund.portfolio.total_value,
            asof_prices=snapshot.prices.copy(),
            valuation_timestamp=snapshot.asof_timestamp,
            positions=positions,
            target_cash_pct=final_output.target_cash_pct,
            max_turnover=fund.policy.max_turnover_daily,
            execution_window_minutes=60,
            sizing_method_used=final_output.sizing_method_used,
            policy_version=final_output.policy_version,
            exit_rules=exit_rules,
        )
