"""
Daily Debate Runner for Backtesting.

Runs the 3-phase AI debate (analyze → propose → decide) for each fund.
Uses LangChain for orchestration with multi-model strategy:
- Gemini for analysis
- GPT for proposals and decisions
- Claude for major trade confirmation

Refactored to use LangChain chains for structured output and better maintainability.
"""

import os
import asyncio
from datetime import date, datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
import logging
from pathlib import Path

from core.data.snapshot import GlobalMarketSnapshot
from core.backtest.portfolio_tracker import BacktestPortfolio
from core.backtest.events import DebateMessageEvent, DecisionEvent
from core.backtest.deidentifier import get_deidentifier
from core.backtest.validator import DecisionValidator, ValidationResult

# Import LangChain chains
from core.langchain.chains import (
    AnalyzeChain,
    ProposeChain,
    DecideChain,
    ConfirmChain,
)
from core.langchain.schemas import (
    MarketAnalysis,
    TradeProposal,
    TradingDecisionOutput,
    RiskConfirmation,
)

logger = logging.getLogger(__name__)

# Load environment variables from .env.local and .env
try:
    from dotenv import load_dotenv
    current = Path(__file__).resolve()
    for parent in [current] + list(current.parents):
        env_local = parent / ".env.local"
        env_file = parent / ".env"
        if env_local.exists():
            load_dotenv(env_local)
            logger.info(f"Loaded env from {env_local}")
            break
        if env_file.exists():
            load_dotenv(env_file)
            logger.info(f"Loaded env from {env_file}")
            break
except ImportError:
    logger.warning("python-dotenv not installed, using system env vars only")


@dataclass
class DebateMessage:
    """A single message in the debate."""
    phase: str  # "analyze", "propose", "decide", "confirm"
    model: str  # "gemini", "gpt", "claude"
    content: str
    tokens_used: int = 0
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class TradingDecision:
    """The result of a debate - a trading decision."""
    action: str  # "buy", "sell", "hold"
    symbol: Optional[str] = None
    quantity: Optional[float] = None
    target_weight: Optional[float] = None
    reasoning: str = ""
    confidence: float = 0.5
    signals_used: Dict[str, float] = field(default_factory=dict)
    debate_transcript: List[DebateMessage] = field(default_factory=list)
    models_used: Dict[str, str] = field(default_factory=dict)
    total_tokens: int = 0
    requires_confirmation: bool = False


class DailyDebateRunner:
    """
    Runs the daily AI debate for a fund using LangChain chains.
    
    3-phase debate:
    1. ANALYZE (Gemini): Analyze market data, identify opportunities
    2. PROPOSE (GPT): Propose specific trades based on analysis
    3. DECIDE (GPT): Make final decision with structured output
    
    For major trades (>5% of portfolio), Claude confirms.
    """
    
    # Thresholds
    MAJOR_TRADE_THRESHOLD = 0.05  # 5% of portfolio
    
    def __init__(
        self,
        analyze_model: str = "gemini-1.5-flash",
        propose_model: str = "gpt-4o-mini",
        decide_model: str = "gpt-4o-mini",
        confirm_model: str = "claude-3-haiku-20240307",
    ):
        """
        Initialize the debate runner with LangChain chains.
        
        Args:
            analyze_model: Model for analysis phase (default: Gemini)
            propose_model: Model for proposal phase (default: GPT-4o-mini)
            decide_model: Model for decision phase (default: GPT-4o-mini)
            confirm_model: Model for confirmation phase (default: Claude Haiku)
        """
        # Initialize LangChain chains
        self.analyze_chain = AnalyzeChain(model=analyze_model)
        self.propose_chain = ProposeChain(model=propose_model)
        self.decide_chain = DecideChain(model=decide_model)
        self.confirm_chain = ConfirmChain(model=confirm_model)
        
        # Store model names for tracking
        self._model_names = {
            "analyze": analyze_model,
            "propose": propose_model,
            "decide": decide_model,
            "confirm": confirm_model,
        }
    
    async def run_debate(
        self,
        fund_id: str,
        fund_name: str,
        fund_thesis: str,
        portfolio: BacktestPortfolio,
        snapshot: GlobalMarketSnapshot,
        simulation_date: date,
        trade_budget: Optional["TradeBudget"] = None,  # NEW: Budget enforcement
    ) -> TradingDecision:
        """
        Run the full 3-phase debate for a fund.
        
        Args:
            fund_id: Fund identifier
            fund_name: Human-readable fund name
            fund_thesis: Fund's investment thesis/strategy
            portfolio: Current portfolio state
            snapshot: Market data snapshot (point-in-time)
            simulation_date: Current simulation date
            trade_budget: TradeBudget for enforcement (replaces available_trades)
            
        Returns:
            TradingDecision with action, reasoning, and transcript
        """
        transcript: List[DebateMessage] = []
        models_used: Dict[str, str] = {}
        total_tokens = 0
        
        # Build context for prompts (with budget constraints)
        context = self._build_context(
            fund_name, fund_thesis, portfolio, snapshot, 
            simulation_date, trade_budget
        )
        
        # Phase 1: ANALYZE (Gemini via LangChain)
        analysis = await self._phase_analyze(context, transcript)
        models_used["analyze"] = self._model_names["analyze"]
        total_tokens += transcript[-1].tokens_used if transcript else 0
        
        # Phase 2: PROPOSE (GPT via LangChain)
        proposal = await self._phase_propose(context, analysis, transcript)
        models_used["propose"] = self._model_names["propose"]
        total_tokens += transcript[-1].tokens_used if transcript else 0
        
        # Phase 3: DECIDE (GPT via LangChain with structured output)
        decision = await self._phase_decide(
            context, analysis, proposal, portfolio, transcript
        )
        models_used["decide"] = self._model_names["decide"]
        total_tokens += transcript[-1].tokens_used if transcript else 0
        
        # Validation gate (safety check)
        if trade_budget and decision.symbol:
            # Create validator
            validator = DecisionValidator(
                max_position_pct=0.20,  # TODO: Get from fund policy
                allowed_features=snapshot.available_features(),
            )
            
            # Validate decision
            decision_dict = {
                "action": decision.action,
                "asset_id": decision.symbol,  # De-identified Asset_###
                "target_weight": decision.target_weight,
                "reasoning": decision.reasoning,
                "confidence": decision.confidence,
            }
            
            validation_result = validator.validate(
                decision_dict,
                budget=trade_budget,
                candidates=None,  # TODO: Pass candidate set
            )
            
            if not validation_result.valid:
                logger.warning(
                    f"[{fund_id}] Decision rejected by validator: "
                    f"{validation_result.reason}"
                )
                decision = TradingDecision(
                    action="hold",
                    reasoning=f"Validator rejected: {validation_result.reason}",
                    confidence=0.0,
                )
        
        # Re-identify asset (convert Asset_### back to ticker)
        if decision.symbol and decision.symbol.startswith("Asset_"):
            deidentifier = get_deidentifier()
            real_ticker = deidentifier.reidentify_ticker(decision.symbol)
            if real_ticker:
                decision.symbol = real_ticker
            else:
                logger.error(
                    f"Could not re-identify {decision.symbol}, converting to HOLD"
                )
                decision = TradingDecision(
                    action="hold",
                    reasoning="Re-identification failed",
                    confidence=0.0,
                )
        
        # Budget validation gate (safety check)
        if trade_budget and decision.action == "buy" and not trade_budget.can_buy():
            logger.warning(
                f"[{fund_id}] Buy decision overridden by budget gate"
            )
            decision = TradingDecision(
                action="hold",
                reasoning=f"Budget denied: {decision.reasoning}",
                confidence=0.1,
            )
        
        # Check if major trade needs Claude confirmation
        if decision.action != "hold" and decision.target_weight:
            if decision.target_weight > self.MAJOR_TRADE_THRESHOLD:
                decision.requires_confirmation = True
                confirmed = await self._phase_confirm(context, decision, transcript)
                models_used["confirm"] = self._model_names["confirm"]
                total_tokens += transcript[-1].tokens_used if transcript else 0
                
                if not confirmed:
                    # Claude rejected - convert to hold
                    decision = TradingDecision(
                        action="hold",
                        reasoning=f"Major trade rejected by risk manager: {decision.reasoning}",
                        confidence=0.3,
                    )
        
        decision.debate_transcript = transcript
        decision.models_used = models_used
        decision.total_tokens = total_tokens
        
        return decision
    
    def _build_context(
        self,
        fund_name: str,
        fund_thesis: str,
        portfolio: BacktestPortfolio,
        snapshot: GlobalMarketSnapshot,
        simulation_date: date,
        trade_budget: Optional["TradeBudget"] = None,
    ) -> str:
        """
        Build context string for prompts with de-identification.
        
        Tickers are replaced with Asset_### to prevent temporal leakage.
        """
        deidentifier = get_deidentifier()
        
        # Portfolio summary (de-identified)
        positions_str = ""
        for sym, pos in portfolio.positions.items():
            asset_id = deidentifier.deidentify_ticker(sym)
            weight = portfolio.get_position_weight(sym)
            positions_str += (
                f"  - {asset_id}: {pos.quantity:.0f} shares @ "
                f"${pos.current_price:.2f} "
                f"({weight:.1%} of portfolio, "
                f"{pos.unrealized_return:+.1%} unrealized)\n"
            )
        
        if not positions_str:
            positions_str = "  (No positions)\n"
        
        # Top movers from snapshot (de-identified)
        top_movers = []
        for sym in snapshot.prices.keys():
            ret_1d = snapshot.get_return(sym, "1d")
            if ret_1d is not None:
                asset_id = deidentifier.deidentify_ticker(sym)
                top_movers.append((asset_id, ret_1d))
        
        top_movers.sort(key=lambda x: abs(x[1]), reverse=True)
        movers_str = "\n".join([
            f"  - {asset_id}: {ret:+.1%} (1d)"
            for asset_id, ret in top_movers[:10]
        ])
        
        # Note: We removed the explicit date and use "Day N" notation
        # to reduce temporal leakage
        context = f"""
FUND: {fund_name}
THESIS: {fund_thesis}

IMPORTANT: Asset identifiers are de-identified (Asset_###).
You may ONLY use numeric features provided. Do NOT reference company
names, products, brands, or external knowledge.

PORTFOLIO STATE:
  Total Value: ${portfolio.total_value:,.2f}
  Cash: ${portfolio.cash:,.2f} ({portfolio.cash/portfolio.total_value:.1%})
  Positions:
{positions_str}
  Cumulative Return: {portfolio.cumulative_return:+.1%}
  Max Drawdown: {portfolio.max_drawdown:.1%}

TOP MARKET MOVERS (1-day):
{movers_str}

AVAILABLE DATA:
  - Prices for {len(snapshot.prices)} de-identified assets
  - Returns: 1d, 5d, 21d, 63d
  - Volatility: 5d, 21d
"""
        
        # Add trade budget constraints if provided
        if trade_budget:
            context += "\n" + trade_budget.to_context_string()
        
        return context
    
    async def _phase_analyze(
        self,
        context: str,
        transcript: List[DebateMessage],
    ) -> str:
        """
        Phase 1: Analyze market conditions using LangChain AnalyzeChain.
        
        Returns the analysis summary as a string.
        """
        try:
            # Use LangChain chain for structured output
            analysis: MarketAnalysis = await self.analyze_chain.ainvoke({
                "context": context
            })
            
            # Format analysis as string for subsequent phases
            analysis_str = analysis.summary
            
            # Estimate tokens (simple word count)
            tokens = len(context.split()) + len(analysis_str.split())
            
            transcript.append(DebateMessage(
                phase="analyze",
                model=self._model_names["analyze"],
                content=analysis_str,
                tokens_used=tokens,
            ))
            
            return analysis_str
            
        except Exception as e:
            logger.error(f"Error in analyze phase: {e}")
            error_msg = f"Analysis unavailable: {e}"
            transcript.append(DebateMessage(
                phase="analyze",
                model=self._model_names["analyze"],
                content=error_msg,
                tokens_used=0,
            ))
            return error_msg
    
    async def _phase_propose(
        self,
        context: str,
        analysis: str,
        transcript: List[DebateMessage],
    ) -> str:
        """
        Phase 2: Propose specific trades using LangChain ProposeChain.
        
        Returns the proposal as a formatted string.
        """
        try:
            # Use LangChain chain for structured output
            proposal: TradeProposal = await self.propose_chain.ainvoke({
                "context": context,
                "analysis": analysis,
            })
            
            # Format proposal as string
            if proposal.action == "hold":
                proposal_str = "HOLD - No compelling trade opportunity"
            else:
                weight_pct = (proposal.target_weight or 0) * 100
                proposal_str = (
                    f"{proposal.action.upper()} {proposal.symbol} {weight_pct:.0f}% - "
                    f"{proposal.reasoning}"
                )
            
            tokens = len(context.split()) + len(analysis.split()) + len(proposal_str.split())
            
            transcript.append(DebateMessage(
                phase="propose",
                model=self._model_names["propose"],
                content=proposal_str,
                tokens_used=tokens,
            ))
            
            return proposal_str
            
        except Exception as e:
            logger.error(f"Error in propose phase: {e}")
            error_msg = "HOLD - Error in proposal generation"
            transcript.append(DebateMessage(
                phase="propose",
                model=self._model_names["propose"],
                content=error_msg,
                tokens_used=0,
            ))
            return error_msg
    
    async def _phase_decide(
        self,
        context: str,
        analysis: str,
        proposal: str,
        portfolio: BacktestPortfolio,
        transcript: List[DebateMessage],
    ) -> TradingDecision:
        """
        Phase 3: Make final decision using LangChain DecideChain.
        
        Uses structured output parsing instead of manual JSON parsing.
        """
        try:
            # Use LangChain chain for structured output
            decision_output: TradingDecisionOutput = await self.decide_chain.ainvoke({
                "context": context,
                "analysis": analysis,
                "proposal": proposal,
            })
            
            # Format decision content for transcript
            decision_content = (
                f"Action: {decision_output.action}, "
                f"Symbol: {decision_output.symbol}, "
                f"Weight: {decision_output.target_weight}, "
                f"Confidence: {decision_output.confidence:.0%}"
            )
            
            tokens = (
                len(context.split()) + len(analysis.split()) + 
                len(proposal.split()) + len(decision_content.split())
            )
            
            transcript.append(DebateMessage(
                phase="decide",
                model=self._model_names["decide"],
                content=decision_content,
                tokens_used=tokens,
            ))
            
            # Convert LangChain output to TradingDecision dataclass
            return TradingDecision(
                action=decision_output.action,
                symbol=decision_output.symbol,
                target_weight=decision_output.target_weight,
                reasoning=decision_output.reasoning,
                confidence=decision_output.confidence,
            )
            
        except Exception as e:
            logger.error(f"Error in decide phase: {e}")
            transcript.append(DebateMessage(
                phase="decide",
                model=self._model_names["decide"],
                content=f"Error: {e}",
                tokens_used=0,
            ))
            return TradingDecision(
                action="hold",
                reasoning=f"Error in decision: {e}",
                confidence=0.0,
            )
    
    async def _phase_confirm(
        self,
        context: str,
        decision: TradingDecision,
        transcript: List[DebateMessage],
    ) -> bool:
        """
        Phase 4 (optional): Claude confirms major trades via LangChain.
        
        Returns True if trade is approved, False if rejected.
        """
        try:
            # Use LangChain chain for structured output
            confirmation: RiskConfirmation = await self.confirm_chain.ainvoke({
                "context": context,
                "action": decision.action,
                "symbol": decision.symbol,
                "target_weight": decision.target_weight or 0,
                "reasoning": decision.reasoning,
                "confidence": decision.confidence,
            })
            
            # Format confirmation content for transcript
            status = "APPROVED" if confirmation.approved else "REJECTED"
            confirm_content = f"{status}: {confirmation.reason} (Risk: {confirmation.risk_score:.0%})"
            
            tokens = len(context.split()) + 50  # Estimate
            
            transcript.append(DebateMessage(
                phase="confirm",
                model=self._model_names["confirm"],
                content=confirm_content,
                tokens_used=tokens,
            ))
            
            return confirmation.approved
            
        except Exception as e:
            logger.error(f"Error in Claude confirmation: {e}")
            transcript.append(DebateMessage(
                phase="confirm",
                model=self._model_names["confirm"],
                content=f"Auto-approved (error: {e})",
                tokens_used=0,
            ))
            return True  # Default to approved on error


# Simple synchronous wrapper for testing
def run_debate_sync(
    fund_id: str,
    fund_name: str,
    fund_thesis: str,
    portfolio: BacktestPortfolio,
    snapshot: GlobalMarketSnapshot,
    simulation_date: date,
) -> TradingDecision:
    """Synchronous wrapper for run_debate."""
    runner = DailyDebateRunner()
    return asyncio.run(runner.run_debate(
        fund_id, fund_name, fund_thesis, portfolio, snapshot, simulation_date
    ))
