"""
Daily Debate Runner for Backtesting.

Uses LangChain with OpenAI and Claude for multi-phase debates.
"""

import os
import json
import asyncio
from datetime import date, datetime
from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass, field
import logging
from pathlib import Path

from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import SystemMessage, HumanMessage

# Use centralized logging
from app.logging_config import debate_log, signals_log, consensus_log

logger = logging.getLogger(__name__)

# Load environment variables
try:
    from dotenv import load_dotenv
    current = Path(__file__).resolve()
    for parent in [current] + list(current.parents):
        env_local = parent / ".env.local"
        env_file = parent / ".env"
        if env_local.exists():
            load_dotenv(env_local)
            break
        if env_file.exists():
            load_dotenv(env_file)
            break
except ImportError:
    pass

from core.data.snapshot import GlobalMarketSnapshot
from core.backtest.portfolio_tracker import BacktestPortfolio


@dataclass
class DebateMessage:
    """A single message in the debate."""
    phase: str
    model: str
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


def get_llm(model: str, temperature: float = 0.7):
    """
    Get the appropriate LangChain LLM based on model name.
    
    Supports:
    - gpt-* models -> OpenAI
    - claude-* models -> Anthropic
    """
    if model.startswith("gpt-"):
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not set")
        return ChatOpenAI(
            model=model,
            temperature=temperature,
            api_key=api_key,
        )
    elif model.startswith("claude-"):
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY not set")
        return ChatAnthropic(
            model=model,
            temperature=temperature,
            api_key=api_key,
        )
    else:
        # Default to OpenAI
        api_key = os.getenv("OPENAI_API_KEY")
        return ChatOpenAI(
            model="gpt-4o-mini",
            temperature=temperature,
            api_key=api_key,
        )


def extract_json(text: str) -> Optional[Dict]:
    """Extract JSON from LLM response, handling markdown code blocks."""
    if not text:
        return None
    
    # Try to find JSON in code blocks
    if "```json" in text:
        start = text.find("```json") + 7
        end = text.find("```", start)
        if end > start:
            text = text[start:end].strip()
    elif "```" in text:
        start = text.find("```") + 3
        end = text.find("```", start)
        if end > start:
            text = text[start:end].strip()
    
    # Find JSON object
    try:
        start = text.find("{")
        end = text.rfind("}") + 1
        if start >= 0 and end > start:
            return json.loads(text[start:end])
    except json.JSONDecodeError as e:
        logger.warning(f"JSON parse error: {e}")
    
    return None


class DailyDebateRunner:
    """
    Runs the daily AI debate for a fund using LangChain.
    
    3-phase debate:
    1. ANALYZE (OpenAI): Analyze market conditions
    2. PROPOSE (OpenAI): Propose specific trades
    3. DECIDE (Claude): Make final decision with risk assessment
    """
    
    MAJOR_TRADE_THRESHOLD = 0.05
    
    def __init__(
        self,
        analyze_model: str = "gpt-4o-mini",
        propose_model: str = "gpt-4o-mini",
        decide_model: str = "claude-3-haiku-20240307",
    ):
        """
        Initialize debate runner with model configuration.
        
        Args:
            analyze_model: Model for market analysis (default: gpt-4o-mini)
            propose_model: Model for trade proposals (default: gpt-4o-mini)
            decide_model: Model for final decision (default: claude-3-haiku)
        """
        self._model_names = {
            "analyze": analyze_model,
            "propose": propose_model,
            "decide": decide_model,
        }
    
    async def run_debate(
        self,
        fund_id: str,
        fund_name: str,
        fund_thesis: str,
        portfolio: BacktestPortfolio,
        snapshot: GlobalMarketSnapshot,
        simulation_date: date,
        trade_budget: Optional[Any] = None,
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
            trade_budget: Optional budget constraints
            
        Returns:
            TradingDecision with action, reasoning, and transcript
        """
        transcript: List[DebateMessage] = []
        models_used: Dict[str, str] = {}
        total_tokens = 0
        
        # Build context for prompts
        context = self._build_context(
            fund_name, fund_thesis, portfolio, snapshot, 
            simulation_date, trade_budget
        )
        
        try:
            # Phase 1: ANALYZE
            print(f"[{fund_id}] Starting debate - Phase 1: ANALYZE with {self._model_names['analyze']}")
            logger.info(f"[{fund_id}] Phase 1: ANALYZE with {self._model_names['analyze']}")
            analysis = await self._phase_analyze(context, fund_thesis, transcript)
            models_used["analyze"] = self._model_names["analyze"]
            if transcript:
                total_tokens += transcript[-1].tokens_used
            
            # Phase 2: PROPOSE
            print(f"[{fund_id}] Phase 2: PROPOSE with {self._model_names['propose']}")
            logger.info(f"[{fund_id}] Phase 2: PROPOSE with {self._model_names['propose']}")
            proposal = await self._phase_propose(
                context, fund_thesis, analysis, portfolio, transcript
            )
            models_used["propose"] = self._model_names["propose"]
            if transcript:
                total_tokens += transcript[-1].tokens_used
            
            # Phase 3: DECIDE (Claude)
            print(f"[{fund_id}] Phase 3: DECIDE with {self._model_names['decide']}")
            logger.info(f"[{fund_id}] Phase 3: DECIDE with {self._model_names['decide']}")
            decision = await self._phase_decide(
                context, fund_thesis, analysis, proposal, portfolio, transcript
            )
            models_used["decide"] = self._model_names["decide"]
            if transcript:
                total_tokens += transcript[-1].tokens_used
            
            # Budget validation
            if trade_budget and decision.action == "buy":
                if not trade_budget.can_buy():
                    logger.warning(f"[{fund_id}] Buy blocked by budget")
                    decision = TradingDecision(
                        action="hold",
                        reasoning="Budget limit reached - holding",
                        confidence=0.1,
                    )
            
            decision.debate_transcript = transcript
            decision.models_used = models_used
            decision.total_tokens = total_tokens
            
            return decision
            
        except Exception as e:
            print(f"[{fund_id}] DEBATE ERROR: {e}")
            logger.error(f"[{fund_id}] Debate error: {e}")
            return TradingDecision(
                action="hold",
                reasoning=f"Debate error: {str(e)[:100]}",
                confidence=0.0,
                debate_transcript=transcript,
                models_used=models_used,
            )
    
    def _build_context(
        self,
        fund_name: str,
        fund_thesis: str,
        portfolio: BacktestPortfolio,
        snapshot: GlobalMarketSnapshot,
        simulation_date: date,
        trade_budget: Optional[Any] = None,
    ) -> str:
        """Build context string for prompts."""
        # Portfolio summary
        positions_str = ""
        for sym, pos in portfolio.positions.items():
            weight = portfolio.get_position_weight(sym)
            positions_str += (
                f"  - {sym}: {pos.quantity:.0f} shares @ "
                f"${pos.current_price:.2f} "
                f"({weight:.1%} of portfolio, "
                f"{pos.unrealized_return:+.1%} unrealized)\n"
            )
        
        if not positions_str:
            positions_str = "  (No positions - 100% cash)\n"
        
        # Top movers with available data
        top_movers = []
        for sym in list(snapshot.prices.keys())[:30]:
            ret_1d = snapshot.get_return(sym, "1d")
            if ret_1d is not None:
                top_movers.append((sym, ret_1d))
        
        top_movers.sort(key=lambda x: abs(x[1]), reverse=True)
        movers_str = "\n".join([
            f"  - {sym}: {ret:+.1%} (1d)"
            for sym, ret in top_movers[:10]
        ])
        
        # Budget info
        budget_str = ""
        if trade_budget:
            budget_str = f"\nTRADE BUDGET: {trade_budget.to_context_string()}"
        
        return f"""DATE: {simulation_date}
FUND: {fund_name}
THESIS: {fund_thesis}

PORTFOLIO:
  Total Value: ${portfolio.total_value:,.2f}
  Cash: ${portfolio.cash:,.2f} ({portfolio.cash/portfolio.total_value:.1%})
  Positions:
{positions_str}
  Cumulative Return: {portfolio.cumulative_return:+.1%}
  Max Drawdown: {portfolio.max_drawdown:.1%}

TOP MOVERS TODAY:
{movers_str}

UNIVERSE: {len(snapshot.prices)} stocks available
{budget_str}"""
    
    async def _phase_analyze(
        self,
        context: str,
        fund_thesis: str,
        transcript: List[DebateMessage],
    ) -> str:
        """Phase 1: Analyze market conditions."""
        llm = get_llm(self._model_names["analyze"], temperature=0.5)
        
        system = f"""You are a market analyst for a fund with thesis: {fund_thesis}

Analyze the market data and identify opportunities or risks.
Be concise - 2-3 sentences max.
Focus on what matters for this fund's strategy."""

        user = f"""{context}

Provide a brief market analysis."""

        try:
            print(f"  -> Calling {self._model_names['analyze']} for analysis...")
            response = await llm.ainvoke([
                SystemMessage(content=system),
                HumanMessage(content=user),
            ])
            print(f"  -> Got response from {self._model_names['analyze']}")
            
            content = response.content if hasattr(response, 'content') else str(response)
            tokens = 0
            if hasattr(response, 'usage_metadata') and response.usage_metadata:
                tokens = response.usage_metadata.get('total_tokens', 0)
            
            transcript.append(DebateMessage(
                phase="analyze",
                model=self._model_names["analyze"],
                content=content[:500],
                tokens_used=tokens,
            ))
            
            return content
            
        except Exception as e:
            print(f"  -> ANALYZE ERROR: {e}")
            logger.error(f"Analyze phase failed: {e}")
            transcript.append(DebateMessage(
                phase="analyze",
                model=self._model_names["analyze"],
                content=f"Analysis error: {str(e)[:100]}",
                tokens_used=0,
            ))
            return "Analysis unavailable due to error."
    
    async def _phase_propose(
        self,
        context: str,
        fund_thesis: str,
        analysis: str,
        portfolio: BacktestPortfolio,
        transcript: List[DebateMessage],
    ) -> str:
        """Phase 2: Propose specific trades."""
        llm = get_llm(self._model_names["propose"], temperature=0.7)
        
        # Determine what actions are possible
        can_buy = portfolio.cash > portfolio.total_value * 0.05
        can_sell = len(portfolio.positions) > 0
        
        system = f"""You are a trade proposer for a fund with thesis: {fund_thesis}

Based on the analysis, propose a specific trade or recommend holding.
{"You have cash available for buying." if can_buy else "Limited cash - consider selling or holding."}
{"You have positions that could be sold." if can_sell else "No positions to sell."}

IMPORTANT RULES:
- This is a LONG-TERM fund. Do NOT panic sell on small daily moves.
- Only sell if a position has lost >15% OR fundamentally broken thesis.
- Small daily dips (-1% to -5%) are NORMAL. Hold through volatility.
- Prefer HOLD over sell unless there's a strong reason.

Be specific: name ONE stock and ONE action (buy/sell/hold).
Keep it brief - 1-2 sentences."""

        user = f"""{context}

ANALYSIS: {analysis}

What trade do you propose?"""

        try:
            print(f"  -> Calling {self._model_names['propose']} for proposal...")
            response = await llm.ainvoke([
                SystemMessage(content=system),
                HumanMessage(content=user),
            ])
            print(f"  -> Got response from {self._model_names['propose']}")
            
            content = response.content if hasattr(response, 'content') else str(response)
            tokens = 0
            if hasattr(response, 'usage_metadata') and response.usage_metadata:
                tokens = response.usage_metadata.get('total_tokens', 0)
            
            transcript.append(DebateMessage(
                phase="propose",
                model=self._model_names["propose"],
                content=content[:500],
                tokens_used=tokens,
            ))
            
            return content
            
        except Exception as e:
            print(f"  -> PROPOSE ERROR: {e}")
            logger.error(f"Propose phase failed: {e}")
            transcript.append(DebateMessage(
                phase="propose",
                model=self._model_names["propose"],
                content=f"Proposal error: {str(e)[:100]}",
                tokens_used=0,
            ))
            return "Hold - unable to generate proposal."
    
    async def _phase_decide(
        self,
        context: str,
        fund_thesis: str,
        analysis: str,
        proposal: str,
        portfolio: BacktestPortfolio,
        transcript: List[DebateMessage],
    ) -> TradingDecision:
        """Phase 3: Make final decision (Claude)."""
        llm = get_llm(self._model_names["decide"], temperature=0.3)
        
        # Build position info with holding periods
        position_info = []
        for sym, pos in portfolio.positions.items():
            unrealized = pos.unrealized_return
            position_info.append(f"{sym}: {unrealized:+.1%} unrealized")
        
        positions_str = ", ".join(position_info) if position_info else "None"
        
        system = f"""You are the final decision maker for a fund with thesis: {fund_thesis}

Review the analysis and proposal, then make a final trading decision.
You must respond with ONLY valid JSON in this exact format:
{{
    "action": "buy" or "sell" or "hold",
    "symbol": "TICKER" or null,
    "target_weight": 0.10 or null,
    "reasoning": "Brief explanation",
    "confidence": 0.75
}}

CRITICAL RULES - READ CAREFULLY:
1. DO NOT PANIC SELL. Small daily losses (-1% to -8%) are NORMAL market noise.
2. Only sell if: loss exceeds -15%, OR thesis is fundamentally broken.
3. Give positions TIME to work - at least 5-10 trading days.
4. HOLD is often the best action. Trading too much destroys returns via fees.
5. If you just bought something, DO NOT sell it the next day!

Rules:
- action: must be "buy", "sell", or "hold"
- symbol: required for buy/sell, null for hold
- target_weight: 0.05 to 0.20 for buys, null for sell/hold
- confidence: 0.0 to 1.0

Current positions: {positions_str}
Cash available: ${portfolio.cash:,.2f}"""

        user = f"""{context}

ANALYSIS: {analysis}

PROPOSAL: {proposal}

Make your final decision. Respond with JSON only."""

        try:
            print(f"  -> Calling {self._model_names['decide']} for decision...")
            response = await llm.ainvoke([
                SystemMessage(content=system),
                HumanMessage(content=user),
            ])
            print(f"  -> Got response from {self._model_names['decide']}")
            
            content = response.content if hasattr(response, 'content') else str(response)
            tokens = 0
            if hasattr(response, 'usage_metadata') and response.usage_metadata:
                tokens = response.usage_metadata.get('total_tokens', 0)
            
            transcript.append(DebateMessage(
                phase="decide",
                model=self._model_names["decide"],
                content=content[:500],
                tokens_used=tokens,
            ))
            
            # Parse the JSON response
            data = extract_json(content)
            print(f"  -> Parsed decision: {data}")
            
            if data:
                return TradingDecision(
                    action=data.get("action", "hold"),
                    symbol=data.get("symbol"),
                    target_weight=data.get("target_weight"),
                    reasoning=data.get("reasoning", ""),
                    confidence=data.get("confidence", 0.5),
                )
            else:
                logger.warning("Failed to parse decision JSON, defaulting to hold")
                return TradingDecision(
                    action="hold",
                    reasoning="Could not parse decision - holding",
                    confidence=0.0,
                )
                
        except Exception as e:
            print(f"  -> DECIDE ERROR: {e}")
            logger.error(f"Decide phase failed: {e}")
            transcript.append(DebateMessage(
                phase="decide",
                model=self._model_names["decide"],
                content=f"Decision error: {str(e)[:100]}",
                tokens_used=0,
            ))
            return TradingDecision(
                action="hold",
                reasoning=f"Decision error: {str(e)[:50]}",
                confidence=0.0,
            )


# Synchronous wrapper for testing
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


# =============================================================================
# Collaborative Debate Runner V2.1
# =============================================================================

from core.collaboration.debate_v2 import (
    AVAILABLE_FEATURES,
    THESIS_REQUIRED_EVIDENCE,
    ThesisType,
    InvalidationRule,
    EvidenceReference,
    ThesisProposal,
    ConfidenceDecomposition,
    DialogMove,
    Counterfactual,
    AgentTurn,
    ConsensusBoard,
    RiskManagerDecision,
    validate_evidence,
    update_consensus_board,
    compute_conservative_weight,
)
from core.ai.experience_memory import ExperienceStore, generate_accountability_brief


# =============================================================================
# Feature Value Lookup (Canonical - One Function)
# =============================================================================

def get_feature_value(
    snapshot: GlobalMarketSnapshot,
    symbol: str,
    feature: str
) -> Optional[float]:
    """
    Canonical feature lookup - one function, no fragile if-elses.
    
    Args:
        snapshot: Market data snapshot
        symbol: Stock symbol
        feature: Feature name (e.g., "return_21d", "volatility_5d", "price")
    
    Returns:
        Feature value or None if not available
    """
    if feature == "price":
        return snapshot.get_price(symbol)
    
    if feature.startswith("return_"):
        # return_21d -> period "21d"
        period = feature[7:]  # Strip "return_"
        return snapshot.get_return(symbol, period)
    
    if feature.startswith("volatility_"):
        # volatility_21d -> period "21d"
        period = feature[11:]  # Strip "volatility_"
        return snapshot.get_volatility(symbol, period)
    
    return None


# =============================================================================
# Decision Audit Trail (Full Traceability)
# =============================================================================

@dataclass
class DecisionAuditTrail:
    """
    Full audit trail for a trading decision.
    
    Contains:
    - signals_used: Actual feature values extracted
    - evidence_used: Full citation details with agent attribution
    - validation_report: Pass/fail with detailed errors
    """
    signals_used: Dict[str, float]  # {symbol_feature: value}
    evidence_used: List[Dict[str, Any]]  # [{symbol, feature, agent, value, ...}]
    validation_report: Dict[str, Any]  # {passed, errors, missing_required, ...}


@dataclass
class ScreenedCandidate:
    """A stock candidate that passed screening with its signals."""
    symbol: str
    score: float  # Composite score for ranking
    signals: Dict[str, float]  # All available signals
    thesis_fit: str  # Which thesis type it fits best
    reasoning: str  # Why it was selected


def screen_universe_for_strategy(
    snapshot: GlobalMarketSnapshot,
    strategy: str,
    portfolio: BacktestPortfolio,
    top_k: int = 5,
) -> List[ScreenedCandidate]:
    """
    Screen the entire universe and return top candidates for a strategy.
    
    This is DATA-FIRST: we look at actual numbers, not LLM guesses.
    
    Args:
        snapshot: Market data with all symbols
        strategy: Fund strategy (momentum, mean_reversion, volatility)
        portfolio: Current portfolio (to exclude existing positions)
        top_k: Number of candidates to return
        
    Returns:
        List of ScreenedCandidate sorted by score (best first)
    """
    candidates: List[ScreenedCandidate] = []
    
    # Get all symbols with data
    symbols = list(snapshot.prices.keys())
    
    # Determine which signals to use based on strategy
    strategy_lower = strategy.lower()
    
    if "momentum" in strategy_lower or "trend" in strategy_lower:
        # Momentum: look for strong positive returns over 21d and 63d
        thesis_fit = "momentum"
        for symbol in symbols:
            ret_21d = snapshot.get_return(symbol, "21d")
            ret_63d = snapshot.get_return(symbol, "63d")
            ret_5d = snapshot.get_return(symbol, "5d")
            vol_21d = snapshot.get_volatility(symbol, "21d")
            price = snapshot.get_price(symbol)
            
            if ret_21d is None or ret_63d is None or price is None:
                continue
            
            # Skip if already held
            if symbol in portfolio.positions:
                continue
            
            # Momentum score: positive returns, skip recent month effect
            # Higher 63d return + positive 21d = strong momentum
            score = (ret_63d * 0.6) + (ret_21d * 0.4)
            
            # Penalize high volatility slightly
            if vol_21d and vol_21d > 0.4:
                score *= 0.8
            
            if score > 0.02:  # Only consider positive momentum
                candidates.append(ScreenedCandidate(
                    symbol=symbol,
                    score=score,
                    signals={
                        "return_21d": ret_21d,
                        "return_63d": ret_63d,
                        "return_5d": ret_5d or 0,
                        "volatility_21d": vol_21d or 0,
                        "price": price,
                    },
                    thesis_fit=thesis_fit,
                    reasoning=(
                        f"Strong momentum: 63d={ret_63d:+.1%}, 21d={ret_21d:+.1%}"
                    ),
                ))
    
    elif "mean_reversion" in strategy_lower or "oversold" in strategy_lower:
        # Mean reversion: look for oversold stocks (negative short-term returns)
        thesis_fit = "mean_reversion"
        for symbol in symbols:
            ret_1d = snapshot.get_return(symbol, "1d")
            ret_5d = snapshot.get_return(symbol, "5d")
            ret_21d = snapshot.get_return(symbol, "21d")
            vol_21d = snapshot.get_volatility(symbol, "21d")
            price = snapshot.get_price(symbol)
            
            if ret_1d is None or ret_5d is None or price is None:
                continue
            
            # Skip if already held
            if symbol in portfolio.positions:
                continue
            
            # Mean reversion score: negative short-term returns = oversold
            # More negative = higher score (potential bounce)
            score = -(ret_1d * 0.4 + ret_5d * 0.6)
            
            # Only consider if actually oversold
            if ret_5d < -0.02 or ret_1d < -0.01:
                # Bonus if longer-term is positive (healthy stock, temporary dip)
                if ret_21d and ret_21d > 0:
                    score *= 1.2
                
                candidates.append(ScreenedCandidate(
                    symbol=symbol,
                    score=score,
                    signals={
                        "return_1d": ret_1d,
                        "return_5d": ret_5d,
                        "return_21d": ret_21d or 0,
                        "volatility_21d": vol_21d or 0,
                        "price": price,
                    },
                    thesis_fit=thesis_fit,
                    reasoning=(
                        f"Oversold bounce: 1d={ret_1d:+.1%}, 5d={ret_5d:+.1%}"
                    ),
                ))
    
    elif "volatility" in strategy_lower:
        # Volatility: look for vol spikes or regime changes
        thesis_fit = "volatility"
        for symbol in symbols:
            vol_5d = snapshot.get_volatility(symbol, "5d")
            vol_21d = snapshot.get_volatility(symbol, "21d")
            ret_5d = snapshot.get_return(symbol, "5d")
            price = snapshot.get_price(symbol)
            
            if vol_5d is None or vol_21d is None or price is None:
                continue
            
            # Skip if already held
            if symbol in portfolio.positions:
                continue
            
            # Vol spike score: short-term vol much higher than longer-term
            vol_ratio = vol_5d / vol_21d if vol_21d > 0 else 1.0
            score = vol_ratio - 1.0  # How much vol has spiked
            
            if vol_ratio > 1.3:  # Significant vol spike
                candidates.append(ScreenedCandidate(
                    symbol=symbol,
                    score=score,
                    signals={
                        "volatility_5d": vol_5d,
                        "volatility_21d": vol_21d,
                        "return_5d": ret_5d or 0,
                        "price": price,
                    },
                    thesis_fit=thesis_fit,
                    reasoning=(
                        f"Vol spike: 5d={vol_5d:.1%} vs 21d={vol_21d:.1%} "
                        f"(ratio={vol_ratio:.1f}x)"
                    ),
                ))
    
    else:
        # Unknown strategy - return empty
        signals_log(
            f"Unknown strategy '{strategy}' - no screening available",
            level=logging.WARNING
        )
        return []
    
    # Sort by score (highest first) and return top_k
    candidates.sort(key=lambda c: c.score, reverse=True)
    
    signals_log(
        f"Screened {len(symbols)} stocks, found {len(candidates)} candidates "
        f"for {thesis_fit}"
    )
    if candidates:
        top_3 = candidates[:3]
        signals_log(
            f"Top 3: {', '.join(f'{c.symbol}({c.score:.2f})' for c in top_3)}"
        )
    
    return candidates[:top_k]


class CollaborativeDebateRunner:
    """
    Investment committee debate system V2.2 - DATA-FIRST.
    
    Key changes from V2.1:
    - SCREEN FIRST: Scan entire universe, rank by actual signals
    - PRESENT CANDIDATES: Both agents see the SAME top candidates
    - DEBATE ON DATA: Agents discuss specific stocks with real numbers
    - NO GUESSING: Agents pick from pre-screened list only
    
    Debate flow:
    1. Screen: Rank all stocks by strategy-specific signals
    2. Present: Show top 5 candidates to both agents with full data
    3. Debate: Agents discuss and converge on ONE candidate
    4. Execute: Trade the agreed stock with validated signals
    """
    
    MAX_ROUNDS = 3
    MIN_CONFIDENCE_COMPONENT = 0.5
    MIN_EDGE_THRESHOLD = 0.6
    
    def __init__(
        self,
        analyst_model: str = "gpt-4o-mini",
        critic_model: str = "claude-3-haiku-20240307",
        experience_db_path: str = "data/experience.db",
    ):
        """
        Initialize collaborative debate runner.
        
        Args:
            analyst_model: Model for analyst agent
            critic_model: Model for critic agent
            experience_db_path: Path to experience database
        """
        self.analyst_model = analyst_model
        self.critic_model = critic_model
        self.experience_store = ExperienceStore(experience_db_path)
    
    def _has_sufficient_data(self, snapshot: GlobalMarketSnapshot) -> bool:
        """
        Check if snapshot has sufficient data for V2 debate.
        
        V2 requires return_21d or return_5d for at least some symbols.
        Early simulation periods won't have this data.
        """
        # Check if we have at least 3 symbols with 21d returns
        symbols_with_21d = 0
        symbols_with_5d = 0
        
        for symbol in snapshot.coverage_symbols[:20]:  # Check first 20
            if snapshot.get_return(symbol, "21d") is not None:
                symbols_with_21d += 1
            if snapshot.get_return(symbol, "5d") is not None:
                symbols_with_5d += 1
        
        # Need at least 3 symbols with medium-term data
        return symbols_with_21d >= 3 or symbols_with_5d >= 5
    
    async def run_debate(
        self,
        fund_id: str,
        fund_name: str,
        fund_thesis: str,
        portfolio: BacktestPortfolio,
        snapshot: GlobalMarketSnapshot,
        simulation_date: date,
        trade_budget: Optional[Any] = None,
    ) -> TradingDecision:
        """
        Run the collaborative debate for a fund.
        
        Args:
            fund_id: Fund identifier
            fund_name: Human-readable fund name
            fund_thesis: Fund's investment thesis
            portfolio: Current portfolio state
            snapshot: Market data snapshot
            simulation_date: Current simulation date
            trade_budget: Optional budget constraints
            
        Returns:
            TradingDecision with action and reasoning
        """
        transcript: List[DebateMessage] = []
        conversation: List[AgentTurn] = []
        
        debate_log(f"Fund {fund_id} starting debate for {simulation_date}")
        debate_log(
            f"Portfolio: ${portfolio.total_value:,.0f}, "
            f"Cash: ${portfolio.cash:,.0f} ({portfolio.cash/portfolio.total_value:.0%}), "
            f"Positions: {len(portfolio.positions)}"
        )
        
        try:
            # 0. Check data availability - V2 requires return data
            if not self._has_sufficient_data(snapshot):
                debate_log(
                    f"Insufficient data for V2 debate on {simulation_date}, "
                    f"defaulting to HOLD",
                    level=logging.WARNING
                )
                return self._create_hold_decision(
                    "Insufficient historical data for analysis (early simulation period)",
                    transcript
                )
            
            # =========================================================
            # PHASE 1: DATA-FIRST SCREENING
            # Scan entire universe, rank by strategy-specific signals
            # =========================================================
            debate_log(f"Phase 1: Screening universe for {fund_thesis[:50]}...")
            
            candidates = screen_universe_for_strategy(
                snapshot=snapshot,
                strategy=fund_thesis,
                portfolio=portfolio,
                top_k=5,
            )
            
            if not candidates:
                debate_log(
                    "No candidates found for strategy, defaulting to HOLD",
                    level=logging.WARNING
                )
                return self._create_hold_decision(
                    f"No stocks in universe match {fund_thesis[:30]} criteria today",
                    transcript
                )
            
            # Log the candidates we're presenting
            debate_log(f"Found {len(candidates)} candidates:")
            for i, c in enumerate(candidates, 1):
                debate_log(f"  #{i} {c.symbol}: {c.reasoning} (score={c.score:.3f})")
            
            # =========================================================
            # PHASE 2: PRE-DEBATE SETUP
            # =========================================================
            accountability_brief = generate_accountability_brief(
                fund_id, simulation_date, self.experience_store
            )
            
            similar_episodes = self.experience_store.retrieve_similar(
                snapshot, fund_id=fund_id, k=3
            )
            
            # Build context WITH screened candidates
            context = self._build_enhanced_context_with_candidates(
                fund_name, fund_thesis, portfolio, snapshot,
                simulation_date, accountability_brief, similar_episodes,
                candidates,  # NEW: pass candidates
            )
            
            # Store candidates for validation later
            valid_symbols = {c.symbol for c in candidates}
            candidate_signals = {c.symbol: c.signals for c in candidates}
            
            # =========================================================
            # PHASE 3: COMMITTEE DEBATE
            # Agents discuss THE SAME candidates, must pick from list
            # =========================================================
            debate_log(f"Phase 2: Committee debate on {len(candidates)} candidates...")
            consensus_board = ConsensusBoard()
            
            for round_num in range(self.MAX_ROUNDS):
                debate_log(f"Round {round_num + 1}/{self.MAX_ROUNDS} - Analyst proposing...")
                
                # Get analyst turn
                analyst_turn, analyst_errors = await self._get_agent_turn_with_repair(
                    agent_id="analyst",
                    model=self.analyst_model,
                    role="analyst",
                    round_num=round_num,
                    context=context,
                    conversation=conversation,
                    fund_thesis=fund_thesis,
                    snapshot=snapshot,
                    transcript=transcript,
                )
                
                if analyst_turn:
                    debate_log(
                        f"Analyst: {analyst_turn.action.upper()} "
                        f"{analyst_turn.symbol or 'N/A'} @ "
                        f"{analyst_turn.suggested_weight:.0%} weight, "
                        f"thesis={analyst_turn.thesis.thesis_type.value}, "
                        f"confidence={analyst_turn.confidence.overall():.0%}"
                    )
                
                debate_log(f"Round {round_num + 1}/{self.MAX_ROUNDS} - Critic responding...")
                
                # Get critic turn
                critic_turn, critic_errors = await self._get_agent_turn_with_repair(
                    agent_id="critic",
                    model=self.critic_model,
                    role="critic",
                    round_num=round_num,
                    context=context,
                    conversation=conversation,
                    fund_thesis=fund_thesis,
                    snapshot=snapshot,
                    transcript=transcript,
                )
                
                if critic_turn:
                    debate_log(
                        f"Critic: {critic_turn.action.upper()} "
                        f"{critic_turn.symbol or 'N/A'} @ "
                        f"{critic_turn.suggested_weight:.0%} weight, "
                        f"thesis={critic_turn.thesis.thesis_type.value}, "
                        f"confidence={critic_turn.confidence.overall():.0%}"
                    )
                
                # If either failed, default to HOLD
                if analyst_turn is None or critic_turn is None:
                    debate_log(
                        f"Agent failed validation, defaulting to HOLD. "
                        f"Errors: {analyst_errors + critic_errors}",
                        level=logging.WARNING
                    )
                    return self._create_hold_decision(
                        f"Validation failed: {analyst_errors + critic_errors}",
                        transcript
                    )
                
                conversation.extend([analyst_turn, critic_turn])
                
                # Update consensus board with fund strategy and valid symbols
                consensus_board = update_consensus_board(
                    consensus_board, analyst_turn, critic_turn, snapshot,
                    fund_strategy=fund_thesis,
                    valid_symbols=valid_symbols,  # V2.2: pass screened candidates
                )
                
                # Log gate status
                gate = consensus_board.gate
                consensus_log(f"Computing gate for {fund_id}...")
                consensus_log(
                    f"action_match={gate.action_match}, "
                    f"symbol_match={gate.symbol_match}, "
                    f"symbol_in_candidates={gate.symbol_in_candidates}, "
                    f"thesis_type_match={gate.thesis_type_match}"
                )
                consensus_log(
                    f"fund_strategy_match={gate.fund_strategy_match} "
                    f"({fund_thesis.split()[0]} -> {analyst_turn.thesis.thesis_type.value})"
                )
                consensus_log(
                    f"min_confidence_met={gate.min_confidence_met} "
                    f"({min(analyst_turn.confidence.min_component(), critic_turn.confidence.min_component()):.2f} >= 0.5)"
                )
                consensus_log(
                    f"min_edge_met={gate.min_edge_met} "
                    f"({min(analyst_turn.confidence.signal_strength, critic_turn.confidence.signal_strength):.2f} >= 0.6)"
                )
                
                # Check if consensus reached
                if consensus_board.can_execute():
                    consensus_log(
                        f"Gate PASSED - all {len(gate.passed_gates())} checks OK",
                        level=logging.INFO
                    )
                    break
                else:
                    consensus_log(
                        f"Gate FAILED - {len(gate.failed_gates())} checks failed: "
                        f"{gate.failed_gates()}",
                        level=logging.WARNING
                    )
            
            # 4. Finalize decision
            if consensus_board.can_execute():
                decision = self._build_consensus_decision(
                    consensus_board, analyst_turn, critic_turn,
                    portfolio, transcript, snapshot, fund_id,
                    valid_symbols, candidate_signals,  # Pass screened candidates
                )
                debate_log(
                    f"Final decision: {decision.action.upper()} "
                    f"{decision.symbol or 'N/A'} @ {decision.target_weight or 0:.0%}, "
                    f"confidence={decision.confidence:.0%}"
                )
            else:
                consensus_log(
                    f"No consensus after {self.MAX_ROUNDS} rounds",
                    level=logging.WARNING
                )
                if consensus_board.gate.rejection_reasons:
                    consensus_log("Rejection reasons:")
                    for reason in consensus_board.gate.rejection_reasons:
                        consensus_log(f"  - {reason}")
                decision = self._create_hold_decision(
                    f"No consensus: {consensus_board.gate.failed_gates()}",
                    transcript
                )
            
            # 5. Record experience
            if analyst_turn and critic_turn:
                self.experience_store.record(
                    snapshot=snapshot,
                    fund_id=fund_id,
                    action=decision.action,
                    symbol=decision.symbol,
                    thesis_type=analyst_turn.thesis.thesis_type,
                    confidence=decision.confidence,
                    conversation=conversation,
                )
            
            return decision
            
        except Exception as e:
            logger.error(f"[{fund_id}] Debate error: {e}")
            return self._create_hold_decision(
                f"Debate error: {str(e)[:100]}",
                transcript
            )
    
    async def _get_agent_turn_with_repair(
        self,
        agent_id: str,
        model: str,
        role: str,
        round_num: int,
        context: str,
        conversation: List[AgentTurn],
        fund_thesis: str,
        snapshot: GlobalMarketSnapshot,
        transcript: List[DebateMessage],
        max_repairs: int = 1,
    ) -> tuple[Optional[AgentTurn], List[str]]:
        """
        Get agent turn with one repair attempt on failure.
        
        Args:
            agent_id: Agent identifier
            model: LLM model to use
            role: "analyst" or "critic"
            round_num: Current round number
            context: Market context string
            conversation: Previous conversation
            fund_thesis: Fund's thesis
            snapshot: Market snapshot
            transcript: Debate transcript
            max_repairs: Maximum repair attempts
            
        Returns:
            Tuple of (AgentTurn or None, list of errors)
        """
        errors: List[str] = []
        prompt = self._build_agent_prompt(
            role, round_num, context, conversation, fund_thesis
        )
        
        for attempt in range(max_repairs + 1):
            try:
                llm = get_llm(model, temperature=0.5)
                
                response = await llm.ainvoke([
                    SystemMessage(content=self._get_agent_system_prompt(
                        role, fund_thesis
                    )),
                    HumanMessage(content=prompt),
                ])
                
                content = (
                    response.content 
                    if hasattr(response, 'content') 
                    else str(response)
                )
                
                # Record in transcript - keep full content for V2 parsing
                transcript.append(DebateMessage(
                    phase=f"round_{round_num}_{role}",
                    model=model,
                    content=content,  # Full content for frontend parsing
                    tokens_used=0,
                ))
                
                # Parse response
                data = extract_json(content)
                if not data:
                    errors.append(f"Attempt {attempt + 1}: Failed to parse JSON")
                    if attempt < max_repairs:
                        prompt = self._build_repair_prompt(errors[-1])
                    continue
                
                # Create AgentTurn
                turn = self._parse_agent_turn(data, agent_id, round_num)
                
                # Validate evidence
                valid, evidence_errors = validate_evidence(
                    turn.evidence_cited,
                    turn.thesis.thesis_type,
                    snapshot
                )
                
                if not valid:
                    errors.extend(evidence_errors)
                    if attempt < max_repairs:
                        prompt = self._build_evidence_repair_prompt(
                            evidence_errors, turn.thesis.thesis_type
                        )
                    continue
                
                return turn, []
                
            except Exception as e:
                errors.append(f"Attempt {attempt + 1}: {str(e)[:100]}")
                if attempt < max_repairs:
                    prompt = self._build_repair_prompt(errors[-1])
        
        return None, errors
    
    def _build_enhanced_context(
        self,
        fund_name: str,
        fund_thesis: str,
        portfolio: BacktestPortfolio,
        snapshot: GlobalMarketSnapshot,
        simulation_date: date,
        accountability_brief: str,
        similar_episodes: Dict[str, List],
    ) -> str:
        """Build enhanced context with accountability and precedents."""
        # Portfolio summary
        positions_str = ""
        for sym, pos in portfolio.positions.items():
            weight = portfolio.get_position_weight(sym)
            positions_str += (
                f"  - {sym}: {pos.quantity:.0f} shares @ "
                f"${pos.current_price:.2f} "
                f"({weight:.1%}, {pos.unrealized_return:+.1%})\n"
            )
        
        if not positions_str:
            positions_str = "  (No positions - 100% cash)\n"
        
        # Market data for top symbols - show all available features
        market_data = []
        available_in_snapshot = set()
        
        for sym in list(snapshot.prices.keys())[:20]:
            parts = [sym]
            
            # Check each feature
            ret_1d = snapshot.get_return(sym, "1d")
            ret_5d = snapshot.get_return(sym, "5d")
            ret_21d = snapshot.get_return(sym, "21d")
            ret_63d = snapshot.get_return(sym, "63d")
            vol_5d = snapshot.get_volatility(sym, "5d")
            vol_21d = snapshot.get_volatility(sym, "21d")
            
            if ret_1d is not None:
                parts.append(f"1d={ret_1d:+.1%}")
                available_in_snapshot.add("return_1d")
            if ret_5d is not None:
                parts.append(f"5d={ret_5d:+.1%}")
                available_in_snapshot.add("return_5d")
            if ret_21d is not None:
                parts.append(f"21d={ret_21d:+.1%}")
                available_in_snapshot.add("return_21d")
            if ret_63d is not None:
                parts.append(f"63d={ret_63d:+.1%}")
                available_in_snapshot.add("return_63d")
            if vol_5d is not None:
                parts.append(f"vol5d={vol_5d:.1%}")
                available_in_snapshot.add("volatility_5d")
            if vol_21d is not None:
                parts.append(f"vol21d={vol_21d:.1%}")
                available_in_snapshot.add("volatility_21d")
            
            if len(parts) > 1:  # Has at least one feature
                market_data.append(": ".join([parts[0], ", ".join(parts[1:])]))
        
        market_str = "\n".join(f"  {d}" for d in market_data[:10])
        
        # Show which features are actually available
        features_available = sorted(available_in_snapshot) if available_in_snapshot else ["price"]
        
        # Similar episodes summary
        episodes_str = ""
        for ep in similar_episodes.get("similar", [])[:2]:
            episodes_str += (
                f"  - {ep.record_date}: {ep.action} {ep.symbol or ''} "
                f"({ep.thesis_type}) -> {ep.outcome_5d or 0:+.1%}\n"
            )
        
        if not episodes_str:
            episodes_str = "  (No similar episodes found)\n"
        
        return f"""DATE: {simulation_date}
FUND: {fund_name}
THESIS: {fund_thesis}

{accountability_brief}

PORTFOLIO:
  Total Value: ${portfolio.total_value:,.2f}
  Cash: ${portfolio.cash:,.2f} ({portfolio.cash/portfolio.total_value:.1%})
{positions_str}

FEATURES AVAILABLE TODAY: {', '.join(features_available)}
(Only cite features from this list - others have no data yet)

MARKET DATA:
{market_str}

SIMILAR PAST EPISODES:
{episodes_str}

AVAILABLE THESIS TYPES: {', '.join(t.value for t in ThesisType)}"""
    
    def _build_enhanced_context_with_candidates(
        self,
        fund_name: str,
        fund_thesis: str,
        portfolio: BacktestPortfolio,
        snapshot: GlobalMarketSnapshot,
        simulation_date: date,
        accountability_brief: str,
        similar_episodes: Dict[str, List],
        candidates: List[ScreenedCandidate],
    ) -> str:
        """
        Build context with PRE-SCREENED candidates.
        
        This is the key change: agents see ONLY the candidates that passed
        data-driven screening. They must pick from this list.
        """
        # Portfolio summary
        positions_str = ""
        for sym, pos in portfolio.positions.items():
            weight = portfolio.get_position_weight(sym)
            positions_str += (
                f"  - {sym}: {pos.quantity:.0f} shares @ "
                f"${pos.current_price:.2f} "
                f"({weight:.1%}, {pos.unrealized_return:+.1%})\n"
            )
        
        if not positions_str:
            positions_str = "  (No positions - 100% cash)\n"
        
        # Build candidate details - this is what agents will debate
        candidates_str = ""
        for i, c in enumerate(candidates, 1):
            signals_str = ", ".join(
                f"{k}={v:+.1%}" if "return" in k or "volatility" in k else f"{k}=${v:.2f}"
                for k, v in c.signals.items()
                if v is not None
            )
            candidates_str += (
                f"  #{i} {c.symbol} (score={c.score:.2f})\n"
                f"      Signals: {signals_str}\n"
                f"      Fit: {c.thesis_fit} - {c.reasoning}\n"
            )
        
        # Similar episodes summary
        episodes_str = ""
        for ep in similar_episodes.get("similar", [])[:2]:
            episodes_str += (
                f"  - {ep.record_date}: {ep.action} {ep.symbol or ''} "
                f"({ep.thesis_type}) -> {ep.outcome_5d or 0:+.1%}\n"
            )
        
        if not episodes_str:
            episodes_str = "  (No similar episodes found)\n"
        
        # Get the thesis type from candidates
        thesis_type = candidates[0].thesis_fit if candidates else "unknown"
        valid_symbols = [c.symbol for c in candidates]
        
        return f"""DATE: {simulation_date}
FUND: {fund_name}
THESIS: {fund_thesis}

{accountability_brief}

PORTFOLIO:
  Total Value: ${portfolio.total_value:,.2f}
  Cash: ${portfolio.cash:,.2f} ({portfolio.cash/portfolio.total_value:.1%})
{positions_str}

============================================================
PRE-SCREENED CANDIDATES (ranked by {thesis_type} signals)
============================================================
These stocks passed quantitative screening. You MUST pick from this list.
Do NOT suggest stocks outside this list - they failed screening.

{candidates_str}
VALID SYMBOLS: {', '.join(valid_symbols)}

============================================================

SIMILAR PAST EPISODES:
{episodes_str}

RULES:
1. You MUST select a symbol from the VALID SYMBOLS list above
2. If you disagree with all candidates, recommend HOLD
3. Cite the actual signal values shown above in your evidence
4. Thesis type must be: {thesis_type}

AVAILABLE THESIS TYPES: {', '.join(t.value for t in ThesisType)}"""
    
    def _get_agent_system_prompt(self, role: str, fund_thesis: str) -> str:
        """Get system prompt for agent role."""
        base = f"""You are the {role.upper()} in an investment committee for a fund.
Fund thesis: {fund_thesis}

You must respond with ONLY valid JSON in this exact format:
{{
    "dialog_move": {{
        "acknowledge": "Restate colleague's key claim",
        "challenge": "One specific disagreement or null",
        "request": "Ask for missing evidence or null",
        "concede_or_hold": "What you updated vs stayed firm on"
    }},
    "action": "buy" or "sell" or "hold",
    "symbol": "TICKER" or null,
    "suggested_weight": 0.10,
    "risk_posture": "normal" or "defensive" or "aggressive",
    "thesis": {{
        "thesis_type": "momentum" or "mean_reversion" or "volatility",
        "horizon_days": 5,
        "primary_signal": "return_1d",
        "secondary_signal": "volatility_21d" or null,
        "invalidation_rules": [
            {{"feature": "return_1d", "symbol": "AAPL", "operator": ">", "value": 0.05}}
        ]
    }},
    "confidence": {{
        "signal_strength": 0.7,
        "regime_fit": 0.6,
        "risk_comfort": 0.55,
        "execution_feasibility": 0.8
    }},
    "evidence_cited": [
        {{"feature": "return_1d", "symbol": "AAPL"}},
        {{"feature": "volatility_21d", "symbol": "AAPL"}}
    ],
    "counterfactual": {{
        "alternative_action": "hold",
        "why_rejected": "Brief reason"
    }}
}}

CRITICAL RULES:
1. SYMBOL SELECTION: You MUST pick from the "VALID SYMBOLS" list in the context
   - These stocks passed quantitative screening for this strategy
   - Do NOT suggest any symbol outside that list - they failed screening
   - If you disagree with ALL candidates, recommend HOLD with symbol=null
2. Use the ACTUAL signal values shown in the PRE-SCREENED CANDIDATES section
   - Copy the exact numbers (return_1d, return_5d, etc.) from the candidate data
   - Do not make up signal values
3. Match thesis_type to the fund's strategy:
   - momentum: needs return_21d or return_63d
   - mean_reversion: needs return_1d or return_5d
   - volatility: needs volatility_5d or volatility_21d
4. All confidence components should be 0.0-1.0
5. HORIZON ALIGNMENT: Use standard horizons:
   - Short-term: 5 days (for mean_reversion)
   - Medium-term: 21 days (for momentum)
6. If colleague proposed a symbol, DISCUSS THAT SAME SYMBOL
   - Agree or disagree, but focus the debate on the same stock
   - Don't randomly switch to a different symbol"""
        
        if role == "analyst":
            return base + """

As ANALYST, focus on:
- Pick the BEST candidate from the pre-screened list
- Cite the specific signal values shown for that candidate
- Explain why this candidate fits the fund's thesis"""
        else:
            return base + """

As CRITIC, focus on:
- RESPOND TO THE ANALYST'S PICK - discuss the SAME symbol
- Challenge: Is this really the best candidate from the list?
- If you agree, confirm. If you disagree, explain why another candidate is better
- Do NOT pick a random different symbol - stay focused on the debate"""
    
    def _build_agent_prompt(
        self,
        role: str,
        round_num: int,
        context: str,
        conversation: List[AgentTurn],
        fund_thesis: str,
    ) -> str:
        """Build prompt for agent turn."""
        # Format conversation history
        history = ""
        for turn in conversation:
            history += f"\n{turn.agent_id.upper()} (Round {turn.round_num}):\n"
            history += f"  Action: {turn.action} {turn.symbol or ''}\n"
            history += f"  Thesis: {turn.thesis.thesis_type.value}\n"
            history += f"  Confidence: {turn.confidence.overall():.0%}\n"
        
        if not history:
            history = "(This is the first round - no prior conversation)"
        
        return f"""{context}

CONVERSATION SO FAR:
{history}

YOUR TASK (Round {round_num + 1}):
1. If there's prior conversation, ACKNOWLEDGE your colleague's position
2. CHALLENGE one specific point if you disagree (or state "none")
3. REQUEST missing evidence if needed (or state "none needed")
4. State what you CONCEDE vs HOLD firm on
5. Provide your proposal with thesis, evidence, and confidence

Respond with JSON only."""
    
    def _build_repair_prompt(self, error: str) -> str:
        """Build repair prompt after validation failure."""
        return f"""Your previous response had an error:
{error}

Please fix and respond again with valid JSON.
Remember:
- evidence_cited must use ONLY: {', '.join(sorted(AVAILABLE_FEATURES))}
- You need at least 2 evidence items
- All fields are required"""
    
    def _build_evidence_repair_prompt(
        self,
        errors: List[str],
        thesis_type: ThesisType,
    ) -> str:
        """Build repair prompt for evidence validation failure."""
        required = THESIS_REQUIRED_EVIDENCE.get(thesis_type, set())
        return f"""Your evidence citations had errors:
{chr(10).join(errors)}

Available features: {', '.join(sorted(AVAILABLE_FEATURES))}
Required for {thesis_type.value}: {', '.join(required)}

Please fix and respond again with valid JSON."""
    
    def _parse_agent_turn(
        self,
        data: Dict[str, Any],
        agent_id: str,
        round_num: int,
    ) -> AgentTurn:
        """Parse agent turn from JSON data."""
        # Parse dialog move
        dm_data = data.get("dialog_move", {})
        dialog_move = DialogMove(
            acknowledge=dm_data.get("acknowledge", ""),
            challenge=dm_data.get("challenge"),
            request=dm_data.get("request"),
            concede_or_hold=dm_data.get("concede_or_hold", ""),
        )
        
        # Parse thesis
        thesis_data = data.get("thesis", {})
        invalidation_rules = [
            InvalidationRule(
                feature=r.get("feature", ""),
                symbol=r.get("symbol", ""),
                operator=r.get("operator", ">"),
                value=float(r.get("value", 0)),
            )
            for r in thesis_data.get("invalidation_rules", [])
        ]
        
        thesis = ThesisProposal(
            thesis_type=ThesisType(thesis_data.get("thesis_type", "momentum")),
            horizon_days=int(thesis_data.get("horizon_days", 5)),
            primary_signal=thesis_data.get("primary_signal", "return_1d"),
            secondary_signal=thesis_data.get("secondary_signal"),
            invalidation_rules=invalidation_rules,
        )
        
        # Parse confidence
        conf_data = data.get("confidence", {})
        confidence = ConfidenceDecomposition(
            signal_strength=float(conf_data.get("signal_strength", 0.5)),
            regime_fit=float(conf_data.get("regime_fit", 0.5)),
            risk_comfort=float(conf_data.get("risk_comfort", 0.5)),
            execution_feasibility=float(conf_data.get("execution_feasibility", 0.5)),
        )
        
        # Parse evidence
        evidence = [
            EvidenceReference(
                feature=e.get("feature", ""),
                symbol=e.get("symbol", ""),
            )
            for e in data.get("evidence_cited", [])
        ]
        
        # Parse counterfactual
        cf_data = data.get("counterfactual", {})
        counterfactual = Counterfactual(
            alternative_action=cf_data.get("alternative_action", "hold"),
            why_rejected=cf_data.get("why_rejected", ""),
        )
        
        return AgentTurn(
            agent_id=agent_id,
            round_num=round_num,
            dialog_move=dialog_move,
            action=data.get("action", "hold"),
            symbol=data.get("symbol"),
            suggested_weight=float(data.get("suggested_weight", 0.0)),
            risk_posture=data.get("risk_posture", "normal"),
            thesis=thesis,
            confidence=confidence,
            evidence_cited=evidence,
            counterfactual=counterfactual,
        )
    
    def _build_consensus_decision(
        self,
        consensus_board: ConsensusBoard,
        analyst_turn: AgentTurn,
        critic_turn: AgentTurn,
        portfolio: BacktestPortfolio,
        transcript: List[DebateMessage],
        snapshot: GlobalMarketSnapshot,
        fund_id: str,
        valid_symbols: Set[str],
        candidate_signals: Dict[str, Dict[str, float]],
    ) -> TradingDecision:
        """
        Build final decision from consensus.
        
        Includes signal extraction and validation blocking.
        Now also validates symbol is from screened candidates.
        """
        action = consensus_board.agreed_action or "hold"
        symbol = consensus_board.agreed_symbol
        
        # GUARDRAIL 0: Symbol must be from screened candidates
        if action != "hold" and symbol:
            if symbol not in valid_symbols:
                debate_log(
                    f"BLOCKING trade - {symbol} not in screened candidates: "
                    f"{valid_symbols}",
                    level=logging.WARNING
                )
                return self._create_hold_decision(
                    f"Trade blocked: {symbol} not in pre-screened candidates",
                    transcript
                )
            
            # Use pre-computed signals from screening (more reliable)
            if symbol in candidate_signals:
                signals_log(
                    f"Using pre-screened signals for {symbol}: "
                    f"{candidate_signals[symbol]}"
                )
        
        # Extract signals from evidence with thesis-specific validation
        thesis_type = (
            consensus_board.agreed_thesis_type or 
            analyst_turn.thesis.thesis_type
        )
        
        # Prefer candidate signals over LLM-cited evidence
        if symbol and symbol in candidate_signals:
            # Build audit from screened signals (more reliable)
            screened_signals = candidate_signals[symbol]
            audit = DecisionAuditTrail(
                signals_used={
                    f"{symbol}_{k}": v 
                    for k, v in screened_signals.items() 
                    if v is not None
                },
                evidence_used=[
                    {"symbol": symbol, "feature": k, "value": v, "source": "screening"}
                    for k, v in screened_signals.items()
                    if v is not None
                ],
                validation_report={
                    "passed": True,
                    "errors": [],
                    "missing_required": [],
                    "source": "pre_screened_candidates",
                },
            )
            signals_log(
                f"Using {len(audit.signals_used)} pre-screened signals for {symbol}"
            )
        else:
            audit = self._extract_signals_from_evidence(
                analyst_turn, critic_turn, snapshot, thesis_type
            )
        
        # GUARDRAIL: Block trades without validated signals
        if action != "hold":
            # Block 1: No signals at all
            if not audit.signals_used:
                debate_log(
                    "BLOCKING trade - no validated signals attached",
                    level=logging.WARNING
                )
                return self._create_hold_decision(
                    "Trade blocked: no validated signals attached",
                    transcript
                )
            
            # Block 2: Required features for thesis missing
            if not audit.validation_report["passed"]:
                debate_log(
                    f"BLOCKING trade - validation failed: "
                    f"{audit.validation_report['errors']}",
                    level=logging.WARNING
                )
                return self._create_hold_decision(
                    f"Trade blocked: {audit.validation_report['errors']}",
                    transcript
                )
        
        # Use conservative weight
        suggested_weight = compute_conservative_weight(analyst_turn, critic_turn)
        
        # Apply risk manager
        risk_decision = RiskManagerDecision.compute(
            action=action,
            symbol=consensus_board.agreed_symbol,
            suggested_weight=suggested_weight,
            risk_posture=analyst_turn.risk_posture,
            portfolio_cash=portfolio.cash,
            portfolio_total_value=portfolio.total_value,
            current_position_weight=portfolio.get_position_weight(
                consensus_board.agreed_symbol
            ) or 0.0,
        )
        
        if not risk_decision.approved:
            return self._create_hold_decision(
                f"Risk manager rejected: {risk_decision.constraints_hit}",
                transcript
            )
        
        # Build reasoning
        reasoning = (
            f"Consensus: {action} "
            f"{consensus_board.agreed_symbol or ''} "
            f"({thesis_type.value if thesis_type else 'unknown'}) "
            f"| Weight: {risk_decision.final_weight:.1%}"
        )
        
        if risk_decision.constraints_hit:
            reasoning += f" | Constraints: {', '.join(risk_decision.constraints_hit)}"
        
        return TradingDecision(
            action=action,
            symbol=consensus_board.agreed_symbol,
            target_weight=risk_decision.final_weight,
            reasoning=reasoning,
            confidence=min(
                analyst_turn.confidence.overall(),
                critic_turn.confidence.overall()
            ),
            signals_used=audit.signals_used,  # Attach signals to decision
            debate_transcript=transcript,
            models_used={
                "analyst": self.analyst_model,
                "critic": self.critic_model,
            },
        )
    
    def _extract_signals_from_evidence(
        self,
        analyst_turn: AgentTurn,
        critic_turn: AgentTurn,
        snapshot: GlobalMarketSnapshot,
        thesis_type: ThesisType,
    ) -> DecisionAuditTrail:
        """
        Extract actual feature values with thesis-specific validation.
        
        Args:
            analyst_turn: Analyst's turn
            critic_turn: Critic's turn
            snapshot: Market data snapshot
            thesis_type: The agreed thesis type
            
        Returns:
            DecisionAuditTrail with signals, evidence, and validation report
        """
        signals: Dict[str, float] = {}
        evidence: List[Dict[str, Any]] = []
        errors: List[str] = []
        
        # Get symbol being traded
        symbol = analyst_turn.symbol or critic_turn.symbol or "?"
        signals_log(f"Extracting signals for {symbol}...")
        
        for agent_id, turn in [("analyst", analyst_turn), ("critic", critic_turn)]:
            for ref in turn.evidence_cited:
                key = f"{ref.symbol}_{ref.feature}"
                value = get_feature_value(snapshot, ref.symbol, ref.feature)
                
                evidence.append({
                    "symbol": ref.symbol,
                    "feature": ref.feature,
                    "agent": agent_id,
                    "value": value,
                })
                
                if value is None:
                    errors.append(f"{agent_id} cited {key} but value is None")
                elif key not in signals:
                    signals[key] = value
        
        # Log found signals
        if signals:
            signal_strs = [f"{k}={v:.4f}" for k, v in list(signals.items())[:5]]
            signals_log(f"Found: {', '.join(signal_strs)}")
        else:
            signals_log("No signals found!", level=logging.WARNING)
        
        # Thesis-specific validation: check required features are present
        required = THESIS_REQUIRED_EVIDENCE.get(thesis_type, set())
        cited_features = {
            ref.feature for ref in 
            analyst_turn.evidence_cited + critic_turn.evidence_cited
        }
        missing_required = required - cited_features
        
        signals_log(f"Thesis {thesis_type.value} requires: {required}")
        
        if missing_required:
            errors.append(
                f"Thesis {thesis_type.value} requires {required}, "
                f"missing: {missing_required}"
            )
        
        # Validation passes if no errors AND we have enough signals
        passed = len(errors) == 0 and len(signals) >= len(required)
        
        if passed:
            signals_log(
                f"Validation PASSED - {len(signals)} signals, "
                f"{len(missing_required)} missing"
            )
        else:
            signals_log(
                f"Validation FAILED - {len(signals)} signals, "
                f"errors: {errors}",
                level=logging.WARNING
            )
        
        return DecisionAuditTrail(
            signals_used=signals,
            evidence_used=evidence,
            validation_report={
                "passed": passed,
                "errors": errors,
                "signal_count": len(signals),
                "evidence_count": len(evidence),
                "required_features": list(required),
                "missing_required": list(missing_required),
            }
        )
    
    def _create_hold_decision(
        self,
        reason: str,
        transcript: List[DebateMessage],
    ) -> TradingDecision:
        """Create a HOLD decision."""
        return TradingDecision(
            action="hold",
            reasoning=reason,
            confidence=0.0,
            debate_transcript=transcript,
            models_used={
                "analyst": self.analyst_model,
                "critic": self.critic_model,
            },
        )


# Factory function to get the appropriate runner
def get_debate_runner(
    version: str = "v2",
    **kwargs,
) -> DailyDebateRunner | CollaborativeDebateRunner:
    """
    Get debate runner by version.
    
    Args:
        version: "v1" for DailyDebateRunner, "v2" for CollaborativeDebateRunner
        **kwargs: Arguments to pass to runner constructor
        
    Returns:
        Debate runner instance
    """
    if version == "v1":
        return DailyDebateRunner(**kwargs)
    else:
        return CollaborativeDebateRunner(**kwargs)
