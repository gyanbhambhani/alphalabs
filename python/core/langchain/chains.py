"""
LangChain chains for debate orchestration.

Simplified chains using direct JSON output instead of complex Pydantic parsing.
"""

import os
import json
import logging
from typing import Dict, Any, Optional
from pathlib import Path

from core.langchain.schemas import (
    MarketAnalysis,
    TradeProposal,
    TradingDecisionOutput,
    RiskConfirmation,
)

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


def get_openai_llm(model: str = "gpt-4o-mini", temperature: float = 0.7):
    """Get OpenAI LLM instance."""
    from langchain_openai import ChatOpenAI
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not set")
    
    return ChatOpenAI(
        model=model,
        temperature=temperature,
        api_key=api_key,
        max_tokens=1000,
    )


def get_anthropic_llm(model: str = "claude-3-haiku-20240307", temperature: float = 0.7):
    """Get Anthropic Claude LLM instance."""
    from langchain_anthropic import ChatAnthropic
    
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY not set")
    
    return ChatAnthropic(
        model=model,
        temperature=temperature,
        api_key=api_key,
        max_tokens=500,
    )


def get_llm_for_model(model: str, temperature: float = 0.7):
    """Get the appropriate LLM based on model name."""
    model_lower = model.lower()
    
    if model_lower.startswith("gpt") or model_lower.startswith("o1"):
        return get_openai_llm(model, temperature)
    
    if model_lower.startswith("claude"):
        return get_anthropic_llm(model, temperature)
    
    # Default to OpenAI
    logger.warning(f"Unknown model {model}, defaulting to gpt-4o-mini")
    return get_openai_llm("gpt-4o-mini", temperature)


def extract_json(text: str) -> Optional[Dict]:
    """Extract JSON from LLM response, handling markdown code blocks."""
    # Try to find JSON in code blocks first
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
    
    # Try to find JSON object
    try:
        # Find first { and last }
        start = text.find("{")
        end = text.rfind("}") + 1
        if start >= 0 and end > start:
            return json.loads(text[start:end])
    except json.JSONDecodeError:
        pass
    
    return None


# ============================================================================
# Phase 1: ANALYZE Chain
# ============================================================================

ANALYZE_SYSTEM = """You are a market analyst for an AI hedge fund.
Analyze market conditions and identify trading opportunities.
Always respond with valid JSON only, no other text."""

ANALYZE_USER = """{context}

Analyze the current market conditions. Respond with this exact JSON format:
{{
    "opportunities": ["opportunity 1", "opportunity 2"],
    "risk_factors": ["risk 1", "risk 2"],
    "recommended_symbols": ["SYM1", "SYM2"],
    "market_regime": "bullish|bearish|neutral|volatile",
    "summary": "Brief summary of analysis"
}}

Respond with JSON only."""


class AnalyzeChain:
    """Phase 1: Market analysis."""
    
    def __init__(self, model: str = "gpt-4o-mini"):
        self.model = model
        self._llm = None
    
    def _get_llm(self):
        if self._llm is None:
            self._llm = get_llm_for_model(self.model)
        return self._llm
    
    async def ainvoke(self, inputs: Dict[str, Any]) -> MarketAnalysis:
        """Run analysis chain asynchronously."""
        from langchain_core.messages import SystemMessage, HumanMessage
        
        llm = self._get_llm()
        context = inputs.get("context", "")
        
        messages = [
            SystemMessage(content=ANALYZE_SYSTEM),
            HumanMessage(content=ANALYZE_USER.format(context=context)),
        ]
        
        response = await llm.ainvoke(messages)
        
        data = extract_json(response.content)
        if data:
            try:
                return MarketAnalysis(**data)
            except Exception as e:
                logger.warning(f"Failed to create MarketAnalysis: {e}")
        
        # Fallback
        return MarketAnalysis(
            opportunities=["Analysis parsing failed"],
            risk_factors=["Unable to parse response"],
            recommended_symbols=[],
            market_regime="unknown",
            summary=response.content[:500] if response.content else "Analysis failed"
        )


# ============================================================================
# Phase 2: PROPOSE Chain
# ============================================================================

PROPOSE_SYSTEM = """You are a portfolio manager for an AI hedge fund.
Propose specific trades based on market analysis.
Always respond with valid JSON only, no other text."""

PROPOSE_USER = """{context}

MARKET ANALYSIS:
{analysis}

Based on the analysis and fund thesis, propose ONE specific trade.
Position sizing: max 20% per position.

Respond with this exact JSON format:
{{
    "action": "buy|sell|hold",
    "symbol": "TICKER" or null,
    "target_weight": 0.10,
    "reasoning": "Why this trade",
    "risk_assessment": "Risks of this trade"
}}

Respond with JSON only."""


class ProposeChain:
    """Phase 2: Trade proposal."""
    
    def __init__(self, model: str = "gpt-4o-mini"):
        self.model = model
        self._llm = None
    
    def _get_llm(self):
        if self._llm is None:
            self._llm = get_llm_for_model(self.model)
        return self._llm
    
    async def ainvoke(self, inputs: Dict[str, Any]) -> TradeProposal:
        """Run proposal chain asynchronously."""
        from langchain_core.messages import SystemMessage, HumanMessage
        
        llm = self._get_llm()
        context = inputs.get("context", "")
        analysis = inputs.get("analysis", "")
        
        messages = [
            SystemMessage(content=PROPOSE_SYSTEM),
            HumanMessage(content=PROPOSE_USER.format(
                context=context,
                analysis=analysis
            )),
        ]
        
        response = await llm.ainvoke(messages)
        
        data = extract_json(response.content)
        if data:
            try:
                return TradeProposal(**data)
            except Exception as e:
                logger.warning(f"Failed to create TradeProposal: {e}")
        
        # Fallback
        return TradeProposal(
            action="hold",
            symbol=None,
            target_weight=None,
            reasoning="Proposal parsing failed",
            risk_assessment="Unable to assess"
        )


# ============================================================================
# Phase 3: DECIDE Chain
# ============================================================================

DECIDE_SYSTEM = """You are the final decision maker for an AI hedge fund.
Make trading decisions based on analysis and proposals.
Always respond with valid JSON only, no other text."""

DECIDE_USER = """{context}

ANALYSIS:
{analysis}

PROPOSAL:
{proposal}

Make the final trading decision.
Rules:
- Only BUY if you have cash and strong conviction
- Only SELL if you have a position to sell
- HOLD is always valid
- target_weight is the desired portfolio weight (0.0 to 0.2)

Respond with this exact JSON format:
{{
    "action": "buy|sell|hold",
    "symbol": "TICKER" or null,
    "target_weight": 0.10 or null,
    "reasoning": "Brief explanation",
    "confidence": 0.75
}}

Respond with JSON only."""


class DecideChain:
    """Phase 3: Final decision."""
    
    def __init__(self, model: str = "gpt-4o-mini"):
        self.model = model
        self._llm = None
    
    def _get_llm(self):
        if self._llm is None:
            self._llm = get_llm_for_model(self.model, temperature=0.5)
        return self._llm
    
    async def ainvoke(self, inputs: Dict[str, Any]) -> TradingDecisionOutput:
        """Run decision chain asynchronously."""
        from langchain_core.messages import SystemMessage, HumanMessage
        
        llm = self._get_llm()
        context = inputs.get("context", "")
        analysis = inputs.get("analysis", "")
        proposal = inputs.get("proposal", "")
        
        messages = [
            SystemMessage(content=DECIDE_SYSTEM),
            HumanMessage(content=DECIDE_USER.format(
                context=context,
                analysis=analysis,
                proposal=proposal
            )),
        ]
        
        response = await llm.ainvoke(messages)
        
        data = extract_json(response.content)
        if data:
            try:
                return TradingDecisionOutput(**data)
            except Exception as e:
                logger.warning(f"Failed to create TradingDecisionOutput: {e}")
        
        # Fallback
        return TradingDecisionOutput(
            action="hold",
            symbol=None,
            target_weight=None,
            reasoning="Decision parsing failed",
            confidence=0.0
        )


# ============================================================================
# Phase 4: CONFIRM Chain (Claude for major trades)
# ============================================================================

CONFIRM_SYSTEM = """You are a risk manager reviewing major trades.
Only approve trades that pass risk criteria.
Always respond with valid JSON only, no other text."""

CONFIRM_USER = """{context}

PROPOSED TRADE:
Action: {action}
Symbol: {symbol}
Target Weight: {target_weight}
Reasoning: {reasoning}
Confidence: {confidence}

This is a major trade (>5% of portfolio).
Should this trade be APPROVED or REJECTED?

Respond with this exact JSON format:
{{
    "approved": true or false,
    "reason": "Why approved or rejected",
    "risk_score": 0.5
}}

Respond with JSON only."""


class ConfirmChain:
    """Phase 4: Risk confirmation for major trades."""
    
    MAJOR_TRADE_THRESHOLD = 0.05
    
    def __init__(self, model: str = "claude-3-haiku-20240307"):
        self.model = model
        self._llm = None
    
    def _get_llm(self):
        if self._llm is None:
            try:
                self._llm = get_llm_for_model(self.model)
            except ValueError:
                logger.warning(f"Model {self.model} not available")
                return None
        return self._llm
    
    async def ainvoke(self, inputs: Dict[str, Any]) -> RiskConfirmation:
        """Run confirmation chain asynchronously."""
        from langchain_core.messages import SystemMessage, HumanMessage
        
        llm = self._get_llm()
        
        if llm is None:
            return RiskConfirmation(
                approved=True,
                reason="Auto-approved (model not available)",
                risk_score=0.5
            )
        
        messages = [
            SystemMessage(content=CONFIRM_SYSTEM),
            HumanMessage(content=CONFIRM_USER.format(
                context=inputs.get("context", ""),
                action=inputs.get("action", ""),
                symbol=inputs.get("symbol", ""),
                target_weight=inputs.get("target_weight", 0),
                reasoning=inputs.get("reasoning", ""),
                confidence=inputs.get("confidence", 0),
            )),
        ]
        
        response = await llm.ainvoke(messages)
        
        data = extract_json(response.content)
        if data:
            try:
                return RiskConfirmation(**data)
            except Exception as e:
                logger.warning(f"Failed to create RiskConfirmation: {e}")
        
        # Default to approved on parse error
        return RiskConfirmation(
            approved=True,
            reason="Parse error, defaulting to approved",
            risk_score=0.5
        )


# ============================================================================
# Debate Sequence - Orchestrates all phases
# ============================================================================

class DebateSequence:
    """Orchestrates the full 4-phase debate sequence."""
    
    def __init__(
        self,
        analyze_model: str = "gpt-4o-mini",
        propose_model: str = "gpt-4o-mini",
        decide_model: str = "gpt-4o-mini",
        confirm_model: str = "claude-3-haiku-20240307",
    ):
        self.analyze_chain = AnalyzeChain(model=analyze_model)
        self.propose_chain = ProposeChain(model=propose_model)
        self.decide_chain = DecideChain(model=decide_model)
        self.confirm_chain = ConfirmChain(model=confirm_model)
    
    async def ainvoke(
        self,
        context: str,
        require_confirmation_threshold: float = 0.05,
    ) -> Dict[str, Any]:
        """Run the full debate sequence asynchronously."""
        result = {"phases": [], "tokens_used": 0}
        
        # Phase 1: ANALYZE
        analysis = await self.analyze_chain.ainvoke({"context": context})
        result["analysis"] = analysis
        result["phases"].append({
            "phase": "analyze",
            "model": self.analyze_chain.model,
            "output": analysis.model_dump(),
        })
        
        # Phase 2: PROPOSE
        proposal = await self.propose_chain.ainvoke({
            "context": context,
            "analysis": analysis.summary,
        })
        result["proposal"] = proposal
        result["phases"].append({
            "phase": "propose",
            "model": self.propose_chain.model,
            "output": proposal.model_dump(),
        })
        
        # Phase 3: DECIDE
        decision = await self.decide_chain.ainvoke({
            "context": context,
            "analysis": analysis.summary,
            "proposal": f"{proposal.action} {proposal.symbol} - {proposal.reasoning}",
        })
        result["decision"] = decision
        result["phases"].append({
            "phase": "decide",
            "model": self.decide_chain.model,
            "output": decision.model_dump(),
        })
        
        # Phase 4: CONFIRM (if major trade)
        if (decision.action != "hold" and 
            decision.target_weight and 
            decision.target_weight > require_confirmation_threshold):
            
            confirmation = await self.confirm_chain.ainvoke({
                "context": context,
                "action": decision.action,
                "symbol": decision.symbol,
                "target_weight": decision.target_weight,
                "reasoning": decision.reasoning,
                "confidence": decision.confidence,
            })
            result["confirmation"] = confirmation
            result["phases"].append({
                "phase": "confirm",
                "model": self.confirm_chain.model,
                "output": confirmation.model_dump(),
            })
            
            if not confirmation.approved:
                result["decision"] = TradingDecisionOutput(
                    action="hold",
                    symbol=None,
                    target_weight=None,
                    reasoning=f"Rejected by risk manager: {confirmation.reason}",
                    confidence=0.3,
                )
        
        return result
