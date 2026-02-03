"""
LangChain chains for debate orchestration.

These chains implement the 4-phase debate:
1. ANALYZE (Gemini) - Analyze market conditions
2. PROPOSE (GPT) - Propose specific trades
3. DECIDE (GPT) - Make final decision with structured output
4. CONFIRM (Claude) - Risk manager review for major trades
"""

import os
import logging
from typing import Dict, Any, Optional
from pathlib import Path

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.runnables import RunnableSequence

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
        max_tokens=500,
    )


def get_gemini_llm(model: str = "gemini-1.5-flash", temperature: float = 0.7):
    """Get Google Gemini LLM instance."""
    from langchain_google_genai import ChatGoogleGenerativeAI
    
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY or GOOGLE_API_KEY not set")
    
    return ChatGoogleGenerativeAI(
        model=model,
        temperature=temperature,
        google_api_key=api_key,
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
        max_tokens=200,
    )


# ============================================================================
# Phase 1: ANALYZE Chain (Gemini)
# ============================================================================

ANALYZE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a market analyst for an AI hedge fund.
Analyze market conditions and identify trading opportunities."""),
    ("human", """{context}

Analyze the current market conditions and identify:
1. Key opportunities based on recent price movements
2. Risk factors to consider
3. Symbols that align with the fund's thesis

Keep your analysis concise (200 words max). Focus on actionable insights.

{format_instructions}""")
])


class AnalyzeChain:
    """
    Phase 1: Market analysis using Gemini.
    
    Analyzes market conditions and identifies opportunities.
    """
    
    def __init__(self, model: str = "gemini-1.5-flash"):
        self.model = model
        self.parser = PydanticOutputParser(pydantic_object=MarketAnalysis)
        self._llm = None
    
    def _get_llm(self):
        if self._llm is None:
            try:
                self._llm = get_gemini_llm(self.model)
            except ValueError:
                # Fallback to OpenAI if Gemini not available
                logger.warning("Gemini not available, falling back to OpenAI")
                self._llm = get_openai_llm()
        return self._llm
    
    async def ainvoke(self, inputs: Dict[str, Any]) -> MarketAnalysis:
        """Run analysis chain asynchronously."""
        llm = self._get_llm()
        
        prompt = ANALYZE_PROMPT.format_messages(
            context=inputs.get("context", ""),
            format_instructions=self.parser.get_format_instructions(),
        )
        
        response = await llm.ainvoke(prompt)
        
        try:
            return self.parser.parse(response.content)
        except Exception as e:
            logger.warning(f"Failed to parse analysis: {e}")
            # Return fallback analysis
            return MarketAnalysis(
                opportunities=["Unable to parse opportunities"],
                risk_factors=["Analysis parsing failed"],
                recommended_symbols=[],
                market_regime="unknown",
                summary=response.content[:500] if response.content else "Analysis failed"
            )
    
    def invoke(self, inputs: Dict[str, Any]) -> MarketAnalysis:
        """Run analysis chain synchronously."""
        import asyncio
        return asyncio.run(self.ainvoke(inputs))


# ============================================================================
# Phase 2: PROPOSE Chain (GPT)
# ============================================================================

PROPOSE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a portfolio manager for an AI hedge fund.
Propose specific trades based on market analysis."""),
    ("human", """{context}

MARKET ANALYSIS:
{analysis}

Based on the analysis and fund thesis, propose specific trades.
Consider:
1. Position sizing (max 20% per position)
2. Current portfolio allocation
3. Risk management

{format_instructions}""")
])


class ProposeChain:
    """
    Phase 2: Trade proposal using GPT.
    
    Proposes specific trades based on analysis.
    """
    
    def __init__(self, model: str = "gpt-4o-mini"):
        self.model = model
        self.parser = PydanticOutputParser(pydantic_object=TradeProposal)
        self._llm = None
    
    def _get_llm(self):
        if self._llm is None:
            self._llm = get_openai_llm(self.model)
        return self._llm
    
    async def ainvoke(self, inputs: Dict[str, Any]) -> TradeProposal:
        """Run proposal chain asynchronously."""
        llm = self._get_llm()
        
        prompt = PROPOSE_PROMPT.format_messages(
            context=inputs.get("context", ""),
            analysis=inputs.get("analysis", ""),
            format_instructions=self.parser.get_format_instructions(),
        )
        
        response = await llm.ainvoke(prompt)
        
        try:
            return self.parser.parse(response.content)
        except Exception as e:
            logger.warning(f"Failed to parse proposal: {e}")
            return TradeProposal(
                action="hold",
                symbol=None,
                target_weight=None,
                reasoning=response.content[:500] if response.content else "Proposal failed",
                risk_assessment="Unable to assess risk due to parsing error"
            )
    
    def invoke(self, inputs: Dict[str, Any]) -> TradeProposal:
        """Run proposal chain synchronously."""
        import asyncio
        return asyncio.run(self.ainvoke(inputs))


# ============================================================================
# Phase 3: DECIDE Chain (GPT with structured output)
# ============================================================================

DECIDE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are the final decision maker for an AI hedge fund.
Make trading decisions based on analysis and proposals."""),
    ("human", """{context}

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

{format_instructions}""")
])


class DecideChain:
    """
    Phase 3: Final decision using GPT with structured output.
    
    Uses Pydantic output parsing for validated decisions.
    """
    
    def __init__(self, model: str = "gpt-4o-mini"):
        self.model = model
        self.parser = PydanticOutputParser(pydantic_object=TradingDecisionOutput)
        self._llm = None
    
    def _get_llm(self):
        if self._llm is None:
            self._llm = get_openai_llm(self.model, temperature=0.5)
        return self._llm
    
    async def ainvoke(self, inputs: Dict[str, Any]) -> TradingDecisionOutput:
        """Run decision chain asynchronously."""
        llm = self._get_llm()
        
        prompt = DECIDE_PROMPT.format_messages(
            context=inputs.get("context", ""),
            analysis=inputs.get("analysis", ""),
            proposal=inputs.get("proposal", ""),
            format_instructions=self.parser.get_format_instructions(),
        )
        
        response = await llm.ainvoke(prompt)
        
        try:
            return self.parser.parse(response.content)
        except Exception as e:
            logger.warning(f"Failed to parse decision: {e}")
            return TradingDecisionOutput(
                action="hold",
                symbol=None,
                target_weight=None,
                reasoning="Decision parsing failed",
                confidence=0.0
            )
    
    def invoke(self, inputs: Dict[str, Any]) -> TradingDecisionOutput:
        """Run decision chain synchronously."""
        import asyncio
        return asyncio.run(self.ainvoke(inputs))


# ============================================================================
# Phase 4: CONFIRM Chain (Claude for major trades)
# ============================================================================

CONFIRM_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a risk manager reviewing major trades.
Only approve trades that pass risk criteria."""),
    ("human", """{context}

PROPOSED TRADE:
Action: {action}
Symbol: {symbol}
Target Weight: {target_weight}
Reasoning: {reasoning}
Confidence: {confidence}

This is a major trade (>{threshold}% of portfolio).

Should this trade be APPROVED or REJECTED?

{format_instructions}""")
])


class ConfirmChain:
    """
    Phase 4: Risk confirmation using Claude for major trades.
    
    Reviews large positions (>5% of portfolio) before execution.
    """
    
    MAJOR_TRADE_THRESHOLD = 0.05
    
    def __init__(self, model: str = "claude-3-haiku-20240307"):
        self.model = model
        self.parser = PydanticOutputParser(pydantic_object=RiskConfirmation)
        self._llm = None
    
    def _get_llm(self):
        if self._llm is None:
            try:
                self._llm = get_anthropic_llm(self.model)
            except ValueError:
                logger.warning("Anthropic not available, auto-approving")
                return None
        return self._llm
    
    async def ainvoke(self, inputs: Dict[str, Any]) -> RiskConfirmation:
        """Run confirmation chain asynchronously."""
        llm = self._get_llm()
        
        if llm is None:
            # Auto-approve if Claude not available
            return RiskConfirmation(
                approved=True,
                reason="Auto-approved (Claude not available)",
                risk_score=0.5
            )
        
        prompt = CONFIRM_PROMPT.format_messages(
            context=inputs.get("context", ""),
            action=inputs.get("action", ""),
            symbol=inputs.get("symbol", ""),
            target_weight=inputs.get("target_weight", 0),
            reasoning=inputs.get("reasoning", ""),
            confidence=inputs.get("confidence", 0),
            threshold=self.MAJOR_TRADE_THRESHOLD * 100,
            format_instructions=self.parser.get_format_instructions(),
        )
        
        response = await llm.ainvoke(prompt)
        
        try:
            return self.parser.parse(response.content)
        except Exception as e:
            logger.warning(f"Failed to parse confirmation: {e}")
            # Default to approved on parse error
            return RiskConfirmation(
                approved=True,
                reason="Parse error, defaulting to approved",
                risk_score=0.5
            )
    
    def invoke(self, inputs: Dict[str, Any]) -> RiskConfirmation:
        """Run confirmation chain synchronously."""
        import asyncio
        return asyncio.run(self.ainvoke(inputs))


# ============================================================================
# Debate Sequence - Orchestrates all phases
# ============================================================================

class DebateSequence:
    """
    Orchestrates the full 4-phase debate sequence.
    
    Runs: ANALYZE → PROPOSE → DECIDE → (optional) CONFIRM
    """
    
    def __init__(
        self,
        analyze_model: str = "gemini-1.5-flash",
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
        """
        Run the full debate sequence asynchronously.
        
        Args:
            context: Market and portfolio context string
            require_confirmation_threshold: Weight threshold for Claude review
            
        Returns:
            Dict with analysis, proposal, decision, and optional confirmation
        """
        result = {
            "phases": [],
            "tokens_used": 0,
        }
        
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
            
            # Update decision if rejected
            if not confirmation.approved:
                result["decision"] = TradingDecisionOutput(
                    action="hold",
                    symbol=None,
                    target_weight=None,
                    reasoning=f"Rejected by risk manager: {confirmation.reason}",
                    confidence=0.3,
                )
        
        return result
    
    def invoke(
        self,
        context: str,
        require_confirmation_threshold: float = 0.05,
    ) -> Dict[str, Any]:
        """Run debate sequence synchronously."""
        import asyncio
        return asyncio.run(self.ainvoke(context, require_confirmation_threshold))
