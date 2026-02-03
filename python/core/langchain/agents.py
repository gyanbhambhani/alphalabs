"""
LangChain agents for autonomous trading and research.

These agents have access to tools and can make decisions autonomously.
Uses langgraph (LangChain 1.2+) for agent orchestration.
"""

import os
import logging
from typing import Dict, Any, List, Optional, Literal
from pathlib import Path

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.prebuilt import create_react_agent

from core.langchain.schemas import EnhancedTradingDecision, ResearchAnalysis
from core.langchain.tools import MARKET_DATA_TOOLS, PORTFOLIO_TOOLS, RESEARCH_TOOLS

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


def get_llm_by_provider(
    provider: Literal["openai", "anthropic", "google"],
    model: Optional[str] = None,
    temperature: float = 0.4,
):
    """Get LLM instance by provider."""
    if provider == "openai":
        from langchain_openai import ChatOpenAI
        api_key = os.getenv("OPENAI_API_KEY")
        return ChatOpenAI(
            model=model or "gpt-4-turbo-preview",
            temperature=temperature,
            api_key=api_key,
            max_tokens=1000,
        )
    elif provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        api_key = os.getenv("ANTHROPIC_API_KEY")
        return ChatAnthropic(
            model=model or "claude-3-5-sonnet-20241022",
            temperature=temperature,
            api_key=api_key,
            max_tokens=1000,
        )
    elif provider == "google":
        from langchain_google_genai import ChatGoogleGenerativeAI
        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        return ChatGoogleGenerativeAI(
            model=model or "gemini-pro",
            temperature=temperature,
            google_api_key=api_key,
        )
    else:
        raise ValueError(f"Unknown provider: {provider}")


# ============================================================================
# Trading Agent
# ============================================================================

TRADING_SYSTEM_PROMPT = """You are {name}, an elite AI hedge fund manager.

You have access to tools for:
- Getting market snapshots and price data
- Calculating returns and volatility
- Checking portfolio state
- Searching historical market patterns

Your job is to make autonomous trading decisions based on:
1. Current market conditions
2. Portfolio state
3. Historical patterns
4. Risk management

Think deeply and write investment theses like a professional hedge fund manager.
Always explain your reasoning thoroughly.

{format_instructions}"""


class TradingAgent:
    """
    Autonomous trading agent with tool access.
    
    Can fetch market data, analyze patterns, and make trading decisions.
    Replaces direct LLM calls in llm_manager.py.
    Uses langgraph's create_react_agent.
    """
    
    def __init__(
        self,
        name: str = "Trading Agent",
        provider: Literal["openai", "anthropic", "google"] = "openai",
        model: Optional[str] = None,
        temperature: float = 0.4,
        tools: Optional[List] = None,
    ):
        self.name = name
        self.provider = provider
        self.model = model
        self.temperature = temperature
        self.tools = tools or (MARKET_DATA_TOOLS + PORTFOLIO_TOOLS)
        
        self.parser = PydanticOutputParser(pydantic_object=EnhancedTradingDecision)
        self._agent = None
    
    def _get_agent(self):
        """Lazily create the react agent."""
        if self._agent is None:
            llm = get_llm_by_provider(self.provider, self.model, self.temperature)
            
            system_prompt = TRADING_SYSTEM_PROMPT.format(
                name=self.name,
                format_instructions=self.parser.get_format_instructions(),
            )
            
            self._agent = create_react_agent(
                llm,
                self.tools,
                prompt=system_prompt,
            )
        
        return self._agent
    
    async def ainvoke(self, inputs: Dict[str, Any]) -> EnhancedTradingDecision:
        """Run trading agent asynchronously."""
        agent = self._get_agent()
        
        context = inputs.get("context", "")
        
        try:
            # Use langgraph agent
            result = await agent.ainvoke({
                "messages": [HumanMessage(content=context)]
            })
            
            # Extract last message content
            messages = result.get("messages", [])
            if messages:
                output = messages[-1].content
            else:
                output = ""
            
            return self.parser.parse(output)
            
        except Exception as e:
            logger.error(f"Trading agent error: {e}")
            # Return empty decision on error
            return EnhancedTradingDecision(
                thesis=f"Error in trading agent: {e}",
                conviction=0.0,
                market_regime="unknown",
                geopolitical_factors=[],
                trades=[],
                risks=["Agent execution failed"],
                market_outlook="Unable to assess",
            )
    
    def invoke(self, inputs: Dict[str, Any]) -> EnhancedTradingDecision:
        """Run trading agent synchronously."""
        import asyncio
        return asyncio.run(self.ainvoke(inputs))


# ============================================================================
# Research Agent
# ============================================================================

RESEARCH_SYSTEM_PROMPT = """You are a financial research analyst.

You have access to tools for:
- Getting market snapshots and price data
- Calculating returns and volatility
- Searching historical market patterns

Your job is to analyze data and provide actionable insights.
Be concise and focus on answering the user's question directly.

{format_instructions}"""


class ResearchAgent:
    """
    Research agent for streaming analyzer.
    
    Analyzes market data and provides insights.
    Can be used for AI synthesis in streaming_analyzer.py.
    Uses langgraph's create_react_agent.
    """
    
    def __init__(
        self,
        provider: Literal["openai", "anthropic", "google"] = "openai",
        model: Optional[str] = None,
        temperature: float = 0.5,
        tools: Optional[List] = None,
    ):
        self.provider = provider
        self.model = model or "gpt-4o-mini"
        self.temperature = temperature
        self.tools = tools or (MARKET_DATA_TOOLS + RESEARCH_TOOLS)
        
        self.parser = PydanticOutputParser(pydantic_object=ResearchAnalysis)
        self._agent = None
    
    def _get_agent(self):
        """Lazily create the react agent."""
        if self._agent is None:
            llm = get_llm_by_provider(self.provider, self.model, self.temperature)
            
            system_prompt = RESEARCH_SYSTEM_PROMPT.format(
                format_instructions=self.parser.get_format_instructions(),
            )
            
            self._agent = create_react_agent(
                llm,
                self.tools,
                prompt=system_prompt,
            )
        
        return self._agent
    
    async def ainvoke(
        self,
        query: str,
        symbols: List[str],
    ) -> ResearchAnalysis:
        """Run research agent asynchronously."""
        agent = self._get_agent()
        
        try:
            human_msg = f"User question: {query}\nSymbols: {', '.join(symbols)}"
            
            result = await agent.ainvoke({
                "messages": [HumanMessage(content=human_msg)]
            })
            
            # Extract last message content
            messages = result.get("messages", [])
            if messages:
                output = messages[-1].content
            else:
                output = ""
            
            return self.parser.parse(output)
            
        except Exception as e:
            logger.error(f"Research agent error: {e}")
            return ResearchAnalysis(
                interpretation=f"Analysis failed: {e}",
                key_insights=["Unable to generate insights"],
                sentiment="neutral",
            )
    
    def invoke(self, query: str, symbols: List[str]) -> ResearchAnalysis:
        """Run research agent synchronously."""
        import asyncio
        return asyncio.run(self.ainvoke(query, symbols))


# ============================================================================
# Streaming LLM for synthesis
# ============================================================================

class StreamingLLM:
    """
    Streaming LLM wrapper for AI synthesis in streaming_analyzer.
    
    Provides async streaming of LLM responses.
    """
    
    def __init__(
        self,
        provider: Literal["openai"] = "openai",
        model: str = "gpt-4o-mini",
    ):
        self.provider = provider
        self.model = model
        self._llm = None
    
    def _get_llm(self):
        """Get streaming LLM instance."""
        if self._llm is None:
            from langchain_openai import ChatOpenAI
            api_key = os.getenv("OPENAI_API_KEY")
            self._llm = ChatOpenAI(
                model=self.model,
                temperature=0.5,
                api_key=api_key,
                max_tokens=200,
                streaming=True,
            )
        return self._llm
    
    async def astream(self, prompt: str):
        """
        Stream LLM response asynchronously.
        
        Yields:
            String chunks as they are generated
        """
        llm = self._get_llm()
        
        async for chunk in llm.astream(prompt):
            if hasattr(chunk, 'content') and chunk.content:
                yield chunk.content


# Factory functions for creating agents
def create_trading_agent(
    name: str = "AI Trader",
    provider: Literal["openai", "anthropic", "google"] = "openai",
) -> TradingAgent:
    """Create a trading agent with default configuration."""
    return TradingAgent(
        name=name,
        provider=provider,
        tools=MARKET_DATA_TOOLS + PORTFOLIO_TOOLS,
    )


def create_research_agent(
    provider: Literal["openai"] = "openai",
) -> ResearchAgent:
    """Create a research agent with default configuration."""
    return ResearchAgent(
        provider=provider,
        tools=MARKET_DATA_TOOLS + RESEARCH_TOOLS,
    )
