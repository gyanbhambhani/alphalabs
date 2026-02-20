"""
Shared utilities for debate runners.

Common dataclasses, LLM initialization, and parsing functions used by all debate systems.
"""

import os
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from pathlib import Path

from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic

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
