"""
Daily Debate Runner V1 - Legacy Implementation

DEPRECATED: This is the original 3-phase debate system.
Kept for backward compatibility and comparison testing.

New implementations should use CollaborativeDebateRunner (V2.2) from debate_runner_v2.py.

3-phase debate:
1. ANALYZE (OpenAI): Analyze market conditions
2. PROPOSE (OpenAI): Propose specific trades
3. DECIDE (Claude): Make final decision with risk assessment
"""

import asyncio
from datetime import date
from typing import Dict, List, Optional, Any
import logging

from langchain_core.messages import SystemMessage, HumanMessage

# Use centralized logging
from app.logging_config import debate_log, signals_log, consensus_log

# Import shared utilities
from core.backtest.debate_common import (
    DebateMessage,
    TradingDecision,
    get_llm,
    extract_json,
)

from core.data.snapshot import GlobalMarketSnapshot
from core.backtest.portfolio_tracker import BacktestPortfolio

logger = logging.getLogger(__name__)


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
        decide_model: str = "claude-3-5-sonnet-20241022",
        temperature: float = 0.7,
        max_retries: int = 2,
    ):
        """
        Initialize the debate runner.
        
        Args:
            analyze_model: Model for analysis phase
            propose_model: Model for proposal phase
            decide_model: Model for final decision
            temperature: Sampling temperature
            max_retries: Max retries for failed API calls
        """
        self.analyze_model = analyze_model
        self.propose_model = propose_model
        self.decide_model = decide_model
        self.temperature = temperature
        self.max_retries = max_retries
        
        # LangChain LLMs
        self.analyze_llm = get_llm(analyze_model, temperature)
        self.propose_llm = get_llm(propose_model, temperature)
        self.decide_llm = get_llm(decide_model, temperature * 0.7)

    async def run_debate(
        self,
        fund_id: str,
        fund_name: str,
        fund_thesis: str,
        portfolio: BacktestPortfolio,
        snapshot: GlobalMarketSnapshot,
        simulation_date: date,
    ) -> TradingDecision:
        """
        Run the full 3-phase debate and return a decision.
        
        Returns:
            TradingDecision with action, symbol, reasoning, etc.
        """
        debate_log(f"\n{'='*80}")
        debate_log(f"DEBATE START - {fund_name} - {simulation_date}")
        debate_log(f"{'='*80}\n")
        
        messages: List[DebateMessage] = []
        total_tokens = 0
        
        # Phase 1: ANALYZE
        analyze_msg, tokens = await self._phase_analyze(
            fund_thesis, portfolio, snapshot, simulation_date
        )
        messages.append(analyze_msg)
        total_tokens += tokens
        
        # Phase 2: PROPOSE
        propose_msg, tokens = await self._phase_propose(
            fund_thesis, portfolio, snapshot, analyze_msg.content, simulation_date
        )
        messages.append(propose_msg)
        total_tokens += tokens
        
        # Phase 3: DECIDE
        decision_msg, tokens, decision_data = await self._phase_decide(
            fund_thesis, portfolio, snapshot, 
            analyze_msg.content, propose_msg.content, simulation_date
        )
        messages.append(decision_msg)
        total_tokens += tokens
        
        # Parse decision
        action = decision_data.get("action", "hold")
        symbol = decision_data.get("symbol")
        quantity = decision_data.get("quantity")
        target_weight = decision_data.get("target_weight")
        reasoning = decision_data.get("reasoning", "")
        confidence = decision_data.get("confidence", 0.5)
        signals_used = decision_data.get("signals_used", {})
        
        # Check if this requires confirmation (major trade)
        requires_confirmation = False
        if action in ["buy", "sell"] and target_weight:
            if abs(target_weight) >= self.MAJOR_TRADE_THRESHOLD:
                requires_confirmation = True
        
        decision = TradingDecision(
            action=action,
            symbol=symbol,
            quantity=quantity,
            target_weight=target_weight,
            reasoning=reasoning,
            confidence=confidence,
            signals_used=signals_used,
            debate_transcript=messages,
            models_used={
                "analyze": self.analyze_model,
                "propose": self.propose_model,
                "decide": self.decide_model,
            },
            total_tokens=total_tokens,
            requires_confirmation=requires_confirmation,
        )
        
        debate_log(f"\n{'='*80}")
        debate_log(f"DEBATE END - Decision: {action.upper()} {symbol or ''}")
        debate_log(f"Confidence: {confidence:.2f} | Tokens: {total_tokens}")
        debate_log(f"{'='*80}\n")
        
        return decision

    async def _phase_analyze(
        self,
        fund_thesis: str,
        portfolio: BacktestPortfolio,
        snapshot: GlobalMarketSnapshot,
        simulation_date: date,
    ) -> tuple[DebateMessage, int]:
        """Phase 1: Analyze market conditions."""
        debate_log("\n[PHASE 1: ANALYZE]")
        
        holdings_summary = self._format_holdings(portfolio, snapshot)
        market_summary = self._format_market_data(snapshot)
        
        system = f"""You are a senior quantitative analyst.

Your role: Analyze current market conditions and our portfolio performance.

Fund Thesis:
{fund_thesis}

Current Date: {simulation_date}"""

        user = f"""Analyze the current situation:

PORTFOLIO:
{holdings_summary}

MARKET DATA:
{market_summary}

Provide your analysis in 2-3 paragraphs covering:
1. How our current holdings align with fund thesis
2. Key market trends and signals
3. Potential opportunities or risks

Be concise and data-driven."""

        messages = [
            SystemMessage(content=system),
            HumanMessage(content=user),
        ]
        
        response = await self.analyze_llm.ainvoke(messages)
        content = response.content
        tokens = response.response_metadata.get("token_usage", {}).get("total_tokens", 0)
        
        debate_log(f"Model: {self.analyze_model}")
        debate_log(f"Output:\n{content}\n")
        
        msg = DebateMessage(
            phase="ANALYZE",
            model=self.analyze_model,
            content=content,
            tokens_used=tokens,
        )
        
        return msg, tokens

    async def _phase_propose(
        self,
        fund_thesis: str,
        portfolio: BacktestPortfolio,
        snapshot: GlobalMarketSnapshot,
        analysis: str,
        simulation_date: date,
    ) -> tuple[DebateMessage, int]:
        """Phase 2: Propose specific trades."""
        debate_log("\n[PHASE 2: PROPOSE]")
        
        holdings_summary = self._format_holdings(portfolio, snapshot)
        top_movers = self._get_top_movers(snapshot, limit=10)
        
        system = f"""You are a portfolio manager.

Your role: Propose specific trades based on the analyst's report.

Fund Thesis:
{fund_thesis}

Current Date: {simulation_date}"""

        user = f"""Based on the analysis, propose a specific trade.

ANALYST REPORT:
{analysis}

PORTFOLIO:
{holdings_summary}

TOP MOVERS TODAY:
{top_movers}

Propose ONE trade (buy/sell/hold) with clear reasoning.
Consider:
- Position sizing (suggest target weight)
- Risk/reward
- Alignment with fund thesis

Format:
Action: [buy/sell/hold]
Symbol: [ticker or N/A]
Target Weight: [decimal like 0.05 for 5%]
Reasoning: [2-3 sentences]"""

        messages = [
            SystemMessage(content=system),
            HumanMessage(content=user),
        ]
        
        response = await self.propose_llm.ainvoke(messages)
        content = response.content
        tokens = response.response_metadata.get("token_usage", {}).get("total_tokens", 0)
        
        debate_log(f"Model: {self.propose_model}")
        debate_log(f"Output:\n{content}\n")
        
        msg = DebateMessage(
            phase="PROPOSE",
            model=self.propose_model,
            content=content,
            tokens_used=tokens,
        )
        
        return msg, tokens

    async def _phase_decide(
        self,
        fund_thesis: str,
        portfolio: BacktestPortfolio,
        snapshot: GlobalMarketSnapshot,
        analysis: str,
        proposal: str,
        simulation_date: date,
    ) -> tuple[DebateMessage, int, Dict]:
        """Phase 3: Make final decision."""
        debate_log("\n[PHASE 3: DECIDE]")
        
        holdings_summary = self._format_holdings(portfolio, snapshot)
        
        system = f"""You are the fund's chief risk officer and final decision maker.

Your role: Review the analysis and proposal, then make the final trading decision.

Fund Thesis:
{fund_thesis}

Current Date: {simulation_date}"""

        user = f"""Review the debate and make a final decision.

ANALYST REPORT:
{analysis}

PORTFOLIO MANAGER PROPOSAL:
{proposal}

PORTFOLIO:
{holdings_summary}

Make your final decision. Return ONLY valid JSON:

{{
  "action": "buy" | "sell" | "hold",
  "symbol": "TICKER" or null,
  "target_weight": 0.05,
  "reasoning": "2-3 sentences explaining your decision",
  "confidence": 0.8,
  "signals_used": {{"momentum": 0.6, "volatility": 0.4}}
}}

Consider:
- Risk/reward ratio
- Portfolio diversification
- Market conditions
- Alignment with fund thesis"""

        messages = [
            SystemMessage(content=system),
            HumanMessage(content=user),
        ]
        
        response = await self.decide_llm.ainvoke(messages)
        content = response.content
        tokens = response.response_metadata.get("token_usage", {}).get("total_tokens", 0)
        
        debate_log(f"Model: {self.decide_model}")
        debate_log(f"Output:\n{content}\n")
        
        # Parse JSON decision
        decision_data = extract_json(content) or {}
        
        msg = DebateMessage(
            phase="DECIDE",
            model=self.decide_model,
            content=content,
            tokens_used=tokens,
        )
        
        return msg, tokens, decision_data

    def _format_holdings(
        self, 
        portfolio: BacktestPortfolio, 
        snapshot: GlobalMarketSnapshot
    ) -> str:
        """Format current holdings for display."""
        if not portfolio.positions:
            return "No positions (100% cash)"
        
        lines = []
        for symbol, shares in portfolio.positions.items():
            price = snapshot.get_price(symbol)
            if price:
                value = shares * price
                weight = value / portfolio.cash if portfolio.cash > 0 else 0
                lines.append(f"  {symbol}: {shares:.2f} shares @ ${price:.2f} = ${value:.2f} ({weight:.1%})")
        
        total_value = portfolio.cash + sum(
            portfolio.positions.get(s, 0) * snapshot.get_price(s)
            for s in portfolio.positions
            if snapshot.get_price(s)
        )
        
        lines.append(f"\nCash: ${portfolio.cash:.2f}")
        lines.append(f"Total Value: ${total_value:.2f}")
        
        return "\n".join(lines)

    def _format_market_data(self, snapshot: GlobalMarketSnapshot) -> str:
        """Format market overview."""
        symbols = list(snapshot.stock_data.keys())[:5]
        lines = []
        
        for symbol in symbols:
            price = snapshot.get_price(symbol)
            ret_1d = snapshot.get_return(symbol, "1d")
            vol_5d = snapshot.get_volatility(symbol, "5d")
            
            if price and ret_1d is not None:
                lines.append(
                    f"  {symbol}: ${price:.2f} | "
                    f"1d: {ret_1d:+.2%} | "
                    f"Vol: {vol_5d:.2%}" if vol_5d else ""
                )
        
        return "\n".join(lines) if lines else "No market data available"

    def _get_top_movers(self, snapshot: GlobalMarketSnapshot, limit: int = 10) -> str:
        """Get top gainers/losers."""
        movers = []
        
        for symbol in snapshot.stock_data.keys():
            ret_1d = snapshot.get_return(symbol, "1d")
            price = snapshot.get_price(symbol)
            if ret_1d is not None and price:
                movers.append((symbol, ret_1d, price))
        
        movers.sort(key=lambda x: abs(x[1]), reverse=True)
        
        lines = []
        for symbol, ret, price in movers[:limit]:
            lines.append(f"  {symbol}: {ret:+.2%} (${price:.2f})")
        
        return "\n".join(lines) if lines else "No movers data"


def create_daily_debate_runner(**kwargs) -> DailyDebateRunner:
    """
    Factory function for V1 debate runner.
    
    Args:
        **kwargs: Arguments to pass to DailyDebateRunner constructor
        
    Returns:
        DailyDebateRunner instance
    """
    return DailyDebateRunner(**kwargs)
