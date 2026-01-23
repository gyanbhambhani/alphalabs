"""
LLM Portfolio Manager

Uses LLMs (GPT-4, Claude, Gemini) for autonomous trading decisions.
Full autonomy to interpret signals and make independent decisions.
"""
import json
from typing import List, Dict, Optional, Literal
from datetime import datetime
from core.managers.base import (
    BaseManager, TradingDecision, ManagerContext,
    Action, RiskLimits
)


class LLMManager(BaseManager):
    """
    LLM-powered portfolio manager with full autonomy.
    
    Can interpret signals, understand context, and make nuanced decisions
    that pure algorithms cannot.
    """
    
    def __init__(
        self,
        manager_id: str,
        name: str,
        provider: Literal["openai", "anthropic", "google"],
        initial_capital: float = 25000.0,
        risk_limits: Optional[RiskLimits] = None,
        model: Optional[str] = None,
        temperature: float = 0.7
    ):
        super().__init__(
            manager_id=manager_id,
            name=name,
            manager_type="llm",
            initial_capital=initial_capital,
            risk_limits=risk_limits
        )
        self.provider = provider
        self.model = model or self._get_default_model(provider)
        self.temperature = temperature
        self._client = None
    
    def _get_default_model(self, provider: str) -> str:
        """Get default model for provider"""
        models = {
            "openai": "gpt-4-turbo-preview",
            "anthropic": "claude-3-sonnet-20240229",
            "google": "gemini-pro"
        }
        return models.get(provider, "gpt-4")
    
    async def _get_client(self):
        """Lazily initialize the LLM client"""
        if self._client is not None:
            return self._client
        
        if self.provider == "openai":
            from openai import AsyncOpenAI
            from app.config import get_settings
            settings = get_settings()
            self._client = AsyncOpenAI(api_key=settings.openai_api_key)
        
        elif self.provider == "anthropic":
            from anthropic import AsyncAnthropic
            from app.config import get_settings
            settings = get_settings()
            self._client = AsyncAnthropic(api_key=settings.anthropic_api_key)
        
        elif self.provider == "google":
            import google.generativeai as genai
            from app.config import get_settings
            settings = get_settings()
            genai.configure(api_key=settings.google_api_key)
            self._client = genai.GenerativeModel(self.model)
        
        return self._client
    
    def _build_prompt(self, context: ManagerContext) -> str:
        """Build the prompt for the LLM"""
        
        # Format portfolio
        portfolio_str = f"""
Portfolio Value: ${context.portfolio.total_value:,.2f}
Cash Balance: ${context.portfolio.cash_balance:,.2f}
Invested: {context.portfolio.invested_pct:.1%}

Current Positions:
"""
        if context.portfolio.positions:
            for symbol, pos in context.portfolio.positions.items():
                portfolio_str += (
                    f"  {symbol}: {pos.quantity} shares @ ${pos.avg_entry_price:.2f} "
                    f"(current: ${pos.current_price:.2f}, "
                    f"P&L: ${pos.unrealized_pnl:,.2f})\n"
                )
        else:
            portfolio_str += "  No open positions\n"
        
        # Format signals
        momentum_str = "\n".join([
            f"  {sym}: {score:+.2f}"
            for sym, score in sorted(
                context.signals.momentum.items(), 
                key=lambda x: x[1], 
                reverse=True
            )[:10]
        ])
        
        mean_rev_str = "\n".join([
            f"  {sym}: {score:+.2f}"
            for sym, score in sorted(
                context.signals.mean_reversion.items(),
                key=lambda x: x[1],
                reverse=True
            )[:10]
        ])
        
        ml_str = "\n".join([
            f"  {sym}: {ret:+.2%} predicted"
            for sym, ret in sorted(
                context.signals.ml_prediction.items(),
                key=lambda x: x[1],
                reverse=True
            )[:10]
        ])
        
        # Format semantic search
        semantic = context.signals.semantic_search
        semantic_str = f"""
Average 5-day return in similar periods: {semantic.get('avg_5d_return', 0):.2%}
Average 20-day return in similar periods: {semantic.get('avg_20d_return', 0):.2%}
Positive outcome rate (5-day): {semantic.get('positive_5d_rate', 0):.0%}
Interpretation: {semantic.get('interpretation', 'N/A')}
"""
        
        prompt = f"""You are {self.name}, an autonomous AI portfolio manager.

## Current Time
{context.timestamp.strftime('%Y-%m-%d %H:%M:%S')} UTC

## Your Portfolio
{portfolio_str}

## Market Regime
{context.signals.volatility_regime}

## Strategy Signals (Your Toolbox)

### Momentum Signals (12-month return, skip 1 month)
{momentum_str}

### Mean Reversion Signals (Bollinger Z-score)
{mean_rev_str}

### ML Model Predictions (5-day forward return)
{ml_str}

### Semantic Market Memory (Similar Historical Periods)
{semantic_str}

## Your Task
Analyze the signals and decide what trades to make. You have FULL AUTONOMY:
- Trust any signals or ignore them
- Go against the signals if you have a reason
- Use any combination of strategies
- Size positions however you want (max 20% per position)

Consider:
1. Signal alignment - are multiple signals agreeing?
2. Market regime - is this a good time to trade?
3. Semantic memory - what happened historically in similar conditions?
4. Current positions - should you add, reduce, or hold?

Respond with your trading decisions in this JSON format:
```json
{{
  "decisions": [
    {{
      "action": "buy" or "sell" or "hold",
      "symbol": "TICKER",
      "size": 0.10,
      "reasoning": "Your explanation"
    }}
  ],
  "market_outlook": "Brief assessment of current conditions"
}}
```

If you decide to hold all positions with no changes, return an empty decisions array.
Only include actionable decisions (buy/sell), not hold decisions for existing positions.
"""
        return prompt
    
    def _parse_response(self, response_text: str) -> List[TradingDecision]:
        """Parse LLM response into trading decisions"""
        decisions = []
        
        try:
            # Extract JSON from response
            json_start = response_text.find('{')
            json_end = response_text.rfind('}') + 1
            
            if json_start == -1 or json_end == 0:
                return decisions
            
            json_str = response_text[json_start:json_end]
            data = json.loads(json_str)
            
            for d in data.get("decisions", []):
                action_str = d.get("action", "hold").lower()
                if action_str == "buy":
                    action = Action.BUY
                elif action_str == "sell":
                    action = Action.SELL
                else:
                    continue  # Skip holds
                
                decisions.append(TradingDecision(
                    action=action,
                    symbol=d.get("symbol", "").upper(),
                    size=float(d.get("size", 0.10)),
                    reasoning=d.get("reasoning", ""),
                    confidence=0.7  # Default confidence
                ))
        
        except json.JSONDecodeError:
            # Try to parse individual decisions from text
            pass
        
        return decisions
    
    async def _call_openai(self, prompt: str) -> str:
        """Call OpenAI API"""
        client = await self._get_client()
        response = await client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an expert AI portfolio manager making autonomous "
                        "trading decisions. Respond only with valid JSON."
                    )
                },
                {"role": "user", "content": prompt}
            ],
            temperature=self.temperature,
            max_tokens=1000
        )
        return response.choices[0].message.content
    
    async def _call_anthropic(self, prompt: str) -> str:
        """Call Anthropic API"""
        client = await self._get_client()
        response = await client.messages.create(
            model=self.model,
            max_tokens=1000,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        return response.content[0].text
    
    async def _call_google(self, prompt: str) -> str:
        """Call Google AI API"""
        client = await self._get_client()
        response = await client.generate_content_async(prompt)
        return response.text
    
    async def make_decisions(
        self,
        context: ManagerContext
    ) -> List[TradingDecision]:
        """
        Make trading decisions using LLM reasoning.
        
        Full autonomy to interpret signals and make independent decisions.
        """
        prompt = self._build_prompt(context)
        
        try:
            # Call appropriate LLM
            if self.provider == "openai":
                response = await self._call_openai(prompt)
            elif self.provider == "anthropic":
                response = await self._call_anthropic(prompt)
            elif self.provider == "google":
                response = await self._call_google(prompt)
            else:
                return []
            
            # Parse response into decisions
            decisions = self._parse_response(response)
            
            # Apply risk limits
            return self.apply_risk_limits(decisions, context)
        
        except Exception as e:
            # Log error and return no decisions
            print(f"Error in {self.name} decision making: {e}")
            return []


# Factory functions for creating specific LLM managers
def create_gpt4_manager(
    initial_capital: float = 25000.0,
    risk_limits: Optional[RiskLimits] = None
) -> LLMManager:
    """Create GPT-4 powered manager"""
    return LLMManager(
        manager_id="gpt4",
        name="GPT-4 Fund",
        provider="openai",
        model="gpt-4-turbo-preview",
        initial_capital=initial_capital,
        risk_limits=risk_limits
    )


def create_claude_manager(
    initial_capital: float = 25000.0,
    risk_limits: Optional[RiskLimits] = None
) -> LLMManager:
    """Create Claude powered manager"""
    return LLMManager(
        manager_id="claude",
        name="Claude Fund",
        provider="anthropic",
        model="claude-3-sonnet-20240229",
        initial_capital=initial_capital,
        risk_limits=risk_limits
    )


def create_gemini_manager(
    initial_capital: float = 25000.0,
    risk_limits: Optional[RiskLimits] = None
) -> LLMManager:
    """Create Gemini powered manager"""
    return LLMManager(
        manager_id="gemini",
        name="Gemini Fund",
        provider="google",
        model="gemini-pro",
        initial_capital=initial_capital,
        risk_limits=risk_limits
    )
