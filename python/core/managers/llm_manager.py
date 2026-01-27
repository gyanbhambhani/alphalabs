"""
LLM Portfolio Manager

Uses LLMs (GPT-4, Claude, Gemini) for autonomous trading decisions.
Full autonomy to interpret signals and make independent decisions.

Enhanced with:
- Deep historical context from semantic search
- Quantitative models (Sharpe, Black-Scholes)
- Narrative-style thesis generation
- Full decision logging and consciousness tracking
"""
import json
from typing import List, Dict, Optional, Literal, Any
from datetime import datetime
from dataclasses import dataclass

from core.managers.base import (
    BaseManager, TradingDecision, ManagerContext,
    Action, RiskLimits
)


@dataclass
class EnhancedDecision(TradingDecision):
    """Enhanced trading decision with full context"""
    thesis: str = ""
    historical_precedent: str = ""
    expected_holding_period: str = ""
    stop_loss: Optional[float] = None
    target_return: Optional[float] = None
    geopolitical_factors: List[str] = None
    market_regime: str = ""
    
    def __post_init__(self):
        if self.geopolitical_factors is None:
            self.geopolitical_factors = []


@dataclass
class DeepContextData:
    """Container for deep context to pass to LLM"""
    market_context: Optional[Any] = None
    quant_analysis: Optional[Any] = None
    recent_performance: str = ""


class LLMManager(BaseManager):
    """
    LLM-powered portfolio manager with full autonomy and deep context.
    
    Can interpret signals, understand context, and make nuanced decisions
    that pure algorithms cannot.
    
    Enhanced features:
    - Deep historical context with narratives
    - Quantitative model integration
    - Full decision consciousness logging
    - Hedge fund-style thesis generation
    """
    
    def __init__(
        self,
        manager_id: str,
        name: str,
        provider: Literal["openai", "anthropic", "google"],
        initial_capital: float = 25000.0,
        risk_limits: Optional[RiskLimits] = None,
        model: Optional[str] = None,
        temperature: float = 0.4,  # Lower for more consistent reasoning
        context_provider: Optional[Any] = None,
        quant_models: Optional[Any] = None,
        decision_logger: Optional[Any] = None
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
        
        # Enhanced context providers
        self.context_provider = context_provider
        self.quant_models = quant_models
        self.decision_logger = decision_logger
        
        # Last response for debugging
        self._last_raw_response: str = ""
        self._last_parsed_response: Dict = {}
    
    def _get_default_model(self, provider: str) -> str:
        """Get default model for provider"""
        models = {
            "openai": "gpt-4-turbo-preview",
            "anthropic": "claude-3-5-sonnet-20241022",
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
        """Build basic prompt for the LLM (fallback)"""
        return self._build_deep_prompt(context, None)
    
    def _build_deep_prompt(
        self, 
        context: ManagerContext, 
        deep_context: Optional[DeepContextData] = None
    ) -> str:
        """
        Build narrative-style prompt with full deep context.
        
        This is the enhanced prompt that enables hedge fund-style reasoning.
        """
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
        
        # Deep context sections
        historical_context = ""
        quant_context = ""
        performance_feedback = ""
        
        if deep_context:
            if deep_context.market_context:
                historical_context = deep_context.market_context.to_prompt_context()
            if deep_context.quant_analysis:
                quant_context = deep_context.quant_analysis
            if deep_context.recent_performance:
                performance_feedback = deep_context.recent_performance
        
        # Fallback semantic search
        semantic = context.signals.semantic_search
        if not historical_context and semantic:
            historical_context = f"""
## Historical Context (Basic)
Average 5-day return in similar periods: {semantic.get('avg_5d_return', 0):.2%}
Average 20-day return in similar periods: {semantic.get('avg_20d_return', 0):.2%}
Positive outcome rate (5-day): {semantic.get('positive_5d_rate', 0):.0%}
Interpretation: {semantic.get('interpretation', 'N/A')}
"""
        
        prompt = f"""You are {self.name}, managing a ${context.portfolio.total_value:,.0f} portfolio.

You are an elite hedge fund manager with access to 45+ years of market history.
Think deeply about market conditions and write investment theses like a professional.

## Current Time
{context.timestamp.strftime('%Y-%m-%d %H:%M:%S')} UTC

## Your Portfolio
{portfolio_str}

## Market Regime
{context.signals.volatility_regime}

{historical_context}

{quant_context if quant_context else ""}

## Technical Signals

### Momentum Signals (12-month return, skip 1 month)
{momentum_str}

### Mean Reversion Signals (Bollinger Z-score)
{mean_rev_str}

{performance_feedback if performance_feedback else ""}

## Your Task: Write an Investment Thesis

As a conscious AI hedge fund manager, you must:

1. **Analyze the Market Regime**
   - What phase are we in? (expansion, contraction, crisis, recovery, euphoria)
   - How certain are you based on historical pattern matches?
   - What geopolitical or economic factors are at play?

2. **Develop Your Trade Rationale**
   - Why this trade, why now?
   - Which historical period is most similar and relevant?
   - What's different about the current situation?

3. **Assess Risks Thoroughly**
   - What could go wrong?
   - What was the worst outcome in similar historical periods?
   - How should you size the position based on conviction?

4. **Plan Your Exit**
   - What conditions would make you exit?
   - What's your time horizon based on historical patterns?
   - Where's your stop loss and target?

## Response Format

Write your complete analysis, then provide structured decisions:

```json
{{
  "thesis": "Your complete investment thesis (500+ words). Explain your thinking like a hedge fund manager writing to investors. Reference specific historical periods. Discuss both bull and bear cases.",
  
  "conviction": 0.75,
  
  "market_regime": "early_recovery",
  
  "geopolitical_factors": [
    "Factor 1 affecting your view",
    "Factor 2 affecting your view"
  ],
  
  "trades": [
    {{
      "action": "buy",
      "symbol": "AAPL",
      "size": 0.15,
      "reasoning": "Detailed trade-specific reasoning...",
      "historical_precedent": "2020-03-23",
      "expected_holding_period": "3-6 months",
      "stop_loss": -0.08,
      "target_return": 0.25
    }}
  ],
  
  "risks": [
    "Key risk 1",
    "Key risk 2",
    "Key risk 3"
  ],
  
  "market_outlook": "Your overall market assessment"
}}
```

Important guidelines:
- Position sizes: max 20% per position, consider volatility
- If no compelling trade, return empty trades array - patience is a strategy
- Be specific about WHY you're making each trade
- Reference historical precedents when possible
- Consider what worked and what failed in similar periods
"""
        return prompt
    
    def _parse_response(
        self, 
        response_text: str
    ) -> tuple[List[EnhancedDecision], Dict[str, Any]]:
        """
        Parse LLM response into enhanced trading decisions.
        
        Returns:
            Tuple of (decisions list, parsed response dict)
        """
        decisions = []
        parsed_data = {}
        
        self._last_raw_response = response_text
        
        try:
            # Extract JSON from response
            json_start = response_text.find('{')
            json_end = response_text.rfind('}') + 1
            
            if json_start == -1 or json_end == 0:
                return decisions, parsed_data
            
            json_str = response_text[json_start:json_end]
            data = json.loads(json_str)
            self._last_parsed_response = data
            parsed_data = data
            
            # Extract top-level fields
            thesis = data.get("thesis", "")
            conviction = float(data.get("conviction", 0.7))
            market_regime = data.get("market_regime", "unknown")
            geopolitical_factors = data.get("geopolitical_factors", [])
            risks = data.get("risks", [])
            
            for d in data.get("trades", data.get("decisions", [])):
                action_str = d.get("action", "hold").lower()
                if action_str == "buy":
                    action = Action.BUY
                elif action_str == "sell":
                    action = Action.SELL
                else:
                    continue  # Skip holds
                
                decisions.append(EnhancedDecision(
                    action=action,
                    symbol=d.get("symbol", "").upper(),
                    size=float(d.get("size", 0.10)),
                    reasoning=d.get("reasoning", ""),
                    confidence=conviction,
                    thesis=thesis,
                    historical_precedent=d.get("historical_precedent", ""),
                    expected_holding_period=d.get("expected_holding_period", ""),
                    stop_loss=d.get("stop_loss"),
                    target_return=d.get("target_return"),
                    geopolitical_factors=geopolitical_factors,
                    market_regime=market_regime
                ))
        
        except json.JSONDecodeError as e:
            print(f"JSON parse error: {e}")
            # Try to extract basic info from text
            pass
        
        return decisions, parsed_data
    
    async def get_deep_context(
        self, 
        symbols: List[str]
    ) -> DeepContextData:
        """
        Get deep context for decision making.
        
        Fetches:
        - Historical context from semantic search
        - Quantitative analysis
        - Recent performance feedback
        """
        deep_data = DeepContextData()
        
        # Get market context for primary symbol
        if self.context_provider and symbols:
            try:
                deep_data.market_context = self.context_provider.get_deep_context(
                    symbol=symbols[0]
                )
            except Exception as e:
                print(f"Error getting market context: {e}")
        
        # Get performance feedback
        if self.decision_logger:
            try:
                deep_data.recent_performance = self.decision_logger.to_prompt_feedback(
                    manager_id=self.manager_id,
                    last_n=5
                )
            except Exception as e:
                print(f"Error getting performance feedback: {e}")
        
        return deep_data
    
    def _log_decision(
        self,
        decision: EnhancedDecision,
        deep_context: Optional[DeepContextData],
        parsed_response: Dict[str, Any]
    ) -> Optional[str]:
        """Log a decision with full consciousness tracking"""
        if not self.decision_logger:
            return None
        
        try:
            # Build historical matches from deep context
            historical_matches = []
            if deep_context and deep_context.market_context:
                for period in deep_context.market_context.similar_periods[:5]:
                    from core.logging.decision_logger import HistoricalMatch
                    historical_matches.append(HistoricalMatch(
                        date=period.date,
                        similarity=period.similarity,
                        regime=period.regime.value,
                        forward_return_1m=period.forward_outcome.return_1m,
                        forward_return_3m=period.forward_outcome.return_3m,
                        narrative=period.narrative,
                        geopolitical_context=period.geopolitical_context
                    ))
            
            # Extract context values
            market_regime = decision.market_regime
            volatility = 0.2
            momentum_1m = 0.0
            momentum_3m = 0.0
            
            if deep_context and deep_context.market_context:
                ctx = deep_context.market_context
                volatility = ctx.current_volatility
                momentum_1m = ctx.current_momentum_1m
                momentum_3m = ctx.current_momentum_3m
            
            # Log the decision
            decision_id = self.decision_logger.log_decision(
                manager_id=self.manager_id,
                symbol=decision.symbol,
                action=decision.action.value,
                size=decision.size,
                price=0.0,  # Will be filled at execution
                thesis=decision.thesis,
                conviction=decision.confidence,
                historical_matches=historical_matches,
                market_regime=market_regime,
                volatility=volatility,
                momentum_1m=momentum_1m,
                momentum_3m=momentum_3m,
                sharpe_expected=parsed_response.get("sharpe_expected", 0),
                sortino_expected=parsed_response.get("sortino_expected", 0),
                optimal_weight=decision.size,
                expected_return=decision.target_return or 0.1,
                max_drawdown_expected=abs(decision.stop_loss or 0.05),
                geopolitical_factors=decision.geopolitical_factors,
                signals_used={},
                stop_loss=decision.stop_loss,
                target_return=decision.target_return
            )
            
            return decision_id
            
        except Exception as e:
            print(f"Error logging decision: {e}")
            return None
    
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
        Make trading decisions using LLM reasoning with deep context.
        
        Full autonomy to interpret signals and make independent decisions.
        Enhanced with:
        - Deep historical context
        - Quantitative analysis
        - Performance feedback
        - Decision logging
        """
        # Get symbols from signals
        symbols = list(context.signals.momentum.keys())[:5]
        
        # Fetch deep context
        deep_context = None
        if self.context_provider or self.decision_logger:
            try:
                deep_context = await self.get_deep_context(symbols)
            except Exception as e:
                print(f"Error fetching deep context: {e}")
        
        # Build enhanced prompt
        prompt = self._build_deep_prompt(context, deep_context)
        
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
            
            # Parse response into enhanced decisions
            decisions, parsed_response = self._parse_response(response)
            
            # Log each decision
            for decision in decisions:
                decision_id = self._log_decision(decision, deep_context, parsed_response)
                if decision_id:
                    print(f"Logged decision {decision_id} for {decision.symbol}")
            
            # Apply risk limits
            return self.apply_risk_limits(decisions, context)
        
        except Exception as e:
            # Log error and return no decisions
            print(f"Error in {self.name} decision making: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    async def make_decisions_simple(
        self,
        context: ManagerContext
    ) -> List[TradingDecision]:
        """
        Make trading decisions with basic prompt (no deep context).
        
        Useful for testing or when context providers aren't available.
        """
        prompt = self._build_deep_prompt(context, None)
        
        try:
            if self.provider == "openai":
                response = await self._call_openai(prompt)
            elif self.provider == "anthropic":
                response = await self._call_anthropic(prompt)
            elif self.provider == "google":
                response = await self._call_google(prompt)
            else:
                return []
            
            decisions, _ = self._parse_response(response)
            return self.apply_risk_limits(decisions, context)
        
        except Exception as e:
            print(f"Error in {self.name} simple decision making: {e}")
            return []
    
    def get_last_thesis(self) -> str:
        """Get the thesis from the last decision"""
        return self._last_parsed_response.get("thesis", "")
    
    def get_last_market_outlook(self) -> str:
        """Get market outlook from the last decision"""
        return self._last_parsed_response.get("market_outlook", "")


# Factory functions for creating specific LLM managers
def create_gpt4_manager(
    initial_capital: float = 25000.0,
    risk_limits: Optional[RiskLimits] = None,
    context_provider: Optional[Any] = None,
    quant_models: Optional[Any] = None,
    decision_logger: Optional[Any] = None
) -> LLMManager:
    """Create GPT-4 powered manager with optional deep context"""
    return LLMManager(
        manager_id="gpt4",
        name="GPT-4 Fund",
        provider="openai",
        model="gpt-4-turbo-preview",
        initial_capital=initial_capital,
        risk_limits=risk_limits,
        context_provider=context_provider,
        quant_models=quant_models,
        decision_logger=decision_logger
    )


def create_claude_manager(
    initial_capital: float = 25000.0,
    risk_limits: Optional[RiskLimits] = None,
    context_provider: Optional[Any] = None,
    quant_models: Optional[Any] = None,
    decision_logger: Optional[Any] = None
) -> LLMManager:
    """Create Claude powered manager with optional deep context"""
    return LLMManager(
        manager_id="claude",
        name="Claude Fund",
        provider="anthropic",
        model="claude-3-5-sonnet-20241022",
        initial_capital=initial_capital,
        risk_limits=risk_limits,
        context_provider=context_provider,
        quant_models=quant_models,
        decision_logger=decision_logger
    )


def create_gemini_manager(
    initial_capital: float = 25000.0,
    risk_limits: Optional[RiskLimits] = None,
    context_provider: Optional[Any] = None,
    quant_models: Optional[Any] = None,
    decision_logger: Optional[Any] = None
) -> LLMManager:
    """Create Gemini powered manager with optional deep context"""
    return LLMManager(
        manager_id="gemini",
        name="Gemini Fund",
        provider="google",
        model="gemini-pro",
        initial_capital=initial_capital,
        risk_limits=risk_limits,
        context_provider=context_provider,
        quant_models=quant_models,
        decision_logger=decision_logger
    )


def create_conscious_manager(
    manager_id: str,
    name: str,
    provider: Literal["openai", "anthropic", "google"],
    persist_directory: str = "./chroma_data"
) -> LLMManager:
    """
    Create a fully conscious LLM manager with all context providers.
    
    This is the recommended way to create managers for production use.
    """
    from core.context.market_context import MarketContextProvider
    from core.quant.models import QuantitativeModels
    from core.logging.decision_logger import DecisionLogger
    
    context_provider = MarketContextProvider(persist_directory=persist_directory)
    quant_models = QuantitativeModels()
    decision_logger = DecisionLogger()
    
    models = {
        "openai": "gpt-4-turbo-preview",
        "anthropic": "claude-3-5-sonnet-20241022",
        "google": "gemini-pro"
    }
    
    return LLMManager(
        manager_id=manager_id,
        name=name,
        provider=provider,
        model=models.get(provider),
        temperature=0.4,
        context_provider=context_provider,
        quant_models=quant_models,
        decision_logger=decision_logger
    )
