"""
Market Context Provider

Provides deep historical context for trading decisions.
Goes beyond simple pattern matching to provide narrative understanding.
"""
import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum

from core.semantic.vector_db import VectorDatabase, SearchResult
from core.semantic.encoder import MarketStateEncoder


class MarketRegime(Enum):
    """Market regime classification"""
    BULL_QUIET = "bull_quiet"           # Low vol uptrend
    BULL_VOLATILE = "bull_volatile"     # High vol uptrend
    BEAR_QUIET = "bear_quiet"           # Low vol downtrend
    BEAR_VOLATILE = "bear_volatile"     # High vol downtrend (crisis)
    SIDEWAYS = "sideways"               # Range-bound
    RECOVERY = "recovery"               # Post-crisis bounce
    EUPHORIA = "euphoria"               # Extreme bullishness
    CAPITULATION = "capitulation"       # Extreme selling


@dataclass
class ForwardOutcome:
    """Forward returns and risk metrics from a historical period"""
    return_1w: float
    return_1m: float
    return_3m: float
    return_6m: float
    max_drawdown_1m: float
    volatility_realized: float
    outcome_category: str  # "strong_bull", "bull", "flat", "bear", "crash"


@dataclass
class HistoricalPeriodAnalysis:
    """Deep analysis of a single historical period"""
    date: str
    similarity: float
    
    # Market conditions at that time
    regime: MarketRegime
    volatility: float
    momentum_1m: float
    momentum_3m: float
    price: float
    
    # What happened next
    forward_outcome: ForwardOutcome
    
    # Narrative context (populated by LLM or rules)
    narrative: str
    geopolitical_context: str
    what_worked: str
    what_failed: str
    
    # Key events around this period
    notable_events: List[str] = field(default_factory=list)


@dataclass
class DeepContext:
    """Complete deep context for trading decisions"""
    symbol: str
    current_date: str
    
    # Current market state
    current_regime: MarketRegime
    current_volatility: float
    current_momentum_1m: float
    current_momentum_3m: float
    current_price: float
    
    # Similar historical periods with full analysis
    similar_periods: List[HistoricalPeriodAnalysis]
    
    # Aggregate statistics from similar periods
    avg_forward_return_1m: float
    avg_forward_return_3m: float
    positive_outcome_rate: float
    worst_case_drawdown: float
    best_case_return: float
    
    # Confidence and interpretation
    confidence_score: float
    market_interpretation: str
    recommended_stance: str  # "aggressive_long", "cautious_long", "neutral", etc.
    
    # Risk factors
    key_risks: List[str] = field(default_factory=list)
    
    def to_prompt_context(self) -> str:
        """Format for LLM prompt"""
        periods_text = ""
        for i, period in enumerate(self.similar_periods[:5], 1):
            periods_text += f"""
    {i}. {period.date} (Similarity: {period.similarity:.1%})
       Regime: {period.regime.value}
       Outcome: {period.forward_outcome.return_1m:+.1%} (1M), 
                {period.forward_outcome.return_3m:+.1%} (3M)
       Max Drawdown: {period.forward_outcome.max_drawdown_1m:.1%}
       Context: {period.narrative}
       Geopolitical: {period.geopolitical_context}
"""
        
        return f"""
## Deep Historical Context for {self.symbol}

### Current Market State
- Regime: {self.current_regime.value}
- Volatility (21D): {self.current_volatility:.1%}
- Momentum (1M): {self.current_momentum_1m:+.1%}
- Momentum (3M): {self.current_momentum_3m:+.1%}

### Similar Historical Periods (Top 5)
{periods_text}

### Statistical Summary
- Average 1M Forward Return: {self.avg_forward_return_1m:+.1%}
- Average 3M Forward Return: {self.avg_forward_return_3m:+.1%}
- Positive Outcome Rate: {self.positive_outcome_rate:.0%}
- Worst Case Drawdown: {self.worst_case_drawdown:.1%}
- Best Case Return: {self.best_case_return:+.1%}

### Market Interpretation
{self.market_interpretation}

### Recommended Stance: {self.recommended_stance}
Confidence: {self.confidence_score:.0%}

### Key Risks
{chr(10).join(f'- {risk}' for risk in self.key_risks)}
"""


# Known historical events for context
HISTORICAL_EVENTS = {
    # Financial crises
    "2008-09": "Global Financial Crisis - Lehman collapse",
    "2008-10": "Global Financial Crisis - Peak panic",
    "2008-11": "Global Financial Crisis - Government bailouts",
    "2009-03": "Financial Crisis bottom - QE begins",
    
    # COVID
    "2020-02": "COVID-19 fears emerging",
    "2020-03": "COVID crash - Fastest bear market in history",
    "2020-04": "COVID recovery begins - Fed unlimited QE",
    
    # Tech bubble
    "2000-03": "Dot-com bubble peak",
    "2000-04": "Dot-com crash begins",
    "2002-10": "Dot-com bear market bottom",
    
    # Recent events
    "2022-01": "Fed pivot to hawkish - Rate hike cycle begins",
    "2022-06": "Bear market confirmed - Inflation peaks",
    "2022-10": "Bear market bottom - Inflation cooling",
    "2023-03": "Regional bank crisis - SVB collapse",
    
    # Historic
    "1987-10": "Black Monday - Single day crash",
    "1997-10": "Asian Financial Crisis",
    "2001-09": "9/11 attacks - Market closed",
    "2011-08": "US debt downgrade - Flash crash",
    "2015-08": "China devaluation - Global selloff",
    "2018-12": "Fed overtightening - Christmas Eve low",
}


class MarketContextProvider:
    """
    Provides deep historical context for trading decisions.
    
    Goes beyond simple pattern matching to provide:
    - Narrative understanding of market conditions
    - Geopolitical and economic context
    - Forward outcome analysis
    - Risk assessment
    """
    
    def __init__(
        self,
        persist_directory: str = "./chroma_data",
        llm_provider: Optional[str] = None
    ):
        """
        Initialize the context provider.
        
        Args:
            persist_directory: ChromaDB storage location
            llm_provider: Optional LLM for narrative generation
        """
        self.persist_directory = persist_directory
        self.llm_provider = llm_provider
        self.encoder = MarketStateEncoder()
        self._vector_dbs: Dict[str, VectorDatabase] = {}
    
    def _get_vector_db(self, symbol: str) -> VectorDatabase:
        """Get or create vector database for symbol"""
        if symbol not in self._vector_dbs:
            self._vector_dbs[symbol] = VectorDatabase(
                persist_directory=self.persist_directory,
                symbol=symbol
            )
        return self._vector_dbs[symbol]
    
    def _classify_regime(
        self,
        momentum_1m: float,
        momentum_3m: float,
        volatility: float
    ) -> MarketRegime:
        """Classify current market regime"""
        is_high_vol = volatility > 0.25
        is_low_vol = volatility < 0.15
        is_bullish = momentum_1m > 0.02 and momentum_3m > 0
        is_bearish = momentum_1m < -0.02 and momentum_3m < 0
        is_extreme_bull = momentum_1m > 0.08
        is_extreme_bear = momentum_1m < -0.08
        
        if is_extreme_bear and is_high_vol:
            return MarketRegime.CAPITULATION
        elif is_extreme_bull and is_high_vol:
            return MarketRegime.EUPHORIA
        elif is_bullish and is_extreme_bull and momentum_3m < 0:
            return MarketRegime.RECOVERY
        elif is_bullish and is_high_vol:
            return MarketRegime.BULL_VOLATILE
        elif is_bullish and is_low_vol:
            return MarketRegime.BULL_QUIET
        elif is_bearish and is_high_vol:
            return MarketRegime.BEAR_VOLATILE
        elif is_bearish:
            return MarketRegime.BEAR_QUIET
        else:
            return MarketRegime.SIDEWAYS
    
    def _categorize_outcome(self, return_1m: float) -> str:
        """Categorize forward outcome"""
        if return_1m > 0.08:
            return "strong_bull"
        elif return_1m > 0.02:
            return "bull"
        elif return_1m > -0.02:
            return "flat"
        elif return_1m > -0.08:
            return "bear"
        else:
            return "crash"
    
    def _get_historical_context(self, date: str) -> Tuple[str, List[str]]:
        """Get geopolitical/economic context for a date"""
        year_month = date[:7]
        
        # Check for known events
        events = []
        context = "Normal market conditions"
        
        if year_month in HISTORICAL_EVENTS:
            events.append(HISTORICAL_EVENTS[year_month])
            context = HISTORICAL_EVENTS[year_month]
        
        # Check surrounding months
        try:
            dt = datetime.strptime(date, "%Y-%m-%d")
            for delta in [-1, 1]:
                check_date = (dt + timedelta(days=delta * 30)).strftime("%Y-%m")
                if check_date in HISTORICAL_EVENTS:
                    events.append(f"Near: {HISTORICAL_EVENTS[check_date]}")
        except ValueError:
            pass
        
        # Generate context based on date patterns
        year = int(date[:4])
        month = int(date[5:7])
        
        if year == 2008:
            context = "Global Financial Crisis era"
        elif year == 2020 and month in [2, 3, 4]:
            context = "COVID-19 pandemic crisis"
        elif year == 2000 and month > 2:
            context = "Dot-com bubble bursting"
        elif year == 2022:
            context = "Fed rate hike cycle, inflation concerns"
        elif year == 2023:
            context = "Post-pandemic normalization, AI boom"
        
        return context, events
    
    def _generate_narrative(
        self,
        regime: MarketRegime,
        volatility: float,
        momentum_1m: float,
        forward_return: float,
        date: str
    ) -> str:
        """Generate narrative description of market period"""
        regime_desc = {
            MarketRegime.BULL_QUIET: "steady uptrend with low volatility",
            MarketRegime.BULL_VOLATILE: "volatile rally with sharp swings",
            MarketRegime.BEAR_QUIET: "grinding decline with low volatility",
            MarketRegime.BEAR_VOLATILE: "sharp selloff with panic",
            MarketRegime.SIDEWAYS: "range-bound consolidation",
            MarketRegime.RECOVERY: "sharp bounce from oversold conditions",
            MarketRegime.EUPHORIA: "euphoric rally with extreme optimism",
            MarketRegime.CAPITULATION: "capitulation with extreme fear",
        }
        
        outcome_desc = self._categorize_outcome(forward_return)
        outcome_text = {
            "strong_bull": "followed by strong gains",
            "bull": "followed by modest gains",
            "flat": "followed by sideways action",
            "bear": "followed by losses",
            "crash": "followed by sharp decline",
        }
        
        return (
            f"Market showed {regime_desc.get(regime, 'mixed conditions')} "
            f"with {volatility:.0%} volatility, "
            f"{outcome_text.get(outcome_desc, 'uncertain outcome')}."
        )
    
    def _generate_what_worked(
        self,
        regime: MarketRegime,
        forward_return: float
    ) -> str:
        """Generate what strategies worked in this period"""
        if forward_return > 0.05:
            if regime in [MarketRegime.RECOVERY, MarketRegime.CAPITULATION]:
                return "Buying the dip, high-beta stocks, growth names"
            elif regime == MarketRegime.BULL_QUIET:
                return "Buy and hold, momentum strategies, quality growth"
            else:
                return "Long positions, trend following"
        elif forward_return < -0.05:
            return "Defensive positions, hedging, cash preservation"
        else:
            return "Range trading, options selling, low volatility strategies"
    
    def _generate_what_failed(
        self,
        regime: MarketRegime,
        forward_return: float
    ) -> str:
        """Generate what strategies failed in this period"""
        if forward_return > 0.05:
            return "Short positions, excessive hedging, sitting in cash"
        elif forward_return < -0.05:
            if regime == MarketRegime.BEAR_VOLATILE:
                return "Buy the dip, leverage, concentrated positions"
            else:
                return "Long-only strategies, momentum following"
        else:
            return "Directional bets, breakout strategies"
    
    def _calculate_forward_outcome(
        self,
        metadata: Dict
    ) -> ForwardOutcome:
        """Calculate forward outcome from embedding metadata"""
        return_1m = metadata.get("return_1m", 0)
        return_3m = metadata.get("return_3m", 0)
        volatility = metadata.get("volatility_21d", 0.2)
        
        # Estimate other metrics from available data
        return ForwardOutcome(
            return_1w=metadata.get("return_1w", return_1m / 4),
            return_1m=return_1m,
            return_3m=return_3m,
            return_6m=metadata.get("return_6m", return_3m * 1.5),
            max_drawdown_1m=-abs(return_1m) if return_1m < 0 else -volatility * 0.5,
            volatility_realized=volatility,
            outcome_category=self._categorize_outcome(return_1m)
        )
    
    def analyze_period(
        self,
        symbol: str,
        date: str,
        similarity: float,
        metadata: Dict
    ) -> HistoricalPeriodAnalysis:
        """
        Deep analysis of a single historical period.
        
        Args:
            symbol: Stock symbol
            date: Historical date
            similarity: Similarity score to current
            metadata: Embedding metadata
        
        Returns:
            Complete period analysis
        """
        # Extract metrics
        momentum_1m = metadata.get("return_1m", 0)
        momentum_3m = metadata.get("return_3m", 0)
        volatility = metadata.get("volatility_21d", 0.2)
        price = metadata.get("price", 0)
        
        # Classify regime
        regime = self._classify_regime(momentum_1m, momentum_3m, volatility)
        
        # Calculate forward outcome
        forward_outcome = self._calculate_forward_outcome(metadata)
        
        # Get historical context
        geo_context, events = self._get_historical_context(date)
        
        # Generate narratives
        narrative = self._generate_narrative(
            regime, volatility, momentum_1m, forward_outcome.return_1m, date
        )
        what_worked = self._generate_what_worked(regime, forward_outcome.return_1m)
        what_failed = self._generate_what_failed(regime, forward_outcome.return_1m)
        
        return HistoricalPeriodAnalysis(
            date=date,
            similarity=similarity,
            regime=regime,
            volatility=volatility,
            momentum_1m=momentum_1m,
            momentum_3m=momentum_3m,
            price=price,
            forward_outcome=forward_outcome,
            narrative=narrative,
            geopolitical_context=geo_context,
            what_worked=what_worked,
            what_failed=what_failed,
            notable_events=events
        )
    
    def get_deep_context(
        self,
        symbol: str,
        current_date: Optional[str] = None,
        top_k: int = 20
    ) -> DeepContext:
        """
        Get deep historical context for a symbol.
        
        Args:
            symbol: Stock symbol
            current_date: Date to analyze (default: latest)
            top_k: Number of similar periods to analyze
        
        Returns:
            Complete deep context with narratives and analysis
        """
        vector_db = self._get_vector_db(symbol)
        
        if current_date is None:
            current_date = datetime.now().strftime("%Y-%m-%d")
        
        # Get all embeddings to find current state
        all_data = vector_db.collection.get(include=["metadatas"])
        
        if not all_data["ids"]:
            raise ValueError(f"No embeddings found for {symbol}")
        
        # Find the most recent embedding
        sorted_dates = sorted(all_data["ids"], reverse=True)
        latest_date = sorted_dates[0]
        latest_idx = all_data["ids"].index(latest_date)
        latest_metadata = all_data["metadatas"][latest_idx]
        
        # Current state metrics
        current_momentum_1m = latest_metadata.get("return_1m", 0)
        current_momentum_3m = latest_metadata.get("return_3m", 0)
        current_volatility = latest_metadata.get("volatility_21d", 0.2)
        current_price = latest_metadata.get("price", 0)
        current_regime = self._classify_regime(
            current_momentum_1m, current_momentum_3m, current_volatility
        )
        
        # Search for similar periods
        # Get the embedding vector for the latest date
        latest_embedding = vector_db.collection.get(
            ids=[latest_date],
            include=["embeddings"]
        )["embeddings"][0]
        
        search_results = vector_db.search(
            query_vector=np.array(latest_embedding),
            top_k=top_k + 50  # Get extra to filter recent
        )
        
        # Analyze similar periods (excluding recent ones)
        cutoff_date = (
            datetime.now() - timedelta(days=63)  # 3 months
        ).strftime("%Y-%m-%d")
        
        similar_periods = []
        for result in search_results:
            if result.date >= cutoff_date:
                continue
            if result.date == latest_date:
                continue
            
            analysis = self.analyze_period(
                symbol=symbol,
                date=result.date,
                similarity=result.similarity,
                metadata=result.metadata
            )
            similar_periods.append(analysis)
            
            if len(similar_periods) >= top_k:
                break
        
        if not similar_periods:
            # Return minimal context if no similar periods found
            return DeepContext(
                symbol=symbol,
                current_date=latest_date,
                current_regime=current_regime,
                current_volatility=current_volatility,
                current_momentum_1m=current_momentum_1m,
                current_momentum_3m=current_momentum_3m,
                current_price=current_price,
                similar_periods=[],
                avg_forward_return_1m=0,
                avg_forward_return_3m=0,
                positive_outcome_rate=0.5,
                worst_case_drawdown=-0.1,
                best_case_return=0.1,
                confidence_score=0.3,
                market_interpretation="Insufficient historical data for analysis",
                recommended_stance="neutral",
                key_risks=["Limited historical context available"]
            )
        
        # Calculate aggregate statistics
        returns_1m = [p.forward_outcome.return_1m for p in similar_periods]
        returns_3m = [p.forward_outcome.return_3m for p in similar_periods]
        
        avg_return_1m = np.mean(returns_1m)
        avg_return_3m = np.mean(returns_3m)
        positive_rate = sum(1 for r in returns_1m if r > 0) / len(returns_1m)
        worst_drawdown = min(p.forward_outcome.max_drawdown_1m for p in similar_periods)
        best_return = max(returns_1m)
        
        # Generate interpretation
        interpretation = self._generate_market_interpretation(
            current_regime, similar_periods, avg_return_1m, positive_rate
        )
        
        # Determine recommended stance
        stance = self._determine_stance(
            current_regime, avg_return_1m, positive_rate, current_volatility
        )
        
        # Calculate confidence
        avg_similarity = np.mean([p.similarity for p in similar_periods[:5]])
        confidence = min(0.95, avg_similarity * positive_rate)
        
        # Identify key risks
        risks = self._identify_risks(
            current_regime, similar_periods, current_volatility
        )
        
        return DeepContext(
            symbol=symbol,
            current_date=latest_date,
            current_regime=current_regime,
            current_volatility=current_volatility,
            current_momentum_1m=current_momentum_1m,
            current_momentum_3m=current_momentum_3m,
            current_price=current_price,
            similar_periods=similar_periods,
            avg_forward_return_1m=avg_return_1m,
            avg_forward_return_3m=avg_return_3m,
            positive_outcome_rate=positive_rate,
            worst_case_drawdown=worst_drawdown,
            best_case_return=best_return,
            confidence_score=confidence,
            market_interpretation=interpretation,
            recommended_stance=stance,
            key_risks=risks
        )
    
    def _generate_market_interpretation(
        self,
        regime: MarketRegime,
        similar_periods: List[HistoricalPeriodAnalysis],
        avg_return: float,
        positive_rate: float
    ) -> str:
        """Generate human-readable market interpretation"""
        regime_text = {
            MarketRegime.BULL_QUIET: "steady uptrend environment",
            MarketRegime.BULL_VOLATILE: "volatile but bullish conditions",
            MarketRegime.BEAR_QUIET: "grinding bear market",
            MarketRegime.BEAR_VOLATILE: "crisis-like selloff",
            MarketRegime.SIDEWAYS: "range-bound market",
            MarketRegime.RECOVERY: "potential recovery setup",
            MarketRegime.EUPHORIA: "euphoric conditions (caution warranted)",
            MarketRegime.CAPITULATION: "capitulation (potential bottom)",
        }
        
        outlook = "bullish" if avg_return > 0.02 else (
            "bearish" if avg_return < -0.02 else "neutral"
        )
        
        historical_support = (
            "strongly supports" if positive_rate > 0.7 else (
                "moderately supports" if positive_rate > 0.55 else (
                    "is mixed on" if positive_rate > 0.45 else "cautions against"
                )
            )
        )
        
        top_period = similar_periods[0] if similar_periods else None
        precedent = ""
        if top_period:
            precedent = (
                f" The most similar period was {top_period.date} "
                f"({top_period.geopolitical_context}), which saw "
                f"{top_period.forward_outcome.return_1m:+.1%} over the next month."
            )
        
        return (
            f"Current market shows {regime_text.get(regime, 'mixed conditions')}. "
            f"Historical analysis {historical_support} a {outlook} outlook, "
            f"with {positive_rate:.0%} of similar periods showing positive returns.{precedent}"
        )
    
    def _determine_stance(
        self,
        regime: MarketRegime,
        avg_return: float,
        positive_rate: float,
        volatility: float
    ) -> str:
        """Determine recommended trading stance"""
        if regime == MarketRegime.CAPITULATION and positive_rate > 0.6:
            return "aggressive_long"
        elif regime == MarketRegime.RECOVERY:
            return "aggressive_long" if avg_return > 0.05 else "cautious_long"
        elif regime == MarketRegime.BULL_QUIET and positive_rate > 0.6:
            return "cautious_long"
        elif regime == MarketRegime.EUPHORIA:
            return "reduce_exposure"
        elif regime in [MarketRegime.BEAR_VOLATILE, MarketRegime.BEAR_QUIET]:
            return "defensive" if avg_return < 0 else "neutral"
        elif positive_rate > 0.6 and avg_return > 0.02:
            return "cautious_long"
        elif positive_rate < 0.4 or avg_return < -0.02:
            return "defensive"
        else:
            return "neutral"
    
    def _identify_risks(
        self,
        regime: MarketRegime,
        similar_periods: List[HistoricalPeriodAnalysis],
        volatility: float
    ) -> List[str]:
        """Identify key risks based on context"""
        risks = []
        
        # Volatility risk
        if volatility > 0.3:
            risks.append("Extremely high volatility increases position sizing risk")
        elif volatility > 0.25:
            risks.append("Elevated volatility may cause whipsaws")
        
        # Regime-specific risks
        if regime == MarketRegime.EUPHORIA:
            risks.append("Euphoric conditions often precede corrections")
        elif regime == MarketRegime.CAPITULATION:
            risks.append("Capitulation can continue longer than expected")
        elif regime == MarketRegime.BEAR_VOLATILE:
            risks.append("Crisis conditions can escalate rapidly")
        
        # Historical outcome variance
        if similar_periods:
            returns = [p.forward_outcome.return_1m for p in similar_periods]
            return_std = np.std(returns)
            if return_std > 0.1:
                risks.append(
                    f"High outcome variance ({return_std:.1%}) in similar periods"
                )
            
            # Check for tail risks
            negative_returns = [r for r in returns if r < -0.1]
            if len(negative_returns) > len(returns) * 0.2:
                risks.append(
                    f"{len(negative_returns)/len(returns):.0%} of similar periods "
                    "saw >10% drawdowns"
                )
        
        # Add general risks if none found
        if not risks:
            risks.append("Standard market risk applies")
        
        return risks[:5]  # Limit to top 5 risks
    
    def compare_periods(
        self,
        symbol: str,
        date1: str,
        date2: str
    ) -> Dict:
        """
        Compare two historical periods.
        
        Args:
            symbol: Stock symbol
            date1: First date
            date2: Second date
        
        Returns:
            Comparison of the two periods
        """
        vector_db = self._get_vector_db(symbol)
        
        # Get data for both dates
        data1 = vector_db.collection.get(ids=[date1], include=["metadatas"])
        data2 = vector_db.collection.get(ids=[date2], include=["metadatas"])
        
        if not data1["ids"] or not data2["ids"]:
            raise ValueError("One or both dates not found in database")
        
        meta1 = data1["metadatas"][0]
        meta2 = data2["metadatas"][0]
        
        # Analyze both periods
        analysis1 = self.analyze_period(symbol, date1, 1.0, meta1)
        analysis2 = self.analyze_period(symbol, date2, 1.0, meta2)
        
        return {
            "period1": {
                "date": date1,
                "regime": analysis1.regime.value,
                "narrative": analysis1.narrative,
                "geopolitical": analysis1.geopolitical_context,
                "outcome_1m": analysis1.forward_outcome.return_1m,
                "outcome_3m": analysis1.forward_outcome.return_3m,
            },
            "period2": {
                "date": date2,
                "regime": analysis2.regime.value,
                "narrative": analysis2.narrative,
                "geopolitical": analysis2.geopolitical_context,
                "outcome_1m": analysis2.forward_outcome.return_1m,
                "outcome_3m": analysis2.forward_outcome.return_3m,
            },
            "comparison": {
                "same_regime": analysis1.regime == analysis2.regime,
                "outcome_difference_1m": (
                    analysis1.forward_outcome.return_1m - 
                    analysis2.forward_outcome.return_1m
                ),
                "volatility_difference": analysis1.volatility - analysis2.volatility,
            }
        }
