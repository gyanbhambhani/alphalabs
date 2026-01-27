"""
External Data Provider

Hybrid approach: Facts from APIs, interpretation from LLMs.
Provides economic data, market sentiment, and geopolitical context.
"""
import yfinance as yf
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum


class SentimentLevel(Enum):
    """Market sentiment levels"""
    EXTREME_FEAR = "extreme_fear"
    FEAR = "fear"
    NEUTRAL = "neutral"
    GREED = "greed"
    EXTREME_GREED = "extreme_greed"


@dataclass
class EconomicEvent:
    """Economic calendar event"""
    date: str
    event_type: str
    description: str
    importance: str  # "high", "medium", "low"
    expected_impact: str


@dataclass
class MarketSentiment:
    """Current market sentiment indicators"""
    vix_level: float
    vix_percentile: float  # Where current VIX is vs history
    put_call_ratio: float
    breadth: float  # Advance/decline ratio
    sentiment_level: SentimentLevel
    interpretation: str


@dataclass
class EconomicContext:
    """Current economic context"""
    fed_funds_rate: float
    ten_year_yield: float
    yield_curve_spread: float  # 10Y - 2Y
    inflation_yoy: float
    gdp_growth: float
    unemployment_rate: float
    is_recession_risk: bool
    interpretation: str


@dataclass
class ExternalContext:
    """Complete external context for trading decisions"""
    timestamp: datetime
    sentiment: MarketSentiment
    economic: EconomicContext
    upcoming_events: List[EconomicEvent]
    geopolitical_summary: str
    market_narrative: str
    
    def to_prompt_context(self) -> str:
        """Format for LLM prompt"""
        events_text = ""
        for event in self.upcoming_events[:5]:
            events_text += f"  - {event.date}: {event.description} ({event.importance})\n"
        
        return f"""
## External Market Context

### Market Sentiment
- VIX: {self.sentiment.vix_level:.1f} ({self.sentiment.vix_percentile:.0%} percentile)
- Put/Call Ratio: {self.sentiment.put_call_ratio:.2f}
- Market Breadth: {self.sentiment.breadth:.2f}
- Overall Sentiment: {self.sentiment.sentiment_level.value}
- Interpretation: {self.sentiment.interpretation}

### Economic Environment
- Fed Funds Rate: {self.economic.fed_funds_rate:.2%}
- 10Y Treasury: {self.economic.ten_year_yield:.2%}
- Yield Curve (10Y-2Y): {self.economic.yield_curve_spread:.2%}
- Inflation (YoY): {self.economic.inflation_yoy:.1%}
- Recession Risk: {"HIGH" if self.economic.is_recession_risk else "LOW"}
- Interpretation: {self.economic.interpretation}

### Upcoming Events
{events_text if events_text else "  No major events scheduled"}

### Geopolitical Summary
{self.geopolitical_summary}

### Market Narrative
{self.market_narrative}
"""


# Known economic events (simplified - would use API in production)
ECONOMIC_CALENDAR = [
    EconomicEvent(
        date="FOMC Meeting",
        event_type="fed",
        description="Federal Reserve Interest Rate Decision",
        importance="high",
        expected_impact="Rates unchanged expected"
    ),
    EconomicEvent(
        date="NFP",
        event_type="employment",
        description="Non-Farm Payrolls Report",
        importance="high",
        expected_impact="Labor market data"
    ),
    EconomicEvent(
        date="CPI",
        event_type="inflation",
        description="Consumer Price Index Release",
        importance="high",
        expected_impact="Inflation trending data"
    ),
]


class ExternalDataProvider:
    """
    Provides external market data with LLM-style interpretation.
    
    Combines:
    - Real market data (VIX, yields, etc.) from yfinance
    - Calculated sentiment indicators
    - Economic context
    - LLM-generated interpretation
    """
    
    def __init__(self, llm_provider: Optional[str] = None):
        """
        Initialize the provider.
        
        Args:
            llm_provider: Optional LLM for interpretation ("openai", "anthropic")
        """
        self.llm_provider = llm_provider
        self._cache: Dict[str, Any] = {}
        self._cache_time: Optional[datetime] = None
        self._cache_duration = timedelta(minutes=15)
    
    def _get_vix_data(self) -> tuple[float, float]:
        """Get current VIX level and historical percentile"""
        try:
            vix = yf.Ticker("^VIX")
            hist = vix.history(period="2y")
            
            if len(hist) == 0:
                return 20.0, 0.5  # Default values
            
            current = hist['Close'].iloc[-1]
            percentile = (hist['Close'] < current).mean()
            
            return float(current), float(percentile)
        except Exception as e:
            print(f"Error fetching VIX: {e}")
            return 20.0, 0.5
    
    def _get_treasury_yields(self) -> tuple[float, float, float]:
        """Get 10Y, 2Y yields and spread"""
        try:
            # Using yfinance tickers for treasury yields
            ten_year = yf.Ticker("^TNX")  # 10-Year Treasury
            two_year = yf.Ticker("^IRX")   # 3-Month (2Y not directly available)
            
            ten_y_hist = ten_year.history(period="5d")
            two_y_hist = two_year.history(period="5d")
            
            if len(ten_y_hist) > 0:
                ten_y_rate = ten_y_hist['Close'].iloc[-1] / 100  # Convert to decimal
            else:
                ten_y_rate = 0.045
            
            if len(two_y_hist) > 0:
                # IRX is 13-week, so estimate 2Y
                two_y_rate = two_y_hist['Close'].iloc[-1] / 100 + 0.005
            else:
                two_y_rate = 0.04
            
            spread = ten_y_rate - two_y_rate
            
            return float(ten_y_rate), float(two_y_rate), float(spread)
        except Exception as e:
            print(f"Error fetching yields: {e}")
            return 0.045, 0.04, 0.005
    
    def _get_market_breadth(self) -> float:
        """
        Calculate market breadth using SPY sector ETFs.
        Returns ratio of advancing to declining sectors.
        """
        try:
            sectors = ['XLK', 'XLF', 'XLV', 'XLE', 'XLI', 'XLC', 'XLY', 'XLP', 'XLU', 'XLRE', 'XLB']
            advancing = 0
            declining = 0
            
            for sector in sectors:
                try:
                    ticker = yf.Ticker(sector)
                    hist = ticker.history(period="5d")
                    if len(hist) >= 2:
                        change = (hist['Close'].iloc[-1] / hist['Close'].iloc[-2]) - 1
                        if change > 0:
                            advancing += 1
                        else:
                            declining += 1
                except Exception:
                    continue
            
            if declining == 0:
                return 2.0
            return advancing / max(declining, 1)
        except Exception as e:
            print(f"Error calculating breadth: {e}")
            return 1.0
    
    def _calculate_sentiment_level(
        self,
        vix: float,
        vix_pct: float,
        breadth: float
    ) -> SentimentLevel:
        """Determine overall sentiment level"""
        # VIX-based sentiment
        if vix > 35 or vix_pct > 0.9:
            vix_sentiment = -2
        elif vix > 25 or vix_pct > 0.7:
            vix_sentiment = -1
        elif vix < 15 or vix_pct < 0.2:
            vix_sentiment = 2
        elif vix < 18 or vix_pct < 0.4:
            vix_sentiment = 1
        else:
            vix_sentiment = 0
        
        # Breadth-based sentiment
        if breadth > 1.5:
            breadth_sentiment = 2
        elif breadth > 1.2:
            breadth_sentiment = 1
        elif breadth < 0.5:
            breadth_sentiment = -2
        elif breadth < 0.8:
            breadth_sentiment = -1
        else:
            breadth_sentiment = 0
        
        # Combined score
        combined = vix_sentiment + breadth_sentiment
        
        if combined >= 3:
            return SentimentLevel.EXTREME_GREED
        elif combined >= 1:
            return SentimentLevel.GREED
        elif combined <= -3:
            return SentimentLevel.EXTREME_FEAR
        elif combined <= -1:
            return SentimentLevel.FEAR
        else:
            return SentimentLevel.NEUTRAL
    
    def _generate_sentiment_interpretation(
        self,
        vix: float,
        vix_pct: float,
        breadth: float,
        sentiment: SentimentLevel
    ) -> str:
        """Generate human-readable sentiment interpretation"""
        interpretations = {
            SentimentLevel.EXTREME_FEAR: (
                f"Extreme fear in markets. VIX at {vix:.0f} "
                f"(higher than {vix_pct:.0%} of history). "
                "This often signals potential buying opportunities, "
                "but caution is warranted as fear can persist."
            ),
            SentimentLevel.FEAR: (
                f"Elevated fear levels. VIX at {vix:.0f}. "
                "Markets are nervous but not panicking. "
                "Historically, these periods have preceded bounces."
            ),
            SentimentLevel.NEUTRAL: (
                f"Balanced sentiment. VIX at {vix:.0f}. "
                f"Market breadth at {breadth:.2f}. "
                "Neither fear nor greed dominating."
            ),
            SentimentLevel.GREED: (
                f"Greedy sentiment building. VIX at low {vix:.0f}. "
                f"Broad market participation with breadth at {breadth:.2f}. "
                "Conditions supportive but watch for complacency."
            ),
            SentimentLevel.EXTREME_GREED: (
                f"Extreme greed warning. VIX at very low {vix:.0f}. "
                "Market very complacent. Historically, these periods "
                "have preceded corrections. Consider hedging."
            ),
        }
        return interpretations.get(sentiment, "Sentiment unclear.")
    
    def _generate_economic_interpretation(
        self,
        ten_y: float,
        spread: float,
        is_inverted: bool
    ) -> str:
        """Generate economic interpretation"""
        if is_inverted:
            return (
                f"Yield curve inverted ({spread:.2%}), historically a recession indicator. "
                f"10Y yield at {ten_y:.2%}. Fed policy likely restrictive. "
                "Economic slowdown risk elevated."
            )
        elif spread < 0.5:
            return (
                f"Yield curve flat ({spread:.2%}), suggesting slowing growth expectations. "
                f"10Y yield at {ten_y:.2%}. Monitor for inversion."
            )
        else:
            return (
                f"Yield curve normal ({spread:.2%}), suggesting healthy growth expectations. "
                f"10Y yield at {ten_y:.2%}. Economic conditions supportive."
            )
    
    def _generate_market_narrative(
        self,
        sentiment: MarketSentiment,
        economic: EconomicContext
    ) -> str:
        """Generate overall market narrative"""
        # Combine sentiment and economic views
        if sentiment.sentiment_level in [SentimentLevel.EXTREME_FEAR, SentimentLevel.FEAR]:
            if economic.is_recession_risk:
                return (
                    "Markets fearful amid recession concerns. "
                    "Defensive positioning warranted, but extreme fear "
                    "often creates opportunities for long-term investors."
                )
            else:
                return (
                    "Fear elevated despite stable economic backdrop. "
                    "May represent buying opportunity if fundamentals hold."
                )
        elif sentiment.sentiment_level in [SentimentLevel.EXTREME_GREED, SentimentLevel.GREED]:
            if economic.is_recession_risk:
                return (
                    "Complacency despite economic headwinds. "
                    "Risk-reward appears unfavorable. "
                    "Consider reducing exposure or hedging."
                )
            else:
                return (
                    "Bullish sentiment aligned with economic strength. "
                    "Conditions supportive but valuations may be stretched."
                )
        else:
            return (
                "Balanced market conditions. "
                "Focus on stock selection over market timing."
            )
    
    def get_market_sentiment(self) -> MarketSentiment:
        """Get current market sentiment indicators"""
        vix, vix_pct = self._get_vix_data()
        breadth = self._get_market_breadth()
        
        # Placeholder for put/call ratio (would need options data)
        put_call = 1.0 + (vix - 20) / 50  # Estimate based on VIX
        
        sentiment_level = self._calculate_sentiment_level(vix, vix_pct, breadth)
        interpretation = self._generate_sentiment_interpretation(
            vix, vix_pct, breadth, sentiment_level
        )
        
        return MarketSentiment(
            vix_level=vix,
            vix_percentile=vix_pct,
            put_call_ratio=put_call,
            breadth=breadth,
            sentiment_level=sentiment_level,
            interpretation=interpretation
        )
    
    def get_economic_context(self) -> EconomicContext:
        """Get current economic context"""
        ten_y, two_y, spread = self._get_treasury_yields()
        
        # Estimate other metrics (would use FRED API in production)
        is_inverted = spread < 0
        is_recession_risk = is_inverted or spread < 0.2
        
        interpretation = self._generate_economic_interpretation(
            ten_y, spread, is_inverted
        )
        
        return EconomicContext(
            fed_funds_rate=0.0525,  # Would fetch from FRED
            ten_year_yield=ten_y,
            yield_curve_spread=spread,
            inflation_yoy=0.03,  # Would fetch from FRED
            gdp_growth=0.02,     # Would fetch from FRED
            unemployment_rate=0.04,  # Would fetch from FRED
            is_recession_risk=is_recession_risk,
            interpretation=interpretation
        )
    
    def get_upcoming_events(self) -> List[EconomicEvent]:
        """Get upcoming economic events"""
        # In production, would fetch from economic calendar API
        return ECONOMIC_CALENDAR[:5]
    
    def get_geopolitical_summary(self) -> str:
        """
        Get geopolitical summary.
        
        In production, would use news API + LLM summarization.
        """
        return (
            "Current geopolitical factors to consider: "
            "Global trade dynamics, central bank policy divergence, "
            "energy markets, and technology sector regulation. "
            "No immediate crisis-level events."
        )
    
    def get_full_context(self) -> ExternalContext:
        """
        Get complete external context for trading decisions.
        
        Caches data for performance.
        """
        # Check cache
        if (self._cache_time and 
            datetime.now() - self._cache_time < self._cache_duration and
            self._cache):
            return self._cache.get("context")
        
        sentiment = self.get_market_sentiment()
        economic = self.get_economic_context()
        events = self.get_upcoming_events()
        geopolitical = self.get_geopolitical_summary()
        narrative = self._generate_market_narrative(sentiment, economic)
        
        context = ExternalContext(
            timestamp=datetime.now(),
            sentiment=sentiment,
            economic=economic,
            upcoming_events=events,
            geopolitical_summary=geopolitical,
            market_narrative=narrative
        )
        
        # Update cache
        self._cache["context"] = context
        self._cache_time = datetime.now()
        
        return context
    
    def refresh(self) -> None:
        """Force refresh of cached data"""
        self._cache = {}
        self._cache_time = None
