"""
Semantic Search Engine

High-level interface for semantic market search.
Combines encoder and vector database with outcome analysis.
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta

from core.semantic.encoder import MarketStateEncoder, MarketState
from core.semantic.vector_db import VectorDatabase, SearchResult


@dataclass
class HistoricalOutcome:
    """Outcome after a similar historical period"""
    date: str
    similarity: float
    return_5d: float
    return_10d: float
    return_20d: float
    max_drawdown_5d: float
    volatility_5d: float


@dataclass
class SemanticSearchResult:
    """Complete semantic search result with outcomes"""
    query_date: str
    similar_periods: List[HistoricalOutcome]
    avg_5d_return: float
    avg_10d_return: float
    avg_20d_return: float
    positive_5d_rate: float
    positive_20d_rate: float
    interpretation: str


class SemanticSearchEngine:
    """
    High-level semantic search for market conditions.
    
    Features:
    - Encode current market state
    - Find similar historical periods
    - Analyze forward outcomes
    - Generate human-readable interpretation
    """
    
    def __init__(
        self,
        persist_directory: str = "./chroma_data",
        in_memory: bool = False
    ):
        """
        Initialize the search engine.
        
        Args:
            persist_directory: Where to store vector data
            in_memory: Use in-memory storage (for testing)
        """
        self.encoder = MarketStateEncoder()
        self.vector_db = VectorDatabase(
            persist_directory=persist_directory,
            in_memory=in_memory
        )
        self._price_data: Optional[pd.DataFrame] = None
    
    def set_price_data(self, data: pd.DataFrame) -> None:
        """
        Set the price data for outcome analysis.
        
        Args:
            data: DataFrame indexed by date with 'close' column
        """
        self._price_data = data.copy()
        if not isinstance(self._price_data.index, pd.DatetimeIndex):
            self._price_data.index = pd.to_datetime(self._price_data.index)
    
    def index_historical_data(
        self,
        data: pd.DataFrame,
        symbol: str = "SPY"
    ) -> int:
        """
        Index historical data into the vector database.
        
        Args:
            data: DataFrame with OHLCV data
            symbol: Symbol being indexed
        
        Returns:
            Number of states indexed
        """
        # Set price data for outcome analysis
        self.set_price_data(data)
        
        # Encode all historical states
        states = self.encoder.encode_batch(data, symbol)
        
        # Add to vector database
        count = self.vector_db.add_batch(states)
        
        return count
    
    def _calculate_outcome(
        self,
        date: str,
        similarity: float,
        days_forward: List[int] = [5, 10, 20]
    ) -> Optional[HistoricalOutcome]:
        """Calculate forward returns from a historical date"""
        if self._price_data is None:
            return None
        
        try:
            date_dt = pd.to_datetime(date)
            
            # Find the date in our data
            if date_dt not in self._price_data.index:
                # Find nearest date
                idx = self._price_data.index.get_indexer([date_dt], method='nearest')
                if idx[0] < 0 or idx[0] >= len(self._price_data):
                    return None
                date_dt = self._price_data.index[idx[0]]
            
            loc = self._price_data.index.get_loc(date_dt)
            start_price = self._price_data.iloc[loc]['close']
            
            # Calculate forward returns
            returns = {}
            for days in days_forward:
                end_loc = loc + days
                if end_loc < len(self._price_data):
                    end_price = self._price_data.iloc[end_loc]['close']
                    returns[days] = (end_price - start_price) / start_price
                else:
                    returns[days] = 0.0
            
            # Calculate max drawdown in 5 days
            max_dd = 0.0
            if loc + 5 < len(self._price_data):
                window = self._price_data.iloc[loc:loc+6]['close']
                peak = window.iloc[0]
                for price in window:
                    if price > peak:
                        peak = price
                    dd = (price - peak) / peak
                    if dd < max_dd:
                        max_dd = dd
            
            # Calculate 5-day volatility
            vol_5d = 0.0
            if loc + 5 < len(self._price_data):
                window = self._price_data.iloc[loc:loc+6]['close']
                log_returns = np.log(window / window.shift(1)).dropna()
                vol_5d = log_returns.std() * np.sqrt(252)
            
            return HistoricalOutcome(
                date=date,
                similarity=similarity,
                return_5d=returns.get(5, 0.0),
                return_10d=returns.get(10, 0.0),
                return_20d=returns.get(20, 0.0),
                max_drawdown_5d=max_dd,
                volatility_5d=vol_5d
            )
        
        except Exception as e:
            print(f"Error calculating outcome for {date}: {e}")
            return None
    
    def _generate_interpretation(
        self,
        outcomes: List[HistoricalOutcome],
        current_state: MarketState
    ) -> str:
        """Generate human-readable interpretation"""
        if not outcomes:
            return "Insufficient historical data for interpretation."
        
        n = len(outcomes)
        avg_5d = np.mean([o.return_5d for o in outcomes])
        avg_20d = np.mean([o.return_20d for o in outcomes])
        positive_rate = sum(1 for o in outcomes if o.return_5d > 0) / n
        
        # Get regime info from metadata
        metadata = current_state.metadata
        returns = metadata.get('returns', {})
        vol = metadata.get('volatility', {})
        
        # Build interpretation
        parts = []
        
        # Describe current conditions
        mom_1m = returns.get('1m', 0)
        mom_3m = returns.get('3m', 0)
        vol_21d = vol.get('21d', 0)
        
        if vol_21d < 0.15:
            vol_desc = "low volatility"
        elif vol_21d > 0.25:
            vol_desc = "high volatility"
        else:
            vol_desc = "moderate volatility"
        
        if mom_1m > 0.02 and mom_3m > 0:
            trend_desc = "uptrending"
        elif mom_1m < -0.02 and mom_3m < 0:
            trend_desc = "downtrending"
        else:
            trend_desc = "ranging"
        
        parts.append(
            f"Current market conditions: {vol_desc}, {trend_desc} market."
        )
        
        # Historical outcomes
        parts.append(
            f"Found {n} similar historical periods."
        )
        
        if positive_rate > 0.65:
            outlook = "bullish"
        elif positive_rate < 0.35:
            outlook = "bearish"
        else:
            outlook = "mixed"
        
        parts.append(
            f"Historical outlook: {outlook} "
            f"({positive_rate:.0%} positive 5-day outcomes)."
        )
        
        parts.append(
            f"Average returns: {avg_5d:+.1%} (5-day), {avg_20d:+.1%} (20-day)."
        )
        
        return " ".join(parts)
    
    def search(
        self,
        close: pd.Series,
        high: Optional[pd.Series] = None,
        low: Optional[pd.Series] = None,
        volume: Optional[pd.Series] = None,
        top_k: int = 50,
        exclude_recent_days: int = 21
    ) -> SemanticSearchResult:
        """
        Search for similar historical market conditions.
        
        Args:
            close: Recent close prices
            high: Recent high prices
            low: Recent low prices
            volume: Recent volume
            top_k: Number of similar periods to return
            exclude_recent_days: Exclude this many recent days from results
        
        Returns:
            SemanticSearchResult with similar periods and analysis
        """
        # Encode current state
        current_date = (
            str(close.index[-1].date()) 
            if hasattr(close.index[-1], 'date') 
            else datetime.now().strftime('%Y-%m-%d')
        )
        
        current_state = self.encoder.encode(
            date=current_date,
            close=close,
            high=high,
            low=low,
            volume=volume
        )
        
        # Search for similar periods
        search_results = self.vector_db.search(
            current_state.vector,
            top_k=top_k + exclude_recent_days  # Get extra to filter
        )
        
        # Filter out recent dates and calculate outcomes
        outcomes = []
        cutoff_date = (
            datetime.now() - timedelta(days=exclude_recent_days)
        ).strftime('%Y-%m-%d')
        
        for result in search_results:
            if result.date >= cutoff_date:
                continue
            
            outcome = self._calculate_outcome(
                result.date,
                result.similarity
            )
            if outcome:
                outcomes.append(outcome)
            
            if len(outcomes) >= top_k:
                break
        
        # Calculate aggregates
        if outcomes:
            avg_5d = np.mean([o.return_5d for o in outcomes])
            avg_10d = np.mean([o.return_10d for o in outcomes])
            avg_20d = np.mean([o.return_20d for o in outcomes])
            pos_5d = sum(1 for o in outcomes if o.return_5d > 0) / len(outcomes)
            pos_20d = sum(1 for o in outcomes if o.return_20d > 0) / len(outcomes)
        else:
            avg_5d = avg_10d = avg_20d = 0.0
            pos_5d = pos_20d = 0.5
        
        interpretation = self._generate_interpretation(outcomes, current_state)
        
        return SemanticSearchResult(
            query_date=current_date,
            similar_periods=outcomes,
            avg_5d_return=avg_5d,
            avg_10d_return=avg_10d,
            avg_20d_return=avg_20d,
            positive_5d_rate=pos_5d,
            positive_20d_rate=pos_20d,
            interpretation=interpretation
        )
    
    def to_dict(self, result: SemanticSearchResult) -> Dict:
        """Convert search result to dictionary for API response"""
        return {
            "query_date": result.query_date,
            "similar_periods": [
                {
                    "date": o.date,
                    "similarity": o.similarity,
                    "return_5d": o.return_5d,
                    "return_20d": o.return_20d
                }
                for o in result.similar_periods[:10]  # Top 10 for API
            ],
            "avg_5d_return": result.avg_5d_return,
            "avg_20d_return": result.avg_20d_return,
            "positive_5d_rate": result.positive_5d_rate,
            "interpretation": result.interpretation
        }
