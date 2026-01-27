"""
GlobalMarketSnapshot - Immutable, fund-agnostic data contract.

This is the single source of truth for market data during a decision cycle.
Models may ONLY reference data from this snapshot - no hallucinations allowed.
"""
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Set, Tuple, Any
import uuid

from core.config.constants import CONSTANTS


@dataclass
class EarningsEvent:
    """Upcoming or recent earnings event."""
    symbol: str
    date: str  # ISO format
    time: str  # "pre_market", "post_market", "during_market"
    estimated_eps: Optional[float] = None
    actual_eps: Optional[float] = None
    surprise_pct: Optional[float] = None


@dataclass
class MacroRelease:
    """Recent macroeconomic data release."""
    name: str  # e.g., "CPI", "NFP", "FOMC"
    date: str  # ISO format
    actual: Optional[float] = None
    expected: Optional[float] = None
    previous: Optional[float] = None
    interpretation: str = ""


@dataclass
class NewsSummary:
    """Summarized news item."""
    headline: str
    source: str
    timestamp: str  # ISO format
    symbols: List[str] = field(default_factory=list)
    sentiment: Optional[str] = None  # "positive", "negative", "neutral"
    summary: str = ""


@dataclass
class DataQuality:
    """
    Snapshot quality flags - auto no-trade if below threshold.
    
    This ensures we don't make decisions on incomplete or stale data.
    """
    coverage_ratio: float  # % of requested symbols with complete data
    staleness_seconds: Dict[str, int] = field(default_factory=dict)  # per source
    missing_fields: Dict[str, List[str]] = field(default_factory=dict)  # per symbol
    warnings: List[str] = field(default_factory=list)
    
    def is_tradeable(self) -> bool:
        """Check if snapshot quality is sufficient for trading."""
        return self.coverage_ratio >= CONSTANTS.snapshot.MIN_COVERAGE_RATIO
    
    def to_summary(self) -> Dict[str, Any]:
        """Compact summary for storing in DecisionRecord."""
        return {
            "coverage_ratio": self.coverage_ratio,
            "max_staleness": max(self.staleness_seconds.values(), default=0),
            "warnings": self.warnings[:5],  # truncate
        }


@dataclass
class GlobalMarketSnapshot:
    """
    Immutable, fund-agnostic data contract.
    
    This is the single source of truth for a decision cycle.
    UniverseResolver filters this per fund - no circular dependency.
    
    Key principles:
    - Fund-agnostic: contains all available market data
    - Immutable: once created, cannot be modified
    - Reproducible: snapshot_id allows exact replay
    - Quality-tracked: explicit flags for data issues
    """
    snapshot_id: str
    asof_timestamp: datetime
    
    # Price data (all available symbols)
    prices: Dict[str, float] = field(default_factory=dict)
    
    # Returns by period: symbol -> {"1d": 0.02, "5d": 0.05, ...}
    returns: Dict[str, Dict[str, float]] = field(default_factory=dict)
    
    # Volatility by period: symbol -> {"5d": 0.15, "21d": 0.18, ...}
    volatility: Dict[str, Dict[str, float]] = field(default_factory=dict)
    
    # Correlations: symbol -> {other_symbol -> correlation}
    # Pre-computed for common pairs, not full pairwise
    correlations: Dict[str, Dict[str, float]] = field(default_factory=dict)
    
    # Known events (from actual data source)
    upcoming_earnings: List[EarningsEvent] = field(default_factory=list)
    recent_macro_releases: List[MacroRelease] = field(default_factory=list)
    
    # News/context (actually fetched, not assumed)
    news_summaries: List[NewsSummary] = field(default_factory=list)
    
    # Quality and coverage
    quality: DataQuality = field(default_factory=DataQuality)
    coverage_symbols: List[str] = field(default_factory=list)
    data_sources: List[str] = field(default_factory=list)
    
    def has_required_data(self, symbol: str, require_vol: bool = True) -> bool:
        """
        Check if symbol has actual data, not just listed in coverage.
        
        Use this for filtering in UniverseResolver, not coverage_symbols.
        This prevents mismatches between "coverage says yes" and "data says no".
        
        Args:
            symbol: The symbol to check
            require_vol: Whether volatility data is required (strategy-dependent)
        
        Returns:
            True if symbol has all required data fields
        """
        if symbol not in self.prices:
            return False
        if symbol not in self.returns:
            return False
        if require_vol and symbol not in self.volatility:
            return False
        return True
    
    def available_features(self) -> Set[str]:
        """
        Dynamic feature allowlist based on what's actually populated.
        
        Used for anti-hallucination validation - models can only reference
        features that exist in this set.
        
        Returns:
            Set of available feature paths (e.g., "returns.1d", "volatility.21d")
        """
        features: Set[str] = set()
        
        if self.prices:
            features.add("prices")
        
        # Check return periods
        for period in ["1d", "5d", "1m", "3m"]:
            if any(period in r for r in self.returns.values()):
                features.add(f"returns.{period}")
        
        # Check volatility periods
        for period in ["5d", "21d", "63d"]:
            if any(period in v for v in self.volatility.values()):
                features.add(f"volatility.{period}")
        
        if self.correlations:
            features.add("correlations")
        
        if self.upcoming_earnings:
            features.add("upcoming_earnings")
        
        if self.recent_macro_releases:
            features.add("recent_macro_releases")
        
        if self.news_summaries:
            features.add("news_summaries")
        
        return features
    
    def validate_or_reject(self) -> Tuple[bool, Optional[str]]:
        """
        Validate snapshot quality for trading.
        
        Returns:
            Tuple of (is_valid, rejection_reason)
            If is_valid is False, rejection_reason explains why.
        """
        if not self.quality.is_tradeable():
            return False, (
                f"Coverage {self.quality.coverage_ratio:.0%} below threshold "
                f"{CONSTANTS.snapshot.MIN_COVERAGE_RATIO:.0%}"
            )
        
        max_stale = max(self.quality.staleness_seconds.values(), default=0)
        if max_stale > CONSTANTS.snapshot.MAX_STALENESS_SECONDS:
            return False, f"Data stale by {max_stale}s (max: {CONSTANTS.snapshot.MAX_STALENESS_SECONDS}s)"
        
        return True, None
    
    def get_price(self, symbol: str) -> Optional[float]:
        """Get price for a symbol, or None if not available."""
        return self.prices.get(symbol)
    
    def get_return(self, symbol: str, period: str) -> Optional[float]:
        """Get return for a symbol and period, or None if not available."""
        if symbol not in self.returns:
            return None
        return self.returns[symbol].get(period)
    
    def get_volatility(self, symbol: str, period: str) -> Optional[float]:
        """Get volatility for a symbol and period, or None if not available."""
        if symbol not in self.volatility:
            return None
        return self.volatility[symbol].get(period)
    
    @staticmethod
    def create_empty(asof_timestamp: Optional[datetime] = None) -> "GlobalMarketSnapshot":
        """Create an empty snapshot for testing."""
        return GlobalMarketSnapshot(
            snapshot_id=str(uuid.uuid4()),
            asof_timestamp=asof_timestamp or datetime.utcnow(),
            quality=DataQuality(coverage_ratio=0.0),
        )


# Instruction for LLM prompts - consistent naming
SNAPSHOT_INSTRUCTION = """
You may ONLY reference data from the provided GlobalMarketSnapshot.
Do NOT assume or hallucinate any market facts, news, or events
not explicitly included in the snapshot.
Your features_used field MUST be a subset of snapshot.available_features().
Use consistent feature paths: returns.1d, volatility.21d, etc.
"""
