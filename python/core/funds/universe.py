"""
Universe Resolver - Resolves universe specs against market snapshots.

Key principle: Filter by actual data availability (has_required_data),
not just coverage_symbols. This prevents mismatches.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, TYPE_CHECKING
import hashlib

from core.config.constants import CONSTANTS

if TYPE_CHECKING:
    from core.data.snapshot import GlobalMarketSnapshot


@dataclass
class UniverseSpec:
    """
    Tag-based universe definition (not brittle symbol lists).
    
    Types:
    - "etf_set": Pre-defined ETF sets like "liquid_macro", "tech_leaders"
    - "screen": Dynamic screen with parameters like min_adv, market_cap
    - "explicit": Explicit list of symbols
    
    Examples:
        UniverseSpec(type="etf_set", params={"name": "liquid_macro"})
        UniverseSpec(type="screen", params={"min_adv": 20_000_000, "sector": "tech"})
        UniverseSpec(type="explicit", params={"symbols": ["AAPL", "MSFT", "GOOGL"]})
    """
    type: str  # "etf_set", "screen", "explicit"
    params: Dict[str, Any] = field(default_factory=dict)
    min_symbols: int = CONSTANTS.universe.DEFAULT_MIN_SYMBOLS


@dataclass
class UniverseResult:
    """
    Rich result with metadata for audit trail.
    
    Store universe_hash in DecisionRecord for replay determinism.
    """
    success: bool
    symbols: List[str]
    
    # Metadata
    coverage_ratio: float  # % of spec that has data
    missing_symbols: List[str]  # symbols in spec but not in snapshot
    notes: List[str] = field(default_factory=list)
    
    # For replay determinism
    universe_hash: str = ""
    
    error: Optional[str] = None
    
    def to_summary(self) -> Dict[str, Any]:
        """Compact summary for storing in DecisionRecord."""
        return {
            "coverage_ratio": self.coverage_ratio,
            "missing_symbols_count": len(self.missing_symbols),
            "notes": self.notes[:3],  # truncate
            "universe_hash": self.universe_hash,
        }


def compute_universe_hash(symbols: List[str]) -> str:
    """
    Deterministic hash of symbol list.
    
    Used to detect if two decisions had the same universe,
    making replays and comparisons sane.
    """
    payload = ",".join(sorted(symbols))
    return hashlib.sha256(payload.encode()).hexdigest()[:12]


# Pre-defined ETF sets
ETF_SETS: Dict[str, List[str]] = {
    "liquid_macro": ["SPY", "QQQ", "IWM", "DIA", "TLT", "GLD", "SLV", "USO", "UNG"],
    "tech_leaders": ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA"],
    "sector_etfs": [
        "XLK", "XLF", "XLE", "XLV", "XLI", "XLP", "XLU", "XLB", "XLRE", "XLC", "XLY"
    ],
    "broad_market": ["SPY", "QQQ", "IWM", "VTI", "VOO"],
}


class UniverseResolver:
    """
    Resolves universe specs against a global snapshot.
    
    Key principle: Filter by actual data availability using
    snapshot.has_required_data(), not just coverage_symbols.
    This prevents mismatches between "coverage says yes" and "data says no".
    """
    
    def __init__(self, custom_etf_sets: Optional[Dict[str, List[str]]] = None):
        """
        Initialize resolver with optional custom ETF sets.
        
        Args:
            custom_etf_sets: Additional ETF sets to merge with defaults
        """
        self.etf_sets = ETF_SETS.copy()
        if custom_etf_sets:
            self.etf_sets.update(custom_etf_sets)
    
    def resolve(
        self,
        spec: UniverseSpec,
        snapshot: "GlobalMarketSnapshot",
        require_vol: bool = True
    ) -> UniverseResult:
        """
        Resolve universe spec against snapshot.
        
        Args:
            spec: Universe specification
            snapshot: Market snapshot to filter against
            require_vol: Whether volatility data is required (strategy-dependent)
        
        Returns:
            UniverseResult with symbols that have actual data
        """
        # Get candidate symbols based on spec type
        if spec.type == "etf_set":
            candidates = self._get_etf_set(spec.params.get("name", ""))
        elif spec.type == "screen":
            candidates = self._apply_screen(spec.params, snapshot)
        elif spec.type == "explicit":
            candidates = spec.params.get("symbols", [])
        else:
            return UniverseResult(
                success=False,
                symbols=[],
                coverage_ratio=0.0,
                missing_symbols=[],
                error=f"Unknown universe type: {spec.type}"
            )
        
        if not candidates:
            return UniverseResult(
                success=False,
                symbols=[],
                coverage_ratio=0.0,
                missing_symbols=[],
                error=f"No candidates found for spec: {spec}"
            )
        
        # Filter by actual data availability, NOT just coverage_symbols
        covered = [
            s for s in candidates
            if snapshot.has_required_data(s, require_vol=require_vol)
        ]
        missing = [
            s for s in candidates
            if not snapshot.has_required_data(s, require_vol=require_vol)
        ]
        
        coverage_ratio = len(covered) / len(candidates) if candidates else 0.0
        
        notes: List[str] = []
        if missing:
            notes.append(f"{len(missing)} symbols missing required data")
        
        # Check minimum symbols requirement
        if len(covered) < spec.min_symbols:
            return UniverseResult(
                success=False,
                symbols=[],
                coverage_ratio=coverage_ratio,
                missing_symbols=missing,
                notes=notes,
                universe_hash="",
                error=(
                    f"Only {len(covered)} symbols have data, "
                    f"need {spec.min_symbols}"
                )
            )
        
        return UniverseResult(
            success=True,
            symbols=covered,
            coverage_ratio=coverage_ratio,
            missing_symbols=missing,
            notes=notes,
            universe_hash=compute_universe_hash(covered)
        )
    
    def _get_etf_set(self, name: str) -> List[str]:
        """Get symbols for a named ETF set."""
        if name not in self.etf_sets:
            return []
        return self.etf_sets[name].copy()
    
    def _apply_screen(
        self,
        params: Dict[str, Any],
        snapshot: "GlobalMarketSnapshot"
    ) -> List[str]:
        """
        Apply screening criteria to snapshot symbols.
        
        For v1, this is a simple filter. Can be extended with:
        - min_adv: Minimum average daily volume
        - market_cap: Market cap filter
        - sector: Sector filter
        - options_liquid: Options liquidity requirement
        """
        # Start with all symbols that have prices
        candidates = list(snapshot.prices.keys())
        
        # Apply filters based on params
        min_price = params.get("min_price")
        if min_price is not None:
            candidates = [
                s for s in candidates
                if snapshot.prices.get(s, 0) >= min_price
            ]
        
        max_price = params.get("max_price")
        if max_price is not None:
            candidates = [
                s for s in candidates
                if snapshot.prices.get(s, float("inf")) <= max_price
            ]
        
        # For v1, we don't have ADV or market cap in snapshot
        # These would be added as snapshot fields in future versions
        
        return candidates
    
    def add_etf_set(self, name: str, symbols: List[str]) -> None:
        """Add or update an ETF set."""
        self.etf_sets[name] = symbols.copy()
