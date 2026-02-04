"""
Feature Pack Builder

Builds complete feature vectors for ML training from market snapshots.

Extracts:
- Price data
- Returns (multiple horizons)
- Volatility (multiple windows)
- Technical indicators (RSI, z-score, etc.)
- Cross-sectional ranks

Output format suitable for:
- Decision candidates logging
- Experience replay
- Supervised learning
"""

from typing import Dict, List, Optional, Set
import numpy as np
import logging

from core.data.snapshot import GlobalMarketSnapshot

logger = logging.getLogger(__name__)


class FeaturePackBuilder:
    """
    Builds complete feature vectors from market snapshots.
    
    All features are numeric and available at decision time.
    """
    
    # Standard feature set
    RETURN_PERIODS = ["1d", "5d", "21d", "63d", "252d"]
    VOL_PERIODS = ["5d", "21d", "63d"]
    
    def __init__(self):
        """Initialize feature pack builder."""
        self.feature_names: Set[str] = set()
    
    def build_feature_pack(
        self,
        snapshot: GlobalMarketSnapshot,
        symbol: str,
    ) -> Dict[str, float]:
        """
        Build complete feature vector for a single asset.
        
        Args:
            snapshot: Market snapshot at decision time
            symbol: Asset symbol
            
        Returns:
            Dict of feature_name -> value
        """
        features = {}
        
        # Price
        price = snapshot.get_price(symbol)
        if price:
            features["price"] = float(price)
        
        # Returns
        for period in self.RETURN_PERIODS:
            ret = snapshot.get_return(symbol, period)
            if ret is not None:
                features[f"return_{period}"] = float(ret)
        
        # Volatility
        for period in self.VOL_PERIODS:
            vol = snapshot.get_volatility(symbol, period)
            if vol is not None:
                features[f"volatility_{period}"] = float(vol)
        
        # Track feature names
        self.feature_names.update(features.keys())
        
        return features
    
    def build_candidate_set(
        self,
        snapshot: GlobalMarketSnapshot,
        symbols: List[str],
        max_candidates: int = 50,
    ) -> List[Dict]:
        """
        Build feature packs for multiple candidates.
        
        Args:
            snapshot: Market snapshot
            symbols: List of symbols to consider
            max_candidates: Maximum number to return
            
        Returns:
            List of candidate dicts with symbol and features
        """
        candidates = []
        
        for symbol in symbols[:max_candidates]:
            features = self.build_feature_pack(snapshot, symbol)
            
            if features:  # Only include if we have data
                candidates.append({
                    "symbol": symbol,
                    "features": features,
                    "selected": False,  # Will be updated if chosen
                })
        
        return candidates
    
    def compute_cross_sectional_ranks(
        self,
        candidates: List[Dict],
        feature_name: str = "return_252d",
    ) -> None:
        """
        Compute cross-sectional percentile ranks for a feature.
        
        Modifies candidates in-place, adding {feature_name}_rank_pct.
        
        Args:
            candidates: List of candidate dicts
            feature_name: Feature to rank on
        """
        # Extract values
        values = []
        indices = []
        
        for i, candidate in enumerate(candidates):
            val = candidate["features"].get(feature_name)
            if val is not None:
                values.append(val)
                indices.append(i)
        
        if not values:
            return
        
        # Compute percentile ranks
        sorted_indices = np.argsort(values)
        ranks = np.empty_like(sorted_indices)
        ranks[sorted_indices] = np.arange(len(values))
        
        # Convert to percentiles (0-1)
        pct_ranks = ranks / (len(values) - 1) if len(values) > 1 else [0.5] * len(values)
        
        # Assign back to candidates
        for i, pct_rank in zip(indices, pct_ranks):
            rank_feature = f"{feature_name}_rank_pct"
            candidates[i]["features"][rank_feature] = float(pct_rank)
            self.feature_names.add(rank_feature)
    
    def normalize_features(
        self,
        features: Dict[str, float],
    ) -> np.ndarray:
        """
        Normalize features to [0, 1] range for similarity search.
        
        Simple min-max normalization (assumes reasonable ranges).
        
        Args:
            features: Feature dict
            
        Returns:
            Normalized feature vector as numpy array
        """
        # Feature normalization ranges (approximate)
        ranges = {
            "price": (0, 500),  # Arbitrary, varies widely
            "return_1d": (-0.10, 0.10),  # ±10%
            "return_5d": (-0.20, 0.20),
            "return_21d": (-0.40, 0.40),
            "return_63d": (-0.60, 0.60),
            "return_252d": (-0.80, 0.80),
            "volatility_5d": (0, 0.10),  # 0-10%
            "volatility_21d": (0, 0.10),
            "volatility_63d": (0, 0.10),
        }
        
        normalized = []
        
        for feature_name in sorted(features.keys()):
            value = features[feature_name]
            
            if feature_name in ranges:
                min_val, max_val = ranges[feature_name]
                norm_val = (value - min_val) / (max_val - min_val)
                norm_val = np.clip(norm_val, 0, 1)
            else:
                # Unknown feature - just clip
                norm_val = np.clip(value, -1, 1)
            
            normalized.append(norm_val)
        
        return np.array(normalized)
    
    def get_feature_names(self) -> List[str]:
        """Get list of all feature names seen."""
        return sorted(list(self.feature_names))


# Singleton instance
_feature_builder: Optional[FeaturePackBuilder] = None


def get_feature_builder() -> FeaturePackBuilder:
    """Get global feature builder instance."""
    global _feature_builder
    if _feature_builder is None:
        _feature_builder = FeaturePackBuilder()
    return _feature_builder
