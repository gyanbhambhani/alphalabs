"""
Funds module - Fund definitions, policies, and universe resolution.
"""
from core.funds.universe import (
    UniverseSpec,
    UniverseResult,
    UniverseResolver,
    compute_universe_hash,
)
from core.funds.fund import (
    Fund,
    FundThesis,
    FundPolicy,
    PMConfig,
    RiskLimits,
    FundPortfolio,
)
from core.funds.baseline import (
    BaselineFallbackPolicy,
    BaselineFallbackHandler,
)

__all__ = [
    # Universe
    "UniverseSpec",
    "UniverseResult",
    "UniverseResolver",
    "compute_universe_hash",
    # Fund
    "Fund",
    "FundThesis",
    "FundPolicy",
    "PMConfig",
    "RiskLimits",
    "FundPortfolio",
    # Baseline
    "BaselineFallbackPolicy",
    "BaselineFallbackHandler",
]
