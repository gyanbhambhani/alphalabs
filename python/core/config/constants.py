"""
Centralized constants for the trading system.
All tolerances and thresholds live here.
Override via environment variables for testing.

Usage:
    from core.config.constants import CONSTANTS
    
    if coverage < CONSTANTS.snapshot.MIN_COVERAGE_RATIO:
        reject()

For tests: set env vars BEFORE importing, or mock this module.
"""
import os
from dataclasses import dataclass


def _env_float(key: str, default: float) -> float:
    """Read float from environment, fall back to default."""
    val = os.environ.get(key)
    if val is None:
        return default
    try:
        return float(val)
    except ValueError:
        return default


def _env_int(key: str, default: int) -> int:
    """Read int from environment, fall back to default."""
    val = os.environ.get(key)
    if val is None:
        return default
    try:
        return int(val)
    except ValueError:
        return default


@dataclass(frozen=True)
class SnapshotConstants:
    """GlobalMarketSnapshot quality thresholds."""
    # Minimum fraction of universe that must have complete data
    MIN_COVERAGE_RATIO: float = _env_float("SNAPSHOT_MIN_COVERAGE", 0.8)
    # Maximum data staleness in seconds before auto no-trade
    MAX_STALENESS_SECONDS: int = _env_int("SNAPSHOT_MAX_STALE_SEC", 3600)


@dataclass(frozen=True)
class ExecutionConstants:
    """Execution engine tolerances."""
    # Skip trades with weight delta smaller than this
    MIN_WEIGHT_DELTA: float = _env_float("EXEC_MIN_WEIGHT_DELTA", 0.001)
    # Tolerance for gross exposure validation
    GROSS_EXPOSURE_TOLERANCE: float = _env_float("EXEC_GROSS_TOLERANCE", 0.01)
    # Default slippage assumption in basis points
    DEFAULT_SLIPPAGE_BPS: int = _env_int("EXEC_DEFAULT_SLIPPAGE_BPS", 10)


@dataclass(frozen=True)
class RiskConstants:
    """Risk check tolerances."""
    # Tolerance for order size exceeding limit (5% over allowed)
    ORDER_SIZE_TOLERANCE: float = _env_float("RISK_ORDER_TOLERANCE", 0.05)
    # Default circuit breaker cooldown in days
    DEFAULT_COOLDOWN_DAYS: int = _env_int("RISK_COOLDOWN_DAYS", 1)
    # Emergency close slippage tolerance in bps
    EMERGENCY_CLOSE_SLIPPAGE_BPS: int = _env_int("RISK_EMERGENCY_SLIPPAGE_BPS", 50)


@dataclass(frozen=True)
class UniverseConstants:
    """Universe resolver thresholds."""
    # Minimum symbols required for a universe to be valid
    DEFAULT_MIN_SYMBOLS: int = _env_int("UNIVERSE_MIN_SYMBOLS", 5)


@dataclass(frozen=True)
class DebateConstants:
    """Debate engine parameters."""
    # Maximum number of positions PM can output
    DEFAULT_MAX_POSITIONS: int = _env_int("DEBATE_MAX_POSITIONS", 20)
    # Consensus threshold below which we consider disagreement
    CONSENSUS_THRESHOLD: float = _env_float("DEBATE_CONSENSUS_THRESHOLD", 0.3)


@dataclass(frozen=True)
class Constants:
    """
    Master constants container - constructed once at import.
    
    All values are read from environment variables at import time.
    To override for tests, set env vars before importing this module.
    """
    snapshot: SnapshotConstants = SnapshotConstants()
    execution: ExecutionConstants = ExecutionConstants()
    risk: RiskConstants = RiskConstants()
    universe: UniverseConstants = UniverseConstants()
    debate: DebateConstants = DebateConstants()


# Single instance, but values read from env at import time
CONSTANTS = Constants()
