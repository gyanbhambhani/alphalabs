"""
Fund Model - Fund definitions, policies, and configuration.

A Fund consists of:
- FundThesis: What we believe and target
- FundPolicy: How we turn ideas into allocations
- PMConfig: Fund-specific PM Finalizer configuration
- RiskLimits: Risk constraints
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
import hashlib

from core.funds.universe import UniverseSpec
from core.config.constants import CONSTANTS


@dataclass
class RiskLimits:
    """
    Risk limits for a fund.
    
    Includes both position limits and circuit breakers.
    """
    # Position limits
    max_position_pct: float = 0.15  # 15% max in single position
    max_sector_pct: float = 0.30  # 30% max in single sector
    
    # Exposure limits (works for long-short)
    max_gross_exposure: float = 1.0  # e.g., 1.5 for 150% gross
    
    # Circuit breakers (HARD STOPS)
    max_daily_loss_pct: float = 0.03  # -3% -> halt trading
    max_weekly_drawdown_pct: float = 0.07  # -7% -> risk-off mode
    max_realized_vol_multiple: float = 2.0  # if vol > 2x target -> reduce
    
    # Breach actions
    breach_action: str = "halt"  # "halt", "reduce_50pct", "go_flat"
    breach_cooldown_days: int = 1


@dataclass
class FundThesis:
    """
    What the fund believes and targets.
    
    This defines the investment strategy and edge.
    """
    name: str
    strategy: str  # "trend_macro", "mean_reversion", "event_driven", "quality_ls"
    description: str
    horizon_days: Tuple[int, int]  # (min, max) holding period
    universe_spec: UniverseSpec
    edge: str  # Description of the fund's edge
    version: str = "1.0"  # For inputs_hash tracking
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON storage."""
        return {
            "name": self.name,
            "strategy": self.strategy,
            "description": self.description,
            "horizon_days": list(self.horizon_days),
            "universe_spec": {
                "type": self.universe_spec.type,
                "params": self.universe_spec.params,
                "min_symbols": self.universe_spec.min_symbols,
            },
            "edge": self.edge,
            "version": self.version,
        }


@dataclass
class FundPolicy:
    """
    How the fund turns ideas into allocations.
    
    This defines sizing, turnover, rebalance cadence, and exit rules.
    """
    # Sizing
    sizing_method: str  # "vol_target", "kelly_fraction", "equal_risk", "fixed_pct"
    vol_target: Optional[float] = None  # e.g., 0.10 for 10% annualized
    max_position_pct: float = 0.15  # Max weight per position
    max_turnover_daily: float = 0.25  # Max portfolio turnover per day
    rebalance_cadence: str = "daily"  # "continuous", "daily", "weekly"
    
    # Position limits
    max_positions: int = CONSTANTS.debate.DEFAULT_MAX_POSITIONS
    
    # Entry/exit templates
    default_stop_loss_pct: float = 0.05  # 5% stop loss
    default_take_profit_pct: float = 0.15  # 15% take profit
    trailing_stop: bool = False
    
    # Exposure limits (works for long-short)
    max_gross_exposure: float = 1.0  # e.g., 1.5 for 150% gross
    min_cash_buffer: float = 0.05  # Minimum 5% cash
    
    # Risk-off triggers
    go_flat_on_circuit_breaker: bool = True
    
    # Version for inputs_hash tracking
    version: str = "1.0"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON storage."""
        return {
            "sizing_method": self.sizing_method,
            "vol_target": self.vol_target,
            "max_position_pct": self.max_position_pct,
            "max_turnover_daily": self.max_turnover_daily,
            "rebalance_cadence": self.rebalance_cadence,
            "max_positions": self.max_positions,
            "default_stop_loss_pct": self.default_stop_loss_pct,
            "default_take_profit_pct": self.default_take_profit_pct,
            "trailing_stop": self.trailing_stop,
            "max_gross_exposure": self.max_gross_exposure,
            "min_cash_buffer": self.min_cash_buffer,
            "go_flat_on_circuit_breaker": self.go_flat_on_circuit_breaker,
            "version": self.version,
        }


@dataclass
class PMConfig:
    """
    Fund-specific PM Finalizer configuration.
    
    Different funds need different PM prompts.
    Trend/Macro PM prompt is different from Event-driven PM prompt.
    """
    model_provider: str  # "openai", "anthropic", "google"
    model_name: str  # e.g., "gpt-4-turbo-preview", "claude-3-5-sonnet"
    prompt_template: str  # Fund-specific instructions
    prompt_hash: str = ""  # Computed from template, included in inputs_hash
    temperature: float = 0.3
    
    def __post_init__(self):
        """Compute prompt hash if not provided."""
        if not self.prompt_hash and self.prompt_template:
            self.prompt_hash = hashlib.sha256(
                self.prompt_template.encode()
            ).hexdigest()[:12]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON storage."""
        return {
            "model_provider": self.model_provider,
            "model_name": self.model_name,
            "prompt_template": self.prompt_template,
            "prompt_hash": self.prompt_hash,
            "temperature": self.temperature,
        }


@dataclass
class Position:
    """Current position in a security."""
    symbol: str
    quantity: float
    avg_entry_price: float
    current_price: float = 0.0
    
    @property
    def market_value(self) -> float:
        """Current market value of position."""
        return self.quantity * self.current_price
    
    @property
    def cost_basis(self) -> float:
        """Total cost basis of position."""
        return self.quantity * self.avg_entry_price
    
    @property
    def unrealized_pnl(self) -> float:
        """Unrealized P&L."""
        return self.market_value - self.cost_basis
    
    @property
    def unrealized_pnl_pct(self) -> float:
        """Unrealized P&L as percentage."""
        if self.cost_basis == 0:
            return 0.0
        return self.unrealized_pnl / abs(self.cost_basis)


@dataclass
class FundPortfolio:
    """Portfolio state for a fund."""
    cash_balance: float = 0.0
    positions: Dict[str, Position] = field(default_factory=dict)
    
    @property
    def total_value(self) -> float:
        """Total portfolio value (cash + positions)."""
        positions_value = sum(p.market_value for p in self.positions.values())
        return self.cash_balance + positions_value
    
    @property
    def gross_exposure(self) -> float:
        """Gross exposure as fraction of total value."""
        if self.total_value == 0:
            return 0.0
        gross = sum(abs(p.market_value) for p in self.positions.values())
        return gross / self.total_value
    
    @property
    def net_exposure(self) -> float:
        """Net exposure as fraction of total value."""
        if self.total_value == 0:
            return 0.0
        net = sum(p.market_value for p in self.positions.values())
        return net / self.total_value
    
    @property
    def cash_pct(self) -> float:
        """Cash as fraction of total value."""
        if self.total_value == 0:
            return 1.0
        return self.cash_balance / self.total_value
    
    def get_weight(self, symbol: str, basis: str = "nav") -> float:
        """
        Get weight of a position.
        
        Args:
            symbol: Symbol to get weight for
            basis: "nav" for NAV-based weight
        
        Returns:
            Weight as fraction (can be negative for shorts)
        """
        if symbol not in self.positions:
            return 0.0
        if self.total_value == 0:
            return 0.0
        return self.positions[symbol].market_value / self.total_value


@dataclass
class Fund:
    """
    A Fund with thesis, policy, and configuration.
    
    This is the main entity that participates in the trading system.
    """
    fund_id: str
    thesis: FundThesis
    policy: FundPolicy
    pm_config: PMConfig
    risk_limits: RiskLimits
    portfolio: FundPortfolio = field(default_factory=FundPortfolio)
    is_active: bool = True
    created_at: datetime = field(default_factory=datetime.utcnow)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON storage."""
        return {
            "fund_id": self.fund_id,
            "thesis": self.thesis.to_dict(),
            "policy": self.policy.to_dict(),
            "pm_config": self.pm_config.to_dict(),
            "risk_limits": {
                "max_position_pct": self.risk_limits.max_position_pct,
                "max_sector_pct": self.risk_limits.max_sector_pct,
                "max_gross_exposure": self.risk_limits.max_gross_exposure,
                "max_daily_loss_pct": self.risk_limits.max_daily_loss_pct,
                "max_weekly_drawdown_pct": self.risk_limits.max_weekly_drawdown_pct,
                "breach_action": self.risk_limits.breach_action,
                "breach_cooldown_days": self.risk_limits.breach_cooldown_days,
            },
            "is_active": self.is_active,
        }


# Factory functions for creating common fund types
def create_trend_macro_fund(
    fund_id: str = "trend_macro",
    initial_capital: float = 100000.0,
) -> Fund:
    """Create a Trend + Macro fund."""
    return Fund(
        fund_id=fund_id,
        thesis=FundThesis(
            name="Trend + Macro Fund",
            strategy="trend_macro",
            description="Regime detection with trend following",
            horizon_days=(1, 20),
            universe_spec=UniverseSpec(
                type="etf_set",
                params={"name": "liquid_macro"},
            ),
            edge="Regime + trend + vol targeting",
        ),
        policy=FundPolicy(
            sizing_method="vol_target",
            vol_target=0.10,
            max_position_pct=0.20,
            max_gross_exposure=1.0,
        ),
        pm_config=PMConfig(
            model_provider="openai",
            model_name="gpt-4-turbo-preview",
            prompt_template="You are managing a trend-following macro fund...",
        ),
        risk_limits=RiskLimits(
            max_position_pct=0.20,
            max_gross_exposure=1.0,
        ),
        portfolio=FundPortfolio(cash_balance=initial_capital),
    )


def create_mean_reversion_fund(
    fund_id: str = "mean_reversion",
    initial_capital: float = 100000.0,
) -> Fund:
    """Create a Mean Reversion fund."""
    return Fund(
        fund_id=fund_id,
        thesis=FundThesis(
            name="Mean Reversion Fund",
            strategy="mean_reversion",
            description="Exploit overreactions in liquid equities",
            horizon_days=(0, 3),  # Intraday to 3 days
            universe_spec=UniverseSpec(
                type="etf_set",
                params={"name": "tech_leaders"},
            ),
            edge="Microstructure signals, vol spikes",
        ),
        policy=FundPolicy(
            sizing_method="equal_risk",
            max_position_pct=0.15,
            max_gross_exposure=1.0,
        ),
        pm_config=PMConfig(
            model_provider="anthropic",
            model_name="claude-3-5-sonnet-20241022",
            prompt_template="You are managing a mean-reversion fund...",
        ),
        risk_limits=RiskLimits(
            max_position_pct=0.15,
            max_gross_exposure=1.0,
        ),
        portfolio=FundPortfolio(cash_balance=initial_capital),
    )


def create_quality_ls_fund(
    fund_id: str = "quality_ls",
    initial_capital: float = 100000.0,
) -> Fund:
    """Create a Quality Long-Short fund."""
    return Fund(
        fund_id=fund_id,
        thesis=FundThesis(
            name="Quality Long-Short Fund",
            strategy="quality_ls",
            description="Fundamentals-driven long-short",
            horizon_days=(20, 90),  # Weeks to months
            universe_spec=UniverseSpec(
                type="etf_set",
                params={"name": "broad_market"},
            ),
            edge="Quality factors + valuation",
        ),
        policy=FundPolicy(
            sizing_method="vol_target",
            vol_target=0.12,
            max_position_pct=0.10,
            max_gross_exposure=1.5,  # Long-short can have higher gross
            min_cash_buffer=0.10,
        ),
        pm_config=PMConfig(
            model_provider="openai",
            model_name="gpt-4-turbo-preview",
            prompt_template="You are managing a quality long-short fund...",
        ),
        risk_limits=RiskLimits(
            max_position_pct=0.10,
            max_gross_exposure=1.5,
        ),
        portfolio=FundPortfolio(cash_balance=initial_capital),
    )
