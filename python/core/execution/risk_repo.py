"""
Fund Risk State Repository - Storage for risk state (cooldowns, P&L tracking).

Key principles:
- Abstract interface for swappable implementations
- In-memory for backtests (fast, ephemeral)
- DB-backed for live trading (persistent)
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Optional


@dataclass
class FundRiskState:
    """
    Risk state for a fund.
    
    Tracks cooldowns, P&L, and breach history.
    """
    fund_id: str
    
    # Cooldown tracking
    risk_off_until: Optional[datetime] = None
    last_breach_reason: Optional[str] = None
    last_breach_time: Optional[datetime] = None
    
    # P&L tracking (for circuit breakers)
    current_daily_pnl_pct: float = 0.0
    current_weekly_drawdown_pct: float = 0.0
    peak_nav: float = 0.0
    
    def is_in_cooldown(self) -> bool:
        """Check if fund is currently in cooldown."""
        if self.risk_off_until is None:
            return False
        return datetime.utcnow() < self.risk_off_until
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON storage."""
        return {
            "fund_id": self.fund_id,
            "risk_off_until": (
                self.risk_off_until.isoformat() if self.risk_off_until else None
            ),
            "last_breach_reason": self.last_breach_reason,
            "last_breach_time": (
                self.last_breach_time.isoformat() if self.last_breach_time else None
            ),
            "current_daily_pnl_pct": self.current_daily_pnl_pct,
            "current_weekly_drawdown_pct": self.current_weekly_drawdown_pct,
            "peak_nav": self.peak_nav,
        }


class FundRiskStateRepo(ABC):
    """
    Abstract repository for fund risk state.
    
    Implementations:
    - InMemoryFundRiskStateRepo: For backtests (fast, ephemeral)
    - DBFundRiskStateRepo: For live trading (persistent)
    """
    
    @abstractmethod
    def get(self, fund_id: str) -> Optional[FundRiskState]:
        """Get risk state for a fund."""
        pass
    
    @abstractmethod
    def upsert(self, state: FundRiskState) -> None:
        """Create or update risk state."""
        pass
    
    @abstractmethod
    def clear(self, fund_id: str) -> None:
        """Clear risk state (useful for tests)."""
        pass
    
    @abstractmethod
    def clear_all(self) -> None:
        """Clear all risk states (useful for tests)."""
        pass


class InMemoryFundRiskStateRepo(FundRiskStateRepo):
    """
    In-memory risk state storage.
    
    Fast and ephemeral - perfect for backtests.
    State is lost when the process ends.
    """
    
    def __init__(self):
        self._states: Dict[str, FundRiskState] = {}
    
    def get(self, fund_id: str) -> Optional[FundRiskState]:
        """Get risk state for a fund."""
        return self._states.get(fund_id)
    
    def upsert(self, state: FundRiskState) -> None:
        """Create or update risk state."""
        self._states[state.fund_id] = state
    
    def clear(self, fund_id: str) -> None:
        """Clear risk state for a fund."""
        self._states.pop(fund_id, None)
    
    def clear_all(self) -> None:
        """Clear all risk states."""
        self._states.clear()


class DBFundRiskStateRepo(FundRiskStateRepo):
    """
    Database-backed risk state storage.
    
    Persistent - survives process restarts.
    Use for live trading where state matters.
    """
    
    def __init__(self, session_factory):
        """
        Initialize with SQLAlchemy session factory.
        
        Args:
            session_factory: Callable that returns a session
        """
        self._session_factory = session_factory
    
    def get(self, fund_id: str) -> Optional[FundRiskState]:
        """Get risk state for a fund."""
        # Import here to avoid circular dependency
        from db.models import FundRiskStateModel
        
        with self._session_factory() as session:
            record = session.query(FundRiskStateModel).filter(
                FundRiskStateModel.fund_id == fund_id
            ).first()
            
            if not record:
                return None
            
            return FundRiskState(
                fund_id=record.fund_id,
                risk_off_until=record.risk_off_until,
                last_breach_reason=record.last_breach_reason,
                last_breach_time=record.last_breach_time,
                current_daily_pnl_pct=record.current_daily_pnl_pct or 0.0,
                current_weekly_drawdown_pct=record.current_weekly_drawdown_pct or 0.0,
                peak_nav=record.peak_nav or 0.0,
            )
    
    def upsert(self, state: FundRiskState) -> None:
        """Create or update risk state."""
        from db.models import FundRiskStateModel
        
        with self._session_factory() as session:
            record = session.query(FundRiskStateModel).filter(
                FundRiskStateModel.fund_id == state.fund_id
            ).first()
            
            if record:
                record.risk_off_until = state.risk_off_until
                record.last_breach_reason = state.last_breach_reason
                record.last_breach_time = state.last_breach_time
                record.current_daily_pnl_pct = state.current_daily_pnl_pct
                record.current_weekly_drawdown_pct = state.current_weekly_drawdown_pct
                record.peak_nav = state.peak_nav
            else:
                record = FundRiskStateModel(
                    fund_id=state.fund_id,
                    risk_off_until=state.risk_off_until,
                    last_breach_reason=state.last_breach_reason,
                    last_breach_time=state.last_breach_time,
                    current_daily_pnl_pct=state.current_daily_pnl_pct,
                    current_weekly_drawdown_pct=state.current_weekly_drawdown_pct,
                    peak_nav=state.peak_nav,
                )
                session.add(record)
            
            session.commit()
    
    def clear(self, fund_id: str) -> None:
        """Clear risk state for a fund."""
        from db.models import FundRiskStateModel
        
        with self._session_factory() as session:
            session.query(FundRiskStateModel).filter(
                FundRiskStateModel.fund_id == fund_id
            ).delete()
            session.commit()
    
    def clear_all(self) -> None:
        """Clear all risk states."""
        from db.models import FundRiskStateModel
        
        with self._session_factory() as session:
            session.query(FundRiskStateModel).delete()
            session.commit()
