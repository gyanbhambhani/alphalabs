from datetime import datetime
from decimal import Decimal
from typing import Optional
from sqlalchemy import (
    Column, String, Integer, Float, Boolean, DateTime, 
    ForeignKey, Text, JSON, DECIMAL, Date, Index
)
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()


class Manager(Base):
    """Portfolio manager (LLM or Quant Bot)"""
    __tablename__ = "managers"
    
    id = Column(String(50), primary_key=True)
    name = Column(String(100), nullable=False)
    type = Column(String(20), nullable=False)  # 'llm' or 'quant'
    provider = Column(String(50), nullable=True)  # 'openai', 'anthropic', 'google'
    description = Column(Text, nullable=True)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    portfolio = relationship("Portfolio", back_populates="manager", uselist=False)
    positions = relationship("Position", back_populates="manager")
    trades = relationship("Trade", back_populates="manager")
    snapshots = relationship("DailySnapshot", back_populates="manager")


class Portfolio(Base):
    """Portfolio state per manager"""
    __tablename__ = "portfolios"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    manager_id = Column(
        String(50), 
        ForeignKey("managers.id"), 
        unique=True, 
        nullable=False
    )
    cash_balance = Column(DECIMAL(18, 2), nullable=False, default=0)
    total_value = Column(DECIMAL(18, 2), nullable=False, default=0)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    manager = relationship("Manager", back_populates="portfolio")


class Position(Base):
    """Open positions per manager"""
    __tablename__ = "positions"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    manager_id = Column(String(50), ForeignKey("managers.id"), nullable=False)
    symbol = Column(String(10), nullable=False)
    quantity = Column(DECIMAL(18, 8), nullable=False)
    avg_entry_price = Column(DECIMAL(18, 8), nullable=False)
    current_price = Column(DECIMAL(18, 8), nullable=True)
    unrealized_pnl = Column(DECIMAL(18, 8), nullable=True)
    opened_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    manager = relationship("Manager", back_populates="positions")
    
    __table_args__ = (
        Index("idx_positions_manager_symbol", "manager_id", "symbol", unique=True),
    )


class Trade(Base):
    """Trade history"""
    __tablename__ = "trades"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    manager_id = Column(String(50), ForeignKey("managers.id"), nullable=False)
    symbol = Column(String(10), nullable=False)
    side = Column(String(4), nullable=False)  # 'buy' or 'sell'
    quantity = Column(DECIMAL(18, 8), nullable=False)
    price = Column(DECIMAL(18, 8), nullable=False)
    order_type = Column(String(20), default="market")
    status = Column(String(20), default="filled")
    reasoning = Column(Text, nullable=True)
    signals_used = Column(JSON, nullable=True)
    alpaca_order_id = Column(String(50), nullable=True)
    executed_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    manager = relationship("Manager", back_populates="trades")
    
    __table_args__ = (
        Index("idx_trades_manager", "manager_id"),
        Index("idx_trades_executed", "executed_at"),
    )


class DailySnapshot(Base):
    """Daily performance snapshots for tracking"""
    __tablename__ = "daily_snapshots"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    manager_id = Column(String(50), ForeignKey("managers.id"), nullable=False)
    date = Column(Date, nullable=False)
    portfolio_value = Column(DECIMAL(18, 2), nullable=False)
    daily_return = Column(DECIMAL(10, 6), nullable=True)
    cumulative_return = Column(DECIMAL(10, 6), nullable=True)
    sharpe_ratio = Column(DECIMAL(10, 4), nullable=True)
    volatility = Column(DECIMAL(10, 6), nullable=True)
    max_drawdown = Column(DECIMAL(10, 6), nullable=True)
    win_rate = Column(DECIMAL(5, 4), nullable=True)
    total_trades = Column(Integer, default=0)
    
    # Relationships
    manager = relationship("Manager", back_populates="snapshots")
    
    __table_args__ = (
        Index("idx_snapshots_manager_date", "manager_id", "date", unique=True),
    )


class SignalSnapshot(Base):
    """Strategy signals at a point in time"""
    __tablename__ = "signal_snapshots"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, nullable=False, default=datetime.utcnow)
    signals = Column(JSON, nullable=False)
    
    __table_args__ = (
        Index("idx_signals_timestamp", "timestamp"),
    )


class MarketData(Base):
    """Historical market data cache"""
    __tablename__ = "market_data"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String(10), nullable=False)
    timestamp = Column(DateTime, nullable=False)
    open = Column(DECIMAL(18, 8), nullable=True)
    high = Column(DECIMAL(18, 8), nullable=True)
    low = Column(DECIMAL(18, 8), nullable=True)
    close = Column(DECIMAL(18, 8), nullable=True)
    volume = Column(Integer, nullable=True)
    vwap = Column(DECIMAL(18, 8), nullable=True)
    
    __table_args__ = (
        Index("idx_market_data_symbol_ts", "symbol", "timestamp", unique=True),
    )
