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


class Stock(Base):
    """S&P 500 stock metadata"""
    __tablename__ = "stocks"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String(10), unique=True, nullable=False)
    name = Column(String(200), nullable=False)
    sector = Column(String(100), nullable=True)
    sub_industry = Column(String(200), nullable=True)
    headquarters = Column(String(200), nullable=True)
    date_added = Column(String(50), nullable=True)
    cik = Column(String(20), nullable=True)
    founded = Column(String(100), nullable=True)
    
    # Embedding metadata
    has_embeddings = Column(Boolean, default=False)
    embeddings_count = Column(Integer, default=0)
    embeddings_date_range_start = Column(String(20), nullable=True)
    embeddings_date_range_end = Column(String(20), nullable=True)
    last_updated = Column(DateTime, default=datetime.utcnow, 
                         onupdate=datetime.utcnow)
    
    __table_args__ = (
        Index("idx_stocks_symbol", "symbol"),
        Index("idx_stocks_sector", "sector"),
    )


# ============================================================================
# NEW: Fund-based trading models (Collaborative AI Funds)
# ============================================================================

class FundModel(Base):
    """Thesis-driven fund where AI models collaborate."""
    __tablename__ = "funds"
    
    id = Column(String(50), primary_key=True)
    name = Column(String(100), nullable=False)
    strategy = Column(String(50), nullable=False)  # trend_macro, mean_reversion, etc.
    description = Column(Text, nullable=True)
    
    # Configuration stored as JSON
    thesis_json = Column(JSON, nullable=True)
    policy_json = Column(JSON, nullable=True)
    pm_config_json = Column(JSON, nullable=True)
    risk_limits_json = Column(JSON, nullable=True)
    baseline_policy_json = Column(JSON, nullable=True)
    
    # Portfolio state
    cash_balance = Column(DECIMAL(18, 2), nullable=False, default=100000)
    total_value = Column(DECIMAL(18, 2), nullable=False, default=100000)
    
    # Status
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    positions = relationship("FundPosition", back_populates="fund")
    decisions = relationship("DecisionRecordModel", back_populates="fund")
    risk_state = relationship(
        "FundRiskStateModel", back_populates="fund", uselist=False
    )
    
    __table_args__ = (
        Index("idx_funds_strategy", "strategy"),
    )


class FundPosition(Base):
    """Position held by a fund."""
    __tablename__ = "fund_positions"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    fund_id = Column(String(50), ForeignKey("funds.id"), nullable=False)
    symbol = Column(String(10), nullable=False)
    quantity = Column(DECIMAL(18, 8), nullable=False)
    avg_entry_price = Column(DECIMAL(18, 8), nullable=False)
    current_price = Column(DECIMAL(18, 8), nullable=True)
    unrealized_pnl = Column(DECIMAL(18, 8), nullable=True)
    opened_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    fund = relationship("FundModel", back_populates="positions")
    
    __table_args__ = (
        Index("idx_fund_positions_fund_symbol", "fund_id", "symbol", unique=True),
    )


class DecisionRecordModel(Base):
    """Structured decision record for audit trail."""
    __tablename__ = "decision_records"
    
    id = Column(String(50), primary_key=True)
    fund_id = Column(String(50), ForeignKey("funds.id"), nullable=False)
    snapshot_id = Column(String(50), nullable=False)
    asof_timestamp = Column(DateTime, nullable=False)
    
    # Idempotency
    idempotency_key = Column(String(32), unique=True, nullable=False)
    run_context = Column(String(20), nullable=False)  # backtest, paper, live
    decision_window_start = Column(DateTime, nullable=False)
    
    # Decision outcome
    decision_type = Column(String(20), nullable=False)  # trade, no_trade
    no_trade_reason = Column(String(30), nullable=True)
    
    # Lifecycle
    status = Column(String(30), nullable=False)
    status_history_json = Column(JSON, nullable=True)
    
    # Intent and risk
    intent_json = Column(JSON, nullable=True)
    risk_result_json = Column(JSON, nullable=True)
    
    # Context for audit
    snapshot_quality_json = Column(JSON, nullable=True)
    universe_result_json = Column(JSON, nullable=True)
    
    # Reproducibility hashes
    universe_hash = Column(String(20), nullable=True)
    inputs_hash = Column(String(20), nullable=True)  # includes pm_prompt_hash
    
    # Model tracking
    model_versions_json = Column(JSON, nullable=True)
    prompt_hashes_json = Column(JSON, nullable=True)
    
    # Predictions for eval
    predicted_directions_json = Column(JSON, nullable=True)
    expected_return = Column(Float, nullable=True)
    expected_holding_days = Column(Integer, nullable=True)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    fund = relationship("FundModel", back_populates="decisions")
    transcript = relationship(
        "DebateTranscriptModel", back_populates="decision", uselist=False
    )
    outcomes = relationship("TradeOutcome", back_populates="decision")
    
    __table_args__ = (
        Index("idx_decisions_fund", "fund_id"),
        Index("idx_decisions_timestamp", "asof_timestamp"),
        Index("idx_decisions_idempotency", "idempotency_key"),
    )


class DebateTranscriptModel(Base):
    """Raw debate transcript for humans and debugging."""
    __tablename__ = "debate_transcripts"
    
    id = Column(String(50), primary_key=True)
    decision_id = Column(
        String(50), ForeignKey("decision_records.id"), nullable=True
    )
    fund_id = Column(String(50), nullable=False)
    snapshot_id = Column(String(50), nullable=False)
    
    started_at = Column(DateTime, nullable=False)
    completed_at = Column(DateTime, nullable=True)
    
    # Messages stored as JSON array
    messages_json = Column(JSON, nullable=True)
    
    # Summary stats
    num_proposals = Column(Integer, default=0)
    num_critiques = Column(Integer, default=0)
    final_consensus_level = Column(Float, nullable=True)
    
    # Token tracking
    total_input_tokens = Column(Integer, default=0)
    total_output_tokens = Column(Integer, default=0)
    
    # Relationships
    decision = relationship("DecisionRecordModel", back_populates="transcript")
    
    __table_args__ = (
        Index("idx_transcripts_fund", "fund_id"),
        Index("idx_transcripts_timestamp", "started_at"),
    )


class TradeOutcome(Base):
    """Trade-level outcome for evaluation."""
    __tablename__ = "trade_outcomes"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    decision_id = Column(
        String(50), ForeignKey("decision_records.id"), nullable=False
    )
    fund_id = Column(String(50), nullable=False)
    
    # Trade details
    symbol = Column(String(10), nullable=False)
    direction = Column(String(10), nullable=False)  # long, short
    entry_price = Column(DECIMAL(18, 8), nullable=False)
    entry_timestamp = Column(DateTime, nullable=False)
    entry_weight = Column(Float, nullable=True)
    
    # Exit details (filled after close)
    exit_price = Column(DECIMAL(18, 8), nullable=True)
    exit_timestamp = Column(DateTime, nullable=True)
    exit_reason = Column(String(50), nullable=True)  # stop_loss, take_profit, manual
    
    # Outcome metrics
    realized_return = Column(Float, nullable=True)  # % return
    realized_pnl = Column(DECIMAL(18, 8), nullable=True)  # Dollar P&L
    holding_days = Column(Integer, nullable=True)
    slippage_bps = Column(Float, nullable=True)
    
    # Prediction tracking (for calibration)
    predicted_direction = Column(String(10), nullable=True)  # up, down
    predicted_confidence = Column(Float, nullable=True)
    was_correct = Column(Boolean, nullable=True)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    decision = relationship("DecisionRecordModel", back_populates="outcomes")
    
    __table_args__ = (
        Index("idx_outcomes_fund", "fund_id"),
        Index("idx_outcomes_symbol", "symbol"),
        Index("idx_outcomes_entry", "entry_timestamp"),
    )


class FundRiskStateModel(Base):
    """Risk state tracking for funds (cooldowns, P&L)."""
    __tablename__ = "fund_risk_states"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    fund_id = Column(
        String(50), ForeignKey("funds.id"), unique=True, nullable=False
    )
    
    # Cooldown tracking
    risk_off_until = Column(DateTime, nullable=True)
    last_breach_reason = Column(String(50), nullable=True)
    last_breach_time = Column(DateTime, nullable=True)
    
    # P&L tracking for circuit breakers
    current_daily_pnl_pct = Column(Float, default=0.0)
    current_weekly_drawdown_pct = Column(Float, default=0.0)
    peak_nav = Column(DECIMAL(18, 2), nullable=True)
    
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    fund = relationship("FundModel", back_populates="risk_state")


class FundDailySnapshot(Base):
    """Daily performance snapshot for funds."""
    __tablename__ = "fund_daily_snapshots"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    fund_id = Column(String(50), ForeignKey("funds.id"), nullable=False)
    date = Column(Date, nullable=False)
    
    # Performance
    portfolio_value = Column(DECIMAL(18, 2), nullable=False)
    daily_return = Column(Float, nullable=True)
    cumulative_return = Column(Float, nullable=True)
    
    # Risk metrics
    volatility_21d = Column(Float, nullable=True)
    sharpe_21d = Column(Float, nullable=True)
    max_drawdown = Column(Float, nullable=True)
    
    # Trade stats
    n_positions = Column(Integer, default=0)
    gross_exposure = Column(Float, nullable=True)
    net_exposure = Column(Float, nullable=True)
    
    # Eval metrics
    n_trades = Column(Integer, default=0)
    hit_rate = Column(Float, nullable=True)
    avg_holding_days = Column(Float, nullable=True)
    turnover = Column(Float, nullable=True)
    
    __table_args__ = (
        Index(
            "idx_fund_snapshots_fund_date", "fund_id", "date", unique=True
        ),
    )


# ============================================================================
# BACKTEST MODELS - AI Fund Time Machine
# ============================================================================

class BacktestRun(Base):
    """
    A single backtest execution.
    
    Tracks the configuration and status of a backtest simulation
    from 2000-2025 across multiple AI funds.
    """
    __tablename__ = "backtest_runs"
    
    id = Column(String(50), primary_key=True)
    name = Column(String(100), nullable=True)
    
    # Date range
    start_date = Column(Date, nullable=False)
    end_date = Column(Date, nullable=False)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)
    
    # Configuration (JSON)
    config = Column(JSON, nullable=True)  # funds, universe, parameters
    fund_ids = Column(JSON, nullable=True)  # List of fund IDs
    universe = Column(JSON, nullable=True)  # List of symbols
    initial_cash = Column(Float, default=100000.0)
    
    # Status
    status = Column(String(20), default="pending")  # pending, running, completed, failed
    error_message = Column(Text, nullable=True)
    
    # Progress
    current_day = Column(Integer, default=0)
    total_days = Column(Integer, default=0)
    
    # Summary stats (filled on completion)
    total_trades = Column(Integer, default=0)
    total_decisions = Column(Integer, default=0)
    elapsed_seconds = Column(Float, nullable=True)
    
    # Relationships
    trades = relationship("BacktestTradeRecord", back_populates="run")
    decisions = relationship("BacktestDecisionRecord", back_populates="run")
    snapshots = relationship("BacktestPortfolioSnapshotRecord", back_populates="run")
    
    __table_args__ = (
        Index("idx_backtest_runs_status", "status"),
        Index("idx_backtest_runs_created", "created_at"),
    )


class BacktestTradeRecord(Base):
    """
    Every trade executed in a backtest.
    
    Stores full details including AI reasoning.
    """
    __tablename__ = "backtest_trades"
    
    id = Column(String(50), primary_key=True)
    run_id = Column(String(50), ForeignKey("backtest_runs.id"), nullable=False)
    fund_id = Column(String(50), nullable=False)
    
    # Trade details
    trade_date = Column(Date, nullable=False)
    symbol = Column(String(10), nullable=False)
    side = Column(String(4), nullable=False)  # buy, sell
    quantity = Column(Float, nullable=False)
    price = Column(Float, nullable=False)
    commission = Column(Float, default=0.0)
    total_cost = Column(Float, nullable=True)  # quantity * price +/- commission
    
    # AI reasoning
    reasoning = Column(Text, nullable=True)
    confidence = Column(Float, nullable=True)
    signals_snapshot = Column(JSON, nullable=True)  # quant signals at decision time
    
    # Execution timing
    executed_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    run = relationship("BacktestRun", back_populates="trades")
    
    __table_args__ = (
        Index("idx_backtest_trades_run", "run_id"),
        Index("idx_backtest_trades_fund", "fund_id"),
        Index("idx_backtest_trades_date", "trade_date"),
        Index("idx_backtest_trades_symbol", "symbol"),
    )


class BacktestDecisionRecord(Base):
    """
    Every AI decision (including HOLDs).
    
    Captures the full debate and reasoning process.
    """
    __tablename__ = "backtest_decisions"
    
    id = Column(String(50), primary_key=True)
    run_id = Column(String(50), ForeignKey("backtest_runs.id"), nullable=False)
    fund_id = Column(String(50), nullable=False)
    
    # Decision details
    decision_date = Column(Date, nullable=False)
    action = Column(String(10), nullable=False)  # buy, sell, hold
    symbol = Column(String(10), nullable=True)  # Null for hold
    quantity = Column(Float, nullable=True)
    target_weight = Column(Float, nullable=True)
    
    # AI reasoning
    confidence = Column(Float, nullable=True)
    reasoning = Column(Text, nullable=True)
    
    # Debate transcript (JSON)
    debate_transcript = Column(JSON, nullable=True)  # Full debate messages
    
    # Signals at decision time
    signals_snapshot = Column(JSON, nullable=True)
    
    # Model info
    models_used = Column(JSON, nullable=True)  # {"analyze": "gemini", "propose": "gpt"}
    tokens_used = Column(Integer, default=0)
    
    # What triggered the decision
    triggered_by = Column(String(50), nullable=True)  # "momentum", "stop_loss", etc.
    
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    run = relationship("BacktestRun", back_populates="decisions")
    
    __table_args__ = (
        Index("idx_backtest_decisions_run", "run_id"),
        Index("idx_backtest_decisions_fund", "fund_id"),
        Index("idx_backtest_decisions_date", "decision_date"),
    )


class BacktestPortfolioSnapshotRecord(Base):
    """
    Daily portfolio state for each fund.
    
    Used for equity curves and performance analysis.
    """
    __tablename__ = "backtest_portfolio_snapshots"
    
    id = Column(String(50), primary_key=True)
    run_id = Column(String(50), ForeignKey("backtest_runs.id"), nullable=False)
    fund_id = Column(String(50), nullable=False)
    
    # Date
    snapshot_date = Column(Date, nullable=False)
    
    # Portfolio state
    cash = Column(Float, nullable=False)
    positions = Column(JSON, nullable=True)  # {symbol: {qty, avg_cost, current_value}}
    total_value = Column(Float, nullable=False)
    invested_pct = Column(Float, nullable=True)
    
    # Performance metrics
    daily_return = Column(Float, nullable=True)
    cumulative_return = Column(Float, nullable=True)
    max_drawdown = Column(Float, nullable=True)
    sharpe_ratio = Column(Float, nullable=True)
    
    # Trade stats
    n_positions = Column(Integer, default=0)
    n_trades_today = Column(Integer, default=0)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    run = relationship("BacktestRun", back_populates="snapshots")
    
    __table_args__ = (
        Index("idx_backtest_snapshots_run", "run_id"),
        Index("idx_backtest_snapshots_fund_date", "fund_id", "snapshot_date"),
    )
