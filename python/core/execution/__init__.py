"""
Execution module - Order execution, risk management, and state machine.
"""
from core.execution.alpaca_client import AlpacaClient
from core.execution.risk_manager import (
    RiskManager,
    RiskCheckResult,
    RiskViolation,
    CloseOrder,
    PostFillAction,
    Fill,
)
from core.execution.risk_repo import (
    FundRiskState,
    FundRiskStateRepo,
    InMemoryFundRiskStateRepo,
    DBFundRiskStateRepo,
)
from core.execution.intent import (
    PortfolioIntent,
    PositionIntent,
    ExitRule,
    WeightBasis,
    Order,
    ExecutionEngine,
)
from core.execution.state_machine import (
    DecisionRecord,
    DecisionStateMachine,
    DecisionStatus,
    DecisionType,
    NoTradeReason,
    RunContext,
    StatusTransition,
    TERMINAL_STATES,
    compute_idempotency_key,
    compute_inputs_hash,
)
from core.execution.trading_engine import TradingEngine
from core.execution.fund_trading_engine import FundTradingEngine, FundTradeResult

__all__ = [
    # Alpaca
    "AlpacaClient",
    # Risk Manager
    "RiskManager",
    "RiskCheckResult",
    "RiskViolation",
    "CloseOrder",
    "PostFillAction",
    "Fill",
    # Risk Repo
    "FundRiskState",
    "FundRiskStateRepo",
    "InMemoryFundRiskStateRepo",
    "DBFundRiskStateRepo",
    # Intent
    "PortfolioIntent",
    "PositionIntent",
    "ExitRule",
    "WeightBasis",
    "Order",
    "ExecutionEngine",
    # State Machine
    "DecisionRecord",
    "DecisionStateMachine",
    "DecisionStatus",
    "DecisionType",
    "NoTradeReason",
    "RunContext",
    "StatusTransition",
    "TERMINAL_STATES",
    "compute_idempotency_key",
    "compute_inputs_hash",
    # Trading Engine
    "TradingEngine",
    # Fund Trading Engine
    "FundTradingEngine",
    "FundTradeResult",
]
