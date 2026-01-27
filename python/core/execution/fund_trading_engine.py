"""
Fund Trading Engine - Orchestrates trading cycles for collaborative funds.

Key principles:
- Snapshot quality checks before trading
- Idempotent execution via state machine
- Quantity-based go_flat (no stale prices)
- Full audit trail
"""
import asyncio
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import logging
import uuid

from core.config.constants import CONSTANTS
from core.data.snapshot import GlobalMarketSnapshot, DataQuality
from core.funds.fund import Fund
from core.funds.universe import UniverseResolver, UniverseResult
from core.execution.intent import PortfolioIntent, Order, ExecutionEngine
from core.execution.state_machine import (
    DecisionStateMachine,
    DecisionRecord,
    DecisionStatus,
    DecisionType,
    NoTradeReason,
    RunContext,
    compute_inputs_hash,
)
from core.execution.risk_manager import (
    RiskManager,
    RiskCheckResult,
    PostFillAction,
    Fill,
    CloseOrder,
)
from core.execution.risk_repo import (
    FundRiskStateRepo,
    InMemoryFundRiskStateRepo,
)
from core.collaboration.debate import DebateEngine, DebateResult
from core.collaboration.participant import AIParticipant

logger = logging.getLogger(__name__)


@dataclass
class FundTradeResult:
    """Result of a fund trading cycle."""
    fund_id: str
    decision_id: str
    decision_type: str  # "trade" or "no_trade"
    success: bool
    orders_sent: int = 0
    orders_filled: int = 0
    error: Optional[str] = None


class FundTradingEngine:
    """
    Trading engine for collaborative funds.
    
    Orchestrates the full cycle:
    1. Build GlobalMarketSnapshot
    2. Validate snapshot quality
    3. Resolve universe
    4. Run debate
    5. Create decision record
    6. Execute with idempotency
    7. Handle post-fill actions (go_flat if needed)
    """
    
    def __init__(
        self,
        funds: Dict[str, Fund],
        debate_engine: DebateEngine,
        risk_manager: RiskManager,
        execution_engine: ExecutionEngine,
        universe_resolver: UniverseResolver,
        risk_state_repo: Optional[FundRiskStateRepo] = None,
        run_context: RunContext = RunContext.PAPER,
    ):
        """
        Initialize fund trading engine.
        
        Args:
            funds: Dictionary of fund_id -> Fund
            debate_engine: Engine for running debates
            risk_manager: Risk manager
            execution_engine: Order execution engine
            universe_resolver: Universe resolver
            risk_state_repo: Repository for risk state (default: in-memory)
            run_context: backtest/paper/live
        """
        self.funds = funds
        self.debate_engine = debate_engine
        self.risk_manager = risk_manager
        self.execution_engine = execution_engine
        self.universe_resolver = universe_resolver
        self.risk_state_repo = risk_state_repo or InMemoryFundRiskStateRepo()
        self.run_context = run_context
        
        self.state_machine = DecisionStateMachine()
        
        # Track executed idempotency keys to prevent duplicates
        self._executed_keys: set = set()
    
    async def run_fund_cycle(
        self,
        fund_id: str,
        snapshot: GlobalMarketSnapshot,
        decision_window_start: datetime,
    ) -> FundTradeResult:
        """
        Run a single trading cycle for a fund.
        
        Args:
            fund_id: Fund to run cycle for
            snapshot: Market snapshot
            decision_window_start: Start of decision window
        
        Returns:
            FundTradeResult with outcome
        """
        fund = self.funds.get(fund_id)
        if fund is None:
            return FundTradeResult(
                fund_id=fund_id,
                decision_id="",
                decision_type="no_trade",
                success=False,
                error=f"Fund {fund_id} not found"
            )
        
        # Create decision record
        decision = self.state_machine.create_decision(
            fund_id=fund_id,
            snapshot_id=snapshot.snapshot_id,
            run_context=self.run_context,
            decision_window_start=decision_window_start,
            asof_timestamp=snapshot.asof_timestamp,
        )
        
        # Check idempotency
        if decision.idempotency_key in self._executed_keys:
            logger.warning(
                f"Duplicate execution prevented: {decision.idempotency_key}"
            )
            return FundTradeResult(
                fund_id=fund_id,
                decision_id=decision.decision_id,
                decision_type="no_trade",
                success=True,
                error="Duplicate execution prevented"
            )
        
        try:
            # Step 1: Validate snapshot quality
            valid, rejection_reason = snapshot.validate_or_reject()
            if not valid:
                decision = self.state_machine.mark_no_trade(
                    decision, NoTradeReason.SNAPSHOT_INVALID
                )
                decision.snapshot_quality_json = snapshot.quality.to_summary()
                return self._complete_no_trade(decision, rejection_reason)
            
            decision.snapshot_quality_json = snapshot.quality.to_summary()
            
            # Step 2: Resolve universe
            universe_result = self.universe_resolver.resolve(
                fund.thesis.universe_spec,
                snapshot,
                require_vol=True
            )
            
            if not universe_result.success:
                decision = self.state_machine.mark_no_trade(
                    decision, NoTradeReason.UNIVERSE_EMPTY
                )
                decision.universe_result_json = universe_result.to_summary()
                return self._complete_no_trade(decision, universe_result.error)
            
            decision.universe_result_json = universe_result.to_summary()
            decision.universe_hash = universe_result.universe_hash
            
            # Compute inputs hash
            decision.inputs_hash = compute_inputs_hash(
                snapshot_id=snapshot.snapshot_id,
                universe_hash=universe_result.universe_hash,
                fund_policy_version=fund.policy.version,
                fund_thesis_version=fund.thesis.version,
                pm_prompt_hash=fund.pm_config.prompt_hash,
            )
            
            # Step 3: Run debate
            self.state_machine.transition(
                decision, DecisionStatus.DEBATED, "Starting debate"
            )
            
            debate_result = self.debate_engine.run_debate(
                fund=fund,
                snapshot=snapshot,
                universe_symbols=universe_result.symbols,
            )
            
            if not debate_result.success:
                decision = self.state_machine.mark_no_trade(
                    decision, NoTradeReason.DISAGREEMENT
                )
                return self._complete_no_trade(decision, debate_result.error)
            
            if debate_result.decision == "no_trade":
                reason = NoTradeReason.DISAGREEMENT
                if debate_result.risk_result and debate_result.risk_result.status == "vetoed":  # noqa
                    reason = NoTradeReason.RISK_VETO
                decision = self.state_machine.mark_no_trade(decision, reason)
                return self._complete_no_trade(decision, "No trade decision")
            
            # Step 4: Finalize decision
            intent = debate_result.intent
            if intent is None:
                decision = self.state_machine.mark_no_trade(
                    decision, NoTradeReason.NO_OPPORTUNITIES
                )
                return self._complete_no_trade(decision, "No intent generated")
            
            # Record predictions
            predicted_directions = {
                pos.symbol: "up" if pos.target_weight > 0 else "down"
                for pos in intent.positions
            }
            
            self.state_machine.mark_trade(
                decision,
                intent_json=intent.to_dict(),
                predicted_directions=predicted_directions,
                expected_holding_days=(
                    debate_result.final_output.expected_holding_days
                    if debate_result.final_output else None
                ),
                expected_return=(
                    debate_result.final_output.expected_return
                    if debate_result.final_output else None
                ),
            )
            
            decision.risk_result_json = (
                debate_result.risk_result.to_dict()
                if debate_result.risk_result else None
            )
            
            self.state_machine.transition(
                decision, DecisionStatus.FINALIZED, "Decision finalized"
            )
            
            # Step 5: Execute
            return await self._execute_intent(fund, intent, decision)
            
        except Exception as e:
            logger.exception(f"Error in fund cycle for {fund_id}")
            self.state_machine.transition(
                decision, DecisionStatus.ERRORED, str(e)
            )
            return FundTradeResult(
                fund_id=fund_id,
                decision_id=decision.decision_id,
                decision_type="no_trade",
                success=False,
                error=str(e)
            )
    
    async def _execute_intent(
        self,
        fund: Fund,
        intent: PortfolioIntent,
        decision: DecisionRecord,
    ) -> FundTradeResult:
        """Execute a portfolio intent."""
        # Get current position weights
        current_weights = {
            symbol: fund.portfolio.get_weight(symbol)
            for symbol in fund.portfolio.positions
        }
        
        # Convert intent to orders
        orders, error = self.execution_engine.execute(
            intent=intent,
            current_positions=current_weights,
            policy_max_gross=fund.policy.max_gross_exposure,
            policy_min_cash=fund.policy.min_cash_buffer,
        )
        
        if error:
            self.state_machine.transition(
                decision, DecisionStatus.ERRORED, error
            )
            return FundTradeResult(
                fund_id=fund.fund_id,
                decision_id=decision.decision_id,
                decision_type="trade",
                success=False,
                error=error
            )
        
        if not orders:
            # No orders needed (positions already match)
            self.state_machine.transition(
                decision, DecisionStatus.FILLED, "No orders needed"
            )
            self._executed_keys.add(decision.idempotency_key)
            return FundTradeResult(
                fund_id=fund.fund_id,
                decision_id=decision.decision_id,
                decision_type="trade",
                success=True,
                orders_sent=0,
                orders_filled=0,
            )
        
        # Order-level risk check
        risk_result = self.risk_manager.check_orders(
            orders, fund, fund.portfolio
        )
        
        if risk_result.status == "vetoed":
            self.state_machine.transition(
                decision, DecisionStatus.RISK_VETOED, "Order risk veto"
            )
            return FundTradeResult(
                fund_id=fund.fund_id,
                decision_id=decision.decision_id,
                decision_type="no_trade",
                success=True,
                error="Order risk veto"
            )
        
        self.state_machine.transition(
            decision, DecisionStatus.SENT_TO_BROKER, f"Sending {len(orders)} orders"
        )
        
        # Execute orders (placeholder - in production, call broker API)
        fills = await self._send_orders(orders)
        
        # Update portfolio state (placeholder)
        self._update_portfolio_from_fills(fund, fills)
        
        # Post-fill risk check
        post_fill = self.risk_manager.check_post_fill(
            fills, fund, fund.portfolio
        )
        
        if post_fill.action == "go_flat":
            await self._execute_go_flat(fund, post_fill.close_orders)
        
        # Mark filled
        if len(fills) == len(orders):
            self.state_machine.transition(
                decision, DecisionStatus.FILLED, "All orders filled"
            )
        elif fills:
            self.state_machine.transition(
                decision, DecisionStatus.PARTIALLY_FILLED,
                f"{len(fills)}/{len(orders)} orders filled"
            )
        else:
            self.state_machine.transition(
                decision, DecisionStatus.ERRORED, "No fills received"
            )
        
        self._executed_keys.add(decision.idempotency_key)
        
        return FundTradeResult(
            fund_id=fund.fund_id,
            decision_id=decision.decision_id,
            decision_type="trade",
            success=True,
            orders_sent=len(orders),
            orders_filled=len(fills),
        )
    
    async def _send_orders(self, orders: List[Order]) -> List[Fill]:
        """
        Send orders to broker.
        
        Placeholder for v1 - in production, call actual broker API.
        """
        fills = []
        for order in orders:
            # Simulate immediate fill at expected price
            fills.append(Fill(
                symbol=order.symbol,
                quantity=order.quantity,
                price=order.expected_price,
                side=order.side,
            ))
        return fills
    
    def _update_portfolio_from_fills(
        self,
        fund: Fund,
        fills: List[Fill]
    ) -> None:
        """Update fund portfolio from fills."""
        from core.funds.fund import Position
        
        for fill in fills:
            if fill.side == "buy":
                # Add or increase position
                if fill.symbol in fund.portfolio.positions:
                    pos = fund.portfolio.positions[fill.symbol]
                    total_qty = pos.quantity + fill.quantity
                    total_cost = (
                        pos.quantity * pos.avg_entry_price +
                        fill.quantity * fill.price
                    )
                    pos.quantity = total_qty
                    pos.avg_entry_price = total_cost / total_qty if total_qty else 0
                    pos.current_price = fill.price
                else:
                    fund.portfolio.positions[fill.symbol] = Position(
                        symbol=fill.symbol,
                        quantity=fill.quantity,
                        avg_entry_price=fill.price,
                        current_price=fill.price,
                    )
                fund.portfolio.cash_balance -= fill.quantity * fill.price
            
            else:  # sell
                if fill.symbol in fund.portfolio.positions:
                    pos = fund.portfolio.positions[fill.symbol]
                    pos.quantity -= fill.quantity
                    pos.current_price = fill.price
                    if pos.quantity <= 0:
                        del fund.portfolio.positions[fill.symbol]
                fund.portfolio.cash_balance += fill.quantity * fill.price
    
    async def _execute_go_flat(
        self,
        fund: Fund,
        close_orders: List[CloseOrder]
    ) -> None:
        """
        Execute go_flat orders.
        
        Uses quantity-based close orders (no stale prices).
        """
        logger.warning(
            f"Executing go_flat for fund {fund.fund_id}: "
            f"{len(close_orders)} positions"
        )
        
        for close_order in close_orders:
            # In production, submit market order to broker
            # For v1, simulate immediate close
            if close_order.side == "sell":
                if close_order.symbol in fund.portfolio.positions:
                    pos = fund.portfolio.positions[close_order.symbol]
                    fund.portfolio.cash_balance += (
                        close_order.quantity * pos.current_price
                    )
                    del fund.portfolio.positions[close_order.symbol]
            else:  # buy to close short
                # Similar logic for shorts
                pass
    
    def _complete_no_trade(
        self,
        decision: DecisionRecord,
        reason: Optional[str]
    ) -> FundTradeResult:
        """Complete a no-trade decision."""
        self._executed_keys.add(decision.idempotency_key)
        return FundTradeResult(
            fund_id=decision.fund_id,
            decision_id=decision.decision_id,
            decision_type="no_trade",
            success=True,
            error=reason
        )
    
    async def run_all_funds(
        self,
        snapshot: GlobalMarketSnapshot,
        decision_window_start: datetime,
    ) -> List[FundTradeResult]:
        """Run trading cycle for all active funds."""
        results = []
        
        for fund_id, fund in self.funds.items():
            if not fund.is_active:
                continue
            
            result = await self.run_fund_cycle(
                fund_id, snapshot, decision_window_start
            )
            results.append(result)
        
        return results
    
    def get_fund_leaderboard(self) -> List[Dict]:
        """Get fund leaderboard sorted by portfolio value."""
        entries = []
        
        for fund_id, fund in self.funds.items():
            entries.append({
                "fund_id": fund_id,
                "name": fund.thesis.name,
                "strategy": fund.thesis.strategy,
                "total_value": fund.portfolio.total_value,
                "cash": fund.portfolio.cash_balance,
                "positions": len(fund.portfolio.positions),
                "gross_exposure": fund.portfolio.gross_exposure,
                "is_active": fund.is_active,
            })
        
        entries.sort(key=lambda x: x["total_value"], reverse=True)
        
        for i, entry in enumerate(entries):
            entry["rank"] = i + 1
        
        return entries
