"""
Baseline Fallback - Deterministic fallback to quant baseline on no consensus.

Key principle: "Defer to baseline" must be deterministic policy, not vibes.
Rules are explicit in code, not in comments.
"""
from dataclasses import dataclass
from typing import Optional, Tuple, TYPE_CHECKING

from core.config.constants import CONSTANTS

if TYPE_CHECKING:
    from core.funds.fund import Fund
    from core.data.snapshot import GlobalMarketSnapshot
    from core.execution.intent import PortfolioIntent
    from core.execution.risk_manager import RiskManager, FundRiskState


@dataclass
class BaselineFallbackPolicy:
    """
    Explicit rules for when to use quant baseline (not vibes).
    
    These are deterministic rules that can be audited.
    """
    enabled: bool = True
    
    # SIMPLE: trigger if consensus below this threshold
    # e.g., if consensus_level < 0.3, consider baseline
    max_consensus_to_trigger: float = CONSTANTS.debate.CONSENSUS_THRESHOLD
    
    # Baseline must pass these checks
    require_risk_approval: bool = True
    min_baseline_confidence: float = 0.6
    fund_must_not_be_in_cooldown: bool = True
    
    # How to use it
    scale_down_factor: float = 0.5  # Reduce position sizes
    max_position_from_baseline: float = 0.10  # Cap size from baseline


class BaselineFallbackHandler:
    """
    Handles fallback to quant baseline on no consensus.
    
    This is invoked when the AI debate doesn't reach sufficient consensus.
    """
    
    def should_use_baseline(
        self,
        consensus_level: float,
        fund: "Fund",
        policy: BaselineFallbackPolicy,
        risk_state: Optional["FundRiskState"]
    ) -> bool:
        """
        Determine if we should fall back to baseline.
        
        Args:
            consensus_level: Level of consensus from debate (0-1)
            fund: The fund in question
            policy: Fallback policy
            risk_state: Current risk state of the fund
        
        Returns:
            True if we should use baseline
        """
        if not policy.enabled:
            return False
        
        # SIMPLE: consensus below threshold triggers baseline consideration
        if consensus_level >= policy.max_consensus_to_trigger:
            return False  # Enough consensus, don't need baseline
        
        # Check cooldown
        if policy.fund_must_not_be_in_cooldown:
            if risk_state and risk_state.is_in_cooldown():
                return False
        
        return True
    
    def get_baseline_decision(
        self,
        fund: "Fund",
        snapshot: "GlobalMarketSnapshot",
        policy: BaselineFallbackPolicy,
        risk_manager: "RiskManager"
    ) -> Tuple[Optional["PortfolioIntent"], Optional[str]]:
        """
        Get baseline decision if it qualifies.
        
        Args:
            fund: The fund
            snapshot: Market snapshot
            policy: Fallback policy
            risk_manager: Risk manager for validation
        
        Returns:
            Tuple of (intent, error_reason)
            - (intent, None) if baseline qualifies
            - (None, reason) if baseline doesn't qualify
        """
        # Run quant baseline
        baseline_result = self._run_quant_baseline(fund, snapshot)
        
        if baseline_result is None:
            return None, "baseline_no_signal"
        
        if baseline_result.confidence < policy.min_baseline_confidence:
            return None, "baseline_low_confidence"
        
        # Scale down positions
        scaled_intent = self._scale_intent(
            baseline_result.intent,
            policy.scale_down_factor,
            policy.max_position_from_baseline
        )
        
        # Must pass risk check
        if policy.require_risk_approval:
            risk_result = risk_manager.check_intent(
                scaled_intent,
                fund,
                fund.portfolio
            )
            if risk_result.status == "vetoed":
                return None, "baseline_risk_veto"
        
        return scaled_intent, None
    
    def _run_quant_baseline(
        self,
        fund: "Fund",
        snapshot: "GlobalMarketSnapshot"
    ) -> Optional["BaselineResult"]:
        """
        Run the quant baseline strategy.
        
        For v1, this is a placeholder. In production, this would run
        the quant bot's decision logic.
        """
        # Placeholder - in production, delegate to QuantBot
        return None
    
    def _scale_intent(
        self,
        intent: "PortfolioIntent",
        scale_factor: float,
        max_position: float
    ) -> "PortfolioIntent":
        """
        Scale down an intent's positions.
        
        Args:
            intent: Original intent
            scale_factor: Factor to scale by (e.g., 0.5 = halve positions)
            max_position: Maximum position size after scaling
        
        Returns:
            Scaled intent
        """
        # Create a copy with scaled positions
        from core.execution.intent import PortfolioIntent, PositionIntent
        
        scaled_positions = []
        for pos in intent.positions:
            scaled_weight = pos.target_weight * scale_factor
            # Cap at max_position
            if abs(scaled_weight) > max_position:
                scaled_weight = max_position if scaled_weight > 0 else -max_position
            
            scaled_positions.append(PositionIntent(
                symbol=pos.symbol,
                target_weight=scaled_weight,
                direction=pos.direction,
                urgency=pos.urgency,
                max_slippage_bps=pos.max_slippage_bps,
                time_limit_minutes=pos.time_limit_minutes,
            ))
        
        return PortfolioIntent(
            intent_id=f"baseline_{intent.intent_id}",
            fund_id=intent.fund_id,
            asof_timestamp=intent.asof_timestamp,
            portfolio_value=intent.portfolio_value,
            asof_prices=intent.asof_prices,
            valuation_timestamp=intent.valuation_timestamp,
            weight_basis=intent.weight_basis,
            positions=scaled_positions,
            target_cash_pct=intent.target_cash_pct,
            max_turnover=intent.max_turnover,
            execution_window_minutes=intent.execution_window_minutes,
            sizing_method_used="baseline_scaled",
            policy_version=intent.policy_version,
            exit_rules=intent.exit_rules,
        )


@dataclass
class BaselineResult:
    """Result from running quant baseline."""
    intent: "PortfolioIntent"
    confidence: float
    reasoning: str = ""
