"""
Outcome Labeler

Post-hoc labeling of decision outcomes for ML training.

After a backtest completes, this fills in:
- Forward returns at multiple horizons (1d, 5d, 21d, 63d)
- Realized P&L (for selected candidates)
- Win/loss labels
- Alpha vs SPY

This converts raw decisions into supervised learning training data.
"""

from typing import Dict, List, Optional
from datetime import date, timedelta
import logging

from core.data.snapshot import GlobalMarketSnapshot
from core.backtest.data_loader import HistoricalDataLoader
from core.backtest.persistence import BacktestPersistence

logger = logging.getLogger(__name__)


class OutcomeLabeler:
    """
    Labels decision outcomes post-hoc for ML training.
    
    Runs after backtest completes to fill in forward returns.
    """
    
    def __init__(
        self,
        data_loader: HistoricalDataLoader,
        persistence: BacktestPersistence,
    ):
        """
        Initialize outcome labeler.
        
        Args:
            data_loader: Data loader for getting future prices
            persistence: Persistence layer for updating records
        """
        self.data_loader = data_loader
        self.persistence = persistence
    
    def label_run_outcomes(
        self,
        run_id: str,
        start_date: date,
        end_date: date,
    ) -> Dict[str, int]:
        """
        Label all outcomes for a backtest run.
        
        Args:
            run_id: Backtest run ID
            start_date: Start of backtest
            end_date: End of backtest
            
        Returns:
            Dict with counts of labeled records
        """
        stats = {
            "candidates_labeled": 0,
            "experiences_labeled": 0,
            "errors": 0,
        }
        
        logger.info(f"Labeling outcomes for run {run_id}...")
        
        # Get all decisions for this run
        # TODO: Query decisions from persistence
        
        # For now, placeholder implementation
        logger.warning("Outcome labeling not yet fully implemented")
        
        return stats
    
    def compute_forward_returns(
        self,
        symbol: str,
        decision_date: date,
        entry_price: float,
    ) -> Dict[str, Optional[float]]:
        """
        Compute forward returns at multiple horizons.
        
        Args:
            symbol: Asset symbol
            decision_date: Date of decision
            entry_price: Entry price
            
        Returns:
            Dict of horizon -> return (e.g., {"1d": 0.02, "5d": 0.05, ...})
        """
        horizons = {
            "1d": 1,
            "5d": 5,
            "21d": 21,
            "63d": 63,
        }
        
        returns = {}
        
        for label, days in horizons.items():
            # Get future date
            target_date = decision_date + timedelta(days=days)
            
            # Get price at future date
            future_price = self.data_loader.get_price_asof(symbol, target_date)
            
            if future_price and entry_price > 0:
                ret = (future_price - entry_price) / entry_price
                returns[label] = float(ret)
            else:
                returns[label] = None
        
        return returns
    
    def compute_alpha_vs_spy(
        self,
        asset_return: float,
        decision_date: date,
        horizon_days: int = 21,
    ) -> Optional[float]:
        """
        Compute alpha (excess return vs SPY) at a horizon.
        
        Args:
            asset_return: Asset return over horizon
            decision_date: Date of decision
            horizon_days: Holding period
            
        Returns:
            Alpha (asset_return - spy_return)
        """
        target_date = decision_date + timedelta(days=horizon_days)
        
        # Get SPY returns
        spy_return = self.data_loader.calc_return("SPY", target_date, horizon_days)
        
        if spy_return is not None:
            return asset_return - spy_return
        
        return None
    
    def label_candidate_outcome(
        self,
        candidate_id: str,
        symbol: str,
        decision_date: date,
        entry_price: float,
        was_selected: bool,
        realized_pnl: Optional[float] = None,
        holding_days: Optional[int] = None,
    ) -> bool:
        """
        Label a single candidate with outcomes.
        
        Args:
            candidate_id: Candidate record ID
            symbol: Asset symbol
            decision_date: Date of decision
            entry_price: Entry price
            was_selected: Whether this candidate was chosen
            realized_pnl: Actual P&L if selected
            holding_days: Actual holding period if selected
            
        Returns:
            True if successful
        """
        try:
            # Compute forward returns
            forward_returns = self.compute_forward_returns(
                symbol, decision_date, entry_price
            )
            
            # Compute realized return if selected
            realized_return = None
            if was_selected and realized_pnl is not None and entry_price > 0:
                # Approximate realized return from P&L
                # This is simplified - real implementation needs position size
                realized_return = realized_pnl / entry_price
            
            # Update candidate record
            self.persistence.update_candidate_outcomes(
                candidate_id=candidate_id,
                outcome_1d=forward_returns.get("1d"),
                outcome_5d=forward_returns.get("5d"),
                outcome_21d=forward_returns.get("21d"),
                outcome_63d=forward_returns.get("63d"),
                realized_pnl=realized_pnl,
                realized_return=realized_return,
                holding_days=holding_days,
            )
            
            return True
            
        except Exception as e:
            logger.error(f"Error labeling candidate {candidate_id}: {e}")
            return False
    
    def label_experience_outcome(
        self,
        experience_id: str,
        symbol: str,
        decision_date: date,
        entry_price: float,
        exit_price: Optional[float] = None,
        commission: Optional[float] = None,
        slippage_bps: Optional[float] = None,
    ) -> bool:
        """
        Label an experience record with outcomes.
        
        Args:
            experience_id: Experience record ID
            symbol: Asset symbol
            decision_date: Date of decision
            entry_price: Entry price
            exit_price: Actual exit price (if closed)
            commission: Commission paid
            slippage_bps: Slippage in basis points
            
        Returns:
            True if successful
        """
        try:
            # Compute forward returns
            forward_returns = self.compute_forward_returns(
                symbol, decision_date, entry_price
            )
            
            # Compute alpha
            outcome_21d = forward_returns.get("21d")
            alpha = None
            if outcome_21d is not None:
                alpha = self.compute_alpha_vs_spy(
                    outcome_21d, decision_date, horizon_days=21
                )
            
            # Determine win/loss
            win = outcome_21d > 0 if outcome_21d is not None else None
            
            # Update experience record
            self.persistence.update_experience_outcomes(
                experience_id=experience_id,
                outcome_5d=forward_returns.get("5d"),
                outcome_21d=outcome_21d,
                outcome_63d=forward_returns.get("63d"),
                realized_commission=commission,
                realized_slippage_bps=slippage_bps,
                win=win,
                alpha_vs_spy=alpha,
            )
            
            return True
            
        except Exception as e:
            logger.error(f"Error labeling experience {experience_id}: {e}")
            return False


def create_outcome_labeler(
    data_loader: HistoricalDataLoader,
    persistence: BacktestPersistence,
) -> OutcomeLabeler:
    """
    Factory function to create outcome labeler.
    
    Args:
        data_loader: Data loader instance
        persistence: Persistence instance
        
    Returns:
        Configured OutcomeLabeler
    """
    return OutcomeLabeler(data_loader, persistence)
