"""
Quant Bot - Pure Systematic Trading Baseline

No LLM, no reasoning. Fixed rules combining signals with predetermined weights.
This is the baseline to answer: "Do LLMs add value over algorithms?"
"""
from typing import List, Dict
from core.managers.base import (
    BaseManager, TradingDecision, ManagerContext, 
    Action, RiskLimits
)


class QuantBot(BaseManager):
    """
    Pure systematic trading bot with fixed rules.
    
    Combines signals using predetermined weights:
    - 30% momentum
    - 20% mean reversion  
    - 20% ML prediction
    - 30% semantic search outcomes
    
    Trading rules:
    - BUY if combined score > 0.6 and regime is trending
    - SELL if combined score < -0.6
    - Position size proportional to score
    """
    
    # Signal weights
    MOMENTUM_WEIGHT = 0.30
    MEAN_REVERSION_WEIGHT = 0.20
    ML_PREDICTION_WEIGHT = 0.20
    SEMANTIC_WEIGHT = 0.30
    
    # Thresholds
    BUY_THRESHOLD = 0.5
    SELL_THRESHOLD = -0.5
    
    # Position sizing
    BASE_POSITION_SIZE = 0.10  # 10% of portfolio per trade
    MAX_POSITION_SIZE = 0.20  # 20% max
    
    def __init__(
        self,
        manager_id: str = "quant_bot",
        name: str = "Quant Bot",
        initial_capital: float = 25000.0,
        risk_limits: RiskLimits | None = None
    ):
        super().__init__(
            manager_id=manager_id,
            name=name,
            manager_type="quant",
            initial_capital=initial_capital,
            risk_limits=risk_limits
        )
    
    def calculate_combined_score(
        self,
        symbol: str,
        signals: Dict
    ) -> float:
        """
        Calculate combined score for a symbol using fixed weights.
        
        Returns score from -1 to +1.
        """
        momentum = signals.get("momentum", {}).get(symbol, 0.0)
        mean_rev = signals.get("mean_reversion", {}).get(symbol, 0.0)
        ml_pred = signals.get("ml_prediction", {}).get(symbol, 0.0)
        
        # Normalize ML prediction (assumed to be return, e.g., 0.02 for 2%)
        # Scale to -1 to +1 range (assuming ±5% is extreme)
        ml_normalized = max(-1, min(1, ml_pred / 0.05))
        
        # Semantic search outcome score
        semantic = signals.get("semantic_search", {})
        semantic_score = 0.0
        if semantic:
            # Use positive rate and average return to create score
            positive_rate = semantic.get("positive_5d_rate", 0.5)
            avg_return = semantic.get("avg_5d_return", 0.0)
            # Score based on historical outcomes
            semantic_score = (positive_rate - 0.5) * 2 + avg_return * 10
            semantic_score = max(-1, min(1, semantic_score))
        
        # Weighted combination
        combined = (
            self.MOMENTUM_WEIGHT * momentum +
            self.MEAN_REVERSION_WEIGHT * mean_rev +
            self.ML_PREDICTION_WEIGHT * ml_normalized +
            self.SEMANTIC_WEIGHT * semantic_score
        )
        
        return combined
    
    def calculate_position_size(self, score: float) -> float:
        """
        Calculate position size based on signal strength.
        
        Stronger signals = larger positions (within limits).
        """
        # Linear scaling from base to max based on score
        score_magnitude = abs(score)
        size = self.BASE_POSITION_SIZE + (
            (self.MAX_POSITION_SIZE - self.BASE_POSITION_SIZE) 
            * score_magnitude
        )
        return min(size, self.MAX_POSITION_SIZE)
    
    def should_trade_in_regime(self, regime: str, action: Action) -> bool:
        """
        Check if trading is appropriate for current regime.
        
        - Trending regimes favor momentum (buys)
        - High vol regimes are more cautious
        """
        if "high_vol" in regime:
            # More cautious in high volatility
            return False
        
        if action == Action.BUY:
            # Buy in trending or low vol environments
            return "trending_up" in regime or "low_vol" in regime
        
        return True  # Sells are always allowed
    
    async def make_decisions(
        self, 
        context: ManagerContext
    ) -> List[TradingDecision]:
        """
        Make trading decisions using fixed systematic rules.
        
        No reasoning, no interpretation - just signal combination.
        """
        decisions = []
        regime = context.signals.volatility_regime
        
        # Build signals dict for easier access
        signals = {
            "momentum": context.signals.momentum,
            "mean_reversion": context.signals.mean_reversion,
            "ml_prediction": context.signals.ml_prediction,
            "semantic_search": context.signals.semantic_search
        }
        
        # Get all symbols we have signals for
        all_symbols = set()
        for signal_type in ["momentum", "mean_reversion", "ml_prediction"]:
            all_symbols.update(signals.get(signal_type, {}).keys())
        
        for symbol in all_symbols:
            # Calculate combined score
            score = self.calculate_combined_score(symbol, signals)
            
            # Current position
            current_position = self.portfolio.positions.get(symbol)
            has_position = current_position is not None and current_position.quantity > 0
            
            # Decision logic
            if score > self.BUY_THRESHOLD and not has_position:
                # Buy signal
                if self.should_trade_in_regime(regime, Action.BUY):
                    size = self.calculate_position_size(score)
                    decisions.append(TradingDecision(
                        action=Action.BUY,
                        symbol=symbol,
                        size=size,
                        reasoning=(
                            f"Combined score {score:.2f} > {self.BUY_THRESHOLD} "
                            f"threshold in {regime} regime"
                        ),
                        confidence=abs(score),
                        signals_used={
                            "momentum": signals["momentum"].get(symbol, 0),
                            "mean_reversion": signals["mean_reversion"].get(symbol, 0),
                            "ml_prediction": signals["ml_prediction"].get(symbol, 0),
                            "combined_score": score
                        }
                    ))
            
            elif score < self.SELL_THRESHOLD and has_position:
                # Sell signal
                decisions.append(TradingDecision(
                    action=Action.SELL,
                    symbol=symbol,
                    size=1.0,  # Sell entire position
                    reasoning=(
                        f"Combined score {score:.2f} < {self.SELL_THRESHOLD} "
                        f"threshold - closing position"
                    ),
                    confidence=abs(score),
                    signals_used={
                        "momentum": signals["momentum"].get(symbol, 0),
                        "mean_reversion": signals["mean_reversion"].get(symbol, 0),
                        "ml_prediction": signals["ml_prediction"].get(symbol, 0),
                        "combined_score": score
                    }
                ))
        
        # Apply risk limits
        return self.apply_risk_limits(decisions, context)
