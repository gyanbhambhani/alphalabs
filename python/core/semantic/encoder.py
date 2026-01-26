"""
Market State Encoder

Encodes market conditions into fixed-length vectors for semantic search.
Features include technical indicators, volatility metrics, and correlations.
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from dataclasses import dataclass


@dataclass
class MarketState:
    """Encoded market state at a point in time"""
    date: str
    vector: np.ndarray
    metadata: Dict


class MarketStateEncoder:
    """
    Encodes market conditions into vectors for semantic similarity search.
    
    Features encoded (50+ dimensions):
    - Price momentum (multiple timeframes)
    - Volatility metrics
    - Technical indicators (RSI, MACD, etc.)
    - Relative strength vs market
    - Volume patterns
    - Cross-asset correlations
    """
    
    # Feature dimensions
    MOMENTUM_DIMS = 5  # 1w, 1m, 3m, 6m, 12m returns
    VOLATILITY_DIMS = 4  # realized vol at different windows
    TECHNICAL_DIMS = 10  # RSI, MACD components, MAs, etc.
    VOLUME_DIMS = 3  # volume changes
    RELATIVE_DIMS = 3  # relative to market
    REGIME_DIMS = 3  # market regime indicators
    
    TOTAL_DIMS = (
        MOMENTUM_DIMS + VOLATILITY_DIMS + TECHNICAL_DIMS + 
        VOLUME_DIMS + RELATIVE_DIMS + REGIME_DIMS
    )  # 28 base dims, expanded to 512 via projection
    
    OUTPUT_DIMS = 512  # Final embedding dimension
    
    def __init__(self, random_seed: int = 42):
        """Initialize encoder with random projection matrix"""
        np.random.seed(random_seed)
        # Random projection to expand to OUTPUT_DIMS
        self.projection_matrix = np.random.randn(
            self.TOTAL_DIMS, 
            self.OUTPUT_DIMS
        ) / np.sqrt(self.TOTAL_DIMS)
    
    def _calculate_returns(
        self, 
        prices: pd.Series, 
        windows: List[int] = [5, 21, 63, 126, 252]
    ) -> np.ndarray:
        """Calculate returns over multiple windows"""
        returns = []
        for window in windows:
            if len(prices) >= window:
                ret = (prices.iloc[-1] / prices.iloc[-window]) - 1
                returns.append(ret)
            else:
                returns.append(0.0)
        return np.array(returns)
    
    def _calculate_volatility(
        self, 
        prices: pd.Series,
        windows: List[int] = [5, 10, 21, 63]
    ) -> np.ndarray:
        """Calculate realized volatility at multiple windows"""
        log_returns = np.log(prices / prices.shift(1)).dropna()
        vols = []
        for window in windows:
            if len(log_returns) >= window:
                vol = log_returns.iloc[-window:].std() * np.sqrt(252)
                vols.append(vol)
            else:
                vols.append(0.0)
        return np.array(vols)
    
    def _calculate_technical(
        self, 
        close: pd.Series,
        high: Optional[pd.Series] = None,
        low: Optional[pd.Series] = None
    ) -> np.ndarray:
        """Calculate technical indicators"""
        features = []
        
        # RSI (normalized to -1 to 1)
        if len(close) >= 15:
            delta = close.diff()
            gain = delta.where(delta > 0, 0).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rs = gain / loss.replace(0, 1e-10)
            rsi = 100 - (100 / (1 + rs))
            rsi_normalized = (rsi.iloc[-1] - 50) / 50  # -1 to 1
            features.append(rsi_normalized)
        else:
            features.append(0.0)
        
        # MACD components (normalized)
        if len(close) >= 26:
            ema12 = close.ewm(span=12).mean()
            ema26 = close.ewm(span=26).mean()
            macd = ema12 - ema26
            signal = macd.ewm(span=9).mean()
            histogram = macd - signal
            
            # Normalize by price
            price = close.iloc[-1]
            features.extend([
                macd.iloc[-1] / price * 100,
                signal.iloc[-1] / price * 100,
                histogram.iloc[-1] / price * 100
            ])
        else:
            features.extend([0.0, 0.0, 0.0])
        
        # Moving average positions (price relative to MAs)
        for window in [20, 50, 200]:
            if len(close) >= window:
                ma = close.iloc[-window:].mean()
                rel_pos = (close.iloc[-1] - ma) / ma
                features.append(rel_pos)
            else:
                features.append(0.0)
        
        # Bollinger Band position
        if len(close) >= 20:
            ma20 = close.iloc[-20:].mean()
            std20 = close.iloc[-20:].std()
            if std20 > 0:
                bb_pos = (close.iloc[-1] - ma20) / (2 * std20)  # -1 to 1 roughly
                features.append(bb_pos)
            else:
                features.append(0.0)
        else:
            features.append(0.0)
        
        # ATR as percentage of price
        if high is not None and low is not None and len(close) >= 15:
            tr = pd.concat([
                high - low,
                abs(high - close.shift(1)),
                abs(low - close.shift(1))
            ], axis=1).max(axis=1)
            atr = tr.rolling(14).mean()
            atr_pct = atr.iloc[-1] / close.iloc[-1]
            features.append(atr_pct)
        else:
            features.append(0.0)
        
        # Rate of change
        if len(close) >= 10:
            roc = (close.iloc[-1] - close.iloc[-10]) / close.iloc[-10]
            features.append(roc)
        else:
            features.append(0.0)
        
        return np.array(features[:self.TECHNICAL_DIMS])
    
    def _calculate_volume_features(
        self,
        volume: pd.Series
    ) -> np.ndarray:
        """Calculate volume-based features"""
        features = []
        
        if len(volume) >= 21:
            # Volume relative to 20-day average
            vol_avg = volume.iloc[-21:-1].mean()
            current_vol = volume.iloc[-1]
            
            if vol_avg > 0 and current_vol > 0:
                vol_ratio = current_vol / vol_avg
                # Clamp ratio to avoid extreme values
                vol_ratio = np.clip(vol_ratio, 0.01, 100.0)
                features.append(np.log1p(vol_ratio - 1))  # log-scaled
            else:
                features.append(0.0)
            
            # Volume trend (5-day vs 20-day)
            vol_5 = volume.iloc[-5:].mean()
            vol_20 = volume.iloc[-20:].mean()
            if vol_20 > 0 and vol_5 > 0:
                vol_trend = np.clip(vol_5 / vol_20 - 1, -0.99, 10.0)
                features.append(vol_trend)
            else:
                features.append(0.0)
            
            # Volume spike indicator
            vol_std = volume.iloc[-21:-1].std()
            if vol_std > 0 and not np.isnan(vol_avg):
                vol_zscore = (current_vol - vol_avg) / vol_std
                features.append(np.tanh(vol_zscore / 2))  # bounded
            else:
                features.append(0.0)
        else:
            features.extend([0.0, 0.0, 0.0])
        
        return np.array(features[:self.VOLUME_DIMS])
    
    def _calculate_relative_features(
        self,
        asset_returns: np.ndarray,
        market_returns: np.ndarray
    ) -> np.ndarray:
        """Calculate features relative to market"""
        features = []
        
        # Relative strength (multiple timeframes)
        for i in range(min(3, len(asset_returns), len(market_returns))):
            rs = asset_returns[i] - market_returns[i]
            features.append(rs)
        
        while len(features) < self.RELATIVE_DIMS:
            features.append(0.0)
        
        return np.array(features[:self.RELATIVE_DIMS])
    
    def _calculate_regime_features(
        self,
        volatility: np.ndarray,
        returns: np.ndarray
    ) -> np.ndarray:
        """Calculate market regime indicators"""
        features = []
        
        # Vol regime (low/normal/high based on 21-day vol)
        if len(volatility) > 0:
            vol_21 = volatility[2] if len(volatility) > 2 else volatility[0]
            # Map to -1 (low), 0 (normal), 1 (high)
            if vol_21 < 0.15:
                features.append(-1.0)
            elif vol_21 > 0.25:
                features.append(1.0)
            else:
                features.append(0.0)
        else:
            features.append(0.0)
        
        # Trend regime
        if len(returns) >= 2:
            # Short vs long term momentum
            short_mom = returns[1] if len(returns) > 1 else 0  # 1 month
            long_mom = returns[3] if len(returns) > 3 else 0  # 6 month
            
            if short_mom > 0.02 and long_mom > 0:
                features.append(1.0)  # Trending up
            elif short_mom < -0.02 and long_mom < 0:
                features.append(-1.0)  # Trending down
            else:
                features.append(0.0)  # Ranging
        else:
            features.append(0.0)
        
        # Momentum consistency
        if len(returns) >= 3:
            signs = np.sign(returns[:3])
            consistency = np.mean(signs)
            features.append(consistency)
        else:
            features.append(0.0)
        
        return np.array(features[:self.REGIME_DIMS])
    
    def encode(
        self,
        date: str,
        close: pd.Series,
        high: Optional[pd.Series] = None,
        low: Optional[pd.Series] = None,
        volume: Optional[pd.Series] = None,
        market_close: Optional[pd.Series] = None
    ) -> MarketState:
        """
        Encode market state into a fixed-length vector.
        
        Args:
            date: Date string for this state
            close: Close prices
            high: High prices (optional)
            low: Low prices (optional)
            volume: Volume data (optional)
            market_close: Market index close for relative features
        
        Returns:
            MarketState with encoded vector and metadata
        """
        # Calculate base features
        returns = self._calculate_returns(close)
        volatility = self._calculate_volatility(close)
        technical = self._calculate_technical(close, high, low)
        
        volume_features = (
            self._calculate_volume_features(volume) 
            if volume is not None 
            else np.zeros(self.VOLUME_DIMS)
        )
        
        market_returns = (
            self._calculate_returns(market_close)
            if market_close is not None
            else returns
        )
        relative = self._calculate_relative_features(returns, market_returns)
        
        regime = self._calculate_regime_features(volatility, returns)
        
        # Concatenate all features
        base_vector = np.concatenate([
            returns,
            volatility,
            technical,
            volume_features,
            relative,
            regime
        ])
        
        # Replace any NaN or inf values with 0
        base_vector = np.nan_to_num(base_vector, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Ensure correct dimension
        if len(base_vector) < self.TOTAL_DIMS:
            base_vector = np.pad(
                base_vector, 
                (0, self.TOTAL_DIMS - len(base_vector))
            )
        elif len(base_vector) > self.TOTAL_DIMS:
            base_vector = base_vector[:self.TOTAL_DIMS]
        
        # Project to output dimensions
        vector = np.dot(base_vector, self.projection_matrix)
        
        # Replace any NaN or inf values from projection
        vector = np.nan_to_num(vector, nan=0.0, posinf=0.0, neginf=0.0)
        
        # L2 normalize
        norm = np.linalg.norm(vector)
        if norm > 1e-10:  # More robust zero check
            vector = vector / norm
        else:
            # If vector is all zeros or near-zeros, use uniform distribution
            vector = np.ones(self.OUTPUT_DIMS) / np.sqrt(self.OUTPUT_DIMS)
        
        # Metadata for interpretability (flattened for ChromaDB)
        metadata = {
            "date": date,
            "return_1w": float(returns[0]),
            "return_1m": float(returns[1]),
            "return_3m": float(returns[2]),
            "return_6m": float(returns[3]),
            "return_12m": float(returns[4]),
            "volatility_5d": float(volatility[0]),
            "volatility_10d": float(volatility[1]),
            "volatility_21d": float(volatility[2]),
            "volatility_63d": float(volatility[3]),
            "price": float(close.iloc[-1]) if len(close) > 0 else 0
        }
        
        return MarketState(
            date=date,
            vector=vector.astype(np.float32),
            metadata=metadata
        )
    
    def encode_batch(
        self,
        data: pd.DataFrame,
        symbol: str = "SPY"
    ) -> List[MarketState]:
        """
        Encode multiple market states from historical data.
        
        Args:
            data: DataFrame with columns: date, open, high, low, close, volume
            symbol: Symbol being encoded (for metadata)
        
        Returns:
            List of MarketState objects
        """
        states = []
        
        # Need at least 252 days for all features
        min_lookback = 252
        
        for i in range(min_lookback, len(data)):
            window = data.iloc[:i+1]
            state = self.encode(
                date=str(window.index[-1].date()) 
                    if hasattr(window.index[-1], 'date') 
                    else str(window.iloc[-1].name),
                close=window['close'],
                high=window.get('high'),
                low=window.get('low'),
                volume=window.get('volume')
            )
            states.append(state)
        
        return states
