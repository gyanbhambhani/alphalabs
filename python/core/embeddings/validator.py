"""
Data Validator

Quality validation for stock data before embedding generation.
Detects common data issues that would produce unreliable embeddings.
"""
import logging
from dataclasses import dataclass, field
from datetime import timedelta
from typing import List, Optional, Tuple

import pandas as pd
import numpy as np

from core.embeddings.config import EmbeddingPipelineConfig, DEFAULT_CONFIG

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result of data validation."""
    is_valid: bool
    issues: List[str] = field(default_factory=list)  # Critical issues (fail)
    warnings: List[str] = field(default_factory=list)  # Non-critical warnings
    stats: dict = field(default_factory=dict)  # Validation statistics
    
    def __bool__(self) -> bool:
        return self.is_valid
    
    def summary(self) -> str:
        """Human-readable summary of validation result."""
        status = "VALID" if self.is_valid else "INVALID"
        parts = [f"[{status}]"]
        
        if self.issues:
            parts.append(f"Issues: {', '.join(self.issues)}")
        if self.warnings:
            parts.append(f"Warnings: {', '.join(self.warnings)}")
            
        return " | ".join(parts)


class DataValidator:
    """
    Validates stock data quality for embedding generation.
    
    Checks for:
    - Missing/NaN values beyond threshold
    - Zero or negative prices
    - Unrealistic daily returns (possible data errors)
    - Trading halts (consecutive zero volume days)
    - Date gaps (missing trading days)
    - Insufficient data length
    """
    
    def __init__(self, config: EmbeddingPipelineConfig = None):
        self.config = config or DEFAULT_CONFIG
    
    def validate(
        self,
        df: pd.DataFrame,
        symbol: str,
        min_rows: Optional[int] = None
    ) -> ValidationResult:
        """
        Validate a stock DataFrame for data quality.
        
        Args:
            df: Normalized DataFrame with close, high, low columns
            symbol: Stock symbol for logging
            min_rows: Minimum required rows (default: config.indicator_lookback_days)
            
        Returns:
            ValidationResult with issues, warnings, and statistics
        """
        if min_rows is None:
            min_rows = self.config.indicator_lookback_days
        
        issues = []
        warnings = []
        stats = {}
        
        # Check 1: Sufficient data length
        if len(df) < min_rows:
            issues.append(f"Insufficient data: {len(df)} rows, need {min_rows}")
            stats['row_count'] = len(df)
            return ValidationResult(False, issues, warnings, stats)
        
        stats['row_count'] = len(df)
        
        # Check 2: NaN values in required columns
        nan_issues = self._check_nan_values(df, symbol)
        for issue, stat_key, stat_val in nan_issues:
            if issue.startswith("WARN:"):
                warnings.append(issue[5:].strip())
            else:
                issues.append(issue)
            stats[stat_key] = stat_val
        
        # Check 3: Zero or negative prices
        price_issues = self._check_price_validity(df, symbol)
        issues.extend(price_issues)
        
        # Check 4: Unrealistic returns
        return_warnings = self._check_unrealistic_returns(df, symbol)
        warnings.extend(return_warnings)
        
        # Check 5: Trading halts
        halt_warnings = self._check_trading_halts(df, symbol)
        warnings.extend(halt_warnings)
        
        # Check 6: Date gaps
        gap_warnings = self._check_date_gaps(df, symbol)
        warnings.extend(gap_warnings)
        
        is_valid = len(issues) == 0
        
        if not is_valid:
            logger.warning(f"{symbol}: Validation failed - {issues}")
        elif warnings:
            logger.info(f"{symbol}: Validation passed with warnings - {warnings}")
        
        return ValidationResult(is_valid, issues, warnings, stats)
    
    def _check_nan_values(
        self,
        df: pd.DataFrame,
        symbol: str
    ) -> List[Tuple[str, str, float]]:
        """Check for NaN values in critical columns."""
        results = []
        
        for col in ['close', 'high', 'low']:
            if col not in df.columns:
                continue
                
            nan_count = df[col].isna().sum()
            nan_ratio = nan_count / len(df)
            
            if nan_ratio > self.config.max_nan_ratio:
                results.append((
                    f"{col} has {nan_ratio:.1%} NaN values (>{self.config.max_nan_ratio:.0%})",
                    f"nan_ratio_{col}",
                    nan_ratio
                ))
            elif nan_ratio > 0:
                results.append((
                    f"WARN: {col} has {nan_count} NaN values ({nan_ratio:.1%})",
                    f"nan_ratio_{col}",
                    nan_ratio
                ))
        
        return results
    
    def _check_price_validity(self, df: pd.DataFrame, symbol: str) -> List[str]:
        """Check for zero or negative prices."""
        issues = []
        
        if 'close' in df.columns:
            zero_prices = (df['close'] <= 0).sum()
            if zero_prices > 0:
                issues.append(f"Found {zero_prices} zero/negative close prices")
        
        return issues
    
    def _check_unrealistic_returns(self, df: pd.DataFrame, symbol: str) -> List[str]:
        """Check for suspiciously large daily returns."""
        warnings = []
        
        if 'close' not in df.columns:
            return warnings
        
        returns = df['close'].pct_change().dropna()
        extreme_returns = returns[abs(returns) > self.config.max_daily_return]
        
        if len(extreme_returns) > 0:
            max_return = extreme_returns.abs().max()
            warnings.append(
                f"Found {len(extreme_returns)} days with >{self.config.max_daily_return:.0%} "
                f"returns (max: {max_return:.1%})"
            )
        
        return warnings
    
    def _check_trading_halts(self, df: pd.DataFrame, symbol: str) -> List[str]:
        """Check for potential trading halts (consecutive zero volume days)."""
        warnings = []
        
        if 'volume' not in df.columns:
            return warnings
        
        # Find consecutive zero volume days
        zero_volume = (df['volume'] == 0).astype(int)
        
        # Rolling sum to find consecutive zeros
        consecutive = zero_volume.rolling(self.config.min_volume_days).sum()
        halt_periods = (consecutive == self.config.min_volume_days).sum()
        
        if halt_periods > 0:
            warnings.append(
                f"Found {halt_periods} potential trading halt periods "
                f"({self.config.min_volume_days}+ consecutive zero volume days)"
            )
        
        return warnings
    
    def _check_date_gaps(self, df: pd.DataFrame, symbol: str) -> List[str]:
        """Check for unexpected gaps in trading dates."""
        warnings = []
        
        if not isinstance(df.index, pd.DatetimeIndex):
            return warnings
        
        # Calculate day differences
        date_diffs = df.index.to_series().diff().dt.days.dropna()
        
        # Find gaps larger than threshold (excluding weekends)
        large_gaps = date_diffs[date_diffs > self.config.max_price_gap_days]
        
        if len(large_gaps) > 0:
            max_gap = large_gaps.max()
            warnings.append(
                f"Found {len(large_gaps)} date gaps >{self.config.max_price_gap_days} days "
                f"(max: {max_gap} days)"
            )
        
        return warnings
    
    def validate_or_reject(
        self,
        df: pd.DataFrame,
        symbol: str
    ) -> Tuple[bool, Optional[str]]:
        """
        Validate and return simple pass/fail with reason.
        
        Follows the pattern used in other modules (e.g., snapshot.validate_or_reject).
        
        Args:
            df: DataFrame to validate
            symbol: Stock symbol
            
        Returns:
            (is_valid, rejection_reason) tuple
        """
        result = self.validate(df, symbol)
        
        if result.is_valid:
            return True, None
        else:
            return False, "; ".join(result.issues)


def calculate_forward_return(
    start_price: float,
    end_price: float
) -> Optional[float]:
    """
    Calculate forward return with proper edge case handling.
    
    Args:
        start_price: Starting price
        end_price: Ending price
        
    Returns:
        Forward return as decimal (e.g., 0.05 for 5%), or None if invalid
    """
    # Handle NaN
    if pd.isna(start_price) or pd.isna(end_price):
        return None
    
    # Handle zero or negative prices
    if start_price <= 0 or end_price <= 0:
        return None
    
    # Handle numpy types
    try:
        start_price = float(start_price)
        end_price = float(end_price)
    except (ValueError, TypeError):
        return None
    
    return (end_price - start_price) / start_price


def calculate_forward_returns(
    df: pd.DataFrame,
    current_idx: int,
    horizons: List[Tuple[int, str]] = None
) -> dict:
    """
    Calculate forward returns for multiple horizons.
    
    Args:
        df: DataFrame with 'close' column
        current_idx: Current index position in DataFrame
        horizons: List of (days, label) tuples, default: [(5, '5d'), (10, '10d'), ...]
        
    Returns:
        Dict mapping f'forward_{label}_return' to return value (or None)
    """
    if horizons is None:
        horizons = [(5, '5d'), (10, '10d'), (21, '1m'), (63, '3m')]
    
    if 'close' not in df.columns:
        return {}
    
    close = df['close']
    results = {}
    
    for days, label in horizons:
        future_idx = current_idx + days
        if future_idx < len(df):
            ret = calculate_forward_return(
                close.iloc[current_idx],
                close.iloc[future_idx]
            )
            results[f'forward_{label}_return'] = ret
    
    return results
