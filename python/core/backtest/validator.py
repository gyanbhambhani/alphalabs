"""
Decision Validator

Validates LLM outputs against hard constraints:
1. Budget compliance (buys only when authorized)
2. Asset in candidate set
3. Weight bounds
4. Factors in allowlist
5. Forbidden tokens (no narrative leakage)

Rejects decisions that violate any constraint.
"""

from typing import Dict, Set, List, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


# Forbidden tokens that indicate narrative/temporal leakage
FORBIDDEN_TOKENS = [
    # Major tickers (common ones - expand as needed)
    "aapl", "msft", "goog", "googl", "amzn", "tsla", "meta", "nvda",
    "fb", "nflx", "baba", "jpm", "wmt", "v", "ma", "dis",
    
    # Products/brands
    "iphone", "ipad", "macbook", "windows", "surface", "azure",
    "model 3", "model s", "model x", "model y", "cybertruck",
    "aws", "prime", "alexa", "kindle", "echo",
    "instagram", "whatsapp", "oculus", "quest",
    
    # Narrative keywords
    "product launch", "services growth", "innovation",
    "market leader", "dominant", "disruption", "disruptive",
    "game-changer", "revolutionary", "iconic brand",
    "brand strength", "brand value", "brand recognition",
    
    # Events/news (temporal leakage indicators)
    "pandemic", "covid", "coronavirus", "lockdown",
    "chip shortage", "supply chain", "earnings beat",
    "guidance raised", "merger", "acquisition",
]


@dataclass
class ValidationResult:
    """Result of decision validation."""
    valid: bool
    violations: List[str] = None
    reason: Optional[str] = None
    
    def __post_init__(self):
        if self.violations is None:
            self.violations = []
        if not self.valid and self.reason is None and self.violations:
            self.reason = "; ".join(self.violations)


class DecisionValidator:
    """
    Validates LLM decisions against hard constraints.
    
    Acts as final safety gate before execution.
    """
    
    def __init__(
        self,
        max_position_pct: float = 0.20,
        allowed_features: Optional[Set[str]] = None,
        forbidden_tokens: Optional[List[str]] = None,
    ):
        """
        Initialize validator.
        
        Args:
            max_position_pct: Maximum position weight (0-1)
            allowed_features: Set of allowed feature names
            forbidden_tokens: List of forbidden tokens (lowercase)
        """
        self.max_position_pct = max_position_pct
        self.allowed_features = allowed_features or set()
        self.forbidden_tokens = forbidden_tokens or FORBIDDEN_TOKENS
        
        # Convert to lowercase for case-insensitive matching
        self.forbidden_tokens_lower = [t.lower() for t in self.forbidden_tokens]
    
    def validate(
        self,
        decision: Dict,
        budget: Optional["TradeBudget"] = None,
        candidates: Optional[List[str]] = None,
    ) -> ValidationResult:
        """
        Validate decision against all constraints.
        
        Args:
            decision: LLM decision dict with action, asset_id, weight, etc.
            budget: Optional TradeBudget for budget checks
            candidates: Optional list of valid asset IDs
            
        Returns:
            ValidationResult with valid flag and violations
        """
        violations = []
        
        # 1. Budget check
        if budget and decision.get("action") == "buy":
            if not budget.can_buy():
                violations.append(
                    f"Budget exhausted - buys not authorized "
                    f"({budget.trades_this_week}/{budget.max_trades_per_week} used)"
                )
        
        # 2. Asset in candidate set
        if candidates:
            asset_id = decision.get("asset_id") or decision.get("symbol")
            if asset_id and asset_id not in candidates:
                violations.append(
                    f"Asset '{asset_id}' not in candidate set "
                    f"({len(candidates)} candidates)"
                )
        
        # 3. Weight bounds
        target_weight = decision.get("target_weight")
        if target_weight is not None:
            if target_weight > self.max_position_pct:
                violations.append(
                    f"Weight {target_weight:.1%} exceeds max "
                    f"{self.max_position_pct:.1%}"
                )
            if target_weight < 0:
                violations.append(
                    f"Weight {target_weight:.1%} is negative "
                    f"(short selling not supported)"
                )
        
        # 4. Factors in allowlist
        factors_used = decision.get("factors_used", {})
        if self.allowed_features and factors_used:
            for factor in factors_used.keys():
                if factor not in self.allowed_features:
                    violations.append(
                        f"Factor '{factor}' not in allowlist "
                        f"({len(self.allowed_features)} allowed)"
                    )
                    break  # Only report first violation
        
        # 5. Forbidden tokens in reasoning
        reasoning = decision.get("reasoning", "")
        if reasoning:
            forbidden_found = self._check_forbidden_tokens(reasoning)
            if forbidden_found:
                violations.append(
                    f"Reasoning contains forbidden token(s): {forbidden_found[:3]}"
                )
        
        # Build result
        if violations:
            return ValidationResult(
                valid=False,
                violations=violations,
            )
        
        return ValidationResult(valid=True)
    
    def _check_forbidden_tokens(self, text: str) -> List[str]:
        """
        Check if text contains any forbidden tokens.
        
        Uses word boundary matching to avoid false positives
        (e.g., "v" in "volatility" shouldn't match ticker "V").
        
        Args:
            text: Text to check (reasoning, analysis, etc.)
            
        Returns:
            List of forbidden tokens found (lowercase)
        """
        import re
        text_lower = text.lower()
        found = []
        
        for token in self.forbidden_tokens_lower:
            # Use word boundaries for single/short tokens
            if len(token) <= 2:
                # Match as whole word only
                pattern = r'\b' + re.escape(token) + r'\b'
                if re.search(pattern, text_lower):
                    found.append(token)
            else:
                # Simple substring match for longer tokens
                if token in text_lower:
                    found.append(token)
        
        return found
    
    def validate_or_hold(
        self,
        decision: Dict,
        budget: Optional["TradeBudget"] = None,
        candidates: Optional[List[str]] = None,
    ) -> tuple[Dict, ValidationResult]:
        """
        Validate decision, convert to HOLD if invalid.
        
        Safety wrapper that ensures we always return a valid decision.
        
        Args:
            decision: Original LLM decision
            budget: Optional TradeBudget
            candidates: Optional candidate set
            
        Returns:
            Tuple of (possibly_modified_decision, validation_result)
        """
        result = self.validate(decision, budget, candidates)
        
        if not result.valid:
            logger.warning(
                f"Decision rejected: {result.reason}. Converting to HOLD."
            )
            
            # Convert to safe HOLD decision
            safe_decision = {
                "action": "hold",
                "symbol": None,
                "asset_id": None,
                "target_weight": None,
                "reasoning": f"Validator rejected: {result.reason}",
                "confidence": 0.0,
                "factors_used": {},
            }
            
            return safe_decision, result
        
        return decision, result
    
    def add_allowed_feature(self, feature: str) -> None:
        """Add a feature to the allowlist."""
        self.allowed_features.add(feature)
    
    def add_forbidden_token(self, token: str) -> None:
        """Add a token to the forbidden list."""
        token_lower = token.lower()
        if token_lower not in self.forbidden_tokens_lower:
            self.forbidden_tokens.append(token)
            self.forbidden_tokens_lower.append(token_lower)
            logger.info(f"Added forbidden token: {token}")


def create_validator_for_fund(
    fund_policy: "FundPolicy",
    snapshot: "GlobalMarketSnapshot",
) -> DecisionValidator:
    """
    Create a validator from fund policy and snapshot.
    
    Args:
        fund_policy: Fund's policy with position limits
        snapshot: Market snapshot with available features
        
    Returns:
        Configured DecisionValidator
    """
    allowed_features = snapshot.available_features()
    max_position = fund_policy.max_position_pct
    
    return DecisionValidator(
        max_position_pct=max_position,
        allowed_features=allowed_features,
        forbidden_tokens=FORBIDDEN_TOKENS,
    )
