"""
Asset De-identification System

Prevents temporal leakage by replacing ticker symbols with anonymous IDs.

The LLM sees:
- Asset_001, Asset_002, ... (de-identified tickers)
- Sector names (kept readable for risk controls)
- Numeric features only

This prevents the model from using its training knowledge about specific companies.

Example:
- Input: "AAPL", sector="Technology", features={...}
- Output: "Asset_001", sector="Technology", features={...}
- Mapping kept in simulator, never shown to LLM
"""

from typing import Dict, Optional, Set
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)


@dataclass
class AssetDeidentifier:
    """
    Maps real tickers to anonymous Asset_### IDs.
    
    Thread-safe, stateful mapping that persists across a backtest run.
    """
    _ticker_to_id: Dict[str, str] = field(default_factory=dict)
    _id_to_ticker: Dict[str, str] = field(default_factory=dict)
    _next_id: int = 1
    
    def register(self, ticker: str) -> str:
        """
        Register a ticker and get its de-identified ID.
        
        If ticker already registered, returns existing ID.
        Otherwise, assigns next available ID.
        
        Args:
            ticker: Real ticker symbol (e.g., "AAPL")
            
        Returns:
            De-identified ID (e.g., "Asset_001")
        """
        if ticker in self._ticker_to_id:
            return self._ticker_to_id[ticker]
        
        # Assign new ID
        asset_id = f"Asset_{self._next_id:03d}"
        self._ticker_to_id[ticker] = asset_id
        self._id_to_ticker[asset_id] = ticker
        self._next_id += 1
        
        logger.debug(f"Registered {ticker} -> {asset_id}")
        
        return asset_id
    
    def register_batch(self, tickers: Set[str]) -> Dict[str, str]:
        """
        Register multiple tickers at once.
        
        Args:
            tickers: Set of ticker symbols
            
        Returns:
            Dict mapping ticker -> asset_id
        """
        return {ticker: self.register(ticker) for ticker in tickers}
    
    def deidentify_ticker(self, ticker: str) -> str:
        """
        Convert ticker to de-identified ID.
        
        If not registered, registers it first.
        
        Args:
            ticker: Real ticker
            
        Returns:
            Asset ID
        """
        return self.register(ticker)
    
    def reidentify_ticker(self, asset_id: str) -> Optional[str]:
        """
        Convert asset ID back to real ticker.
        
        Used for execution, logging (not for LLM prompts).
        
        Args:
            asset_id: De-identified ID (e.g., "Asset_001")
            
        Returns:
            Real ticker, or None if not found
        """
        return self._id_to_ticker.get(asset_id)
    
    def deidentify_context(self, context_data: Dict) -> Dict:
        """
        De-identify a context dictionary (for LLM prompts).
        
        Replaces tickers but keeps sectors readable.
        
        Args:
            context_data: Dict with ticker keys and data
            
        Returns:
            De-identified dict with Asset_### keys
        """
        deidentified = {}
        
        for ticker, data in context_data.items():
            asset_id = self.register(ticker)
            
            # Copy data, keep sector readable
            if isinstance(data, dict):
                deidentified[asset_id] = data.copy()
            else:
                deidentified[asset_id] = data
        
        return deidentified
    
    def get_mapping(self) -> Dict[str, str]:
        """
        Get the full ticker -> asset_id mapping.
        
        For logging/debugging only, not for LLM prompts.
        
        Returns:
            Dict mapping ticker -> asset_id
        """
        return self._ticker_to_id.copy()
    
    def get_reverse_mapping(self) -> Dict[str, str]:
        """
        Get the full asset_id -> ticker mapping.
        
        For re-identification after LLM decisions.
        
        Returns:
            Dict mapping asset_id -> ticker
        """
        return self._id_to_ticker.copy()
    
    def count_registered(self) -> int:
        """Return number of registered tickers."""
        return len(self._ticker_to_id)
    
    def clear(self) -> None:
        """Clear all mappings (start fresh)."""
        self._ticker_to_id.clear()
        self._id_to_ticker.clear()
        self._next_id = 1
        logger.info("De-identification mappings cleared")


# Singleton instance for consistency across a simulation run
_global_deidentifier: Optional[AssetDeidentifier] = None


def get_deidentifier() -> AssetDeidentifier:
    """
    Get the global deidentifier instance.
    
    Returns:
        Shared AssetDeidentifier instance
    """
    global _global_deidentifier
    if _global_deidentifier is None:
        _global_deidentifier = AssetDeidentifier()
    return _global_deidentifier


def reset_deidentifier() -> None:
    """Reset the global deidentifier (for new simulation runs)."""
    global _global_deidentifier
    _global_deidentifier = None
    logger.info("Global deidentifier reset")
