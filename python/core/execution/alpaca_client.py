"""
Alpaca Trading Client

Wrapper for Alpaca API for paper and live trading.
"""
from typing import Dict, List, Optional, Literal
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal


@dataclass
class Order:
    """Order representation"""
    id: str
    symbol: str
    side: Literal["buy", "sell"]
    quantity: float
    order_type: str
    status: str
    filled_price: Optional[float] = None
    filled_at: Optional[datetime] = None
    submitted_at: Optional[datetime] = None


@dataclass
class AccountInfo:
    """Account information"""
    equity: float
    cash: float
    buying_power: float
    portfolio_value: float


class AlpacaClient:
    """
    Alpaca trading client for paper/live trading.
    
    Supports:
    - Market and limit orders
    - Position management
    - Account information
    - Market data
    """
    
    def __init__(
        self,
        api_key: str,
        secret_key: str,
        paper: bool = True
    ):
        """
        Initialize Alpaca client.
        
        Args:
            api_key: Alpaca API key
            secret_key: Alpaca secret key
            paper: Use paper trading (default True)
        """
        self.api_key = api_key
        self.secret_key = secret_key
        self.paper = paper
        self._trading_client = None
        self._data_client = None
    
    def _get_trading_client(self):
        """Lazily initialize trading client"""
        if self._trading_client is None:
            try:
                from alpaca.trading.client import TradingClient
                self._trading_client = TradingClient(
                    api_key=self.api_key,
                    secret_key=self.secret_key,
                    paper=self.paper
                )
            except ImportError:
                print("alpaca-py not installed. Using mock client.")
                return None
        return self._trading_client
    
    def _get_data_client(self):
        """Lazily initialize data client"""
        if self._data_client is None:
            try:
                from alpaca.data.historical import StockHistoricalDataClient
                self._data_client = StockHistoricalDataClient(
                    api_key=self.api_key,
                    secret_key=self.secret_key
                )
            except ImportError:
                return None
        return self._data_client
    
    def get_account(self) -> AccountInfo:
        """Get account information"""
        client = self._get_trading_client()
        
        if client is None:
            # Return mock data
            return AccountInfo(
                equity=100000.0,
                cash=100000.0,
                buying_power=100000.0,
                portfolio_value=100000.0
            )
        
        account = client.get_account()
        
        return AccountInfo(
            equity=float(account.equity),
            cash=float(account.cash),
            buying_power=float(account.buying_power),
            portfolio_value=float(account.portfolio_value)
        )
    
    def get_positions(self) -> Dict[str, Dict]:
        """Get all current positions"""
        client = self._get_trading_client()
        
        if client is None:
            return {}
        
        positions = client.get_all_positions()
        
        return {
            pos.symbol: {
                "quantity": float(pos.qty),
                "avg_entry_price": float(pos.avg_entry_price),
                "current_price": float(pos.current_price),
                "market_value": float(pos.market_value),
                "unrealized_pl": float(pos.unrealized_pl),
                "unrealized_plpc": float(pos.unrealized_plpc)
            }
            for pos in positions
        }
    
    def submit_market_order(
        self,
        symbol: str,
        quantity: float,
        side: Literal["buy", "sell"]
    ) -> Optional[Order]:
        """
        Submit a market order.
        
        Args:
            symbol: Stock symbol
            quantity: Number of shares
            side: 'buy' or 'sell'
        
        Returns:
            Order object or None if failed
        """
        client = self._get_trading_client()
        
        if client is None:
            # Return mock order
            return Order(
                id=f"mock_{datetime.now().timestamp()}",
                symbol=symbol,
                side=side,
                quantity=quantity,
                order_type="market",
                status="filled",
                filled_price=100.0,
                filled_at=datetime.now(),
                submitted_at=datetime.now()
            )
        
        try:
            from alpaca.trading.requests import MarketOrderRequest
            from alpaca.trading.enums import OrderSide, TimeInForce
            
            order_data = MarketOrderRequest(
                symbol=symbol,
                qty=quantity,
                side=OrderSide.BUY if side == "buy" else OrderSide.SELL,
                time_in_force=TimeInForce.DAY
            )
            
            order = client.submit_order(order_data)
            
            return Order(
                id=str(order.id),
                symbol=order.symbol,
                side=side,
                quantity=float(order.qty),
                order_type="market",
                status=order.status.value,
                submitted_at=order.submitted_at
            )
        
        except Exception as e:
            print(f"Error submitting order: {e}")
            return None
    
    def submit_limit_order(
        self,
        symbol: str,
        quantity: float,
        side: Literal["buy", "sell"],
        limit_price: float
    ) -> Optional[Order]:
        """Submit a limit order"""
        client = self._get_trading_client()
        
        if client is None:
            return Order(
                id=f"mock_{datetime.now().timestamp()}",
                symbol=symbol,
                side=side,
                quantity=quantity,
                order_type="limit",
                status="submitted",
                submitted_at=datetime.now()
            )
        
        try:
            from alpaca.trading.requests import LimitOrderRequest
            from alpaca.trading.enums import OrderSide, TimeInForce
            
            order_data = LimitOrderRequest(
                symbol=symbol,
                qty=quantity,
                side=OrderSide.BUY if side == "buy" else OrderSide.SELL,
                time_in_force=TimeInForce.DAY,
                limit_price=limit_price
            )
            
            order = client.submit_order(order_data)
            
            return Order(
                id=str(order.id),
                symbol=order.symbol,
                side=side,
                quantity=float(order.qty),
                order_type="limit",
                status=order.status.value,
                submitted_at=order.submitted_at
            )
        
        except Exception as e:
            print(f"Error submitting limit order: {e}")
            return None
    
    def cancel_order(self, order_id: str) -> bool:
        """Cancel an order"""
        client = self._get_trading_client()
        
        if client is None:
            return True
        
        try:
            client.cancel_order_by_id(order_id)
            return True
        except Exception as e:
            print(f"Error canceling order: {e}")
            return False
    
    def get_latest_price(self, symbol: str) -> Optional[float]:
        """Get latest price for a symbol"""
        data_client = self._get_data_client()
        
        if data_client is None:
            return None
        
        try:
            from alpaca.data.requests import StockLatestQuoteRequest
            
            request = StockLatestQuoteRequest(symbol_or_symbols=symbol)
            quote = data_client.get_stock_latest_quote(request)
            
            return float(quote[symbol].ask_price)
        
        except Exception as e:
            print(f"Error getting price for {symbol}: {e}")
            return None
    
    def get_latest_prices(self, symbols: List[str]) -> Dict[str, float]:
        """Get latest prices for multiple symbols"""
        prices = {}
        
        for symbol in symbols:
            price = self.get_latest_price(symbol)
            if price:
                prices[symbol] = price
        
        return prices
