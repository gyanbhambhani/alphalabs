"""
LangChain tools for market data and portfolio operations.

These tools wrap existing functionality to be used by LangChain agents.
Tools can fetch market data, calculate metrics, and query portfolio state.
"""

from typing import Optional, Dict, Any, List
from datetime import date
from langchain_core.tools import tool

# Import existing modules for wrapping
from core.data.snapshot import GlobalMarketSnapshot


@tool
def get_market_snapshot(
    symbols: List[str],
    as_of_date: str,
) -> Dict[str, Any]:
    """
    Get point-in-time market snapshot for given symbols.
    
    Args:
        symbols: List of stock symbols (e.g., ["AAPL", "MSFT"])
        as_of_date: Date string in YYYY-MM-DD format
        
    Returns:
        Dict with prices, returns, and volatility for each symbol
    """
    from core.backtest.data_loader import HistoricalDataLoader
    
    loader = HistoricalDataLoader()
    if not loader._loaded:
        loader.load_all()
    
    target_date = date.fromisoformat(as_of_date)
    
    result = {
        "date": as_of_date,
        "symbols": {}
    }
    
    for symbol in symbols:
        price = loader.get_price_asof(symbol, target_date)
        if price is not None:
            result["symbols"][symbol] = {
                "price": price,
                "return_1d": loader.calc_return(symbol, target_date, lookback_days=1),
                "return_5d": loader.calc_return(symbol, target_date, lookback_days=5),
                "return_21d": loader.calc_return(symbol, target_date, lookback_days=21),
                "volatility_5d": loader.calc_volatility(symbol, target_date, lookback_days=5),
                "volatility_21d": loader.calc_volatility(symbol, target_date, lookback_days=21),
            }
    
    return result


@tool
def get_portfolio_state(portfolio_json: str) -> Dict[str, Any]:
    """
    Get current portfolio state summary.
    
    Args:
        portfolio_json: JSON string with portfolio data (or portfolio ID)
        
    Returns:
        Dict with portfolio summary including positions, cash, returns
    """
    import json
    
    try:
        data = json.loads(portfolio_json)
    except json.JSONDecodeError:
        return {"error": "Invalid portfolio JSON"}
    
    # Extract key metrics
    total_value = data.get("total_value", 0)
    cash = data.get("cash", 0)
    positions = data.get("positions", {})
    
    summary = {
        "total_value": total_value,
        "cash": cash,
        "cash_percentage": cash / total_value if total_value > 0 else 0,
        "num_positions": len(positions),
        "positions": []
    }
    
    for symbol, pos in positions.items():
        summary["positions"].append({
            "symbol": symbol,
            "quantity": pos.get("quantity", 0),
            "current_value": pos.get("current_value", 0),
            "weight": pos.get("current_value", 0) / total_value if total_value > 0 else 0,
            "unrealized_return": pos.get("unrealized_return", 0),
        })
    
    return summary


@tool
def calculate_returns(
    symbol: str,
    as_of_date: str,
    lookback_days: int = 21,
) -> Dict[str, float]:
    """
    Calculate returns for a symbol over specified period.
    
    Args:
        symbol: Stock symbol (e.g., "AAPL")
        as_of_date: Date string in YYYY-MM-DD format
        lookback_days: Number of trading days to look back
        
    Returns:
        Dict with return calculation result
    """
    from core.backtest.data_loader import HistoricalDataLoader
    
    loader = HistoricalDataLoader()
    if not loader._loaded:
        loader.load_all()
    
    target_date = date.fromisoformat(as_of_date)
    ret = loader.calc_return(symbol, target_date, lookback_days=lookback_days)
    
    return {
        "symbol": symbol,
        "date": as_of_date,
        "lookback_days": lookback_days,
        "return": ret if ret is not None else 0.0,
    }


@tool
def calculate_volatility(
    symbol: str,
    as_of_date: str,
    lookback_days: int = 21,
) -> Dict[str, float]:
    """
    Calculate volatility for a symbol over specified period.
    
    Args:
        symbol: Stock symbol (e.g., "AAPL")
        as_of_date: Date string in YYYY-MM-DD format
        lookback_days: Number of trading days for volatility calculation
        
    Returns:
        Dict with volatility calculation result
    """
    from core.backtest.data_loader import HistoricalDataLoader
    
    loader = HistoricalDataLoader()
    if not loader._loaded:
        loader.load_all()
    
    target_date = date.fromisoformat(as_of_date)
    vol = loader.calc_volatility(symbol, target_date, lookback_days=lookback_days)
    
    return {
        "symbol": symbol,
        "date": as_of_date,
        "lookback_days": lookback_days,
        "volatility": vol if vol is not None else 0.0,
    }


@tool
def semantic_search_history(
    query: str,
    n_results: int = 5,
) -> Dict[str, Any]:
    """
    Search historical market data using semantic similarity.
    
    Args:
        query: Natural language query about market conditions
        n_results: Number of similar historical periods to return
        
    Returns:
        Dict with similar historical periods and their outcomes
    """
    try:
        from core.semantic.search import SemanticSearch
        
        searcher = SemanticSearch()
        results = searcher.search(query, n_results=n_results)
        
        return {
            "query": query,
            "results": [
                {
                    "date": r.date,
                    "similarity": r.similarity,
                    "context": r.context,
                    "forward_return": r.forward_return if hasattr(r, 'forward_return') else None,
                }
                for r in results
            ]
        }
    except Exception as e:
        return {"error": str(e), "query": query, "results": []}


@tool
def get_top_movers(
    as_of_date: str,
    n_top: int = 10,
) -> Dict[str, Any]:
    """
    Get top market movers (by absolute 1-day return) for a date.
    
    Args:
        as_of_date: Date string in YYYY-MM-DD format
        n_top: Number of top movers to return
        
    Returns:
        Dict with top gainers and losers
    """
    from core.backtest.data_loader import HistoricalDataLoader, BACKTEST_UNIVERSE
    
    loader = HistoricalDataLoader()
    if not loader._loaded:
        loader.load_all()
    
    target_date = date.fromisoformat(as_of_date)
    
    movers = []
    for symbol in BACKTEST_UNIVERSE:
        ret = loader.calc_return(symbol, target_date, lookback_days=1)
        if ret is not None:
            movers.append({"symbol": symbol, "return_1d": ret})
    
    # Sort by absolute return
    movers.sort(key=lambda x: abs(x["return_1d"]), reverse=True)
    
    return {
        "date": as_of_date,
        "top_movers": movers[:n_top],
        "top_gainers": sorted(movers, key=lambda x: x["return_1d"], reverse=True)[:5],
        "top_losers": sorted(movers, key=lambda x: x["return_1d"])[:5],
    }


@tool
def check_trade_limits(
    fund_id: str,
    trade_count_json: str,
    max_trades: int = 3,
    window_days: int = 5,
) -> Dict[str, Any]:
    """
    Check if fund can make new trades based on rate limits.
    
    Args:
        fund_id: Fund identifier
        trade_count_json: JSON string with recent trade dates
        max_trades: Maximum trades allowed in window
        window_days: Rolling window in days
        
    Returns:
        Dict with available trades and limit info
    """
    import json
    
    try:
        recent_trades = json.loads(trade_count_json)
    except json.JSONDecodeError:
        recent_trades = []
    
    available = max_trades - len(recent_trades)
    
    return {
        "fund_id": fund_id,
        "available_trades": max(0, available),
        "trades_in_window": len(recent_trades),
        "max_trades": max_trades,
        "window_days": window_days,
        "can_trade": available > 0,
    }


# List of all available tools for agents
MARKET_DATA_TOOLS = [
    get_market_snapshot,
    calculate_returns,
    calculate_volatility,
    get_top_movers,
]

PORTFOLIO_TOOLS = [
    get_portfolio_state,
    check_trade_limits,
]

RESEARCH_TOOLS = [
    semantic_search_history,
]

ALL_TOOLS = MARKET_DATA_TOOLS + PORTFOLIO_TOOLS + RESEARCH_TOOLS
