"""
Natural Language Query Parser for ChromaDB Semantic Search

Parses user-friendly queries into ChromaDB metadata filters.
"""
from typing import Optional, Dict, Any
import re


def parse_query(query: str) -> tuple[Optional[Dict[str, Any]], str]:
    """
    Parse natural language query into ChromaDB metadata filters.
    
    Args:
        query: Natural language query string
        
    Returns:
        tuple: (where_filter, interpretation_text)
    """
    query_lower = query.lower()
    filters = {}
    interpretations = []
    
    # Check for volatility keywords
    if "high volatility" in query_lower or "high vol" in query_lower:
        filters["volatility_21d"] = {"$gt": 0.25}
        interpretations.append("high volatility (>25%)")
    elif "low volatility" in query_lower or "low vol" in query_lower:
        filters["volatility_21d"] = {"$lt": 0.15}
        interpretations.append("low volatility (<15%)")
    elif "normal volatility" in query_lower:
        filters["volatility_21d"] = {"$gte": 0.15, "$lte": 0.25}
        interpretations.append("normal volatility (15-25%)")
    
    # Check for trend keywords
    if any(word in query_lower for word in ["uptrend", "bullish", "up trend"]):
        filters["return_1m"] = {"$gt": 0.02}
        interpretations.append("bullish trend (>2% monthly return)")
    elif any(
        word in query_lower for word in 
        ["downtrend", "bearish", "down trend"]
    ):
        filters["return_1m"] = {"$lt": -0.02}
        interpretations.append("bearish trend (<-2% monthly return)")
    
    # Check for extreme moves
    if any(word in query_lower for word in ["crash", "selloff", "sell off"]):
        filters["return_1m"] = {"$lt": -0.05}
        interpretations.append("market crash (<-5% monthly return)")
    elif "rally" in query_lower:
        filters["return_1m"] = {"$gt": 0.05}
        interpretations.append("strong rally (>5% monthly return)")
    
    # Check for date patterns (YYYY-MM or YYYY)
    date_match = re.search(r'(\d{4})-?(\d{2})?', query)
    if date_match:
        year = date_match.group(1)
        month = date_match.group(2)
        if month:
            date_pattern = f"{year}-{month}"
            interpretations.append(f"dates matching {date_pattern}")
        else:
            date_pattern = f"{year}"
            interpretations.append(f"year {year}")
    
    # Build interpretation text
    if interpretations:
        interpretation = "Searching for: " + ", ".join(interpretations)
    else:
        interpretation = "Showing all market states (no filters applied)"
    
    # Return None if no filters, so ChromaDB returns all results
    where_filter = filters if filters else None
    
    return where_filter, interpretation
