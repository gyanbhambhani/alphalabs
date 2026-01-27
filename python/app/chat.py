"""
Trading Lab Chat Backend

Conversational interface for market research and analysis.
Supports semantic queries across 45+ years of market history.
"""
import re
import json
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

from core.context.market_context import MarketContextProvider, DeepContext
from core.context.external_data import ExternalDataProvider
from core.semantic.vector_db import VectorDatabase
from app.query_parser import parse_query


class QueryType(Enum):
    """Types of queries the chat can handle"""
    SEMANTIC_SEARCH = "semantic_search"      # Find similar periods
    STOCK_ANALYSIS = "stock_analysis"        # Analyze a specific stock
    COMPARISON = "comparison"                # Compare stocks or periods
    MARKET_OVERVIEW = "market_overview"      # Current market state
    HISTORICAL_EVENT = "historical_event"    # What happened on date X
    RESEARCH = "research"                    # Generate research report
    GENERAL = "general"                      # General question


@dataclass
class ChatMessage:
    """A single chat message"""
    role: str  # "user" or "assistant"
    content: str
    timestamp: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ChatResponse:
    """Response from the chat system"""
    message: str
    query_type: QueryType
    data: Optional[Dict[str, Any]] = None
    visualizations: List[Dict[str, Any]] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)
    sources: List[str] = field(default_factory=list)


class QueryClassifier:
    """Classifies user queries into types"""
    
    # Keywords for classification
    SEMANTIC_KEYWORDS = [
        "similar", "like", "find", "search", "when", "periods",
        "times when", "historical", "past", "before"
    ]
    
    STOCK_KEYWORDS = [
        "analyze", "analysis", "what about", "how is", "tell me about",
        "show me", "give me"
    ]
    
    COMPARISON_KEYWORDS = [
        "compare", "versus", "vs", "difference", "between",
        "better", "worse"
    ]
    
    MARKET_KEYWORDS = [
        "market", "overall", "current", "today", "now",
        "sentiment", "outlook"
    ]
    
    EVENT_KEYWORDS = [
        "what happened", "on", "in", "during", "crash", "crisis",
        "2008", "2020", "2000"
    ]
    
    RESEARCH_KEYWORDS = [
        "report", "research", "deep dive", "comprehensive",
        "thesis", "strategy"
    ]
    
    @classmethod
    def classify(cls, query: str) -> QueryType:
        """Classify a query into a type"""
        query_lower = query.lower()
        
        # Check for specific patterns first
        if any(kw in query_lower for kw in cls.COMPARISON_KEYWORDS):
            return QueryType.COMPARISON
        
        if any(kw in query_lower for kw in cls.EVENT_KEYWORDS):
            # Check for date patterns
            if re.search(r'\d{4}|\d{1,2}/\d{1,2}', query_lower):
                return QueryType.HISTORICAL_EVENT
        
        if any(kw in query_lower for kw in cls.RESEARCH_KEYWORDS):
            return QueryType.RESEARCH
        
        if any(kw in query_lower for kw in cls.MARKET_KEYWORDS):
            return QueryType.MARKET_OVERVIEW
        
        if any(kw in query_lower for kw in cls.SEMANTIC_KEYWORDS):
            return QueryType.SEMANTIC_SEARCH
        
        # Check for stock symbols
        symbols = re.findall(r'\b[A-Z]{2,5}\b', query)
        if symbols and any(kw in query_lower for kw in cls.STOCK_KEYWORDS):
            return QueryType.STOCK_ANALYSIS
        
        return QueryType.GENERAL


class TradingLabChat:
    """
    Conversational interface for market research.
    
    Supports:
    - Natural language queries about market conditions
    - Semantic search across historical data
    - Stock-specific analysis
    - Comparison between periods or stocks
    - Research report generation
    """
    
    def __init__(
        self,
        persist_directory: str = "./chroma_data",
        llm_provider: Optional[str] = None
    ):
        """
        Initialize the chat system.
        
        Args:
            persist_directory: ChromaDB storage location
            llm_provider: Optional LLM for enhanced responses
        """
        self.context_provider = MarketContextProvider(persist_directory)
        self.external_provider = ExternalDataProvider()
        self.persist_directory = persist_directory
        self.llm_provider = llm_provider
        
        # Conversation history
        self.history: List[ChatMessage] = []
        
        # Example queries for suggestions
        self.example_queries = [
            "Find periods similar to current market conditions",
            "What happened after the 2008 financial crisis?",
            "Compare AAPL and MSFT over the last year",
            "Show me high volatility periods with positive outcomes",
            "What's the current market sentiment?",
            "Generate a research report on tech stocks",
            "When did the market crash more than 10% in a week?",
            "Analyze NVDA for potential entry points",
        ]
    
    def _get_symbols_from_query(self, query: str) -> List[str]:
        """Extract stock symbols from query"""
        # Common patterns
        symbols = re.findall(r'\b[A-Z]{2,5}\b', query)
        
        # Filter out common words that might be capitalized
        common_words = {
            'THE', 'AND', 'FOR', 'WITH', 'FROM', 'INTO',
            'WHAT', 'WHEN', 'WHERE', 'HOW', 'WHY', 'WHO'
        }
        
        return [s for s in symbols if s not in common_words]
    
    async def handle_query(self, user_query: str) -> ChatResponse:
        """
        Handle a user query and return a response.
        
        Args:
            user_query: The user's natural language query
            
        Returns:
            ChatResponse with analysis and data
        """
        # Add to history
        self.history.append(ChatMessage(role="user", content=user_query))
        
        # Classify query
        query_type = QueryClassifier.classify(user_query)
        
        # Route to appropriate handler
        if query_type == QueryType.SEMANTIC_SEARCH:
            response = await self._handle_semantic_search(user_query)
        elif query_type == QueryType.STOCK_ANALYSIS:
            response = await self._handle_stock_analysis(user_query)
        elif query_type == QueryType.COMPARISON:
            response = await self._handle_comparison(user_query)
        elif query_type == QueryType.MARKET_OVERVIEW:
            response = await self._handle_market_overview(user_query)
        elif query_type == QueryType.HISTORICAL_EVENT:
            response = await self._handle_historical_event(user_query)
        elif query_type == QueryType.RESEARCH:
            response = await self._handle_research(user_query)
        else:
            response = await self._handle_general(user_query)
        
        # Add response to history
        self.history.append(ChatMessage(
            role="assistant",
            content=response.message,
            metadata={"query_type": query_type.value}
        ))
        
        return response
    
    async def _handle_semantic_search(self, query: str) -> ChatResponse:
        """Handle semantic search queries"""
        # Parse the query into filters
        where_filter, interpretation = parse_query(query)
        
        # Get symbols from query or default to SPY
        symbols = self._get_symbols_from_query(query)
        symbol = symbols[0] if symbols else "SPY"
        
        try:
            # Get vector database for symbol
            vector_db = VectorDatabase(
                persist_directory=self.persist_directory,
                symbol=symbol
            )
            
            # Get matching results
            if where_filter:
                results = vector_db.collection.get(
                    where=where_filter,
                    limit=20,
                    include=['metadatas']
                )
            else:
                # Get deep context instead
                context = self.context_provider.get_deep_context(symbol)
                
                return ChatResponse(
                    message=self._format_deep_context(context),
                    query_type=QueryType.SEMANTIC_SEARCH,
                    data={
                        "symbol": symbol,
                        "similar_periods": [
                            {
                                "date": p.date,
                                "similarity": p.similarity,
                                "regime": p.regime.value,
                                "narrative": p.narrative
                            }
                            for p in context.similar_periods[:10]
                        ]
                    },
                    suggestions=[
                        f"What happened after {context.similar_periods[0].date}?",
                        f"Compare current {symbol} to 2020 crash",
                        "Show me the risk factors"
                    ]
                )
            
            # Format results
            if results and results['ids']:
                message = f"Found {len(results['ids'])} matching periods for {symbol}:\n\n"
                message += f"**Search interpretation:** {interpretation}\n\n"
                
                for i, (doc_id, metadata) in enumerate(
                    zip(results['ids'][:10], results['metadatas'][:10])
                ):
                    ret_1m = metadata.get('return_1m', 0) * 100
                    vol = metadata.get('volatility_21d', 0) * 100
                    message += (
                        f"{i+1}. **{doc_id}**: "
                        f"1M Return: {ret_1m:+.1f}%, "
                        f"Volatility: {vol:.1f}%\n"
                    )
                
                return ChatResponse(
                    message=message,
                    query_type=QueryType.SEMANTIC_SEARCH,
                    data={
                        "symbol": symbol,
                        "interpretation": interpretation,
                        "results": [
                            {"date": doc_id, "metadata": meta}
                            for doc_id, meta in zip(
                                results['ids'][:10], 
                                results['metadatas'][:10]
                            )
                        ]
                    },
                    suggestions=[
                        f"Analyze {results['ids'][0]} in detail",
                        f"What was the average outcome?",
                        f"Compare to current conditions"
                    ]
                )
            else:
                return ChatResponse(
                    message=f"No matching periods found for {symbol} with those criteria.",
                    query_type=QueryType.SEMANTIC_SEARCH,
                    suggestions=[
                        "Try broader search criteria",
                        f"Show all {symbol} embeddings",
                        "What periods are available?"
                    ]
                )
                
        except Exception as e:
            return ChatResponse(
                message=f"Error performing search: {str(e)}",
                query_type=QueryType.SEMANTIC_SEARCH,
                suggestions=["Try a different symbol", "Check if data exists"]
            )
    
    async def _handle_stock_analysis(self, query: str) -> ChatResponse:
        """Handle stock analysis queries"""
        symbols = self._get_symbols_from_query(query)
        
        if not symbols:
            return ChatResponse(
                message="Please specify a stock symbol (e.g., AAPL, MSFT, GOOGL).",
                query_type=QueryType.STOCK_ANALYSIS,
                suggestions=self.example_queries[:3]
            )
        
        symbol = symbols[0]
        
        try:
            # Get deep context
            context = self.context_provider.get_deep_context(symbol)
            
            # Get external context
            external = self.external_provider.get_full_context()
            
            message = f"## Analysis of {symbol}\n\n"
            message += f"### Current State\n"
            message += f"- **Regime:** {context.current_regime.value}\n"
            message += f"- **Volatility (21D):** {context.current_volatility:.1%}\n"
            message += f"- **Momentum (1M):** {context.current_momentum_1m:+.1%}\n"
            message += f"- **Momentum (3M):** {context.current_momentum_3m:+.1%}\n\n"
            
            message += f"### Historical Context\n"
            message += context.market_interpretation + "\n\n"
            
            message += f"### Similar Historical Periods\n"
            for i, period in enumerate(context.similar_periods[:5], 1):
                message += (
                    f"{i}. **{period.date}** "
                    f"(similarity: {period.similarity:.1%})\n"
                    f"   {period.narrative}\n"
                    f"   Outcome: {period.forward_outcome.return_1m:+.1%} (1M)\n\n"
                )
            
            message += f"### Recommendation\n"
            message += f"**Stance:** {context.recommended_stance}\n"
            message += f"**Confidence:** {context.confidence_score:.0%}\n\n"
            
            message += f"### Key Risks\n"
            for risk in context.key_risks:
                message += f"- {risk}\n"
            
            return ChatResponse(
                message=message,
                query_type=QueryType.STOCK_ANALYSIS,
                data={
                    "symbol": symbol,
                    "regime": context.current_regime.value,
                    "recommendation": context.recommended_stance,
                    "confidence": context.confidence_score
                },
                suggestions=[
                    f"Compare {symbol} to SPY",
                    f"What happened after similar periods?",
                    f"Generate research report for {symbol}"
                ]
            )
            
        except Exception as e:
            return ChatResponse(
                message=f"Error analyzing {symbol}: {str(e)}",
                query_type=QueryType.STOCK_ANALYSIS,
                suggestions=["Check if embeddings exist for this symbol"]
            )
    
    async def _handle_comparison(self, query: str) -> ChatResponse:
        """Handle comparison queries"""
        symbols = self._get_symbols_from_query(query)
        
        if len(symbols) < 2:
            return ChatResponse(
                message=(
                    "Please specify at least two symbols to compare "
                    "(e.g., 'Compare AAPL and MSFT')."
                ),
                query_type=QueryType.COMPARISON,
                suggestions=[
                    "Compare AAPL and MSFT",
                    "Compare tech vs financials",
                    "Compare 2020 crash to 2008"
                ]
            )
        
        try:
            contexts = {}
            for symbol in symbols[:3]:  # Max 3 for comparison
                contexts[symbol] = self.context_provider.get_deep_context(symbol)
            
            message = f"## Comparison: {' vs '.join(symbols[:3])}\n\n"
            
            # Table header
            message += "| Metric | " + " | ".join(symbols[:3]) + " |\n"
            message += "|--------|" + "|".join(["--------"] * len(symbols[:3])) + "|\n"
            
            # Regime
            message += "| Regime | " + " | ".join([
                ctx.current_regime.value for ctx in contexts.values()
            ]) + " |\n"
            
            # Momentum 1M
            message += "| Momentum 1M | " + " | ".join([
                f"{ctx.current_momentum_1m:+.1%}" for ctx in contexts.values()
            ]) + " |\n"
            
            # Volatility
            message += "| Volatility | " + " | ".join([
                f"{ctx.current_volatility:.1%}" for ctx in contexts.values()
            ]) + " |\n"
            
            # Expected Return
            message += "| Avg Fwd Return (1M) | " + " | ".join([
                f"{ctx.avg_forward_return_1m:+.1%}" for ctx in contexts.values()
            ]) + " |\n"
            
            # Confidence
            message += "| Confidence | " + " | ".join([
                f"{ctx.confidence_score:.0%}" for ctx in contexts.values()
            ]) + " |\n"
            
            # Stance
            message += "| Recommended | " + " | ".join([
                ctx.recommended_stance for ctx in contexts.values()
            ]) + " |\n"
            
            message += "\n### Summary\n"
            
            # Determine winner
            best_symbol = max(
                contexts.items(), 
                key=lambda x: x[1].avg_forward_return_1m * x[1].confidence_score
            )[0]
            
            message += (
                f"Based on historical patterns, **{best_symbol}** shows the most "
                f"favorable risk-adjusted outlook.\n"
            )
            
            return ChatResponse(
                message=message,
                query_type=QueryType.COMPARISON,
                data={
                    "symbols": symbols[:3],
                    "comparison": {
                        s: {
                            "regime": ctx.current_regime.value,
                            "momentum_1m": ctx.current_momentum_1m,
                            "volatility": ctx.current_volatility,
                            "recommendation": ctx.recommended_stance
                        }
                        for s, ctx in contexts.items()
                    }
                },
                suggestions=[
                    f"Deep dive into {best_symbol}",
                    "Show risk factors for each",
                    "Historical correlation analysis"
                ]
            )
            
        except Exception as e:
            return ChatResponse(
                message=f"Error in comparison: {str(e)}",
                query_type=QueryType.COMPARISON
            )
    
    async def _handle_market_overview(self, query: str) -> ChatResponse:
        """Handle market overview queries"""
        try:
            external = self.external_provider.get_full_context()
            
            message = f"## Market Overview\n\n"
            message += f"*As of {external.timestamp.strftime('%Y-%m-%d %H:%M')} UTC*\n\n"
            
            message += f"### Sentiment\n"
            message += (
                f"- **VIX:** {external.sentiment.vix_level:.1f} "
                f"({external.sentiment.vix_percentile:.0%} percentile)\n"
            )
            message += (
                f"- **Overall:** {external.sentiment.sentiment_level.value.replace('_', ' ').title()}\n"
            )
            message += f"- {external.sentiment.interpretation}\n\n"
            
            message += f"### Economic Context\n"
            message += f"- **10Y Yield:** {external.economic.ten_year_yield:.2%}\n"
            message += (
                f"- **Yield Curve:** {external.economic.yield_curve_spread:.2%}\n"
            )
            message += (
                f"- **Recession Risk:** "
                f"{'HIGH' if external.economic.is_recession_risk else 'LOW'}\n"
            )
            message += f"- {external.economic.interpretation}\n\n"
            
            message += f"### Market Narrative\n"
            message += external.market_narrative + "\n\n"
            
            message += f"### Geopolitical Factors\n"
            message += external.geopolitical_summary + "\n"
            
            return ChatResponse(
                message=message,
                query_type=QueryType.MARKET_OVERVIEW,
                data={
                    "vix": external.sentiment.vix_level,
                    "sentiment": external.sentiment.sentiment_level.value,
                    "yield_curve": external.economic.yield_curve_spread,
                    "recession_risk": external.economic.is_recession_risk
                },
                suggestions=[
                    "How does this compare to 2008?",
                    "What usually happens after this sentiment level?",
                    "Show me defensive positioning strategies"
                ]
            )
            
        except Exception as e:
            return ChatResponse(
                message=f"Error fetching market data: {str(e)}",
                query_type=QueryType.MARKET_OVERVIEW
            )
    
    async def _handle_historical_event(self, query: str) -> ChatResponse:
        """Handle historical event queries"""
        # Extract date or year from query
        year_match = re.search(r'(19|20)\d{2}', query)
        
        if not year_match:
            return ChatResponse(
                message="Please specify a year or date (e.g., '2008', '2020-03').",
                query_type=QueryType.HISTORICAL_EVENT,
                suggestions=[
                    "What happened in 2008?",
                    "Tell me about 2020-03",
                    "How did the market recover in 2009?"
                ]
            )
        
        year = year_match.group()
        
        # Get symbols from query or default
        symbols = self._get_symbols_from_query(query)
        symbol = symbols[0] if symbols else "SPY"
        
        try:
            vector_db = VectorDatabase(
                persist_directory=self.persist_directory,
                symbol=symbol
            )
            
            # Get all data from that year
            all_data = vector_db.collection.get(include=['metadatas'])
            
            year_data = [
                (doc_id, meta)
                for doc_id, meta in zip(all_data['ids'], all_data['metadatas'])
                if doc_id.startswith(year)
            ]
            
            if not year_data:
                return ChatResponse(
                    message=f"No data found for {symbol} in {year}.",
                    query_type=QueryType.HISTORICAL_EVENT,
                    suggestions=["Try a different year", "Check data availability"]
                )
            
            # Sort by date
            year_data.sort(key=lambda x: x[0])
            
            # Calculate statistics
            returns_1m = [m.get('return_1m', 0) for _, m in year_data]
            vols = [m.get('volatility_21d', 0) for _, m in year_data]
            
            message = f"## {symbol} in {year}\n\n"
            message += f"**Data points:** {len(year_data)}\n"
            message += f"**Period:** {year_data[0][0]} to {year_data[-1][0]}\n\n"
            
            message += f"### Performance\n"
            message += f"- **Avg 1M Return:** {np.mean(returns_1m)*100:+.2f}%\n"
            message += f"- **Best Month:** {max(returns_1m)*100:+.2f}%\n"
            message += f"- **Worst Month:** {min(returns_1m)*100:+.2f}%\n"
            message += f"- **Avg Volatility:** {np.mean(vols)*100:.1f}%\n\n"
            
            # Notable periods
            message += f"### Notable Periods\n"
            
            # Find extreme periods
            extreme_periods = sorted(year_data, key=lambda x: x[1].get('return_1m', 0))
            
            message += f"\n**Worst periods:**\n"
            for date, meta in extreme_periods[:3]:
                message += (
                    f"- {date}: {meta.get('return_1m', 0)*100:+.1f}% "
                    f"(vol: {meta.get('volatility_21d', 0)*100:.0f}%)\n"
                )
            
            message += f"\n**Best periods:**\n"
            for date, meta in extreme_periods[-3:]:
                message += (
                    f"- {date}: {meta.get('return_1m', 0)*100:+.1f}% "
                    f"(vol: {meta.get('volatility_21d', 0)*100:.0f}%)\n"
                )
            
            return ChatResponse(
                message=message,
                query_type=QueryType.HISTORICAL_EVENT,
                data={
                    "symbol": symbol,
                    "year": year,
                    "data_points": len(year_data),
                    "avg_return": float(np.mean(returns_1m)),
                    "avg_volatility": float(np.mean(vols))
                },
                suggestions=[
                    f"Compare {year} to current market",
                    f"What were the recovery patterns?",
                    f"Show similar periods across all years"
                ]
            )
            
        except Exception as e:
            return ChatResponse(
                message=f"Error fetching historical data: {str(e)}",
                query_type=QueryType.HISTORICAL_EVENT
            )
    
    async def _handle_research(self, query: str) -> ChatResponse:
        """Handle research report generation"""
        symbols = self._get_symbols_from_query(query)
        
        if not symbols:
            symbols = ["SPY"]  # Default to market overview
        
        symbol = symbols[0]
        
        try:
            # Get comprehensive context
            context = self.context_provider.get_deep_context(symbol, top_k=30)
            external = self.external_provider.get_full_context()
            
            message = f"# Research Report: {symbol}\n\n"
            message += f"*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')} UTC*\n\n"
            
            message += "## Executive Summary\n\n"
            message += (
                f"{symbol} is currently in a **{context.current_regime.value}** regime "
                f"with {context.current_volatility:.0%} volatility. "
                f"Based on analysis of {len(context.similar_periods)} similar historical periods, "
                f"our recommendation is **{context.recommended_stance}** "
                f"with {context.confidence_score:.0%} confidence.\n\n"
            )
            
            message += "## Current Market Conditions\n\n"
            message += f"- **Regime:** {context.current_regime.value}\n"
            message += f"- **Volatility (21D):** {context.current_volatility:.1%}\n"
            message += f"- **Momentum (1M):** {context.current_momentum_1m:+.1%}\n"
            message += f"- **Momentum (3M):** {context.current_momentum_3m:+.1%}\n\n"
            
            message += f"### Market Sentiment\n"
            message += (
                f"VIX at {external.sentiment.vix_level:.1f} indicates "
                f"{external.sentiment.sentiment_level.value.replace('_', ' ')} sentiment. "
                f"{external.sentiment.interpretation}\n\n"
            )
            
            message += "## Historical Analysis\n\n"
            message += context.market_interpretation + "\n\n"
            
            message += "### Most Relevant Historical Precedents\n\n"
            for i, period in enumerate(context.similar_periods[:5], 1):
                message += f"#### {i}. {period.date}\n"
                message += f"- **Similarity:** {period.similarity:.1%}\n"
                message += f"- **Regime:** {period.regime.value}\n"
                message += f"- **Context:** {period.geopolitical_context}\n"
                message += f"- **Narrative:** {period.narrative}\n"
                message += (
                    f"- **Outcome:** "
                    f"{period.forward_outcome.return_1m:+.1%} (1M), "
                    f"{period.forward_outcome.return_3m:+.1%} (3M)\n\n"
                )
            
            message += "## Statistical Analysis\n\n"
            message += f"| Metric | Value |\n"
            message += f"|--------|-------|\n"
            message += f"| Avg Forward Return (1M) | {context.avg_forward_return_1m:+.1%} |\n"
            message += f"| Avg Forward Return (3M) | {context.avg_forward_return_3m:+.1%} |\n"
            message += f"| Positive Outcome Rate | {context.positive_outcome_rate:.0%} |\n"
            message += f"| Worst Case Drawdown | {context.worst_case_drawdown:.1%} |\n"
            message += f"| Best Case Return | {context.best_case_return:+.1%} |\n\n"
            
            message += "## Risk Assessment\n\n"
            for risk in context.key_risks:
                message += f"- {risk}\n"
            message += "\n"
            
            message += "## Recommendation\n\n"
            message += f"**Stance:** {context.recommended_stance}\n\n"
            message += f"**Confidence:** {context.confidence_score:.0%}\n\n"
            
            # Position sizing recommendation
            if context.confidence_score > 0.7:
                sizing = "Full position (15-20%)"
            elif context.confidence_score > 0.5:
                sizing = "Half position (8-10%)"
            else:
                sizing = "Small position (3-5%) or pass"
            
            message += f"**Suggested Position Size:** {sizing}\n\n"
            
            message += "---\n"
            message += (
                "*This report is generated by AI analysis of historical market data. "
                "Past performance does not guarantee future results. "
                "Always conduct your own research.*\n"
            )
            
            return ChatResponse(
                message=message,
                query_type=QueryType.RESEARCH,
                data={
                    "symbol": symbol,
                    "recommendation": context.recommended_stance,
                    "confidence": context.confidence_score,
                    "avg_forward_return": context.avg_forward_return_1m,
                    "positive_rate": context.positive_outcome_rate
                },
                suggestions=[
                    f"Compare {symbol} to other tech stocks",
                    "What are the key risk factors?",
                    "Show me the exit strategy"
                ],
                sources=[
                    f"ChromaDB embeddings ({len(context.similar_periods)} similar periods)",
                    "Market sentiment data (VIX, breadth)",
                    "Economic indicators (yields, spreads)"
                ]
            )
            
        except Exception as e:
            return ChatResponse(
                message=f"Error generating research report: {str(e)}",
                query_type=QueryType.RESEARCH
            )
    
    async def _handle_general(self, query: str) -> ChatResponse:
        """Handle general queries"""
        return ChatResponse(
            message=(
                "I can help you with market research and analysis. Try asking:\n\n"
                "- **Semantic search:** 'Find periods similar to current conditions'\n"
                "- **Stock analysis:** 'Analyze AAPL'\n"
                "- **Comparison:** 'Compare MSFT and GOOGL'\n"
                "- **Market overview:** 'What's the current market sentiment?'\n"
                "- **Historical:** 'What happened in 2008?'\n"
                "- **Research:** 'Generate a research report on NVDA'\n"
            ),
            query_type=QueryType.GENERAL,
            suggestions=self.example_queries[:5]
        )
    
    def _format_deep_context(self, context: DeepContext) -> str:
        """Format deep context into readable message"""
        message = f"## Similar Historical Periods for {context.symbol}\n\n"
        message += f"Current market regime: **{context.current_regime.value}**\n\n"
        message += context.market_interpretation + "\n\n"
        
        message += "### Top Similar Periods\n"
        for i, period in enumerate(context.similar_periods[:10], 1):
            message += (
                f"{i}. **{period.date}** "
                f"(similarity: {period.similarity:.1%})\n"
                f"   - {period.narrative}\n"
                f"   - Outcome: {period.forward_outcome.return_1m:+.1%} (1M)\n\n"
            )
        
        return message
    
    def get_conversation_history(self) -> List[Dict]:
        """Get formatted conversation history"""
        return [
            {
                "role": msg.role,
                "content": msg.content,
                "timestamp": msg.timestamp.isoformat()
            }
            for msg in self.history
        ]
    
    def clear_history(self) -> None:
        """Clear conversation history"""
        self.history = []


# Utility import for numpy
import numpy as np
