"""
Research Query Engine

AI-powered market research and report generation.
Combines semantic search, quantitative analysis, and LLM reasoning.
"""
import numpy as np
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

from core.context.market_context import MarketContextProvider, DeepContext
from core.context.external_data import ExternalDataProvider, ExternalContext
from core.quant.models import QuantitativeModels, RiskMetrics
from core.semantic.vector_db import VectorDatabase


class ReportType(Enum):
    """Types of research reports"""
    STOCK_ANALYSIS = "stock_analysis"
    SECTOR_ANALYSIS = "sector_analysis"
    MARKET_OUTLOOK = "market_outlook"
    HISTORICAL_COMPARISON = "historical_comparison"
    RISK_ASSESSMENT = "risk_assessment"
    TRADE_IDEA = "trade_idea"


@dataclass
class ReportSection:
    """A section of a research report"""
    title: str
    content: str
    data: Optional[Dict[str, Any]] = None
    charts: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class ResearchReport:
    """Complete research report"""
    title: str
    report_type: ReportType
    generated_at: datetime
    symbols: List[str]
    
    # Report sections
    executive_summary: str
    sections: List[ReportSection]
    
    # Key findings
    recommendation: str
    confidence: float
    key_risks: List[str]
    
    # Supporting data
    historical_context: Optional[DeepContext] = None
    market_context: Optional[ExternalContext] = None
    quantitative_data: Optional[Dict[str, Any]] = None
    
    # Metadata
    sources: List[str] = field(default_factory=list)
    methodology: str = ""
    
    def to_markdown(self) -> str:
        """Convert report to markdown format"""
        md = f"# {self.title}\n\n"
        md += f"*Generated: {self.generated_at.strftime('%Y-%m-%d %H:%M')} UTC*\n\n"
        md += f"**Symbols:** {', '.join(self.symbols)}\n\n"
        
        md += "## Executive Summary\n\n"
        md += self.executive_summary + "\n\n"
        
        for section in self.sections:
            md += f"## {section.title}\n\n"
            md += section.content + "\n\n"
        
        md += "## Recommendation\n\n"
        md += f"**{self.recommendation}** (Confidence: {self.confidence:.0%})\n\n"
        
        if self.key_risks:
            md += "## Key Risks\n\n"
            for risk in self.key_risks:
                md += f"- {risk}\n"
            md += "\n"
        
        if self.methodology:
            md += "---\n\n"
            md += f"*Methodology: {self.methodology}*\n"
        
        return md
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API response"""
        return {
            "title": self.title,
            "report_type": self.report_type.value,
            "generated_at": self.generated_at.isoformat(),
            "symbols": self.symbols,
            "executive_summary": self.executive_summary,
            "sections": [
                {
                    "title": s.title,
                    "content": s.content,
                    "data": s.data
                }
                for s in self.sections
            ],
            "recommendation": self.recommendation,
            "confidence": self.confidence,
            "key_risks": self.key_risks,
            "sources": self.sources
        }


@dataclass
class ResearchQuery:
    """Research query specification"""
    query_type: ReportType
    symbols: List[str]
    topic: Optional[str] = None
    time_period: str = "1y"
    compare_to: Optional[List[str]] = None
    focus_areas: List[str] = field(default_factory=list)


class ResearchEngine:
    """
    AI-powered research engine.
    
    Generates comprehensive research reports by combining:
    - Semantic search across historical data
    - Quantitative analysis
    - Market context
    - LLM-generated narratives
    """
    
    def __init__(
        self,
        persist_directory: str = "./chroma_data",
        llm_provider: Optional[str] = None
    ):
        """
        Initialize the research engine.
        
        Args:
            persist_directory: ChromaDB storage location
            llm_provider: Optional LLM for enhanced narratives
        """
        self.persist_directory = persist_directory
        self.llm_provider = llm_provider
        
        # Initialize providers
        self.context_provider = MarketContextProvider(persist_directory)
        self.external_provider = ExternalDataProvider()
        self.quant_models = QuantitativeModels()
    
    async def generate_report(self, query: ResearchQuery) -> ResearchReport:
        """
        Generate a research report based on the query.
        
        Args:
            query: Research query specification
            
        Returns:
            Complete research report
        """
        if query.query_type == ReportType.STOCK_ANALYSIS:
            return await self._generate_stock_report(query)
        elif query.query_type == ReportType.MARKET_OUTLOOK:
            return await self._generate_market_outlook(query)
        elif query.query_type == ReportType.HISTORICAL_COMPARISON:
            return await self._generate_historical_comparison(query)
        elif query.query_type == ReportType.RISK_ASSESSMENT:
            return await self._generate_risk_assessment(query)
        elif query.query_type == ReportType.TRADE_IDEA:
            return await self._generate_trade_idea(query)
        else:
            # Default to stock analysis
            return await self._generate_stock_report(query)
    
    async def _generate_stock_report(self, query: ResearchQuery) -> ResearchReport:
        """Generate comprehensive stock analysis report"""
        symbol = query.symbols[0] if query.symbols else "SPY"
        
        # Get deep context
        context = self.context_provider.get_deep_context(symbol, top_k=30)
        
        # Get external context
        external = self.external_provider.get_full_context()
        
        # Generate sections
        sections = []
        
        # Current Market State
        sections.append(ReportSection(
            title="Current Market State",
            content=self._format_current_state(context, external),
            data={
                "regime": context.current_regime.value,
                "volatility": context.current_volatility,
                "momentum_1m": context.current_momentum_1m
            }
        ))
        
        # Historical Analysis
        sections.append(ReportSection(
            title="Historical Analysis",
            content=self._format_historical_analysis(context),
            data={
                "similar_periods_count": len(context.similar_periods),
                "avg_forward_return": context.avg_forward_return_1m,
                "positive_rate": context.positive_outcome_rate
            }
        ))
        
        # Similar Periods Detail
        sections.append(ReportSection(
            title="Most Relevant Historical Precedents",
            content=self._format_similar_periods(context.similar_periods[:5])
        ))
        
        # Statistical Analysis
        sections.append(ReportSection(
            title="Statistical Analysis",
            content=self._format_statistics(context),
            data={
                "avg_1m_return": context.avg_forward_return_1m,
                "avg_3m_return": context.avg_forward_return_3m,
                "positive_rate": context.positive_outcome_rate,
                "worst_drawdown": context.worst_case_drawdown,
                "best_return": context.best_case_return
            }
        ))
        
        # Risk Assessment
        sections.append(ReportSection(
            title="Risk Assessment",
            content=self._format_risks(context.key_risks, external)
        ))
        
        # Generate executive summary
        exec_summary = self._generate_executive_summary(
            symbol, context, external
        )
        
        return ResearchReport(
            title=f"Research Report: {symbol}",
            report_type=ReportType.STOCK_ANALYSIS,
            generated_at=datetime.utcnow(),
            symbols=[symbol],
            executive_summary=exec_summary,
            sections=sections,
            recommendation=context.recommended_stance,
            confidence=context.confidence_score,
            key_risks=context.key_risks,
            historical_context=context,
            market_context=external,
            sources=[
                f"ChromaDB semantic search ({len(context.similar_periods)} similar periods)",
                "VIX and market sentiment data",
                "Treasury yield data"
            ],
            methodology=(
                "This report analyzes historical market data using semantic similarity "
                "matching to identify comparable periods. Statistical analysis of forward "
                "returns from similar periods informs the recommendation."
            )
        )
    
    async def _generate_market_outlook(self, query: ResearchQuery) -> ResearchReport:
        """Generate market outlook report"""
        external = self.external_provider.get_full_context()
        
        # Analyze major indices
        indices = ["SPY", "QQQ", "IWM", "DIA"]
        contexts = {}
        
        for symbol in indices:
            try:
                contexts[symbol] = self.context_provider.get_deep_context(symbol)
            except Exception:
                continue
        
        sections = []
        
        # Market Sentiment
        sections.append(ReportSection(
            title="Market Sentiment",
            content=self._format_sentiment(external),
            data={
                "vix": external.sentiment.vix_level,
                "sentiment_level": external.sentiment.sentiment_level.value
            }
        ))
        
        # Economic Context
        sections.append(ReportSection(
            title="Economic Context",
            content=self._format_economic(external)
        ))
        
        # Index Analysis
        if contexts:
            sections.append(ReportSection(
                title="Index Analysis",
                content=self._format_index_analysis(contexts)
            ))
        
        # Market Narrative
        sections.append(ReportSection(
            title="Market Narrative",
            content=external.market_narrative + "\n\n" + external.geopolitical_summary
        ))
        
        # Determine overall stance
        if "fear" in external.sentiment.sentiment_level.value:
            stance = "cautious_long" if contexts.get("SPY") and \
                contexts["SPY"].confidence_score > 0.5 else "defensive"
        elif "greed" in external.sentiment.sentiment_level.value:
            stance = "reduce_exposure"
        else:
            stance = "neutral"
        
        confidence = np.mean([
            c.confidence_score for c in contexts.values()
        ]) if contexts else 0.5
        
        return ResearchReport(
            title="Market Outlook Report",
            report_type=ReportType.MARKET_OUTLOOK,
            generated_at=datetime.utcnow(),
            symbols=list(contexts.keys()),
            executive_summary=self._generate_market_summary(external, contexts),
            sections=sections,
            recommendation=stance,
            confidence=confidence,
            key_risks=[
                "Geopolitical uncertainty",
                "Central bank policy risk",
                "Earnings outlook uncertainty"
            ],
            market_context=external,
            sources=[
                "VIX and sentiment indicators",
                "Treasury yield data",
                "Sector analysis"
            ]
        )
    
    async def _generate_historical_comparison(
        self, 
        query: ResearchQuery
    ) -> ResearchReport:
        """Generate historical comparison report"""
        symbol = query.symbols[0] if query.symbols else "SPY"
        compare_periods = query.compare_to or []
        
        context = self.context_provider.get_deep_context(symbol, top_k=50)
        
        sections = []
        
        # Current vs Historical
        sections.append(ReportSection(
            title="Current Period Analysis",
            content=self._format_current_state(context, None)
        ))
        
        # Historical periods comparison
        for i, period in enumerate(context.similar_periods[:5], 1):
            sections.append(ReportSection(
                title=f"Historical Period {i}: {period.date}",
                content=self._format_period_detail(period)
            ))
        
        # Statistical comparison
        sections.append(ReportSection(
            title="Comparative Statistics",
            content=self._format_comparison_stats(context)
        ))
        
        return ResearchReport(
            title=f"Historical Comparison: {symbol}",
            report_type=ReportType.HISTORICAL_COMPARISON,
            generated_at=datetime.utcnow(),
            symbols=[symbol],
            executive_summary=(
                f"Analysis of {symbol} comparing current conditions to "
                f"{len(context.similar_periods)} similar historical periods. "
                f"The most relevant comparison is {context.similar_periods[0].date}."
            ),
            sections=sections,
            recommendation=context.recommended_stance,
            confidence=context.confidence_score,
            key_risks=context.key_risks,
            historical_context=context
        )
    
    async def _generate_risk_assessment(self, query: ResearchQuery) -> ResearchReport:
        """Generate risk assessment report"""
        symbol = query.symbols[0] if query.symbols else "SPY"
        
        context = self.context_provider.get_deep_context(symbol)
        external = self.external_provider.get_full_context()
        
        sections = []
        
        # Volatility Analysis
        sections.append(ReportSection(
            title="Volatility Analysis",
            content=self._format_volatility_analysis(context, external)
        ))
        
        # Drawdown Analysis
        sections.append(ReportSection(
            title="Historical Drawdown Analysis",
            content=self._format_drawdown_analysis(context)
        ))
        
        # Risk Factors
        sections.append(ReportSection(
            title="Risk Factors",
            content=self._format_risks(context.key_risks, external)
        ))
        
        # Tail Risk Analysis
        sections.append(ReportSection(
            title="Tail Risk Analysis",
            content=self._format_tail_risk(context)
        ))
        
        return ResearchReport(
            title=f"Risk Assessment: {symbol}",
            report_type=ReportType.RISK_ASSESSMENT,
            generated_at=datetime.utcnow(),
            symbols=[symbol],
            executive_summary=(
                f"Comprehensive risk analysis for {symbol}. "
                f"Current volatility at {context.current_volatility:.1%}. "
                f"Worst case historical drawdown in similar periods: "
                f"{context.worst_case_drawdown:.1%}."
            ),
            sections=sections,
            recommendation="risk_aware" if context.current_volatility > 0.25 else "normal",
            confidence=context.confidence_score,
            key_risks=context.key_risks,
            historical_context=context
        )
    
    async def _generate_trade_idea(self, query: ResearchQuery) -> ResearchReport:
        """Generate trade idea report"""
        symbol = query.symbols[0] if query.symbols else "SPY"
        
        context = self.context_provider.get_deep_context(symbol)
        external = self.external_provider.get_full_context()
        
        sections = []
        
        # Trade Setup
        sections.append(ReportSection(
            title="Trade Setup",
            content=self._format_trade_setup(context, external),
            data={
                "entry_signal": context.recommended_stance,
                "confidence": context.confidence_score
            }
        ))
        
        # Historical Edge
        sections.append(ReportSection(
            title="Historical Edge",
            content=self._format_historical_edge(context)
        ))
        
        # Position Sizing
        sections.append(ReportSection(
            title="Position Sizing",
            content=self._format_position_sizing(context)
        ))
        
        # Exit Strategy
        sections.append(ReportSection(
            title="Exit Strategy",
            content=self._format_exit_strategy(context)
        ))
        
        return ResearchReport(
            title=f"Trade Idea: {symbol}",
            report_type=ReportType.TRADE_IDEA,
            generated_at=datetime.utcnow(),
            symbols=[symbol],
            executive_summary=(
                f"Trade idea for {symbol} based on historical pattern matching. "
                f"Recommendation: {context.recommended_stance} "
                f"with {context.confidence_score:.0%} confidence."
            ),
            sections=sections,
            recommendation=context.recommended_stance,
            confidence=context.confidence_score,
            key_risks=context.key_risks,
            historical_context=context
        )
    
    # Helper methods for formatting
    def _format_current_state(
        self, 
        context: DeepContext, 
        external: Optional[ExternalContext]
    ) -> str:
        text = f"""
**Regime:** {context.current_regime.value}
**Volatility (21D):** {context.current_volatility:.1%}
**Momentum (1M):** {context.current_momentum_1m:+.1%}
**Momentum (3M):** {context.current_momentum_3m:+.1%}

{context.market_interpretation}
"""
        if external:
            text += f"\n**Market Sentiment:** {external.sentiment.sentiment_level.value}\n"
            text += f"**VIX:** {external.sentiment.vix_level:.1f}\n"
        
        return text
    
    def _format_historical_analysis(self, context: DeepContext) -> str:
        return f"""
Analysis based on {len(context.similar_periods)} similar historical periods.

**Average 1-Month Forward Return:** {context.avg_forward_return_1m:+.1%}
**Average 3-Month Forward Return:** {context.avg_forward_return_3m:+.1%}
**Positive Outcome Rate:** {context.positive_outcome_rate:.0%}
**Worst Case Drawdown:** {context.worst_case_drawdown:.1%}
**Best Case Return:** {context.best_case_return:+.1%}

{context.market_interpretation}
"""
    
    def _format_similar_periods(self, periods) -> str:
        text = ""
        for i, period in enumerate(periods, 1):
            text += f"""
### {i}. {period.date}
- **Similarity:** {period.similarity:.1%}
- **Regime:** {period.regime.value}
- **Context:** {period.geopolitical_context}
- **Narrative:** {period.narrative}
- **Outcome (1M):** {period.forward_outcome.return_1m:+.1%}
- **Outcome (3M):** {period.forward_outcome.return_3m:+.1%}
"""
        return text
    
    def _format_statistics(self, context: DeepContext) -> str:
        return f"""
| Metric | Value |
|--------|-------|
| Average Forward Return (1M) | {context.avg_forward_return_1m:+.1%} |
| Average Forward Return (3M) | {context.avg_forward_return_3m:+.1%} |
| Positive Outcome Rate | {context.positive_outcome_rate:.0%} |
| Worst Case Drawdown | {context.worst_case_drawdown:.1%} |
| Best Case Return | {context.best_case_return:+.1%} |
| Confidence Score | {context.confidence_score:.0%} |
"""
    
    def _format_risks(
        self, 
        risks: List[str], 
        external: Optional[ExternalContext]
    ) -> str:
        text = "### Identified Risks\n\n"
        for risk in risks:
            text += f"- {risk}\n"
        
        if external and external.economic.is_recession_risk:
            text += "\n**Warning:** Elevated recession risk detected.\n"
        
        return text
    
    def _generate_executive_summary(
        self,
        symbol: str,
        context: DeepContext,
        external: ExternalContext
    ) -> str:
        return (
            f"{symbol} is currently in a **{context.current_regime.value}** regime "
            f"with {context.current_volatility:.0%} volatility. "
            f"Based on analysis of {len(context.similar_periods)} similar historical periods, "
            f"we expect a **{context.avg_forward_return_1m:+.1%}** return over the next month "
            f"with {context.positive_outcome_rate:.0%} probability of positive outcome. "
            f"\n\n**Recommendation:** {context.recommended_stance} "
            f"(Confidence: {context.confidence_score:.0%})"
        )
    
    def _format_sentiment(self, external: ExternalContext) -> str:
        return f"""
**VIX Level:** {external.sentiment.vix_level:.1f} 
({external.sentiment.vix_percentile:.0%} percentile)

**Market Breadth:** {external.sentiment.breadth:.2f}

**Sentiment Level:** {external.sentiment.sentiment_level.value.replace('_', ' ').title()}

{external.sentiment.interpretation}
"""
    
    def _format_economic(self, external: ExternalContext) -> str:
        return f"""
**10-Year Treasury Yield:** {external.economic.ten_year_yield:.2%}
**Yield Curve Spread (10Y-2Y):** {external.economic.yield_curve_spread:.2%}
**Recession Risk:** {'ELEVATED' if external.economic.is_recession_risk else 'LOW'}

{external.economic.interpretation}
"""
    
    def _format_index_analysis(self, contexts: Dict[str, DeepContext]) -> str:
        text = "| Index | Regime | Momentum | Volatility | Recommendation |\n"
        text += "|-------|--------|----------|------------|----------------|\n"
        
        for symbol, ctx in contexts.items():
            text += (
                f"| {symbol} | {ctx.current_regime.value} | "
                f"{ctx.current_momentum_1m:+.1%} | {ctx.current_volatility:.1%} | "
                f"{ctx.recommended_stance} |\n"
            )
        
        return text
    
    def _generate_market_summary(
        self,
        external: ExternalContext,
        contexts: Dict[str, DeepContext]
    ) -> str:
        avg_momentum = np.mean([c.current_momentum_1m for c in contexts.values()])
        avg_vol = np.mean([c.current_volatility for c in contexts.values()])
        
        return (
            f"Current market sentiment is "
            f"{external.sentiment.sentiment_level.value.replace('_', ' ')}. "
            f"VIX at {external.sentiment.vix_level:.1f} "
            f"({external.sentiment.vix_percentile:.0%} percentile). "
            f"Average index momentum is {avg_momentum:+.1%} "
            f"with {avg_vol:.1%} volatility. "
            f"{external.market_narrative}"
        )
    
    def _format_period_detail(self, period) -> str:
        return f"""
**Similarity to Current:** {period.similarity:.1%}
**Regime:** {period.regime.value}
**Geopolitical Context:** {period.geopolitical_context}

{period.narrative}

**Outcome:**
- 1-Month Return: {period.forward_outcome.return_1m:+.1%}
- 3-Month Return: {period.forward_outcome.return_3m:+.1%}
- Max Drawdown: {period.forward_outcome.max_drawdown_1m:.1%}
"""
    
    def _format_comparison_stats(self, context: DeepContext) -> str:
        returns = [p.forward_outcome.return_1m for p in context.similar_periods]
        
        return f"""
Based on {len(returns)} similar historical periods:

| Metric | Value |
|--------|-------|
| Median Return | {np.median(returns)*100:+.1f}% |
| Mean Return | {np.mean(returns)*100:+.1f}% |
| Std Dev | {np.std(returns)*100:.1f}% |
| Best Case | {max(returns)*100:+.1f}% |
| Worst Case | {min(returns)*100:+.1f}% |
| % Positive | {sum(1 for r in returns if r > 0)/len(returns)*100:.0f}% |
"""
    
    def _format_volatility_analysis(
        self, 
        context: DeepContext, 
        external: ExternalContext
    ) -> str:
        return f"""
**Current Volatility (21D):** {context.current_volatility:.1%}
**VIX Level:** {external.sentiment.vix_level:.1f}
**VIX Percentile:** {external.sentiment.vix_percentile:.0%}

Volatility assessment:
- {'HIGH' if context.current_volatility > 0.25 else 'MODERATE' if context.current_volatility > 0.15 else 'LOW'} volatility environment
- Historical similar periods averaged {np.mean([p.volatility for p in context.similar_periods]):.1%} volatility
"""
    
    def _format_drawdown_analysis(self, context: DeepContext) -> str:
        drawdowns = [p.forward_outcome.max_drawdown_1m for p in context.similar_periods]
        
        return f"""
Analysis of drawdowns in similar historical periods:

| Metric | Value |
|--------|-------|
| Average Drawdown | {np.mean(drawdowns)*100:.1f}% |
| Median Drawdown | {np.median(drawdowns)*100:.1f}% |
| Worst Drawdown | {min(drawdowns)*100:.1f}% |
| Periods with >10% Drawdown | {sum(1 for d in drawdowns if d < -0.1)} of {len(drawdowns)} |
"""
    
    def _format_tail_risk(self, context: DeepContext) -> str:
        returns = [p.forward_outcome.return_1m for p in context.similar_periods]
        
        var_95 = np.percentile(returns, 5)
        var_99 = np.percentile(returns, 1)
        
        return f"""
**Value at Risk (95%):** {var_95*100:.1f}%
**Value at Risk (99%):** {var_99*100:.1f}%

Tail risk events in similar periods:
- {sum(1 for r in returns if r < -0.1)} periods with >10% loss
- {sum(1 for r in returns if r < -0.2)} periods with >20% loss
"""
    
    def _format_trade_setup(
        self, 
        context: DeepContext, 
        external: ExternalContext
    ) -> str:
        return f"""
**Entry Signal:** {context.recommended_stance}
**Confidence:** {context.confidence_score:.0%}
**Market Regime:** {context.current_regime.value}
**Market Sentiment:** {external.sentiment.sentiment_level.value}

Setup rationale:
{context.market_interpretation}
"""
    
    def _format_historical_edge(self, context: DeepContext) -> str:
        return f"""
Historical edge analysis:

- **Expected Return (1M):** {context.avg_forward_return_1m:+.1%}
- **Win Rate:** {context.positive_outcome_rate:.0%}
- **Risk/Reward:** {abs(context.avg_forward_return_1m / context.worst_case_drawdown):.2f}

Similar periods showed:
- {context.positive_outcome_rate*100:.0f}% were profitable
- Average winner: {context.best_case_return/2:+.1%}
- Average loser: {context.worst_case_drawdown/2:.1%}
"""
    
    def _format_position_sizing(self, context: DeepContext) -> str:
        if context.confidence_score > 0.7:
            size = "Full position (15-20%)"
        elif context.confidence_score > 0.5:
            size = "Half position (8-10%)"
        else:
            size = "Small position (3-5%)"
        
        return f"""
**Recommended Size:** {size}

Sizing based on:
- Confidence score: {context.confidence_score:.0%}
- Current volatility: {context.current_volatility:.1%}
- Historical win rate: {context.positive_outcome_rate:.0%}

Kelly criterion suggests: {min(0.25, context.positive_outcome_rate * 0.4):.0%} of portfolio
"""
    
    def _format_exit_strategy(self, context: DeepContext) -> str:
        stop_loss = context.worst_case_drawdown / 2
        target = context.avg_forward_return_1m * 1.5
        
        return f"""
**Stop Loss:** {stop_loss:.1%}
**Target:** {target:+.1%}
**Time Horizon:** 1-3 months (based on historical patterns)

Exit conditions:
1. Stop loss triggered at {stop_loss:.1%}
2. Target reached at {target:+.1%}
3. Regime change detected
4. Confidence score drops below 40%

Trailing stop: Consider 50% of gains after {context.avg_forward_return_1m/2:+.1%} profit
"""
