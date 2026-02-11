"""Research Query Engine API endpoints"""
from typing import Optional, List as TypingList
from pydantic import BaseModel
from fastapi import APIRouter, HTTPException, Query

router = APIRouter(prefix="/api/lab", tags=["research"])


class ResearchQueryRequest(BaseModel):
    """Research query request"""
    query_type: str = "stock_analysis"
    symbols: TypingList[str]
    topic: Optional[str] = None
    time_period: str = "1y"
    compare_to: Optional[TypingList[str]] = None


class ResearchReportResponse(BaseModel):
    """Research report response"""
    title: str
    report_type: str
    generated_at: str
    symbols: TypingList[str]
    executive_summary: str
    sections: TypingList[dict]
    recommendation: str
    confidence: float
    key_risks: TypingList[str]
    sources: TypingList[str]
    markdown: str


@router.post("/research", response_model=ResearchReportResponse)
async def generate_research_report(request: ResearchQueryRequest):
    """
    Generate an AI-powered research report.

    Report types:
    - stock_analysis: Deep dive into a single stock
    - market_outlook: Overall market conditions
    - historical_comparison: Compare current to historical periods
    - risk_assessment: Focus on risk factors
    - trade_idea: Actionable trade setup
    """
    try:
        from core.research.engine import ResearchEngine, ResearchQuery, ReportType

        engine = ResearchEngine(persist_directory="./chroma_data")

        # Map string to enum
        report_type_map = {
            "stock_analysis": ReportType.STOCK_ANALYSIS,
            "market_outlook": ReportType.MARKET_OUTLOOK,
            "historical_comparison": ReportType.HISTORICAL_COMPARISON,
            "risk_assessment": ReportType.RISK_ASSESSMENT,
            "trade_idea": ReportType.TRADE_IDEA
        }

        query = ResearchQuery(
            query_type=report_type_map.get(
                request.query_type,
                ReportType.STOCK_ANALYSIS
            ),
            symbols=request.symbols,
            topic=request.topic,
            time_period=request.time_period,
            compare_to=request.compare_to
        )

        report = await engine.generate_report(query)

        return ResearchReportResponse(
            title=report.title,
            report_type=report.report_type.value,
            generated_at=report.generated_at.isoformat(),
            symbols=report.symbols,
            executive_summary=report.executive_summary,
            sections=[
                {"title": s.title, "content": s.content, "data": s.data}
                for s in report.sections
            ],
            recommendation=report.recommendation,
            confidence=report.confidence,
            key_risks=report.key_risks,
            sources=report.sources,
            markdown=report.to_markdown()
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Research generation failed: {str(e)}"
        )


@router.get("/research/{symbol}")
async def get_stock_research(
    symbol: str,
    report_type: str = Query(
        default="stock_analysis",
        pattern="^(stock_analysis|risk_assessment|trade_idea)$"
    )
):
    """
    Quick research report for a single symbol.

    Shortcut endpoint that generates the specified report type.
    """
    try:
        from core.research.engine import ResearchEngine, ResearchQuery, ReportType

        engine = ResearchEngine(persist_directory="./chroma_data")

        type_map = {
            "stock_analysis": ReportType.STOCK_ANALYSIS,
            "risk_assessment": ReportType.RISK_ASSESSMENT,
            "trade_idea": ReportType.TRADE_IDEA
        }

        query = ResearchQuery(
            query_type=type_map.get(report_type, ReportType.STOCK_ANALYSIS),
            symbols=[symbol.upper()]
        )

        report = await engine.generate_report(query)

        return {
            "title": report.title,
            "report_type": report.report_type.value,
            "symbol": symbol.upper(),
            "executive_summary": report.executive_summary,
            "recommendation": report.recommendation,
            "confidence": report.confidence,
            "key_risks": report.key_risks,
            "markdown": report.to_markdown()
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Research generation failed: {str(e)}"
        )


@router.get("/research/market/outlook")
async def get_market_outlook():
    """
    Get current market outlook report.

    Analyzes major indices and market sentiment.
    """
    try:
        from core.research.engine import ResearchEngine, ResearchQuery, ReportType

        engine = ResearchEngine(persist_directory="./chroma_data")

        query = ResearchQuery(
            query_type=ReportType.MARKET_OUTLOOK,
            symbols=["SPY", "QQQ", "IWM"]
        )

        report = await engine.generate_report(query)

        return {
            "title": report.title,
            "executive_summary": report.executive_summary,
            "recommendation": report.recommendation,
            "confidence": report.confidence,
            "key_risks": report.key_risks,
            "markdown": report.to_markdown()
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Market outlook generation failed: {str(e)}"
        )
