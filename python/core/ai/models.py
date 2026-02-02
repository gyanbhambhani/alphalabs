"""
Pydantic models for the AI Stock Terminal.

All data structures used in streaming analysis.
"""

from datetime import datetime
from typing import Literal, Optional, Any
from pydantic import BaseModel, Field


class ChartSpec(BaseModel):
    """Specification for a chart to render on the frontend."""
    
    type: str = Field(
        ...,
        description="Chart type: volatility_regime, sharpe_evolution, "
                    "correlation_heatmap, returns_distribution, similar_periods"
    )
    data: dict = Field(
        ...,
        description="Chart data (dates, values, etc.)"
    )
    config: dict = Field(
        default_factory=dict,
        description="Optional chart configuration (title, colors, etc.)"
    )


class TableSpec(BaseModel):
    """Specification for a data table to render."""
    
    title: str = Field(default="", description="Table title")
    columns: list[str] = Field(..., description="Column headers")
    rows: list[dict] = Field(..., description="Row data as list of dicts")
    highlight_rows: list[int] = Field(
        default_factory=list,
        description="Indices of rows to highlight"
    )


class StreamChunk(BaseModel):
    """A single chunk in the SSE stream."""
    
    type: Literal['text', 'chart', 'table', 'error', 'complete'] = Field(
        ...,
        description="Chunk type determines how frontend renders it"
    )
    content: Any = Field(
        ...,
        description="Chunk content: str for text, ChartSpec for chart, etc."
    )
    metadata: dict = Field(
        default_factory=dict,
        description="Optional metadata (stage, tool name, etc.)"
    )


class AnalysisSession(BaseModel):
    """Analysis session stored in Redis."""
    
    id: str = Field(..., description="Unique session ID")
    query: str = Field(..., description="User's search query")
    symbols: list[str] = Field(..., description="Stock symbols to analyze")
    context: dict = Field(
        default_factory=dict,
        description="Additional context (cache_key, etc.)"
    )
    created_at: datetime = Field(
        default_factory=datetime.now,
        description="Session creation timestamp"
    )
    user_id: Optional[str] = Field(
        default=None,
        description="User ID for rate limiting (if authenticated)"
    )
    
    def to_dict(self) -> dict:
        """Convert to dict for Redis storage."""
        return {
            'id': self.id,
            'query': self.query,
            'symbols': self.symbols,
            'context': self.context,
            'created_at': self.created_at.isoformat(),
            'user_id': self.user_id
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "AnalysisSession":
        """Create from Redis dict."""
        return cls(
            id=data['id'],
            query=data['query'],
            symbols=data['symbols'],
            context=data.get('context', {}),
            created_at=datetime.fromisoformat(data['created_at']),
            user_id=data.get('user_id')
        )


class ToolResult(BaseModel):
    """Result from executing a quant tool."""
    
    name: str = Field(..., description="Tool name that generated this result")
    chart: Optional[ChartSpec] = Field(
        default=None,
        description="Chart specification if tool generates a chart"
    )
    table: Optional[TableSpec] = Field(
        default=None,
        description="Table specification if tool generates a table"
    )
    explanation: str = Field(
        ...,
        description="Pre-written explanation of the result (no AI needed)"
    )
    metrics: dict = Field(
        default_factory=dict,
        description="Key metrics for AI synthesis context"
    )


class ToolExecutionError(Exception):
    """Raised when a tool fails to execute."""
    
    def __init__(self, tool_name: str, reason: str):
        self.tool_name = tool_name
        self.reason = reason
        super().__init__(f"{tool_name} failed: {reason}")


class AnalysisPlan(BaseModel):
    """Plan for what analysis to perform."""
    
    tools: list[str] = Field(
        ...,
        description="List of tools to execute"
    )
    reasoning: str = Field(
        ...,
        description="Human-readable explanation of what we're analyzing"
    )
    needs_ai_synthesis: bool = Field(
        default=False,
        description="Whether to call OpenAI for interpretation"
    )
    confidence: Literal['high', 'medium', 'low'] = Field(
        default='medium',
        description="Confidence in the plan based on keyword matching"
    )
