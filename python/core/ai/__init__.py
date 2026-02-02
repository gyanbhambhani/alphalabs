"""
AI Stock Terminal Module

Stream-first architecture for real-time stock analysis with:
- Smart query planning (95% coverage)
- ChromaDB similar_periods integration
- Graceful error degradation
- Optional AI synthesis
"""

# Lazy imports to avoid circular dependencies
__all__ = [
    "ChartSpec",
    "TableSpec",
    "StreamChunk",
    "AnalysisSession",
    "ToolResult",
    "ToolExecutionError",
    "AnalysisPlan",
    "QueryPlanner",
    "StreamingQuantAnalyzer",
]


def __getattr__(name):
    """Lazy import to avoid circular dependencies."""
    if name in ("ChartSpec", "TableSpec", "StreamChunk", "AnalysisSession", 
                "ToolResult", "ToolExecutionError", "AnalysisPlan"):
        from core.ai import models
        return getattr(models, name)
    elif name == "QueryPlanner":
        from core.ai.query_planner import QueryPlanner
        return QueryPlanner
    elif name == "StreamingQuantAnalyzer":
        from core.ai.streaming_analyzer import StreamingQuantAnalyzer
        return StreamingQuantAnalyzer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
