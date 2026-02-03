"""
Streaming Quant Analyzer for AI Stock Terminal.

Stream-first architecture:
1. Stream text immediately (0.5s to first content)
2. Execute tools in parallel
3. Stream charts/tables as they complete
4. Optional AI synthesis at the end (via LangChain)

Graceful error handling:
- Per-tool try/catch
- Partial results always shown
- Warnings for failed tools (not errors)

Refactored to use LangChain for AI synthesis streaming.
"""

import asyncio
import os
import logging
from typing import AsyncGenerator, Optional

from core.ai.models import (
    StreamChunk,
    AnalysisSession,
    ToolResult,
    ToolExecutionError,
)
from core.ai.query_planner import QueryPlanner
from core.ai.quant_tools import execute_tool

# Import LangChain streaming components
from core.langchain.agents import StreamingLLM

logger = logging.getLogger(__name__)


class StreamingQuantAnalyzer:
    """
    Stream-first analyzer with graceful error degradation.
    
    Key principles:
    1. Always show something (partial results > no results)
    2. Fail individual tools, not entire stream
    3. Pre-written explanations (no AI wait for basic insights)
    4. Optional AI synthesis only when needed (via LangChain)
    """
    
    TOOL_TIMEOUT = 30  # Max seconds per tool
    
    def __init__(self, openai_api_key: Optional[str] = None):
        """
        Initialize analyzer.
        
        Args:
            openai_api_key: OpenAI API key for optional synthesis
        """
        self.planner = QueryPlanner()
        
        # Store API key for LangChain
        self._api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        
        # LangChain streaming LLM (lazily initialized)
        self._streaming_llm: Optional[StreamingLLM] = None
    
    def _get_streaming_llm(self) -> Optional[StreamingLLM]:
        """Lazily initialize LangChain streaming LLM."""
        if self._streaming_llm is None and self._api_key:
            self._streaming_llm = StreamingLLM(
                provider="openai",
                model="gpt-4o-mini",
            )
        return self._streaming_llm
    
    async def analyze_stream(
        self,
        session: AnalysisSession
    ) -> AsyncGenerator[StreamChunk, None]:
        """
        Stream analysis results as they become available.
        
        Yields StreamChunk objects in order:
        1. Initial text (immediately)
        2. Charts/tables as tools complete
        3. Explanations for each result
        4. Optional AI synthesis (via LangChain streaming)
        5. Complete marker
        
        Handles errors gracefully - individual tool failures don't
        stop the stream, they produce warnings.
        """
        try:
            # 1. IMMEDIATE TEXT - show user we're working
            yield StreamChunk(
                type='text',
                content=f"Analyzing {', '.join(session.symbols)}...",
                metadata={'stage': 'init'}
            )
            
            # 2. PLAN ANALYSIS (no OpenAI, instant)
            plan = self.planner.plan(session.query, session.symbols)
            
            yield StreamChunk(
                type='text',
                content=f"{plan.reasoning}. Generating {len(plan.tools)} visualizations.",
                metadata={'stage': 'planning', 'confidence': plan.confidence}
            )
            
            # 3. EXECUTE TOOLS IN PARALLEL with individual error handling
            successful_results: list[ToolResult] = []
            failed_tools: list[str] = []
            
            # Create tasks that return (tool_name, result) tuples
            async def run_tool(tool_name: str) -> tuple[str, ToolResult]:
                result = await self._execute_tool_safe(tool_name, session.symbols[0])
                return (tool_name, result)
            
            tasks = [
                asyncio.create_task(run_tool(tool))
                for tool in plan.tools
            ]
            
            # Process results as they complete using asyncio.wait
            pending = set(tasks)
            while pending:
                done, pending = await asyncio.wait(
                    pending,
                    return_when=asyncio.FIRST_COMPLETED
                )
                
                for task in done:
                    try:
                        tool_name, result = task.result()
                        
                        # Stream chart if present
                        if result.chart:
                            yield StreamChunk(
                                type='chart',
                                content=result.chart.model_dump(),
                                metadata={'tool': tool_name}
                            )
                        
                        # Stream table if present
                        if result.table:
                            yield StreamChunk(
                                type='table',
                                content=result.table.model_dump(),
                                metadata={'tool': tool_name}
                            )
                        
                        # Stream explanation
                        yield StreamChunk(
                            type='text',
                            content=result.explanation,
                            metadata={'tool': tool_name, 'stage': 'result'}
                        )
                        
                        successful_results.append(result)
                        
                    except ToolExecutionError as e:
                        # Tool failed - warn user but CONTINUE
                        failed_tools.append(e.tool_name)
                        yield StreamChunk(
                            type='text',
                            content=f"Could not generate {e.tool_name} ({e.reason})",
                            metadata={'stage': 'warning', 'tool': e.tool_name}
                        )
                        
                    except Exception as e:
                        # Unexpected error - log and warn
                        logger.error(f"Unexpected tool error: {e}", exc_info=True)
                        failed_tools.append("unknown")
                        yield StreamChunk(
                            type='text',
                            content=f"Could not generate analysis ({type(e).__name__})",
                            metadata={'stage': 'warning'}
                        )
            
            # 4. CHECK IF ALL TOOLS FAILED
            if not successful_results:
                yield StreamChunk(
                    type='error',
                    content={
                        'message': 'Could not generate any visualizations. '
                                   'Please try again later.',
                        'failed_tools': failed_tools
                    },
                    metadata={'stage': 'error'}
                )
                return
            
            # 5. OPTIONAL AI SYNTHESIS (via LangChain streaming)
            streaming_llm = self._get_streaming_llm()
            if plan.needs_ai_synthesis and streaming_llm:
                try:
                    async for chunk in self._synthesize_with_langchain(
                        session, 
                        successful_results
                    ):
                        yield chunk
                except Exception as e:
                    logger.warning(f"AI synthesis failed: {e}")
                    yield StreamChunk(
                        type='text',
                        content=(
                            "(AI analysis unavailable, but charts above "
                            "show the data you need)"
                        ),
                        metadata={'stage': 'ai_fallback'}
                    )
            
            # 6. COMPLETE
            summary = f"Analyzed {len(successful_results)} metrics"
            if failed_tools:
                summary += f" ({len(failed_tools)} unavailable)"
            
            yield StreamChunk(
                type='complete',
                content=summary,
                metadata={
                    'successful': len(successful_results),
                    'failed': len(failed_tools)
                }
            )
            
        except Exception as e:
            # Stream-level error - shouldn't happen but handle gracefully
            logger.error(f"Stream failed: {e}", exc_info=True)
            yield StreamChunk(
                type='error',
                content={'message': 'Analysis failed. Please try again.'},
                metadata={'stage': 'fatal_error'}
            )
    
    async def _execute_tool_safe(
        self,
        tool_name: str,
        symbol: str
    ) -> ToolResult:
        """
        Execute a tool with timeout and error handling.
        
        Raises ToolExecutionError on failure (never crashes).
        """
        try:
            async with asyncio.timeout(self.TOOL_TIMEOUT):
                return await execute_tool(tool_name, symbol)
                
        except asyncio.TimeoutError:
            raise ToolExecutionError(
                tool_name, 
                f"Timeout (>{self.TOOL_TIMEOUT}s)"
            )
        except ToolExecutionError:
            # Re-raise - already properly formatted
            raise
        except Exception as e:
            # Wrap unexpected errors
            error_type = type(e).__name__
            raise ToolExecutionError(tool_name, error_type)
    
    async def _synthesize_with_langchain(
        self,
        session: AnalysisSession,
        results: list[ToolResult]
    ) -> AsyncGenerator[StreamChunk, None]:
        """
        Optional AI synthesis of results using LangChain streaming.
        
        Only called when:
        1. User asked a question (needs interpretation)
        2. LangChain is configured
        3. We have successful results to interpret
        """
        streaming_llm = self._get_streaming_llm()
        if not streaming_llm:
            return
        
        # Build context from results
        metrics_summary = "\n".join([
            f"- {r.name}: {r.explanation}"
            for r in results
        ])
        
        prompt = f"""You are a concise financial analyst. Give direct answers based on data.

User question: {session.query}
Symbol(s): {', '.join(session.symbols)}

Analysis results:
{metrics_summary}

Key metrics:
{self._format_metrics(results)}

Provide a 2-3 sentence interpretation that directly answers the user's question.
Focus on actionable insights. Don't repeat the raw numbers. No disclaimers."""

        yield StreamChunk(
            type='text',
            content="\n---\n**AI Analysis:**\n",
            metadata={'stage': 'ai_synthesis'}
        )
        
        try:
            # Stream AI response using LangChain
            async for text_chunk in streaming_llm.astream(prompt):
                yield StreamChunk(
                    type='text',
                    content=text_chunk,
                    metadata={'stage': 'ai_synthesis', 'streaming': True}
                )
                    
        except Exception as e:
            logger.warning(f"LangChain streaming failed: {e}")
            raise
    
    def _format_metrics(self, results: list[ToolResult]) -> str:
        """Format key metrics for AI context."""
        lines = []
        
        for r in results:
            if r.metrics:
                for key, value in r.metrics.items():
                    if isinstance(value, float):
                        lines.append(f"{r.name}.{key}: {value:.3f}")
                    else:
                        lines.append(f"{r.name}.{key}: {value}")
        
        return "\n".join(lines) if lines else "No numeric metrics available."


# Convenience function for simple usage
async def analyze_query(
    query: str,
    symbols: list[str],
    openai_api_key: Optional[str] = None
) -> AsyncGenerator[StreamChunk, None]:
    """
    Simple interface to stream analysis.
    
    Usage:
        async for chunk in analyze_query("Is NVDA risky?", ["NVDA"]):
            print(chunk)
    """
    session = AnalysisSession(
        id="temp",
        query=query,
        symbols=symbols
    )
    
    analyzer = StreamingQuantAnalyzer(openai_api_key)
    
    async for chunk in analyzer.analyze_stream(session):
        yield chunk
