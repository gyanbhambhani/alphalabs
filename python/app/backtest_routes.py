"""
Backtest API Routes - AI Fund Time Machine.

SSE streaming endpoints for backtesting visualization.
Includes persistence for training data and exports.
"""

import asyncio
import json
import uuid
from datetime import date, datetime
from typing import Dict, List, Optional
from pydantic import BaseModel, Field

from fastapi import APIRouter, HTTPException, Query, BackgroundTasks
from fastapi.responses import StreamingResponse, Response

from core.backtest import (
    HistoricalDataLoader,
    SimulationEngine,
    FundConfig,
    BACKTEST_UNIVERSE,
)
from core.backtest.persistence import BacktestPersistence, get_persistence

router = APIRouter(prefix="/api/backtest", tags=["backtest"])

# Initialize persistence layer (singleton)
_persistence: Optional[BacktestPersistence] = None


def get_backtest_persistence() -> BacktestPersistence:
    """Get or create the persistence instance."""
    global _persistence
    if _persistence is None:
        _persistence = get_persistence()
    return _persistence

# Store active simulations
_simulations: Dict[str, SimulationEngine] = {}


# =============================================================================
# Request/Response Models
# =============================================================================

class FundConfigRequest(BaseModel):
    """Fund configuration for backtest."""
    fund_id: str = Field(..., alias="fundId")
    name: str
    thesis: str
    initial_cash: float = Field(default=100_000.0, alias="initialCash")
    
    model_config = {"populate_by_name": True}


class BacktestStartRequest(BaseModel):
    """Request to start a backtest simulation."""
    funds: List[FundConfigRequest]
    start_date: Optional[str] = Field(default=None, alias="startDate")
    end_date: Optional[str] = Field(default=None, alias="endDate")
    speed: float = 100.0
    initial_cash: float = Field(default=100_000.0, alias="initialCash")
    
    model_config = {"populate_by_name": True}


class BacktestStatusResponse(BaseModel):
    """Backtest simulation status."""
    simulation_id: str = Field(..., alias="simulationId")
    status: str
    progress: float
    current_date: Optional[str] = Field(default=None, alias="currentDate")
    funds: List[Dict]
    
    model_config = {"populate_by_name": True}


class BacktestControlRequest(BaseModel):
    """Control request for backtest (pause/resume/speed)."""
    action: str  # "pause", "resume", "stop", "speed"
    speed: Optional[float] = None


# =============================================================================
# Endpoints
# =============================================================================

@router.get("/universe")
async def get_backtest_universe():
    """Get the default universe of stocks for backtesting."""
    return {
        "universe": BACKTEST_UNIVERSE,
        "count": len(BACKTEST_UNIVERSE),
        "description": "Liquid stocks that existed throughout 2000-2025"
    }


@router.get("/data/status")
async def get_data_status():
    """Check if historical data is cached and ready (without triggering downloads)."""
    try:
        loader = HistoricalDataLoader()
        status = loader.cache_status()
        
        return {
            "ready": status["ready"],
            "symbols_loaded": status["symbols_cached"],
            "total_symbols": status["total_symbols"],
            "trading_days": status["trading_days"],
            "date_range": {
                "start": status["start_date"],
                "end": status["end_date"]
            },
            "cache_dir": status["cache_dir"],
            "chroma_embeddings": {
                "symbols": status.get("chroma_symbols", 0),
                "date_range": status.get("chroma_date_range"),
            }
        }
    except Exception as e:
        return {
            "ready": False,
            "error": str(e)
        }


# Track download progress
_download_progress = {"status": "idle", "current": 0, "total": 0, "symbol": ""}


@router.post("/data/download")
async def download_historical_data(
    background_tasks: BackgroundTasks,
    force_refresh: bool = False
):
    """
    Download and cache historical data.
    
    This fetches 25 years of daily OHLCV data for the backtest universe.
    Data is cached to disk as parquet files - only needs to run ONCE.
    """
    global _download_progress
    
    if _download_progress["status"] == "downloading":
        return {
            "status": "already_running",
            "message": "Download already in progress",
            "progress": _download_progress
        }
    
    def download_task():
        global _download_progress
        loader = HistoricalDataLoader()
        universe = loader.universe
        
        _download_progress = {
            "status": "downloading",
            "current": 0,
            "total": len(universe),
            "symbol": ""
        }
        
        for i, symbol in enumerate(universe):
            _download_progress["current"] = i + 1
            _download_progress["symbol"] = symbol
            
            # Skip if already cached (unless force refresh)
            if not force_refresh and loader._is_cached(symbol):
                continue
            
            # Fetch and cache
            df = loader._fetch_from_yfinance(symbol)
            if df is not None and not df.empty:
                loader._save_to_cache(symbol, df)
        
        _download_progress["status"] = "complete"
        _download_progress["symbol"] = ""
    
    # Run in background thread (not async - yfinance is blocking)
    import threading
    thread = threading.Thread(target=download_task)
    thread.start()
    
    return {
        "status": "started",
        "message": "Historical data download started. Check /api/backtest/data/progress"
    }


@router.get("/data/progress")
async def get_download_progress():
    """Get current download progress."""
    return _download_progress


@router.post("/start")
async def start_backtest(request: BacktestStartRequest):
    """
    Start a new backtest simulation.
    
    Returns a simulation_id to use for streaming and control.
    """
    try:
        # Parse dates
        start = None
        end = None
        
        if request.start_date:
            start = date.fromisoformat(request.start_date)
        if request.end_date:
            end = date.fromisoformat(request.end_date)
        
        # Create fund configs
        funds = [
            FundConfig(
                fund_id=f.fund_id,
                name=f.name,
                thesis=f.thesis,
                initial_cash=f.initial_cash,
            )
            for f in request.funds
        ]
        
        if not funds:
            raise HTTPException(
                status_code=400,
                detail="At least one fund configuration required"
            )
        
        # Load data
        loader = HistoricalDataLoader()
        
        # Create simulation engine
        simulation_id = str(uuid.uuid4())[:8]
        
        engine = SimulationEngine(
            funds=funds,
            data_loader=loader,
            start_date=start,
            end_date=end,
            speed_multiplier=request.speed,
            initial_cash=request.initial_cash,
        )
        
        _simulations[simulation_id] = engine
        
        return {
            "simulation_id": simulation_id,
            "status": "created",
            "funds": [{"id": f.fund_id, "name": f.name} for f in funds],
            "total_days": len(engine.trading_days),
            "date_range": {
                "start": engine.start_date.isoformat(),
                "end": engine.end_date.isoformat()
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/stream/{simulation_id}")
async def stream_backtest(simulation_id: str):
    """
    Stream backtest events via Server-Sent Events (SSE).
    
    Connect after creating simulation with POST /api/backtest/start.
    
    Event types:
    - simulation_start: Simulation beginning
    - day_tick: New trading day
    - debate_start: Fund starting debate
    - debate_message: Message in debate
    - decision: Trading decision made
    - trade_executed: Trade filled
    - portfolio_update: Portfolio state update
    - leaderboard: Fund rankings
    - simulation_end: Simulation complete
    - error: Error occurred
    
    Example frontend:
    ```javascript
    const eventSource = new EventSource(`/api/backtest/stream/${simulationId}`);
    eventSource.onmessage = (event) => {
        const data = JSON.parse(event.data);
        // Handle event based on data.event_type
    };
    ```
    """
    if simulation_id not in _simulations:
        raise HTTPException(status_code=404, detail="Simulation not found")
    
    engine = _simulations[simulation_id]
    
    async def generate_events():
        """Generate SSE events from simulation."""
        # Start simulation in background
        simulation_task = asyncio.create_task(engine.run())
        
        try:
            # Stream events
            async for event in engine.events():
                data = event.to_dict()
                yield f"data: {json.dumps(data)}\n\n"
            
            # Wait for simulation to complete
            await simulation_task
            
        except asyncio.CancelledError:
            engine.stop()
        except Exception as e:
            error_data = json.dumps({
                "event_type": "error",
                "message": str(e),
                "timestamp": datetime.utcnow().isoformat()
            })
            yield f"data: {error_data}\n\n"
        finally:
            # Cleanup
            if simulation_id in _simulations:
                del _simulations[simulation_id]
    
    return StreamingResponse(
        generate_events(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
            "Access-Control-Allow-Origin": "*",
        }
    )


@router.post("/control/{simulation_id}")
async def control_backtest(simulation_id: str, request: BacktestControlRequest):
    """
    Control a running backtest simulation.
    
    Actions:
    - pause: Pause simulation
    - resume: Resume simulation
    - stop: Stop simulation
    - speed: Change simulation speed (requires speed parameter)
    """
    if simulation_id not in _simulations:
        raise HTTPException(status_code=404, detail="Simulation not found")
    
    engine = _simulations[simulation_id]
    
    if request.action == "pause":
        engine.pause()
        return {"status": "paused"}
    
    elif request.action == "resume":
        engine.resume()
        return {"status": "resumed"}
    
    elif request.action == "stop":
        engine.stop()
        return {"status": "stopped"}
    
    elif request.action == "speed":
        if request.speed is None:
            raise HTTPException(
                status_code=400,
                detail="Speed parameter required for speed action"
            )
        engine.set_speed(request.speed)
        return {"status": "speed_changed", "new_speed": request.speed}
    
    else:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown action: {request.action}"
        )


@router.get("/status/{simulation_id}", response_model=BacktestStatusResponse)
async def get_backtest_status(simulation_id: str):
    """Get current status of a backtest simulation."""
    if simulation_id not in _simulations:
        raise HTTPException(status_code=404, detail="Simulation not found")
    
    engine = _simulations[simulation_id]
    
    # Determine status
    if engine.stopped:
        status = "stopped"
    elif engine.paused:
        status = "paused"
    elif engine.progress >= 1.0:
        status = "completed"
    else:
        status = "running"
    
    return BacktestStatusResponse(
        simulation_id=simulation_id,
        status=status,
        progress=engine.progress,
        current_date=engine.current_date.isoformat() if engine.current_date else None,
        funds=[
            {
                "fund_id": f.fund_id,
                "name": f.name,
                "total_value": engine.portfolios[f.fund_id].total_value,
                "cumulative_return": engine.portfolios[f.fund_id].cumulative_return,
            }
            for f in engine.funds
        ]
    )


@router.get("/list")
async def list_simulations():
    """List all active simulations."""
    return {
        "simulations": [
            {
                "simulation_id": sim_id,
                "progress": engine.progress,
                "status": "paused" if engine.paused else "running",
                "current_date": (
                    engine.current_date.isoformat() if engine.current_date else None
                ),
                "funds": len(engine.funds),
            }
            for sim_id, engine in _simulations.items()
        ]
    }


@router.delete("/{simulation_id}")
async def delete_simulation(simulation_id: str):
    """Stop and delete a simulation."""
    if simulation_id not in _simulations:
        raise HTTPException(status_code=404, detail="Simulation not found")
    
    engine = _simulations[simulation_id]
    engine.stop()
    del _simulations[simulation_id]
    
    return {"status": "deleted"}


# =============================================================================
# Quick Start Templates
# =============================================================================

@router.get("/templates")
async def get_fund_templates():
    """Get pre-configured fund templates for quick start."""
    return {
        "templates": [
            {
                "id": "momentum",
                "name": "Momentum Fund",
                "thesis": (
                    "Buy stocks with strong 12-month momentum, skip the most recent month. "
                    "Focus on tech and growth stocks. Sell when momentum weakens."
                ),
                "description": "Classic momentum strategy"
            },
            {
                "id": "value",
                "name": "Value Fund",
                "thesis": (
                    "Buy undervalued stocks with low P/E ratios and strong fundamentals. "
                    "Focus on financials and industrials. Patient, long-term holding."
                ),
                "description": "Warren Buffett style value investing"
            },
            {
                "id": "mean_reversion",
                "name": "Mean Reversion Fund",
                "thesis": (
                    "Buy oversold stocks (RSI < 30, negative 1-month returns). "
                    "Sell when they revert to mean. Quick trades, high turnover."
                ),
                "description": "Buy the dip, sell the rip"
            },
            {
                "id": "quant",
                "name": "Quant Fund",
                "thesis": (
                    "Combine multiple signals: momentum, mean reversion, volatility. "
                    "Trade based on signal strength, not gut feel. Systematic approach."
                ),
                "description": "Multi-factor quantitative strategy"
            }
        ]
    }


@router.post("/quick-start")
async def quick_start_backtest(
    template_ids: List[str] = Query(
        default=["momentum", "value", "mean_reversion"],
        description="Template IDs to use"
    ),
    speed: float = Query(default=100.0, description="Simulation speed"),
    initial_cash: float = Query(default=100_000.0, description="Initial cash per fund"),
):
    """
    Quick start a backtest with pre-configured fund templates.
    
    Returns a simulation_id to use for streaming.
    """
    templates = {
        "momentum": FundConfig(
            fund_id="momentum_fund",
            name="Momentum Fund",
            thesis=(
                "Buy stocks with strong 12-month momentum, skip the most recent month. "
                "Focus on tech and growth stocks. Sell when momentum weakens."
            ),
            initial_cash=initial_cash,
        ),
        "value": FundConfig(
            fund_id="value_fund",
            name="Value Fund",
            thesis=(
                "Buy undervalued stocks with low P/E ratios and strong fundamentals. "
                "Focus on financials and industrials. Patient, long-term holding."
            ),
            initial_cash=initial_cash,
        ),
        "mean_reversion": FundConfig(
            fund_id="mean_reversion_fund",
            name="Mean Reversion Fund",
            thesis=(
                "Buy oversold stocks (RSI < 30, negative 1-month returns). "
                "Sell when they revert to mean. Quick trades, high turnover."
            ),
            initial_cash=initial_cash,
        ),
        "quant": FundConfig(
            fund_id="quant_fund",
            name="Quant Fund",
            thesis=(
                "Combine multiple signals: momentum, mean reversion, volatility. "
                "Trade based on signal strength, not gut feel. Systematic approach."
            ),
            initial_cash=initial_cash,
        ),
    }
    
    funds = [templates[tid] for tid in template_ids if tid in templates]
    
    if not funds:
        raise HTTPException(
            status_code=400,
            detail="No valid template IDs provided"
        )
    
    # Load data
    loader = HistoricalDataLoader()
    
    # Create simulation with persistence
    simulation_id = str(uuid.uuid4())[:8]
    persistence = get_backtest_persistence()
    
    engine = SimulationEngine(
        funds=funds,
        data_loader=loader,
        speed_multiplier=speed,
        initial_cash=initial_cash,
        persistence=persistence,
    )
    
    _simulations[simulation_id] = engine
    
    return {
        "simulation_id": simulation_id,
        "status": "created",
        "funds": [{"id": f.fund_id, "name": f.name} for f in funds],
        "total_days": len(engine.trading_days),
        "message": f"Connect to /api/backtest/stream/{simulation_id} to start streaming"
    }


# =============================================================================
# History & Export Endpoints
# =============================================================================

@router.get("/history/runs")
async def get_all_runs():
    """Get all historical backtest runs."""
    persistence = get_backtest_persistence()
    runs = persistence.get_all_runs()
    return {"runs": runs, "total": len(runs)}


@router.get("/history/runs/{run_id}")
async def get_run_details(run_id: str):
    """Get details for a specific run."""
    persistence = get_backtest_persistence()
    run = persistence.get_run(run_id)
    
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")
    
    # Get statistics
    stats = persistence.get_statistics(run_id)
    
    return {
        "run": run,
        "statistics": stats,
    }


@router.get("/history/runs/{run_id}/trades")
async def get_run_trades(
    run_id: str,
    fund_id: Optional[str] = None,
):
    """Get all trades for a run."""
    persistence = get_backtest_persistence()
    trades = persistence.get_trades_for_run(run_id, fund_id)
    return {"trades": trades, "total": len(trades)}


@router.get("/history/runs/{run_id}/decisions")
async def get_run_decisions(
    run_id: str,
    fund_id: Optional[str] = None,
    action: Optional[str] = None,
):
    """Get all decisions for a run."""
    persistence = get_backtest_persistence()
    decisions = persistence.get_decisions_for_run(run_id, fund_id, action)
    return {"decisions": decisions, "total": len(decisions)}


@router.get("/history/runs/{run_id}/debates")
async def get_run_debates(
    run_id: str,
    fund_id: Optional[str] = None,
):
    """
    Get all debate transcripts for a run.
    
    Returns the full 4-phase LangChain debate for each decision:
    - analyze: Market analysis (Gemini)
    - propose: Trade proposal (GPT-4o-mini)
    - decide: Final decision with structured output (GPT-4o-mini)
    - confirm: Risk confirmation for major trades (Claude Haiku)
    
    Each phase includes the model used, content, and token count.
    """
    persistence = get_backtest_persistence()
    transcripts = persistence.get_debate_transcripts_for_run(run_id, fund_id)
    
    return {
        "debates": transcripts,
        "total": len(transcripts),
        "phases": ["analyze", "propose", "decide", "confirm"],
        "description": "Full LangChain debate transcripts for each trading decision"
    }


@router.get("/history/decisions/{decision_id}")
async def get_decision_detail(decision_id: str):
    """
    Get a single decision with its full debate transcript.
    
    Returns complete debate including all phases and full content.
    """
    persistence = get_backtest_persistence()
    decision = persistence.get_decision_with_transcript(decision_id)
    
    if not decision:
        raise HTTPException(status_code=404, detail="Decision not found")
    
    return decision


@router.get("/history/runs/{run_id}/snapshots")
async def get_run_snapshots(
    run_id: str,
    fund_id: Optional[str] = None,
):
    """Get portfolio snapshots for a run (equity curve data)."""
    persistence = get_backtest_persistence()
    snapshots = persistence.get_snapshots_for_run(run_id, fund_id)
    return {"snapshots": snapshots, "total": len(snapshots)}


@router.get("/history/runs/{run_id}/metrics")
async def get_run_metrics(
    run_id: str,
    fund_id: Optional[str] = None,
):
    """
    Get fund evaluation metrics for a run.
    
    Returns metrics including:
    - hit_rate: Percentage of trades with correct direction prediction
    - brier_score: Calibration score (lower is better, 0 = perfect)
    - turnover: Annualized portfolio turnover
    - avg_slippage_bps: Average slippage in basis points
    - n_trades: Total number of trades
    - avg_return: Average return per trade
    - max_drawdown: Maximum portfolio drawdown
    """
    persistence = get_backtest_persistence()
    metrics = persistence.get_fund_metrics(run_id, fund_id)
    
    if not metrics:
        return {
            "metrics": [],
            "total": 0,
            "message": "No metrics found. Metrics are computed when simulation completes."
        }
    
    return {"metrics": metrics, "total": len(metrics)}


@router.get("/metrics/{simulation_id}")
async def get_live_metrics(simulation_id: str):
    """
    Get current metrics from a running or completed simulation.
    
    Returns fund metrics directly from the simulation engine if still active,
    or from persistence if the simulation has completed.
    """
    # Check if simulation is still active
    if simulation_id in _simulations:
        engine = _simulations[simulation_id]
        
        # Return metrics from active simulation
        metrics_list = []
        for fund_id, metrics in engine.fund_metrics.items():
            from dataclasses import asdict
            metrics_dict = asdict(metrics)
            metrics_dict["fund_id"] = fund_id
            metrics_list.append(metrics_dict)
        
        return {
            "simulation_id": simulation_id,
            "status": "running" if not engine.stopped else "completed",
            "metrics": metrics_list,
            "source": "live"
        }
    
    # Check persistence for completed run
    persistence = get_backtest_persistence()
    metrics = persistence.get_fund_metrics(simulation_id)
    
    if metrics:
        return {
            "simulation_id": simulation_id,
            "status": "completed",
            "metrics": metrics,
            "source": "persistence"
        }
    
    raise HTTPException(
        status_code=404, 
        detail=f"Simulation {simulation_id} not found"
    )


# =============================================================================
# Export Endpoints
# =============================================================================

@router.get("/export/{run_id}/trades.csv")
async def export_trades_csv(run_id: str):
    """Export trades as CSV."""
    persistence = get_backtest_persistence()
    csv_content = persistence.export_trades_csv(run_id)
    
    if not csv_content:
        raise HTTPException(status_code=404, detail="No trades found for run")
    
    return Response(
        content=csv_content,
        media_type="text/csv",
        headers={
            "Content-Disposition": f"attachment; filename=trades_{run_id}.csv"
        }
    )


@router.get("/export/{run_id}/decisions.csv")
async def export_decisions_csv(run_id: str):
    """Export decisions as CSV."""
    persistence = get_backtest_persistence()
    csv_content = persistence.export_decisions_csv(run_id)
    
    if not csv_content:
        raise HTTPException(status_code=404, detail="No decisions found for run")
    
    return Response(
        content=csv_content,
        media_type="text/csv",
        headers={
            "Content-Disposition": f"attachment; filename=decisions_{run_id}.csv"
        }
    )


@router.get("/export/{run_id}/snapshots.csv")
async def export_snapshots_csv(run_id: str):
    """Export portfolio snapshots as CSV."""
    persistence = get_backtest_persistence()
    csv_content = persistence.export_snapshots_csv(run_id)
    
    if not csv_content:
        raise HTTPException(status_code=404, detail="No snapshots found for run")
    
    return Response(
        content=csv_content,
        media_type="text/csv",
        headers={
            "Content-Disposition": f"attachment; filename=snapshots_{run_id}.csv"
        }
    )


@router.get("/export/{run_id}/full.json")
async def export_full_json(run_id: str):
    """Export complete run data as JSON (for training)."""
    persistence = get_backtest_persistence()
    data = persistence.export_run_json(run_id)
    
    if not data:
        raise HTTPException(status_code=404, detail="Run not found")
    
    return Response(
        content=json.dumps(data, default=str, indent=2),
        media_type="application/json",
        headers={
            "Content-Disposition": f"attachment; filename=backtest_{run_id}.json"
        }
    )


@router.get("/export/training-data")
async def export_training_data(run_id: Optional[str] = None):
    """
    Export data formatted for ML training.
    
    Each record contains market state, decision, and can be used
    to train models to make better trading decisions.
    """
    persistence = get_backtest_persistence()
    training_data = persistence.export_training_data(run_id)
    
    return {
        "training_data": training_data,
        "total_records": len(training_data),
        "description": "Each record contains signals, decision, and outcome for training",
    }
