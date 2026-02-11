"""
AI Trading Lab FastAPI Application

Main application entry point with middleware and router configuration.
"""
from datetime import datetime
from pathlib import Path
import time
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
import logging

# Load environment variables FIRST before anything else
from dotenv import load_dotenv
# Look for .env.local in project root (parent of python/)
project_root = Path(__file__).resolve().parent.parent.parent
env_local = project_root / ".env.local"
env_file = project_root / ".env"
if env_local.exists():
    load_dotenv(env_local)
elif env_file.exists():
    load_dotenv(env_file)

# Setup logging AFTER env vars loaded
from app.logging_config import setup_logging
setup_logging()

from app.config import get_settings

# Import all routers
from app.backtest_routes import router as backtest_router
from app.replay_routes import router as replay_router
from app.manager_routes import router as manager_router
from app.portfolio_routes import router as portfolio_router
from app.signals_routes import router as signals_router
from app.stock_routes import router as stock_router
from app.embeddings_routes import router as embeddings_router
from app.chat_routes import router as chat_router
from app.market_routes import router as market_router
from app.semantic_routes import router as semantic_router
from app.research_routes import router as research_router
from app.fund_routes import router as fund_router
from app.terminal_routes import router as terminal_router

settings = get_settings()
logger = logging.getLogger(__name__)

app = FastAPI(
    title=settings.app_name,
    description="AI Trading Lab - Where AI Portfolio Managers Compete",
    version="0.1.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify actual origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request logging middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all incoming requests with timing."""
    start_time = time.time()

    # Skip logging for health checks and static files
    path = request.url.path
    skip_paths = ["/health", "/docs", "/openapi.json", "/favicon.ico"]

    if any(path.startswith(p) for p in skip_paths):
        return await call_next(request)

    # Log request
    logger.info(f"[API] {request.method} {path}")

    # Process request
    response = await call_next(request)

    # Log response with timing
    duration_ms = (time.time() - start_time) * 1000
    status = response.status_code

    if status >= 400:
        logger.warning(f"[API] {request.method} {path} -> {status} ({duration_ms:.0f}ms)")
    else:
        logger.info(f"[API] {request.method} {path} -> {status} ({duration_ms:.0f}ms)")

    return response


# Include all routers
app.include_router(backtest_router)
app.include_router(replay_router)
app.include_router(manager_router)
app.include_router(portfolio_router)
app.include_router(signals_router)
app.include_router(stock_router)
app.include_router(embeddings_router)
app.include_router(chat_router)
app.include_router(market_router)
app.include_router(semantic_router)
app.include_router(research_router)
app.include_router(fund_router)
app.include_router(terminal_router)


# Basic endpoints
@app.get("/")
async def root():
    """API root endpoint"""
    return {
        "name": settings.app_name,
        "status": "running",
        "timestamp": datetime.utcnow().isoformat()
    }


@app.get("/health")
async def health():
    """Health check endpoint"""
    return {
        "status": "ok",
        "timestamp": datetime.utcnow().isoformat(),
        "service": "ai-trading-lab-api"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
