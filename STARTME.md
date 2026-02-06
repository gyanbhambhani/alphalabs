# Quick Start Guide

## Option 1: Docker (Recommended)

Start everything with one command:

```bash
# From project root
docker-compose -f docker/docker-compose.yml up
```

### Docker Commands

```bash
# Start all services
docker-compose -f docker/docker-compose.yml up

# Start in background
docker-compose -f docker/docker-compose.yml up -d

# View backend logs (with all the debug info)
docker-compose -f docker/docker-compose.yml logs -f api

# View last 100 lines of backend logs
docker-compose -f docker/docker-compose.yml logs --tail=100 api

# View frontend logs
docker-compose -f docker/docker-compose.yml logs -f frontend

# Restart just the backend
docker-compose -f docker/docker-compose.yml restart api

# Restart just the frontend
docker-compose -f docker/docker-compose.yml restart frontend

# Stop everything
docker-compose -f docker/docker-compose.yml down

# Rebuild after dependency changes
docker-compose -f docker/docker-compose.yml up --build
```

### Log Levels

Set `LOG_LEVEL` in `docker/docker-compose.yml`:
- `DEBUG` - Everything including raw LLM responses
- `INFO` - Decisions, trades, consensus (recommended)
- `WARNING` - Only failures and rejections
- `ERROR` - Only errors

## Option 2: Manual Start

### 1. Backend (Terminal 1)
```bash
cd /Users/gyanb/Desktop/alphalabs/python
python3.11 -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### 2. Frontend (Terminal 2)
```bash
cd /Users/gyanb/Desktop/alphalabs/frontend
npm run dev
```

### 3. Populate Database
```bash
cd /Users/gyanb/Desktop/alphalabs/python && python3.11 scripts/populate_stocks_table.py
```

## Access Points

| URL | Description |
|-----|-------------|
| http://localhost:3000 | Main Dashboard |
| http://localhost:3000/lab | Trading Lab (AI Chat + Market Intelligence) |
| http://localhost:3000/backtest | Time Machine Backtest |
| http://localhost:8000/docs | API Documentation |

## Trading Lab Features

Ask questions like:
- "Analyze AAPL"
- "What's the current market sentiment?"
- "Compare NVDA and AMD"
- "Generate a research report on MSFT"
- "What happened in 2008?"

## Prerequisites

Make sure these are running:
- PostgreSQL database (or use Docker)
- ChromaDB data in `python/chroma_data/`

## Environment Variables

Create `.env.local` in project root with:
```
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
```
