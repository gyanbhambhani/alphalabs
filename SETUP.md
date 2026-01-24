# AI Trading Lab - Setup & Run Guide

This guide will help you get the AI Trading Lab up and running.

## Prerequisites

- Python 3.11+
- Node.js 18+
- PostgreSQL 14+
- Git

## Quick Start

### 1. Clone and Install Dependencies

```bash
cd alphalabs

# Backend dependencies
cd python
pip install -r requirements.txt
cd ..

# Frontend dependencies
cd frontend
npm install
cd ..
```

### 2. Configure Environment

Copy the example environment file and fill in your API keys:

```bash
cp env.example.txt .env
```

Edit `.env` and add your keys:

```env
# Database
DATABASE_URL=postgresql://localhost:5432/trading_lab

# Alpaca API (Paper Trading) - Get from https://alpaca.markets
ALPACA_API_KEY=your_alpaca_api_key
ALPACA_SECRET_KEY=your_alpaca_secret_key
ALPACA_PAPER=true

# LLM API Keys
OPENAI_API_KEY=your_openai_key      # Get from https://platform.openai.com
ANTHROPIC_API_KEY=your_anthropic_key # Get from https://console.anthropic.com
GOOGLE_API_KEY=your_google_key       # Get from https://makersuite.google.com

# ChromaDB
CHROMA_PERSIST_DIRECTORY=./chroma_data

# App Settings
DEBUG=true
INITIAL_CAPITAL=100000
```

### 3. Set Up PostgreSQL Database

```bash
# Create database
createdb trading_lab

# Or using psql
psql -c "CREATE DATABASE trading_lab;"
```

### 4. Initialize Database

This creates tables and seeds initial manager data:

```bash
cd python
python scripts/init_database.py
```

You should see:
```
✓ Tables created
✓ Created 4 managers
✓ Created 4 portfolios with $100,000.00 each
✓ Created 4 initial snapshots
```

### 5. Generate Historical Embeddings (One-Time Setup)

This fetches 10 years of market data and generates embeddings for semantic search:

```bash
python scripts/generate_embeddings.py
```

This will take 5-10 minutes. You'll see:
```
✓ Fetched data for 20 symbols
✓ Generated 2,500+ market state embeddings
✓ Stored in ChromaDB
```

### 6. Start Backend API

```bash
cd python
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Backend will be available at: http://localhost:8000

Test it: http://localhost:8000/health

### 7. Start Frontend

In a new terminal:

```bash
cd frontend
npm run dev
```

Frontend will be available at: http://localhost:3000

## Running a Trading Cycle

### Option A: Manual Trigger (Recommended for Testing)

```bash
# From python directory
python -c "import asyncio; from core.execution.scheduler import TradingScheduler; asyncio.run(TradingScheduler().run_once())"
```

Or via API:

```bash
curl -X POST http://localhost:8000/api/trading/cycle
```

### Option B: Continuous Mode

```bash
python core/execution/scheduler.py
```

This will run trading cycles automatically during market hours.

## Project Structure

```
alphalabs/
├── frontend/              # Next.js dashboard
│   ├── src/
│   │   ├── app/          # Pages and API routes
│   │   ├── components/   # React components
│   │   ├── lib/          # API client
│   │   └── types/        # TypeScript types
│   └── package.json
│
├── python/               # Python backend
│   ├── app/             # FastAPI application
│   │   ├── main.py      # Main API endpoints
│   │   ├── config.py    # Settings
│   │   └── schemas.py   # Pydantic models
│   ├── core/
│   │   ├── managers/    # Portfolio managers (LLM + Quant)
│   │   ├── strategies/  # Strategy toolbox
│   │   ├── semantic/    # Semantic search engine
│   │   └── execution/   # Trading engine & scheduler
│   ├── data/            # Data ingestion
│   ├── db/              # Database models
│   ├── scripts/         # Setup scripts
│   └── requirements.txt
│
└── docker/              # Docker configs (optional)
```

## The 4 Portfolio Managers

### 1. GPT-4 Fund (LLM)
- Provider: OpenAI
- Full autonomy to interpret signals and make decisions
- Can reason about market conditions

### 2. Claude Fund (LLM)
- Provider: Anthropic
- Deep reasoning and careful analysis
- Conservative approach to uncertainty

### 3. Gemini Fund (LLM)
- Provider: Google
- Fast decision making
- Balanced risk approach

### 4. Quant Bot (Baseline)
- No LLM, pure algorithms
- Fixed rules: 30% momentum, 20% mean reversion, 20% ML, 30% semantic
- **Answers the question: "Do LLMs add value?"**

## Viewing Results

### Dashboard (http://localhost:3000)
- Leaderboard ranked by Sharpe ratio
- Portfolio values and positions
- Recent trades with reasoning
- Current strategy signals

### API Endpoints

```bash
# Leaderboard
curl http://localhost:8000/api/leaderboard

# Strategy signals
curl http://localhost:8000/api/signals

# Recent trades
curl http://localhost:8000/api/trades

# Manager details
curl http://localhost:8000/api/managers

# Trigger trading cycle
curl -X POST http://localhost:8000/api/trading/cycle
```

## Troubleshooting

### Database Connection Error

```bash
# Check PostgreSQL is running
pg_isready

# Verify database exists
psql -l | grep trading_lab

# Recreate if needed
dropdb trading_lab && createdb trading_lab
python scripts/init_database.py
```

### ChromaDB Empty

If semantic search fails:

```bash
cd python
python scripts/generate_embeddings.py
```

### API Keys Missing

Make sure `.env` file exists and has all required keys:

```bash
cat .env | grep API_KEY
```

### Port Already in Use

Change ports in respective configs:
- Backend: Add `--port 8001` to uvicorn command
- Frontend: Create `frontend/.env.local` with `PORT=3001`

## Development Workflow

### 1. Make Code Changes

Edit Python or TypeScript files.

### 2. Test Trading Cycle

```bash
# Trigger one cycle
curl -X POST http://localhost:8000/api/trading/cycle
```

### 3. View Results

Check dashboard at http://localhost:3000

### 4. Check Logs

Backend terminal shows:
- Signal calculations
- Manager decisions
- Trade executions
- Performance updates

## Production Deployment

### Option A: Docker Compose

```bash
cd docker
docker-compose up -d
```

### Option B: Manual

**Backend (Railway/Render):**
```bash
cd python
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port $PORT
```

**Frontend (Vercel):**
```bash
cd frontend
vercel deploy
```

## Key Questions Answered

### Do LLMs add value over systematic strategies?

Compare the Sharpe ratios:
- If **Quant Bot** ranks highest → LLMs don't add value
- If **LLM managers** rank highest → AI reasoning creates alpha

### How does semantic search help?

The semantic memory finds similar historical market periods and shows what happened next:
- "Current market resembles low-volatility tech rallies"
- "Historically, 72% positive outcomes over 5 days"
- "Average return: +1.8%"

This gives managers (both LLM and Quant) forward-looking context.

## Next Steps

1. Run for 30 days of paper trading
2. Analyze which manager performs best
3. Iterate on losing strategies
4. Adjust risk parameters
5. Consider live trading (at your own risk!)

## License

MIT

## Support

For issues, check:
- Backend logs (terminal running uvicorn)
- Frontend console (browser DevTools)
- Database queries (psql)

---

**Built to answer:** *"Do LLMs actually add value over systematic quant strategies?"*
