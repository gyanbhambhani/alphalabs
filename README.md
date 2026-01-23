# AI Trading Lab

> Do LLMs add value over systematic quant strategies?

An AI Trading Lab where 4 portfolio managers (3 LLMs + 1 Quant Bot baseline) compete using a shared strategy toolbox, with **Semantic Market Memory** as the core differentiator. Ranked by Sharpe ratio.

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          AI TRADING LAB                                 │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐         │
│  │    GPT-4        │  │    Claude       │  │    Gemini       │         │
│  │    Fund         │  │    Fund         │  │    Fund         │         │
│  │  LLM + Tools    │  │  LLM + Tools    │  │  LLM + Tools    │         │
│  │  Full Autonomy  │  │  Full Autonomy  │  │  Full Autonomy  │         │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘         │
│                                                                         │
│  ┌─────────────────┐                                                    │
│  │   Quant Bot     │  ← BASELINE (no LLM, pure algorithms)             │
│  │   (Systematic)  │                                                    │
│  │   Rule-based    │    "Do LLMs actually add value?"                  │
│  └─────────────────┘                                                    │
│                                                                         │
│  ════════════════════════════════════════════════════════════════════  │
│                     SHARED STRATEGY TOOLBOX                            │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐     │
│  │ Momentum │ │Mean Rev. │ │ Semantic │ │Technical │ │ ML Pred. │     │
│  │ Signals  │ │ Signals  │ │ Memory   │ │Indicators│ │ (LSTM)   │     │
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘ └──────────┘     │
└─────────────────────────────────────────────────────────────────────────┘
```

## Features

- **4 Portfolio Managers**: GPT-4, Claude, Gemini (LLMs) + Quant Bot (baseline)
- **Shared Strategy Toolbox**: Momentum, mean reversion, technical indicators, ML predictions
- **Semantic Market Memory**: Vector embeddings of market states for pattern matching
- **Risk Management**: Per-manager limits + global circuit breaker
- **Sharpe Ratio Ranking**: Risk-adjusted performance comparison
- **Paper Trading**: Alpaca integration for realistic simulation

## The Key Question

**If Quant Bot beats LLMs** → LLM reasoning doesn't add value over systematic rules  
**If LLMs beat Quant Bot** → AI reasoning creates alpha

## Tech Stack

### Frontend
- Next.js 14 (App Router)
- TypeScript
- Tailwind CSS + shadcn/ui
- Lightweight Charts

### Backend
- FastAPI (Python)
- PostgreSQL + ChromaDB
- Alpaca API (paper trading)

### LLM Providers
- OpenAI (GPT-4)
- Anthropic (Claude)
- Google (Gemini)

## Quick Start

### 1. Clone and Setup

```bash
cd qf

# Frontend
cd frontend
npm install

# Backend
cd ../python
pip install -r requirements.txt
```

### 2. Configure Environment

Copy `env.example.txt` to `.env` and fill in your API keys:

```bash
# Required
ALPACA_API_KEY=your_key
ALPACA_SECRET_KEY=your_secret
OPENAI_API_KEY=your_key
ANTHROPIC_API_KEY=your_key
GOOGLE_API_KEY=your_key

# Database
DATABASE_URL=postgresql://localhost:5432/trading_lab
```

### 3. Start Services

**Option A: Docker Compose**
```bash
cd docker
docker-compose up
```

**Option B: Manual**
```bash
# Terminal 1: Frontend
cd frontend
npm run dev

# Terminal 2: Backend
cd python
uvicorn app.main:app --reload
```

### 4. Access Dashboard

Open http://localhost:3000

## Project Structure

```
qf/
├── frontend/              # Next.js dashboard
│   ├── app/              # Pages and API routes
│   ├── components/       # React components
│   └── lib/              # Utilities
│
├── python/               # Python backend
│   ├── app/             # FastAPI application
│   ├── core/
│   │   ├── managers/    # LLM and Quant managers
│   │   ├── strategies/  # Strategy toolbox
│   │   ├── semantic/    # Vector search engine
│   │   └── execution/   # Trading engine
│   ├── data/            # Data ingestion
│   └── db/              # Database models
│
└── docker/              # Docker configuration
```

## Strategy Toolbox

### Momentum Signals
12-month returns with 1-month skip. Score -1 to +1.

### Mean Reversion
Bollinger Band Z-score. Positive = oversold (buy), Negative = overbought (sell).

### Technical Indicators
- RSI (14-day)
- MACD (12, 26, 9)
- Moving Averages (20, 50, 200)
- ATR (volatility)

### Semantic Market Memory
Vector embeddings of market states. Find similar historical periods and analyze outcomes.

```python
# Example: Find similar conditions
result = search_engine.search(current_close_prices)
print(f"Similar periods: {len(result.similar_periods)}")
print(f"Avg 5-day return: {result.avg_5d_return:.2%}")
print(f"Positive rate: {result.positive_5d_rate:.0%}")
```

## API Endpoints

- `GET /api/managers` - List all managers
- `GET /api/leaderboard` - Sharpe-ranked leaderboard
- `GET /api/signals` - Current strategy signals
- `GET /api/portfolios/{id}` - Manager portfolio
- `GET /api/trades` - Trade history
- `POST /api/trading/cycle` - Trigger trading cycle

## Deployment

### Frontend (Vercel)
```bash
cd frontend
vercel
```

### Backend (Railway)
```bash
cd python
railway up
```

## Cost Estimate

| Service | Monthly Cost |
|---------|--------------|
| Vercel (Hobby) | $0 |
| Railway (Free) | $0 |
| Supabase (Free) | $0 |
| LLM APIs | ~$70 |
| **Total** | **~$70/month** |

## License

MIT

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

---

Built with the question: *"Do LLMs actually add value over systematic quant strategies?"*
