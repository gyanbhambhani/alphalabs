---
name: Strategy Toolbox Implementation
overview: Complete the core strategy toolbox and semantic memory engine, then implement the portfolio managers and trading execution layer to create a functional AI trading competition platform.
todos:
  - id: semantic-encoder
    content: Complete market state encoder with 50+ features (returns, volatility, technical indicators, correlations)
    status: completed
  - id: generate-embeddings
    content: Create script to generate and store 10 years of historical market embeddings in ChromaDB
    status: completed
  - id: semantic-search
    content: Enhance semantic search engine to analyze outcomes and generate interpretations
    status: completed
  - id: quant-bot-logic
    content: Implement Quant Bot decision logic with fixed signal weighting
    status: completed
  - id: llm-managers
    content: Complete LLM manager implementations (GPT-4, Claude, Gemini) with prompt engineering
    status: completed
  - id: trading-engine
    content: Build trading engine execution flow (signals → decisions → trades → portfolio updates)
    status: completed
  - id: init-database
    content: Create database initialization script to seed managers and portfolios
    status: completed
  - id: performance-tracking
    content: Implement Sharpe ratio and performance metric calculations
    status: completed
  - id: trading-scheduler
    content: Create trading loop scheduler for daily cycles
    status: completed
  - id: connect-frontend
    content: Replace mock data in frontend with real API calls
    status: completed
isProject: false
---

# Strategy Toolbox & Trading Engine Implementation

## Overview

Based on the existing codebase analysis, you have a solid foundation with frontend UI, database schema, and basic strategy implementations. The next critical phase is to complete the **strategy toolbox**, build the **semantic market memory engine**, implement the **portfolio managers** (LLM + Quant), and connect everything through a **trading execution layer**.

## Current State

**✅ Completed:**

- Next.js dashboard with components ([`frontend/src/app/page.tsx`](frontend/src/app/page.tsx))
- Database models ([`python/db/models.py`](python/db/models.py))
- FastAPI endpoints ([`python/app/main.py`](python/app/main.py))
- Basic strategies: momentum, mean reversion, technical indicators, volatility
- Semantic search infrastructure skeleton
- Data ingestion pipeline ([`python/data/ingest.py`](python/data/ingest.py))

**❌ Missing Critical Components:**

- Complete semantic market encoder (feature engineering)
- Historical data embedding generation
- LLM manager implementations
- Quant Bot decision logic
- Trading engine execution
- Database initialization with seed data
- Trading loop scheduler

## Implementation Plan

### Phase 1: Complete Semantic Market Memory

**Goal:** Build a working semantic search engine that encodes market states and finds similar historical periods.

#### 1.1 Enhance Market State Encoder

The current [`python/core/semantic/encoder.py`](python/core/semantic/encoder.py) needs full feature engineering to capture market conditions:

- **Price features**: Returns (1d, 5d, 20d, 60d), momentum scores
- **Volatility features**: Realized vol (20d, 60d), ATR ratios, Bollinger Band width
- **Technical features**: RSI, MACD histogram, moving average relationships
- **Market structure**: Correlation matrix (avg correlation, max correlation)
- **Sentiment proxies**: VIX level, put/call ratio (if available)

Create a 50-100 dimensional feature vector that captures market regime.

#### 1.2 Generate Historical Embeddings

Create a script to:

1. Load 10 years of historical data for the universe (using existing [`python/data/ingest.py`](python/data/ingest.py))
2. For each trading day, encode the market state using the enhanced encoder
3. Store embeddings in ChromaDB with metadata (date, subsequent returns)
4. This creates ~2,500 historical states for semantic search

Path: `python/scripts/generate_embeddings.py`

#### 1.3 Complete Semantic Search Engine

Enhance [`python/core/semantic/search.py`](python/core/semantic/search.py) to:

- Query current market state
- Retrieve top-k similar periods
- Analyze outcomes (5-day, 20-day forward returns)
- Generate interpretation text for LLM context

### Phase 2: Implement Portfolio Managers

**Goal:** Create functioning portfolio managers that make trading decisions.

#### 2.1 Complete Quant Bot

In [`python/core/managers/quant_bot.py`](python/core/managers/quant_bot.py), implement systematic decision logic:

```python
def make_decisions(context):
    # Combine signals with fixed weights
    for symbol in universe:
        score = (
            0.3 * momentum[symbol] +
            0.2 * mean_reversion[symbol] +
            0.2 * ml_prediction[symbol] +
            0.3 * semantic_outcome[symbol]
        )
        
        if score > 0.6 and volatility_regime == "trending":
            decisions.append(BUY)
        elif score < -0.6:
            decisions.append(SELL)
```

This provides the **baseline** that LLMs must beat.

#### 2.2 Build LLM Managers

Complete [`python/core/managers/llm_manager.py`](python/core/managers/llm_manager.py):

1. **Build context prompt** with current portfolio, market data, all strategy signals
2. **Include semantic search results** with historical outcomes
3. **Call LLM API** (OpenAI, Anthropic, Google)
4. **Parse response** to extract trading decisions
5. **Validate decisions** before returning

Each LLM manager receives identical signals but can reason differently about them.

#### 2.3 Test Managers in Isolation

Create unit tests to verify:

- Managers can process context
- Decisions follow correct format
- Risk limits are respected
- No API errors

### Phase 3: Build Trading Execution Layer

**Goal:** Connect managers → decisions → actual trades → portfolio updates.

#### 3.1 Complete Trading Engine

Enhance [`python/core/execution/trading_engine.py`](python/core/execution/trading_engine.py):

1. **Gather signals** from all strategies
2. **Query each manager** for decisions
3. **Apply risk checks** via [`python/core/execution/risk_manager.py`](python/core/execution/risk_manager.py)
4. **Execute orders** (paper trading via Alpaca or simulation)
5. **Update portfolios** in database
6. **Log trades** with reasoning

#### 3.2 Integrate Alpaca API

Complete [`python/core/execution/alpaca_client.py`](python/core/execution/alpaca_client.py):

- Connect to Alpaca paper trading API
- Submit market orders for approved decisions
- Track order status
- Handle partial fills and errors

Alternative: Build a simple simulator if Alpaca setup is complex.

#### 3.3 Portfolio & Performance Tracking

After each trading cycle:

- Update [`Portfolio`](python/db/models.py) cash and positions
- Calculate unrealized P&L for open positions
- Create [`DailySnapshot`](python/db/models.py) with Sharpe ratio, returns, volatility
- Store [`Trade`](python/db/models.py) records with reasoning

### Phase 4: Database Initialization & Seeding

**Goal:** Populate database with initial manager records.

#### 4.1 Create Initialization Script

`python/scripts/init_database.py`:

1. Create database tables (using SQLAlchemy models)
2. Seed 4 managers: GPT-4, Claude, Gemini, Quant Bot
3. Initialize portfolios with $25,000 each
4. Create initial daily snapshots

#### 4.2 Database Migrations

Set up Alembic migrations for schema changes (already included in requirements.txt).

### Phase 5: Trading Loop & Scheduler

**Goal:** Automated daily trading cycles.

#### 5.1 Create Trading Loop

`python/core/execution/scheduler.py` or enhance existing:

```python
async def run_trading_cycle():
    # 1. Fetch latest market data
    # 2. Calculate all strategy signals
    # 3. Query semantic search
    # 4. Get decisions from all managers
    # 5. Execute trades via trading engine
    # 6. Update database
    # 7. Calculate performance metrics
```

#### 5.2 Scheduler Options

- **Manual trigger**: via API endpoint `/api/trading/cycle` (already exists)
- **Cron job**: for daily execution
- **Event-driven**: on market open/close

Start with manual triggers for testing, add automation later.

### Phase 6: Connect Backend to Frontend

**Goal:** Replace mock data with real API calls.

#### 6.1 Update Frontend API Calls

In [`frontend/src/app/page.tsx`](frontend/src/app/page.tsx):

- Replace `MOCK_LEADERBOARD` with `fetch('/api/leaderboard')`
- Replace `MOCK_SIGNALS` with `fetch('/api/signals')`
- Replace `MOCK_TRADES` with `fetch('/api/trades')`

#### 6.2 Fix API Response Models

Ensure FastAPI response schemas ([`python/app/schemas.py`](python/app/schemas.py)) match TypeScript types ([`frontend/src/types/index.ts`](frontend/src/types/index.ts)).

#### 6.3 Real-time Updates

Add polling or WebSocket for live leaderboard updates (optional enhancement).

## Implementation Priority

### High Priority (Week 1)

1. Complete semantic market encoder with full features
2. Generate historical embeddings (run overnight)
3. Implement Quant Bot decision logic
4. Build LLM manager prompt + API integration
5. Database initialization script

### Medium Priority (Week 2)

1. Complete trading engine execution flow
2. Integrate Alpaca API or build simulator
3. Performance calculation (Sharpe ratio)
4. Trading loop scheduler
5. Connect frontend to backend APIs

### Nice-to-Have (Later)

1. ML prediction model (LSTM)
2. Advanced risk management rules
3. Real-time WebSocket updates
4. Backtesting framework
5. Strategy optimization tools

## Key Technical Decisions

### Semantic Encoder Architecture

**Recommended approach:** Custom feature engineering + simple MLP

- Extract 50-100 quantitative features from market data
- Normalize/standardize features
- Use a simple feedforward network or even just concatenate normalized features
- This is simpler and more interpretable than transformers

### LLM Manager Prompt Structure

```
You are [Manager Name], an autonomous portfolio manager.

## Current Portfolio
Cash: $X, Positions: {...}

## Strategy Signals
Momentum: NVDA +0.85, MSFT +0.62
Mean Reversion: TSLA +0.72
Volatility Regime: low_vol_trending_up

## Semantic Search Results
Current market resembles [similar periods]. 
Historical outcomes: +1.8% (5d), 72% positive rate.

## Task
Decide trades. Format:
BUY NVDA 0.10 (10% of portfolio)
Reasoning: Strong momentum + favorable historical pattern
```

### Paper Trading vs Simulation

Start with **simple simulation** (just update database positions) to test the system end-to-end. Add Alpaca integration later for realism.

## Success Metrics

**Phase 1 Complete:**

- ChromaDB contains 2,500+ historical market states
- Semantic search returns relevant similar periods
- Can query "current market" and get interpretations

**Phase 2 Complete:**

- All 4 managers can generate decisions from context
- Quant Bot uses fixed rules
- LLM managers call APIs successfully

**Phase 3 Complete:**

- Trading cycle executes without errors
- Portfolios update correctly after trades
- Database shows trade history with reasoning

**Phase 4 Complete:**

- Dashboard shows real leaderboard data
- Sharpe ratios calculated correctly
- Can see differentiation between manager styles

## Next Steps After This Plan

Once the core system is working:

1. **Paper trading**: Run for 30 days, track real performance
2. **Analysis**: Did LLMs beat Quant Bot?
3. **Iteration**: Improve losing strategies, adjust risk limits
4. **Deployment**: Deploy to Vercel + Railway for live operation