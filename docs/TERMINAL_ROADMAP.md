# Open-Source Bloomberg Terminal - Roadmap

> Real-time market data + LLM-powered research. Fully free for students.
> Powered by your own API keys.

## Vision

Extend AI Trading Lab into an **open-source Bloomberg-style terminal** where:

- **Real-time data**: Live prices, indices, FX (free tier: Yahoo Finance ~15min delayed)
- **LLM-powered**: Ask anything in natural language - "What's AAPL doing?", "Compare NVDA to AMD"
- **BYOK (Bring Your Own Key)**: Users provide their own OpenAI/Anthropic keys - no server costs
- **Student-focused**: Educational, accessible, free to run locally

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    OPEN-SOURCE BLOOMBERG TERMINAL                       │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐        │
│  │  Ticker Tape    │  │  LLM Chat       │  │  Market Panels   │        │
│  │  (Live Prices)  │  │  (BYOK)         │  │  (Context)      │        │
│  │  SPY AAPL NVDA  │  │  "Compare X/Y"  │  │  VIX, Sentiment │        │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘        │
│                                                                         │
│  ═══════════════════════════════════════════════════════════════════   │
│                         DATA LAYER (Free)                              │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐ ┌──────────────┐   │
│  │ Yahoo Finance│ │ Alpha Vantage│ │ FRED         │ │ SEC EDGAR    │   │
│  │ (prices)     │ │ (optional)   │ │ (rates)      │ │ (filings)    │   │
│  └──────────────┘ └──────────────┘ └──────────────┘ └──────────────┘   │
└─────────────────────────────────────────────────────────────────────────┘
```

## Phase 1: Core Terminal (Current)

- [x] BYOK support - `X-OpenAI-API-Key` header for user keys
- [x] Real-time data endpoint - `GET /api/terminal/live?symbols=AAPL,MSFT`
- [x] Terminal page - Bloomberg-style dark UI with ticker + chat
- [x] Key setup modal - Store key in localStorage (never sent to server except per-request)

## Phase 2: Enhanced Data

- [ ] Streaming ticker - WebSocket or SSE for live price updates
- [ ] Indices panel - SPY, QQQ, VIX, sector ETFs
- [ ] FX & commodities - Major pairs, gold, oil (via yfinance)
- [ ] News feed - RSS or free news API

## Phase 3: LLM Tools

- [ ] Multi-provider - OpenAI, Anthropic, Gemini (user picks + provides key)
- [ ] Chart generation - "Show me AAPL 1Y chart" → Lightweight Charts
- [ ] Export - "Export this analysis as PDF" for student reports
- [ ] Citation mode - Show data sources for educational transparency

## Phase 4: Student Features

- [ ] Tutorial mode - Guided walkthrough of terminal features
- [ ] Glossary - Hover definitions for financial terms
- [ ] Assignment templates - "Compare 3 tech stocks" preset
- [ ] Local-first - Optional SQLite for offline history

## Data Sources (Free Tier)

| Source        | Data              | Rate Limit   | Delay    |
|---------------|-------------------|--------------|----------|
| Yahoo Finance | Prices, volume    | Generous     | ~15 min  |
| Alpha Vantage | Prices, FX        | 25/day free  | Real-time|
| FRED          | Rates, macro      | Unlimited    | Daily    |
| SEC EDGAR     | Filings           | 10 req/sec   | Real-time|

## API: BYOK Flow

```
Frontend                          Backend
   │                                 │
   │  POST /api/search/session       │
   │  Headers: X-OpenAI-API-Key: sk-xxx
   │  Body: { query, symbols }       │
   │ ──────────────────────────────>│
   │                                 │  Use user key for this request
   │                                 │  (never stored)
   │  <──────────────────────────────│
   │  200 { sessionId, cached }      │
```

Keys are passed per-request. Server never persists them.

## File Structure

```
python/app/
  terminal_routes.py    # Existing - add BYOK, live endpoint
  ...

frontend/src/
  app/terminal/         # NEW: Full terminal page
    page.tsx
  components/
    TerminalTicker.tsx  # Scrolling ticker tape
    TerminalChat.tsx    # BYOK chat panel
    KeySetupModal.tsx   # API key config
```

## License

MIT - Fully open source. Students can fork, self-host, learn.
