# Quick Start Guide

## Start the Application

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

cd /Users/gyanb/Desktop/alphalabs/python && python3.11 scripts/populate_stocks_table.py

## Access Points

| URL | Description |
|-----|-------------|
| http://localhost:3000 | Main Dashboard |
| http://localhost:3000/lab | Trading Lab (AI Chat + Market Intelligence) |
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
- PostgreSQL database
- ChromaDB data in `python/chroma_data/`
