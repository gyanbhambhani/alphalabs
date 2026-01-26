# S&P 500 Embeddings Setup Guide

## Overview

This guide walks you through setting up the system to generate embeddings for all S&P 500 stocks with maximum historical data.

## Step-by-Step Setup

### 1. Update Database Schema

First, apply the new database migrations to add the `stocks` table:

```bash
cd python
python3.11 scripts/init_database.py
```

This will create the new `stocks` table in PostgreSQL.

### 2. Fetch S&P 500 Stock List

Download the current S&P 500 constituents from Wikipedia:

```bash
python3.11 scripts/fetch_sp500_list.py
```

This creates `python/data/sp500_list.csv` with ~500 stocks including:
- Symbol, Name
- Sector, Sub-Industry
- Headquarters, Founded date
- CIK number

### 3. Generate Embeddings for All Stocks

**WARNING: This will take several hours to complete (~2-6 hours)**

```bash
python3.11 scripts/generate_sp500_embeddings.py
```

This script will:
- Fetch maximum available historical data for each stock (up to 20+ years)
- Generate 512-dimensional embeddings for each trading day
- Store in separate ChromaDB collections (one per stock)
- Create ~200k-1M+ embeddings total

**Progress tracking:**
- Updates every 10 stocks processed
- Shows ETA and success/failure counts
- Automatically skips stocks that already have embeddings

**Memory usage:**
- Processes one stock at a time
- Each stock ~50-200 MB during processing
- Final ChromaDB size: ~5-15 GB for all stocks

### 4. Populate Stock Metadata

Sync the stocks table with ChromaDB embedding status:

```bash
python3.11 scripts/populate_stocks_table.py
```

This updates the PostgreSQL `stocks` table with:
- Which stocks have embeddings
- How many embeddings per stock
- Date range covered

### 5. Start the Services

**Backend:**
```bash
cd python
python3.11 -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

**Frontend:**
```bash
cd frontend
npm run dev
```

Navigate to `http://localhost:3000`

## New Features

### Stock Selector
- Search by symbol or company name
- Filter by sector (11 GICS sectors)
- Toggle "With Data" to show only stocks with embeddings
- Shows embedding count and date range for each stock

### Single Stock View
- View embeddings for one stock at a time
- Full table and timeline visualization
- Semantic search within that stock's history

### Multi-Stock Comparison
- Select multiple stocks (up to 6 recommended)
- Compare returns and volatility over time
- Side-by-side line charts
- Statistics table

## API Endpoints

### Stock Endpoints
- `GET /api/stocks` - List all stocks with filters
  - Query params: `sector`, `has_embeddings`, `search`
- `GET /api/stocks/{symbol}` - Get specific stock details

### Symbol-Specific Embeddings
- `GET /api/embeddings/stats/{symbol}` - Stats for one stock
- `GET /api/embeddings/{symbol}` - List embeddings with pagination
- `POST /api/embeddings/search/{symbol}` - Search within one stock

### Legacy Endpoints (SPY only)
- `GET /api/embeddings/stats` - Stats for SPY
- `GET /api/embeddings` - List SPY embeddings
- `POST /api/embeddings/search` - Search SPY embeddings

## Database Structure

### ChromaDB Collections
- `market_states` - Legacy SPY collection
- `market_states_AAPL` - Apple embeddings
- `market_states_MSFT` - Microsoft embeddings
- ... (one collection per stock)

### PostgreSQL Tables
- `stocks` - Stock metadata and embedding status
  - symbol, name, sector, sub_industry
  - has_embeddings, embeddings_count
  - embeddings_date_range_start, embeddings_date_range_end

## Performance Tips

### Initial Data Generation
- Run overnight or during off-hours
- Can be interrupted and resumed (skips existing collections)
- Consider starting with top 50 most liquid stocks first

### Quick Start (Top 50 Stocks)
If you want to test with a subset first:

```python
# Edit scripts/generate_sp500_embeddings.py
# After loading sp500_df, add:
sp500_df = sp500_df.head(50)  # Only process first 50
```

### Query Performance
- Each stock collection is independent
- Queries are fast (< 100ms per stock)
- Multi-stock comparison loads in parallel

## Troubleshooting

### "Stock not found" errors
- Run `populate_stocks_table.py` to sync database

### Missing embeddings
- Check if stock was successfully processed
- Some stocks may have insufficient history (< 252 days)
- Re-run `generate_sp500_embeddings.py` (it will skip existing)

### Frontend shows no stocks
- Ensure backend is running on port 8000
- Check that `stocks` table is populated
- Verify CORS is enabled in backend

## Next Steps

After setup is complete, you can:
1. Browse any S&P 500 stock's historical embeddings
2. Search for specific market conditions across all history
3. Compare multiple stocks side-by-side
4. Use semantic search to find similar market periods

The system now scales to handle 500+ stocks with decades of market history!
