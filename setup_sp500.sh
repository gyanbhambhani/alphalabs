#!/bin/bash

# S&P 500 Setup Quick Start Script
# Run this from the alphalabs root directory

set -e

echo "=========================================="
echo "S&P 500 Embeddings Setup"
echo "=========================================="
echo ""

cd python

echo "[1/4] Updating database schema..."
python3.11 scripts/init_database.py
echo "✓ Database schema updated"
echo ""

echo "[2/4] Fetching S&P 500 list..."
python3.11 scripts/fetch_sp500_list.py
echo "✓ S&P 500 list downloaded"
echo ""

echo "[3/4] Populating stocks metadata table..."
python3.11 scripts/populate_stocks_table.py
echo "✓ Stock metadata populated"
echo ""

echo "[4/4] Ready to generate embeddings!"
echo ""
echo "=========================================="
echo "✓ Setup Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo ""
echo "1. Generate embeddings (this takes several hours):"
echo "   cd python"
echo "   python3.11 scripts/generate_sp500_embeddings.py"
echo ""
echo "2. Start the backend:"
echo "   cd python"
echo "   python3.11 -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000"
echo ""
echo "3. Start the frontend:"
echo "   cd frontend"
echo "   npm run dev"
echo ""
echo "4. Visit http://localhost:3000"
echo ""
