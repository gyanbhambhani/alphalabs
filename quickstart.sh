#!/bin/bash

# AI Trading Lab - Quick Start Script

set -e

echo "=================================="
echo "AI Trading Lab - Quick Start"
echo "=================================="
echo

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Check if .env exists
if [ ! -f .env ]; then
    echo -e "${YELLOW}⚠️  .env file not found${NC}"
    echo "Creating from env.example.txt..."
    cp env.example.txt .env
    echo -e "${YELLOW}Please edit .env and add your API keys before continuing${NC}"
    exit 1
fi

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check prerequisites
echo "Checking prerequisites..."

if ! command_exists python3; then
    echo -e "${RED}✗ Python 3 not found${NC}"
    exit 1
fi

if ! command_exists node; then
    echo -e "${RED}✗ Node.js not found${NC}"
    exit 1
fi

if ! command_exists psql; then
    echo -e "${RED}✗ PostgreSQL not found${NC}"
    exit 1
fi

echo -e "${GREEN}✓ All prerequisites found${NC}"
echo

# Install Python dependencies
echo "Installing Python dependencies..."
cd python
pip install -q -r requirements.txt
cd ..
echo -e "${GREEN}✓ Python dependencies installed${NC}"
echo

# Install Node dependencies
echo "Installing Node dependencies..."
cd frontend
npm install --silent
cd ..
echo -e "${GREEN}✓ Node dependencies installed${NC}"
echo

# Check if database exists
DB_NAME="trading_lab"
if psql -lqt | cut -d \| -f 1 | grep -qw $DB_NAME; then
    echo -e "${YELLOW}⚠️  Database '$DB_NAME' already exists${NC}"
    read -p "Recreate database? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Dropping and recreating database..."
        dropdb $DB_NAME || true
        createdb $DB_NAME
        echo -e "${GREEN}✓ Database recreated${NC}"
        
        # Initialize database
        echo
        echo "Initializing database..."
        cd python
        python scripts/init_database.py <<< "y"
        cd ..
        echo -e "${GREEN}✓ Database initialized${NC}"
    fi
else
    echo "Creating database..."
    createdb $DB_NAME
    echo -e "${GREEN}✓ Database created${NC}"
    
    # Initialize database
    echo
    echo "Initializing database..."
    cd python
    python scripts/init_database.py <<< "y"
    cd ..
    echo -e "${GREEN}✓ Database initialized${NC}"
fi

# Check if embeddings exist
if [ ! -d "python/chroma_data" ]; then
    echo
    echo -e "${YELLOW}⚠️  Historical embeddings not found${NC}"
    read -p "Generate embeddings now? (recommended, takes 5-10 min) (Y/n): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Nn]$ ]]; then
        echo "Generating historical embeddings..."
        cd python
        python scripts/generate_embeddings.py <<< "y"
        cd ..
        echo -e "${GREEN}✓ Embeddings generated${NC}"
    fi
fi

echo
echo "=================================="
echo -e "${GREEN}✓ Setup Complete!${NC}"
echo "=================================="
echo
echo "To start the application:"
echo
echo "1. Start backend (in one terminal):"
echo "   cd python && uvicorn app.main:app --reload"
echo
echo "2. Start frontend (in another terminal):"
echo "   cd frontend && npm run dev"
echo
echo "3. Open http://localhost:3000"
echo
echo "To run a trading cycle:"
echo "   curl -X POST http://localhost:8000/api/trading/cycle"
echo
