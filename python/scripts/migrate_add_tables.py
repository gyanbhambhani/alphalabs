"""
Database Migration - Add New Tables

Adds:
- backtest_decision_candidates
- experience_records

Run this after updating the codebase to add the new tables.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from db.database import engine
from db.models import Base

def run_migration():
    """Create all tables (only creates new ones, doesn't modify existing)."""
    print("=" * 60)
    print("Running Database Migration")
    print("=" * 60)
    print()
    
    print("Creating new tables:")
    print("  - backtest_decision_candidates")
    print("  - experience_records")
    print()
    
    try:
        Base.metadata.create_all(engine)
        print("✓ Migration complete!")
        print()
        print("New tables added successfully.")
        print("Existing tables unchanged.")
        
    except Exception as e:
        print(f"✗ Migration failed: {e}")
        sys.exit(1)
    
    print("=" * 60)


if __name__ == "__main__":
    run_migration()
