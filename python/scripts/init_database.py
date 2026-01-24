"""
Database Initialization Script

Creates database tables and seeds initial data:
- 4 portfolio managers (GPT-4, Claude, Gemini, Quant Bot)
- Initial portfolios with starting capital
- Initial daily snapshots
"""
import sys
import asyncio
from pathlib import Path
from datetime import date, datetime
from decimal import Decimal

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from db.database import (
    sync_engine, async_engine, Base, get_async_session
)
from db.models import Manager, Portfolio, DailySnapshot
from app.config import get_settings


async def create_tables():
    """Create all database tables"""
    print("Creating database tables...")
    async with async_engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
        await conn.run_sync(Base.metadata.create_all)
    print("✓ Tables created")


async def seed_managers(db):
    """Seed initial managers"""
    print("Seeding portfolio managers...")
    
    settings = get_settings()
    initial_capital = settings.initial_capital
    
    managers = [
        Manager(
            id="gpt4",
            name="GPT-4 Fund",
            type="llm",
            provider="openai",
            description="OpenAI GPT-4 powered autonomous trading manager",
            is_active=True
        ),
        Manager(
            id="claude",
            name="Claude Fund",
            type="llm",
            provider="anthropic",
            description="Anthropic Claude powered autonomous trading manager",
            is_active=True
        ),
        Manager(
            id="gemini",
            name="Gemini Fund",
            type="llm",
            provider="google",
            description="Google Gemini powered autonomous trading manager",
            is_active=True
        ),
        Manager(
            id="quant_bot",
            name="Quant Bot",
            type="quant",
            provider=None,
            description=(
                "Pure systematic trading baseline. No LLM, fixed rules. "
                "Answers: Do LLMs add value?"
            ),
            is_active=True
        )
    ]
    
    for manager in managers:
        db.add(manager)
    
    await db.commit()
    print(f"✓ Created {len(managers)} managers")
    return managers


async def seed_portfolios(db, managers):
    """Seed initial portfolios"""
    print("Creating initial portfolios...")
    
    settings = get_settings()
    initial_capital = Decimal(str(settings.initial_capital))
    
    portfolios = []
    for manager in managers:
        portfolio = Portfolio(
            manager_id=manager.id,
            cash_balance=initial_capital,
            total_value=initial_capital,
            updated_at=datetime.utcnow()
        )
        portfolios.append(portfolio)
        db.add(portfolio)
    
    await db.commit()
    print(f"✓ Created {len(portfolios)} portfolios with "
          f"${initial_capital:,.2f} each")
    return portfolios


async def seed_snapshots(db, managers):
    """Create initial daily snapshots"""
    print("Creating initial performance snapshots...")
    
    settings = get_settings()
    initial_capital = Decimal(str(settings.initial_capital))
    today = date.today()
    
    snapshots = []
    for manager in managers:
        snapshot = DailySnapshot(
            manager_id=manager.id,
            date=today,
            portfolio_value=initial_capital,
            daily_return=Decimal('0.0'),
            cumulative_return=Decimal('0.0'),
            sharpe_ratio=Decimal('0.0'),
            volatility=Decimal('0.0'),
            max_drawdown=Decimal('0.0'),
            win_rate=Decimal('0.0'),
            total_trades=0
        )
        snapshots.append(snapshot)
        db.add(snapshot)
    
    await db.commit()
    print(f"✓ Created {len(snapshots)} initial snapshots")
    return snapshots


async def verify_data(db):
    """Verify seeded data"""
    print("\nVerifying database...")
    
    from sqlalchemy import select
    
    # Count managers
    result = await db.execute(select(Manager))
    managers = result.scalars().all()
    print(f"  Managers: {len(managers)}")
    for m in managers:
        print(f"    - {m.name} ({m.type})")
    
    # Count portfolios
    result = await db.execute(select(Portfolio))
    portfolios = result.scalars().all()
    print(f"  Portfolios: {len(portfolios)}")
    
    # Count snapshots
    result = await db.execute(select(DailySnapshot))
    snapshots = result.scalars().all()
    print(f"  Daily Snapshots: {len(snapshots)}")


async def main():
    """Main initialization flow"""
    print("=" * 60)
    print("Database Initialization")
    print("=" * 60)
    print()
    
    # Check settings
    settings = get_settings()
    print(f"Database URL: {settings.database_url}")
    print(f"Initial Capital: ${settings.initial_capital:,.2f}")
    print()
    
    # Confirm
    response = input(
        "⚠️  This will DROP and recreate all tables. Continue? (y/N): "
    )
    if response.lower() != 'y':
        print("Cancelled.")
        return
    
    print()
    
    # Create tables
    await create_tables()
    print()
    
    # Seed data
    async with get_async_session() as db:
        managers = await seed_managers(db)
        await seed_portfolios(db, managers)
        await seed_snapshots(db, managers)
        print()
        await verify_data(db)
    
    print()
    print("=" * 60)
    print("✓ Database initialization complete!")
    print("=" * 60)
    print()
    print("You can now start the trading engine.")
    print()


if __name__ == "__main__":
    asyncio.run(main())
