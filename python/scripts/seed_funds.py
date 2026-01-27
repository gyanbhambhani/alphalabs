"""
Seed Initial Collaborative Funds

Creates the 4 thesis-driven funds:
- Trend + Macro Fund
- Mean Reversion Fund
- Event-Driven Fund
- Quality L/S Fund
"""
import sys
import asyncio
from pathlib import Path
from datetime import datetime
from decimal import Decimal
import json

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from db.database import get_async_session
from db.models import FundModel


# Fund definitions based on the architecture
FUNDS = [
    {
        "id": "trend_macro_fund",
        "name": "Trend + Macro",
        "strategy": "trend_macro",
        "description": (
            "Regime detection + trend following + vol targeting. "
            "Focuses on liquid ETFs and index futures with 1-20 day horizon."
        ),
        "thesis_json": {
            "name": "Trend + Macro",
            "strategy": "trend_macro",
            "description": "Regime detection, trend following, vol targeting",
            "horizon_days": [1, 20],
            "edge": "Regime detection + trend following + vol targeting",
            "universe_spec": {
                "type": "etf_set",
                "params": {"name": "liquid_macro"}
            },
            "version": "1.0"
        },
        "policy_json": {
            "sizing_method": "vol_target",
            "vol_target": 0.15,
            "max_position_pct": 0.20,
            "max_turnover_daily": 0.30,
            "rebalance_cadence": "daily",
            "max_positions": 15,
            "default_stop_loss_pct": 0.05,
            "default_take_profit_pct": 0.15,
            "trailing_stop": True,
            "max_gross_exposure": 1.0,
            "min_cash_buffer": 0.05,
            "go_flat_on_circuit_breaker": True,
            "version": "1.0"
        },
        "risk_limits_json": {
            "max_position_pct": 0.20,
            "max_sector_pct": 0.40,
            "max_gross_exposure": 1.0,
            "max_daily_loss_pct": 0.03,
            "max_weekly_drawdown_pct": 0.07,
            "breach_action": "halt",
            "breach_cooldown_days": 1
        },
        "cash_balance": 100000,
        "total_value": 100000,
    },
    {
        "id": "mean_reversion_fund",
        "name": "Mean Reversion",
        "strategy": "mean_reversion",
        "description": (
            "Exploit overreactions in liquid names. "
            "Microstructure signals, intraday to 3 day horizon."
        ),
        "thesis_json": {
            "name": "Mean Reversion",
            "strategy": "mean_reversion",
            "description": "Exploit overreactions, volatility spikes",
            "horizon_days": [0, 3],
            "edge": "Microstructure signals, overreaction, volatility spikes",
            "universe_spec": {
                "type": "screen",
                "params": {
                    "min_adv": 10000000,
                    "market_cap": "large",
                    "options_liquid": True
                }
            },
            "version": "1.0"
        },
        "policy_json": {
            "sizing_method": "equal_risk",
            "vol_target": 0.20,
            "max_position_pct": 0.10,
            "max_turnover_daily": 0.50,
            "rebalance_cadence": "intraday",
            "max_positions": 20,
            "default_stop_loss_pct": 0.03,
            "default_take_profit_pct": 0.05,
            "trailing_stop": False,
            "max_gross_exposure": 1.2,
            "min_cash_buffer": 0.10,
            "go_flat_on_circuit_breaker": True,
            "version": "1.0"
        },
        "risk_limits_json": {
            "max_position_pct": 0.10,
            "max_sector_pct": 0.30,
            "max_gross_exposure": 1.2,
            "max_daily_loss_pct": 0.02,
            "max_weekly_drawdown_pct": 0.05,
            "breach_action": "reduce",
            "breach_cooldown_days": 1
        },
        "cash_balance": 100000,
        "total_value": 100000,
    },
    {
        "id": "event_driven_fund",
        "name": "Event-Driven",
        "strategy": "event_driven",
        "description": (
            "Earnings plays and event catalysts. "
            "-3 to +5 days around events, focus on guidance surprise."
        ),
        "thesis_json": {
            "name": "Event-Driven",
            "strategy": "event_driven",
            "description": "Earnings plays, guidance surprise, implied vs realized vol",
            "horizon_days": [-3, 5],
            "edge": "Guidance surprise, transcript sentiment, implied vs realized vol",
            "universe_spec": {
                "type": "screen",
                "params": {
                    "upcoming_earnings": True,
                    "options_liquid": True,
                    "min_adv": 5000000
                }
            },
            "version": "1.0"
        },
        "policy_json": {
            "sizing_method": "event_based",
            "vol_target": 0.25,
            "max_position_pct": 0.08,
            "max_turnover_daily": 0.40,
            "rebalance_cadence": "event_driven",
            "max_positions": 10,
            "default_stop_loss_pct": 0.08,
            "default_take_profit_pct": 0.12,
            "trailing_stop": False,
            "max_gross_exposure": 0.8,
            "min_cash_buffer": 0.20,
            "go_flat_on_circuit_breaker": True,
            "version": "1.0"
        },
        "risk_limits_json": {
            "max_position_pct": 0.08,
            "max_sector_pct": 0.25,
            "max_gross_exposure": 0.8,
            "max_daily_loss_pct": 0.04,
            "max_weekly_drawdown_pct": 0.08,
            "breach_action": "halt",
            "breach_cooldown_days": 2
        },
        "cash_balance": 100000,
        "total_value": 100000,
    },
    {
        "id": "quality_ls_fund",
        "name": "Quality L/S",
        "strategy": "quality_ls",
        "description": (
            "Fundamental long-short strategies. "
            "Weeks to months horizon, slow-moving fundamentals + quality factors."
        ),
        "thesis_json": {
            "name": "Quality Long-Short",
            "strategy": "quality_ls",
            "description": "Slow-moving fundamentals, valuation, quality factors",
            "horizon_days": [20, 90],
            "edge": "Fundamental analysis, quality factors, value spread",
            "universe_spec": {
                "type": "screen",
                "params": {
                    "market_cap": "large",
                    "min_adv": 20000000
                }
            },
            "version": "1.0"
        },
        "policy_json": {
            "sizing_method": "fundamental_weight",
            "vol_target": 0.12,
            "max_position_pct": 0.10,
            "max_turnover_daily": 0.10,
            "rebalance_cadence": "weekly",
            "max_positions": 30,
            "default_stop_loss_pct": 0.10,
            "default_take_profit_pct": 0.20,
            "trailing_stop": True,
            "max_gross_exposure": 1.6,
            "min_cash_buffer": 0.05,
            "go_flat_on_circuit_breaker": True,
            "version": "1.0"
        },
        "risk_limits_json": {
            "max_position_pct": 0.10,
            "max_sector_pct": 0.25,
            "max_gross_exposure": 1.6,
            "max_daily_loss_pct": 0.02,
            "max_weekly_drawdown_pct": 0.05,
            "breach_action": "reduce",
            "breach_cooldown_days": 3
        },
        "cash_balance": 100000,
        "total_value": 100000,
    },
]


async def seed_funds():
    """Seed the collaborative funds into the database."""
    print("=" * 60)
    print("Seeding Collaborative AI Funds")
    print("=" * 60)
    print()
    
    async with get_async_session() as db:
        for fund_data in FUNDS:
            # Check if fund already exists
            from sqlalchemy import select
            result = await db.execute(
                select(FundModel).where(FundModel.id == fund_data["id"])
            )
            existing = result.scalar_one_or_none()
            
            if existing:
                print(f"  ⚠️  {fund_data['name']} already exists, skipping")
                continue
            
            fund = FundModel(
                id=fund_data["id"],
                name=fund_data["name"],
                strategy=fund_data["strategy"],
                description=fund_data["description"],
                thesis_json=fund_data["thesis_json"],
                policy_json=fund_data["policy_json"],
                risk_limits_json=fund_data["risk_limits_json"],
                cash_balance=Decimal(str(fund_data["cash_balance"])),
                total_value=Decimal(str(fund_data["total_value"])),
                is_active=True,
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow()
            )
            db.add(fund)
            print(f"  ✓ Created {fund_data['name']} ({fund_data['strategy']})")
        
        await db.commit()
    
    print()
    print("=" * 60)
    print("✓ Fund seeding complete!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(seed_funds())
