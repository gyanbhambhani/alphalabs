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


# Fund definitions with MECHANICAL rules (no narrative vibes)
FUNDS = [
    {
        "id": "momentum_fund",
        "name": "Momentum Cross-Sectional",
        "strategy": "momentum",
        "description": (
            "12-1 cross-sectional momentum with vol targeting. "
            "Entry: Top 15% momentum rank. Exit: Falls below 50% rank. "
            "Rebalance: Monthly."
        ),
        "thesis_json": {
            "name": "Momentum Cross-Sectional",
            "strategy": "momentum",
            "description": "12-month return skipping last month, cross-sectional rank",
            "horizon_days": [20, 60],
            "edge": "Cross-sectional momentum persistence",
            "rules": {
                "signal": "12-month return skipping last month (252d - 21d)",
                "entry": "Top 15% momentum rank",
                "exit": "Falls below 50% rank",
                "sizing": "Inverse vol, cap 15%",
                "rebalance": "Monthly (20 trading days)"
            },
            "factors": ["return_252d", "return_21d", "momentum_rank_pct", "volatility_21d"],
            "universe_spec": {
                "type": "screen",
                "params": {
                    "min_adv": 10000000,
                    "market_cap": "large"
                }
            },
            "version": "1.0"
        },
        "policy_json": {
            "sizing_method": "vol_target",
            "vol_target": 0.15,
            "max_position_pct": 0.15,
            "max_turnover_daily": 0.20,
            "rebalance_cadence": "monthly",
            "max_positions": 15,
            "default_stop_loss_pct": 0.10,
            "default_take_profit_pct": 0.25,
            "trailing_stop": False,
            "max_gross_exposure": 1.0,
            "min_cash_buffer": 0.05,
            "go_flat_on_circuit_breaker": True,
            "version": "1.0"
        },
        "risk_limits_json": {
            "max_position_pct": 0.15,
            "max_sector_pct": 0.35,
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
        "name": "Mean Reversion Short-Horizon",
        "strategy": "mean_reversion",
        "description": (
            "Oversold + extreme + vol filter. "
            "Entry: z_score < -2.0 AND rsi < 30 AND vol_spike < 2.5x. "
            "Exit: z_score > -0.5 OR holding >= 10 days. "
            "Rebalance: Daily."
        ),
        "thesis_json": {
            "name": "Mean Reversion Short-Horizon",
            "strategy": "mean_reversion",
            "description": "Trade statistical extremes with vol filter",
            "horizon_days": [1, 10],
            "edge": "Statistical extremes mean-revert short-term",
            "rules": {
                "entry": "z_score_20d < -2.0 AND rsi_14 < 30 AND vol_spike_ratio < 2.5",
                "exit": "z_score_20d > -0.5 OR holding_days >= 10",
                "sizing": "Small bets, inverse vol, cap 7%",
                "liquidity_filter": "min_adv > 10M",
                "rebalance": "Daily"
            },
            "factors": ["z_score_20d", "rsi_14", "volatility_21d", "vol_spike_ratio", "adv_20d"],
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
            "max_position_pct": 0.07,
            "max_turnover_daily": 0.50,
            "rebalance_cadence": "daily",
            "max_positions": 20,
            "default_stop_loss_pct": 0.03,
            "default_take_profit_pct": 0.05,
            "trailing_stop": False,
            "max_gross_exposure": 1.2,
            "min_cash_buffer": 0.05,
            "go_flat_on_circuit_breaker": True,
            "version": "1.0"
        },
        "risk_limits_json": {
            "max_position_pct": 0.07,
            "max_sector_pct": 0.30,
            "max_gross_exposure": 1.2,
            "max_daily_loss_pct": 0.025,
            "max_weekly_drawdown_pct": 0.06,
            "breach_action": "reduce",
            "breach_cooldown_days": 1
        },
        "cash_balance": 100000,
        "total_value": 100000,
    },
    {
        "id": "value_fund",
        "name": "Quality Value",
        "strategy": "value",
        "description": (
            "Quality value composite. "
            "Entry: Top 20% composite score (earnings yield + FCF yield + ROIC - leverage). "
            "Exit: Falls below 40% OR holding > 90 days. "
            "Rebalance: Quarterly."
        ),
        "thesis_json": {
            "name": "Quality Value",
            "strategy": "value",
            "description": "Cheap, profitable firms with stable balance sheets",
            "horizon_days": [60, 180],
            "edge": "Value without traps (profitability filter)",
            "rules": {
                "signal": "composite = z(earnings_yield) + z(fcf_yield) + z(roic) - z(leverage)",
                "entry": "Top 20% composite score",
                "exit": "Falls below 40% score OR holding > 90 days",
                "sizing": "Equal weight, cap 10%, max 25% per sector",
                "rebalance": "Quarterly"
            },
            "factors": ["earnings_yield", "fcf_yield", "roic", "leverage", "value_rank_pct"],
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
            "rebalance_cadence": "quarterly",
            "max_positions": 30,
            "default_stop_loss_pct": 0.10,
            "default_take_profit_pct": 0.20,
            "trailing_stop": True,
            "max_gross_exposure": 1.0,
            "min_cash_buffer": 0.05,
            "go_flat_on_circuit_breaker": True,
            "version": "1.0"
        },
        "risk_limits_json": {
            "max_position_pct": 0.10,
            "max_sector_pct": 0.25,
            "max_gross_exposure": 1.0,
            "max_daily_loss_pct": 0.03,
            "max_weekly_drawdown_pct": 0.08,
            "breach_action": "reduce",
            "breach_cooldown_days": 3
        },
        "cash_balance": 100000,
        "total_value": 100000,
    },
    {
        "id": "low_vol_fund",
        "name": "Defensive Low Volatility",
        "strategy": "low_vol",
        "description": (
            "Defensive low volatility. "
            "Entry: Bottom 20% vol + profitability > 0. "
            "Exit: Vol rank > 40% OR profit < 0. "
            "Rebalance: Monthly."
        ),
        "thesis_json": {
            "name": "Defensive Low Volatility",
            "strategy": "low_vol",
            "description": "Low vol outperforms in drawdowns",
            "horizon_days": [30, 120],
            "edge": "Low vol outperforms in drawdowns",
            "rules": {
                "signal": "Realized volatility rank (lowest = best)",
                "entry": "Bottom 20% vol + profitability > 0",
                "exit": "Vol rank > 40% OR profitability < 0",
                "sizing": "Equal weight, cap 10%",
                "rebalance": "Monthly"
            },
            "factors": ["volatility_63d", "vol_rank_pct", "profitability", "beta"],
            "universe_spec": {
                "type": "screen",
                "params": {
                    "market_cap": "large",
                    "min_adv": 10000000
                }
            },
            "version": "1.0"
        },
        "policy_json": {
            "sizing_method": "equal_weight",
            "vol_target": 0.10,
            "max_position_pct": 0.10,
            "max_turnover_daily": 0.15,
            "rebalance_cadence": "monthly",
            "max_positions": 25,
            "default_stop_loss_pct": 0.08,
            "default_take_profit_pct": 0.15,
            "trailing_stop": False,
            "max_gross_exposure": 1.0,
            "min_cash_buffer": 0.05,
            "go_flat_on_circuit_breaker": True,
            "version": "1.0"
        },
        "risk_limits_json": {
            "max_position_pct": 0.10,
            "max_sector_pct": 0.30,
            "max_gross_exposure": 1.0,
            "max_daily_loss_pct": 0.025,
            "max_weekly_drawdown_pct": 0.06,
            "breach_action": "reduce",
            "breach_cooldown_days": 2
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
