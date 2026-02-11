"""Collaborative Funds API endpoints"""
from typing import Optional
from pydantic import BaseModel, Field
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, desc

from db import get_db

router = APIRouter(prefix="/api", tags=["funds"])


class FundResponse(BaseModel):
    """Fund summary response"""
    fund_id: str = Field(serialization_alias="fundId")
    name: str
    strategy: str
    description: Optional[str] = None
    total_value: float = Field(serialization_alias="totalValue")
    cash_balance: float = Field(serialization_alias="cashBalance")
    gross_exposure: float = Field(serialization_alias="grossExposure")
    net_exposure: float = Field(serialization_alias="netExposure")
    n_positions: int = Field(serialization_alias="nPositions")
    is_active: bool = Field(serialization_alias="isActive")

    model_config = {"from_attributes": True, "populate_by_name": True}


class FundDetailResponse(FundResponse):
    """Detailed fund response including thesis and policy"""
    thesis: Optional[dict] = None
    policy: Optional[dict] = None
    risk_limits: Optional[dict] = Field(serialization_alias="riskLimits", default=None)


class FundPositionResponse(BaseModel):
    """Fund position response"""
    symbol: str
    quantity: float
    avg_entry_price: float = Field(serialization_alias="avgEntryPrice")
    current_price: float = Field(serialization_alias="currentPrice")
    market_value: float = Field(serialization_alias="marketValue")
    unrealized_pnl: float = Field(serialization_alias="unrealizedPnl")
    weight_pct: float = Field(serialization_alias="weightPct")

    model_config = {"populate_by_name": True}


class DecisionResponse(BaseModel):
    """Decision record response"""
    decision_id: str = Field(serialization_alias="decisionId")
    fund_id: str = Field(serialization_alias="fundId")
    asof_timestamp: str = Field(serialization_alias="asofTimestamp")
    decision_type: str = Field(serialization_alias="decisionType")
    status: str
    no_trade_reason: Optional[str] = Field(
        serialization_alias="noTradeReason", default=None
    )
    universe_hash: Optional[str] = Field(
        serialization_alias="universeHash", default=None
    )
    inputs_hash: Optional[str] = Field(
        serialization_alias="inputsHash", default=None
    )
    predicted_directions: Optional[dict] = Field(
        serialization_alias="predictedDirections", default=None
    )
    expected_return: Optional[float] = Field(
        serialization_alias="expectedReturn", default=None
    )

    model_config = {"populate_by_name": True}


class DebateResponse(BaseModel):
    """Debate transcript summary response"""
    transcript_id: str = Field(serialization_alias="transcriptId")
    fund_id: str = Field(serialization_alias="fundId")
    started_at: str = Field(serialization_alias="startedAt")
    completed_at: Optional[str] = Field(
        serialization_alias="completedAt", default=None
    )
    num_proposals: int = Field(serialization_alias="numProposals")
    num_critiques: int = Field(serialization_alias="numCritiques")
    final_consensus_level: float = Field(serialization_alias="finalConsensusLevel")

    model_config = {"populate_by_name": True}


class FundLeaderboardEntry(BaseModel):
    """Fund leaderboard entry"""
    rank: int
    fund_id: str = Field(serialization_alias="fundId")
    name: str
    strategy: str
    total_value: float = Field(serialization_alias="totalValue")
    gross_exposure: float = Field(serialization_alias="grossExposure")
    is_active: bool = Field(serialization_alias="isActive")

    model_config = {"populate_by_name": True}


@router.get("/funds", response_model=list[FundResponse])
async def get_funds(db: AsyncSession = Depends(get_db)):
    """Get all funds"""
    from db.models import FundModel

    result = await db.execute(select(FundModel))
    funds = result.scalars().all()

    return [
        FundResponse(
            fund_id=f.id,
            name=f.name,
            strategy=f.strategy,
            description=f.description,
            total_value=float(f.total_value or 0),
            cash_balance=float(f.cash_balance or 0),
            gross_exposure=0.0,  # Computed on demand
            net_exposure=0.0,
            n_positions=0,
            is_active=f.is_active,
        )
        for f in funds
    ]


@router.get("/funds/{fund_id}", response_model=FundDetailResponse)
async def get_fund(fund_id: str, db: AsyncSession = Depends(get_db)):
    """Get fund details"""
    from db.models import FundModel, FundPosition

    result = await db.execute(
        select(FundModel).where(FundModel.id == fund_id)
    )
    fund = result.scalar_one_or_none()
    if not fund:
        raise HTTPException(status_code=404, detail="Fund not found")

    # Get positions
    pos_result = await db.execute(
        select(FundPosition).where(FundPosition.fund_id == fund_id)
    )
    positions = pos_result.scalars().all()

    total_value = float(fund.total_value or 0)
    gross_exposure = 0.0
    net_exposure = 0.0

    if total_value > 0:
        for pos in positions:
            market_value = float(pos.quantity or 0) * float(pos.current_price or 0)
            weight = market_value / total_value
            gross_exposure += abs(weight)
            net_exposure += weight

    return FundDetailResponse(
        fund_id=fund.id,
        name=fund.name,
        strategy=fund.strategy,
        description=fund.description,
        total_value=total_value,
        cash_balance=float(fund.cash_balance or 0),
        gross_exposure=gross_exposure,
        net_exposure=net_exposure,
        n_positions=len(positions),
        is_active=fund.is_active,
        thesis=fund.thesis_json,
        policy=fund.policy_json,
        risk_limits=fund.risk_limits_json,
    )


@router.get("/funds/{fund_id}/positions", response_model=list[FundPositionResponse])
async def get_fund_positions(fund_id: str, db: AsyncSession = Depends(get_db)):
    """Get positions for a fund"""
    from db.models import FundModel, FundPosition

    # Get fund for total value
    fund_result = await db.execute(
        select(FundModel).where(FundModel.id == fund_id)
    )
    fund = fund_result.scalar_one_or_none()
    if not fund:
        raise HTTPException(status_code=404, detail="Fund not found")

    total_value = float(fund.total_value or 1)

    result = await db.execute(
        select(FundPosition).where(FundPosition.fund_id == fund_id)
    )
    positions = result.scalars().all()

    return [
        FundPositionResponse(
            symbol=pos.symbol,
            quantity=float(pos.quantity or 0),
            avg_entry_price=float(pos.avg_entry_price or 0),
            current_price=float(pos.current_price or 0),
            market_value=float(pos.quantity or 0) * float(pos.current_price or 0),
            unrealized_pnl=float(pos.unrealized_pnl or 0),
            weight_pct=(
                (float(pos.quantity or 0) * float(pos.current_price or 0))
                / total_value * 100
            ),
        )
        for pos in positions
    ]


@router.get("/funds/{fund_id}/decisions", response_model=list[DecisionResponse])
async def get_fund_decisions(
    fund_id: str,
    limit: int = Query(50, ge=1, le=500),
    db: AsyncSession = Depends(get_db)
):
    """Get recent decisions for a fund"""
    from db.models import DecisionRecordModel

    result = await db.execute(
        select(DecisionRecordModel)
        .where(DecisionRecordModel.fund_id == fund_id)
        .order_by(desc(DecisionRecordModel.asof_timestamp))
        .limit(limit)
    )
    decisions = result.scalars().all()

    return [
        DecisionResponse(
            decision_id=d.id,
            fund_id=d.fund_id,
            asof_timestamp=d.asof_timestamp.isoformat(),
            decision_type=d.decision_type,
            status=d.status,
            no_trade_reason=d.no_trade_reason,
            universe_hash=d.universe_hash,
            inputs_hash=d.inputs_hash,
            predicted_directions=d.predicted_directions_json,
            expected_return=d.expected_return,
        )
        for d in decisions
    ]


@router.get("/decisions/{decision_id}")
async def get_decision_detail(decision_id: str, db: AsyncSession = Depends(get_db)):
    """Get full decision record with intent and risk result"""
    from db.models import DecisionRecordModel

    result = await db.execute(
        select(DecisionRecordModel).where(DecisionRecordModel.id == decision_id)
    )
    decision = result.scalar_one_or_none()
    if not decision:
        raise HTTPException(status_code=404, detail="Decision not found")

    return {
        "decision_id": decision.id,
        "fund_id": decision.fund_id,
        "snapshot_id": decision.snapshot_id,
        "asof_timestamp": decision.asof_timestamp.isoformat(),
        "idempotency_key": decision.idempotency_key,
        "run_context": decision.run_context,
        "decision_type": decision.decision_type,
        "no_trade_reason": decision.no_trade_reason,
        "status": decision.status,
        "status_history": decision.status_history_json,
        "intent": decision.intent_json,
        "risk_result": decision.risk_result_json,
        "snapshot_quality": decision.snapshot_quality_json,
        "universe_result": decision.universe_result_json,
        "universe_hash": decision.universe_hash,
        "inputs_hash": decision.inputs_hash,
        "model_versions": decision.model_versions_json,
        "prompt_hashes": decision.prompt_hashes_json,
        "predicted_directions": decision.predicted_directions_json,
        "expected_return": decision.expected_return,
        "expected_holding_days": decision.expected_holding_days,
    }


@router.get("/decisions/{decision_id}/debate", response_model=DebateResponse)
async def get_decision_debate(decision_id: str, db: AsyncSession = Depends(get_db)):
    """Get debate transcript for a decision (drill-down)"""
    from db.models import DebateTranscriptModel

    result = await db.execute(
        select(DebateTranscriptModel)
        .where(DebateTranscriptModel.decision_id == decision_id)
    )
    transcript = result.scalar_one_or_none()
    if not transcript:
        raise HTTPException(status_code=404, detail="Debate transcript not found")

    return DebateResponse(
        transcript_id=transcript.id,
        fund_id=transcript.fund_id,
        started_at=transcript.started_at.isoformat(),
        completed_at=(
            transcript.completed_at.isoformat() if transcript.completed_at else None
        ),
        num_proposals=transcript.num_proposals,
        num_critiques=transcript.num_critiques,
        final_consensus_level=transcript.final_consensus_level or 0.0,
    )


@router.get("/debates/{transcript_id}")
async def get_debate_detail(transcript_id: str, db: AsyncSession = Depends(get_db)):
    """Get full debate transcript with all messages"""
    from db.models import DebateTranscriptModel

    result = await db.execute(
        select(DebateTranscriptModel).where(DebateTranscriptModel.id == transcript_id)
    )
    transcript = result.scalar_one_or_none()
    if not transcript:
        raise HTTPException(status_code=404, detail="Debate transcript not found")

    return {
        "transcript_id": transcript.id,
        "fund_id": transcript.fund_id,
        "snapshot_id": transcript.snapshot_id,
        "started_at": transcript.started_at.isoformat(),
        "completed_at": (
            transcript.completed_at.isoformat() if transcript.completed_at else None
        ),
        "messages": transcript.messages_json or [],
        "num_proposals": transcript.num_proposals,
        "num_critiques": transcript.num_critiques,
        "final_consensus_level": transcript.final_consensus_level,
        "total_input_tokens": transcript.total_input_tokens,
        "total_output_tokens": transcript.total_output_tokens,
    }


@router.get("/funds/leaderboard", response_model=list[FundLeaderboardEntry])
async def get_funds_leaderboard(db: AsyncSession = Depends(get_db)):
    """Get fund leaderboard sorted by total value"""
    from db.models import FundModel

    result = await db.execute(
        select(FundModel).order_by(desc(FundModel.total_value))
    )
    funds = result.scalars().all()

    entries = []
    for i, fund in enumerate(funds):
        entries.append(FundLeaderboardEntry(
            rank=i + 1,
            fund_id=fund.id,
            name=fund.name,
            strategy=fund.strategy,
            total_value=float(fund.total_value or 0),
            gross_exposure=0.0,  # Would need to compute from positions
            is_active=fund.is_active,
        ))

    return entries


@router.post("/funds/trading/cycle")
async def trigger_fund_trading_cycle():
    """Manually trigger a trading cycle for all funds"""
    # Placeholder - would instantiate FundTradingEngine and run
    return {
        "success": True,
        "message": "Fund trading cycle not yet implemented",
        "note": "This will run debates for all active funds"
    }
