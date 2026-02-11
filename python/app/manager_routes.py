"""Manager API endpoints"""
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from db import get_db, Manager
from app.schemas import ManagerResponse

router = APIRouter(prefix="/api/managers", tags=["managers"])


@router.get("", response_model=list[ManagerResponse])
async def get_managers(db: AsyncSession = Depends(get_db)):
    """Get all active managers"""
    result = await db.execute(
        select(Manager).where(Manager.is_active == True)
    )
    managers = result.scalars().all()
    return managers


@router.get("/{manager_id}", response_model=ManagerResponse)
async def get_manager(manager_id: str, db: AsyncSession = Depends(get_db)):
    """Get a specific manager by ID"""
    result = await db.execute(
        select(Manager).where(Manager.id == manager_id)
    )
    manager = result.scalar_one_or_none()
    if not manager:
        raise HTTPException(status_code=404, detail="Manager not found")
    return manager
