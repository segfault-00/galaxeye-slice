"""Tile ingestion and query endpoints."""

from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile

from app.db.repository import TileRepository
from app.dependencies import get_repository, get_tile_service
from app.exceptions import TileNotFoundError
from app.models.tile import (
    TileDetailResponse,
    TileIngestResponse,
    TileListResponse,
)
from app.services.tile_service import TileService

router = APIRouter(prefix="/tiles", tags=["Tiles"])


@router.post("", response_model=TileIngestResponse, status_code=200)
async def ingest_tile(
    file: UploadFile = File(...),
    service: TileService = Depends(get_tile_service),
) -> TileIngestResponse:
    """Accept one image file, persist it, run inference, and store the result.

    Mirrors the design note's state machine, collapsed into one synchronous
    request since there's no broker in this slice::

        RECEIVED → PERSISTED → INFERRING → COMPLETED | FLAGGED_REVIEW | FAILED
    """
    raw_bytes = await file.read()
    if not raw_bytes:
        raise HTTPException(status_code=400, detail="Empty file")
    return await service.ingest(raw_bytes, file.filename)


@router.get("/{tile_id}", response_model=TileDetailResponse)
def get_tile(
    tile_id: str,
    repo: TileRepository = Depends(get_repository),
) -> TileDetailResponse:
    """Fetch a single tile by ID."""
    row = repo.get_by_id(tile_id)
    if row is None:
        raise TileNotFoundError(tile_id)
    return TileDetailResponse(**row)


@router.get("", response_model=TileListResponse)
def query_tiles(
    predicted_class: str | None = Query(default=None),
    min_confidence: float | None = Query(default=None),
    state: str | None = Query(default=None),
    limit: int = Query(default=50, le=500),
    repo: TileRepository = Depends(get_repository),
) -> TileListResponse:
    """Analyst query endpoint — filters map onto the indexed columns
    described in the design note (Section 3.4)."""
    rows = repo.query(
        predicted_class=predicted_class,
        min_confidence=min_confidence,
        state=state,
        limit=limit,
    )
    return TileListResponse(count=len(rows), results=rows)
