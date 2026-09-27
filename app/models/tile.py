"""
Pydantic schemas for tile-related request and response payloads.

These enforce type safety at the API boundary and auto-generate the
OpenAPI schema shown in ``/docs``.
"""

from pydantic import BaseModel, Field


# ── Ingestion ───────────────────────────────────────────────────────


class TileIngestResponse(BaseModel):
    """Returned from ``POST /tiles`` on successful processing."""

    id: str
    state: str
    predicted_class: str
    confidence: float = Field(
        ..., description="Softmax confidence for the predicted class"
    )
    checksum_sha256: str


class TileFailureResponse(BaseModel):
    """Returned from ``POST /tiles`` when inference fails."""

    id: str
    state: str = "FAILED"
    failure_reason: str


# ── Detail ──────────────────────────────────────────────────────────


class TileDetailResponse(BaseModel):
    """Returned from ``GET /tiles/{tile_id}``."""

    id: str
    artifact_path: str
    checksum_sha256: str
    original_filename: str | None = None
    ingested_at: str
    state: str
    predicted_class: str | None = None
    confidence: float | None = None
    all_scores: dict[str, float] | None = None
    failure_reason: str | None = None


# ── Query / List ────────────────────────────────────────────────────


class TileListItem(BaseModel):
    """Single item in the analyst query results."""

    id: str
    predicted_class: str | None = None
    confidence: float | None = None
    state: str
    ingested_at: str


class TileListResponse(BaseModel):
    """Returned from ``GET /tiles`` (analyst query)."""

    count: int
    results: list[TileListItem]


# ── Health ──────────────────────────────────────────────────────────


class HealthResponse(BaseModel):
    """Returned from ``GET /health``."""

    status: str = "ok"
    model_loaded: bool
