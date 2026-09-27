"""Health-check endpoint."""

from fastapi import APIRouter

from app.dependencies import get_engine
from app.models.tile import HealthResponse

router = APIRouter(tags=["Health"])


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Liveness / readiness probe."""
    engine = get_engine()
    return HealthResponse(model_loaded=engine is not None)
