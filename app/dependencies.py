"""
FastAPI dependency injection providers.

Replaces module-level globals with properly scoped, testable dependencies.
Each provider is cached so the heavy objects (ONNX session, etc.) are
created once and reused across requests.
"""

from functools import lru_cache

from app.config import get_settings
from app.db.repository import TileRepository
from app.inference.engine import InferenceEngine
from app.services.tile_service import TileService


@lru_cache
def get_repository() -> TileRepository:
    """Singleton tile repository."""
    return TileRepository(get_settings().db_path)


@lru_cache
def get_engine() -> InferenceEngine:
    """Singleton inference engine — loads the ONNX model once."""
    settings = get_settings()
    return InferenceEngine(
        settings.model_path,
        resize_dim=settings.resize_dim,
        intra_op_threads=settings.onnx_intra_op_threads,
    )


@lru_cache
def get_tile_service() -> TileService:
    """Singleton tile service wired with its dependencies."""
    return TileService(
        repo=get_repository(),
        engine=get_engine(),
        settings=get_settings(),
    )
