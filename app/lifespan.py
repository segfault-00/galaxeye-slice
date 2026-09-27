"""
Application lifespan context manager.

Replaces the deprecated ``@app.on_event("startup")`` / ``@app.on_event("shutdown")``
with the modern ``lifespan`` protocol introduced in FastAPI 0.93+.
"""

import logging
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.config import get_settings
from app.db.connection import init_db
from app.dependencies import get_engine

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Runs once on startup; yields while the app serves; cleans up on shutdown."""
    settings = get_settings()

    # ── Startup ─────────────────────────────────────────────────────
    settings.artifact_dir.mkdir(parents=True, exist_ok=True)
    init_db(settings.db_path)

    if not settings.model_path.exists():
        raise RuntimeError(
            f"{settings.model_path} not found. "
            "Run `python scripts/build_model.py` first (see README)."
        )

    # Eagerly load the model so a bad .onnx file fails fast at startup.
    get_engine()
    logger.info("GalaxEye Tile Classification service started")

    yield  # ← app is running

    # ── Shutdown ────────────────────────────────────────────────────
    logger.info("GalaxEye Tile Classification service shutting down")
