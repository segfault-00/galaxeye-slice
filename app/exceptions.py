"""
Application-specific exceptions and FastAPI exception handlers.

Custom exceptions make the service layer expressive (``raise TileNotFoundError``)
while the handlers ensure consistent JSON error responses.
"""

import logging

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)


# ── Exceptions ──────────────────────────────────────────────────────


class TileNotFoundError(Exception):
    """Raised when a tile ID does not exist in the database."""

    def __init__(self, tile_id: str) -> None:
        self.tile_id = tile_id
        super().__init__(f"Tile '{tile_id}' not found")


class ArtifactCorruptedError(Exception):
    """Raised when a persisted artifact fails checksum verification."""

    def __init__(self, tile_id: str) -> None:
        self.tile_id = tile_id
        super().__init__(f"Artifact for tile '{tile_id}' failed integrity check")


class InferenceError(Exception):
    """Wraps any exception that occurs during model inference."""

    def __init__(self, tile_id: str, reason: str) -> None:
        self.tile_id = tile_id
        self.reason = reason
        super().__init__(f"Inference failed for tile '{tile_id}': {reason}")


# ── Handlers ────────────────────────────────────────────────────────


def register_exception_handlers(app: FastAPI) -> None:
    """Attach custom handlers so errors produce consistent JSON responses."""

    @app.exception_handler(TileNotFoundError)
    async def _tile_not_found(
        request: Request, exc: TileNotFoundError
    ) -> JSONResponse:
        return JSONResponse(
            status_code=404, content={"detail": str(exc)}
        )

    @app.exception_handler(ArtifactCorruptedError)
    async def _artifact_corrupted(
        request: Request, exc: ArtifactCorruptedError
    ) -> JSONResponse:
        logger.error("Artifact corruption detected: %s", exc)
        return JSONResponse(
            status_code=422, content={"detail": str(exc)}
        )

    @app.exception_handler(InferenceError)
    async def _inference_error(
        request: Request, exc: InferenceError
    ) -> JSONResponse:
        logger.error("Inference failed: %s", exc)
        return JSONResponse(
            status_code=422,
            content={
                "id": exc.tile_id,
                "state": "FAILED",
                "failure_reason": exc.reason,
            },
        )
