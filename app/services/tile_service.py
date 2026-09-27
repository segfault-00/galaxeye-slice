"""
Tile ingestion and query business logic.

Orchestrates the lifecycle state machine documented in the design note::

    RECEIVED → PERSISTED → INFERRING → COMPLETED | FLAGGED_REVIEW | FAILED
"""

import hashlib
import logging
import os
import uuid
from datetime import datetime, timezone

from app.config import Settings
from app.db.repository import TileRepository
from app.exceptions import ArtifactCorruptedError, InferenceError
from app.inference.engine import InferenceEngine
from app.models.tile import TileIngestResponse

logger = logging.getLogger(__name__)


class TileService:
    """Encapsulates the core tile ingestion workflow.

    Receives its collaborators via constructor injection so the service
    is easy to test with fakes/stubs.
    """

    def __init__(
        self,
        repo: TileRepository,
        engine: InferenceEngine,
        settings: Settings,
    ) -> None:
        self._repo = repo
        self._engine = engine
        self._settings = settings

    # ── Public API ──────────────────────────────────────────────────

    async def ingest(
        self,
        raw_bytes: bytes,
        original_filename: str | None,
    ) -> TileIngestResponse:
        """Run the full lifecycle for one uploaded tile image."""
        tile_id = str(uuid.uuid4())
        checksum = hashlib.sha256(raw_bytes).hexdigest()
        ingested_at = datetime.now(timezone.utc).isoformat()

        # ── PERSISTED: write artifact + row before touching the model ──
        ext = os.path.splitext(original_filename or "")[1] or ".bin"
        artifact_path = str(self._settings.artifact_dir / f"{checksum}{ext}")

        with open(artifact_path, "wb") as fh:
            fh.write(raw_bytes)

        self._repo.insert(
            tile_id=tile_id,
            artifact_path=artifact_path,
            checksum=checksum,
            original_filename=original_filename,
            ingested_at=ingested_at,
            state="PERSISTED",
        )
        logger.info(
            "Tile %s persisted (%s, %d bytes)",
            tile_id,
            checksum[:12],
            len(raw_bytes),
        )

        # ── INFERRING → terminal state ──
        self._repo.update_state(tile_id, "INFERRING")

        try:
            self._verify_artifact(artifact_path, checksum, tile_id)
            predicted_class, confidence, all_scores = self._run_inference(
                artifact_path
            )
            final_state = self._determine_state(confidence)

            self._repo.update_result(
                tile_id,
                state=final_state,
                predicted_class=predicted_class,
                confidence=confidence,
                all_scores=all_scores,
            )
            logger.info(
                "Tile %s → %s (class=%s, conf=%.4f)",
                tile_id,
                final_state,
                predicted_class,
                confidence,
            )
            return TileIngestResponse(
                id=tile_id,
                state=final_state,
                predicted_class=predicted_class,
                confidence=round(confidence, 4),
                checksum_sha256=checksum,
            )

        except (ArtifactCorruptedError, InferenceError):
            raise
        except Exception as exc:
            reason = str(exc)
            self._repo.update_failure(tile_id, reason)
            logger.error("Tile %s FAILED: %s", tile_id, reason)
            raise InferenceError(tile_id, reason) from exc

    # ── Private helpers ─────────────────────────────────────────────

    def _verify_artifact(
        self, artifact_path: str, expected_checksum: str, tile_id: str
    ) -> None:
        """Re-verify integrity before feeding the model (design note §2.3)."""
        with open(artifact_path, "rb") as fh:
            actual = hashlib.sha256(fh.read()).hexdigest()
        if actual != expected_checksum:
            raise ArtifactCorruptedError(tile_id)

    def _run_inference(
        self, artifact_path: str
    ) -> tuple[str, float, dict[str, float]]:
        """Read the persisted artifact and run the model."""
        with open(artifact_path, "rb") as fh:
            raw_bytes = fh.read()
        return self._engine.predict(raw_bytes)

    def _determine_state(self, confidence: float) -> str:
        """Route to COMPLETED or FLAGGED_REVIEW based on threshold."""
        if confidence >= self._settings.confidence_threshold:
            return "COMPLETED"
        return "FLAGGED_REVIEW"
