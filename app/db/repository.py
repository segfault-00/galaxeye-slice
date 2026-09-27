"""
Tile repository — all SQL lives here.

Keeps raw SQL out of the service layer so the business logic reads as
operations on tiles, not operations on rows.
"""

import json
import sqlite3
from pathlib import Path
from typing import Any

from app.db.connection import get_connection


class TileRepository:
    """CRUD operations on the ``tiles`` table."""

    def __init__(self, db_path: Path) -> None:
        self._db_path = db_path

    # ── Writes ──────────────────────────────────────────────────────

    def insert(
        self,
        *,
        tile_id: str,
        artifact_path: str,
        checksum: str,
        original_filename: str | None,
        ingested_at: str,
        state: str,
    ) -> None:
        """Insert a new tile row in its initial state."""
        with get_connection(self._db_path) as conn:
            conn.execute(
                """INSERT INTO tiles
                   (id, artifact_path, checksum_sha256, original_filename,
                    ingested_at, state)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (tile_id, artifact_path, checksum, original_filename,
                 ingested_at, state),
            )

    def update_state(self, tile_id: str, state: str) -> None:
        """Transition a tile to a new lifecycle state."""
        with get_connection(self._db_path) as conn:
            conn.execute(
                "UPDATE tiles SET state = ? WHERE id = ?",
                (state, tile_id),
            )

    def update_result(
        self,
        tile_id: str,
        *,
        state: str,
        predicted_class: str,
        confidence: float,
        all_scores: dict[str, float],
    ) -> None:
        """Record a successful inference result."""
        with get_connection(self._db_path) as conn:
            conn.execute(
                """UPDATE tiles
                   SET state = ?, predicted_class = ?, confidence = ?,
                       all_scores_json = ?
                   WHERE id = ?""",
                (state, predicted_class, confidence,
                 json.dumps(all_scores), tile_id),
            )

    def update_failure(self, tile_id: str, reason: str) -> None:
        """Mark a tile as FAILED with the exception message."""
        with get_connection(self._db_path) as conn:
            conn.execute(
                "UPDATE tiles SET state = ?, failure_reason = ? WHERE id = ?",
                ("FAILED", reason, tile_id),
            )

    # ── Reads ───────────────────────────────────────────────────────

    def get_by_id(self, tile_id: str) -> dict[str, Any] | None:
        """Fetch a single tile by primary key, or ``None``."""
        with get_connection(self._db_path) as conn:
            row = conn.execute(
                "SELECT * FROM tiles WHERE id = ?", (tile_id,)
            ).fetchone()
        if row is None:
            return None
        result = dict(row)
        if result.get("all_scores_json"):
            result["all_scores"] = json.loads(result.pop("all_scores_json"))
        return result

    def query(
        self,
        *,
        predicted_class: str | None = None,
        min_confidence: float | None = None,
        state: str | None = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """Filtered listing — uses the indexed columns from the schema."""
        clauses: list[str] = []
        params: list[Any] = []

        if predicted_class:
            clauses.append("predicted_class = ?")
            params.append(predicted_class)
        if min_confidence is not None:
            clauses.append("confidence >= ?")
            params.append(min_confidence)
        if state:
            clauses.append("state = ?")
            params.append(state)

        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        sql = (
            "SELECT id, predicted_class, confidence, state, ingested_at "
            f"FROM tiles {where} ORDER BY ingested_at DESC LIMIT ?"
        )
        params.append(limit)

        with get_connection(self._db_path) as conn:
            rows = conn.execute(sql, params).fetchall()
        return [dict(r) for r in rows]
