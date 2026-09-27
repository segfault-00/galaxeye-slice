"""
SQLite connection management.

SQLite is used instead of Postgres purely to keep this thin slice zero-setup.
The design note specifies Postgres for the real system — this is a scoped-down
substitution, not a design reversal.  Schema mirrors the design note's
``tiles`` table (minus PostGIS ``bbox``, which SQLite has no equivalent for).
"""

import logging
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Generator

logger = logging.getLogger(__name__)

SCHEMA = """\
CREATE TABLE IF NOT EXISTS tiles (
    id              TEXT PRIMARY KEY,
    artifact_path   TEXT NOT NULL,
    checksum_sha256 TEXT NOT NULL,
    original_filename TEXT,
    ingested_at     TEXT NOT NULL,
    state           TEXT NOT NULL,
    predicted_class TEXT,
    confidence      REAL,
    all_scores_json TEXT,
    failure_reason  TEXT
);
CREATE INDEX IF NOT EXISTS idx_tiles_state ON tiles (state);
CREATE INDEX IF NOT EXISTS idx_tiles_class_conf ON tiles (predicted_class, confidence);
"""


def init_db(db_path: Path) -> None:
    """Create the schema if it doesn't exist yet."""
    conn = sqlite3.connect(str(db_path))
    conn.executescript(SCHEMA)
    conn.commit()
    conn.close()
    logger.info("Database initialized at %s", db_path)


@contextmanager
def get_connection(db_path: Path) -> Generator[sqlite3.Connection, None, None]:
    """Yields a connection with ``row_factory=sqlite3.Row``, auto-commits."""
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()
