"""
Centralized configuration via Pydantic Settings.

All hardcoded paths and tuning knobs from the original codebase are now
driven by environment variables (with sensible defaults matching the
original behavior).  See ``.env.example`` for the full list.
"""

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application configuration loaded from environment variables."""

    # ── Paths ───────────────────────────────────────────────────────
    artifact_dir: Path = Path("data/artifacts")
    model_path: Path = Path("model.onnx")
    db_path: Path = Path("tiles.db")

    # ── Inference tuning ────────────────────────────────────────────
    confidence_threshold: float = 0.20
    resize_dim: int = 32
    onnx_intra_op_threads: int = 1

    # ── API defaults ────────────────────────────────────────────────
    default_query_limit: int = 50
    max_query_limit: int = 500

    model_config = {"env_prefix": "GALAXEYE_", "env_file": ".env", "extra": "ignore"}


@lru_cache
def get_settings() -> Settings:
    """Cached singleton — avoids re-reading env on every request."""
    return Settings()
