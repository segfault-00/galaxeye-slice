"""Shared test fixtures."""

import io
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.config import Settings, get_settings
from app.db.connection import init_db
from app.db.repository import TileRepository
from app.dependencies import get_engine, get_repository, get_tile_service
from app.inference.engine import InferenceEngine
from app.main import app
from app.services.tile_service import TileService


def _test_settings(tmp_path: Path) -> Settings:
    """Create settings pointing at a temporary directory."""
    artifact_dir = tmp_path / "artifacts"
    artifact_dir.mkdir()
    return Settings(
        artifact_dir=artifact_dir,
        model_path=Path("model.onnx"),  # real model.onnx from project root
        db_path=tmp_path / "test_tiles.db",
    )


@pytest.fixture()
def settings(tmp_path: Path) -> Settings:
    """Test-scoped settings backed by a temporary directory."""
    return _test_settings(tmp_path)


@pytest.fixture()
def client(tmp_path: Path) -> TestClient:
    """``TestClient`` with all dependencies overridden to use temp dirs."""
    test_settings = _test_settings(tmp_path)
    init_db(test_settings.db_path)

    repo = TileRepository(test_settings.db_path)
    engine = InferenceEngine(
        test_settings.model_path,
        resize_dim=test_settings.resize_dim,
        intra_op_threads=test_settings.onnx_intra_op_threads,
    )
    service = TileService(repo=repo, engine=engine, settings=test_settings)

    app.dependency_overrides[get_settings] = lambda: test_settings
    app.dependency_overrides[get_repository] = lambda: repo
    app.dependency_overrides[get_engine] = lambda: engine
    app.dependency_overrides[get_tile_service] = lambda: service

    with TestClient(app) as tc:
        yield tc

    app.dependency_overrides.clear()


@pytest.fixture()
def sample_image_bytes() -> bytes:
    """Minimal valid PNG image (64×64 synthetic tile)."""
    img = Image.new("RGB", (64, 64), color=(128, 64, 32))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()
