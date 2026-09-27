"""Tests for ``POST /tiles`` (tile ingestion)."""

from fastapi.testclient import TestClient


def test_ingest_valid_image(client: TestClient, sample_image_bytes: bytes) -> None:
    """A valid image should be classified and persisted."""
    resp = client.post(
        "/tiles",
        files={"file": ("tile.png", sample_image_bytes, "image/png")},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["state"] in ("COMPLETED", "FLAGGED_REVIEW")
    assert "predicted_class" in data
    assert "confidence" in data
    assert "checksum_sha256" in data
    assert "id" in data


def test_ingest_empty_file_returns_400(client: TestClient) -> None:
    """An empty upload should be rejected immediately."""
    resp = client.post(
        "/tiles",
        files={"file": ("empty.png", b"", "image/png")},
    )
    assert resp.status_code == 400


def test_ingest_creates_retrievable_tile(
    client: TestClient, sample_image_bytes: bytes
) -> None:
    """After ingestion, the tile should be retrievable by ID."""
    resp = client.post(
        "/tiles",
        files={"file": ("tile.png", sample_image_bytes, "image/png")},
    )
    tile_id = resp.json()["id"]

    detail = client.get(f"/tiles/{tile_id}")
    assert detail.status_code == 200
    assert detail.json()["id"] == tile_id
    assert detail.json()["state"] in ("COMPLETED", "FLAGGED_REVIEW")
