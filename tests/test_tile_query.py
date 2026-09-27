"""Tests for ``GET /tiles`` and ``GET /tiles/{tile_id}``."""

from fastapi.testclient import TestClient


def test_query_empty_db(client: TestClient) -> None:
    """Querying an empty database should return an empty list."""
    resp = client.get("/tiles")
    assert resp.status_code == 200
    data = resp.json()
    assert data["count"] == 0
    assert data["results"] == []


def test_get_nonexistent_tile_returns_404(client: TestClient) -> None:
    """Fetching a tile that doesn't exist should return 404."""
    resp = client.get("/tiles/nonexistent-id")
    assert resp.status_code == 404


def test_query_filters_by_state(
    client: TestClient, sample_image_bytes: bytes
) -> None:
    """Results filtered by state should only contain matching tiles."""
    client.post(
        "/tiles",
        files={"file": ("tile.png", sample_image_bytes, "image/png")},
    )

    # Query for COMPLETED tiles specifically
    resp = client.get("/tiles", params={"state": "COMPLETED"})
    assert resp.status_code == 200
    for item in resp.json()["results"]:
        assert item["state"] == "COMPLETED"


def test_query_respects_limit(
    client: TestClient, sample_image_bytes: bytes
) -> None:
    """The ``limit`` parameter should cap the number of results."""
    # Ingest several tiles
    for _ in range(3):
        client.post(
            "/tiles",
            files={"file": ("tile.png", sample_image_bytes, "image/png")},
        )

    resp = client.get("/tiles", params={"limit": 2})
    assert resp.status_code == 200
    assert len(resp.json()["results"]) <= 2


def test_query_filters_by_class(
    client: TestClient, sample_image_bytes: bytes
) -> None:
    """Filtering by predicted_class should return only matching tiles."""
    ingest_resp = client.post(
        "/tiles",
        files={"file": ("tile.png", sample_image_bytes, "image/png")},
    )
    predicted = ingest_resp.json()["predicted_class"]

    resp = client.get("/tiles", params={"predicted_class": predicted})
    assert resp.status_code == 200
    for item in resp.json()["results"]:
        assert item["predicted_class"] == predicted
