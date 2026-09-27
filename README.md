# GalaxEye Take-Home — Part 2: Working Slice

Implements the core path from the design note: **one endpoint that takes a
tile, runs a classifier, and stores the result** — plus two minimal
read/query endpoints so the "analyst can query results" half of the design
is at least demonstrable, not just described.

## Run it

```bash
pip install -r requirements.txt          # or: pip install ".[dev]"
python scripts/build_model.py            # generates model.onnx (see "About the model")
uvicorn app.main:app --reload
```

Server runs at `http://127.0.0.1:8000`. Interactive docs at `/docs`.

## Try it

```bash
# ingest a tile
curl -X POST http://127.0.0.1:8000/tiles -F "file=@some_tile.png"
# -> {"id": "...", "state": "COMPLETED", "predicted_class": "Forest", "confidence": 0.19, "checksum_sha256": "..."}

# fetch one result
curl http://127.0.0.1:8000/tiles/<id>

# analyst query: filter by class / confidence / state
curl "http://127.0.0.1:8000/tiles?predicted_class=Forest&min_confidence=0.15"
curl "http://127.0.0.1:8000/tiles?state=FLAGGED_REVIEW"
```

Accepts any standard image format (PNG/JPEG/TIFF via Pillow). If you have
the assignment's tile zip, point curl at individual extracted tile files.

## Run the tests

```bash
pip install ".[dev]"                     # installs pytest + httpx
pytest -v
```

## Configuration

All settings are configurable via environment variables (prefix:
`GALAXEYE_`). See [`.env.example`](.env.example) for the full list.
Defaults match the original hardcoded values, so zero configuration is
needed to run locally.

| Variable | Default | Description |
|---|---|---|
| `GALAXEYE_ARTIFACT_DIR` | `data/artifacts` | Where raw tile bytes are persisted |
| `GALAXEYE_MODEL_PATH` | `model.onnx` | Path to the ONNX classifier |
| `GALAXEYE_DB_PATH` | `tiles.db` | SQLite database path |
| `GALAXEYE_CONFIDENCE_THRESHOLD` | `0.20` | Below this → `FLAGGED_REVIEW` |
| `GALAXEYE_RESIZE_DIM` | `32` | Image resize dimension |
| `GALAXEYE_ONNX_INTRA_OP_THREADS` | `1` | ONNX Runtime thread count |

## What this actually does

`POST /tiles` runs the lifecycle from the design note, collapsed into one
synchronous request (see "What's stubbed" below for why):

```
RECEIVED -> PERSISTED -> INFERRING -> COMPLETED | FLAGGED_REVIEW | FAILED
```

1. Reads the upload, computes SHA-256.
2. Writes the raw bytes to `data/artifacts/<checksum>.<ext>` — artifact
   storage is untouched original bytes, never mutated.
3. Inserts a metadata row (SQLite) with `state=PERSISTED`.
4. Re-reads the artifact and re-verifies the checksum before inference —
   catches a partial write or corruption between steps 2 and 4.
5. Preprocesses deterministically (resize to 32×32, normalize to [0,1],
   flatten) and runs it through an ONNX Runtime session.
6. If confidence is below threshold, the result lands in `FLAGGED_REVIEW`
   instead of `COMPLETED` — nothing is silently dropped (see design note
   Section 3.1 for why a hard cutoff or forced prediction was rejected).
7. Any exception in steps 4–6 (corrupt image, checksum mismatch) writes
   `state=FAILED` with the actual exception message recorded, and returns
   HTTP 422 — not a crash, not a silent 500.

## Project structure

```
app/
  main.py              Slim FastAPI app factory — wires router + lifespan
  config.py            Pydantic Settings (env-driven configuration)
  lifespan.py          Modern lifespan context manager (startup/shutdown)
  dependencies.py      FastAPI Depends() providers (replaces globals)
  exceptions.py        Custom exceptions + JSON error handlers

  models/
    tile.py            Pydantic request/response schemas

  routes/
    tiles.py           POST /tiles, GET /tiles/{id}, GET /tiles
    health.py          GET /health

  services/
    tile_service.py    Business logic (ingestion lifecycle)

  db/
    connection.py      SQLite connection management
    repository.py      Tile CRUD operations (repository pattern)

  inference/
    constants.py       CLASSES list (single source of truth)
    engine.py          Preprocessing + ONNX Runtime wrapper

scripts/
  build_model.py       Generates model.onnx (see "About the model")

tests/
  conftest.py          Shared fixtures (TestClient, temp dirs)
  test_health.py       Health endpoint tests
  test_tile_ingest.py  Tile ingestion tests
  test_tile_query.py   Query endpoint tests

pyproject.toml         Project metadata + dependencies
.env.example           Environment variable template
requirements.txt       Flat dependency list (for compatibility)
```

## About the model

There's no real trained model here, and that's intentional, not a
shortcut I'm hiding: the assignment explicitly says accuracy isn't graded,
and getting a legitimately trained EuroSAT classifier meant either (a)
downloading a real pretrained model from a hub, which conflicts with
"offline, no internet at runtime" as a *demonstration* of the constraint,
or (b) training one from scratch on the provided tiles, which is a
reasonable next step but not the point of a *thin slice*.

So `scripts/build_model.py` builds a small deterministic ONNX graph
(seeded random `Gemm -> Softmax` over a flattened, resized image) directly
via the `onnx` package — no training data, no downloads, fully
reproducible from the seed. It exercises the real architecture end-to-end
(ONNX Runtime CPU inference, the exact input/output contract a real model
would have) while being honest that it has learned nothing. Confidence
scores cluster near uniform (1/7 ≈ 0.143) as a result, which is why
`CONFIDENCE_THRESHOLD` in `app/config.py` is set low (0.20) — a
realistic production threshold (e.g. 0.6+) would flag nearly every tile,
which would be true-to-the-model but useless as a demo of the
`FLAGGED_REVIEW` branch actually triggering sometimes.

**Swapping in a real model** is a drop-in replacement: train/export any
image classifier to ONNX with the same input contract (a fixed-size
normalized tensor) and output contract (N-way softmax over the same
`CLASSES` list, same index order, defined in `app/inference/constants.py`)
— replace `model.onnx` and nothing else in `app/` needs to change.

## What's stubbed, and why

This is a thin slice proving the core path runs, not the full system from
the design note. Each stub below is a scoped-down substitution, not a
design reversal — the design note's Section 1/4 covers what each of these
looks like at production scale:

| Design note | This slice | Why |
|---|---|---|
| Message broker + worker pool (async, decoupled) | Inference runs inline in the request handler | At single-tile-at-a-time demo volume there's no burst to decouple from. The queue/worker split is exactly what you'd re-introduce the moment ingestion could outpace inference latency. |
| PostgreSQL | SQLite | Zero-setup for a take-home reviewer — no server, no connection string. Schema (see `app/db/connection.py`) mirrors the design note's `tiles` table, minus PostGIS (SQLite has no spatial index equivalent). |
| Retry budget / claim timeout on `INFERRING` | None — a crash mid-inference just leaves a row stuck in `INFERRING` | No worker pool to crash independently of the API process in this slice; the pattern is documented but not worth implementing without the process boundary it protects. |
| Reconciliation sweep (catches `PERSISTED` rows that never got enqueued) | N/A | No enqueue step exists in this synchronous slice — nothing to reconcile. |
| OOM isolation / cgroup limits per worker | None | No worker pool to isolate. The placeholder model's memory footprint is negligible. |
| Spatial bounding-box query | Not implemented | Real tiles' georeferencing metadata (CRS, bounds) wasn't something I fabricated for synthetic test tiles — GET /tiles supports the non-spatial filters (class, confidence, state) that don't depend on it. |
