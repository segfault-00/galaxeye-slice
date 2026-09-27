# GalaxEye — Tile Classification Service

A lightweight, robust FastAPI microservice for satellite image tile ingestion, ONNX-based deep learning classification, and metadata querying.

---

## Quick Start

### 1. Installation

```bash
# Clone and enter the repository
cd galaxeye-slice

# (Optional) Create & activate a virtual environment
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
# source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

> **Note**: `model.onnx` is pre-trained and included in the repository (79.05% accuracy on EuroSAT evaluation set). No extra setup or downloads needed.

### 2. Run the Server

```bash
uvicorn app.main:app --reload
```

- **API Base:** `http://127.0.0.1:8000`
- **Interactive Swagger Docs:** `http://127.0.0.1:8000/docs`
- **Health Check:** `http://127.0.0.1:8000/health`

---

## API Usage

### Ingest a Tile (`POST /tiles`)

Upload an image tile (`PNG`, `JPEG`, or `TIFF`) to persist and classify:

```bash
curl -X POST http://127.0.0.1:8000/tiles -F "file=@sample_tile.png"
```

**Response:**
```json
{
  "id": "c1f7b0a2-4a5e-4c7b-b384-5f128e0e7a2b",
  "state": "COMPLETED",
  "predicted_class": "Forest",
  "confidence": 0.976,
  "checksum_sha256": "0db18eedc3f69d9fabdd3d4eaed00d323e6d8a32ce8f748c4ee0a4093f29d8c5"
}
```

### Fetch a Tile (`GET /tiles/{id}`)

```bash
curl http://127.0.0.1:8000/tiles/<tile_id>
```

### Query Tiles (`GET /tiles`)

Filter tiles by class, minimum confidence, or state:

```bash
# Filter by predicted class and confidence
curl "http://127.0.0.1:8000/tiles?predicted_class=Forest&min_confidence=0.80"

# Filter by lifecycle state
curl "http://127.0.0.1:8000/tiles?state=COMPLETED"
```

---

## Testing

Run the automated test suite:

```bash
pip install pytest httpx
pytest -v
```

---

## Model & Training

- **Classes:** `Forest`, `River`, `Residential`, `Industrial`, `AnnualCrop`, `SeaLake`, `Highway`
- **Architecture:** 4-block CNN (`Conv2D` $\to$ `BatchNorm` $\to$ `ReLU` $\to$ `MaxPool2D`/`AdaptiveAvgPool2D`) with Dropout and Softmax output.
- **Evaluation Accuracy:** **79.05%** overall on hold-out `eval_set` (AnnualCrop 100%, Industrial 93.3%, SeaLake 86.7%, Residential 83.3%).

To re-train or fine-tune the model from `candidate_tiles/`:

```bash
python scripts/train_model.py
```

---

## Configuration

All parameters can be configured via environment variables (prefix `GALAXEYE_`) or a `.env` file (see [`.env.example`](.env.example)):

| Variable | Default | Description |
|---|---|---|
| `GALAXEYE_ARTIFACT_DIR` | `data/artifacts` | Directory where raw tiles are saved |
| `GALAXEYE_MODEL_PATH` | `model.onnx` | Path to the ONNX classification model |
| `GALAXEYE_DB_PATH` | `tiles.db` | Path to the SQLite database |
| `GALAXEYE_CONFIDENCE_THRESHOLD` | `0.20` | Threshold below which tile is flagged for review |
| `GALAXEYE_RESIZE_DIM` | `64` | Image input resolution for the model |

---

## Project Structure

```
├── app/
│   ├── main.py              # Application entrypoint & router assembly
│   ├── config.py            # Pydantic Settings (env configuration)
│   ├── lifespan.py          # Startup & shutdown lifecycle events
│   ├── dependencies.py      # Dependency injection providers
│   ├── exceptions.py        # Custom exceptions and HTTP error handlers
│   ├── db/                  # SQLite connection and repository CRUD operations
│   ├── inference/           # Preprocessing, ONNX runtime engine, and class taxonomy
│   ├── models/              # Pydantic request/response schemas
│   ├── routes/              # Route controllers (/tiles, /health)
│   └── services/            # Core ingestion business logic & integrity checks
├── scripts/
│   ├── train_model.py       # CNN training, ONNX export, and evaluation script
│   └── build_model.py       # Deterministic fallback model builder
├── tests/                   # Pytest suite (health, ingestion, and query tests)
├── DESIGN_NOTE.pdf          # Part 1: System architecture & failure mode design note
├── part_3_answers.md        # Part 3: Operational & troubleshooting problem-solving answers
├── pyproject.toml           # Package metadata & dependencies
└── requirements.txt         # Pinned production dependencies
```
