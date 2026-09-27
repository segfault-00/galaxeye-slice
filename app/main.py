"""
GalaxEye Tile Classification — Thin Slice

Application factory: creates a FastAPI app with routes, exception handlers,
and lifespan management.  All business logic lives in ``app.services``,
all data access in ``app.db``, and all ML code in ``app.inference``.
"""

import logging

from fastapi import FastAPI

from app.exceptions import register_exception_handlers
from app.lifespan import lifespan
from app.routes import api_router

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)

app = FastAPI(
    title="GalaxEye Tile Classification — Thin Slice",
    description=(
        "Ingests satellite image tiles, classifies them via ONNX, "
        "and stores results."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.include_router(api_router)
register_exception_handlers(app)
