from fastapi import APIRouter

from app.routes.health import router as health_router
from app.routes.tiles import router as tiles_router

api_router = APIRouter()
api_router.include_router(tiles_router)
api_router.include_router(health_router)

__all__ = ["api_router"]
