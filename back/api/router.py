from fastapi import APIRouter

from back.api.routes.database import router as database_router
from back.api.routes.SARIMAX_model import router as sarimax_router


api_router = APIRouter(prefix="/api")
api_router.include_router(database_router)
api_router.include_router(sarimax_router)
