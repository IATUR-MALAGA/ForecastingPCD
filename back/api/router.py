from fastapi import APIRouter

from back.api.routes.database import router as database_router
from back.api.routes.SARIMAX_model import router as sarimax_router
from back.api.routes.XGBoost_model import router as xgboost_router
from back.api.routes.scenarios import router as scenarios_router
from back.config import settings


api_router = APIRouter(prefix=settings.get("server.api_prefix", "/api"))
api_router.include_router(database_router)
api_router.include_router(sarimax_router)
api_router.include_router(xgboost_router)
api_router.include_router(scenarios_router)
