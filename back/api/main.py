import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from back.api.router import api_router
from back.config import settings


def create_app() -> FastAPI:
    logging.basicConfig(
        level=getattr(logging, str(settings.get("logging.level", "INFO")).upper(), logging.INFO),
        format=settings.get("logging.format", "%(asctime)s %(levelname)s %(name)s: %(message)s"),
    )

    app = FastAPI(
        title=settings.get("server.title", "Back API"),
        version=settings.get("server.version", "0.1.0"),
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.get("server.cors.allow_origins", ["*"]),
        allow_credentials=settings.get("server.cors.allow_credentials", True),
        allow_methods=settings.get("server.cors.allow_methods", ["*"]),
        allow_headers=settings.get("server.cors.allow_headers", ["*"]),
    )

    app.include_router(api_router)
    return app


app = create_app()
