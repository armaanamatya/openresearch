"""FastAPI application factory."""

from __future__ import annotations

from fastapi import FastAPI

from backend import __version__
from backend.config import get_settings


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    settings = get_settings()

    app = FastAPI(
        title="ReproLab Agent",
        version=__version__,
        debug=settings.debug,
    )

    @app.get("/health")
    async def health() -> dict:
        return {"status": "ok", "version": __version__}

    return app
