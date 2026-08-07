from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI, Request, Response

from tender_review.shared.logging import configure_logging

from .dependencies import ApiContainer
from .errors import install_error_handlers
from .routes import build_health_router, build_v1_router


def create_app(container: ApiContainer) -> FastAPI:
    configure_logging(
        container.settings.log_level,
        json_output=container.settings.log_json,
    )
    app = FastAPI(
        title="Tender Review API",
        version=container.settings.version,
        description="Evidence-driven tender review application API",
        lifespan=_lifespan(container),
    )
    app.state.container = container
    install_error_handlers(app)
    app.include_router(build_health_router(container))
    app.include_router(build_v1_router(container))
    _install_request_logging(app, container)
    return app


def _lifespan(container: ApiContainer):
    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        try:
            yield
        finally:
            close = getattr(container, "close", None)
            if close is not None:
                close()

    return lifespan


def _install_request_logging(app: FastAPI, container: ApiContainer) -> None:
    logger = logging.getLogger("tender_review.api")

    @app.middleware("http")
    async def request_context(request: Request, call_next) -> Response:
        request_id = _incoming_request_id(request) or container.ids.new()
        request.state.request_id = request_id
        started = time.perf_counter()
        response: Response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        logger.info(
            "HTTP request completed",
            extra={
                "event": "http.request",
                "request_id": request_id,
                "method": request.method,
                "path": request.url.path,
                "status_code": response.status_code,
                "duration_ms": round((time.perf_counter() - started) * 1000, 3),
            },
        )
        return response


def _incoming_request_id(request: Request) -> str | None:
    value = request.headers.get("X-Request-ID", "").strip()
    if not value or len(value) > 128 or any(ord(character) < 32 for character in value):
        return None
    return value
