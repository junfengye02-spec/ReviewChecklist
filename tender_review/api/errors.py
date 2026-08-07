from __future__ import annotations

import logging
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException

from tender_review.shared.errors import ErrorCategory, ServiceError

from .schemas import ApiError, ErrorDetail, ErrorResponse


_STATUS_BY_CATEGORY = {
    ErrorCategory.RETRYABLE: 503,
    ErrorCategory.PERMANENT: 422,
    ErrorCategory.CANCELLED: 409,
    ErrorCategory.INSUFFICIENT_EVIDENCE: 422,
    ErrorCategory.NOT_FOUND: 404,
    ErrorCategory.CONFLICT: 409,
    ErrorCategory.INVALID_REQUEST: 422,
    ErrorCategory.INTERNAL: 500,
}


def install_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(ServiceError)
    async def service_error_handler(
        request: Request, exc: ServiceError
    ) -> JSONResponse:
        return _response(
            request,
            status_code=_STATUS_BY_CATEGORY[exc.category],
            code=exc.code,
            message=exc.message,
            category=exc.category,
            retryable=exc.retryable,
            details=exc.details,
        )

    @app.exception_handler(RequestValidationError)
    async def validation_error_handler(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        validation_errors = tuple(
            ErrorDetail(
                location=tuple(error.get("loc") or ()),
                message=str(error.get("msg") or "Invalid value"),
                type=str(error.get("type") or ""),
            )
            for error in exc.errors()
        )
        return _response(
            request,
            status_code=422,
            code="request_validation_failed",
            message="Request validation failed",
            category=ErrorCategory.INVALID_REQUEST,
            retryable=False,
            validation_errors=validation_errors,
        )

    @app.exception_handler(HTTPException)
    async def http_error_handler(request: Request, exc: HTTPException) -> JSONResponse:
        category = _http_category(exc.status_code)
        message = exc.detail if isinstance(exc.detail, str) else "HTTP request failed"
        return _response(
            request,
            status_code=exc.status_code,
            code="http_error",
            message=message,
            category=category,
            retryable=exc.status_code >= 500,
        )

    @app.exception_handler(Exception)
    async def unexpected_error_handler(
        request: Request, exc: Exception
    ) -> JSONResponse:
        logging.getLogger("tender_review.api").exception(
            "Unhandled API error",
            extra={"event": "api.unhandled_error", "request_id": _request_id(request)},
        )
        return _response(
            request,
            status_code=500,
            code="internal_error",
            message="An unexpected error occurred",
            category=ErrorCategory.INTERNAL,
            retryable=False,
        )


def _response(
    request: Request,
    *,
    status_code: int,
    code: str,
    message: str,
    category: ErrorCategory,
    retryable: bool,
    details: dict[str, Any] | None = None,
    validation_errors: tuple[ErrorDetail, ...] = (),
) -> JSONResponse:
    body = ErrorResponse(
        error=ApiError(
            code=code,
            message=message,
            category=category.value,
            retryable=retryable,
            request_id=_request_id(request),
            details=details or {},
            validation_errors=validation_errors,
        )
    )
    return JSONResponse(status_code=status_code, content=body.model_dump(mode="json"))


def _request_id(request: Request) -> str:
    return str(getattr(request.state, "request_id", "unknown"))


def _http_category(status_code: int) -> ErrorCategory:
    if status_code == 404:
        return ErrorCategory.NOT_FOUND
    if status_code == 409:
        return ErrorCategory.CONFLICT
    if status_code >= 500:
        return ErrorCategory.INTERNAL
    return ErrorCategory.INVALID_REQUEST
