from __future__ import annotations

from enum import Enum
from typing import Any


class ErrorCategory(str, Enum):
    RETRYABLE = "retryable"
    PERMANENT = "permanent"
    CANCELLED = "cancelled"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    NOT_FOUND = "not_found"
    CONFLICT = "conflict"
    INVALID_REQUEST = "invalid_request"
    INTERNAL = "internal"


class ServiceError(Exception):
    """Stable application error safe to translate at process boundaries."""

    category = ErrorCategory.PERMANENT
    default_code = "service_error"

    def __init__(
        self,
        message: str,
        *,
        code: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.code = code or self.default_code
        self.details = dict(details or {})

    @property
    def retryable(self) -> bool:
        return self.category is ErrorCategory.RETRYABLE


class RetryableError(ServiceError):
    category = ErrorCategory.RETRYABLE
    default_code = "temporarily_unavailable"


class PermanentError(ServiceError):
    category = ErrorCategory.PERMANENT
    default_code = "operation_rejected"


class CancelledError(ServiceError):
    category = ErrorCategory.CANCELLED
    default_code = "operation_cancelled"


class InsufficientEvidenceError(ServiceError):
    category = ErrorCategory.INSUFFICIENT_EVIDENCE
    default_code = "insufficient_evidence"


class NotFoundError(ServiceError):
    category = ErrorCategory.NOT_FOUND
    default_code = "not_found"


class ConflictError(ServiceError):
    category = ErrorCategory.CONFLICT
    default_code = "conflict"
