from __future__ import annotations

import json
import logging
import re
import sys
from datetime import datetime, timezone
from typing import Any, TextIO


_STANDARD_FIELDS = set(logging.makeLogRecord({}).__dict__) | {
    "message",
    "asctime",
}

_CORRELATION_FIELDS = (
    "job_id",
    "thread_id",
    "checkpoint_id",
    "call_id",
    "rule_version",
    "dataset_version",
    "model_config",
)
_SENSITIVE_FIELD_NAMES = frozenset(
    {
        "api_key",
        "authorization",
        "content",
        "excerpt",
        "input_json",
        "llm_api_key",
        "messages",
        "output_json",
        "password",
        "prompt",
        "query",
        "rationale",
        "reason",
        "refresh_token",
        "secret",
        "secret_key",
        "text",
        "access_token",
    }
)
_SECRET_PATTERNS = (
    re.compile(r"(?i)(bearer\s+)[a-z0-9._~+/=-]+"),
    re.compile(r"(?i)((?:api[_-]?key|secret|password)\s*[=:]\s*)[^\s,;]+"),
    re.compile(r"\bsk-[A-Za-z0-9_-]{8,}\b"),
)


def _sanitize_string(value: str) -> str:
    sanitized = value
    for pattern in _SECRET_PATTERNS:
        sanitized = pattern.sub(r"\1<redacted>" if pattern.groups else "<redacted>", sanitized)
    return sanitized


def _is_sensitive_key(key: str) -> bool:
    normalized = key.strip().casefold().replace("-", "_")
    return normalized in _SENSITIVE_FIELD_NAMES or normalized.endswith(
        ("_api_key", "_password", "_secret", "_reason", "_rationale", "_excerpt")
    )


def _sanitize(value: Any, *, key: str = "") -> Any:
    if _is_sensitive_key(key):
        return "<redacted>"
    if isinstance(value, dict):
        return {
            str(child_key): _sanitize(child_value, key=str(child_key))
            for child_key, child_value in value.items()
        }
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_sanitize(item) for item in value]
    if isinstance(value, str):
        return _sanitize_string(value)
    return value


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "logger": record.name,
            "message": _sanitize_string(record.getMessage()),
        }
        for key, value in record.__dict__.items():
            if key not in _STANDARD_FIELDS and not key.startswith("_"):
                payload[key] = _sanitize(value, key=key)
        if "event" in payload:
            for field in _CORRELATION_FIELDS:
                payload.setdefault(field, None)
        if record.exc_info:
            exception_type = record.exc_info[0]
            payload["exception_type"] = (
                exception_type.__name__ if exception_type is not None else "Exception"
            )
            exception = record.exc_info[1]
            code = getattr(exception, "code", None)
            if code:
                payload["error_code"] = str(code)
        return json.dumps(payload, ensure_ascii=False, default=str)


def configure_logging(
    level: str = "INFO",
    *,
    json_output: bool = True,
    stream: TextIO | None = None,
    force: bool = False,
) -> logging.Logger:
    logger = logging.getLogger("tender_review")
    logger.setLevel(level.upper())
    logger.propagate = False
    if force:
        for handler in tuple(logger.handlers):
            logger.removeHandler(handler)
            handler.close()

    formatter: logging.Formatter = (
        JsonFormatter()
        if json_output
        else logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")
    )
    owned_handlers = [
        handler
        for handler in logger.handlers
        if getattr(handler, "_tender_review_handler", False)
    ]
    if not owned_handlers:
        handler = logging.StreamHandler(stream or sys.stdout)
        handler._tender_review_handler = True  # type: ignore[attr-defined]
        logger.addHandler(handler)
        owned_handlers.append(handler)
    for handler in owned_handlers:
        handler.setLevel(level.upper())
        handler.setFormatter(formatter)
    return logger
