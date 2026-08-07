"""OpenAI-compatible adapters for the review and retrieval provider ports."""

from __future__ import annotations

import json
import logging
import math
import time
from collections.abc import Mapping, Sequence
from typing import Any
from urllib.parse import urlsplit

import requests
from pydantic import ValidationError

from tender_review.retrieval.public import EmbeddingRequest, EmbeddingResult
from tender_review.review.public import (
    LlmRequest,
    LlmResponse,
    StructuredExtraction,
)
from tender_review.shared.contracts import CallContext, ensure_call_active
from tender_review.shared.errors import PermanentError, RetryableError
from tender_review.shared.observability import (
    CorrelationContext,
    log_event,
    record_metric,
)


_RETRYABLE_HTTP_STATUSES = frozenset({408, 429, 500, 502, 503, 504})
_STRUCTURED_EXTRACTION_SCHEMA = "StructuredExtraction.v1"


class _OpenAICompatibleTransport:
    def __init__(
        self,
        *,
        api_key: str,
        base_url: str,
        session: requests.Session | None,
    ) -> None:
        if not isinstance(api_key, str) or not api_key.strip():
            raise ValueError("api_key must not be blank")
        self.base_url = _validate_base_url(base_url)
        self._api_key = api_key
        self._session = session or requests.Session()

    def __repr__(self) -> str:
        return f"{type(self).__name__}(base_url={self.base_url!r}, api_key=<redacted>)"

    def post_json(
        self,
        *,
        path: str,
        payload: Mapping[str, Any],
        request_id: str,
        timeout_seconds: float,
        service: str,
        attempt: int,
    ) -> Mapping[str, Any]:
        details: dict[str, Any] = {
            "call_id": request_id,
            "request_id": request_id,
            "attempt": attempt,
        }
        try:
            response = self._session.post(
                f"{self.base_url}/{path.lstrip('/')}",
                headers={
                    "Authorization": f"Bearer {self._api_key}",
                    "Content-Type": "application/json",
                    "X-Request-ID": request_id,
                },
                json=dict(payload),
                timeout=timeout_seconds,
            )
        except requests.Timeout:
            raise RetryableError(
                f"{service} request timed out",
                code=f"{service}_timeout",
                details=details,
            ) from None
        except requests.RequestException:
            raise RetryableError(
                f"{service} transport request failed",
                code=f"{service}_transport_error",
                details=details,
            ) from None

        status_code = int(response.status_code)
        if not 200 <= status_code < 300:
            details["status_code"] = status_code
            if status_code in _RETRYABLE_HTTP_STATUSES:
                code = (
                    f"{service}_rate_limited"
                    if status_code == 429
                    else f"{service}_timeout"
                    if status_code == 408
                    else f"{service}_upstream_unavailable"
                )
                raise RetryableError(
                    f"{service} provider returned HTTP {status_code}",
                    code=code,
                    details=details,
                )
            raise PermanentError(
                f"{service} provider rejected the request with HTTP {status_code}",
                code=f"{service}_request_rejected",
                details=details,
            )

        try:
            body = response.json()
        except (requests.JSONDecodeError, ValueError):
            raise RetryableError(
                f"{service} provider returned invalid response JSON",
                code=f"{service}_invalid_response_json",
                details=details,
            ) from None
        if not isinstance(body, Mapping):
            raise RetryableError(
                f"{service} provider returned an invalid response schema",
                code=f"{service}_response_schema_mismatch",
                details=details,
            )
        return body


class OpenAICompatibleLlmProvider:
    """Synchronous chat-completions adapter implementing ``LlmProvider``.

    Constructor timeout and attempt values are hard upper bounds. A call may
    request stricter limits through ``CallContext``. When ``temperature`` is
    omitted, the per-request temperature is sent unchanged.
    """

    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        base_url: str,
        timeout_seconds: float = 30.0,
        max_attempts: int = 3,
        temperature: float | None = None,
        session: requests.Session | None = None,
    ) -> None:
        self.model = _validate_non_blank(model, name="model")
        self.timeout_seconds = _validate_positive_number(
            timeout_seconds, name="timeout_seconds"
        )
        self.max_attempts = _validate_positive_integer(
            max_attempts, name="max_attempts"
        )
        self.temperature = (
            None
            if temperature is None
            else _validate_temperature(temperature, name="temperature")
        )
        self._transport = _OpenAICompatibleTransport(
            api_key=api_key,
            base_url=base_url,
            session=session,
        )

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(model={self.model!r}, "
            f"base_url={self._transport.base_url!r}, "
            f"timeout_seconds={self.timeout_seconds!r}, "
            f"max_attempts={self.max_attempts!r}, "
            f"temperature={self.temperature!r}, api_key=<redacted>)"
        )

    def complete(self, request: LlmRequest) -> LlmResponse:
        request = LlmRequest.model_validate(request)
        timeout_seconds, max_attempts = _effective_limits(
            request.call,
            timeout_seconds=self.timeout_seconds,
            max_attempts=self.max_attempts,
        )
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": message.role, "content": message.content}
                for message in request.messages
            ],
            "temperature": (
                request.temperature if self.temperature is None else self.temperature
            ),
        }
        if request.response_schema_name == _STRUCTURED_EXTRACTION_SCHEMA:
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "StructuredExtraction_v1",
                    "strict": True,
                    "schema": StructuredExtraction.model_json_schema(),
                },
            }
        elif request.response_schema_name is not None:
            payload["response_format"] = {"type": "json_object"}

        last_error: RetryableError | None = None
        logger = logging.getLogger("tender_review.model")
        context = _provider_context(request.call.call_id, self.model)
        for attempt in range(1, max_attempts + 1):
            ensure_call_active(request.call)
            started = time.perf_counter()
            try:
                body = self._transport.post_json(
                    path="chat/completions",
                    payload=payload,
                    request_id=request.call.call_id,
                    timeout_seconds=timeout_seconds,
                    service="model",
                    attempt=attempt,
                )
                response = self._parse_response(
                    body,
                    request=request,
                    attempt=attempt,
                )
            except RetryableError as exc:
                last_error = exc
                duration_ms = max(0.0, (time.perf_counter() - started) * 1000.0)
                log_event(
                    logger,
                    logging.WARNING,
                    event="model.call_attempt",
                    message="Model call attempt failed",
                    context=context,
                    model_name=self.model,
                    attempt=attempt,
                    outcome="retryable_error",
                    error_code=exc.code,
                    retryable=attempt < max_attempts,
                )
                record_metric(
                    logger,
                    name="model_call_duration",
                    value=duration_ms,
                    unit="ms",
                    source="process_monotonic",
                    context=context,
                    model_name=self.model,
                    attempt=attempt,
                    outcome="retryable_error",
                )
                if attempt < max_attempts:
                    record_metric(
                        logger,
                        name="model_retry",
                        value=1,
                        unit="count",
                        source="adapter_retry_policy",
                        context=context,
                        model_name=self.model,
                        attempt=attempt,
                        error_code=exc.code,
                    )
                continue
            except PermanentError as exc:
                duration_ms = max(0.0, (time.perf_counter() - started) * 1000.0)
                log_event(
                    logger,
                    logging.WARNING,
                    event="model.call_attempt",
                    message="Model call attempt was permanently rejected",
                    context=context,
                    model_name=self.model,
                    attempt=attempt,
                    outcome="permanent_error",
                    error_code=exc.code,
                    retryable=False,
                )
                record_metric(
                    logger,
                    name="model_call_duration",
                    value=duration_ms,
                    unit="ms",
                    source="process_monotonic",
                    context=context,
                    model_name=self.model,
                    attempt=attempt,
                    outcome="permanent_error",
                )
                raise
            duration_ms = max(0.0, (time.perf_counter() - started) * 1000.0)
            log_event(
                logger,
                logging.INFO,
                event="model.call_attempt",
                message="Model call attempt completed",
                context=context,
                model_name=response.model,
                attempt=attempt,
                outcome="success",
            )
            record_metric(
                logger,
                name="model_call_duration",
                value=duration_ms,
                unit="ms",
                source="process_monotonic",
                context=context,
                model_name=response.model,
                attempt=attempt,
                outcome="success",
            )
            for name, value in (
                ("model_prompt_tokens", response.prompt_tokens),
                ("model_completion_tokens", response.completion_tokens),
            ):
                record_metric(
                    logger,
                    name=name,
                    value=value,
                    unit="tokens",
                    source="provider_response",
                    context=context,
                    model_name=response.model,
                    attempt=attempt,
                )
            record_metric(
                logger,
                name="model_cost",
                value=None,
                unit="currency",
                source="pricing_configuration",
                status="not_collected",
                context=context,
                model_name=response.model,
                attempt=attempt,
            )
            return response
        if last_error is None:  # pragma: no cover - validated max_attempts is positive
            raise AssertionError("max_attempts must be positive")
        raise last_error

    def _parse_response(
        self,
        body: Mapping[str, Any],
        *,
        request: LlmRequest,
        attempt: int,
    ) -> LlmResponse:
        details = {
            "call_id": request.call.call_id,
            "request_id": request.call.call_id,
            "attempt": attempt,
        }
        choices = body.get("choices")
        if not isinstance(choices, list) or not choices:
            raise _retryable_schema_error("model", details)
        choice = choices[0]
        if not isinstance(choice, Mapping):
            raise _retryable_schema_error("model", details)
        message = choice.get("message")
        if not isinstance(message, Mapping):
            raise _retryable_schema_error("model", details)
        content = message.get("content")
        if not isinstance(content, str):
            raise _retryable_schema_error("model", details)
        if not content.strip():
            raise RetryableError(
                "model provider returned empty output",
                code="model_empty_output",
                details=details,
            )

        if request.response_schema_name is not None:
            try:
                decoded = json.loads(content)
            except (json.JSONDecodeError, TypeError):
                raise RetryableError(
                    "model provider returned invalid output JSON",
                    code="model_invalid_output_json",
                    details=details,
                ) from None
            if not isinstance(decoded, Mapping):
                raise RetryableError(
                    "model output did not match the requested schema",
                    code="model_schema_mismatch",
                    details=details,
                )
            if request.response_schema_name == _STRUCTURED_EXTRACTION_SCHEMA:
                try:
                    StructuredExtraction.model_validate(decoded)
                except (ValidationError, ValueError, TypeError):
                    raise RetryableError(
                        "model output did not match StructuredExtraction.v1",
                        code="model_schema_mismatch",
                        details=details,
                    ) from None

        usage = body.get("usage", {})
        if not isinstance(usage, Mapping):
            raise _retryable_schema_error("model", details)
        prompt_tokens = _non_negative_integer_or_error(
            usage.get("prompt_tokens", 0), service="model", details=details
        )
        completion_tokens = _non_negative_integer_or_error(
            usage.get("completion_tokens", 0), service="model", details=details
        )
        response_model = body.get("model", self.model)
        if not isinstance(response_model, str) or not response_model.strip():
            raise _retryable_schema_error("model", details)
        finish_reason = choice.get("finish_reason", "stop")
        if finish_reason is None:
            finish_reason = "unknown"
        if not isinstance(finish_reason, str):
            raise _retryable_schema_error("model", details)
        return LlmResponse(
            model=response_model,
            content=content,
            finish_reason=finish_reason,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )


class OpenAICompatibleEmbeddingProvider:
    """Batched embeddings adapter implementing ``EmbeddingProvider``."""

    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        dimensions: int,
        batch_size: int,
        base_url: str,
        timeout_seconds: float = 30.0,
        max_attempts: int = 3,
        session: requests.Session | None = None,
    ) -> None:
        self.model = _validate_non_blank(model, name="model")
        self.dimensions = _validate_positive_integer(dimensions, name="dimensions")
        self.batch_size = _validate_positive_integer(batch_size, name="batch_size")
        self.timeout_seconds = _validate_positive_number(
            timeout_seconds, name="timeout_seconds"
        )
        self.max_attempts = _validate_positive_integer(
            max_attempts, name="max_attempts"
        )
        self._transport = _OpenAICompatibleTransport(
            api_key=api_key,
            base_url=base_url,
            session=session,
        )

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(model={self.model!r}, "
            f"dimensions={self.dimensions!r}, batch_size={self.batch_size!r}, "
            f"base_url={self._transport.base_url!r}, "
            f"timeout_seconds={self.timeout_seconds!r}, "
            f"max_attempts={self.max_attempts!r}, api_key=<redacted>)"
        )

    def embed(self, request: EmbeddingRequest) -> EmbeddingResult:
        request = EmbeddingRequest.model_validate(request)
        timeout_seconds, max_attempts = _effective_limits(
            request.call,
            timeout_seconds=self.timeout_seconds,
            max_attempts=self.max_attempts,
        )
        vectors: list[tuple[float, ...]] = []
        batches = tuple(_batched(request.texts, self.batch_size))
        for batch_number, texts in enumerate(batches, start=1):
            request_id = _batch_request_id(
                request.call.call_id,
                batch_number=batch_number,
                batch_count=len(batches),
            )
            vectors.extend(
                self._embed_batch(
                    texts,
                    call=request.call,
                    request_id=request_id,
                    timeout_seconds=timeout_seconds,
                    max_attempts=max_attempts,
                )
            )
        return EmbeddingResult(
            model=self.model,
            dimensions=self.dimensions,
            vectors=tuple(vectors),
        )

    def _embed_batch(
        self,
        texts: tuple[str, ...],
        *,
        call: CallContext,
        request_id: str,
        timeout_seconds: float,
        max_attempts: int,
    ) -> tuple[tuple[float, ...], ...]:
        payload = {
            "model": self.model,
            "input": list(texts),
            "dimensions": self.dimensions,
        }
        last_error: RetryableError | None = None
        logger = logging.getLogger("tender_review.model.embedding")
        context = _provider_context(call.call_id, self.model)
        for attempt in range(1, max_attempts + 1):
            ensure_call_active(call)
            started = time.perf_counter()
            try:
                body = self._transport.post_json(
                    path="embeddings",
                    payload=payload,
                    request_id=request_id,
                    timeout_seconds=timeout_seconds,
                    service="embedding",
                    attempt=attempt,
                )
                vectors = self._parse_batch(
                    body,
                    expected_count=len(texts),
                    call_id=call.call_id,
                    request_id=request_id,
                    attempt=attempt,
                )
            except RetryableError as exc:
                last_error = exc
                outcome = "retryable_error"
                error_code = exc.code
            except PermanentError as exc:
                duration_ms = max(0.0, (time.perf_counter() - started) * 1000.0)
                log_event(
                    logger,
                    logging.WARNING,
                    event="embedding.call_attempt",
                    message="Embedding call attempt was permanently rejected",
                    context=context,
                    call_id=request_id,
                    model_name=self.model,
                    attempt=attempt,
                    outcome="permanent_error",
                    error_code=exc.code,
                    retryable=False,
                    batch_size=len(texts),
                )
                record_metric(
                    logger,
                    name="embedding_call_duration",
                    value=duration_ms,
                    unit="ms",
                    source="process_monotonic",
                    context=context,
                    call_id=request_id,
                    model_name=self.model,
                    attempt=attempt,
                    outcome="permanent_error",
                    batch_size=len(texts),
                )
                raise
            else:
                outcome = "success"
                error_code = None
            duration_ms = max(0.0, (time.perf_counter() - started) * 1000.0)
            log_event(
                logger,
                logging.INFO if outcome == "success" else logging.WARNING,
                event="embedding.call_attempt",
                message=(
                    "Embedding call attempt completed"
                    if outcome == "success"
                    else "Embedding call attempt failed"
                ),
                context=context,
                call_id=request_id,
                model_name=self.model,
                attempt=attempt,
                outcome=outcome,
                error_code=error_code,
                batch_size=len(texts),
            )
            record_metric(
                logger,
                name="embedding_call_duration",
                value=duration_ms,
                unit="ms",
                source="process_monotonic",
                context=context,
                call_id=request_id,
                model_name=self.model,
                attempt=attempt,
                outcome=outcome,
                batch_size=len(texts),
            )
            if outcome == "success":
                return vectors
            if attempt < max_attempts:
                record_metric(
                    logger,
                    name="embedding_retry",
                    value=1,
                    unit="count",
                    source="adapter_retry_policy",
                    context=context,
                    call_id=request_id,
                    model_name=self.model,
                    attempt=attempt,
                    error_code=error_code,
                )
        if last_error is None:  # pragma: no cover - validated max_attempts is positive
            raise AssertionError("max_attempts must be positive")
        raise last_error

    def _parse_batch(
        self,
        body: Mapping[str, Any],
        *,
        expected_count: int,
        call_id: str,
        request_id: str,
        attempt: int,
    ) -> tuple[tuple[float, ...], ...]:
        details = {
            "call_id": call_id,
            "request_id": request_id,
            "attempt": attempt,
        }
        data = body.get("data")
        if not isinstance(data, list):
            raise _retryable_schema_error("embedding", details)
        if not data:
            raise RetryableError(
                "embedding provider returned empty output",
                code="embedding_empty_output",
                details=details,
            )
        if len(data) != expected_count:
            raise _retryable_schema_error("embedding", details)

        indexed: dict[int, tuple[float, ...]] = {}
        for item in data:
            if not isinstance(item, Mapping):
                raise _retryable_schema_error("embedding", details)
            index = item.get("index")
            raw_vector = item.get("embedding")
            if (
                isinstance(index, bool)
                or not isinstance(index, int)
                or index < 0
                or index >= expected_count
                or index in indexed
                or not isinstance(raw_vector, Sequence)
                or isinstance(raw_vector, (str, bytes, bytearray))
            ):
                raise _retryable_schema_error("embedding", details)
            vector = _validate_vector(
                raw_vector,
                dimensions=self.dimensions,
                details=details,
            )
            indexed[index] = vector
        if set(indexed) != set(range(expected_count)):
            raise _retryable_schema_error("embedding", details)
        return tuple(indexed[index] for index in range(expected_count))


def _retryable_schema_error(service: str, details: Mapping[str, Any]) -> RetryableError:
    return RetryableError(
        f"{service} provider returned an invalid response schema",
        code=f"{service}_response_schema_mismatch",
        details=dict(details),
    )


def _validate_vector(
    raw_vector: Sequence[Any],
    *,
    dimensions: int,
    details: Mapping[str, Any],
) -> tuple[float, ...]:
    if len(raw_vector) != dimensions:
        raise RetryableError(
            "embedding vector dimensions did not match the configured dimensions",
            code="embedding_dimension_mismatch",
            details=dict(details),
        )
    vector: list[float] = []
    for value in raw_vector:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise _retryable_schema_error("embedding", details)
        numeric = float(value)
        if not math.isfinite(numeric):
            raise _retryable_schema_error("embedding", details)
        vector.append(numeric)
    return tuple(vector)


def _non_negative_integer_or_error(
    value: Any,
    *,
    service: str,
    details: Mapping[str, Any],
) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise _retryable_schema_error(service, details)
    return value


def _effective_limits(
    call: CallContext,
    *,
    timeout_seconds: float,
    max_attempts: int,
) -> tuple[float, int]:
    return min(timeout_seconds, call.timeout_seconds), min(
        max_attempts, call.max_attempts
    )


def _batch_request_id(call_id: str, *, batch_number: int, batch_count: int) -> str:
    if batch_count == 1:
        return call_id
    suffix = f":batch:{batch_number}"
    return f"{call_id[: 128 - len(suffix)]}{suffix}"


def _batched(values: tuple[str, ...], batch_size: int) -> Sequence[tuple[str, ...]]:
    return tuple(
        values[offset : offset + batch_size]
        for offset in range(0, len(values), batch_size)
    )


def _provider_context(call_id: str, model: str) -> CorrelationContext:
    job_id = None
    if call_id.startswith("review:"):
        job_id = call_id.removeprefix("review:").split(":", 1)[0]
    elif call_id.startswith("document-parse:"):
        job_id = call_id.removeprefix("document-parse:").split(":", 1)[0]
    return CorrelationContext(
        job_id=job_id,
        thread_id=job_id,
        call_id=call_id,
        model_config=model,
    )


def _validate_base_url(value: str) -> str:
    value = _validate_non_blank(value, name="base_url").rstrip("/")
    parsed = urlsplit(value)
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.netloc
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError(
            "base_url must be an HTTP(S) URL without credentials, query, or fragment"
        )
    return value


def _validate_non_blank(value: str, *, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must not be blank")
    return value


def _validate_positive_number(value: float, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a positive finite number")
    numeric = float(value)
    if not math.isfinite(numeric) or numeric <= 0:
        raise ValueError(f"{name} must be a positive finite number")
    return numeric


def _validate_positive_integer(value: int, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _validate_temperature(value: float, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be between 0 and 2")
    numeric = float(value)
    if not math.isfinite(numeric) or not 0 <= numeric <= 2:
        raise ValueError(f"{name} must be between 0 and 2")
    return numeric
