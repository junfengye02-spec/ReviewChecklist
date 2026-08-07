from __future__ import annotations

import unittest
from collections import deque
from typing import Any

import requests

from tender_review.infrastructure.ai import (
    OpenAICompatibleEmbeddingProvider,
    OpenAICompatibleLlmProvider,
)
from tender_review.retrieval.public import (
    EmbeddingProvider,
    EmbeddingRequest,
)
from tender_review.review.public import (
    LlmMessage,
    LlmProvider,
    LlmRequest,
)
from tender_review.shared.contracts import CallContext
from tender_review.shared.errors import ErrorCategory, PermanentError, RetryableError


API_KEY = "test-only-secret-key"


class StubResponse:
    def __init__(
        self,
        status_code: int = 200,
        payload: Any = None,
        *,
        json_error: BaseException | None = None,
    ) -> None:
        self.status_code = status_code
        self.payload = payload
        self.json_error = json_error

    def json(self) -> Any:
        if self.json_error is not None:
            raise self.json_error
        return self.payload


class StubSession:
    def __init__(self, outcomes: tuple[StubResponse | BaseException, ...]) -> None:
        self.outcomes = deque(outcomes)
        self.calls: list[dict[str, Any]] = []

    def post(self, url: str, **kwargs: Any) -> StubResponse:
        self.calls.append({"url": url, **kwargs})
        outcome = self.outcomes.popleft()
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


def structured_extraction_json() -> str:
    return (
        '{"schema_version":1,"review_item_id":"authorization-letter",'
        '"fields":[{"schema_version":1,"field_name":"authorization_text",'
        '"value_type":"text","value":"signed authorization",'
        '"sources":[{"schema_version":1,"source_id":"source-1",'
        '"document_id":"document-1","chunk_id":"chunk-1",'
        '"page_number":7,"section_path":["Authorization"],'
        '"excerpt":"signed authorization"}]}]}'
    )


def chat_response(
    content: str | None = None,
    *,
    prompt_tokens: int = 11,
    completion_tokens: int = 7,
) -> StubResponse:
    return StubResponse(
        payload={
            "model": "served-model-revision",
            "choices": [
                {
                    "message": {
                        "content": (
                            structured_extraction_json() if content is None else content
                        )
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
        }
    )


def llm_request(
    *,
    call_id: str = "review-call-1",
    timeout_seconds: float = 5.0,
    max_attempts: int = 1,
    temperature: float = 0.35,
) -> LlmRequest:
    return LlmRequest(
        messages=(
            LlmMessage(role="system", content="Return strict JSON."),
            LlmMessage(role="user", content="Extract the authorization."),
        ),
        response_schema_name="StructuredExtraction.v1",
        temperature=temperature,
        call=CallContext(
            call_id=call_id,
            timeout_seconds=timeout_seconds,
            max_attempts=max_attempts,
        ),
    )


class OpenAICompatibleLlmProviderTests(unittest.TestCase):
    def test_permanent_rejection_emits_a_safe_correlated_attempt(self) -> None:
        session = StubSession((StubResponse(400),))
        provider = OpenAICompatibleLlmProvider(
            api_key=API_KEY,
            model="chat-model",
            base_url="https://models.example.test/v1",
            session=session,  # type: ignore[arg-type]
        )

        with self.assertLogs("tender_review.model", level="WARNING") as captured:
            with self.assertRaises(PermanentError):
                provider.complete(llm_request(call_id="review:job-observed"))

        attempt = next(
            record
            for record in captured.records
            if getattr(record, "event", None) == "model.call_attempt"
        )
        self.assertEqual(attempt.job_id, "job-observed")
        self.assertEqual(attempt.thread_id, "job-observed")
        self.assertEqual(attempt.call_id, "review:job-observed")
        self.assertEqual(attempt.model_config, "chat-model")
        self.assertEqual(attempt.outcome, "permanent_error")
        self.assertFalse(attempt.retryable)
        self.assertNotIn(API_KEY, attempt.getMessage())

    def test_sends_openai_chat_request_and_returns_model_usage(self) -> None:
        session = StubSession((chat_response(),))
        provider = OpenAICompatibleLlmProvider(
            api_key=API_KEY,
            model="configured-chat-model",
            base_url="https://models.example.test/v1/",
            timeout_seconds=9.0,
            max_attempts=3,
            session=session,  # type: ignore[arg-type]
        )

        response = provider.complete(llm_request())

        self.assertIsInstance(provider, LlmProvider)
        self.assertEqual(response.model, "served-model-revision")
        self.assertEqual(response.prompt_tokens, 11)
        self.assertEqual(response.completion_tokens, 7)
        self.assertEqual(response.finish_reason, "stop")
        call = session.calls[0]
        self.assertEqual(call["url"], "https://models.example.test/v1/chat/completions")
        self.assertEqual(call["timeout"], 5.0)
        self.assertEqual(call["headers"]["X-Request-ID"], "review-call-1")
        self.assertEqual(call["headers"]["Authorization"], f"Bearer {API_KEY}")
        self.assertEqual(call["json"]["model"], "configured-chat-model")
        self.assertEqual(call["json"]["temperature"], 0.35)
        self.assertEqual(
            call["json"]["response_format"]["json_schema"]["name"],
            "StructuredExtraction_v1",
        )
        self.assertTrue(call["json"]["response_format"]["json_schema"]["strict"])

    def test_configured_temperature_pins_the_request_value(self) -> None:
        session = StubSession((chat_response(),))
        provider = OpenAICompatibleLlmProvider(
            api_key=API_KEY,
            model="chat-model",
            base_url="http://127.0.0.1:8080/v1",
            temperature=0.1,
            session=session,  # type: ignore[arg-type]
        )

        provider.complete(llm_request(temperature=1.2))

        self.assertEqual(session.calls[0]["json"]["temperature"], 0.1)

    def test_429_retries_within_bounded_attempts(self) -> None:
        session = StubSession((StubResponse(429), chat_response()))
        provider = OpenAICompatibleLlmProvider(
            api_key=API_KEY,
            model="chat-model",
            base_url="https://models.example.test/v1",
            max_attempts=4,
            session=session,  # type: ignore[arg-type]
        )

        response = provider.complete(llm_request(max_attempts=2))

        self.assertEqual(response.prompt_tokens, 11)
        self.assertEqual(len(session.calls), 2)
        self.assertEqual(
            [call["headers"]["X-Request-ID"] for call in session.calls],
            ["review-call-1", "review-call-1"],
        )

    def test_exhausted_429_maps_to_retryable_rate_limit(self) -> None:
        session = StubSession((StubResponse(429),))
        provider = OpenAICompatibleLlmProvider(
            api_key=API_KEY,
            model="chat-model",
            base_url="https://models.example.test/v1",
            max_attempts=1,
            session=session,  # type: ignore[arg-type]
        )

        with self.assertRaises(RetryableError) as raised:
            provider.complete(llm_request())

        self.assertEqual(raised.exception.code, "model_rate_limited")
        self.assertEqual(raised.exception.category, ErrorCategory.RETRYABLE)

    def test_timeout_uses_provider_attempt_cap_and_maps_to_retryable(self) -> None:
        session = StubSession(
            (
                requests.Timeout(f"timeout {API_KEY}"),
                requests.Timeout(f"timeout {API_KEY}"),
            )
        )
        provider = OpenAICompatibleLlmProvider(
            api_key=API_KEY,
            model="chat-model",
            base_url="https://models.example.test/v1",
            timeout_seconds=2.5,
            max_attempts=2,
            session=session,  # type: ignore[arg-type]
        )

        with self.assertRaises(RetryableError) as raised:
            provider.complete(llm_request(timeout_seconds=20.0, max_attempts=5))

        self.assertEqual(raised.exception.category, ErrorCategory.RETRYABLE)
        self.assertEqual(raised.exception.code, "model_timeout")
        self.assertEqual(raised.exception.details["call_id"], "review-call-1")
        self.assertEqual(len(session.calls), 2)
        self.assertEqual([call["timeout"] for call in session.calls], [2.5, 2.5])
        self.assertNotIn(API_KEY, repr(raised.exception))
        self.assertNotIn(API_KEY, str(raised.exception.details))

    def test_invalid_json_empty_output_and_schema_mismatch_are_typed(self) -> None:
        cases = (
            (
                StubResponse(json_error=ValueError(f"bad JSON {API_KEY}")),
                "model_invalid_response_json",
            ),
            (chat_response("not-json"), "model_invalid_output_json"),
            (chat_response("   "), "model_empty_output"),
            (
                chat_response(
                    '{"schema_version":1,"review_item_id":"item","fields":[]}'
                ),
                "model_schema_mismatch",
            ),
        )
        for outcome, expected_code in cases:
            with self.subTest(expected_code=expected_code):
                session = StubSession((outcome,))
                provider = OpenAICompatibleLlmProvider(
                    api_key=API_KEY,
                    model="chat-model",
                    base_url="https://models.example.test/v1",
                    max_attempts=1,
                    session=session,  # type: ignore[arg-type]
                )

                with self.assertRaises(RetryableError) as raised:
                    provider.complete(llm_request())

                self.assertEqual(raised.exception.code, expected_code)
                self.assertTrue(raised.exception.retryable)
                self.assertNotIn(API_KEY, repr(raised.exception))

    def test_non_retryable_http_rejection_is_permanent_and_not_retried(self) -> None:
        session = StubSession((StubResponse(400), chat_response()))
        provider = OpenAICompatibleLlmProvider(
            api_key=API_KEY,
            model="chat-model",
            base_url="https://models.example.test/v1",
            max_attempts=2,
            session=session,  # type: ignore[arg-type]
        )

        with self.assertRaises(PermanentError) as raised:
            provider.complete(llm_request(max_attempts=2))

        self.assertEqual(raised.exception.category, ErrorCategory.PERMANENT)
        self.assertEqual(raised.exception.code, "model_request_rejected")
        self.assertEqual(raised.exception.details["status_code"], 400)
        self.assertEqual(len(session.calls), 1)

    def test_api_key_is_absent_from_repr_response_and_error_metadata(self) -> None:
        success_session = StubSession((chat_response(),))
        provider = OpenAICompatibleLlmProvider(
            api_key=API_KEY,
            model="chat-model",
            base_url="https://models.example.test/v1",
            session=success_session,  # type: ignore[arg-type]
        )

        response = provider.complete(llm_request())

        self.assertNotIn(API_KEY, repr(provider))
        self.assertNotIn(API_KEY, repr(provider.__dict__))
        self.assertNotIn(API_KEY, response.model_dump_json())


def embedding_response(*vectors: tuple[float, ...]) -> StubResponse:
    return StubResponse(
        payload={
            "model": "embedding-model",
            "data": [
                {"index": index, "embedding": list(vector)}
                for index, vector in enumerate(vectors)
            ],
            "usage": {"prompt_tokens": len(vectors), "total_tokens": len(vectors)},
        }
    )


def embedding_request(
    texts: tuple[str, ...] = ("first", "second", "third"),
    *,
    max_attempts: int = 1,
) -> EmbeddingRequest:
    return EmbeddingRequest(
        texts=texts,
        call=CallContext(
            call_id="embedding-call-1",
            timeout_seconds=4.0,
            max_attempts=max_attempts,
        ),
    )


class OpenAICompatibleEmbeddingProviderTests(unittest.TestCase):
    def test_permanent_rejection_emits_a_safe_correlated_attempt(self) -> None:
        session = StubSession((StubResponse(400),))
        provider = OpenAICompatibleEmbeddingProvider(
            api_key=API_KEY,
            model="embedding-model",
            dimensions=3,
            batch_size=8,
            base_url="https://models.example.test/v1",
            session=session,  # type: ignore[arg-type]
        )

        with self.assertLogs(
            "tender_review.model.embedding", level="WARNING"
        ) as captured:
            with self.assertRaises(PermanentError):
                provider.embed(embedding_request(("sensitive source text",)))

        attempt = next(
            record
            for record in captured.records
            if getattr(record, "event", None) == "embedding.call_attempt"
        )
        self.assertEqual(attempt.call_id, "embedding-call-1")
        self.assertEqual(attempt.model_config, "embedding-model")
        self.assertEqual(attempt.outcome, "permanent_error")
        self.assertFalse(attempt.retryable)
        self.assertNotIn("sensitive source text", attempt.getMessage())

    def test_batches_inputs_preserves_order_and_enforces_dimensions(self) -> None:
        session = StubSession(
            (
                StubResponse(
                    payload={
                        "data": [
                            {"index": 1, "embedding": [0.0, 1.0, 0.0]},
                            {"index": 0, "embedding": [1.0, 0.0, 0.0]},
                        ]
                    }
                ),
                embedding_response((0.0, 0.0, 1.0)),
            )
        )
        provider = OpenAICompatibleEmbeddingProvider(
            api_key=API_KEY,
            model="embedding-model",
            dimensions=3,
            batch_size=2,
            base_url="https://models.example.test/v1/",
            timeout_seconds=8.0,
            session=session,  # type: ignore[arg-type]
        )

        result = provider.embed(embedding_request())

        self.assertIsInstance(provider, EmbeddingProvider)
        self.assertEqual(result.model, "embedding-model")
        self.assertEqual(result.dimensions, 3)
        self.assertEqual(
            result.vectors,
            ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
        )
        self.assertEqual(
            [call["json"]["input"] for call in session.calls],
            [["first", "second"], ["third"]],
        )
        self.assertEqual([call["json"]["dimensions"] for call in session.calls], [3, 3])
        self.assertEqual(
            [call["headers"]["X-Request-ID"] for call in session.calls],
            ["embedding-call-1:batch:1", "embedding-call-1:batch:2"],
        )
        self.assertEqual([call["timeout"] for call in session.calls], [4.0, 4.0])

    def test_embedding_429_retries_only_the_current_batch(self) -> None:
        session = StubSession((StubResponse(429), embedding_response((1.0, 0.0, 0.0))))
        provider = OpenAICompatibleEmbeddingProvider(
            api_key=API_KEY,
            model="embedding-model",
            dimensions=3,
            batch_size=8,
            base_url="https://models.example.test/v1",
            max_attempts=3,
            session=session,  # type: ignore[arg-type]
        )

        result = provider.embed(embedding_request(("only",), max_attempts=2))

        self.assertEqual(result.vectors, ((1.0, 0.0, 0.0),))
        self.assertEqual(len(session.calls), 2)
        self.assertEqual(
            [call["headers"]["X-Request-ID"] for call in session.calls],
            ["embedding-call-1", "embedding-call-1"],
        )

    def test_exhausted_embedding_429_maps_to_retryable_rate_limit(self) -> None:
        session = StubSession((StubResponse(429),))
        provider = OpenAICompatibleEmbeddingProvider(
            api_key=API_KEY,
            model="embedding-model",
            dimensions=3,
            batch_size=8,
            base_url="https://models.example.test/v1",
            max_attempts=1,
            session=session,  # type: ignore[arg-type]
        )

        with self.assertRaises(RetryableError) as raised:
            provider.embed(embedding_request(("only",)))

        self.assertEqual(raised.exception.code, "embedding_rate_limited")
        self.assertEqual(raised.exception.category, ErrorCategory.RETRYABLE)

    def test_invalid_json_empty_and_dimension_mismatch_are_typed(self) -> None:
        cases = (
            (
                StubResponse(json_error=ValueError(f"bad JSON {API_KEY}")),
                "embedding_invalid_response_json",
            ),
            (StubResponse(payload={"data": []}), "embedding_empty_output"),
            (
                embedding_response((1.0, 0.0)),
                "embedding_dimension_mismatch",
            ),
        )
        for outcome, expected_code in cases:
            with self.subTest(expected_code=expected_code):
                session = StubSession((outcome,))
                provider = OpenAICompatibleEmbeddingProvider(
                    api_key=API_KEY,
                    model="embedding-model",
                    dimensions=3,
                    batch_size=8,
                    base_url="https://models.example.test/v1",
                    max_attempts=1,
                    session=session,  # type: ignore[arg-type]
                )

                with self.assertRaises(RetryableError) as raised:
                    provider.embed(embedding_request(("only",)))

                self.assertEqual(raised.exception.code, expected_code)
                self.assertTrue(raised.exception.retryable)
                self.assertNotIn(API_KEY, repr(raised.exception))

    def test_embedding_timeout_and_repr_do_not_expose_api_key(self) -> None:
        session = StubSession((requests.Timeout(f"timeout {API_KEY}"),))
        provider = OpenAICompatibleEmbeddingProvider(
            api_key=API_KEY,
            model="embedding-model",
            dimensions=3,
            batch_size=8,
            base_url="https://models.example.test/v1",
            max_attempts=1,
            session=session,  # type: ignore[arg-type]
        )

        with self.assertRaises(RetryableError) as raised:
            provider.embed(embedding_request(("only",)))

        self.assertEqual(raised.exception.code, "embedding_timeout")
        self.assertNotIn(API_KEY, repr(provider))
        self.assertNotIn(API_KEY, repr(provider.__dict__))
        self.assertNotIn(API_KEY, repr(raised.exception))
        self.assertNotIn(API_KEY, str(raised.exception.details))


if __name__ == "__main__":
    unittest.main()
