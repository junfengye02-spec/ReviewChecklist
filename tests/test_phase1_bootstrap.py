from __future__ import annotations

import unittest
from typing import Any
from unittest.mock import patch

from tender_review.bootstrap import build_container
from tender_review.infrastructure.ai import (
    OpenAICompatibleEmbeddingProvider,
    OpenAICompatibleLlmProvider,
)
from tender_review.infrastructure.database.langgraph_checkpoints import (
    SqlAlchemyCheckpointSaver,
)
from tender_review.infrastructure.object_storage import MinioArtifactStore
from tender_review.review.workflow import SingleReviewWorkflow
from tender_review.shared.contracts import CallContext
from tender_review.shared.config import AppSettings
from tender_review.shared.errors import PermanentError

from test_phase5_review import review_request, text_extraction


LLM_TEST_KEY = "llm-test-key"
EMBEDDING_TEST_KEY = "embedding-test-key"


def production_settings(**overrides: Any) -> AppSettings:
    values: dict[str, Any] = {
        "environment": "local",
        "adapter_mode": "production",
        "database_url": "mysql+pymysql://user:pass@127.0.0.1/review",
        "minio_endpoint": "127.0.0.1:9000",
        "minio_access_key": "access",
        "minio_secret_key": "secret",
        "minio_bucket": "artifacts",
        "llm_base_url": "https://llm.example.test/v1",
        "llm_api_key": LLM_TEST_KEY,
        "llm_model": "review-model",
        "llm_timeout_seconds": 12.5,
        "llm_max_attempts": 3,
        "llm_temperature": 0.2,
        "embedding_base_url": "https://embedding.example.test/v1",
        "embedding_api_key": EMBEDDING_TEST_KEY,
        "embedding_model": "embedding-model",
        "embedding_timeout_seconds": 8.5,
        "embedding_max_attempts": 2,
        "embedding_dimensions": 1024,
        "embedding_batch_size": 32,
    }
    values.update(overrides)
    return AppSettings(**values)


class RateLimitedResponse:
    status_code = 429


class RateLimitedSession:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def post(self, url: str, **kwargs: Any) -> RateLimitedResponse:
        self.calls.append({"url": url, **kwargs})
        return RateLimitedResponse()


class ProductionBootstrapTests(unittest.TestCase):
    def test_production_explicitly_assembles_external_adapters_and_readiness(self):
        container = build_container(production_settings())
        try:
            self.assertEqual(container.database_engine.url.drivername, "mysql+pymysql")
            self.assertIsNotNone(container.session_factory)
            self.assertIsInstance(container.artifact_store, MinioArtifactStore)
            self.assertEqual(
                [check.name for check in container.readiness_checks],
                ["database", "object_storage", "review_model_config"],
            )
            self.assertIsNone(container.document_parser)
            self.assertIsInstance(
                container.embedding_provider, OpenAICompatibleEmbeddingProvider
            )
            self.assertEqual(container.embedding_provider.model, "embedding-model")
            self.assertEqual(container.embedding_provider.dimensions, 1024)
            self.assertEqual(container.embedding_provider.batch_size, 32)
            self.assertEqual(container.embedding_provider.timeout_seconds, 8.5)
            self.assertEqual(container.embedding_provider.max_attempts, 2)
            self.assertIsInstance(container.llm_provider, OpenAICompatibleLlmProvider)
            self.assertEqual(container.llm_provider.model, "review-model")
            self.assertEqual(container.llm_provider.timeout_seconds, 12.5)
            self.assertEqual(container.llm_provider.max_attempts, 3)
            self.assertEqual(container.llm_provider.temperature, 0.2)
            self.assertIsInstance(container.checkpoint_saver, SqlAlchemyCheckpointSaver)
            self.assertIs(
                container.checkpoint_saver._sessions, container.session_factory
            )
            self.assertIsNone(container.retriever)
            production_components = (
                container.job_repository,
                container.lease_manager,
                container.artifact_store,
                container.embedding_provider,
                container.llm_provider,
                container.checkpoint_saver,
            )
            self.assertTrue(
                all(
                    ".fakes" not in type(value).__module__
                    for value in production_components
                )
            )
        finally:
            container.close()

    def test_production_rejects_missing_external_configuration(self):
        with self.assertRaises(PermanentError) as raised:
            build_container(AppSettings(adapter_mode="production"))
        self.assertEqual(raised.exception.code, "production_configuration_invalid")
        self.assertIn("database_url", raised.exception.details["missing_fields"])
        self.assertIn("llm_base_url", raised.exception.details["missing_fields"])
        self.assertIn("llm_api_key", raised.exception.details["missing_fields"])
        self.assertIn("llm_model", raised.exception.details["missing_fields"])
        self.assertIn("embedding_base_url", raised.exception.details["missing_fields"])
        self.assertIn("embedding_api_key", raised.exception.details["missing_fields"])
        self.assertIn("embedding_model", raised.exception.details["missing_fields"])

    def test_production_rejects_invalid_ai_configuration_without_key_disclosure(self):
        with self.assertRaises(PermanentError) as raised:
            build_container(production_settings(llm_base_url="not-a-url"))

        self.assertEqual(raised.exception.code, "production_ai_configuration_invalid")
        self.assertNotIn(LLM_TEST_KEY, repr(raised.exception))
        self.assertNotIn(EMBEDDING_TEST_KEY, repr(raised.exception))

    def test_workflow_owns_the_total_llm_attempt_budget(self):
        session = RateLimitedSession()
        with patch(
            "tender_review.infrastructure.ai.openai_compatible.requests.Session",
            return_value=session,
        ):
            container = build_container(production_settings(llm_max_attempts=3))
            try:
                request, _ = review_request(
                    text_extraction(),
                    call=CallContext(
                        call_id="total-budget",
                        timeout_seconds=1.0,
                        max_attempts=3,
                    ),
                )
                state = SingleReviewWorkflow(container.llm_provider).run(request)

                self.assertEqual(len(session.calls), 3)
                self.assertEqual(len(state.call_records), 3)
                self.assertEqual(
                    [call["headers"]["X-Request-ID"] for call in session.calls],
                    [
                        "total-budget:extract:1",
                        "total-budget:extract:2",
                        "total-budget:extract:3",
                    ],
                )
            finally:
                container.close()

    def test_fake_mode_does_not_require_ai_configuration(self):
        container = build_container(AppSettings(environment="test"))

        self.assertIsNotNone(container.llm_provider)
        self.assertIsNotNone(container.embedding_provider)
        self.assertIsNone(container.checkpoint_saver)

    def test_fake_mode_is_rejected_for_production_environment(self):
        with self.assertRaises(PermanentError) as raised:
            build_container(AppSettings(environment="production", adapter_mode="fake"))
        self.assertEqual(raised.exception.code, "fake_adapters_forbidden")


if __name__ == "__main__":
    unittest.main()
