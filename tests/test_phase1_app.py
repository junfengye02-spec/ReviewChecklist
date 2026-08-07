import io
import json
import unittest
from dataclasses import dataclass

from fastapi.testclient import TestClient

from tender_review.api import create_app
from tender_review.bootstrap import build_container, create_api_app
from tender_review.shared.config import AppSettings
from tender_review.shared.errors import RetryableError
from tender_review.shared.health import StaticReadinessCheck
from tender_review.shared.ids import SequentialIdGenerator
from tender_review.shared.logging import configure_logging


@dataclass(frozen=True)
class ForeignHealthResult:
    name: str
    ready: bool
    detail: str


class ForeignReadinessCheck:
    name = "foreign"

    def check(self):
        return ForeignHealthResult(name=self.name, ready=True, detail="ok")


class FastApiApplicationTests(unittest.TestCase):
    def setUp(self):
        settings = AppSettings(environment="test", log_json=False)
        self.container = build_container(settings).with_overrides(
            ids=SequentialIdGenerator(prefix="request")
        )
        self.app = create_app(self.container)
        self.client = TestClient(self.app)

    def test_liveness_readiness_and_v1_index_are_offline(self):
        live = self.client.get("/health/live")
        ready = self.client.get("/health/ready")
        index = self.client.get("/api/v1")

        self.assertEqual(live.status_code, 200)
        self.assertEqual(live.json()["status"], "ok")
        self.assertEqual(ready.status_code, 200)
        self.assertEqual(ready.json()["status"], "ready")
        self.assertEqual(ready.json()["checks"]["dependencies"]["status"], "ready")
        self.assertEqual(index.status_code, 200)
        self.assertEqual(index.json()["api_version"], "v1")
        self.assertEqual(index.json()["schema_version"], 1)

    def test_request_id_is_preserved_or_generated(self):
        supplied = self.client.get("/health/live", headers={"X-Request-ID": "caller-1"})
        generated = self.client.get("/health/live")

        self.assertEqual(supplied.headers["X-Request-ID"], "caller-1")
        self.assertEqual(generated.headers["X-Request-ID"], "request-1")

    def test_framework_and_service_errors_use_the_same_envelope(self):
        @self.app.get("/api/v1/temporary-failure")
        def temporary_failure():
            raise RetryableError("Dependency is unavailable", code="dependency_down")

        missing = self.client.get(
            "/does-not-exist", headers={"X-Request-ID": "missing-1"}
        )
        failed = self.client.get(
            "/api/v1/temporary-failure", headers={"X-Request-ID": "failure-1"}
        )

        self.assertEqual(missing.status_code, 404)
        self.assertEqual(missing.json()["error"]["category"], "not_found")
        self.assertEqual(missing.json()["error"]["request_id"], "missing-1")
        self.assertEqual(failed.status_code, 503)
        self.assertEqual(failed.json()["error"]["code"], "dependency_down")
        self.assertTrue(failed.json()["error"]["retryable"])

    def test_unexpected_errors_do_not_expose_exception_details(self):
        @self.app.get("/api/v1/unexpected-failure")
        def unexpected_failure():
            raise RuntimeError("private adapter detail")

        client = TestClient(self.app, raise_server_exceptions=False)
        response = client.get(
            "/api/v1/unexpected-failure", headers={"X-Request-ID": "internal-1"}
        )

        self.assertEqual(response.status_code, 500)
        body = response.json()["error"]
        self.assertEqual(body["code"], "internal_error")
        self.assertEqual(body["request_id"], "internal-1")
        self.assertNotIn("private adapter detail", response.text)

    def test_openapi_exposes_versioned_and_health_routes(self):
        schema = self.client.get("/openapi.json").json()

        self.assertIn("/api/v1", schema["paths"])
        self.assertIn("/health/live", schema["paths"])
        self.assertIn("/health/ready", schema["paths"])
        self.assertEqual(schema["info"]["version"], self.container.settings.version)

    def test_request_validation_uses_error_envelope(self):
        @self.app.get("/api/v1/validated")
        def validated(limit: int):
            return {"limit": limit}

        response = self.client.get("/api/v1/validated?limit=not-an-integer")

        self.assertEqual(response.status_code, 422)
        body = response.json()["error"]
        self.assertEqual(body["code"], "request_validation_failed")
        self.assertEqual(body["category"], "invalid_request")
        self.assertEqual(body["validation_errors"][0]["location"], ["query", "limit"])

    def test_failed_readiness_check_returns_503_without_affecting_liveness(self):
        container = self.container.with_overrides(
            readiness_checks=(
                StaticReadinessCheck("database", ready=False, detail="offline"),
            )
        )
        client = TestClient(create_app(container))

        self.assertEqual(client.get("/health/live").status_code, 200)
        response = client.get("/health/ready")
        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.json()["status"], "not_ready")
        self.assertEqual(response.json()["checks"]["database"]["status"], "not_ready")

    def test_structural_readiness_result_is_accepted(self):
        container = self.container.with_overrides(
            readiness_checks=(ForeignReadinessCheck(),)
        )

        response = TestClient(create_app(container)).get("/health/ready")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["checks"]["foreign"]["status"], "ready")


class AppSettingsTests(unittest.TestCase):
    def test_bootstrap_assembles_an_offline_api(self):
        app = create_api_app(AppSettings(environment="test", log_json=False))

        self.assertEqual(app.state.container.settings.adapter_mode, "fake")

    def test_environment_loading_is_central_and_typed(self):
        settings = AppSettings.from_env(
            {
                "TENDER_REVIEW_ENVIRONMENT": "test",
                "TENDER_REVIEW_LOG_JSON": "false",
                "TENDER_REVIEW_WORKER_LEASE_SECONDS": "45",
                "TENDER_REVIEW_API_PREFIX": "custom/v1/",
                "DATABASE_URL": "mysql+pymysql://local/test",
                "MINIO_SECURE": "true",
                "TENDER_REVIEW_LLM_BASE_URL": "https://llm.example.test/v1",
                "TENDER_REVIEW_LLM_API_KEY": "llm-config-test-key",
                "TENDER_REVIEW_LLM_MODEL": "review-model",
                "TENDER_REVIEW_LLM_TIMEOUT_SECONDS": "11.5",
                "TENDER_REVIEW_LLM_MAX_ATTEMPTS": "4",
                "TENDER_REVIEW_LLM_TEMPERATURE": "0.25",
                "TENDER_REVIEW_EMBEDDING_BASE_URL": (
                    "https://embedding.example.test/v1"
                ),
                "TENDER_REVIEW_EMBEDDING_API_KEY": "embedding-config-test-key",
                "TENDER_REVIEW_EMBEDDING_MODEL": "embedding-model",
                "TENDER_REVIEW_EMBEDDING_TIMEOUT_SECONDS": "7.5",
                "TENDER_REVIEW_EMBEDDING_MAX_ATTEMPTS": "2",
                "TENDER_REVIEW_EMBEDDING_DIMENSIONS": "1024",
                "TENDER_REVIEW_EMBEDDING_BATCH_SIZE": "48",
            }
        )

        self.assertEqual(settings.environment, "test")
        self.assertFalse(settings.log_json)
        self.assertEqual(settings.worker_lease_seconds, 45)
        self.assertEqual(settings.api_prefix, "/custom/v1")
        self.assertEqual(settings.database_url, "mysql+pymysql://local/test")
        self.assertTrue(settings.minio_secure)
        self.assertEqual(settings.llm_base_url, "https://llm.example.test/v1")
        self.assertEqual(settings.llm_model, "review-model")
        self.assertEqual(settings.llm_timeout_seconds, 11.5)
        self.assertEqual(settings.llm_max_attempts, 4)
        self.assertEqual(settings.llm_temperature, 0.25)
        self.assertEqual(
            settings.embedding_base_url, "https://embedding.example.test/v1"
        )
        self.assertEqual(settings.embedding_model, "embedding-model")
        self.assertEqual(settings.embedding_timeout_seconds, 7.5)
        self.assertEqual(settings.embedding_max_attempts, 2)
        self.assertEqual(settings.embedding_dimensions, 1024)
        self.assertEqual(settings.embedding_batch_size, 48)

        rendered = repr(settings) + settings.model_dump_json()
        self.assertNotIn("llm-config-test-key", rendered)
        self.assertNotIn("embedding-config-test-key", rendered)

    def test_structured_logger_is_independent_from_root_handlers(self):
        stream = io.StringIO()
        logger = configure_logging("INFO", stream=stream, force=True)

        logger.info("ready", extra={"event": "test.ready", "job_id": "job-1"})
        record = json.loads(stream.getvalue())

        self.assertEqual(record["level"], "INFO")
        self.assertEqual(record["event"], "test.ready")
        self.assertEqual(record["job_id"], "job-1")
        self.assertEqual(record["message"], "ready")


if __name__ == "__main__":
    unittest.main()
