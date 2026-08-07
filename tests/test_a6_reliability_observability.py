from __future__ import annotations

import io
import json
import logging
import unittest

from fastapi.testclient import TestClient

from tender_review.api import create_app
from tender_review.bootstrap import build_container
from tender_review.config import PROJECT_DIR
from tender_review.shared.config import AppSettings
from tender_review.shared.logging import JsonFormatter
from tender_review.shared.observability import CorrelationContext, log_event


class A6StructuredLoggingTests(unittest.TestCase):
    def test_json_event_has_correlation_fields_and_redacts_nested_sensitive_data(
        self,
    ) -> None:
        stream = io.StringIO()
        handler = logging.StreamHandler(stream)
        handler.setFormatter(JsonFormatter())
        logger = logging.getLogger("a6.structured-log-contract")
        previous_level = logger.level
        previous_propagate = logger.propagate
        logger.handlers.clear()
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
        try:
            log_event(
                logger,
                logging.INFO,
                event="a6.contract_checked",
                message="api_key=example-redacted-api-key",
                context=CorrelationContext(
                    job_id="job-1",
                    thread_id="thread-1",
                    checkpoint_id="checkpoint-1",
                    call_id="call-1",
                    rule_version="rule-1",
                    dataset_version="dataset-1",
                    model_config="model-1",
                ),
                authorization="Bearer another-secret",
                reason="完整复核理由",
                nested={
                    "excerpt": "完整敏感原文",
                    "content": "模型完整响应",
                    "safe_status": "blocked",
                },
            )
        finally:
            logger.removeHandler(handler)
            logger.setLevel(previous_level)
            logger.propagate = previous_propagate

        serialized = stream.getvalue()
        payload = json.loads(serialized)
        self.assertEqual(
            {
                key: payload[key]
                for key in (
                    "job_id",
                    "thread_id",
                    "checkpoint_id",
                    "call_id",
                    "rule_version",
                    "dataset_version",
                    "model_config",
                )
            },
            {
                "job_id": "job-1",
                "thread_id": "thread-1",
                "checkpoint_id": "checkpoint-1",
                "call_id": "call-1",
                "rule_version": "rule-1",
                "dataset_version": "dataset-1",
                "model_config": "model-1",
            },
        )
        self.assertEqual(payload["authorization"], "<redacted>")
        self.assertEqual(payload["reason"], "<redacted>")
        self.assertEqual(payload["nested"]["excerpt"], "<redacted>")
        self.assertEqual(payload["nested"]["content"], "<redacted>")
        self.assertEqual(payload["nested"]["safe_status"], "blocked")
        for secret in (
            "example-redacted-api-key",
            "another-secret",
            "完整复核理由",
            "完整敏感原文",
            "模型完整响应",
        ):
            self.assertNotIn(secret, serialized)


class A6ApiAndWorkbenchContractTests(unittest.TestCase):
    def test_runtime_api_exposes_safe_recovery_fields_without_changing_frozen_projection(
        self,
    ) -> None:
        container = build_container(
            AppSettings(environment="test", adapter_mode="fake", log_json=False)
        )
        try:
            with TestClient(create_app(container)) as client:
                response = client.post(
                    "/api/v1/review-jobs",
                    headers={"Idempotency-Key": "a6-job"},
                    json={
                        "document_snapshot_id": "document-1",
                        "document_sha256": "a" * 64,
                        "rule_version_id": "rule-1",
                        "rule_version_hash": "b" * 64,
                        "model_config_id": "model-1",
                        "model_config_hash": "c" * 64,
                    },
                )
                schema = client.get("/openapi.json").json()
        finally:
            container.close()

        self.assertEqual(response.status_code, 201, response.text)
        body = response.json()
        self.assertEqual(body["recovery_count"], 0)
        self.assertEqual(body["recovery_metric_source"], "review_jobs.attempt_count")
        self.assertIsNone(body["safe_failure_code"])
        self.assertIsNone(body["safe_failure_category"])
        self.assertIsNone(body["safe_failure_retryable"])
        properties = schema["components"]["schemas"]["ReviewJobResponse"]["properties"]
        self.assertTrue(
            {
                "recovery_count",
                "recovery_metric_source",
                "safe_failure_code",
                "safe_failure_category",
                "safe_failure_retryable",
            }.issubset(properties)
        )

    def test_workbench_uses_safe_failure_recovery_metric_and_audit_trace_fields(
        self,
    ) -> None:
        source = (PROJECT_DIR / "workbench" / "src" / "App.vue").read_text(
            encoding="utf-8"
        )

        for contract in (
            "reviewJob.safe_failure_code",
            "reviewJob.safe_failure_category",
            "reviewJob.safe_failure_retryable",
            "reviewJob.recovery_count",
            "reviewJob.recovery_metric_source",
            "node_duration_ms:",
            "event.checkpoint_id",
        ):
            self.assertIn(contract, source)
        self.assertNotIn("reviewJob.failure.message", source)
        self.assertNotIn("event.provenance.source_text", source)


if __name__ == "__main__":
    unittest.main()
