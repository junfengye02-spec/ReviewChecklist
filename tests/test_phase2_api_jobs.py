from __future__ import annotations

import unittest
from datetime import datetime, timezone

from fastapi.testclient import TestClient

from tender_review.api import create_app
from tender_review.bootstrap import build_container
from tender_review.jobs.public import ReviewJobService
from tender_review.shared.clock import FixedClock
from tender_review.shared.config import AppSettings
from tender_review.shared.ids import SequentialIdGenerator


NOW = datetime(2026, 7, 27, 9, 30, tzinfo=timezone.utc)


def request_body(**updates):
    body = {
        "schema_version": 1,
        "document_snapshot_id": "document-1",
        "document_sha256": "a" * 64,
        "rule_version_id": "rule-1",
        "rule_version_hash": "b" * 64,
        "model_config_id": "model-1",
        "model_config_hash": "c" * 64,
        "max_attempts": 3,
    }
    body.update(updates)
    return body


class ReviewJobApiTests(unittest.TestCase):
    def setUp(self):
        container = build_container(
            AppSettings(environment="test", adapter_mode="fake", log_json=False)
        )
        clock = FixedClock(NOW)
        ids = SequentialIdGenerator(prefix="api-resource")
        service = ReviewJobService(
            repository=container.job_repository,
            ids=ids,
            clock=clock,
        )
        self.container = container.with_overrides(
            clock=clock,
            ids=ids,
            review_jobs=service,
        )
        self.client = TestClient(create_app(self.container))

    def test_create_replay_query_cancel_and_explicit_rerun(self):
        headers = {"Idempotency-Key": "create-1", "X-Caller-ID": "caller-1"}
        created = self.client.post(
            "/api/v1/review-jobs", json=request_body(), headers=headers
        )
        replay = self.client.post(
            "/api/v1/review-jobs", json=request_body(), headers=headers
        )

        self.assertEqual(created.status_code, 201)
        self.assertEqual(replay.status_code, 200)
        self.assertEqual(created.json(), replay.json())
        job_id = created.json()["id"]
        self.assertEqual(created.json()["status"], "QUEUED")
        self.assertIsNone(created.json()["stage"])
        self.assertEqual(created.json()["schema_version"], 1)

        queried = self.client.get(f"/api/v1/review-jobs/{job_id}")
        cancelled = self.client.post(f"/api/v1/review-jobs/{job_id}/cancel")
        cancelled_again = self.client.post(f"/api/v1/review-jobs/{job_id}/cancel")
        rerun = self.client.post(f"/api/v1/review-jobs/{job_id}/rerun")

        self.assertEqual(queried.status_code, 200)
        self.assertEqual(cancelled.status_code, 200)
        self.assertEqual(cancelled.json()["status"], "CANCELLED")
        self.assertEqual(cancelled_again.json(), cancelled.json())
        self.assertEqual(rerun.status_code, 201)
        self.assertEqual(rerun.json()["rerun_of"], job_id)
        self.assertEqual(rerun.json()["status"], "QUEUED")
        self.assertNotEqual(rerun.json()["id"], job_id)

    def test_same_key_different_request_is_a_unified_409(self):
        headers = {
            "Idempotency-Key": "conflicting-key",
            "X-Caller-ID": "caller-1",
            "X-Request-ID": "conflict-request",
        }
        self.client.post("/api/v1/review-jobs", json=request_body(), headers=headers)

        response = self.client.post(
            "/api/v1/review-jobs",
            json=request_body(model_config_hash="d" * 64),
            headers=headers,
        )

        self.assertEqual(response.status_code, 409)
        error = response.json()["error"]
        self.assertEqual(error["code"], "idempotency_key_reused")
        self.assertEqual(error["category"], "conflict")
        self.assertFalse(error["retryable"])
        self.assertEqual(error["request_id"], "conflict-request")

    def test_idempotency_key_is_scoped_by_caller(self):
        first = self.client.post(
            "/api/v1/review-jobs",
            json=request_body(),
            headers={"Idempotency-Key": "shared", "X-Caller-ID": "caller-a"},
        )
        second = self.client.post(
            "/api/v1/review-jobs",
            json=request_body(),
            headers={"Idempotency-Key": "shared", "X-Caller-ID": "caller-b"},
        )

        self.assertEqual(first.status_code, 201)
        self.assertEqual(second.status_code, 201)
        self.assertNotEqual(first.json()["id"], second.json()["id"])

    def test_validation_not_found_and_illegal_rerun_use_error_envelope(self):
        missing_key = self.client.post("/api/v1/review-jobs", json=request_body())
        invalid_hash = self.client.post(
            "/api/v1/review-jobs",
            json=request_body(document_sha256="z" * 64),
            headers={"Idempotency-Key": "invalid"},
        )
        missing = self.client.get("/api/v1/review-jobs/missing")
        created = self.client.post(
            "/api/v1/review-jobs",
            json=request_body(),
            headers={"Idempotency-Key": "active"},
        )
        rerun = self.client.post(
            f"/api/v1/review-jobs/{created.json()['id']}/rerun"
        )

        self.assertEqual(missing_key.status_code, 422)
        self.assertEqual(
            missing_key.json()["error"]["code"], "request_validation_failed"
        )
        self.assertEqual(invalid_hash.status_code, 422)
        self.assertEqual(
            invalid_hash.json()["error"]["category"], "invalid_request"
        )
        self.assertEqual(missing.status_code, 404)
        self.assertEqual(missing.json()["error"]["code"], "review_job_not_found")
        self.assertEqual(rerun.status_code, 409)
        self.assertEqual(
            rerun.json()["error"]["code"], "review_job_rerun_not_allowed"
        )

    def test_openapi_includes_the_first_review_job_routes(self):
        paths = self.client.get("/openapi.json").json()["paths"]
        self.assertIn("/api/v1/review-jobs", paths)
        self.assertIn("/api/v1/review-jobs/{job_id}", paths)
        self.assertIn("/api/v1/review-jobs/{job_id}/cancel", paths)
        self.assertIn("/api/v1/review-jobs/{job_id}/rerun", paths)


if __name__ == "__main__":
    unittest.main()
