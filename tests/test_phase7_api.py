from __future__ import annotations

import unittest

from fastapi.testclient import TestClient

from tender_review.api.app import create_app
from tender_review.bootstrap.assembly import build_container
from tender_review.evaluation.public import (
    CreateDatasetVersion,
    DatasetProvenance,
)
from tender_review.rule_management.public import CreateRuleVersion, RuleProvenance
from tender_review.shared.config import AppSettings

from test_phase7_optimization import _dataset_sample, _optimization_sample, _provenance
from tender_review.evaluation.public import DatasetSplit
from tender_review.optimization.public import FailureSignals, SampleRole


class Phase7ApiTests(unittest.TestCase):
    def setUp(self) -> None:
        self.container = build_container(AppSettings(environment="test", log_json=False))
        self.base = self.container.rule_versions.create_version(
            CreateRuleVersion(
                rule_set_id="set-api-optimization",
                rule_key="api-optimization",
                rule_set_name="API optimization",
                content_json='{"rule_text":"base"}',
                change_summary="base version",
                provenance=RuleProvenance(
                    source_type="manual", status="verified", claims_allowed=True
                ),
            )
        )
        self.dataset = self.container.dataset_versions.create_version(
            CreateDatasetVersion(
                dataset_name="api-phase7",
                requested_status="PROVISIONAL",
                change_summary="API flow demonstration",
                provenance=DatasetProvenance(
                    status="provisional",
                    claims_allowed=False,
                    source_description="synthetic examples; human labels 0/4",
                    source_manifest_sha256="a" * 64,
                ),
                samples=(
                    _dataset_sample(
                        "target-1", "doc-target", DatasetSplit.OPTIMIZATION
                    ),
                    _dataset_sample(
                        "protection-1", "doc-protection", DatasetSplit.VALIDATION
                    ),
                ),
            )
        )
        self.client = TestClient(create_app(self.container))

    def _payload(self):
        samples = [
            _optimization_sample(
                "target-1",
                SampleRole.TARGET,
                "doc-target",
                FailureSignals(
                    failure_summary="bounded rule gap",
                    evidence_in_top_k=True,
                    extraction_matches_expected=True,
                    tool_matches_expected=True,
                    repeated_outputs_consistent=True,
                ),
            ).model_dump(mode="json"),
            _optimization_sample(
                "protection-1",
                SampleRole.PROTECTION,
                "doc-protection",
                None,
            ).model_dump(mode="json"),
        ]
        for sample in samples:
            sample.update(
                {
                    "source_type": "EXTERNAL_PLATFORM",
                    "provenance_status": "provisional",
                    "claims_allowed": False,
                    "finding_id": None,
                    "human_decision_id": None,
                }
            )
        provenance = _provenance().model_dump(mode="json")
        provenance.update(
            {
                "source_type": "EXTERNAL_PLATFORM",
                "status": "provisional",
                "claims_allowed": False,
                "human_annotation_cases": 0,
            }
        )
        return {
            "dataset_version_id": self.dataset.dataset_version_id,
            "max_rounds": 2,
            "candidates_per_round": 2,
            "required_stability_runs": 2,
            "model_sha256": "b" * 64,
            "prompt_sha256": "c" * 64,
            "retriever_sha256": "d" * 64,
            "tool_sha256": "e" * 64,
            "samples": samples,
            "provenance": provenance,
        }

    def test_create_query_attempts_and_cancel_contract(self):
        created = self.client.post(
            f"/api/v1/rule-versions/{self.base.rule_version_id}/optimize",
            json=self._payload(),
        )
        self.assertEqual(created.status_code, 202, created.text)
        job = created.json()
        self.assertEqual(job["status"], "NOT_READY")
        self.assertEqual(job["hashes"]["rule_sha256"], self.base.content_sha256)
        self.assertEqual(job["provenance"]["status"], "provisional")
        self.assertFalse(job["provenance"]["claims_allowed"])
        self.assertFalse(job["readiness"]["claims_allowed"])
        self.assertTrue(job["readiness"]["blockers"])

        job_id = job["optimization_job_id"]
        queried = self.client.get(f"/api/v1/optimization-jobs/{job_id}")
        self.assertEqual(queried.status_code, 200)
        self.assertEqual(queried.json(), job)
        attempts = self.client.get(
            f"/api/v1/optimization-jobs/{job_id}/attempts"
        )
        self.assertEqual(attempts.status_code, 200)
        self.assertEqual(attempts.json(), [])
        cancelled = self.client.post(
            f"/api/v1/optimization-jobs/{job_id}/cancel"
        )
        self.assertEqual(cancelled.status_code, 409)
        self.assertEqual(
            cancelled.json()["error"]["code"], "optimization_cancel_invalid"
        )

    def test_api_rejects_provisional_claim_escalation(self):
        payload = self._payload()
        payload["provenance"]["claims_allowed"] = True
        response = self.client.post(
            f"/api/v1/rule-versions/{self.base.rule_version_id}/optimize",
            json=payload,
        )
        self.assertEqual(response.status_code, 422)
        self.assertEqual(
            response.json()["error"]["code"], "request_validation_failed"
        )

    def test_a5_endpoint_accepts_only_evidence_identity_and_remains_not_ready(self):
        payload = self._payload()
        payload.update(
            {
                "a4_evaluation_run_id": "missing-a4-run",
                "a4_report_sha256": "f" * 64,
            }
        )
        response = self.client.post(
            f"/api/v1/a5/rule-versions/{self.base.rule_version_id}/optimize",
            json=payload,
        )

        self.assertEqual(response.status_code, 202, response.text)
        body = response.json()
        self.assertEqual(body["status"], "NOT_READY")
        self.assertEqual(
            body["readiness"]["a4_evaluation_run_id"], "missing-a4-run"
        )
        self.assertFalse(body["readiness"]["claims_allowed"])


if __name__ == "__main__":
    unittest.main()
