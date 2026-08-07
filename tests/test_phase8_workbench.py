from __future__ import annotations

import json
import unittest

from fastapi.testclient import TestClient
from pydantic import ValidationError

from tender_review.api.app import create_app
from tender_review.bootstrap.assembly import build_container
from tender_review.shared.config import AppSettings
from tender_review.stage8.public import ReportMetric


RUN_ID = "synthetic-demo-run-v1"


class Phase8ContractTests(unittest.TestCase):
    def test_non_real_metrics_cannot_escalate_claims(self):
        with self.assertRaises(ValidationError):
            ReportMetric(
                metric_id="recall-at-10",
                label="Recall@10",
                value=0.99,
                unit="ratio",
                source_type="provisional",
                status="provisional",
                claims_allowed=True,
                collected=True,
                interpretation="invalid escalation",
            )

    def test_demo_report_is_deterministic_traceable_and_non_claimable(self):
        settings = AppSettings(environment="local", workbench_demo_enabled=True)
        first = build_container(settings)
        second = build_container(settings)
        try:
            first_run = first.stage8.get_evaluation_run(RUN_ID)
            second_run = second.stage8.get_evaluation_run(RUN_ID)
            first_report = first.stage8.get_evaluation_report(RUN_ID)
            second_report = second.stage8.get_evaluation_report(RUN_ID)
            self.assertEqual(first_run, second_run)
            self.assertEqual(first_report, second_report)
            self.assertEqual(first_run.report_sha256, first_report.report_sha256)
            self.assertEqual(first_report.status, "provisional")
            self.assertFalse(first_report.claims_allowed)
            self.assertEqual(first_report.human_annotation_cases, 0)
            self.assertEqual(first_report.required_human_cases, 4)
            metrics = [
                metric
                for section in first_report.sections
                for metric in section.metrics
            ]
            self.assertTrue(all(not metric.claims_allowed for metric in metrics))
            for metric_id in ("recall-at-10", "mrr", "model-cost", "production-latency"):
                metric = next(item for item in metrics if item.metric_id == metric_id)
                self.assertIsNone(metric.value)
                self.assertFalse(metric.collected)
                self.assertEqual(metric.status.value, "unknown")
        finally:
            first.close()
            second.close()


class Phase8ApiTests(unittest.TestCase):
    def setUp(self) -> None:
        self.container = build_container(
            AppSettings(
                environment="local",
                log_json=False,
                workbench_demo_enabled=True,
            )
        )
        self.client = TestClient(create_app(self.container))

    def tearDown(self) -> None:
        self.client.close()

    def test_workbench_queries_cover_progress_evidence_rules_reports_and_attempts(self):
        index = self.client.get("/api/v1/workbench")
        self.assertEqual(index.status_code, 200, index.text)
        body = index.json()
        self.assertTrue(body["demo_mode"])
        self.assertEqual(body["source_type"], "synthetic")
        self.assertEqual(body["status"], "provisional")
        self.assertFalse(body["claims_allowed"])
        self.assertEqual(body["human_annotation_cases"], 0)
        self.assertEqual(body["required_human_cases"], 4)

        job_id = body["review_job_ids"][0]
        job = self.client.get(f"/api/v1/review-jobs/{job_id}")
        checkpoints = self.client.get(f"/api/v1/review-jobs/{job_id}/checkpoints")
        findings = self.client.get(f"/api/v1/review-jobs/{job_id}/findings")
        self.assertEqual(job.status_code, 200)
        self.assertEqual(job.json()["status"], "WAITING_HUMAN")
        self.assertEqual(len(checkpoints.json()), 7)
        self.assertEqual(findings.status_code, 200)
        evidence = findings.json()[0]["evidence"][0]
        self.assertEqual(evidence["page_number"], 1)
        self.assertTrue(evidence["section_path"])
        self.assertTrue(evidence["excerpt"])
        self.assertEqual(len(evidence["text_sha256"]), 64)

        self.assertEqual(body["rule_set_ids"], [])
        self.assertEqual(body["optimization_job_ids"], [])

        run = self.client.get(f"/api/v1/evaluation-runs/{RUN_ID}")
        report = self.client.get(f"/api/v1/evaluation-runs/{RUN_ID}/report")
        self.assertEqual(run.status_code, 200)
        self.assertEqual(report.status_code, 200)
        self.assertEqual(run.json()["report_sha256"], report.json()["report_sha256"])
        self.assertEqual(len(run.json()["hashes"]), 6)


    def test_provisional_approval_is_rejected_and_audited(self):
        reason = "Synthetic sensitive reason that must not enter audit logs"
        approval = self.client.post(
            "/api/v1/findings/demo-finding-1/decisions",
            headers={"X-Request-ID": "req-finding-approval", "X-Call-ID": "call-1"},
            json={
                "reviewer_id": "reviewer-example-a",
                "decision": "APPROVE",
                "reason": reason,
            },
        )
        self.assertEqual(approval.status_code, 422, approval.text)
        self.assertEqual(
            approval.json()["error"]["code"],
            "provisional_finding_approval_forbidden",
        )

        events = self.client.get("/api/v1/audit-events").json()
        finding_event = next(
            item for item in events if item["request_id"] == "req-finding-approval"
        )
        self.assertEqual(finding_event["actor"]["kind"], "human")
        self.assertEqual(finding_event["result"], "rejected")
        self.assertEqual(finding_event["call_id"], "call-1")
        serialized = json.dumps(events, ensure_ascii=False)
        self.assertNotIn(reason, serialized)

    def test_named_reviewer_can_record_rejection_through_real_api(self):
        response = self.client.post(
            "/api/v1/findings/demo-finding-1/decisions",
            json={
                "reviewer_id": "reviewer-example-b",
                "decision": "REJECT",
                "reason": "证据仍不足，保持不可声明",
            },
        )
        self.assertEqual(response.status_code, 201, response.text)
        self.assertEqual(response.json()["finding"]["status"], "REJECTED")
        decisions = self.client.get(
            "/api/v1/findings/demo-finding-1/decisions"
        )
        self.assertEqual(decisions.status_code, 200)
        self.assertEqual(decisions.json()[0]["reviewer_id"], "reviewer-example-b")

    def test_empty_non_demo_workbench_does_not_fall_back_to_fake(self):
        container = build_container(
            AppSettings(
                environment="test",
                log_json=False,
                workbench_demo_enabled=False,
            )
        )
        with TestClient(create_app(container)) as client:
            index = client.get("/api/v1/workbench")
            self.assertEqual(index.status_code, 200)
            self.assertFalse(index.json()["demo_mode"])
            self.assertEqual(index.json()["evaluation_run_ids"], [])
            missing = client.get(f"/api/v1/evaluation-runs/{RUN_ID}")
            self.assertEqual(missing.status_code, 404)
            self.assertEqual(
                missing.json()["error"]["code"], "evaluation_run_not_found"
            )


if __name__ == "__main__":
    unittest.main()
