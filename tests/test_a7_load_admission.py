from __future__ import annotations

import json
import io
import logging
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from fastapi.testclient import TestClient
from pydantic import ValidationError

from tender_review.api import create_app
from tender_review.bootstrap import build_container
from tender_review.config import PROJECT_DIR
from tender_review.performance.public import (
    A7Authenticity,
    A7EvidenceBundle,
    A7ExecutionBinding,
    AdmissionEvidence,
    AdmissionEvidenceType,
    FrozenA7ThresholdPolicy,
    MetricId,
    ObservationKind,
    ObservationSource,
    QueueDecision,
    RawObservation,
    RedisDecision,
    RunStatus,
    ScenarioCapture,
    SourceType,
    ThresholdRule,
    assess_evidence,
    create_not_run_plan,
    expected_scenarios,
    load_report,
    scenario_id,
    seal_model,
    stable_sha256,
)
from tender_review.performance.cli import main as a7_cli_main
from tender_review.jobs.fakes import FakeLeaseManager, InMemoryJobRepository
from tender_review.shared.clock import SystemClock
from tender_review.shared.config import AppSettings
from tender_review.shared.logging import JsonFormatter
from tender_review.worker.runner import Worker


NOW = datetime(2026, 8, 6, 8, 0, tzinfo=timezone.utc)
SHA = "a" * 64
ATTESTATION_KEY = "test-only-a7-attestation-key-32-bytes-minimum"
ATTESTATION_KEY_ID = "test-only-a7-key"


def _observation(
    *,
    capture_id: str,
    sequence: int,
    kind: ObservationKind,
    value: float,
    unit: str,
    source: ObservationSource,
    worker_id: str | None = None,
    job_id: str | None = None,
    node_name: str | None = None,
) -> RawObservation:
    return seal_model(
        RawObservation,
        {
            "observation_id": f"{capture_id}-o{sequence}",
            "scenario_id": capture_id,
            "observed_at": NOW + timedelta(milliseconds=sequence),
            "kind": kind,
            "value": value,
            "unit": unit,
            "source": source,
            "source_artifact_sha256": SHA,
            "worker_process_id": worker_id,
            "job_id": job_id,
            "node_name": node_name,
        },
        "record_sha256",
    )


def _capture(profile, concurrency: int, workers: int) -> ScenarioCapture:
    capture_id = scenario_id(profile, concurrency, workers)
    worker_ids = tuple(f"{capture_id}-worker-{index}" for index in range(workers))
    observations: list[RawObservation] = []

    def add(kind, value, unit, source, **fields):
        observations.append(
            _observation(
                capture_id=capture_id,
                sequence=len(observations) + 1,
                kind=kind,
                value=value,
                unit=unit,
                source=source,
                **fields,
            )
        )

    add(
        ObservationKind.MYSQL_CPU_PERCENT,
        10,
        "percent",
        ObservationSource.MYSQL_SERVER_STATUS,
    )
    add(
        ObservationKind.LOCK_WAIT_MS,
        0,
        "milliseconds",
        ObservationSource.MYSQL_PERFORMANCE_SCHEMA,
    )
    for worker_id in worker_ids:
        add(
            ObservationKind.CLAIM_LATENCY_MS,
            1,
            "milliseconds",
            ObservationSource.WORKER_STRUCTURED_LOG,
            worker_id=worker_id,
        )
        add(
            ObservationKind.EMPTY_POLL,
            0,
            "boolean",
            ObservationSource.WORKER_STRUCTURED_LOG,
            worker_id=worker_id,
        )
    add(
        ObservationKind.QUEUE_LATENCY_MS,
        1,
        "milliseconds",
        ObservationSource.WORKER_STRUCTURED_LOG,
        worker_id=worker_ids[0],
    )
    for node in ("parsing", "retrieval", "review", "report"):
        add(
            ObservationKind.NODE_DURATION_MS,
            2,
            "milliseconds",
            ObservationSource.WORKER_STRUCTURED_LOG,
            worker_id=worker_ids[0],
            node_name=node,
        )
    for index in range(concurrency):
        add(
            ObservationKind.JOB_OUTCOME,
            1,
            "boolean",
            ObservationSource.API_JOB_SNAPSHOT,
            job_id=f"{capture_id}-job-{index}",
        )
    add(
        ObservationKind.RECOVERY_OUTCOME,
        1,
        "boolean",
        ObservationSource.PROCESS_SUPERVISOR,
        worker_id=worker_ids[0],
    )
    return seal_model(
        ScenarioCapture,
        {
            "scenario_id": capture_id,
            "file_profile": profile,
            "page_count": {"TEXT_20": 20, "MIXED_100": 100, "SCANNED_300": 300}[
                profile.value
            ],
            "concurrency": concurrency,
            "workers": workers,
            "status": RunStatus.COMPLETED,
            "started_at": NOW,
            "finished_at": NOW + timedelta(minutes=1),
            "worker_process_ids": worker_ids,
            "observations": tuple(observations),
        },
        "capture_sha256",
    )


def _binding() -> A7ExecutionBinding:
    return seal_model(
        A7ExecutionBinding,
        {
            "environment_id": "test-only-real-contract-fixture",
            "compose_file_sha256": "1" * 64,
            "compose_config_sha256": "2" * 64,
            "api_image_digest": "3" * 64,
            "worker_image_digest": "4" * 64,
            "mysql_image_digest": "5" * 64,
            "minio_image_digest": "6" * 64,
            "git_commit": "7" * 40,
            "git_dirty": False,
            "code_sha256": "8" * 64,
            "dataset_sha256": "9" * 64,
            "model_config_sha256": "b" * 64,
            "workload_sha256": "c" * 64,
        },
        "binding_sha256",
    )


def _authenticity(**updates) -> A7Authenticity:
    payload = {
        "source_type": SourceType.REAL,
        "provenance_status": "verified",
        "claims_allowed": True,
        "environment_kind": "dedicated-real",
        "adapter_mode": "production",
        "database_dialect": "mysql",
        "mysql_version": "8.4.test-contract",
        "mysql_server_uuid_sha256": "d" * 64,
        "real_mysql_exercised": True,
        "real_minio_exercised": True,
        "real_model_exercised": True,
        "real_pdf_end_to_end": True,
        "independent_worker_processes_verified": True,
        "fake_adapters_used": False,
        "sqlite_used": False,
        "synthetic_artifacts_used": False,
        "attested_by": "test-contract-attestor",
        "attested_at": NOW,
    }
    payload.update(updates)
    return A7Authenticity.model_validate(payload)


def _bundle(
    evidence_type: AdmissionEvidenceType | None = None,
) -> A7EvidenceBundle:
    captures = tuple(_capture(*item) for item in expected_scenarios())
    admission_evidence = ()
    if evidence_type is not None:
        required_kinds = {
            AdmissionEvidenceType.MULTI_CONSUMER_REQUIREMENT: {
                ObservationKind.JOB_OUTCOME
            },
        }.get(evidence_type, {ObservationKind.CLAIM_LATENCY_MS})
        referenced = tuple(
            item.record_sha256
            for item in captures[0].observations
            if item.kind in required_kinds
        )
        evidence_payload = {
            "evidence_id": "test-only-admission-evidence",
            "evidence_type": evidence_type,
            "status": "verified",
            "claims_allowed": True,
            "verified_by": "test-contract-reviewer",
            "verified_at": NOW,
            "scenario_ids": (captures[0].scenario_id,),
            "observation_sha256s": referenced,
            "consumer_ids": (
                ("notification", "billing")
                if evidence_type is AdmissionEvidenceType.MULTI_CONSUMER_REQUIREMENT
                else ()
            ),
            "detail_sha256": "e" * 64,
        }
        admission_evidence = (
            seal_model(AdmissionEvidence, evidence_payload, "evidence_sha256"),
        )
    observations = [
        item.model_dump(mode="json")
        for capture in captures
        for item in capture.observations
    ]
    payload = {
        "run_id": "test-only-a7-contract-run",
        "status": RunStatus.COMPLETED,
        "binding": _binding(),
        "authenticity": _authenticity(),
        "captures": captures,
        "admission_evidence": admission_evidence,
        "collected_at": NOW + timedelta(hours=1),
        "raw_observations_sha256": stable_sha256(observations),
    }
    return seal_model(
        A7EvidenceBundle,
        payload,
        "evidence_sha256",
        attestation_key=ATTESTATION_KEY,
        attestation_key_id=ATTESTATION_KEY_ID,
    )


def _policy(bundle: A7EvidenceBundle, **threshold_updates) -> FrozenA7ThresholdPolicy:
    thresholds = {
        MetricId.MYSQL_CPU_P95_PERCENT: 50,
        MetricId.CLAIM_LATENCY_P95_MS: 10,
        MetricId.EMPTY_POLL_RATIO: 0.5,
        MetricId.LOCK_WAIT_P95_MS: 10,
        MetricId.QUEUE_LATENCY_P95_MS: 10,
        MetricId.NODE_PARSING_P95_MS: 10,
        MetricId.NODE_RETRIEVAL_P95_MS: 10,
        MetricId.NODE_REVIEW_P95_MS: 10,
        MetricId.NODE_REPORT_P95_MS: 10,
        MetricId.THROUGHPUT_JOBS_PER_MINUTE: 0.5,
        MetricId.FAILURE_RATE: 0.1,
        MetricId.RECOVERY_RATE: 0.9,
    }
    thresholds.update(threshold_updates)
    rules = tuple(
        ThresholdRule(
            metric_id=metric_id,
            operator=(
                "gte"
                if metric_id
                in {
                    MetricId.THROUGHPUT_JOBS_PER_MINUTE,
                    MetricId.RECOVERY_RATE,
                }
                else "lte"
            ),
            threshold=value,
        )
        for metric_id, value in thresholds.items()
    )
    return seal_model(
        FrozenA7ThresholdPolicy,
        {
            "policy_id": "test-only-frozen-policy",
            "baseline_evidence_sha256": bundle.evidence_sha256,
            "source_type": SourceType.REAL,
            "provenance_status": "verified",
            "claims_allowed": True,
            "approved_by": "test-contract-threshold-owner",
            "frozen_at": NOW + timedelta(hours=2),
            "rules": rules,
        },
        "policy_sha256",
        attestation_key=ATTESTATION_KEY,
        attestation_key_id=ATTESTATION_KEY_ID,
    )


def _assess(bundle, policy):
    return assess_evidence(
        bundle,
        policy,
        now=NOW,
        trusted_attestation_key=ATTESTATION_KEY,
        trusted_attestation_key_id=ATTESTATION_KEY_ID,
    )


class A7MatrixAndBoundaryTests(unittest.TestCase):
    def test_not_run_plan_has_exact_matrix_and_no_metrics(self) -> None:
        plan = create_not_run_plan(now=NOW)
        matrix = plan["matrix"]
        self.assertEqual(len(matrix), 60)
        self.assertEqual(len({item["scenario_id"] for item in matrix}), 60)
        self.assertEqual(
            {item["file_profile"] for item in matrix},
            {"TEXT_20", "MIXED_100", "SCANNED_300"},
        )
        self.assertEqual({item["concurrency"] for item in matrix}, {1, 5, 10, 20, 50})
        self.assertEqual({item["workers"] for item in matrix}, {1, 2, 5, 10})
        self.assertTrue(all(item["status"] == "NOT_RUN" for item in matrix))
        report = plan["report"]
        self.assertEqual(report["status"], "NOT_RUN")
        self.assertEqual(report["queue_decision"], "NO_DECISION")
        self.assertEqual(report["operational_action"], "KEEP_MYSQL_QUEUE")
        self.assertFalse(report["claims_allowed"])
        self.assertEqual(report["scenario_metrics"], [])

    def test_no_evidence_is_not_ready_and_cannot_change_stack(self) -> None:
        report = assess_evidence(None, None, now=NOW)
        self.assertEqual(report.status, RunStatus.NOT_READY)
        self.assertEqual(report.queue_decision, QueueDecision.NO_DECISION)
        self.assertEqual(report.redis_decision, RedisDecision.NO_DECISION)
        self.assertEqual(report.operational_action, "KEEP_MYSQL_QUEUE")
        self.assertFalse(report.automatic_stack_change_allowed)
        self.assertFalse(report.claims_allowed)

    def test_completed_evidence_rejects_fake_or_sqlite_provenance(self) -> None:
        bundle = _bundle()
        payload = bundle.model_dump(mode="json")
        payload["authenticity"].update(
            {
                "source_type": "sqlite",
                "environment_kind": "local",
                "adapter_mode": "fake",
                "database_dialect": "sqlite",
                "claims_allowed": False,
                "real_mysql_exercised": False,
                "fake_adapters_used": True,
                "sqlite_used": True,
            }
        )
        with self.assertRaisesRegex(
            ValidationError, "completed A7 evidence requires verified real provenance"
        ):
            A7EvidenceBundle.model_validate(payload)

    def test_observation_tamper_and_matrix_omission_are_rejected(self) -> None:
        bundle = _bundle()
        tampered = bundle.model_dump(mode="json")
        tampered["captures"][0]["observations"][0]["value"] = 99
        with self.assertRaisesRegex(ValidationError, "record_sha256"):
            A7EvidenceBundle.model_validate(tampered)

        incomplete = bundle.model_dump(mode="json")
        incomplete["captures"].pop()
        incomplete["evidence_sha256"] = stable_sha256(
            {
                key: value
                for key, value in incomplete.items()
                if key != "evidence_sha256"
            }
        )
        with self.assertRaisesRegex(ValidationError, "exact 60-scenario matrix"):
            A7EvidenceBundle.model_validate(incomplete)


class A7DecisionTests(unittest.TestCase):
    def test_all_frozen_thresholds_pass_keeps_mysql_queue(self) -> None:
        bundle = _bundle()
        report = _assess(bundle, _policy(bundle))
        self.assertEqual(report.status, RunStatus.COMPLETED)
        self.assertEqual(report.queue_decision, QueueDecision.KEEP_MYSQL_QUEUE)
        self.assertEqual(report.redis_decision, RedisDecision.KEEP_REDIS_OUT)
        self.assertTrue(report.claims_allowed)
        self.assertEqual(len(report.scenario_metrics), 60)
        self.assertEqual(len(report.threshold_assessments), 60 * len(MetricId))
        self.assertTrue(all(item.passed for item in report.threshold_assessments))
        self.assertFalse(report.automatic_stack_change_allowed)

    def test_failed_threshold_without_verified_root_cause_is_no_decision(self) -> None:
        bundle = _bundle()
        report = _assess(
            bundle, _policy(bundle, **{MetricId.CLAIM_LATENCY_P95_MS: 0.5})
        )
        self.assertEqual(report.status, RunStatus.NOT_READY)
        self.assertEqual(report.queue_decision, QueueDecision.NO_DECISION)
        self.assertFalse(report.claims_allowed)
        self.assertEqual(report.scenario_metrics, ())

    def test_frozen_threshold_tamper_is_rejected(self) -> None:
        bundle = _bundle()
        payload = _policy(bundle).model_dump(mode="json")
        payload["rules"][0]["threshold"] = 99
        with self.assertRaisesRegex(ValidationError, "policy_sha256"):
            FrozenA7ThresholdPolicy.model_validate(payload)

    def test_forged_attestation_is_rejected(self) -> None:
        bundle = _bundle()
        forged = bundle.model_copy(update={"attestation_hmac_sha256": "0" * 64})
        with self.assertRaisesRegex(ValueError, "HMAC verification failed"):
            _assess(forged, _policy(bundle))

    def test_rehashed_report_without_trusted_hmac_is_rejected(self) -> None:
        bundle = _bundle()
        payload = _assess(bundle, _policy(bundle)).model_dump(mode="json")
        payload["generated_at"] = "2026-08-06T08:01:00Z"
        payload["report_sha256"] = stable_sha256(
            {
                key: value
                for key, value in payload.items()
                if key not in {"report_sha256", "attestation_hmac_sha256"}
            }
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "forged-report.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "HMAC verification failed"):
                load_report(
                    path,
                    trusted_attestation_key=ATTESTATION_KEY,
                    trusted_attestation_key_id=ATTESTATION_KEY_ID,
                )

    def test_verified_multi_consumer_need_only_proposes_separate_admission(
        self,
    ) -> None:
        bundle = _bundle(AdmissionEvidenceType.MULTI_CONSUMER_REQUIREMENT)
        report = _assess(bundle, None)
        self.assertEqual(
            report.queue_decision, QueueDecision.PROPOSE_ROCKETMQ_ADMISSION
        )
        self.assertEqual(report.operational_action, "KEEP_MYSQL_QUEUE")
        self.assertFalse(report.automatic_stack_change_allowed)
        self.assertIn("verified:MULTI_CONSUMER_REQUIREMENT", report.decision_reasons)


class A7ApiCliAndWorkbenchTests(unittest.TestCase):
    def test_api_defaults_to_read_only_not_run_report(self) -> None:
        container = build_container(
            AppSettings(environment="test", adapter_mode="fake", log_json=False)
        )
        try:
            with TestClient(create_app(container)) as client:
                response = client.get("/api/v1/a7/admission-report")
        finally:
            container.close()
        self.assertEqual(response.status_code, 200, response.text)
        body = response.json()
        self.assertEqual(body["status"], "NOT_RUN")
        self.assertEqual(body["matrix_completed"], 0)
        self.assertEqual(body["scenario_metrics"], [])
        self.assertEqual(body["queue_decision"], "NO_DECISION")
        self.assertEqual(body["operational_action"], "KEEP_MYSQL_QUEUE")

    def test_cli_plan_is_not_run_and_workbench_is_read_only(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "plan.json"
            self.assertEqual(a7_cli_main(["plan", "--output", str(output)]), 0)
            payload = json.loads(output.read_text(encoding="utf-8"))
        self.assertEqual(payload["status"], "NOT_RUN")
        self.assertEqual(payload["report"]["scenario_metrics"], [])

        client_source = (
            PROJECT_DIR / "workbench" / "src" / "api" / "client.js"
        ).read_text(encoding="utf-8")
        app_source = (PROJECT_DIR / "workbench" / "src" / "App.vue").read_text(
            encoding="utf-8"
        )
        self.assertIn("a7AdmissionReport", client_source)
        self.assertIn("/a7/admission-report", client_source)
        self.assertIn("a7AdmissionReport.operational_action", app_source)
        self.assertNotIn("createA7", client_source)
        self.assertNotIn("updateA7", client_source)

    def test_worker_emits_claim_latency_and_empty_poll_observations(self) -> None:
        stream = io.StringIO()
        handler = logging.StreamHandler(stream)
        handler.setFormatter(JsonFormatter())
        logger = logging.getLogger("tender_review.worker")
        previous_handlers = tuple(logger.handlers)
        previous_level = logger.level
        previous_propagate = logger.propagate
        logger.handlers.clear()
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
        try:
            worker = Worker(
                worker_id="a7-test-worker",
                repository=InMemoryJobRepository(),
                leases=FakeLeaseManager(),
                handlers={},
                clock=SystemClock(),
                poll_interval_seconds=0,
            )
            self.assertFalse(worker.run_once())
        finally:
            logger.handlers.clear()
            for previous in previous_handlers:
                logger.addHandler(previous)
            logger.setLevel(previous_level)
            logger.propagate = previous_propagate

        observations = [json.loads(line) for line in stream.getvalue().splitlines()]
        by_name = {item["metric_name"]: item for item in observations}
        self.assertEqual(by_name["worker_empty_poll"]["metric_value"], 1)
        self.assertEqual(by_name["worker_empty_poll"]["metric_unit"], "boolean")
        self.assertGreaterEqual(by_name["job_claim_latency"]["metric_value"], 0)
        self.assertEqual(
            by_name["job_claim_latency"]["metric_source"],
            "worker_process_monotonic",
        )


if __name__ == "__main__":
    unittest.main()
