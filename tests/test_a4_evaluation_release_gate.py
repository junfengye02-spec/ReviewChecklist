from __future__ import annotations

import unittest
from datetime import datetime, timezone

from fastapi.testclient import TestClient
from pydantic import ValidationError
from sqlalchemy import create_engine

from tender_review.api.app import create_app
from tender_review.bootstrap.assembly import build_container
from tender_review.evaluation.public import (
    CompleteEvaluationRun,
    CreateEvaluationRun,
    EngineeringMetrics,
    EvaluationDatasetSnapshot,
    EvaluationMetrics,
    EvaluationPurpose,
    EvaluationResult,
    EvaluationRunBinding,
    EvaluationRunService,
    EvaluationRunStatus,
    EvaluationSourceType,
    FreezeThresholdPolicy,
    InMemoryEvaluationRunRepository,
    RetrievalMetrics,
    ReviewMetrics,
    StabilityMetrics,
    ThresholdRule,
)
from tender_review.infrastructure.database import Base, create_session_factory
from tender_review.infrastructure.database.rule_versions import SqlAlchemyRuleVersionRepository
from tender_review.rule_management.public import (
    CompleteEvaluationGate,
    CreateRuleVersion,
    PublishRuleVersion,
    RuleProvenance,
    RuleVersionService,
)
from tender_review.shared.clock import FixedClock
from tender_review.shared.config import AppSettings
from tender_review.shared.errors import PermanentError, ServiceError
from tender_review.shared.ids import SequentialIdGenerator


NOW = datetime(2026, 8, 6, 14, 0, tzinfo=timezone.utc)


class SnapshotResolver:
    def __init__(self, snapshot: EvaluationDatasetSnapshot) -> None:
        self.snapshot = snapshot

    def resolve(self, dataset_version_id: str) -> EvaluationDatasetSnapshot:
        if dataset_version_id != self.snapshot.dataset_version_id:
            raise AssertionError("unexpected dataset")
        return self.snapshot


class AllowingVerifier:
    def assert_dataset_release_ready(self, dataset_version_id: str) -> None:
        del dataset_version_id

    def assert_release_eligible(self, **identity: str) -> None:
        del identity


def _snapshot(
    *,
    source_type: EvaluationSourceType = EvaluationSourceType.REAL,
    status: str = "FROZEN",
    provenance_status: str = "verified",
    claims_allowed: bool = True,
    required: int = 4,
    verified: int = 4,
) -> EvaluationDatasetSnapshot:
    return EvaluationDatasetSnapshot(
        dataset_version_id="dataset-a4",
        manifest_sha256="a" * 64,
        source_type=source_type,
        status=status,
        provenance_status=provenance_status,
        claims_allowed=claims_allowed,
        required_human_cases=required,
        independently_verified_cases=verified,
        frozen_test_cases=1,
    )


def _metrics() -> EvaluationMetrics:
    return EvaluationMetrics(
        retrieval=RetrievalMetrics(
            evidence_recall_at_5=0.8,
            evidence_recall_at_10=0.9,
            mrr=0.7,
            cross_section_bilateral_hit_rate=0.75,
            no_answer_false_retrieval_rate=0.05,
        ),
        review=ReviewMetrics(
            precision=0.86,
            recall=0.84,
            f1=0.85,
            false_positive_rate=0.04,
            false_negative_rate=0.06,
            evidence_completeness_rate=0.9,
            evidence_conclusion_consistency_rate=0.92,
        ),
        stability=StabilityMetrics(
            repeated_run_consistency_rate=0.98,
            model_exception_rate=0.01,
            human_handoff_rate=0.08,
        ),
        engineering=EngineeringMetrics(
            task_success_rate=0.99,
            worker_recovery_success_rate=0.95,
            latency_p50_ms=1500,
            latency_p95_ms=3200,
            token_usage=125000,
            cost_per_document=0.42,
        ),
    )


def _result(binding_sha256: str) -> EvaluationResult:
    payload = {
        "schema_version": 1,
        "binding_sha256": binding_sha256,
        "metrics": _metrics().model_dump(mode="json"),
        "failure_samples": (),
        "difference_sources": (),
        "case_results_sha256": "b" * 64,
        "repeated_runs_sha256": "c" * 64,
        "engineering_telemetry_sha256": "d" * 64,
    }
    from tender_review.evaluation.runs import stable_sha256

    return EvaluationResult(**payload, result_sha256=stable_sha256(payload))


def _create_command(rule_version_id: str, purpose: EvaluationPurpose, *, split: str = "FROZEN_TEST") -> CreateEvaluationRun:
    return CreateEvaluationRun(
        rule_version_id=rule_version_id,
        dataset_version_id="dataset-a4",
        purpose=purpose,
        dataset_split=split,
        model_config_id="model-config-a4",
        retriever_version="hybrid-rrf-v1",
        evaluator_version="a4-evaluator-v1",
        input_sha256="1" * 64,
        config_sha256="2" * 64,
        code_sha256="3" * 64,
        model_sha256="4" * 64,
        prompt_sha256="5" * 64,
        reproducibility_command=(
            "python -m tender_review.evaluation a4-verify "
            "--run run.json --report report.json"
        ),
    )


class A4EvaluationDomainTests(unittest.TestCase):
    def _system(self, snapshot: EvaluationDatasetSnapshot):
        container = build_container(AppSettings(environment="test", log_json=False))
        evaluations = EvaluationRunService(
            InMemoryEvaluationRunRepository(),
            SnapshotResolver(snapshot),
            container.rule_version_repository,
            SequentialIdGenerator(prefix="evaluation"),
            FixedClock(NOW),
        )
        container.rule_version_repository.set_release_gate_verifier(evaluations)
        rules = RuleVersionService(
            container.rule_version_repository,
            SequentialIdGenerator(prefix="rule"),
            FixedClock(NOW),
            evaluations,
        )
        version = rules.create_version(CreateRuleVersion(
            rule_set_id="rule-set-a4",
            rule_key="a4-release-rule",
            rule_set_name="A4 release rule",
            content_json='{"threshold":10}',
            change_summary="A4 test fixture",
            provenance=RuleProvenance(
                source_type="manual", status="verified", claims_allowed=True
            ),
        ))
        return container, evaluations, rules, version

    def test_unready_or_forbidden_dataset_produces_no_metrics(self) -> None:
        unready = _snapshot(
            status="DRAFT",
            provenance_status="provisional",
            claims_allowed=False,
            required=4,
            verified=0,
        )
        container, evaluations, _, version = self._system(unready)
        try:
            run = evaluations.create(_create_command(version.rule_version_id, EvaluationPurpose.RELEASE_GATE))
            report = evaluations.get_report(run.run_id)
            self.assertEqual(run.status, EvaluationRunStatus.NOT_READY)
            self.assertFalse(run.claims_allowed)
            self.assertIn("dataset status is not FROZEN", run.blockers)
            self.assertTrue(all(value is None for value in report.metrics.values().values()))
            self.assertFalse(report.release_gate.eligible)
            self.assertFalse(report.release_gate.passed)
            with self.assertRaises(ServiceError):
                evaluations.complete(CompleteEvaluationRun(run_id=run.run_id, result=_result(run.binding.binding_sha256)))
        finally:
            container.close()

        for source_type in (
            EvaluationSourceType.PROVISIONAL,
            EvaluationSourceType.SYNTHETIC,
            EvaluationSourceType.EXTERNAL_PLATFORM,
        ):
            with self.subTest(source_type=source_type):
                container, evaluations, _, version = self._system(
                    _snapshot(source_type=source_type, claims_allowed=False)
                )
                try:
                    run = evaluations.create(
                        _create_command(
                            version.rule_version_id,
                            EvaluationPurpose.RELEASE_GATE,
                        )
                    )
                    self.assertEqual(run.status, EvaluationRunStatus.NOT_READY)
                    self.assertIn("dataset source_type is not real", run.blockers)
                    self.assertFalse(evaluations.get_report(run.run_id).claims_allowed)
                finally:
                    container.close()

        container, evaluations, _, version = self._system(_snapshot())
        try:
            run = evaluations.create(
                _create_command(
                    version.rule_version_id,
                    EvaluationPurpose.CANDIDATE_DIAGNOSTIC,
                    split="FROZEN_TEST",
                )
            )
            self.assertEqual(run.status, EvaluationRunStatus.NOT_READY)
            self.assertIn("frozen-test is forbidden for candidate generation or diagnostics", run.blockers)
        finally:
            container.close()

    def test_binding_and_result_hash_tampering_is_rejected(self) -> None:
        container, evaluations, _, version = self._system(_snapshot())
        try:
            run = evaluations.create(_create_command(version.rule_version_id, EvaluationPurpose.REAL_BASELINE))
            tampered = run.binding.model_dump(mode="json")
            tampered["prompt_sha256"] = "9" * 64
            with self.assertRaisesRegex(ValidationError, "binding_sha256"):
                EvaluationRunBinding.model_validate(tampered)
            result = _result(run.binding.binding_sha256).model_dump(mode="json")
            result["metrics"]["review"]["f1"] = 0.99
            with self.assertRaisesRegex(ValidationError, "result_sha256"):
                EvaluationResult.model_validate(result)
        finally:
            container.close()

    def test_real_baseline_freezes_thresholds_before_release_can_publish(self) -> None:
        container, evaluations, rules, version = self._system(_snapshot())
        try:
            baseline = evaluations.create(_create_command(version.rule_version_id, EvaluationPurpose.REAL_BASELINE))
            evaluations.complete(CompleteEvaluationRun(run_id=baseline.run_id, result=_result(baseline.binding.binding_sha256)))
            baseline_report = evaluations.get_report(baseline.run_id)
            self.assertTrue(baseline_report.claims_allowed)
            self.assertFalse(baseline_report.release_gate.passed)

            threshold_rules = tuple(
                ThresholdRule(
                    metric_id=metric_id,
                    operator="lte" if any(token in metric_id for token in ("false_", "exception", "handoff", "latency", "token", "cost")) else "gte",
                    threshold=float(value),
                    baseline_value=float(value),
                )
                for metric_id, value in baseline_report.metrics.values().items()
                if value is not None
            )
            policy = evaluations.freeze_threshold_policy(FreezeThresholdPolicy(
                baseline_run_id=baseline.run_id,
                approved_by="quality-owner-li",
                rules=threshold_rules,
            ))
            release = evaluations.create(_create_command(version.rule_version_id, EvaluationPurpose.RELEASE_GATE))
            release = evaluations.complete(CompleteEvaluationRun(
                run_id=release.run_id,
                result=_result(release.binding.binding_sha256),
                threshold_policy_id=policy.policy_id,
            ))
            report = evaluations.get_report(release.run_id)
            self.assertTrue(report.release_gate.passed)
            evaluating = rules.request_evaluation(version.rule_version_id, "dataset-a4")
            rules.complete_evaluation(CompleteEvaluationGate(
                rule_version_id=version.rule_version_id,
                gate_id=evaluating.evaluation_gate.gate_id,
                evaluation_run_id=release.run_id,
                status="PASSED",
                provisional=False,
                claims_allowed=True,
                report_sha256=report.report_sha256,
            ))
            published = rules.publish(PublishRuleVersion(
                rule_version_id=version.rule_version_id,
                approver_id="release-owner-zhang",
            ))
            self.assertEqual(published.status.value, "PUBLISHED")
        finally:
            container.close()


class A4ApiAndTransactionTests(unittest.TestCase):
    def test_api_exposes_not_ready_run_without_inventing_metrics(self) -> None:
        base = build_container(AppSettings(environment="test", log_json=False))
        snapshot = _snapshot(status="DRAFT", provenance_status="provisional", claims_allowed=False, verified=0)
        evaluations = EvaluationRunService(
            base.evaluation_run_repository,
            SnapshotResolver(snapshot),
            base.rule_version_repository,
            SequentialIdGenerator(prefix="api-evaluation"),
            FixedClock(NOW),
        )
        container = base.with_overrides(evaluations=evaluations)
        with TestClient(create_app(container)) as client:
            created_rule = client.post("/api/v1/rule-sets/api-a4/versions", json={
                "rule_key": "api-a4",
                "rule_set_name": "API A4",
                "content": {"threshold": 10},
                "change_summary": "API A4 fixture",
                "provenance": {"source_type": "manual", "status": "verified", "claims_allowed": True},
            })
            version_id = created_rule.json()["rule_version_id"]
            payload = _create_command(version_id, EvaluationPurpose.RELEASE_GATE).model_dump(mode="json", exclude={"rule_version_id"})
            response = client.post(f"/api/v1/a4/rule-versions/{version_id}/evaluation-runs", json=payload)
            self.assertEqual(response.status_code, 201, response.text)
            self.assertEqual(response.json()["status"], "NOT_READY")
            report = client.get(f"/api/v1/a4/evaluation-runs/{response.json()['run_id']}/report")
            self.assertEqual(report.status_code, 200, report.text)
            values = report.json()["metrics"]
            self.assertTrue(all(
                value is None
                for group_name, group in values.items()
                if group_name != "schema_version"
                for field_name, value in group.items()
                if field_name != "schema_version"
            ))
            self.assertFalse(report.json()["release_gate"]["eligible"])

    def test_sql_transaction_rejects_direct_repository_publish_without_a4_rows(self) -> None:
        engine = create_engine("sqlite:///:memory:")
        Base.metadata.create_all(engine)
        sessions = create_session_factory(engine)
        repository = SqlAlchemyRuleVersionRepository(sessions)
        service = RuleVersionService(
            repository,
            SequentialIdGenerator(prefix="db-rule"),
            FixedClock(NOW),
            AllowingVerifier(),
        )
        version = service.create_version(CreateRuleVersion(
            rule_set_id="db-rule-set-a4",
            rule_key="db-a4",
            rule_set_name="DB A4",
            content_json="{}",
            change_summary="transaction bypass fixture",
            provenance=RuleProvenance(source_type="manual", status="verified", claims_allowed=True),
        ))
        evaluating = service.request_evaluation(version.rule_version_id, "missing-dataset")
        waiting = service.complete_evaluation(CompleteEvaluationGate(
            rule_version_id=version.rule_version_id,
            gate_id=evaluating.evaluation_gate.gate_id,
            evaluation_run_id="missing-run",
            status="PASSED",
            provisional=False,
            claims_allowed=True,
            report_sha256="f" * 64,
        ))
        candidate = waiting.model_copy(update={
            "status": "PUBLISHED",
            "published_at": NOW,
            "published_by": "bypass-actor",
        })
        with self.assertRaisesRegex(PermanentError, "persisted release evidence"):
            repository.publish(repository.get_rule_set(version.rule_set_id), candidate)
        self.assertIsNone(repository.get_rule_set(version.rule_set_id).current_version_id)
        self.assertEqual(repository.get_version(version.rule_version_id).status.value, "WAITING_APPROVAL")
        engine.dispose()


if __name__ == "__main__":
    unittest.main()
