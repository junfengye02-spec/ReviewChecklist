from __future__ import annotations

import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

from langgraph.checkpoint.memory import InMemorySaver
from alembic import command as alembic_command
from alembic.config import Config as AlembicConfig
from pydantic import ValidationError
from sqlalchemy import create_engine, inspect
from sqlalchemy.orm import sessionmaker

from tender_review.api.schemas import A5OptimizeRuleVersionRequest
from tender_review.evaluation.public import (
    CreateDatasetVersion,
    DatasetProvenance,
    DatasetSampleInput,
    DatasetSourceType,
    DatasetSplit,
    DatasetVersionService,
    InMemoryAnnotationDatasetRepository,
    InMemoryDatasetVersionRepository,
    InMemoryEvaluationRunRepository,
    AnnotationSampleStatus,
    DatasetStatus,
    EvaluationPurpose,
    EvaluationRunStatus,
    EvaluationSourceType,
)
from tender_review.infrastructure.database.base import Base
from tender_review.infrastructure.database.optimization_jobs import (
    SqlAlchemyOptimizationRepository,
)
from tender_review.optimization.public import (
    ALLOWED_CHANGE_PATHS,
    A4OptimizationReadinessVerifier,
    CandidateChange,
    CandidateType,
    CLASSIFY_ROOT_CAUSE,
    GENERATE_MINIMAL_CANDIDATE,
    LOAD_FAILURE_SAMPLES,
    RUN_PROTECTION_GATE,
    RUN_STABILITY_GATE,
    RUN_TARGET_GATE,
    STAGE_DRAFT_RULE,
    WAIT_FOR_HUMAN_APPROVAL,
    CreateOptimizationJob,
    FailureSignals,
    FakeCandidateGenerator,
    FakeRegressionEvaluator,
    InMemoryOptimizationRepository,
    OptimizationProvenance,
    OptimizationReadiness,
    OptimizationReadinessStatus,
    OptimizationSample,
    OptimizationService,
    OptimizationStatus,
    RootCause,
    RootCauseAnalyzer,
    RuleVersionCandidateStager,
    SampleRole,
    SourceArtifact,
    SourceType,
)
from tender_review.review.public import FakeLlmProvider
from tender_review.rule_management.public import (
    CreateRuleVersion,
    InMemoryRuleVersionRepository,
    PublishRuleVersion,
    RollbackRuleSet,
    RuleProvenance,
    RuleVersionService,
)
from tender_review.shared.clock import FixedClock
from tender_review.shared.errors import ConflictError, PermanentError
from tender_review.shared.ids import SequentialIdGenerator


NOW = datetime(2026, 8, 6, 15, 0, tzinfo=timezone.utc)


class LocalReadyVerifier:
    """Exercises A5 control flow without claiming project A3/A4 readiness."""

    def __init__(self, dataset_manifest_sha256: str) -> None:
        self.dataset_manifest_sha256 = dataset_manifest_sha256

    def assess(self, command: CreateOptimizationJob) -> OptimizationReadiness:
        return OptimizationReadiness(
            status=OptimizationReadinessStatus.READY,
            claims_allowed=True,
            dataset_manifest_sha256=self.dataset_manifest_sha256,
            a4_evaluation_run_id=command.a4_evaluation_run_id,
            a4_run_sha256="8" * 64,
            a4_report_sha256=command.a4_report_sha256,
            a4_binding_sha256="9" * 64,
            a4_result_sha256="a" * 64,
            verified_failure_sample_ids=("target-1",),
            dataset_sample_ids=("protection-1", "target-1"),
            assessed_at=NOW,
        )


class A5Fixture:
    def __init__(
        self,
        *,
        signals: FailureSignals | None = None,
        plans=(),
        max_rounds: int = 2,
        candidates_per_round: int = 2,
        llm=None,
        readiness=None,
    ) -> None:
        self.clock = FixedClock(NOW)
        self.rule_repository = InMemoryRuleVersionRepository()
        self.rule_service = RuleVersionService(
            self.rule_repository,
            SequentialIdGenerator(prefix="a5-rule"),
            self.clock,
        )
        self.base = self.rule_service.create_version(
            CreateRuleVersion(
                rule_set_id="a5-set",
                rule_key="qualification",
                rule_set_name="A5 local contract",
                content_json='{"rule_text":"base"}',
                execution_config_json="{}",
                change_summary="test-only immutable base",
                provenance=RuleProvenance(
                    source_type="manual", status="verified", claims_allowed=True
                ),
            )
        )
        self.dataset_repository = InMemoryDatasetVersionRepository()
        self.dataset = DatasetVersionService(
            self.dataset_repository,
            SequentialIdGenerator(prefix="a5-dataset"),
            self.clock,
        ).create_version(
            CreateDatasetVersion(
                dataset_name="a5-test-only-ready",
                requested_status="FROZEN",
                change_summary="test-only state-machine fixture",
                provenance=DatasetProvenance(
                    status="verified",
                    claims_allowed=True,
                    source_description="test-only; not project labels or metrics",
                    source_manifest_sha256="1" * 64,
                ),
                samples=(
                    _dataset_sample("target-1", DatasetSplit.OPTIMIZATION, "1"),
                    _dataset_sample("protection-1", DatasetSplit.VALIDATION, "2"),
                ),
            )
        )
        self.repository = InMemoryOptimizationRepository()
        self.generator = FakeCandidateGenerator(
            base_content_json=self.base.content_json,
            base_execution_config_json=self.base.execution_config_json,
        )
        self.evaluator = FakeRegressionEvaluator(plans)
        self.llm = llm or FakeLlmProvider()
        self.ids = SequentialIdGenerator(prefix="a5-optimization")
        self.checkpointer = InMemorySaver()
        self.readiness = readiness or LocalReadyVerifier(
            self.dataset.manifest_sha256
        )
        self.command = CreateOptimizationJob(
            base_rule_version_id=self.base.rule_version_id,
            dataset_version_id=self.dataset.dataset_version_id,
            max_rounds=max_rounds,
            candidates_per_round=candidates_per_round,
            required_stability_runs=2,
            model_sha256="b" * 64,
            prompt_sha256="c" * 64,
            retriever_sha256="d" * 64,
            tool_sha256="e" * 64,
            a4_evaluation_run_id="a4-test-only-run",
            a4_report_sha256="7" * 64,
            samples=(
                _optimization_sample(
                    "target-1",
                    SampleRole.TARGET,
                    "1",
                    signals
                    or FailureSignals(
                        failure_summary="verified local contract failure",
                        evidence_in_top_k=True,
                        extraction_matches_expected=True,
                        tool_matches_expected=True,
                        repeated_outputs_consistent=True,
                    ),
                ),
                _optimization_sample(
                    "protection-1", SampleRole.PROTECTION, "2", None
                ),
            ),
            provenance=OptimizationProvenance(
                source_type=SourceType.REAL,
                status="verified",
                claims_allowed=True,
                source_description="test-only A5 state-machine evidence",
                source_artifacts=(
                    SourceArtifact(
                        path="tests/test_a5_langgraph_optimization.py",
                        sha256="f" * 64,
                        kind="manifest",
                    ),
                ),
                human_annotation_cases=2,
                required_human_cases=2,
            ),
        )
        self.service = self._service()

    def _service(self) -> OptimizationService:
        return OptimizationService(
            repository=self.repository,
            rule_versions=self.rule_repository,
            datasets=self.dataset_repository,
            ids=self.ids,
            clock=self.clock,
            root_causes=RootCauseAnalyzer(self.llm),
            candidates=self.generator,
            evaluator=self.evaluator,
            stager=RuleVersionCandidateStager(
                self.rule_service, self.rule_repository
            ),
            readiness=self.readiness,
            checkpointer=self.checkpointer,
        )


def _dataset_sample(
    sample_id: str, split: DatasetSplit, digest_digit: str
) -> DatasetSampleInput:
    return DatasetSampleInput(
        sample_id=sample_id,
        finding_id=f"finding-{sample_id}",
        human_decision_id=f"decision-{sample_id}",
        document_id=f"document-{sample_id}",
        document_sha256=digest_digit * 64,
        split=split,
        source_type=DatasetSourceType.REAL,
        provenance_status="verified",
        label_version="test-only-v1",
        label_json='{"label":"test-only"}',
        review_input_sha256=("3" if sample_id == "target-1" else "4") * 64,
        evidence_sha256=("5" if sample_id == "target-1" else "6") * 64,
    )


def _optimization_sample(
    sample_id: str,
    role: SampleRole,
    digest_digit: str,
    signals: FailureSignals | None,
) -> OptimizationSample:
    target = role is SampleRole.TARGET
    return OptimizationSample(
        sample_id=sample_id,
        role=role,
        document_id=f"document-{sample_id}",
        document_sha256=digest_digit * 64,
        source_type=SourceType.REAL,
        provenance_status="verified",
        claims_allowed=True,
        source_reference=f"test-only://a5/{sample_id}",
        review_input_sha256=("3" if target else "4") * 64,
        evidence_sha256=("5" if target else "6") * 64,
        finding_id=f"finding-{sample_id}",
        human_decision_id=f"decision-{sample_id}",
        signals=signals,
    )


class A5TopologyAndRouteTests(unittest.TestCase):
    def test_graph_contains_the_bounded_a5_topology_and_only_expected_loops(self):
        graph = A5Fixture().service.compiled_graph.get_graph()
        actual = {(edge.source, edge.target) for edge in graph.edges}
        expected = {
            ("__start__", LOAD_FAILURE_SAMPLES),
            (LOAD_FAILURE_SAMPLES, CLASSIFY_ROOT_CAUSE),
            (LOAD_FAILURE_SAMPLES, "__end__"),
            (CLASSIFY_ROOT_CAUSE, GENERATE_MINIMAL_CANDIDATE),
            (CLASSIFY_ROOT_CAUSE, LOAD_FAILURE_SAMPLES),
            (CLASSIFY_ROOT_CAUSE, "__end__"),
            (GENERATE_MINIMAL_CANDIDATE, RUN_TARGET_GATE),
            (GENERATE_MINIMAL_CANDIDATE, LOAD_FAILURE_SAMPLES),
            (GENERATE_MINIMAL_CANDIDATE, "__end__"),
            (RUN_TARGET_GATE, RUN_PROTECTION_GATE),
            (RUN_TARGET_GATE, LOAD_FAILURE_SAMPLES),
            (RUN_TARGET_GATE, "__end__"),
            (RUN_PROTECTION_GATE, RUN_STABILITY_GATE),
            (RUN_STABILITY_GATE, STAGE_DRAFT_RULE),
            (RUN_STABILITY_GATE, GENERATE_MINIMAL_CANDIDATE),
            (RUN_STABILITY_GATE, LOAD_FAILURE_SAMPLES),
            (RUN_STABILITY_GATE, "__end__"),
            (STAGE_DRAFT_RULE, WAIT_FOR_HUMAN_APPROVAL),
            (STAGE_DRAFT_RULE, LOAD_FAILURE_SAMPLES),
            (STAGE_DRAFT_RULE, "__end__"),
            (WAIT_FOR_HUMAN_APPROVAL, "__end__"),
        }
        self.assertEqual(actual, expected)

    def test_all_root_causes_route_to_only_their_allowed_minimal_patch(self):
        cases = (
            (FailureSignals(failure_summary="miss", evidence_in_top_k=False), RootCause.RETRIEVAL_MISS),
            (FailureSignals(failure_summary="extract", evidence_in_top_k=True, extraction_matches_expected=False), RootCause.EXTRACTION_ERROR),
            (FailureSignals(failure_summary="tool", evidence_in_top_k=True, extraction_matches_expected=True, tool_matches_expected=False), RootCause.TOOL_ERROR),
            (FailureSignals(failure_summary="gap", evidence_in_top_k=True, extraction_matches_expected=True, tool_matches_expected=True, repeated_outputs_consistent=True), RootCause.RULE_GAP),
            (FailureSignals(failure_summary="unstable", repeated_outputs_consistent=False), RootCause.MODEL_INSTABILITY),
        )
        for signals, root_cause in cases:
            with self.subTest(root_cause=root_cause):
                fixture = A5Fixture(signals=signals)
                job = fixture.service.create(fixture.command)
                fixture.service.run(
                    job.optimization_job_id,
                    interrupt_after=(GENERATE_MINIMAL_CANDIDATE,),
                )
                attempt = fixture.service.list_attempts(job.optimization_job_id)[0]
                candidate = attempt.candidates[0]
                self.assertEqual(candidate.root_cause, root_cause)
                self.assertTrue(
                    any(
                        candidate.change.scope == scope
                        and candidate.change.path.startswith(path)
                        for scope, path in ALLOWED_CHANGE_PATHS[candidate.candidate_type]
                    )
                )

    def test_similar_prefix_cannot_escape_the_candidate_change_scope(self):
        fixture = A5Fixture(
            signals=FailureSignals(
                failure_summary="unstable", repeated_outputs_consistent=False
            )
        )
        job = fixture.service.create(fixture.command)
        decision = RootCauseAnalyzer().analyze(job, 1)
        candidate = fixture.generator.generate(job, 1, decision, 1)[0]
        payload = candidate.model_dump(mode="json")
        payload["candidate_type"] = CandidateType.STABILITY_CONFIG
        payload["change"] = CandidateChange(
            scope="execution_config",
            path="$.model.seed_override",
            before_json=None,
            after_json="1",
        ).model_dump(mode="json")

        with self.assertRaisesRegex(ValidationError, "exceeds"):
            type(candidate).model_validate(payload)

    def test_label_uncertain_routes_to_human_without_generating_a_candidate(self):
        fixture = A5Fixture(
            signals=FailureSignals(
                failure_summary="conflicting verified labels", label_conflict=True
            )
        )
        job = fixture.service.create(fixture.command)
        completed = fixture.service.run(job.optimization_job_id)
        attempt = fixture.service.list_attempts(job.optimization_job_id)[0]

        self.assertEqual(completed.status, OptimizationStatus.WAITING_HUMAN)
        self.assertEqual(attempt.candidates, ())
        self.assertEqual(fixture.generator.calls, [])


class A5ExecutionBoundaryTests(unittest.TestCase):
    def test_three_gates_stage_only_a_draft_and_stop_at_human_approval(self):
        fixture = A5Fixture(plans=((True, True, True),), candidates_per_round=1)
        job = fixture.service.create(fixture.command)
        completed = fixture.service.run(job.optimization_job_id)
        candidate = fixture.rule_repository.get_version(
            completed.candidate_rule_version_id
        )

        self.assertEqual(completed.status, OptimizationStatus.WAITING_APPROVAL)
        self.assertEqual(candidate.status.value, "DRAFT")
        self.assertIsNone(candidate.evaluation_gate)
        self.assertEqual(
            [item.node for item in completed.graph_trace],
            [
                LOAD_FAILURE_SAMPLES,
                CLASSIFY_ROOT_CAUSE,
                GENERATE_MINIMAL_CANDIDATE,
                RUN_TARGET_GATE,
                RUN_PROTECTION_GATE,
                RUN_STABILITY_GATE,
                STAGE_DRAFT_RULE,
                WAIT_FOR_HUMAN_APPROVAL,
            ],
        )
        with self.assertRaisesRegex(ConflictError, "not waiting for approval"):
            fixture.rule_service.publish(
                PublishRuleVersion(
                    rule_version_id=candidate.rule_version_id,
                    approver_id="Zhang Reviewer",
                )
            )
        with self.assertRaisesRegex(ConflictError, "no published version"):
            fixture.rule_service.rollback(
                RollbackRuleSet(
                    rule_set_id="a5-set",
                    target_version_id=fixture.base.rule_version_id,
                    approver_id="Zhang Reviewer",
                    reason="test boundary",
                )
            )

    def test_round_and_candidate_limits_end_in_optimization_failed(self):
        fixture = A5Fixture(
            plans=((False, True, True),) * 4,
            max_rounds=2,
            candidates_per_round=2,
        )
        job = fixture.service.create(fixture.command)
        completed = fixture.service.run(job.optimization_job_id)

        self.assertEqual(completed.status, OptimizationStatus.OPTIMIZATION_FAILED)
        self.assertEqual(completed.current_round, 2)
        self.assertEqual(len(fixture.evaluator.calls), 4)
        self.assertEqual(
            [len(item.candidates) for item in fixture.service.list_attempts(job.optimization_job_id)],
            [2, 2],
        )

    def test_checkpoint_resume_does_not_repeat_the_root_cause_model_call(self):
        llm = FakeLlmProvider(
            ('{"rationale":"semantic gap","root_cause":"RULE_GAP"}',)
        )
        fixture = A5Fixture(
            signals=FailureSignals(failure_summary="unresolved semantics"),
            plans=((True, True, True),),
            candidates_per_round=1,
            llm=llm,
        )
        job = fixture.service.create(fixture.command)
        fixture.service.run(
            job.optimization_job_id,
            interrupt_after=(CLASSIFY_ROOT_CAUSE,),
        )
        self.assertEqual(len(llm.calls), 1)

        fixture.service = fixture._service()
        completed = fixture.service.run(job.optimization_job_id)

        self.assertEqual(completed.status, OptimizationStatus.WAITING_APPROVAL)
        self.assertEqual(len(llm.calls), 1)
        self.assertEqual(
            sum(item.node == CLASSIFY_ROOT_CAUSE for item in completed.graph_trace), 1
        )


class A5ReadinessAndBypassTests(unittest.TestCase):
    def test_persisted_real_a3_a4_failure_evidence_can_become_ready(self):
        fixture = A5Fixture()
        source_samples = []
        for supplied in fixture.command.samples:
            source_samples.append(
                SimpleNamespace(
                    sample_id=supplied.sample_id,
                    split=(
                        DatasetSplit.OPTIMIZATION
                        if supplied.role is SampleRole.TARGET
                        else DatasetSplit.VALIDATION
                    ),
                    status=AnnotationSampleStatus.FROZEN,
                    annotation=SimpleNamespace(
                        human_decision_id=supplied.human_decision_id
                    ),
                    review=SimpleNamespace(
                        human_decision_id=f"review-{supplied.sample_id}"
                    ),
                    adjudication=None,
                    document_snapshot_id=supplied.document_id,
                    document_sha256=supplied.document_sha256,
                    source_pdf_reference=supplied.source_reference,
                    source_case_sha256=supplied.review_input_sha256,
                    evidence_catalog_sha256=supplied.evidence_sha256,
                    finding_id=supplied.finding_id,
                )
            )
        dataset = SimpleNamespace(
            dataset_version_id=fixture.command.dataset_version_id,
            manifest_sha256=fixture.dataset.manifest_sha256,
            status=DatasetStatus.FROZEN,
            provenance=SimpleNamespace(status="verified", claims_allowed=True),
            required_human_cases=2,
            samples=tuple(source_samples),
        )
        snapshot = SimpleNamespace(
            dataset_version_id=dataset.dataset_version_id,
            manifest_sha256=dataset.manifest_sha256,
        )
        binding = SimpleNamespace(
            dataset_manifest_sha256=dataset.manifest_sha256,
            dataset_split=DatasetSplit.OPTIMIZATION,
            binding_sha256="9" * 64,
        )
        run = SimpleNamespace(
            run_id=fixture.command.a4_evaluation_run_id,
            status=EvaluationRunStatus.COMPLETED,
            purpose=EvaluationPurpose.CANDIDATE_DIAGNOSTIC,
            source_type=EvaluationSourceType.REAL,
            provenance_status="verified",
            claims_allowed=True,
            dataset=snapshot,
            binding=binding,
            result_sha256="a" * 64,
            report_sha256=fixture.command.a4_report_sha256,
            run_sha256="8" * 64,
        )
        report = SimpleNamespace(
            run_id=run.run_id,
            purpose=run.purpose,
            source_type=EvaluationSourceType.REAL,
            status="verified",
            claims_allowed=True,
            dataset=snapshot,
            binding=binding,
            result_sha256=run.result_sha256,
            report_sha256=run.report_sha256,
            failure_samples=(
                SimpleNamespace(
                    sample_id="target-1",
                    evidence_sha256s=(fixture.command.samples[0].evidence_sha256,),
                ),
            ),
        )

        class AnnotationRepository:
            def get_version(self, dataset_version_id):
                self.requested = dataset_version_id
                return dataset

        class EvaluationRepository:
            def get(self, run_id):
                self.requested = run_id
                return run

            def get_report(self, run_id):
                self.report_requested = run_id
                return report

        readiness = A4OptimizationReadinessVerifier(
            AnnotationRepository(), EvaluationRepository(), fixture.clock
        ).assess(fixture.command)

        self.assertEqual(readiness.status, OptimizationReadinessStatus.READY)
        self.assertTrue(readiness.claims_allowed)
        self.assertEqual(readiness.verified_failure_sample_ids, ("target-1",))

    def test_missing_real_a3_a4_evidence_creates_not_ready_without_running(self):
        fixture = A5Fixture()
        fixture.readiness = A4OptimizationReadinessVerifier(
            InMemoryAnnotationDatasetRepository(),
            InMemoryEvaluationRunRepository(),
            fixture.clock,
        )
        fixture.service = fixture._service()
        job = fixture.service.create(fixture.command)
        completed = fixture.service.run(job.optimization_job_id)

        self.assertEqual(job.status, OptimizationStatus.NOT_READY)
        self.assertEqual(completed, job)
        self.assertFalse(job.readiness.claims_allowed)
        self.assertIn("A3 annotation dataset is unavailable", job.readiness.blockers)
        self.assertEqual(fixture.generator.calls, [])
        self.assertEqual(fixture.evaluator.calls, [])

    def test_api_contract_rejects_client_supplied_readiness_or_status(self):
        fixture = A5Fixture()
        payload = fixture.command.model_dump(
            mode="json", exclude={"base_rule_version_id"}
        )
        payload.update({"readiness": {"status": "READY"}, "status": "PENDING"})
        with self.assertRaises(ValidationError):
            A5OptimizeRuleVersionRequest.model_validate(payload)

    def test_sql_repository_rejects_fabricated_ready_job_without_a4_rows(self):
        fixture = A5Fixture()
        job = fixture.service.create(fixture.command)
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "a5.sqlite3"
            engine = create_engine(f"sqlite+pysqlite:///{database.as_posix()}")
            try:
                Base.metadata.create_all(engine)
                repository = SqlAlchemyOptimizationRepository(
                    sessionmaker(bind=engine, expire_on_commit=False)
                )
                with self.assertRaisesRegex(
                    PermanentError, "does not authorize optimization"
                ) as raised:
                    repository.create_job(job)
                self.assertEqual(
                    raised.exception.code,
                    "optimization_readiness_bypass_forbidden",
                )
            finally:
                engine.dispose()

    def test_a5_migration_round_trip_keeps_one_head(self):
        config = AlembicConfig(str(Path(__file__).parents[1] / "alembic.ini"))
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "a5-migration.sqlite3"
            database_url = f"sqlite:///{database.as_posix()}"
            config.set_main_option("sqlalchemy.url", database_url)
            alembic_command.upgrade(config, "head")
            engine = create_engine(database_url)
            try:
                columns = {
                    item["name"]
                    for item in inspect(engine).get_columns("optimization_jobs")
                }
                self.assertTrue({"readiness_json", "graph_trace_json"}.issubset(columns))
            finally:
                engine.dispose()

            alembic_command.downgrade(config, "c4a9e2d7f103")
            engine = create_engine(database_url)
            try:
                columns = {
                    item["name"]
                    for item in inspect(engine).get_columns("optimization_jobs")
                }
                self.assertNotIn("readiness_json", columns)
                self.assertNotIn("graph_trace_json", columns)
            finally:
                engine.dispose()

            alembic_command.upgrade(config, "head")


if __name__ == "__main__":
    unittest.main()
