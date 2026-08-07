from __future__ import annotations

import unittest

from sqlalchemy import create_engine, inspect

from tender_review.evaluation.public import (
    CreateDatasetVersion,
    DatasetProvenance,
    DatasetSplit,
    DatasetVersionService,
)
from tender_review.infrastructure.database import Base, create_session_factory
from tender_review.infrastructure.database.dataset_versions import (
    SqlAlchemyDatasetVersionRepository,
)
from tender_review.infrastructure.database.optimization_jobs import (
    SqlAlchemyOptimizationRepository,
)
from tender_review.infrastructure.database.rule_versions import (
    SqlAlchemyRuleVersionRepository,
)
from tender_review.optimization.public import (
    CreateOptimizationJob,
    FailureSignals,
    FakeCandidateGenerator,
    FakeRegressionEvaluator,
    OptimizationService,
    OptimizationStatus,
    RootCauseAnalyzer,
    RuleVersionCandidateStager,
    SampleRole,
)
from tender_review.rule_management.public import (
    CreateRuleVersion,
    RuleProvenance,
    RuleVersionService,
)
from tender_review.shared.clock import FixedClock
from tender_review.shared.ids import SequentialIdGenerator

from test_phase7_optimization import (
    NOW,
    _dataset_sample,
    _optimization_sample,
    _provenance,
)


class Phase7DatabaseTests(unittest.TestCase):
    def setUp(self) -> None:
        self.engine = create_engine("sqlite:///:memory:")
        Base.metadata.create_all(self.engine)
        self.sessions = create_session_factory(self.engine)
        self.clock = FixedClock(NOW)

    def tearDown(self) -> None:
        self.engine.dispose()

    def test_provisional_job_round_trips_as_not_ready_without_attempts(self):
        rule_repository = SqlAlchemyRuleVersionRepository(self.sessions)
        rule_service = RuleVersionService(
            rule_repository,
            SequentialIdGenerator(
                values=(
                    "00000000-0000-0000-0000-000000000101",
                    "00000000-0000-0000-0000-000000000102",
                )
            ),
            self.clock,
        )
        base = rule_service.create_version(
            CreateRuleVersion(
                rule_set_id="00000000-0000-0000-0000-000000000100",
                rule_key="phase7-db",
                rule_set_name="Phase 7 database",
                content_json='{"rule_text":"base"}',
                change_summary="base version",
                provenance=RuleProvenance(
                    source_type="manual", status="verified", claims_allowed=True
                ),
            )
        )
        dataset_repository = SqlAlchemyDatasetVersionRepository(self.sessions)
        dataset_service = DatasetVersionService(
            dataset_repository,
            SequentialIdGenerator(
                values=("00000000-0000-0000-0000-000000000110",)
            ),
            self.clock,
        )
        dataset = dataset_service.create_version(
            CreateDatasetVersion(
                dataset_name="phase7-db",
                requested_status="PROVISIONAL",
                change_summary="database adapter flow",
                provenance=DatasetProvenance(
                    status="provisional",
                    claims_allowed=False,
                    source_description="synthetic examples; labels 0/4",
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
        optimization_repository = SqlAlchemyOptimizationRepository(self.sessions)
        service = OptimizationService(
            repository=optimization_repository,
            rule_versions=rule_repository,
            datasets=dataset_repository,
            ids=SequentialIdGenerator(
                values=(
                    "00000000-0000-0000-0000-000000000120",
                    "00000000-0000-0000-0000-000000000121",
                )
            ),
            clock=self.clock,
            root_causes=RootCauseAnalyzer(),
            candidates=FakeCandidateGenerator(
                base_content_json=base.content_json,
                base_execution_config_json=base.execution_config_json,
            ),
            evaluator=FakeRegressionEvaluator(((True, True, True),)),
            stager=RuleVersionCandidateStager(rule_service, rule_repository),
        )
        job = service.create(
            CreateOptimizationJob(
                base_rule_version_id=base.rule_version_id,
                dataset_version_id=dataset.dataset_version_id,
                max_rounds=1,
                candidates_per_round=1,
                required_stability_runs=2,
                model_sha256="b" * 64,
                prompt_sha256="c" * 64,
                retriever_sha256="d" * 64,
                tool_sha256="e" * 64,
                samples=(
                    _optimization_sample(
                        "target-1",
                        SampleRole.TARGET,
                        "doc-target",
                        FailureSignals(
                            failure_summary="rule gap",
                            evidence_in_top_k=True,
                            extraction_matches_expected=True,
                            tool_matches_expected=True,
                            repeated_outputs_consistent=True,
                        ),
                    ),
                    _optimization_sample(
                        "protection-1",
                        SampleRole.PROTECTION,
                        "doc-protection",
                        None,
                    ),
                ),
                provenance=_provenance(),
            )
        )

        completed = service.run(job.optimization_job_id)

        restored = optimization_repository.get_job(job.optimization_job_id)
        attempts = optimization_repository.list_attempts(job.optimization_job_id)
        self.assertEqual(restored, completed)
        self.assertEqual(restored.status, OptimizationStatus.NOT_READY)
        self.assertEqual(attempts, ())
        self.assertIsNone(restored.last_checkpoint_sha256)
        self.assertIsNone(restored.candidate_rule_version_id)
        self.assertFalse(restored.readiness.claims_allowed)

    def test_phase7_columns_exist_in_metadata(self):
        job_columns = {
            item["name"]
            for item in inspect(self.engine).get_columns("optimization_jobs")
        }
        attempt_columns = {
            item["name"]
            for item in inspect(self.engine).get_columns("optimization_attempts")
        }
        self.assertTrue(
            {
                "candidates_per_round",
                "required_stability_runs",
                "hashes_json",
                "provenance_json",
                "samples_json",
                "last_checkpoint_sha256",
                "readiness_json",
                "graph_trace_json",
            }.issubset(job_columns)
        )
        self.assertTrue(
            {
                "status",
                "root_cause_json",
                "candidates_json",
                "evaluations_json",
                "checkpoint_sha256",
            }.issubset(attempt_columns)
        )


if __name__ == "__main__":
    unittest.main()
