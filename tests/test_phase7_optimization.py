from __future__ import annotations

import time
import unittest
from datetime import datetime, timezone

from pydantic import ValidationError

from tender_review.evaluation.public import (
    CreateDatasetVersion,
    DatasetProvenance,
    DatasetSampleInput,
    DatasetSourceType,
    DatasetSplit,
    DatasetVersionService,
    InMemoryDatasetVersionRepository,
)
from tender_review.optimization.public import (
    CandidateChange,
    CandidateType,
    CreateOptimizationJob,
    FailureSignals,
    FakeCandidateGenerator,
    FakeRegressionEvaluator,
    InMemoryOptimizationRepository,
    OptimizationCandidate,
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
    canonical_json,
)
from tender_review.review.public import FakeLlmProvider, LlmResponse
from tender_review.rule_management.public import (
    CreateRuleVersion,
    InMemoryRuleVersionRepository,
    RuleProvenance,
    RuleVersionService,
)
from tender_review.shared.clock import FixedClock
from tender_review.shared.errors import RetryableError
from tender_review.shared.ids import SequentialIdGenerator


NOW = datetime(2026, 7, 28, 7, 0, tzinfo=timezone.utc)


class SlowLlm:
    def complete(self, request):
        del request
        time.sleep(0.05)
        return LlmResponse(
            model="slow",
            content='{"rationale":"semantic gap","root_cause":"RULE_GAP"}',
        )


class CrashAfterCreateStager:
    def __init__(self, delegate) -> None:
        self.delegate = delegate
        self.crashed = False

    def stage_candidate(self, job, attempt, candidate):
        version_id = self.delegate.stage_candidate(job, attempt, candidate)
        if not self.crashed:
            self.crashed = True
            raise KeyboardInterrupt()
        return version_id


class LocalReadyVerifier:
    """Test-only proof object; it is not project A3/A4 evidence."""

    def __init__(self, dataset) -> None:
        self.dataset = dataset

    def assess(self, command):
        return OptimizationReadiness(
            status=OptimizationReadinessStatus.READY,
            claims_allowed=True,
            dataset_manifest_sha256=self.dataset.manifest_sha256,
            a4_evaluation_run_id=command.a4_evaluation_run_id,
            a4_run_sha256="8" * 64,
            a4_report_sha256=command.a4_report_sha256,
            a4_binding_sha256="9" * 64,
            a4_result_sha256="a" * 64,
            verified_failure_sample_ids=("target-1",),
            dataset_sample_ids=("target-1", "protection-1"),
            assessed_at=NOW,
        )


class Phase7Fixture:
    def __init__(
        self,
        *,
        signals: FailureSignals | None = None,
        plans=(),
        max_rounds: int = 2,
        candidates_per_round: int = 2,
        llm=None,
    ) -> None:
        self.clock = FixedClock(NOW)
        self.rule_repository = InMemoryRuleVersionRepository()
        self.rule_service = RuleVersionService(
            self.rule_repository,
            SequentialIdGenerator(prefix="rule"),
            self.clock,
        )
        self.base = self.rule_service.create_version(
            CreateRuleVersion(
                rule_set_id="set-1",
                rule_key="qualification",
                rule_set_name="Qualification",
                content_json='{"rule_text":"base"}',
                execution_config_json="{}",
                change_summary="base immutable version",
                provenance=RuleProvenance(
                    source_type="manual", status="verified", claims_allowed=True
                ),
            )
        )
        self.dataset_repository = InMemoryDatasetVersionRepository()
        self.dataset_service = DatasetVersionService(
            self.dataset_repository,
            SequentialIdGenerator(prefix="dataset"),
            self.clock,
        )
        self.dataset = self.dataset_service.create_version(
            CreateDatasetVersion(
                dataset_name="phase7-local-ready-contract",
                requested_status="FROZEN",
                change_summary="test-only state-machine fixture",
                provenance=DatasetProvenance(
                    status="verified",
                    claims_allowed=True,
                    source_description="test-only verified contract fixture",
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
        self.optimization_repository = InMemoryOptimizationRepository()
        self.generator = FakeCandidateGenerator(
            base_content_json=self.base.content_json,
            base_execution_config_json=self.base.execution_config_json,
        )
        self.evaluator = FakeRegressionEvaluator(plans)
        self.service = OptimizationService(
            repository=self.optimization_repository,
            rule_versions=self.rule_repository,
            datasets=self.dataset_repository,
            ids=SequentialIdGenerator(prefix="optimization"),
            clock=self.clock,
            root_causes=RootCauseAnalyzer(llm or FakeLlmProvider()),
            candidates=self.generator,
            evaluator=self.evaluator,
            stager=RuleVersionCandidateStager(
                self.rule_service, self.rule_repository
            ),
            readiness=LocalReadyVerifier(self.dataset),
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
            a4_evaluation_run_id="a4-local-contract-run",
            a4_report_sha256="7" * 64,
            samples=(
                _optimization_sample(
                    "target-1",
                    SampleRole.TARGET,
                    "doc-target",
                    signals
                    or FailureSignals(
                        failure_summary="historical workflow missed confirmed platform expectation",
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

    def create(self):
        return self.service.create(self.command)


def _dataset_sample(
    sample_id: str, document_id: str, split: DatasetSplit
) -> DatasetSampleInput:
    return DatasetSampleInput(
        sample_id=sample_id,
        document_id=document_id,
        document_sha256=("1" if "target" in document_id else "2") * 64,
        split=split,
        finding_id=f"finding-{sample_id}",
        human_decision_id=f"decision-{sample_id}",
        source_type=DatasetSourceType.REAL,
        provenance_status="verified",
        label_version="test-only-verified-v1",
        label_json='{"label":"test-only-verified"}',
        review_input_sha256=("3" if "target" in document_id else "4") * 64,
        evidence_sha256=("5" if "target" in document_id else "6") * 64,
    )


def _optimization_sample(
    sample_id: str,
    role: SampleRole,
    document_id: str,
    signals: FailureSignals | None,
) -> OptimizationSample:
    target = role is SampleRole.TARGET
    return OptimizationSample(
        sample_id=sample_id,
        role=role,
        document_id=document_id,
        document_sha256=("1" if target else "2") * 64,
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


def _provenance() -> OptimizationProvenance:
    return OptimizationProvenance(
        source_type=SourceType.REAL,
        status="verified",
        claims_allowed=True,
        source_description="Synthetic verified optimization fixture.",
        source_artifacts=(
            SourceArtifact(
                path="tests/fixtures/templates/review_case.example.json",
                sha256="7" * 64,
                kind="manifest",
            ),
        ),
        human_annotation_cases=4,
        required_human_cases=4,
    )


class RootCauseTests(unittest.TestCase):
    def test_deterministic_checks_take_priority_over_llm(self):
        cases = (
            (
                FailureSignals(
                    failure_summary="conflict",
                    label_conflict=True,
                    evidence_in_top_k=False,
                ),
                RootCause.LABEL_UNCERTAIN,
            ),
            (
                FailureSignals(
                    failure_summary="miss", evidence_in_top_k=False
                ),
                RootCause.RETRIEVAL_MISS,
            ),
            (
                FailureSignals(
                    failure_summary="extract",
                    evidence_in_top_k=True,
                    extraction_matches_expected=False,
                ),
                RootCause.EXTRACTION_ERROR,
            ),
            (
                FailureSignals(
                    failure_summary="tool",
                    evidence_in_top_k=True,
                    extraction_matches_expected=True,
                    tool_matches_expected=False,
                ),
                RootCause.TOOL_ERROR,
            ),
            (
                FailureSignals(
                    failure_summary="unstable",
                    repeated_outputs_consistent=False,
                ),
                RootCause.MODEL_INSTABILITY,
            ),
            (
                FailureSignals(
                    failure_summary="gap",
                    evidence_in_top_k=True,
                    extraction_matches_expected=True,
                    tool_matches_expected=True,
                    repeated_outputs_consistent=True,
                ),
                RootCause.RULE_GAP,
            ),
        )
        llm = FakeLlmProvider(
            ('{"rationale":"wrong","root_cause":"RULE_GAP"}',)
        )
        for signals, expected in cases:
            with self.subTest(expected=expected):
                fixture = Phase7Fixture(signals=signals, llm=llm)
                decision = RootCauseAnalyzer(llm).analyze(fixture.create(), 1)
                self.assertEqual(decision.root_cause, expected)
                self.assertEqual(decision.classifier, "deterministic")
        self.assertEqual(llm.calls, [])

    def test_llm_only_handles_unresolved_semantics_with_schema_and_call_id(self):
        llm = FakeLlmProvider(
            ('{"rationale":"semantic rule omission","root_cause":"RULE_GAP"}',)
        )
        fixture = Phase7Fixture(
            signals=FailureSignals(failure_summary="semantic ambiguity"), llm=llm
        )
        job = fixture.create()

        decision = RootCauseAnalyzer(llm).analyze(job, 1)

        self.assertEqual(decision.root_cause, RootCause.RULE_GAP)
        self.assertEqual(decision.classifier, "llm")
        self.assertEqual(decision.call_id, f"{job.optimization_job_id}:root-cause:1")
        self.assertEqual(llm.calls[0].response_schema_name, "RootCauseLlmOutput.v1")
        self.assertEqual(llm.calls[0].temperature, 0)

    def test_llm_timeout_is_typed_and_bounded(self):
        fixture = Phase7Fixture(
            signals=FailureSignals(failure_summary="semantic ambiguity")
        )
        with self.assertRaisesRegex(RetryableError, "timed out") as raised:
            RootCauseAnalyzer(SlowLlm(), timeout_seconds=0.001).analyze(
                fixture.create(), 1
            )
        self.assertEqual(
            raised.exception.details["call_id"],
            "optimization-1:root-cause:1",
        )

    def test_llm_extra_fields_are_rejected_by_strict_schema(self):
        llm = FakeLlmProvider(
            (
                '{"extra":"forbidden","rationale":"gap",'
                '"root_cause":"RULE_GAP"}',
            )
        )
        fixture = Phase7Fixture(
            signals=FailureSignals(failure_summary="semantic ambiguity"), llm=llm
        )
        with self.assertRaisesRegex(Exception, "invalid strict-schema"):
            RootCauseAnalyzer(llm).analyze(fixture.create(), 1)


class OptimizationLoopTests(unittest.TestCase):
    def test_label_uncertain_stops_for_human_without_candidate(self):
        fixture = Phase7Fixture(
            signals=FailureSignals(
                failure_summary="conflicting label", label_conflict=True
            )
        )
        job = fixture.create()

        completed = fixture.service.run(job.optimization_job_id)

        self.assertEqual(completed.status, OptimizationStatus.WAITING_HUMAN)
        self.assertEqual(fixture.generator.calls, [])
        attempt = fixture.service.list_attempts(job.optimization_job_id)[0]
        self.assertEqual(attempt.root_cause.root_cause, RootCause.LABEL_UNCERTAIN)

    def test_successful_local_candidate_passes_three_gates_but_stays_draft(self):
        fixture = Phase7Fixture(plans=((True, False, True), (True, True, True)))
        job = fixture.create()

        completed = fixture.service.run(job.optimization_job_id)

        self.assertEqual(completed.status, OptimizationStatus.WAITING_APPROVAL)
        attempt = fixture.service.list_attempts(job.optimization_job_id)[0]
        self.assertEqual(len(attempt.evaluations), 2)
        accepted = attempt.evaluations[-1]
        self.assertTrue(accepted.target_gate_passed)
        self.assertTrue(accepted.protection_gate_passed)
        self.assertTrue(accepted.stability_gate_passed)
        self.assertEqual(accepted.status.value, "PASSED")
        self.assertTrue(accepted.claims_allowed)
        candidate_version = self.rule_repository_get(
            fixture, completed.candidate_rule_version_id
        )
        self.assertEqual(candidate_version.status.value, "DRAFT")
        self.assertIsNone(candidate_version.evaluation_gate)
        self.assertEqual(candidate_version.provenance.status, "verified")
        self.assertTrue(candidate_version.provenance.claims_allowed)

    @staticmethod
    def rule_repository_get(fixture, version_id):
        return fixture.rule_repository.get_version(version_id)

    def test_failed_candidates_reach_bounded_terminal_and_keep_trajectory(self):
        fixture = Phase7Fixture(
            plans=((False, True, True),) * 4,
            max_rounds=2,
            candidates_per_round=2,
        )
        job = fixture.create()

        completed = fixture.service.run(job.optimization_job_id)

        self.assertEqual(completed.status, OptimizationStatus.OPTIMIZATION_FAILED)
        self.assertEqual(completed.current_round, 2)
        attempts = fixture.service.list_attempts(job.optimization_job_id)
        self.assertEqual(len(attempts), 2)
        self.assertEqual([len(item.evaluations) for item in attempts], [2, 2])
        self.assertTrue(all(item.selected_candidate_id is None for item in attempts))

    def test_resume_skips_completed_candidate_evaluation(self):
        fixture = Phase7Fixture(
            plans=((False, True, True), KeyboardInterrupt(), (True, True, True)),
            candidates_per_round=2,
        )
        job = fixture.create()
        with self.assertRaises(KeyboardInterrupt):
            fixture.service.run(job.optimization_job_id)
        partial = fixture.service.list_attempts(job.optimization_job_id)[0]
        first_candidate_id = partial.candidates[0].candidate_id
        self.assertEqual([item.candidate_id for item in partial.evaluations], [first_candidate_id])

        completed = fixture.service.run(job.optimization_job_id)

        self.assertEqual(completed.status, OptimizationStatus.WAITING_APPROVAL)
        final_attempt = fixture.service.list_attempts(job.optimization_job_id)[0]
        self.assertEqual(len(final_attempt.evaluations), 2)
        self.assertEqual(fixture.evaluator.calls.count(first_candidate_id), 1)
        self.assertEqual(len(fixture.generator.calls), 1)

    def test_resume_after_rule_staging_does_not_duplicate_candidate_version(self):
        fixture = Phase7Fixture(
            plans=((True, True, True),),
            candidates_per_round=1,
        )
        crashing_stager = CrashAfterCreateStager(
            RuleVersionCandidateStager(
                fixture.rule_service, fixture.rule_repository
            )
        )
        fixture.service._stager = crashing_stager
        fixture.service._workflow._nodes._stager = crashing_stager
        job = fixture.create()
        with self.assertRaises(KeyboardInterrupt):
            fixture.service.run(job.optimization_job_id)
        partial = fixture.service.list_attempts(job.optimization_job_id)[0]
        self.assertIsNotNone(partial.selected_candidate_id)
        self.assertIsNone(partial.candidate_rule_version_id)
        self.assertEqual(len(fixture.rule_repository.list_versions("set-1")), 2)

        completed = fixture.service.run(job.optimization_job_id)

        self.assertEqual(completed.status, OptimizationStatus.WAITING_APPROVAL)
        self.assertEqual(len(fixture.rule_repository.list_versions("set-1")), 2)
        restored = fixture.service.list_attempts(job.optimization_job_id)[0]
        self.assertEqual(
            restored.candidate_rule_version_id,
            completed.candidate_rule_version_id,
        )

    def test_wrong_candidate_route_is_rejected_by_contract(self):
        fixture = Phase7Fixture()
        job = fixture.create()
        decision = RootCauseAnalyzer().analyze(job, 1)
        valid = fixture.generator.generate(job, 1, decision, 1)[0]
        payload = valid.model_dump(mode="json")
        payload["candidate_type"] = CandidateType.TOOL_CONFIG
        payload["change"] = CandidateChange(
            scope="execution_config",
            path="$.tools.version",
            before_json=None,
            after_json='"v2"',
        ).model_dump(mode="json")
        with self.assertRaisesRegex(ValidationError, "not allowed"):
            OptimizationCandidate.model_validate(payload)

    def test_candidate_provenance_contains_all_execution_hashes(self):
        fixture = Phase7Fixture()
        job = fixture.create()
        decision = RootCauseAnalyzer().analyze(job, 1)
        candidate = fixture.generator.generate(job, 1, decision, 1)[0]
        self.assertEqual(candidate.provenance.hashes, job.hashes)
        self.assertEqual(candidate.provenance.status, "verified")
        self.assertTrue(candidate.provenance.claims_allowed)
        self.assertEqual(candidate.change.path, "$.rule_text")
        self.assertEqual(canonical_json({"rule_text": "base"}), job and fixture.base.content_json)


if __name__ == "__main__":
    unittest.main()
