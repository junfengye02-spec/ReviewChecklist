from __future__ import annotations

import hashlib
import unittest
from datetime import datetime, timezone

from fastapi.testclient import TestClient
from pydantic import ValidationError

from tender_review.api import create_app
from tender_review.bootstrap import build_container
from tender_review.evaluation.public import (
    CreateDatasetVersion,
    DatasetProvenance,
    DatasetSampleInput,
    DatasetSourceType,
    DatasetSplit,
    DatasetStatus,
    DatasetVersionService,
    InMemoryDatasetVersionRepository,
    deterministic_document_splits,
    samples_from_human_decision,
)
from tender_review.findings.public import (
    DocumentIdentity,
    EvidenceReference,
    FindingDecisionService,
    FindingProvenance,
    FindingWorkflowState,
    HumanDecisionType,
    InMemoryFindingRepository,
    SubmitHumanDecision,
    build_finding,
)
from tender_review.rule_management.public import (
    CompleteEvaluationGate,
    CreateRuleVersion,
    InMemoryRuleVersionRepository,
    PublishRuleVersion,
    RollbackRuleSet,
    RuleProvenance,
    RuleVersion,
    RuleVersionService,
    canonical_json,
)
from tender_review.shared.clock import FixedClock
from tender_review.shared.config import AppSettings
from tender_review.shared.errors import ConflictError, PermanentError
from tender_review.shared.ids import SequentialIdGenerator
from tender_review.review.public import approval_finding_from_review_state
from tender_review.review.public import SingleReviewWorkflow
from test_phase5_review import review_request, text_extraction


NOW = datetime(2026, 7, 28, 5, 0, tzinfo=timezone.utc)


class _TrustedPhase6Verifier:
    """Keeps the legacy state-machine tests isolated from A4 evidence storage."""

    def assert_dataset_release_ready(self, dataset_version_id: str) -> None:
        del dataset_version_id

    def assert_release_eligible(self, **identity: str) -> None:
        del identity


def _evidence(text: str = "source evidence") -> EvidenceReference:
    return EvidenceReference(
        document_id="doc-1",
        chunk_id="chunk-1",
        page_number=3,
        section_path=("qualification",),
        excerpt=text,
        text_sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(),
    )


def _finding(*, workflow_state: FindingWorkflowState = FindingWorkflowState.DONE):
    done = workflow_state is FindingWorkflowState.DONE
    return build_finding(
        finding_id="finding-1",
        review_job_id="job-1",
        rule_version_id="rule-1",
        review_item_id="item-1",
        workflow_state=workflow_state,
        message="deterministic result" if done else "requires explicit work",
        documents=(DocumentIdentity(document_id="doc-1", document_sha256="a" * 64),),
        provenance=FindingProvenance(
            source_kind="provisional_retrieval",
            status="provisional",
            claims_allowed=False,
            dataset_version_id="phase4-provisional",
            review_input_sha256="b" * 64,
            retrieval_results_sha256="c" * 64,
            retrieval_variant="bm25",
        ),
        created_at=NOW,
        conclusion="noncompliant" if done else None,
        evidence=(_evidence(),) if done else (),
    )


def _verified_rule_provenance() -> RuleProvenance:
    return RuleProvenance(source_type="manual", status="verified", claims_allowed=True)


class FindingDecisionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.repository = InMemoryFindingRepository()
        self.repository.add_finding(_finding())
        self.service = FindingDecisionService(
            self.repository,
            SequentialIdGenerator(prefix="decision"),
            FixedClock(NOW),
        )

    def test_decision_is_immutable_and_correction_must_supersede_latest(self):
        first = self.service.submit(SubmitHumanDecision(
            finding_id="finding-1",
            reviewer_id="reviewer-42",
            decision=HumanDecisionType.APPROVE,
            reason="checked source page",
        ))
        self.assertEqual(first.finding.status.value, "APPROVED")
        self.assertEqual(first.decision.review_input_sha256, "b" * 64)

        with self.assertRaisesRegex(ConflictError, "supersede"):
            self.service.submit(SubmitHumanDecision(
                finding_id="finding-1",
                reviewer_id="reviewer-42",
                decision=HumanDecisionType.REJECT,
                reason="correction",
            ))

        corrected = self.service.submit(SubmitHumanDecision(
            finding_id="finding-1",
            reviewer_id="reviewer-42",
            decision=HumanDecisionType.REJECT,
            reason="second source check",
            supersedes_decision_id=first.decision.decision_id,
        ))
        self.assertEqual(corrected.finding.status.value, "REJECTED")
        self.assertEqual(len(self.repository.list_decisions("finding-1")), 2)
        tampered = corrected.decision.model_dump(mode="json")
        tampered["reason"] = "overwritten"
        with self.assertRaisesRegex(ValidationError, "decision_sha256"):
            type(corrected.decision).model_validate(tampered)

    def test_ai_or_provisional_identity_is_rejected(self):
        for reviewer_id in ("ai", "AI-reviewer", "provisional:worker", "system"):
            with self.subTest(reviewer_id=reviewer_id):
                with self.assertRaisesRegex(ValidationError, "named human"):
                    SubmitHumanDecision(
                        finding_id="finding-1",
                        reviewer_id=reviewer_id,
                        decision="APPROVE",
                        reason="invalid actor",
                    )

    def test_handoff_branch_cannot_be_approved_as_a_done_finding(self):
        repository = InMemoryFindingRepository()
        repository.add_finding(_finding(workflow_state=FindingWorkflowState.WAITING_HUMAN))
        service = FindingDecisionService(
            repository, SequentialIdGenerator(), FixedClock(NOW)
        )
        with self.assertRaisesRegex(PermanentError, "work items"):
            service.submit(SubmitHumanDecision(
                finding_id="finding-1",
                reviewer_id="reviewer-42",
                decision="APPROVE",
                reason="invalid shortcut",
            ))

    def test_stage5_done_state_projects_to_approvable_finding(self):
        request, llm = review_request(text_extraction())
        workflow = SingleReviewWorkflow(
            llm,
            id_generator=SequentialIdGenerator(("finding-from-stage5",)),
        )
        state = workflow.run(request)
        projected = approval_finding_from_review_state(
            state,
            rule_version_id="rule-v1",
            documents=(
                DocumentIdentity(document_id="document-1", document_sha256="a" * 64),
            ),
            created_at=NOW,
        )
        self.assertEqual(projected.workflow_state, FindingWorkflowState.DONE)
        self.assertEqual(projected.status.value, "PENDING_DECISION")
        self.assertEqual(
            projected.provenance.review_input_sha256,
            state.provenance.input_sha256,
        )


class RuleVersionTests(unittest.TestCase):
    def setUp(self) -> None:
        verifier = _TrustedPhase6Verifier()
        self.repository = InMemoryRuleVersionRepository(verifier)
        self.service = RuleVersionService(
            self.repository,
            SequentialIdGenerator(prefix="rule"),
            FixedClock(NOW),
            verifier,
        )

    def _create(self, content: dict, parent: str | None = None):
        return self.service.create_version(CreateRuleVersion(
            rule_set_id="set-1",
            rule_key="qualification",
            rule_set_name="Qualification rules",
            parent_version_id=parent,
            content_json=canonical_json(content),
            change_summary="bounded change",
            provenance=_verified_rule_provenance(),
        ))

    def _pass_gate(self, version):
        evaluating = self.service.request_evaluation(version.rule_version_id, "dataset-real-1")
        return self.service.complete_evaluation(CompleteEvaluationGate(
            rule_version_id=version.rule_version_id,
            gate_id=evaluating.evaluation_gate.gate_id,
            evaluation_run_id=f"run-{version.version_number}",
            status="PASSED",
            provisional=False,
            claims_allowed=True,
            report_sha256="d" * 64,
        ))

    def test_structured_diff_and_content_hash_detect_mutation(self):
        first = self._create({"rules": [{"id": "r1", "threshold": 10}]})
        second = self._create(
            {"rules": [{"id": "r1", "threshold": 12}], "mode": "strict"},
            first.rule_version_id,
        )
        diff = self.service.diff(first.rule_version_id, second.rule_version_id)
        self.assertEqual(
            [(item.path, item.operation) for item in diff.changes],
            [("$.mode", "add"), ("$.rules[0].threshold", "replace")],
        )
        tampered = second.model_dump(mode="json")
        tampered["content_json"] = canonical_json({"rules": []})
        with self.assertRaisesRegex(ValidationError, "content_sha256"):
            RuleVersion.model_validate(tampered)

    def test_failed_or_provisional_gate_cannot_publish(self):
        version = self._create({"rules": []})
        evaluating = self.service.request_evaluation(version.rule_version_id, "dataset-1")
        rejected = self.service.complete_evaluation(CompleteEvaluationGate(
            rule_version_id=version.rule_version_id,
            gate_id=evaluating.evaluation_gate.gate_id,
            evaluation_run_id="run-failed",
            status="FAILED",
            provisional=False,
            claims_allowed=True,
            report_sha256="e" * 64,
        ))
        self.assertEqual(rejected.status.value, "REJECTED")
        with self.assertRaises(ConflictError):
            self.service.publish(PublishRuleVersion(
                rule_version_id=version.rule_version_id,
                approver_id="reviewer-42",
            ))

        provisional_service = RuleVersionService(
            InMemoryRuleVersionRepository(_TrustedPhase6Verifier()),
            SequentialIdGenerator(prefix="provisional-rule"),
            FixedClock(NOW),
            _TrustedPhase6Verifier(),
        )
        provisional = provisional_service.create_version(CreateRuleVersion(
            rule_set_id="set-p",
            rule_key="p",
            rule_set_name="Provisional",
            content_json="{}",
            change_summary="engineering flow only",
            provenance=RuleProvenance(
                source_type="provisional", status="provisional", claims_allowed=False
            ),
        ))
        pending = provisional_service.request_evaluation(
            provisional.rule_version_id, "dataset-provisional"
        )
        with self.assertRaisesRegex(PermanentError, "provisional-only"):
            provisional_service.complete_evaluation(CompleteEvaluationGate(
                rule_version_id=provisional.rule_version_id,
                gate_id=pending.evaluation_gate.gate_id,
                evaluation_run_id="run-provisional",
                status="PASSED",
                provisional=True,
                claims_allowed=False,
                report_sha256="f" * 64,
            ))

    def test_publish_atomically_switches_current_and_rollback_keeps_history(self):
        first = self._create({"threshold": 10})
        self._pass_gate(first)
        self.service.publish(PublishRuleVersion(
            rule_version_id=first.rule_version_id, approver_id="reviewer-42"
        ))
        second = self._create({"threshold": 12}, first.rule_version_id)
        self._pass_gate(second)
        self.service.publish(PublishRuleVersion(
            rule_version_id=second.rule_version_id, approver_id="reviewer-42"
        ))

        restored = self.service.rollback(RollbackRuleSet(
            rule_set_id="set-1",
            target_version_id=first.rule_version_id,
            approver_id="reviewer-42",
            reason="verified regression",
        ))
        self.assertEqual(restored.rule_version_id, first.rule_version_id)
        self.assertEqual(
            self.repository.get_rule_set("set-1").current_version_id,
            first.rule_version_id,
        )
        versions = self.repository.list_versions("set-1")
        self.assertEqual(len(versions), 2)
        self.assertEqual(versions[1].status.value, "ROLLED_BACK")


class DatasetVersionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.service = DatasetVersionService(
            InMemoryDatasetVersionRepository(),
            SequentialIdGenerator(prefix="dataset"),
            FixedClock(NOW),
        )
        self.provenance = DatasetProvenance(
            status="provisional",
            claims_allowed=False,
            source_description="synthetic provisional inputs; human labels 0/4",
            source_manifest_sha256="1" * 64,
        )

    def _sample(self, sample_id: str, split: DatasetSplit) -> DatasetSampleInput:
        return DatasetSampleInput(
            sample_id=sample_id,
            document_id="doc-shared",
            document_sha256="2" * 64,
            split=split,
            source_type=DatasetSourceType.PROVISIONAL,
            provenance_status="provisional",
            label_version="provisional-v1",
            label_json='{"label":"navigation-hint-only"}',
            review_input_sha256="3" * 64,
            evidence_sha256="4" * 64,
        )

    def test_document_level_split_rejects_leakage(self):
        with self.assertRaisesRegex(PermanentError, "one split"):
            self.service.create_version(CreateDatasetVersion(
                dataset_name="phase6-flow",
                requested_status=DatasetStatus.PROVISIONAL,
                change_summary="test leakage guard",
                provenance=self.provenance,
                samples=(
                    self._sample("s1", DatasetSplit.OPTIMIZATION),
                    self._sample("s2", DatasetSplit.FROZEN_TEST),
                ),
            ))

    def test_provisional_sample_cannot_be_frozen_as_real(self):
        with self.assertRaisesRegex(PermanentError, "real human labels"):
            self.service.create_version(CreateDatasetVersion(
                dataset_name="phase6-flow",
                requested_status=DatasetStatus.FROZEN,
                change_summary="invalid freeze",
                provenance=self.provenance,
                samples=(self._sample("s1", DatasetSplit.FROZEN_TEST),),
            ))
        version = self.service.create_version(CreateDatasetVersion(
            dataset_name="phase6-flow",
            requested_status=DatasetStatus.PROVISIONAL,
            change_summary="validate version flow only",
            provenance=self.provenance,
            samples=(self._sample("s1", DatasetSplit.FROZEN_TEST),),
        ))
        self.assertEqual(version.status, DatasetStatus.PROVISIONAL)
        self.assertFalse(version.provenance.claims_allowed)

    def test_decision_projection_retains_provisional_review_provenance(self):
        finding_repository = InMemoryFindingRepository()
        finding_repository.add_finding(_finding())
        outcome = FindingDecisionService(
            finding_repository,
            SequentialIdGenerator(prefix="decision"),
            FixedClock(NOW),
        ).submit(SubmitHumanDecision(
            finding_id="finding-1",
            reviewer_id="reviewer-42",
            decision="APPROVE",
            reason="human checked the cited page",
        ))
        samples = samples_from_human_decision(
            outcome, split=DatasetSplit.OPTIMIZATION
        )
        self.assertEqual(samples[0].human_decision_id, outcome.decision.decision_id)
        self.assertEqual(samples[0].provenance_status, "provisional")
        self.assertEqual(samples[0].review_input_sha256, "b" * 64)

    def test_deterministic_split_operates_on_unique_documents(self):
        first = deterministic_document_splits(("doc-b", "doc-a", "doc-c"))
        second = deterministic_document_splits(("doc-c", "doc-b", "doc-a"))
        self.assertEqual(first, second)
        self.assertEqual(set(first), {"doc-a", "doc-b", "doc-c"})


class Phase6ApiTests(unittest.TestCase):
    def setUp(self) -> None:
        self.container = build_container(AppSettings(environment="test", log_json=False))
        self.container.finding_repository.add_finding(_finding())
        self.client = TestClient(create_app(self.container))

    def test_decision_and_rule_lifecycle_api(self):
        decision = self.client.post(
            "/api/v1/findings/finding-1/decisions",
            json={
                "reviewer_kind": "human",
                "reviewer_id": "reviewer-42",
                "decision": "APPROVE",
                "reason": "checked source page",
            },
        )
        self.assertEqual(decision.status_code, 201, decision.text)
        self.assertEqual(decision.json()["decision"]["review_input_sha256"], "b" * 64)

        created = self.client.post(
            "/api/v1/rule-sets/set-api/versions",
            json={
                "rule_key": "api-rule",
                "rule_set_name": "API rules",
                "content": {"threshold": 10},
                "change_summary": "initial version",
                "provenance": {
                    "source_type": "manual",
                    "status": "verified",
                    "claims_allowed": True,
                },
            },
        )
        self.assertEqual(created.status_code, 201, created.text)
        version_id = created.json()["rule_version_id"]
        evaluating = self.client.post(
            f"/api/v1/rule-versions/{version_id}/evaluate",
            json={"dataset_version_id": "dataset-real"},
        )
        self.assertEqual(evaluating.status_code, 404, evaluating.text)
        self.assertEqual(evaluating.json()["error"]["code"], "annotation_dataset_not_found")
        self.assertEqual(
            self.container.rule_version_repository.get_version(version_id).status.value,
            "DRAFT",
        )

    def test_api_rejects_nonhuman_reviewer_without_leaking_internals(self):
        response = self.client.post(
            "/api/v1/findings/finding-1/decisions",
            json={
                "reviewer_kind": "human",
                "reviewer_id": "AI-reviewer",
                "decision": "APPROVE",
                "reason": "invalid",
            },
        )
        self.assertEqual(response.status_code, 422)
        self.assertEqual(response.json()["error"]["code"], "request_validation_failed")
        self.assertNotIn("Traceback", response.text)


if __name__ == "__main__":
    unittest.main()
