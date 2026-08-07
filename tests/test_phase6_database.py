from __future__ import annotations

import unittest

from sqlalchemy import create_engine, inspect

from tender_review.evaluation.public import (
    CreateDatasetVersion,
    DatasetProvenance,
    DatasetSampleInput,
    DatasetVersionService,
)
from tender_review.findings.public import (
    FindingDecisionService,
    SubmitHumanDecision,
)
from tender_review.infrastructure.database import Base, create_session_factory
from tender_review.infrastructure.database.dataset_versions import (
    SqlAlchemyDatasetVersionRepository,
)
from tender_review.infrastructure.database.finding_records import (
    SqlAlchemyFindingRepository,
)
from tender_review.infrastructure.database.rule_versions import (
    SqlAlchemyRuleVersionRepository,
)
from tender_review.rule_management.public import (
    CompleteEvaluationGate,
    CreateRuleVersion,
    PublishRuleVersion,
    RuleProvenance,
    RuleVersionService,
)
from tender_review.shared.clock import FixedClock
from tender_review.shared.ids import SequentialIdGenerator
from tender_review.shared.errors import PermanentError

from test_phase6_governance import NOW, _finding


class _LegacyVerifier:
    def assert_dataset_release_ready(self, dataset_version_id: str) -> None:
        del dataset_version_id

    def assert_release_eligible(self, **identity: str) -> None:
        del identity


class Phase6DatabaseAdapterTests(unittest.TestCase):
    def setUp(self) -> None:
        self.engine = create_engine("sqlite:///:memory:")
        Base.metadata.create_all(self.engine)
        self.sessions = create_session_factory(self.engine)
        self.clock = FixedClock(NOW)

    def tearDown(self) -> None:
        self.engine.dispose()

    def test_finding_decision_round_trip_preserves_hashes_and_history(self):
        repository = SqlAlchemyFindingRepository(self.sessions)
        repository.add_finding(_finding())
        outcome = FindingDecisionService(
            repository,
            SequentialIdGenerator(values=("00000000-0000-0000-0000-000000000001",)),
            self.clock,
        ).submit(SubmitHumanDecision(
            finding_id="finding-1",
            reviewer_id="reviewer-42",
            decision="APPROVE",
            reason="checked source page",
        ))
        restored = repository.get_finding("finding-1")
        decisions = repository.list_decisions("finding-1")
        self.assertEqual(restored.finding_content_sha256, _finding().finding_content_sha256)
        self.assertEqual(restored.status.value, "APPROVED")
        self.assertEqual(decisions, (outcome.decision,))

    def test_database_repository_rejects_publish_without_persisted_a4_evidence(self):
        service = RuleVersionService(
            SqlAlchemyRuleVersionRepository(self.sessions),
            SequentialIdGenerator(values=(
                "00000000-0000-0000-0000-000000000010",
                "00000000-0000-0000-0000-000000000011",
            )),
            self.clock,
            _LegacyVerifier(),
        )
        version = service.create_version(CreateRuleVersion(
            rule_set_id="00000000-0000-0000-0000-000000000009",
            rule_key="db-rule",
            rule_set_name="Database rule",
            content_json='{"threshold":10}',
            change_summary="initial version",
            provenance=RuleProvenance(
                source_type="manual", status="verified", claims_allowed=True
            ),
        ))
        evaluating = service.request_evaluation(
            version.rule_version_id, "dataset-real"
        )
        service.complete_evaluation(CompleteEvaluationGate(
            rule_version_id=version.rule_version_id,
            gate_id=evaluating.evaluation_gate.gate_id,
            evaluation_run_id="run-db",
            status="PASSED",
            provisional=False,
            claims_allowed=True,
            report_sha256="a" * 64,
        ))
        with self.assertRaisesRegex(PermanentError, "persisted release evidence"):
            service.publish(PublishRuleVersion(
                rule_version_id=version.rule_version_id,
                approver_id="reviewer-42",
            ))
        self.assertEqual(
            service.list_versions(version.rule_set_id)[0].status.value,
            "WAITING_APPROVAL",
        )

    def test_provisional_dataset_manifest_and_samples_round_trip(self):
        repository = SqlAlchemyDatasetVersionRepository(self.sessions)
        service = DatasetVersionService(
            repository,
            SequentialIdGenerator(
                values=("00000000-0000-0000-0000-000000000020",)
            ),
            self.clock,
        )
        version = service.create_version(CreateDatasetVersion(
            dataset_name="phase6-provisional",
            requested_status="PROVISIONAL",
            change_summary="engineering flow only",
            provenance=DatasetProvenance(
                status="provisional",
                claims_allowed=False,
                source_description="synthetic human labels remain 0/4",
                source_manifest_sha256="b" * 64,
            ),
            samples=(DatasetSampleInput(
                sample_id="sample-1",
                document_id="doc-1",
                document_sha256="c" * 64,
                split="FROZEN_TEST",
                source_type="PROVISIONAL",
                provenance_status="provisional",
                label_version="provisional-v1",
                label_json='{"label":"navigation-hint-only"}',
                review_input_sha256="d" * 64,
                evidence_sha256="e" * 64,
            ),),
        ))
        self.assertEqual(repository.get_version(version.dataset_version_id), version)

    def test_phase6_columns_exist_in_metadata(self):
        columns = {
            table: {item["name"] for item in inspect(self.engine).get_columns(table)}
            for table in ("findings", "human_decisions", "rule_versions", "dataset_versions")
        }
        self.assertIn("finding_content_sha256", columns["findings"])
        self.assertIn("decision_sha256", columns["human_decisions"])
        self.assertIn("evaluation_gate_json", columns["rule_versions"])
        self.assertIn("manifest_json", columns["dataset_versions"])


if __name__ == "__main__":
    unittest.main()
