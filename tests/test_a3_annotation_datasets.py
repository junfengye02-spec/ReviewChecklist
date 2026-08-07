from __future__ import annotations

import hashlib
import unittest
from datetime import datetime, timezone
from types import SimpleNamespace

from fastapi.testclient import TestClient
from pydantic import ValidationError
from alembic import command
from alembic.config import Config
from sqlalchemy import create_engine, inspect
from tempfile import TemporaryDirectory
from pathlib import Path

from tender_review.api.app import create_app
from tender_review.bootstrap.assembly import build_container
from tender_review.evaluation.public import (
    AnnotationDatasetService,
    AnnotationEvidenceChunk,
    AnnotationSampleInput,
    AnnotationSampleStatus,
    ChunkRelevanceLabel,
    CreateAnnotationDatasetRevision,
    CreateAnnotationDatasetVersion,
    DatasetAnnotationSample,
    DatasetSplit,
    DatasetStatus,
    InMemoryAnnotationDatasetRepository,
    RepositoryAnnotationReferenceValidator,
    SubmitAnnotationLabel,
)
from tender_review.findings.public import HumanDecision, HumanDecisionType, stable_sha256
from tender_review.infrastructure.database import Base, create_session_factory
from tender_review.infrastructure.database.annotation_datasets import (
    SqlAlchemyAnnotationDatasetRepository,
)
from tender_review.shared.clock import FixedClock
from tender_review.shared.config import AppSettings
from tender_review.shared.errors import ConflictError, PermanentError
from tender_review.shared.ids import SequentialIdGenerator


NOW = datetime(2026, 8, 6, 12, 0, tzinfo=timezone.utc)


def _decision(decision_id: str, actor_id: str, minute: int) -> HumanDecision:
    payload = {
        "schema_version": 1,
        "decision_id": decision_id,
        "finding_id": "finding-1",
        "reviewer_kind": "human",
        "reviewer_id": actor_id,
        "decision": HumanDecisionType.APPROVE,
        "reason": "isolated A3 workflow contract fixture",
        "revision": None,
        "supersedes_decision_id": None,
        "decided_at": datetime(2026, 8, 6, 12, minute, tzinfo=timezone.utc),
        "review_input_sha256": "1" * 64,
        "finding_content_sha256": "2" * 64,
        "evidence_sha256": "3" * 64,
    }
    return HumanDecision(**payload, decision_sha256=stable_sha256(payload))


class DecisionResolver:
    def __init__(self, *decisions: HumanDecision) -> None:
        self.decisions = {item.decision_id: item for item in decisions}

    def get_decision(self, finding_id: str, decision_id: str) -> HumanDecision:
        decision = self.decisions[decision_id]
        if decision.finding_id != finding_id:
            raise AssertionError("fixture decision belongs to another finding")
        return decision


def _chunk(chunk_id: str, excerpt: str) -> AnnotationEvidenceChunk:
    return AnnotationEvidenceChunk(
        chunk_id=chunk_id,
        page_start=1,
        page_end=1,
        section_path=("资格要求",),
        excerpt=excerpt,
        text_sha256=hashlib.sha256(excerpt.encode("utf-8")).hexdigest(),
    )


def _sample(
    sample_id: str = "sample-1",
    *,
    document_sha256: str = "4" * 64,
    split: DatasetSplit = DatasetSplit.FROZEN_TEST,
) -> AnnotationSampleInput:
    return AnnotationSampleInput(
        sample_id=sample_id,
        finding_id="finding-1",
        document_snapshot_id="00000000-0000-0000-0000-000000000101",
        source_pdf_reference="minio://documents/4/source.pdf",
        document_sha256=document_sha256,
        source_case_sha256="5" * 64,
        rule_version_id="00000000-0000-0000-0000-000000000102",
        rule_sha256="6" * 64,
        query_id=f"query-{sample_id}",
        query="投标人需要提供什么资格证明？",
        question_label="资格审查条件",
        split=split,
        candidate_chunks=(
            _chunk("chunk-a", "投标人应提供有效资格证明。"),
            _chunk("chunk-b", "本项目不接受联合体投标。"),
        ),
    )


def _label(*chunk_ids: str) -> ChunkRelevanceLabel:
    return ChunkRelevanceLabel(no_answer=False, relevant_chunk_ids=chunk_ids)


def _command(
    dataset_version_id: str,
    decision: HumanDecision,
    label: ChunkRelevanceLabel,
) -> SubmitAnnotationLabel:
    return SubmitAnnotationLabel(
        dataset_version_id=dataset_version_id,
        sample_id="sample-1",
        actor_id=decision.reviewer_id,
        human_decision_id=decision.decision_id,
        label=label,
    )


class AnnotationDatasetWorkflowTests(unittest.TestCase):
    def setUp(self) -> None:
        self.annotation = _decision("decision-annotation", "annotator-alpha", 1)
        self.review = _decision("decision-review", "reviewer-beta", 2)
        self.adjudication = _decision("decision-adjudication", "adjudicator-gamma", 3)
        self.repository = InMemoryAnnotationDatasetRepository()
        self.service = AnnotationDatasetService(
            self.repository,
            DecisionResolver(self.annotation, self.review, self.adjudication),
            SequentialIdGenerator(prefix="dataset"),
            FixedClock(NOW),
        )

    def _create(self):
        return self.service.create_version(
            CreateAnnotationDatasetVersion(
                dataset_name="a3-isolated-contract",
                change_summary="isolated workflow fixture only",
                source_description="candidate navigation hints; no labels imported",
                source_manifest_sha256="7" * 64,
                source_work_package_sha256="8" * 64,
                samples=(_sample(),),
            )
        )

    def test_independent_review_conflict_adjudication_and_freeze(self) -> None:
        created = self._create()
        self.assertEqual(created.status, DatasetStatus.DRAFT)
        self.assertEqual(
            created.samples[0].status, AnnotationSampleStatus.PENDING_ANNOTATION
        )
        self.assertFalse(created.provenance.claims_allowed)

        annotated = self.service.submit_annotation(
            _command(created.dataset_version_id, self.annotation, _label("chunk-a"))
        )
        self.assertEqual(
            annotated.samples[0].status, AnnotationSampleStatus.PENDING_REVIEW
        )
        conflicted = self.service.submit_review(
            _command(created.dataset_version_id, self.review, _label("chunk-b"))
        )
        self.assertEqual(conflicted.status, DatasetStatus.DRAFT)
        self.assertEqual(conflicted.samples[0].status, AnnotationSampleStatus.CONFLICT)
        with self.assertRaisesRegex(PermanentError, "independently verified"):
            self.service.freeze(created.dataset_version_id)

        verified = self.service.adjudicate(
            _command(created.dataset_version_id, self.adjudication, _label("chunk-a"))
        )
        self.assertEqual(verified.status, DatasetStatus.VERIFIED)
        self.assertEqual(verified.samples[0].status, AnnotationSampleStatus.VERIFIED)
        self.assertEqual(
            verified.samples[0].final_label_sha256,
            verified.samples[0].adjudication.label_sha256,
        )
        frozen = self.service.freeze(created.dataset_version_id)
        self.assertEqual(frozen.status, DatasetStatus.FROZEN)
        self.assertEqual(frozen.samples[0].status, AnnotationSampleStatus.FROZEN)
        self.assertTrue(frozen.provenance.claims_allowed)

        with self.assertRaisesRegex(ConflictError, "immutable"):
            self.service.submit_annotation(
                _command(created.dataset_version_id, self.annotation, _label("chunk-a"))
            )

    def test_matching_independent_review_verifies_without_adjudication(self) -> None:
        created = self._create()
        label = _label("chunk-a")
        self.service.submit_annotation(
            _command(created.dataset_version_id, self.annotation, label)
        )
        verified = self.service.submit_review(
            _command(created.dataset_version_id, self.review, label)
        )
        self.assertEqual(verified.status, DatasetStatus.VERIFIED)
        self.assertIsNone(verified.samples[0].adjudication)

    def test_new_version_preserves_frozen_parent_and_resets_selected_samples(self) -> None:
        created = self._create()
        label = _label("chunk-a")
        self.service.submit_annotation(
            _command(created.dataset_version_id, self.annotation, label)
        )
        self.service.submit_review(_command(created.dataset_version_id, self.review, label))
        parent = self.service.freeze(created.dataset_version_id)

        revision = self.service.create_revision(
            CreateAnnotationDatasetRevision(
                parent_version_id=parent.dataset_version_id,
                change_summary="reset one sample after source correction",
                reset_sample_ids=("sample-1",),
            )
        )
        self.assertEqual(revision.version_number, 2)
        self.assertEqual(revision.parent_version_id, parent.dataset_version_id)
        self.assertEqual(revision.samples[0].status, AnnotationSampleStatus.PENDING_ANNOTATION)
        self.assertEqual(
            self.service.get_version(parent.dataset_version_id).status,
            DatasetStatus.FROZEN,
        )

    def test_document_hash_cannot_cross_splits_within_or_across_versions(self) -> None:
        with self.assertRaisesRegex(PermanentError, "cannot cross"):
            self.service.create_version(
                CreateAnnotationDatasetVersion(
                    dataset_name="leak-test",
                    change_summary="must reject leakage",
                    source_description="isolated",
                    source_manifest_sha256="7" * 64,
                    source_work_package_sha256="8" * 64,
                    samples=(
                        _sample("sample-1", split=DatasetSplit.OPTIMIZATION),
                        _sample("sample-2", split=DatasetSplit.VALIDATION),
                    ),
                )
            )

    def test_hash_and_chunk_reference_tampering_is_rejected(self) -> None:
        created = self._create()
        with self.assertRaisesRegex(PermanentError, "outside"):
            self.service.submit_annotation(
                _command(created.dataset_version_id, self.annotation, _label("missing"))
            )
        sample_payload = created.samples[0].model_dump(mode="json")
        sample_payload["question_label"] = "tampered"
        with self.assertRaisesRegex(ValidationError, "sample_sha256"):
            DatasetAnnotationSample.model_validate(sample_payload)

    def test_document_finding_and_rule_references_are_hash_checked(self) -> None:
        sample = _sample()
        validator = RepositoryAnnotationReferenceValidator(
            SimpleNamespace(
                get_snapshot=lambda _snapshot_id: SimpleNamespace(
                    object=SimpleNamespace(
                        sha256=sample.document_sha256,
                        object_key=sample.source_pdf_reference,
                    )
                )
            ),
            SimpleNamespace(
                get_finding=lambda _finding_id: SimpleNamespace(
                    documents=(
                        SimpleNamespace(
                            document_id=sample.document_snapshot_id,
                            document_sha256=sample.document_sha256,
                        ),
                    ),
                    rule_version_id=sample.rule_version_id,
                )
            ),
            SimpleNamespace(
                get_version=lambda _version_id: SimpleNamespace(
                    content_sha256=sample.rule_sha256
                )
            ),
        )
        validator.validate_sample(sample)
        with self.assertRaisesRegex(PermanentError, "document_sha256"):
            validator.validate_sample(
                sample.model_copy(update={"document_sha256": "9" * 64})
            )


class AnnotationDatasetDatabaseTests(unittest.TestCase):
    def setUp(self) -> None:
        self.engine = create_engine("sqlite:///:memory:")
        Base.metadata.create_all(self.engine)
        self.sessions = create_session_factory(self.engine)

    def tearDown(self) -> None:
        self.engine.dispose()

    def test_sql_repository_round_trip_and_status_projection(self) -> None:
        annotation = _decision("00000000-0000-0000-0000-000000000201", "annotator-alpha", 1)
        repository = SqlAlchemyAnnotationDatasetRepository(self.sessions)
        service = AnnotationDatasetService(
            repository,
            DecisionResolver(annotation),
            SequentialIdGenerator(
                values=("00000000-0000-0000-0000-000000000202",)
            ),
            FixedClock(NOW),
        )
        version = service.create_version(
            CreateAnnotationDatasetVersion(
                dataset_name="a3-sqlite-contract",
                change_summary="isolated sqlite fixture",
                source_description="no real gate claim",
                source_manifest_sha256="7" * 64,
                source_work_package_sha256="8" * 64,
                samples=(_sample(),),
            )
        )
        service.submit_annotation(
            _command(version.dataset_version_id, annotation, _label("chunk-a"))
        )
        restored = repository.get_version(version.dataset_version_id)
        self.assertEqual(restored.samples[0].status, AnnotationSampleStatus.PENDING_REVIEW)
        filtered = repository.list_versions(
            sample_status=AnnotationSampleStatus.PENDING_REVIEW
        )
        self.assertEqual(filtered, (restored,))
        columns = {
            item["name"]
            for item in inspect(self.engine).get_columns("dataset_annotation_samples")
        }
        self.assertIn("annotation_human_decision_id", columns)
        self.assertIn("sample_json", columns)

    def test_a3_migration_upgrades_and_downgrades_cleanly(self) -> None:
        config = Config(str(Path(__file__).parents[1] / "alembic.ini"))
        with TemporaryDirectory() as directory:
            database_url = f"sqlite:///{(Path(directory) / 'a3.sqlite3').as_posix()}"
            config.set_main_option("sqlalchemy.url", database_url)
            command.upgrade(config, "head")
            engine = create_engine(database_url)
            self.assertIn(
                "dataset_annotation_samples", inspect(engine).get_table_names()
            )
            engine.dispose()
            command.downgrade(config, "a2e7c4d9b801")
            engine = create_engine(database_url)
            self.assertNotIn(
                "dataset_annotation_samples", inspect(engine).get_table_names()
            )
            engine.dispose()


class AnnotationDatasetApiTests(unittest.TestCase):
    def setUp(self) -> None:
        self.annotation = _decision("decision-api-annotation", "annotator-alpha", 1)
        self.review = _decision("decision-api-review", "reviewer-beta", 2)
        container = build_container(AppSettings(environment="test", log_json=False))
        service = AnnotationDatasetService(
            container.annotation_dataset_repository,
            DecisionResolver(self.annotation, self.review),
            container.ids,
            container.clock,
        )
        self.client = TestClient(
            create_app(container.with_overrides(annotation_datasets=service))
        )

    def _create_payload(self) -> dict[str, object]:
        return {
            "dataset_name": "a3-api-contract",
            "change_summary": "isolated API workflow fixture",
            "source_description": "candidate hints only",
            "source_manifest_sha256": "7" * 64,
            "source_work_package_sha256": "8" * 64,
            "samples": [_sample().model_dump(mode="json")],
        }

    def test_api_create_filter_annotate_and_review_contract(self) -> None:
        self.assertIn(
            "/api/v1/annotation-datasets", self.client.app.openapi()["paths"]
        )
        created = self.client.post(
            "/api/v1/annotation-datasets", json=self._create_payload()
        )
        self.assertEqual(created.status_code, 201, created.text)
        version_id = created.json()["dataset_version_id"]
        filtered = self.client.get(
            "/api/v1/annotation-datasets",
            params={"sample_status": "PENDING_ANNOTATION"},
        )
        self.assertEqual(filtered.status_code, 200, filtered.text)
        self.assertEqual(len(filtered.json()), 1)

        annotated = self.client.post(
            f"/api/v1/annotation-datasets/{version_id}/samples/sample-1/annotations",
            json={
                "actor_id": self.annotation.reviewer_id,
                "human_decision_id": self.annotation.decision_id,
                "label": _label("chunk-a").model_dump(mode="json"),
            },
        )
        self.assertEqual(annotated.status_code, 200, annotated.text)
        self.assertEqual(annotated.json()["samples"][0]["status"], "PENDING_REVIEW")
        reviewed = self.client.post(
            f"/api/v1/annotation-datasets/{version_id}/samples/sample-1/reviews",
            json={
                "actor_id": self.review.reviewer_id,
                "human_decision_id": self.review.decision_id,
                "label": _label("chunk-a").model_dump(mode="json"),
            },
        )
        self.assertEqual(reviewed.status_code, 200, reviewed.text)
        self.assertEqual(reviewed.json()["status"], "VERIFIED")
        self.assertFalse(reviewed.json()["provenance"]["claims_allowed"])

    def test_api_rejects_nonhuman_identity_and_invalid_state(self) -> None:
        created = self.client.post(
            "/api/v1/annotation-datasets", json=self._create_payload()
        ).json()
        version_id = created["dataset_version_id"]
        invalid = self.client.post(
            f"/api/v1/annotation-datasets/{version_id}/samples/sample-1/annotations",
            json={
                "actor_id": "AI-reviewer",
                "human_decision_id": self.annotation.decision_id,
                "label": _label("chunk-a").model_dump(mode="json"),
            },
        )
        self.assertEqual(invalid.status_code, 422)
        self.assertEqual(invalid.json()["error"]["code"], "request_validation_failed")


if __name__ == "__main__":
    unittest.main()
