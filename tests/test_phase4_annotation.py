from __future__ import annotations

import hashlib
import unittest

from pydantic import ValidationError

from tender_review.evaluation import (
    AnnotationWorkPackage,
    AutomatedCandidatePrefill,
    CandidateChunk,
    WorkPackageCase,
    WorkPackageDocument,
    annotation_gaps,
    annotation_template,
    freeze_dataset,
    import_annotations,
    stable_sha256,
)


def chunk(chunk_id: str, document_id: str = "doc.pdf") -> CandidateChunk:
    return CandidateChunk(
        chunk_id=chunk_id,
        document_id=document_id,
        document_sha256="1" * 64,
        page_start=1,
        page_end=1,
        text=chunk_id,
        text_sha256=hashlib.sha256(chunk_id.encode()).hexdigest(),
    )


def package() -> tuple[AnnotationWorkPackage, tuple[CandidateChunk, ...]]:
    chunks = (chunk("chunk-a"), chunk("chunk-b"))
    document = WorkPackageDocument(
        document_id="doc.pdf",
        source_manifest_document_id="source-doc",
        source_relative_path="doc.pdf",
        document_sha256="1" * 64,
        parse_artifact_sha256="2" * 64,
        chunk_set_sha256="3" * 64,
        chunk_count=2,
    )
    cases = tuple(
        WorkPackageCase(
            case_id=f"case-{index}",
            query_id=f"case-{index}",
            query=f"query {index}",
            source_case_sha256=str(index) * 64,
            source_opinion_json="approval.json",
            source_opinion_json_sha256="4" * 64,
            review_item=str(index),
            document_id="doc.pdf",
            source_manifest_document_id="source-doc",
            document_sha256="1" * 64,
            automated_prefill=AutomatedCandidatePrefill(
                candidate_chunk_ids=("chunk-a",)
            ),
        )
        for index in (5, 6)
    )
    payload = {
        "schema_version": 1,
        "package_id": "test-package",
        "package_kind": "chunk_retrieval_candidate_annotation",
        "source_description": "real source test package",
        "source_dataset_sha256": "5" * 64,
        "source_manifest_sha256": "6" * 64,
        "phase3_audit_sha256": "7" * 64,
        "phase3_audit_report_sha256": "8" * 64,
        "chunk_catalog_sha256": stable_sha256(
            [item.model_dump(mode="json") for item in chunks]
        ),
        "documents": [document.model_dump(mode="json")],
        "cases": [item.model_dump(mode="json") for item in cases],
        "required_human_cases": 2,
        "annotation_policy": "human annotator and independent human reviewer",
    }
    return (
        AnnotationWorkPackage(
            **payload,
            work_package_sha256=stable_sha256(payload),
        ),
        chunks,
    )


def raw_decision(case_id: str, *, reviewer: str = "reviewer-b") -> dict[str, object]:
    return {
        "schema_version": 1,
        "case_id": case_id,
        "query_id": case_id,
        "document_id": "doc.pdf",
        "no_answer": False,
        "chunk_labels": [
            {
                "schema_version": 1,
                "chunk_id": "chunk-a",
                "relevance": "relevant",
            },
            {
                "schema_version": 1,
                "chunk_id": "chunk-b",
                "relevance": "not_relevant",
            },
        ],
        "required_chunk_groups": [["chunk-a"]],
        "annotator_id": "annotator-a",
        "annotation_source": "human",
        "annotated_at": "2026-07-28T10:00:00+08:00",
        "reviewer_id": reviewer,
        "review_source": "human",
        "review_state": "approved",
        "reviewed_at": "2026-07-28T11:00:00+08:00",
        "notes": "checked against PDF",
    }


class RetrievalAnnotationPipelineTests(unittest.TestCase):
    def test_template_keeps_automated_candidates_visibly_unlabeled(self) -> None:
        work_package, _ = package()
        template = annotation_template(work_package)

        decision = template["decisions"][0]
        self.assertIsNone(decision["annotator_id"])
        self.assertIsNone(decision["reviewer_id"])
        self.assertIsNone(decision["review_state"])
        self.assertIsNone(decision["chunk_labels"][0]["relevance"])
        self.assertFalse(
            work_package.cases[0].automated_prefill.is_human_annotation
        )

    def test_import_checks_human_identity_chunk_ownership_and_package_hash(self) -> None:
        work_package, chunks = package()
        raw = {
            "schema_version": 1,
            "work_package_sha256": work_package.work_package_sha256,
            "decisions": [raw_decision("case-5")],
        }
        imported = import_annotations(
            package=work_package,
            chunks=chunks,
            raw_bundle=raw,
        )
        self.assertRegex(imported.annotation_bundle_sha256, r"^[0-9a-f]{64}$")

        raw["work_package_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "different work package"):
            import_annotations(
                package=work_package,
                chunks=chunks,
                raw_bundle=raw,
            )
        raw["work_package_sha256"] = work_package.work_package_sha256
        raw["decisions"][0]["chunk_labels"][0]["chunk_id"] = "missing"
        raw["decisions"][0]["required_chunk_groups"] = [["missing"]]
        with self.assertRaisesRegex(ValueError, "unknown chunk_id"):
            import_annotations(
                package=work_package,
                chunks=chunks,
                raw_bundle=raw,
            )

    def test_same_person_review_and_non_human_provenance_cannot_import(self) -> None:
        work_package, chunks = package()
        same_person = {
            "schema_version": 1,
            "work_package_sha256": work_package.work_package_sha256,
            "decisions": [raw_decision("case-5", reviewer="annotator-a")],
        }
        with self.assertRaisesRegex(ValidationError, "different people"):
            import_annotations(
                package=work_package,
                chunks=chunks,
                raw_bundle=same_person,
            )
        same_person["decisions"][0]["reviewer_id"] = "reviewer-b"
        same_person["decisions"][0]["annotation_source"] = "ai_prefill"
        with self.assertRaises(ValidationError):
            import_annotations(
                package=work_package,
                chunks=chunks,
                raw_bundle=same_person,
            )

    def test_freeze_fails_on_gap_then_creates_hashed_dataset_version(self) -> None:
        work_package, chunks = package()
        partial = import_annotations(
            package=work_package,
            chunks=chunks,
            raw_bundle={
                "schema_version": 1,
                "work_package_sha256": work_package.work_package_sha256,
                "decisions": [raw_decision("case-5")],
            },
        )
        gaps = annotation_gaps(work_package, partial)
        self.assertEqual(len(gaps.missing_case_ids), 1)
        self.assertFalse(gaps.real_dataset_ready)
        with self.assertRaisesRegex(ValueError, "freeze gate failed"):
            freeze_dataset(
                package=work_package,
                bundle=partial,
                dataset_version_id="dataset-v1",
            )

        complete = import_annotations(
            package=work_package,
            chunks=chunks,
            raw_bundle={
                "schema_version": 1,
                "work_package_sha256": work_package.work_package_sha256,
                "decisions": [
                    raw_decision("case-5"),
                    raw_decision("case-6"),
                ],
            },
        )
        dataset = freeze_dataset(
            package=work_package,
            bundle=complete,
            dataset_version_id="dataset-v1",
        )
        self.assertEqual(dataset.status, "frozen")
        self.assertEqual(len(dataset.labels), 2)
        self.assertRegex(dataset.dataset_sha256 or "", r"^[0-9a-f]{64}$")
        self.assertTrue(
            all(label.review_status == "approved" for label in dataset.labels)
        )


if __name__ == "__main__":
    unittest.main()
