from __future__ import annotations

import unittest

from tender_review.evaluation import (
    ProvisionalEvaluationInput,
    ProvisionalVariantRun,
    build_provisional_input,
    render_provisional_artifacts,
    run_provisional_comparison,
    stable_sha256,
)
from tender_review.evaluation import default_provisional_config
from tender_review.evaluation.annotation import (
    AnnotationWorkPackage,
    AutomatedCandidatePrefill,
    CandidateChunk,
    WorkPackageCase,
    WorkPackageDocument,
)
from tender_review.evaluation.provisional import ProvisionalCandidateCase


def _chunk(chunk_id: str, text: str = "alpha beta") -> CandidateChunk:
    return CandidateChunk(
        chunk_id=chunk_id,
        document_id="doc-1",
        document_sha256="1" * 64,
        page_start=1,
        page_end=1,
        text=text,
        text_sha256="2" * 64,
    )


class ProvisionalContractTests(unittest.TestCase):
    def test_input_has_navigation_only_provenance_and_hash(self) -> None:
        case_payload = ProvisionalCandidateCase(
            case_id="case-1",
            query_id="query-1",
            query="alpha",
            document_id="doc-1",
            document_sha256="1" * 64,
            source_case_sha256="3" * 64,
            candidate_chunk_ids=("chunk-1",),
        ).model_dump(mode="json")
        payload = {
            "schema_version": 1,
            "input_version": "phase4-provisional-input-v1",
            "dataset_version_id": "candidate",
            "source_work_package_sha256": "4" * 64,
            "chunk_catalog_sha256": "5" * 64,
            "cases": [case_payload],
            "status": "provisional",
            "claims_allowed": False,
        }
        canonical = payload
        contract = ProvisionalEvaluationInput(
            **canonical,
            input_sha256=stable_sha256(canonical),
        )
        self.assertFalse(contract.claims_allowed)
        self.assertFalse(contract.cases[0].is_human_annotation)
        self.assertEqual(contract.cases[0].candidate_label_provenance, "navigation_hint")

    def test_three_variants_share_input_config_and_case_coverage(self) -> None:
        package_payload = {
            "package_id": "package",
            "source_description": "candidate package",
            "source_dataset_sha256": "6" * 64,
            "source_manifest_sha256": "7" * 64,
            "phase3_audit_sha256": "8" * 64,
            "phase3_audit_report_sha256": "9" * 64,
            "chunk_catalog_sha256": "a" * 64,
            "documents": [WorkPackageDocument(
                document_id="doc-1",
                source_manifest_document_id="doc-1",
                source_relative_path="doc.pdf",
                document_sha256="1" * 64,
                parse_artifact_sha256="b" * 64,
                chunk_set_sha256="c" * 64,
                chunk_count=2,
            ).model_dump(mode="json")],
            "cases": [WorkPackageCase(
                case_id="case-1",
                query_id="query-1",
                query="alpha",
                source_case_sha256="d" * 64,
                source_opinion_json="opinion.json",
                source_opinion_json_sha256="e" * 64,
                review_item="1",
                document_id="doc-1",
                source_manifest_document_id="doc-1",
                document_sha256="1" * 64,
                automated_prefill=AutomatedCandidatePrefill(
                    candidate_chunk_ids=("chunk-1",),
                ),
            ).model_dump(mode="json")],
            "required_human_cases": 1,
            "annotation_policy": "human labels required; hints are navigation only",
        }
        package_payload["schema_version"] = 1
        package_payload["package_kind"] = "chunk_retrieval_candidate_annotation"
        canonical_package = package_payload
        package = AnnotationWorkPackage(
            **canonical_package,
            work_package_sha256=stable_sha256(canonical_package),
        )
        chunks = (_chunk("chunk-1", "alpha"), _chunk("chunk-2", "beta"))
        input_contract = build_provisional_input(package, chunks)
        runs, report = run_provisional_comparison(
            input_contract=input_contract,
            chunks=chunks,
        )
        self.assertEqual({run.variant for run in runs}, {"bm25", "vector", "hybrid_rrf"})
        self.assertEqual({run.input_sha256 for run in runs}, {input_contract.input_sha256})
        self.assertEqual({run.chunk_catalog_sha256 for run in runs}, {input_contract.chunk_catalog_sha256})
        self.assertFalse(report.claims_allowed)
        self.assertEqual(report.required_human_cases, 1)
        self.assertEqual(report.human_annotation_cases, 0)
        self.assertIn(report.default_strategy_candidate, {"bm25", "vector", "hybrid_rrf"})

        artifacts = render_provisional_artifacts(
            input_contract=input_contract,
            runs=runs,
            report=report,
            config=default_provisional_config(),
        )
        self.assertIn("error_analysis.json", artifacts)
        self.assertIn("claims_allowed", artifacts["README.md"] or "claims_allowed")

    def test_provisional_run_rejects_tampered_hash(self) -> None:
        with self.assertRaises(ValueError):
            ProvisionalVariantRun.model_validate(
                {
                    "variant": "bm25",
                    "input_sha256": "1" * 64,
                    "dataset_version_id": "candidate",
                    "source_work_package_sha256": "2" * 64,
                    "chunk_catalog_sha256": "3" * 64,
                    "shared_config_sha256": "4" * 64,
                    "retriever_config_sha256": "5" * 64,
                    "cases": [],
                    "results_sha256": "6" * 64,
                }
            )


if __name__ == "__main__":
    unittest.main()
