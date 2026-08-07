from __future__ import annotations

import unittest

from pydantic import ValidationError

from tender_review.evaluation import (
    RankedChunkHit,
    RankedSearchResult,
    RetrievalDataset,
    RetrievalEvaluationInput,
    RetrievalLabel,
    build_evaluation_report,
    evaluate_variant,
    stable_sha256,
)


def result(*chunk_ids: str) -> RankedSearchResult:
    return RankedSearchResult(
        retriever="test",
        hits=tuple(
            RankedChunkHit(
                chunk_id=chunk_id,
                document_id="document-1",
                text=chunk_id,
                score=float(len(chunk_ids) - index),
                source="test",
                rank=index + 1,
            )
            for index, chunk_id in enumerate(chunk_ids)
        ),
    )


def human_label(
    query_id: str = "q1",
    *,
    review_status: str = "approved",
    annotation_source: str = "human",
    reviewer_id: str | None = "reviewer-2",
    review_source: str | None = "human",
) -> RetrievalLabel:
    return RetrievalLabel(
        query_id=query_id,
        case_id=query_id,
        query="eligibility",
        document_id="document-1",
        document_sha256="1" * 64,
        source_case_sha256="2" * 64,
        relevant_chunk_ids=("a",),
        annotator_id="reviewer-1",
        annotation_source=annotation_source,
        reviewer_id=reviewer_id,
        review_source=review_source,
        review_status=review_status,
    )


def frozen_dataset(*labels: RetrievalLabel) -> RetrievalDataset:
    payload = {
        "schema_version": 1,
        "dataset_version_id": "dataset-v1",
        "source_description": "test human labels",
        "status": "frozen",
        "source_package_sha256": "3" * 64,
        "labels": [label.model_dump(mode="json") for label in labels],
    }
    payload["dataset_sha256"] = stable_sha256(payload)
    return RetrievalDataset.model_validate(payload)


class RetrievalEvaluationTests(unittest.TestCase):
    def test_metrics_use_explicit_chunk_labels_and_ranked_results(self) -> None:
        relevant = RetrievalLabel(
            query_id="q1",
            query="eligibility",
            relevant_chunk_ids=("a", "b"),
            required_chunk_groups=(("a",), ("b",)),
            annotator_id="reviewer-1",
            review_status="approved",
        )
        no_answer = RetrievalLabel(
            query_id="q2",
            query="unrelated",
            no_answer=True,
            annotator_id="reviewer-1",
            review_status="approved",
        )

        metrics = evaluate_variant(
            variant="bm25",
            cases=(
                RetrievalEvaluationInput(relevant, result("x", "a", "b"), 3.0),
                RetrievalEvaluationInput(no_answer, result("noise"), 5.0),
            ),
        )

        self.assertEqual(metrics.cases_evaluated, 2)
        self.assertEqual(metrics.relevant_cases, 1)
        self.assertEqual(metrics.no_answer_cases, 1)
        self.assertEqual(metrics.recall_at_5, 1.0)
        self.assertEqual(metrics.recall_at_10, 1.0)
        self.assertAlmostEqual(metrics.mrr or 0.0, 0.5)
        self.assertEqual(metrics.two_sided_evidence_rate_at_10, 1.0)
        self.assertEqual(metrics.no_answer_false_positive_rate, 1.0)
        self.assertEqual(metrics.latency_p50_ms, 3.0)
        self.assertEqual(metrics.latency_p95_ms, 5.0)

    def test_recall_and_two_sided_top_k_boundaries_are_explicit(self) -> None:
        label = RetrievalLabel(
            query_id="q1",
            query="cross section",
            relevant_chunk_ids=("left", "right"),
            required_chunk_groups=(("left",), ("right",)),
            annotator_id="human-a",
            review_status="approved",
        )
        ranked = result(*([f"noise-{index}" for index in range(5)] + ["left"] + [f"more-{index}" for index in range(4)] + ["right"]))
        metrics = evaluate_variant(
            variant="hybrid",
            cases=(RetrievalEvaluationInput(label, ranked, 0.0),),
        )

        self.assertEqual(metrics.recall_at_5, 0.0)
        self.assertEqual(metrics.recall_at_10, 0.5)
        self.assertEqual(metrics.mrr, 1 / 6)
        self.assertEqual(metrics.two_sided_evidence_rate_at_10, 0.0)

    def test_metrics_leave_undefined_categories_as_none(self) -> None:
        label = human_label(review_status="reviewed")
        metrics = evaluate_variant(
            variant="vector",
            cases=(RetrievalEvaluationInput(label, result("missing")),),
        )

        self.assertEqual(metrics.recall_at_5, 0.0)
        self.assertEqual(metrics.recall_at_10, 0.0)
        self.assertEqual(metrics.mrr, 0.0)
        self.assertIsNone(metrics.two_sided_evidence_rate_at_10)
        self.assertIsNone(metrics.no_answer_false_positive_rate)
        self.assertIsNone(metrics.latency_p50_ms)

    def test_rank_duplicate_query_latency_and_label_contradictions_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValidationError, "contiguous"):
            RankedSearchResult(
                retriever="bad",
                hits=(
                    RankedChunkHit(
                        chunk_id="a",
                        document_id="d",
                        rank=2,
                        score=1,
                        source="bad",
                    ),
                ),
            )
        with self.assertRaisesRegex(ValidationError, "no-answer"):
            RetrievalLabel(
                query_id="bad",
                query="bad",
                relevant_chunk_ids=("a",),
                no_answer=True,
                annotator_id="human",
                review_status="approved",
            )
        label = human_label()
        with self.assertRaisesRegex(ValueError, "latency_ms"):
            evaluate_variant(
                variant="bm25",
                cases=(RetrievalEvaluationInput(label, result(), -1.0),),
            )
        with self.assertRaisesRegex(ValueError, "exactly once"):
            evaluate_variant(
                variant="bm25",
                cases=(
                    RetrievalEvaluationInput(label, result()),
                    RetrievalEvaluationInput(label, result()),
                ),
            )

    def test_real_report_requires_frozen_independently_reviewed_human_labels(self) -> None:
        metric = evaluate_variant(
            variant="bm25",
            cases=(RetrievalEvaluationInput(human_label(), result("a"), 1.0),),
        )
        provisional = RetrievalDataset(
            dataset_version_id="candidate",
            source_description="AI navigation candidates",
            labels=(
                human_label(
                    annotation_source="ai_prefill",
                    reviewer_id=None,
                    review_source=None,
                    review_status="draft",
                ),
            ),
        )
        with self.assertRaisesRegex(ValueError, "real baseline gate failed"):
            build_evaluation_report(dataset=provisional, variants=(metric,))

        report = build_evaluation_report(
            dataset=provisional,
            variants=(metric,),
            require_real_baseline=False,
        )
        self.assertEqual(report.report_kind, "provisional")
        self.assertFalse(report.claims_allowed)
        self.assertIn("must not be claimed", report.interpretation)

        dataset = frozen_dataset(human_label())
        with self.assertRaisesRegex(ValueError, "persisted A4 FROZEN"):
            build_evaluation_report(dataset=dataset, variants=(metric,))
        bounded = build_evaluation_report(
            dataset=dataset,
            variants=(metric,),
            require_real_baseline=False,
        )
        self.assertEqual(bounded.report_kind, "provisional")
        self.assertFalse(bounded.claims_allowed)
        self.assertRegex(bounded.report_sha256, r"^[0-9a-f]{64}$")


if __name__ == "__main__":
    unittest.main()
