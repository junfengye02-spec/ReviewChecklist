from __future__ import annotations

import hashlib
import time
import unittest
from datetime import date

from pydantic import ValidationError

from tender_review.evaluation.public import (
    ProvisionalCandidateCase,
    ProvisionalEvaluationInput,
    ProvisionalVariantRun,
    stable_sha256,
)
from tender_review.retrieval.public import (
    BM25Retriever,
    RetrievalDocument,
    SearchHit,
    SearchResult,
)
from tender_review.review.public import (
    ComparisonResult,
    ComparisonToolInput,
    DateComparisonTool,
    DateExtraction,
    DateRule,
    ExtractionSource,
    FakeLlmProvider,
    FakeReviewTool,
    NumberExtraction,
    NumericRangeComparisonTool,
    NumericRangeRule,
    ReviewGraphNode,
    ReviewGraphState,
    ReviewInputProvenance,
    ReviewLifecycle,
    ReviewProcessingStage,
    ReviewRequest,
    SetComparisonTool,
    SetExtraction,
    SetRule,
    SingleReviewWorkflow,
    StructuredExtraction,
    TextExtraction,
    TextPresenceComparisonTool,
    TextPresenceRule,
    ToolRequest,
    provisional_review_provenance,
    transition_review_state,
)
from tender_review.shared.contracts import CallContext
from tender_review.shared.errors import ConflictError, ErrorCategory, RetryableError
from tender_review.shared.ids import SequentialIdGenerator


SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64
CALL = CallContext(call_id="phase5-test", timeout_seconds=1.0, max_attempts=1)


def verified_provenance() -> ReviewInputProvenance:
    return ReviewInputProvenance(
        source_kind="verified_retrieval",
        status="verified",
        claims_allowed=True,
        dataset_version_id="verified-dataset-v1",
        input_sha256=SHA_A,
        results_sha256=SHA_B,
        variant="bm25",
    )


def evidence_hit(*, locatable: bool = True) -> SearchHit:
    return SearchHit(
        chunk_id="chunk-authorization",
        document_id="document-1",
        text="The bid includes a signed authorization letter.",
        section_path=("Submission", "Attachments") if locatable else (),
        page_start=7 if locatable else None,
        page_end=7 if locatable else None,
        score=3.5,
        source="bm25",
        rank=1,
    )


def extraction_source(*, excerpt: str | None = None) -> ExtractionSource:
    return ExtractionSource(
        source_id="source-1",
        document_id="document-1",
        chunk_id="chunk-authorization",
        page_number=7,
        section_path=("Submission", "Attachments"),
        excerpt=excerpt or "signed authorization letter",
    )


def text_rule() -> TextPresenceRule:
    return TextPresenceRule(
        review_item_id="authorization-letter",
        field_name="authorization_text",
        required_terms=("signed", "authorization"),
        mode="all",
    )


def text_extraction(*, source: ExtractionSource | None = None) -> StructuredExtraction:
    return StructuredExtraction(
        review_item_id="authorization-letter",
        fields=(
            TextExtraction(
                field_name="authorization_text",
                value="The bid includes a signed authorization letter.",
                sources=(source or extraction_source(),),
            ),
        ),
    )


def review_request(
    extraction: StructuredExtraction,
    *,
    call: CallContext = CALL,
    provenance: ReviewInputProvenance | None = None,
    result: SearchResult | None = None,
) -> tuple[ReviewRequest, FakeLlmProvider]:
    llm = FakeLlmProvider((extraction.model_dump_json(),))
    request = ReviewRequest(
        review_job_id="review-job-1",
        query="Does the bid include a signed authorization letter?",
        document_ids=("document-1",),
        rule=text_rule(),
        provenance=provenance or verified_provenance(),
        call=call,
        retrieval_result=result
        or SearchResult(retriever="bm25", hits=(evidence_hit(),)),
    )
    return request, llm


def execute_tool(tool, rule, field) -> ComparisonResult:
    extraction = StructuredExtraction(
        review_item_id=rule.review_item_id,
        fields=(field,),
    )
    result = tool.execute(
        ToolRequest(
            tool_name=tool.name,
            input_json=ComparisonToolInput(
                rule=rule,
                extraction=extraction,
            ).model_dump_json(),
            call=CALL,
        )
    )
    return ComparisonResult.model_validate_json(result.output_json)


class StructuredExtractionContractTests(unittest.TestCase):
    def test_invalid_empty_extra_wrong_type_and_out_of_bounds_outputs_are_rejected(
        self,
    ) -> None:
        with self.assertRaises(ValidationError):
            StructuredExtraction(review_item_id="item", fields=())
        with self.assertRaises(ValidationError):
            StructuredExtraction.model_validate(
                {
                    "schema_version": 1,
                    "review_item_id": "item",
                    "fields": [],
                    "free_text_conclusion": "compliant",
                }
            )
        with self.assertRaises(ValidationError):
            ExtractionSource.model_validate(
                {
                    "source_id": "source",
                    "document_id": "document-1",
                    "chunk_id": "chunk-1",
                    "page_number": "7",
                    "section_path": ["Section"],
                    "excerpt": "evidence",
                }
            )
        with self.assertRaises(ValidationError):
            NumberExtraction(
                field_name="score",
                value=1_000_000_000_001.0,
                sources=(extraction_source(),),
            )
        with self.assertRaises(ValidationError):
            StructuredExtraction.model_validate(
                {
                    "schema_version": 2,
                    "review_item_id": "item",
                    "fields": [
                        {
                            "value_type": "text",
                            "field_name": "value",
                            "value": "text",
                            "sources": [],
                        }
                    ],
                }
            )

    def test_provisional_provenance_cannot_be_upgraded_to_claimable(self) -> None:
        with self.assertRaisesRegex(ValidationError, "claims_allowed=false"):
            ReviewInputProvenance(
                source_kind="provisional_retrieval",
                status="provisional",
                claims_allowed=True,
                dataset_version_id="provisional-v1",
                input_sha256=SHA_A,
                results_sha256=SHA_B,
                variant="bm25",
            )


class DeterministicReviewToolTests(unittest.TestCase):
    def test_date_set_numeric_range_and_text_tools_return_auditable_results(
        self,
    ) -> None:
        source = extraction_source()
        cases = (
            (
                DateComparisonTool(),
                DateRule(
                    review_item_id="date-item",
                    field_name="deadline",
                    operator="on_or_before",
                    expected=date(2026, 8, 1),
                ),
                DateExtraction(
                    field_name="deadline",
                    value=date(2026, 7, 31),
                    sources=(source,),
                ),
            ),
            (
                SetComparisonTool(),
                SetRule(
                    review_item_id="set-item",
                    field_name="attachments",
                    mode="contains_all",
                    expected_values=("license", "authorization"),
                ),
                SetExtraction(
                    field_name="attachments",
                    values=("license", "authorization", "price-list"),
                    sources=(source,),
                ),
            ),
            (
                NumericRangeComparisonTool(),
                NumericRangeRule(
                    review_item_id="score-item",
                    field_name="score",
                    expected_minimum=80.0,
                    expected_maximum=100.0,
                    unit="points",
                ),
                NumberExtraction(
                    field_name="score",
                    value=88.0,
                    unit="points",
                    sources=(source,),
                ),
            ),
            (
                TextPresenceComparisonTool(),
                text_rule(),
                TextExtraction(
                    field_name="authorization_text",
                    value="A signed authorization letter is attached.",
                    sources=(source,),
                ),
            ),
        )
        for tool, rule, field in cases:
            with self.subTest(tool=tool.name):
                result = execute_tool(tool, rule, field)
                self.assertTrue(result.passed)
                self.assertEqual(result.tool_name, tool.name)
                self.assertEqual(result.tool_version, "1.0.0")
                self.assertEqual(result.sources, (source,))


class ReviewStateGraphTests(unittest.TestCase):
    def test_graph_node_processing_stage_and_lifecycle_are_separate(self) -> None:
        state = ReviewGraphState(
            review_job_id="review-job-1",
            rule=text_rule(),
            provenance=verified_provenance(),
        )
        with self.assertRaisesRegex(ConflictError, "INPUT to DONE"):
            transition_review_state(state, ReviewGraphNode.DONE)
        with self.assertRaisesRegex(ConflictError, "cannot rewrite"):
            transition_review_state(
                state,
                ReviewGraphNode.RETRIEVAL,
                provenance=verified_provenance(),
            )

        retrieving = transition_review_state(state, ReviewGraphNode.RETRIEVAL)
        self.assertEqual(retrieving.node, ReviewGraphNode.RETRIEVAL)
        self.assertEqual(retrieving.stage, ReviewProcessingStage.RETRIEVING)
        self.assertEqual(retrieving.lifecycle, ReviewLifecycle.RUNNING)

    def test_normal_loop_closes_only_with_page_and_section_evidence(self) -> None:
        request, llm = review_request(text_extraction())
        workflow = SingleReviewWorkflow(
            llm,
            id_generator=SequentialIdGenerator(("finding-1",)),
        )

        state = workflow.run(request)

        self.assertEqual(state.node, ReviewGraphNode.DONE)
        self.assertEqual(state.lifecycle, ReviewLifecycle.COMPLETED)
        self.assertEqual(state.stage, ReviewProcessingStage.REPORTING)
        self.assertEqual(
            state.visited_nodes,
            (
                ReviewGraphNode.INPUT,
                ReviewGraphNode.RETRIEVAL,
                ReviewGraphNode.EVIDENCE_VALIDATION,
                ReviewGraphNode.EXTRACTION,
                ReviewGraphNode.COMPARISON,
                ReviewGraphNode.CONCLUSION,
                ReviewGraphNode.EVIDENCE_INTEGRITY,
                ReviewGraphNode.DONE,
            ),
        )
        self.assertIsNotNone(state.finding)
        assert state.finding is not None
        self.assertEqual(state.finding.finding_id, "finding-1")
        self.assertEqual(state.finding.conclusion, "compliant")
        self.assertEqual(len(state.finding.evidence), 1)
        evidence = state.finding.evidence[0]
        self.assertEqual(evidence.chunk_id, "chunk-authorization")
        self.assertEqual(evidence.page_number, 7)
        self.assertEqual(evidence.section_path, ("Submission", "Attachments"))
        self.assertEqual(
            evidence.text_sha256,
            hashlib.sha256(evidence.excerpt.encode("utf-8")).hexdigest(),
        )
        self.assertEqual(
            [record.outcome for record in state.call_records],
            [
                "success",
                "success",
            ],
        )
        self.assertTrue(llm.calls[0].call.call_id.startswith("phase5-test:extract:"))

    def test_workflow_can_retrieve_through_the_stable_retriever_port(self) -> None:
        request, llm = review_request(text_extraction())
        request = request.model_copy(update={"retrieval_result": None})
        retriever = BM25Retriever(
            (
                RetrievalDocument(
                    chunk_id="chunk-authorization",
                    document_id="document-1",
                    text="The bid includes a signed authorization letter.",
                    section_path=("Submission", "Attachments"),
                    page_start=7,
                    page_end=7,
                ),
            )
        )

        state = SingleReviewWorkflow(llm, retriever=retriever).run(request)

        self.assertEqual(state.node, ReviewGraphNode.DONE)
        self.assertEqual(
            [record.operation for record in state.call_records],
            ["retrieve", "extract", "tool_text_presence"],
        )
        self.assertTrue(
            state.call_records[0].call_id.startswith("phase5-test:retrieve:")
        )

    def test_unlocatable_retrieval_and_bad_extraction_cannot_create_findings(
        self,
    ) -> None:
        no_location = SearchResult(
            retriever="bm25",
            hits=(evidence_hit(locatable=False),),
        )
        request, llm = review_request(text_extraction(), result=no_location)
        state = SingleReviewWorkflow(llm).run(request)
        self.assertEqual(state.node, ReviewGraphNode.NEED_MORE_EVIDENCE)
        self.assertEqual(state.failure.category, ErrorCategory.INSUFFICIENT_EVIDENCE)
        self.assertIsNone(state.finding)
        self.assertEqual(llm.calls, [])

        bad_source = extraction_source(excerpt="not present in the chunk")
        request, llm = review_request(text_extraction(source=bad_source))
        state = SingleReviewWorkflow(llm).run(request)
        self.assertEqual(state.node, ReviewGraphNode.NEED_MORE_EVIDENCE)
        self.assertEqual(state.failure.code, "extraction_evidence_invalid")
        self.assertIsNone(state.finding)

        fabricated = StructuredExtraction(
            review_item_id="authorization-letter",
            fields=(
                TextExtraction(
                    field_name="authorization_text",
                    value="Fabricated signed authorization statement.",
                    sources=(extraction_source(),),
                ),
            ),
        )
        request, llm = review_request(fabricated)
        state = SingleReviewWorkflow(llm).run(request)
        self.assertEqual(state.node, ReviewGraphNode.NEED_MORE_EVIDENCE)
        self.assertIn("not grounded", state.reason)
        self.assertIsNone(state.finding)

    def test_invalid_json_retries_then_hands_off_to_a_human(self) -> None:
        call = CallContext(
            call_id="invalid-json",
            timeout_seconds=1.0,
            max_attempts=2,
        )
        request, _ = review_request(text_extraction(), call=call)
        llm = FakeLlmProvider(("not-json", '{"fields": []}'))

        state = SingleReviewWorkflow(llm).run(request)

        self.assertEqual(state.node, ReviewGraphNode.HUMAN_HANDOFF)
        self.assertEqual(state.lifecycle, ReviewLifecycle.WAITING_HUMAN)
        self.assertEqual(state.failure.code, "invalid_external_output")
        self.assertTrue(state.failure.retryable)
        self.assertEqual(
            [record.outcome for record in state.call_records],
            ["invalid_output", "invalid_output"],
        )
        self.assertEqual(len({record.call_id for record in state.call_records}), 2)
        self.assertIsNone(state.finding)

    def test_rate_limit_is_retried_then_handed_off_without_a_conclusion(self) -> None:
        call = CallContext(
            call_id="rate-limited-model",
            timeout_seconds=1.0,
            max_attempts=2,
        )
        request, _ = review_request(text_extraction(), call=call)
        llm = FakeLlmProvider(
            (
                RetryableError("model returned HTTP 429", code="model_rate_limited"),
                RetryableError("model returned HTTP 429", code="model_rate_limited"),
            )
        )

        state = SingleReviewWorkflow(llm).run(request)

        self.assertEqual(state.node, ReviewGraphNode.HUMAN_HANDOFF)
        self.assertEqual(state.lifecycle, ReviewLifecycle.WAITING_HUMAN)
        self.assertEqual(state.failure.code, "model_rate_limited")
        self.assertTrue(state.failure.retryable)
        self.assertEqual(
            [record.outcome for record in state.call_records],
            ["retryable_error", "retryable_error"],
        )
        self.assertEqual(len(llm.calls), 2)
        self.assertIsNone(state.finding)

    def test_empty_model_output_is_retried_then_handed_off(self) -> None:
        call = CallContext(
            call_id="empty-model-output",
            timeout_seconds=1.0,
            max_attempts=2,
        )
        request, _ = review_request(text_extraction(), call=call)
        llm = FakeLlmProvider(("", ""))

        state = SingleReviewWorkflow(llm).run(request)

        self.assertEqual(state.node, ReviewGraphNode.HUMAN_HANDOFF)
        self.assertEqual(state.lifecycle, ReviewLifecycle.WAITING_HUMAN)
        self.assertEqual(state.failure.code, "invalid_external_output")
        self.assertEqual(
            [record.outcome for record in state.call_records],
            ["invalid_output", "invalid_output"],
        )
        self.assertEqual(len(llm.calls), 2)
        self.assertIsNone(state.finding)

    def test_timeout_is_enforced_retried_and_handed_off(self) -> None:
        class SlowLlm:
            def complete(self, request):
                time.sleep(0.05)
                return FakeLlmProvider((text_extraction().model_dump_json(),)).complete(
                    request
                )

        call = CallContext(
            call_id="slow-model",
            timeout_seconds=0.005,
            max_attempts=2,
        )
        request, _ = review_request(text_extraction(), call=call)

        state = SingleReviewWorkflow(SlowLlm()).run(request)

        self.assertEqual(state.node, ReviewGraphNode.HUMAN_HANDOFF)
        self.assertEqual(state.failure.code, "external_call_timeout")
        self.assertEqual(
            [record.outcome for record in state.call_records],
            ["timeout", "timeout"],
        )
        self.assertTrue(all(record.retryable for record in state.call_records))
        self.assertTrue(
            all(record.timeout_seconds == 0.005 for record in state.call_records)
        )

    def test_invalid_tool_output_hands_off_without_a_fake_conclusion(self) -> None:
        request, llm = review_request(text_extraction())
        workflow = SingleReviewWorkflow(
            llm,
            tools=(FakeReviewTool("text_presence", output_json="not-json"),),
        )

        state = workflow.run(request)

        self.assertEqual(state.node, ReviewGraphNode.HUMAN_HANDOFF)
        self.assertEqual(state.failure.code, "invalid_external_output")
        self.assertIsNone(state.finding)

    def test_stage4_provisional_contract_propagates_unchanged_to_completed_state(
        self,
    ) -> None:
        candidate = ProvisionalCandidateCase(
            case_id="case-1",
            query_id="query-1",
            query="authorization letter",
            document_id="document-1",
            document_sha256=SHA_A,
            source_case_sha256=SHA_B,
            candidate_chunk_ids=("chunk-authorization",),
        )
        input_payload = {
            "dataset_version_id": "phase4-provisional-v1",
            "source_work_package_sha256": SHA_B,
            "chunk_catalog_sha256": SHA_C,
            "cases": (candidate,),
        }
        evaluation_input = ProvisionalEvaluationInput(
            **input_payload,
            input_sha256=stable_sha256(
                {
                    "schema_version": 1,
                    "input_version": "phase4-provisional-input-v1",
                    **{
                        **input_payload,
                        "cases": [candidate.model_dump(mode="json")],
                    },
                    "status": "provisional",
                    "claims_allowed": False,
                }
            ),
        )
        run_payload = {
            "input_sha256": evaluation_input.input_sha256,
            "dataset_version_id": evaluation_input.dataset_version_id,
            "source_work_package_sha256": evaluation_input.source_work_package_sha256,
            "chunk_catalog_sha256": evaluation_input.chunk_catalog_sha256,
            "shared_config_sha256": SHA_A,
            "implementation_version_sha256": SHA_B,
            "variant": "bm25",
            "retriever_config_sha256": SHA_C,
            "cases": (),
        }
        variant_run = ProvisionalVariantRun(
            **run_payload,
            results_sha256=stable_sha256(
                {
                    "schema_version": 1,
                    "artifact_version": "phase4-provisional-run-v1",
                    **{**run_payload, "cases": []},
                }
            ),
        )
        provenance = provisional_review_provenance(evaluation_input, variant_run)
        request, llm = review_request(text_extraction(), provenance=provenance)

        state = SingleReviewWorkflow(llm).run(request)

        self.assertEqual(state.node, ReviewGraphNode.DONE)
        self.assertEqual(state.provenance, provenance)
        self.assertEqual(state.provenance.status, "provisional")
        self.assertFalse(state.provenance.claims_allowed)
        self.assertEqual(state.provenance.input_sha256, evaluation_input.input_sha256)
        self.assertEqual(state.provenance.results_sha256, variant_run.results_sha256)


if __name__ == "__main__":
    unittest.main()
