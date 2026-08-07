from __future__ import annotations

import json
import unittest

from tender_review.findings.public import FindingSummary, HumanDecision
from tender_review.retrieval.public import SearchHit, SearchResult
from tender_review.review.langgraph_workflow import LangGraphReviewWorkflow
from tender_review.review.public import (
    ExtractionSource,
    FakeLlmProvider,
    ReviewGraphState,
    ReviewRequest,
    ReviewWorkflow,
    SingleReviewWorkflow,
    StructuredExtraction,
    TextExtraction,
    TextPresenceRule,
)
from tender_review.rule_management.public import RuleVersion
from tender_review.shared.contracts import CallContext
from tender_review.shared.ids import SequentialIdGenerator


SHA_A = "a" * 64
SHA_B = "b" * 64


def _request(
    *,
    retrieval_result: SearchResult | None,
    max_attempts: int = 1,
) -> ReviewRequest:
    return ReviewRequest(
        review_job_id="a0-review-job",
        query="Does the bid include a signed authorization letter?",
        document_ids=("document-1",),
        rule=TextPresenceRule(
            review_item_id="authorization-letter",
            field_name="authorization_text",
            required_terms=("signed", "authorization"),
            mode="all",
        ),
        provenance={
            "source_kind": "verified_retrieval",
            "status": "verified",
            "claims_allowed": True,
            "dataset_version_id": "verified-dataset-v1",
            "input_sha256": SHA_A,
            "results_sha256": SHA_B,
            "variant": "bm25",
        },
        call=CallContext(
            call_id="a0-contract",
            timeout_seconds=1.0,
            max_attempts=max_attempts,
        ),
        retrieval_result=retrieval_result,
    )


def _hit(*, locatable: bool = True) -> SearchHit:
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


def _extraction() -> StructuredExtraction:
    return StructuredExtraction(
        review_item_id="authorization-letter",
        fields=(
            TextExtraction(
                field_name="authorization_text",
                value="The bid includes a signed authorization letter.",
                sources=(
                    ExtractionSource(
                        source_id="source-1",
                        document_id="document-1",
                        chunk_id="chunk-authorization",
                        page_number=7,
                        section_path=("Submission", "Attachments"),
                        excerpt="signed authorization letter",
                    ),
                ),
            ),
        ),
    )


def _single_review_scenarios() -> dict[str, tuple[ReviewWorkflow, ReviewRequest]]:
    locatable = SearchResult(retriever="bm25", hits=(_hit(),))
    unlocatable = SearchResult(retriever="bm25", hits=(_hit(locatable=False),))
    return {
        "DONE": (
            SingleReviewWorkflow(
                FakeLlmProvider((_extraction().model_dump_json(),)),
                id_generator=SequentialIdGenerator(("a0-finding",)),
            ),
            _request(retrieval_result=locatable),
        ),
        "NEED_MORE_EVIDENCE": (
            SingleReviewWorkflow(FakeLlmProvider()),
            _request(retrieval_result=unlocatable),
        ),
        "WAITING_HUMAN": (
            SingleReviewWorkflow(FakeLlmProvider(("not-json",))),
            _request(retrieval_result=locatable),
        ),
        "FAILED": (
            SingleReviewWorkflow(FakeLlmProvider()),
            _request(retrieval_result=None),
        ),
    }


def _normalized_state(state: ReviewGraphState) -> dict[str, object]:
    payload = state.model_dump(mode="json")
    for record in payload["call_records"]:
        # Wall-clock duration is evidence, but it is not a stable contract value.
        record["duration_ms"] = 0.0
    return payload


def _terminal_snapshots() -> dict[str, dict[str, object]]:
    snapshots: dict[str, dict[str, object]] = {}
    for terminal, (workflow, request) in _single_review_scenarios().items():
        state = workflow.run(request)
        if not isinstance(state, ReviewGraphState):
            raise TypeError(f"{terminal} did not return ReviewGraphState")
        snapshots[terminal] = _normalized_state(state)
    return snapshots


class A0ReviewWorkflowContractTests(unittest.TestCase):
    def test_review_workflow_implementations_satisfy_public_protocol(self) -> None:
        workflows = {
            "single": SingleReviewWorkflow(FakeLlmProvider()),
            "langgraph": LangGraphReviewWorkflow(
                SingleReviewWorkflow(FakeLlmProvider())
            ),
        }

        for name, workflow in workflows.items():
            with self.subTest(implementation=name):
                self.assertIsInstance(workflow, ReviewWorkflow)

    def test_review_workflows_match_the_same_terminal_golden_snapshot(self) -> None:
        expected = _terminal_snapshots()
        adapters = {
            "single": lambda workflow: workflow,
            "langgraph": LangGraphReviewWorkflow,
        }

        for implementation, adapt in adapters.items():
            scenarios = {
                terminal: (adapt(workflow), request)
                for terminal, (workflow, request) in _single_review_scenarios().items()
            }
            for terminal, (workflow, request) in scenarios.items():
                with self.subTest(
                    implementation=implementation,
                    terminal=terminal,
                ):
                    state = workflow.run(request)
                    self.assertIsInstance(state, ReviewGraphState)
                    self.assertEqual(_normalized_state(state), expected[terminal])


class A0SchemaContractTests(unittest.TestCase):
    def test_domain_schemas_are_deterministic(self) -> None:
        models = {
            "ReviewRequest": ReviewRequest,
            "ReviewGraphState": ReviewGraphState,
            "FindingSummary": FindingSummary,
            "HumanDecision": HumanDecision,
            "RuleVersion": RuleVersion,
        }

        for name, model in models.items():
            with self.subTest(schema=name):
                first = json.dumps(
                    model.model_json_schema(),
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                second = json.dumps(
                    model.model_json_schema(),
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                self.assertEqual(first, second)
                self.assertTrue(first.startswith("{"))


if __name__ == "__main__":
    unittest.main()
