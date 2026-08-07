from __future__ import annotations

import unittest

from tender_review.retrieval.fakes import FakeRetriever
from tender_review.retrieval.public import SearchResult
from tender_review.review.fakes import FakeLlmProvider
from tender_review.review.langgraph_workflow import (
    BUILD_FINDING,
    DONE,
    EXTRACT_STRUCTURED_FIELDS,
    FAILED,
    NEED_MORE_EVIDENCE,
    PERSIST_FINDING,
    RETRIEVE_EVIDENCE,
    RUN_DETERMINISTIC_TOOL,
    VALIDATE_EVIDENCE,
    VALIDATE_FINDING_EVIDENCE,
    WAITING_HUMAN,
    LangGraphReviewWorkflow,
)
from tender_review.review.models import (
    ReviewGraphNode,
    ReviewGraphState,
    ReviewLifecycle,
)
from tender_review.review.nodes import ReviewNodes, review_state
from tender_review.review.ports import ReviewWorkflow
from tender_review.review.tools import TextPresenceComparisonTool
from tender_review.review.workflow import SingleReviewWorkflow
from tender_review.shared.contracts import CallContext
from tender_review.shared.errors import ConflictError
from tender_review.shared.ids import SequentialIdGenerator

from test_phase5_review import (
    evidence_hit,
    review_request,
    text_extraction,
)


class CountingTool:
    def __init__(self) -> None:
        self._delegate = TextPresenceComparisonTool()
        self.calls = []

    @property
    def name(self) -> str:
        return self._delegate.name

    def execute(self, request):
        self.calls.append(request)
        return self._delegate.execute(request)


def _payload(state: ReviewGraphState) -> dict:
    return {
        "review_state": state.model_dump(mode="json"),
        "request_fingerprint": "a" * 64,
    }


def _contract_projection(state: ReviewGraphState) -> dict:
    payload = state.model_dump(mode="json")
    for record in payload["call_records"]:
        record["duration_ms"] = 0.0
    return payload


class ReviewNodeTests(unittest.TestCase):
    def test_each_business_node_delegates_and_advances_one_stage(self) -> None:
        request, llm = review_request(text_extraction())
        tool = CountingTool()
        persisted = []
        nodes = ReviewNodes(
            SingleReviewWorkflow(
                llm,
                tools=(tool,),
                id_generator=SequentialIdGenerator(("finding-1",)),
            ),
            finding_persister=persisted.append,
        )
        state = ReviewGraphState(
            review_job_id=request.review_job_id,
            rule=request.rule,
            provenance=request.provenance,
        )

        state = review_state(nodes.retrieve_evidence(_payload(state), request))
        self.assertEqual(state.node, ReviewGraphNode.RETRIEVAL)
        self.assertIsNotNone(state.retrieval_result)

        state = review_state(nodes.validate_evidence(_payload(state), request))
        self.assertEqual(state.node, ReviewGraphNode.EXTRACTION)
        self.assertEqual(len(state.eligible_hits), 1)

        state = review_state(
            nodes.extract_structured_fields(_payload(state), request)
        )
        self.assertEqual(state.node, ReviewGraphNode.COMPARISON)
        self.assertEqual(len(llm.calls), 1)

        state = review_state(
            nodes.run_deterministic_tool(_payload(state), request)
        )
        self.assertEqual(state.node, ReviewGraphNode.CONCLUSION)
        self.assertEqual(len(tool.calls), 1)

        state = review_state(nodes.build_finding(_payload(state), request))
        self.assertEqual(state.node, ReviewGraphNode.EVIDENCE_INTEGRITY)
        self.assertEqual(state.finding.finding_id, "finding-1")

        state = review_state(
            nodes.validate_finding_evidence(_payload(state), request)
        )
        self.assertEqual(state.node, ReviewGraphNode.EVIDENCE_INTEGRITY)

        state = review_state(nodes.persist_finding(_payload(state), request))
        self.assertEqual(state.node, ReviewGraphNode.DONE)
        self.assertEqual(persisted[0].node, ReviewGraphNode.EVIDENCE_INTEGRITY)

    def test_node_rejects_an_illegal_entry_state(self) -> None:
        request, llm = review_request(text_extraction())
        nodes = ReviewNodes(SingleReviewWorkflow(llm))
        initial = ReviewGraphState(
            review_job_id=request.review_job_id,
            rule=request.rule,
            provenance=request.provenance,
        )

        with self.assertRaisesRegex(ConflictError, "expected RETRIEVAL"):
            nodes.validate_evidence(_payload(initial), request)


class LangGraphTopologyTests(unittest.TestCase):
    def test_workflow_satisfies_the_a0_protocol_structurally(self) -> None:
        request, llm = review_request(text_extraction())
        del request
        workflow = LangGraphReviewWorkflow(SingleReviewWorkflow(llm))

        self.assertIsInstance(workflow, ReviewWorkflow)

    def test_topology_contains_exactly_the_legal_edges(self) -> None:
        request, llm = review_request(text_extraction())
        del request
        graph = LangGraphReviewWorkflow(
            SingleReviewWorkflow(llm)
        ).compiled_graph.get_graph()
        actual = {(edge.source, edge.target) for edge in graph.edges}
        expected = {
            ("__start__", RETRIEVE_EVIDENCE),
            (RETRIEVE_EVIDENCE, VALIDATE_EVIDENCE),
            (RETRIEVE_EVIDENCE, NEED_MORE_EVIDENCE),
            (RETRIEVE_EVIDENCE, FAILED),
            (VALIDATE_EVIDENCE, EXTRACT_STRUCTURED_FIELDS),
            (VALIDATE_EVIDENCE, NEED_MORE_EVIDENCE),
            (EXTRACT_STRUCTURED_FIELDS, RUN_DETERMINISTIC_TOOL),
            (EXTRACT_STRUCTURED_FIELDS, NEED_MORE_EVIDENCE),
            (EXTRACT_STRUCTURED_FIELDS, WAITING_HUMAN),
            (EXTRACT_STRUCTURED_FIELDS, FAILED),
            (RUN_DETERMINISTIC_TOOL, BUILD_FINDING),
            (RUN_DETERMINISTIC_TOOL, WAITING_HUMAN),
            (RUN_DETERMINISTIC_TOOL, FAILED),
            (BUILD_FINDING, VALIDATE_FINDING_EVIDENCE),
            (VALIDATE_FINDING_EVIDENCE, PERSIST_FINDING),
            (VALIDATE_FINDING_EVIDENCE, NEED_MORE_EVIDENCE),
            (PERSIST_FINDING, DONE),
            (DONE, "__end__"),
            (NEED_MORE_EVIDENCE, "__end__"),
            (WAITING_HUMAN, "__end__"),
            (FAILED, "__end__"),
        }

        self.assertEqual(actual, expected)
        self.assertNotIn((VALIDATE_EVIDENCE, DONE), actual)
        self.assertNotIn((BUILD_FINDING, FAILED), actual)
        self.assertNotIn((PERSIST_FINDING, WAITING_HUMAN), actual)


class LangGraphTerminalContractTests(unittest.TestCase):
    def _assert_same_contract(
        self,
        request,
        old_workflow: SingleReviewWorkflow,
        new_workflow: SingleReviewWorkflow,
    ) -> ReviewGraphState:
        expected = old_workflow.run(request)
        actual = LangGraphReviewWorkflow(new_workflow).run(request)
        self.assertEqual(
            _contract_projection(actual),
            _contract_projection(expected),
        )
        return actual

    def test_done_matches_the_existing_workflow(self) -> None:
        request, old_llm = review_request(text_extraction())
        new_llm = FakeLlmProvider((text_extraction().model_dump_json(),))
        state = self._assert_same_contract(
            request,
            SingleReviewWorkflow(
                old_llm,
                id_generator=SequentialIdGenerator(("finding-1",)),
            ),
            SingleReviewWorkflow(
                new_llm,
                id_generator=SequentialIdGenerator(("finding-1",)),
            ),
        )
        self.assertEqual(state.lifecycle, ReviewLifecycle.COMPLETED)

    def test_need_more_evidence_matches_the_existing_workflow(self) -> None:
        result = SearchResult(
            retriever="fake",
            hits=(evidence_hit(locatable=False),),
        )
        request, old_llm = review_request(
            text_extraction(),
            result=result,
        )
        state = self._assert_same_contract(
            request,
            SingleReviewWorkflow(old_llm),
            SingleReviewWorkflow(FakeLlmProvider(())),
        )
        self.assertEqual(state.lifecycle, ReviewLifecycle.NEED_MORE_EVIDENCE)

    def test_waiting_human_matches_the_existing_workflow(self) -> None:
        request, _ = review_request(
            text_extraction(),
            call=CallContext(
                call_id="a1-invalid-output",
                timeout_seconds=1.0,
                max_attempts=1,
            ),
        )
        state = self._assert_same_contract(
            request,
            SingleReviewWorkflow(FakeLlmProvider(("not-json",))),
            SingleReviewWorkflow(FakeLlmProvider(("not-json",))),
        )
        self.assertEqual(state.lifecycle, ReviewLifecycle.WAITING_HUMAN)

    def test_failed_matches_the_existing_workflow(self) -> None:
        request, old_llm = review_request(text_extraction())
        request = request.model_copy(update={"retrieval_result": None})
        state = self._assert_same_contract(
            request,
            SingleReviewWorkflow(old_llm),
            SingleReviewWorkflow(FakeLlmProvider(())),
        )
        self.assertEqual(state.lifecycle, ReviewLifecycle.FAILED)


class LangGraphRecoveryTests(unittest.TestCase):
    def test_recovery_after_each_major_stage_does_not_repeat_completed_work(
        self,
    ) -> None:
        cases = (
            (RETRIEVE_EVIDENCE, (1, 0, 0, 0)),
            (EXTRACT_STRUCTURED_FIELDS, (1, 1, 0, 0)),
            (RUN_DETERMINISTIC_TOOL, (1, 1, 1, 0)),
            (BUILD_FINDING, (1, 1, 1, 0)),
        )
        for index, (pause_after, expected_paused_counts) in enumerate(cases):
            with self.subTest(pause_after=pause_after):
                request, _ = review_request(text_extraction())
                request = request.model_copy(
                    update={
                        "review_job_id": f"a1-recovery-{index}",
                        "retrieval_result": None,
                    }
                )
                retriever = FakeRetriever((evidence_hit(),))
                llm = FakeLlmProvider((text_extraction().model_dump_json(),))
                tool = CountingTool()
                persisted = []
                workflow = LangGraphReviewWorkflow(
                    SingleReviewWorkflow(
                        llm,
                        retriever=retriever,
                        tools=(tool,),
                        id_generator=SequentialIdGenerator(("finding-1",)),
                    ),
                    finding_persister=persisted.append,
                )

                paused = workflow.run(
                    request,
                    interrupt_after=(pause_after,),
                )
                paused_counts = (
                    len(retriever.calls),
                    len(llm.calls),
                    len(tool.calls),
                    len(persisted),
                )
                self.assertEqual(paused_counts, expected_paused_counts)
                self.assertEqual(paused.lifecycle, ReviewLifecycle.RUNNING)
                pointer = workflow.latest_checkpoint(request.review_job_id)
                self.assertEqual(pointer.thread_id, request.review_job_id)

                completed = workflow.run(request)

                self.assertEqual(completed.node, ReviewGraphNode.DONE)
                self.assertEqual(len(retriever.calls), 1)
                self.assertEqual(len(llm.calls), 1)
                self.assertEqual(len(tool.calls), 1)
                self.assertEqual(len(persisted), 1)
                self.assertEqual(completed.finding.finding_id, "finding-1")
                self.assertEqual(workflow.run(request), completed)
                self.assertEqual(len(persisted), 1)

    def test_recovery_may_refresh_call_context(self) -> None:
        request, llm = review_request(text_extraction())
        workflow = LangGraphReviewWorkflow(
            SingleReviewWorkflow(
                llm,
                id_generator=SequentialIdGenerator(("finding-1",)),
            )
        )
        workflow.run(request, interrupt_after=(RETRIEVE_EVIDENCE,))
        refreshed = request.model_copy(
            update={
                "call": CallContext(
                    call_id="replacement-worker-call",
                    timeout_seconds=2.0,
                    max_attempts=2,
                )
            }
        )

        completed = workflow.run(refreshed)

        self.assertEqual(completed.node, ReviewGraphNode.DONE)
        self.assertTrue(
            completed.call_records[0].call_id.startswith(
                "replacement-worker-call:extract:"
            )
        )

    def test_same_thread_rejects_changed_input(self) -> None:
        request, llm = review_request(text_extraction())
        workflow = LangGraphReviewWorkflow(SingleReviewWorkflow(llm))
        workflow.run(request, interrupt_after=(RETRIEVE_EVIDENCE,))
        changed = request.model_copy(update={"query": "different query"})

        with self.assertRaisesRegex(ConflictError, "different input"):
            workflow.run(changed)


if __name__ == "__main__":
    unittest.main()
