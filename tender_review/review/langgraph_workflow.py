from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from tender_review.shared.errors import ConflictError
from tender_review.shared.observability import CorrelationContext

from .checkpoints import ReviewCheckpointPointer, review_thread_id
from .models import (
    ReviewGraphNode,
    ReviewGraphState,
    ReviewLifecycle,
    ReviewRequest,
)
from .nodes import (
    FindingPersister,
    LangGraphReviewState,
    ReviewNodes,
    bind_request,
    review_state,
)
from .workflow import SingleReviewWorkflow


RETRIEVE_EVIDENCE = "retrieve_evidence"
VALIDATE_EVIDENCE = "validate_evidence"
EXTRACT_STRUCTURED_FIELDS = "extract_structured_fields"
RUN_DETERMINISTIC_TOOL = "run_deterministic_tool"
BUILD_FINDING = "build_finding"
VALIDATE_FINDING_EVIDENCE = "validate_finding_evidence"
PERSIST_FINDING = "persist_finding"
DONE = "done"
NEED_MORE_EVIDENCE = "need_more_evidence"
WAITING_HUMAN = "waiting_human"
FAILED = "failed"

BUSINESS_NODES = (
    RETRIEVE_EVIDENCE,
    VALIDATE_EVIDENCE,
    EXTRACT_STRUCTURED_FIELDS,
    RUN_DETERMINISTIC_TOOL,
    BUILD_FINDING,
    VALIDATE_FINDING_EVIDENCE,
    PERSIST_FINDING,
)
TERMINAL_NODES = (DONE, NEED_MORE_EVIDENCE, WAITING_HUMAN, FAILED)

_TERMINAL_ROUTE = {
    ReviewGraphNode.DONE: DONE,
    ReviewGraphNode.NEED_MORE_EVIDENCE: NEED_MORE_EVIDENCE,
    ReviewGraphNode.HUMAN_HANDOFF: WAITING_HUMAN,
    ReviewGraphNode.FAILED: FAILED,
}


@dataclass(frozen=True, slots=True)
class ReviewRuntimeContext:
    request: ReviewRequest
    correlation: CorrelationContext


def _route_to(next_node: str):
    def route(payload: LangGraphReviewState) -> str:
        state = review_state(payload)
        terminal = _TERMINAL_ROUTE.get(state.node)
        return terminal or next_node

    return route


def build_review_graph(
    nodes: ReviewNodes,
    checkpointer: BaseCheckpointSaver[Any],
) -> CompiledStateGraph:
    builder = StateGraph(
        LangGraphReviewState,
        context_schema=ReviewRuntimeContext,
    )
    node_functions = {
        RETRIEVE_EVIDENCE: nodes.retrieve_evidence,
        VALIDATE_EVIDENCE: nodes.validate_evidence,
        EXTRACT_STRUCTURED_FIELDS: nodes.extract_structured_fields,
        RUN_DETERMINISTIC_TOOL: nodes.run_deterministic_tool,
        BUILD_FINDING: nodes.build_finding,
        VALIDATE_FINDING_EVIDENCE: nodes.validate_finding_evidence,
        PERSIST_FINDING: nodes.persist_finding,
        DONE: nodes.terminal,
        NEED_MORE_EVIDENCE: nodes.terminal,
        WAITING_HUMAN: nodes.wait_for_human,
        FAILED: nodes.terminal,
    }
    for name, node in node_functions.items():
        builder.add_node(name, bind_request(name, node))

    builder.add_edge(START, RETRIEVE_EVIDENCE)
    conditional_steps = (
        (
            RETRIEVE_EVIDENCE,
            VALIDATE_EVIDENCE,
            (NEED_MORE_EVIDENCE, FAILED),
        ),
        (VALIDATE_EVIDENCE, EXTRACT_STRUCTURED_FIELDS, (NEED_MORE_EVIDENCE,)),
        (
            EXTRACT_STRUCTURED_FIELDS,
            RUN_DETERMINISTIC_TOOL,
            (NEED_MORE_EVIDENCE, WAITING_HUMAN, FAILED),
        ),
        (
            RUN_DETERMINISTIC_TOOL,
            BUILD_FINDING,
            (WAITING_HUMAN, FAILED),
        ),
        (BUILD_FINDING, VALIDATE_FINDING_EVIDENCE, ()),
        (
            VALIDATE_FINDING_EVIDENCE,
            PERSIST_FINDING,
            (NEED_MORE_EVIDENCE,),
        ),
    )
    for source, target, terminal_routes in conditional_steps:
        builder.add_conditional_edges(
            source,
            _route_to(target),
            {
                target: target,
                **{terminal: terminal for terminal in terminal_routes},
            },
        )
    builder.add_edge(PERSIST_FINDING, DONE)
    for terminal in TERMINAL_NODES:
        builder.add_edge(terminal, END)
    return builder.compile(checkpointer=checkpointer, name="review")


class LangGraphReviewWorkflow:
    """Checkpointed implementation of the existing ``run(request)`` contract."""

    def __init__(
        self,
        workflow: SingleReviewWorkflow,
        *,
        checkpointer: BaseCheckpointSaver[Any] | None = None,
        finding_persister: FindingPersister | None = None,
    ) -> None:
        self._checkpointer = checkpointer or InMemorySaver()
        self._nodes = ReviewNodes(
            workflow,
            finding_persister=finding_persister,
        )
        self._graph = build_review_graph(self._nodes, self._checkpointer)

    @property
    def compiled_graph(self) -> CompiledStateGraph:
        return self._graph

    def run(
        self,
        request: ReviewRequest,
        *,
        interrupt_after: Sequence[str] = (),
        correlation: CorrelationContext | None = None,
    ) -> ReviewGraphState:
        config = _thread_config(request.review_job_id)
        fingerprint = _request_fingerprint(request)
        snapshot = self._graph.get_state(config)
        context = ReviewRuntimeContext(
            request=request,
            correlation=correlation
            or CorrelationContext(
                job_id=request.review_job_id,
                thread_id=request.review_job_id,
                call_id=request.call.call_id,
                dataset_version=request.provenance.dataset_version_id,
            ),
        )
        if snapshot.values:
            _assert_request_matches(snapshot.values, fingerprint)
            current = review_state(snapshot.values)
            if current.lifecycle is not ReviewLifecycle.RUNNING:
                return current
            output = self._graph.invoke(
                None,
                config,
                context=context,
                interrupt_after=tuple(interrupt_after),
            )
        else:
            initial = ReviewGraphState(
                review_job_id=request.review_job_id,
                rule=request.rule,
                provenance=request.provenance,
            )
            output = self._graph.invoke(
                {
                    "review_state": initial.model_dump(mode="json"),
                    "request_fingerprint": fingerprint,
                    "node_records": [],
                },
                config,
                context=context,
                interrupt_after=tuple(interrupt_after),
            )
        return review_state(output)

    def node_records(self, review_job_id: str) -> tuple[dict[str, object], ...]:
        snapshot = self._graph.get_state(_thread_config(review_job_id))
        records = snapshot.values.get("node_records", []) if snapshot.values else []
        return tuple(dict(record) for record in records)

    def latest_state(self, review_job_id: str) -> ReviewGraphState | None:
        """Read the last durable business state without advancing the graph."""

        snapshot = self._graph.get_state(_thread_config(review_job_id))
        if not snapshot.values:
            return None
        return review_state(snapshot.values)

    def latest_checkpoint(
        self, review_job_id: str
    ) -> ReviewCheckpointPointer | None:
        thread_id = review_thread_id(review_job_id)
        snapshot = self._graph.get_state(_thread_config(review_job_id))
        if not snapshot.values:
            return None
        state = review_state(snapshot.values)
        checkpoint_id = snapshot.config["configurable"].get("checkpoint_id")
        if not checkpoint_id:
            return None
        return ReviewCheckpointPointer(
            review_job_id=review_job_id,
            thread_id=thread_id,
            checkpoint_id=checkpoint_id,
            node=state.node,
            stage=state.stage,
            lifecycle=state.lifecycle,
        )

    def checkpoint_trace(
        self, review_job_id: str
    ) -> tuple[ReviewCheckpointPointer, ...]:
        """Return one latest pointer per durable graph state, oldest first."""

        snapshots = tuple(
            reversed(tuple(self._graph.get_state_history(_thread_config(review_job_id))))
        )
        pointers: dict[ReviewGraphNode, ReviewCheckpointPointer] = {}
        for snapshot in snapshots:
            if not snapshot.values:
                continue
            checkpoint_id = snapshot.config["configurable"].get("checkpoint_id")
            if not checkpoint_id:
                continue
            state = review_state(snapshot.values)
            pointers[state.node] = ReviewCheckpointPointer(
                review_job_id=review_job_id,
                thread_id=review_thread_id(review_job_id),
                checkpoint_id=checkpoint_id,
                node=state.node,
                stage=state.stage,
                lifecycle=state.lifecycle,
            )
        return tuple(pointers.values())


def _thread_config(review_job_id: str) -> dict[str, dict[str, str]]:
    return {"configurable": {"thread_id": review_thread_id(review_job_id)}}


def _request_fingerprint(request: ReviewRequest) -> str:
    payload = request.model_dump(mode="json", exclude={"call"})
    retrieval_result = payload.get("retrieval_result")
    if isinstance(retrieval_result, dict):
        provenance = retrieval_result.get("provenance")
        if isinstance(provenance, dict):
            provenance.pop("latency_ms", None)
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _assert_request_matches(
    payload: dict[str, Any], request_fingerprint: str
) -> None:
    if payload.get("request_fingerprint") != request_fingerprint:
        raise ConflictError(
            "A review thread cannot resume with different input",
            code="review_checkpoint_input_mismatch",
        )
