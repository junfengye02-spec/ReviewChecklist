from __future__ import annotations

import logging
import time
from collections.abc import Callable
from typing import Any, Protocol, TypedDict

from langgraph.types import interrupt

from tender_review.findings.public import FindingSummary
from tender_review.shared.errors import ConflictError

from tender_review.shared.observability import log_event, record_metric

from .models import ReviewGraphNode, ReviewGraphState, ReviewRequest
from .workflow import (
    SingleReviewWorkflow,
    _resolve_extraction_evidence,
    _resolve_sources,
    _validate_extraction_grounding,
    _validate_finding_evidence,
    transition_review_state,
)


class LangGraphReviewState(TypedDict):
    review_state: dict[str, Any]
    request_fingerprint: str
    node_records: list[dict[str, object]]


class FindingPersister(Protocol):
    """Idempotent A2 boundary invoked only after evidence validation succeeds."""

    def __call__(self, state: ReviewGraphState) -> None: ...


class ReviewNodes:
    """Thin LangGraph nodes over the existing single-review business logic."""

    def __init__(
        self,
        workflow: SingleReviewWorkflow,
        *,
        finding_persister: FindingPersister | None = None,
    ) -> None:
        self._workflow = workflow
        self._finding_persister = finding_persister

    def retrieve_evidence(
        self, payload: LangGraphReviewState, request: ReviewRequest
    ) -> dict[str, object]:
        state = _state_at(payload, ReviewGraphNode.INPUT)
        state = transition_review_state(state, ReviewGraphNode.RETRIEVAL)
        result, state = self._workflow._retrieve(request, state)
        if result is not None:
            state = state.model_copy(update={"retrieval_result": result})
        return _state_update(state)

    def validate_evidence(
        self, payload: LangGraphReviewState, request: ReviewRequest
    ) -> dict[str, object]:
        del request
        state = _state_at(payload, ReviewGraphNode.RETRIEVAL)
        if state.retrieval_result is None:
            raise ConflictError(
                "Retrieval completed without a result or terminal state",
                code="review_retrieval_result_missing",
            )
        state = transition_review_state(
            state,
            ReviewGraphNode.EVIDENCE_VALIDATION,
            retrieval_result=state.retrieval_result,
        )
        eligible_hits = tuple(
            hit
            for hit in state.retrieval_result.hits
            if hit.page_start is not None
            and hit.page_end is not None
            and bool(hit.section_path)
        )
        if not eligible_hits:
            state = self._workflow._need_more_evidence(
                state,
                code="no_locatable_evidence",
                message=(
                    "Retrieval returned no evidence with chunk, page, and section "
                    "provenance."
                ),
            )
        else:
            state = transition_review_state(
                state,
                ReviewGraphNode.EXTRACTION,
                eligible_hits=eligible_hits,
            )
        return _state_update(state)

    def extract_structured_fields(
        self, payload: LangGraphReviewState, request: ReviewRequest
    ) -> dict[str, object]:
        state = _state_at(payload, ReviewGraphNode.EXTRACTION)
        extraction, state = self._workflow._extract(request, state)
        if extraction is None:
            return _state_update(state)
        try:
            evidence = _resolve_extraction_evidence(
                extraction, state.eligible_hits
            )
            _validate_extraction_grounding(extraction, state.eligible_hits)
        except ValueError as exc:
            state = self._workflow._need_more_evidence(
                state,
                code="extraction_evidence_invalid",
                message=str(exc),
                extraction=extraction,
            )
        else:
            state = transition_review_state(
                state,
                ReviewGraphNode.COMPARISON,
                extraction=extraction,
                validated_evidence=evidence,
            )
        return _state_update(state)

    def run_deterministic_tool(
        self, payload: LangGraphReviewState, request: ReviewRequest
    ) -> dict[str, object]:
        state = _state_at(payload, ReviewGraphNode.COMPARISON)
        if state.extraction is None:
            raise ConflictError(
                "Comparison requires a structured extraction",
                code="review_extraction_missing",
            )
        comparison, state = self._workflow._compare(
            request, state, state.extraction
        )
        if comparison is not None:
            state = transition_review_state(
                state,
                ReviewGraphNode.CONCLUSION,
                comparison=comparison,
            )
        return _state_update(state)

    def build_finding(
        self, payload: LangGraphReviewState, request: ReviewRequest
    ) -> dict[str, object]:
        state = _state_at(payload, ReviewGraphNode.CONCLUSION)
        if state.comparison is None:
            raise ConflictError(
                "Finding construction requires a comparison",
                code="review_comparison_missing",
            )
        finding = FindingSummary(
            finding_id=self._workflow._ids.new(),
            review_job_id=request.review_job_id,
            conclusion=(
                "compliant" if state.comparison.passed else "noncompliant"
            ),
            message=state.comparison.message,
            evidence=_resolve_sources(
                state.comparison.sources, state.eligible_hits
            ),
        )
        state = transition_review_state(
            state,
            ReviewGraphNode.EVIDENCE_INTEGRITY,
            finding=finding,
        )
        return _state_update(state)

    def validate_finding_evidence(
        self, payload: LangGraphReviewState, request: ReviewRequest
    ) -> dict[str, object]:
        del request
        state = _state_at(payload, ReviewGraphNode.EVIDENCE_INTEGRITY)
        if state.finding is None:
            raise ConflictError(
                "Evidence validation requires a finding",
                code="review_finding_missing",
            )
        try:
            _validate_finding_evidence(state.finding, state.eligible_hits)
        except ValueError as exc:
            state = self._workflow._need_more_evidence(
                state,
                code="finding_evidence_invalid",
                message=str(exc),
            )
        return _state_update(state)

    def persist_finding(
        self, payload: LangGraphReviewState, request: ReviewRequest
    ) -> dict[str, object]:
        del request
        state = _state_at(payload, ReviewGraphNode.EVIDENCE_INTEGRITY)
        if self._finding_persister is not None:
            self._finding_persister(state)
        state = transition_review_state(state, ReviewGraphNode.DONE)
        return _state_update(state)

    @staticmethod
    def terminal(
        payload: LangGraphReviewState, request: ReviewRequest
    ) -> dict[str, object]:
        del payload, request
        return {}

    @staticmethod
    def wait_for_human(
        payload: LangGraphReviewState, request: ReviewRequest
    ) -> dict[str, object]:
        del request
        state = _state_at(payload, ReviewGraphNode.HUMAN_HANDOFF)
        interrupt(
            {
                "kind": "review_waiting_human",
                "review_job_id": state.review_job_id,
                "failure_code": state.failure.code if state.failure else None,
            }
        )
        return {}


def bind_request(
    node_name: str,
    node: Callable[
        [LangGraphReviewState, ReviewRequest], dict[str, object]
    ],
) -> Callable[[LangGraphReviewState, Any], dict[str, object]]:
    def invoke(payload: LangGraphReviewState, runtime: Any) -> dict[str, object]:
        request = runtime.context.request
        context = runtime.context.correlation
        logger = logging.getLogger("tender_review.review.graph")
        started = time.perf_counter()
        log_event(
            logger,
            logging.INFO,
            event="review.node_started",
            message="Review graph node started",
            context=context,
            node_name=node_name,
        )
        try:
            update = node(payload, request)
        except Exception as exc:
            duration_ms = max(0.0, (time.perf_counter() - started) * 1000.0)
            log_event(
                logger,
                logging.WARNING,
                event="review.node_failed",
                message="Review graph node failed",
                context=context,
                node_name=node_name,
                error_type=type(exc).__name__,
                error_code=getattr(exc, "code", None),
            )
            record_metric(
                logger,
                name="review_node_duration",
                value=duration_ms,
                unit="ms",
                source="process_monotonic",
                context=context,
                node_name=node_name,
                outcome="failed",
            )
            raise
        duration_ms = max(0.0, (time.perf_counter() - started) * 1000.0)
        merged = dict(payload)
        merged.update(update)
        state = review_state(merged)
        update = {
            **update,
            "node_records": [
                *payload.get("node_records", []),
                {
                    "schema_version": 1,
                    "node_name": node_name,
                    "duration_ms": duration_ms,
                    "metric_source": "process_monotonic",
                },
            ],
        }
        log_event(
            logger,
            logging.INFO,
            event="review.node_completed",
            message="Review graph node completed",
            context=context,
            node_name=node_name,
            outcome_node=state.node.value,
        )
        record_metric(
            logger,
            name="review_node_duration",
            value=duration_ms,
            unit="ms",
            source="process_monotonic",
            context=context,
            node_name=node_name,
            outcome="completed",
        )
        return update

    invoke.__name__ = node.__name__
    return invoke


def review_state(payload: LangGraphReviewState) -> ReviewGraphState:
    return ReviewGraphState.model_validate(payload["review_state"])


def _state_at(
    payload: LangGraphReviewState, expected: ReviewGraphNode
) -> ReviewGraphState:
    state = review_state(payload)
    if state.node is not expected:
        raise ConflictError(
            f"Review node expected {expected.value}, got {state.node.value}",
            code="review_graph_node_mismatch",
            details={"expected": expected.value, "actual": state.node.value},
        )
    return state


def _state_update(state: ReviewGraphState) -> dict[str, object]:
    return {"review_state": state.model_dump(mode="json")}
