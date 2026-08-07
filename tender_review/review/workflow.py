from __future__ import annotations

import hashlib
import json
import re
import time
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from decimal import Decimal, InvalidOperation
from typing import TypeVar

from pydantic import ValidationError

from tender_review.evaluation.public import (
    ProvisionalEvaluationInput,
    ProvisionalVariantRun,
)
from tender_review.findings.public import EvidenceReference, FindingSummary
from tender_review.retrieval.public import (
    Retriever,
    SearchHit,
    SearchRequest,
    SearchResult,
)
from tender_review.shared.contracts import CallContext
from tender_review.shared.errors import ConflictError, ErrorCategory, ServiceError
from tender_review.shared.ids import IdGenerator, UuidGenerator

from .models import (
    ComparisonResult,
    ComparisonToolInput,
    DateExtraction,
    ExternalCallRecord,
    ExtractionSource,
    LlmMessage,
    LlmRequest,
    LlmResponse,
    NumberExtraction,
    NumericRangeExtraction,
    ReviewFailure,
    ReviewGraphNode,
    ReviewGraphState,
    ReviewInputProvenance,
    ReviewLifecycle,
    ReviewProcessingStage,
    ReviewRequest,
    SetExtraction,
    StructuredExtraction,
    TextExtraction,
    ToolRequest,
    ToolResult,
)
from .ports import LlmProvider, ReviewTool
from .tools import default_review_tools


T = TypeVar("T")
R = TypeVar("R")


_LEGAL_TRANSITIONS: dict[ReviewGraphNode, frozenset[ReviewGraphNode]] = {
    ReviewGraphNode.INPUT: frozenset({ReviewGraphNode.RETRIEVAL}),
    ReviewGraphNode.RETRIEVAL: frozenset(
        {
            ReviewGraphNode.EVIDENCE_VALIDATION,
            ReviewGraphNode.NEED_MORE_EVIDENCE,
            ReviewGraphNode.FAILED,
        }
    ),
    ReviewGraphNode.EVIDENCE_VALIDATION: frozenset(
        {ReviewGraphNode.EXTRACTION, ReviewGraphNode.NEED_MORE_EVIDENCE}
    ),
    ReviewGraphNode.EXTRACTION: frozenset(
        {
            ReviewGraphNode.COMPARISON,
            ReviewGraphNode.NEED_MORE_EVIDENCE,
            ReviewGraphNode.HUMAN_HANDOFF,
            ReviewGraphNode.FAILED,
        }
    ),
    ReviewGraphNode.COMPARISON: frozenset(
        {
            ReviewGraphNode.CONCLUSION,
            ReviewGraphNode.HUMAN_HANDOFF,
            ReviewGraphNode.FAILED,
        }
    ),
    ReviewGraphNode.CONCLUSION: frozenset({ReviewGraphNode.EVIDENCE_INTEGRITY}),
    ReviewGraphNode.EVIDENCE_INTEGRITY: frozenset(
        {
            ReviewGraphNode.DONE,
            ReviewGraphNode.NEED_MORE_EVIDENCE,
            ReviewGraphNode.HUMAN_HANDOFF,
        }
    ),
    ReviewGraphNode.DONE: frozenset(),
    ReviewGraphNode.NEED_MORE_EVIDENCE: frozenset(),
    ReviewGraphNode.HUMAN_HANDOFF: frozenset(),
    ReviewGraphNode.FAILED: frozenset(),
}

_NODE_STAGE: dict[ReviewGraphNode, ReviewProcessingStage] = {
    ReviewGraphNode.RETRIEVAL: ReviewProcessingStage.RETRIEVING,
    ReviewGraphNode.EVIDENCE_VALIDATION: ReviewProcessingStage.VERIFYING_EVIDENCE,
    ReviewGraphNode.EXTRACTION: ReviewProcessingStage.EXTRACTING,
    ReviewGraphNode.COMPARISON: ReviewProcessingStage.COMPARING,
    ReviewGraphNode.CONCLUSION: ReviewProcessingStage.REPORTING,
    ReviewGraphNode.EVIDENCE_INTEGRITY: ReviewProcessingStage.REPORTING,
}

_TERMINAL_LIFECYCLE: dict[ReviewGraphNode, ReviewLifecycle] = {
    ReviewGraphNode.DONE: ReviewLifecycle.COMPLETED,
    ReviewGraphNode.NEED_MORE_EVIDENCE: ReviewLifecycle.NEED_MORE_EVIDENCE,
    ReviewGraphNode.HUMAN_HANDOFF: ReviewLifecycle.WAITING_HUMAN,
    ReviewGraphNode.FAILED: ReviewLifecycle.FAILED,
}


class _InvocationFailure(Exception):
    def __init__(
        self,
        *,
        code: str,
        message: str,
        category: ErrorCategory,
        retryable: bool,
        call_id: str,
        records: tuple[ExternalCallRecord, ...],
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.category = category
        self.retryable = retryable
        self.call_id = call_id
        self.records = records


def transition_review_state(
    state: ReviewGraphState,
    target: ReviewGraphNode,
    **updates: object,
) -> ReviewGraphState:
    """Apply one declared graph edge while keeping stage and lifecycle separate."""

    if target not in _LEGAL_TRANSITIONS[state.node]:
        raise ConflictError(
            f"Cannot transition review graph from {state.node.value} to {target.value}",
            code="invalid_review_graph_transition",
            details={"from_node": state.node.value, "to_node": target.value},
        )
    reserved_updates = {
        "review_job_id",
        "rule",
        "provenance",
        "node",
        "stage",
        "lifecycle",
        "visited_nodes",
    }
    attempted = sorted(reserved_updates & updates.keys())
    if attempted:
        raise ConflictError(
            "Graph transitions cannot rewrite immutable control or input fields",
            code="immutable_review_state_field",
            details={"fields": attempted},
        )
    payload = state.model_dump()
    payload.update(updates)
    payload["node"] = target
    payload["visited_nodes"] = (*state.visited_nodes, target)
    if target in _NODE_STAGE:
        payload["stage"] = _NODE_STAGE[target]
        payload["lifecycle"] = ReviewLifecycle.RUNNING
    else:
        payload["lifecycle"] = _TERMINAL_LIFECYCLE[target]
    return ReviewGraphState.model_validate(payload)


def provisional_review_provenance(
    evaluation_input: ProvisionalEvaluationInput,
    variant_run: ProvisionalVariantRun,
) -> ReviewInputProvenance:
    """Build Stage 5 provenance without upgrading Stage 4 navigation hints."""

    if variant_run.input_sha256 != evaluation_input.input_sha256:
        raise ValueError("provisional input and variant run hashes do not match")
    if variant_run.dataset_version_id != evaluation_input.dataset_version_id:
        raise ValueError("provisional input and variant run datasets do not match")
    if (
        variant_run.source_work_package_sha256
        != evaluation_input.source_work_package_sha256
        or variant_run.chunk_catalog_sha256 != evaluation_input.chunk_catalog_sha256
    ):
        raise ValueError("provisional input and variant run sources do not match")
    return ReviewInputProvenance(
        source_kind="provisional_retrieval",
        status=evaluation_input.status,
        claims_allowed=evaluation_input.claims_allowed,
        dataset_version_id=evaluation_input.dataset_version_id,
        input_sha256=evaluation_input.input_sha256,
        results_sha256=variant_run.results_sha256,
        variant=variant_run.variant,
    )


class SingleReviewWorkflow:
    """Explicit replaceable state graph for one deterministic review item."""

    def __init__(
        self,
        llm: LlmProvider,
        *,
        retriever: Retriever | None = None,
        tools: Iterable[ReviewTool] | None = None,
        id_generator: IdGenerator | None = None,
    ) -> None:
        self._llm = llm
        self._retriever = retriever
        configured_tools = tuple(tools) if tools is not None else default_review_tools()
        self._tools = {tool.name: tool for tool in configured_tools}
        if len(self._tools) != len(configured_tools):
            raise ValueError("review tool names must be unique")
        self._ids = id_generator or UuidGenerator()

    def run(self, request: ReviewRequest) -> ReviewGraphState:
        state = ReviewGraphState(
            review_job_id=request.review_job_id,
            rule=request.rule,
            provenance=request.provenance,
        )
        state = transition_review_state(state, ReviewGraphNode.RETRIEVAL)
        retrieval_result, state = self._retrieve(request, state)
        if retrieval_result is None:
            return state

        state = transition_review_state(
            state,
            ReviewGraphNode.EVIDENCE_VALIDATION,
            retrieval_result=retrieval_result,
        )
        eligible_hits = tuple(
            hit
            for hit in retrieval_result.hits
            if hit.page_start is not None
            and hit.page_end is not None
            and bool(hit.section_path)
        )
        if not eligible_hits:
            return self._need_more_evidence(
                state,
                code="no_locatable_evidence",
                message=(
                    "Retrieval returned no evidence with chunk, page, and section "
                    "provenance."
                ),
            )

        state = transition_review_state(
            state,
            ReviewGraphNode.EXTRACTION,
            eligible_hits=eligible_hits,
        )
        extraction, state = self._extract(request, state)
        if extraction is None:
            return state
        try:
            validated_evidence = _resolve_extraction_evidence(extraction, eligible_hits)
            _validate_extraction_grounding(extraction, eligible_hits)
        except ValueError as exc:
            return self._need_more_evidence(
                state,
                code="extraction_evidence_invalid",
                message=str(exc),
                extraction=extraction,
            )

        state = transition_review_state(
            state,
            ReviewGraphNode.COMPARISON,
            extraction=extraction,
            validated_evidence=validated_evidence,
        )
        comparison, state = self._compare(request, state, extraction)
        if comparison is None:
            return state

        state = transition_review_state(
            state,
            ReviewGraphNode.CONCLUSION,
            comparison=comparison,
        )
        finding_evidence = _resolve_sources(comparison.sources, eligible_hits)
        finding = FindingSummary(
            finding_id=self._ids.new(),
            review_job_id=request.review_job_id,
            conclusion="compliant" if comparison.passed else "noncompliant",
            message=comparison.message,
            evidence=finding_evidence,
        )
        state = transition_review_state(
            state,
            ReviewGraphNode.EVIDENCE_INTEGRITY,
            finding=finding,
        )
        try:
            _validate_finding_evidence(finding, eligible_hits)
        except ValueError as exc:
            return self._need_more_evidence(
                state,
                code="finding_evidence_invalid",
                message=str(exc),
            )
        return transition_review_state(state, ReviewGraphNode.DONE)

    def _retrieve(
        self, request: ReviewRequest, state: ReviewGraphState
    ) -> tuple[SearchResult | None, ReviewGraphState]:
        if request.retrieval_result is not None:
            return request.retrieval_result, state
        if self._retriever is None:
            return None, self._failed(
                state,
                code="retriever_not_configured",
                message="A retriever is required when no retrieval result is supplied.",
                category=ErrorCategory.PERMANENT,
                retryable=False,
            )
        try:
            result, records = _invoke_with_retries(
                operation="retrieve",
                root_call=request.call,
                invoke=lambda call: self._retriever.search(
                    SearchRequest(
                        query=request.query,
                        document_ids=request.document_ids,
                        limit=10,
                        call=call,
                    )
                ),
                validate=SearchResult.model_validate,
                invalid_output_retryable=False,
            )
        except _InvocationFailure as exc:
            state = _append_records(state, exc.records)
            if exc.retryable:
                return None, self._need_more_evidence(
                    state,
                    code=exc.code,
                    message=exc.message,
                    call_failure=exc,
                )
            return None, self._failed_from_invocation(state, exc)
        return result, _append_records(state, records)

    def _extract(
        self, request: ReviewRequest, state: ReviewGraphState
    ) -> tuple[StructuredExtraction | None, ReviewGraphState]:
        llm_request_payload = _extraction_messages(request, state.eligible_hits)

        def invoke(call: CallContext) -> LlmResponse:
            return self._llm.complete(
                LlmRequest(
                    messages=llm_request_payload,
                    response_schema_name="StructuredExtraction.v1",
                    temperature=0.0,
                    call=call,
                )
            )

        def validate(response: LlmResponse) -> StructuredExtraction:
            validated_response = LlmResponse.model_validate(response)
            extraction = StructuredExtraction.model_validate_json(
                validated_response.content
            )
            if extraction.review_item_id != request.rule.review_item_id:
                raise ValueError("extraction review_item_id does not match the rule")
            matching_fields = tuple(
                field
                for field in extraction.fields
                if field.field_name == request.rule.field_name
            )
            if len(matching_fields) != 1:
                raise ValueError("extraction is missing the rule field")
            return extraction

        try:
            extraction, records = _invoke_with_retries(
                operation="extract",
                root_call=request.call,
                invoke=invoke,
                validate=validate,
                invalid_output_retryable=True,
            )
        except _InvocationFailure as exc:
            state = _append_records(state, exc.records)
            if exc.category is ErrorCategory.CANCELLED:
                return None, self._failed_from_invocation(state, exc)
            return None, self._human_handoff(state, exc)
        return extraction, _append_records(state, records)

    def _compare(
        self,
        request: ReviewRequest,
        state: ReviewGraphState,
        extraction: StructuredExtraction,
    ) -> tuple[ComparisonResult | None, ReviewGraphState]:
        tool = self._tools.get(request.rule.tool_name)
        if tool is None:
            failure = _InvocationFailure(
                code="review_tool_not_configured",
                message=f"Review tool {request.rule.tool_name!r} is not configured.",
                category=ErrorCategory.PERMANENT,
                retryable=False,
                call_id=request.call.call_id,
                records=(),
            )
            return None, self._human_handoff(state, failure)
        input_value = ComparisonToolInput(rule=request.rule, extraction=extraction)

        def invoke(call: CallContext) -> ToolResult:
            return tool.execute(
                ToolRequest(
                    tool_name=tool.name,
                    input_json=input_value.model_dump_json(),
                    call=call,
                )
            )

        def validate(raw_result: ToolResult) -> ComparisonResult:
            tool_result = ToolResult.model_validate(raw_result)
            if tool_result.tool_name != tool.name:
                raise ValueError("tool result name does not match the requested tool")
            comparison = ComparisonResult.model_validate_json(tool_result.output_json)
            if comparison.tool_name != tool.name:
                raise ValueError(
                    "comparison tool_name does not match the requested tool"
                )
            if comparison.tool_version != tool_result.tool_version:
                raise ValueError("comparison and tool result versions do not match")
            if comparison.review_item_id != request.rule.review_item_id:
                raise ValueError("comparison review_item_id does not match the rule")
            relevant_field = next(
                field
                for field in extraction.fields
                if field.field_name == request.rule.field_name
            )
            if comparison.sources != relevant_field.sources:
                raise ValueError("comparison sources must match the compared field")
            return comparison

        try:
            comparison, records = _invoke_with_retries(
                operation=f"tool_{tool.name}",
                root_call=request.call,
                invoke=invoke,
                validate=validate,
                invalid_output_retryable=False,
            )
        except _InvocationFailure as exc:
            state = _append_records(state, exc.records)
            if exc.category is ErrorCategory.CANCELLED:
                return None, self._failed_from_invocation(state, exc)
            return None, self._human_handoff(state, exc)
        return comparison, _append_records(state, records)

    @staticmethod
    def _need_more_evidence(
        state: ReviewGraphState,
        *,
        code: str,
        message: str,
        extraction: StructuredExtraction | None = None,
        call_failure: _InvocationFailure | None = None,
    ) -> ReviewGraphState:
        failure = ReviewFailure(
            code=code,
            message=message,
            category=ErrorCategory.INSUFFICIENT_EVIDENCE,
            retryable=call_failure.retryable if call_failure else False,
            call_id=call_failure.call_id if call_failure else None,
        )
        updates: dict[str, object] = {
            "reason": message,
            "failure": failure,
            "finding": None,
        }
        if extraction is not None:
            updates["extraction"] = extraction
        return transition_review_state(
            state,
            ReviewGraphNode.NEED_MORE_EVIDENCE,
            **updates,
        )

    @staticmethod
    def _human_handoff(
        state: ReviewGraphState, failure: _InvocationFailure
    ) -> ReviewGraphState:
        return transition_review_state(
            state,
            ReviewGraphNode.HUMAN_HANDOFF,
            reason=failure.message,
            failure=_review_failure(failure),
        )

    @staticmethod
    def _failed_from_invocation(
        state: ReviewGraphState, failure: _InvocationFailure
    ) -> ReviewGraphState:
        return transition_review_state(
            state,
            ReviewGraphNode.FAILED,
            reason=failure.message,
            failure=_review_failure(failure),
        )

    @staticmethod
    def _failed(
        state: ReviewGraphState,
        *,
        code: str,
        message: str,
        category: ErrorCategory,
        retryable: bool,
    ) -> ReviewGraphState:
        return transition_review_state(
            state,
            ReviewGraphNode.FAILED,
            reason=message,
            failure=ReviewFailure(
                code=code,
                message=message,
                category=category,
                retryable=retryable,
            ),
        )


def _extraction_messages(
    request: ReviewRequest, hits: tuple[SearchHit, ...]
) -> tuple[LlmMessage, ...]:
    evidence = [hit.model_dump(mode="json") for hit in hits]
    rule = request.rule.model_dump(mode="json")
    return (
        LlmMessage(
            role="system",
            content=(
                "Return only JSON matching StructuredExtraction.v1. Extract facts, "
                "not conclusions. Every field needs exact source_id, document_id, "
                "chunk_id, page_number, section_path, and an excerpt copied from the "
                "provided evidence."
            ),
        ),
        LlmMessage(
            role="user",
            content=json.dumps(
                {"query": request.query, "rule": rule, "evidence": evidence},
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ),
        ),
    )


def _resolve_extraction_evidence(
    extraction: StructuredExtraction,
    hits: tuple[SearchHit, ...],
) -> tuple[EvidenceReference, ...]:
    return _resolve_sources(
        tuple(source for field in extraction.fields for source in field.sources),
        hits,
    )


def _validate_extraction_grounding(
    extraction: StructuredExtraction,
    hits: tuple[SearchHit, ...],
) -> None:
    hit_by_chunk = {hit.chunk_id: hit for hit in hits}
    for field in extraction.fields:
        excerpts = "\n".join(source.excerpt for source in field.sources)
        if isinstance(field, DateExtraction):
            value = field.value
            forms = {
                value.isoformat(),
                value.strftime("%Y/%m/%d"),
                f"{value.year}年{value.month}月{value.day}日",
                f"{value.year}年{value.month:02d}月{value.day:02d}日",
            }
            if not any(form in excerpts for form in forms):
                raise ValueError(
                    f"date field {field.field_name!r} is not grounded in its excerpts"
                )
        elif isinstance(field, SetExtraction):
            folded = excerpts.casefold()
            missing = tuple(
                value for value in field.values if value.casefold() not in folded
            )
            if missing:
                raise ValueError(
                    f"set field {field.field_name!r} has ungrounded values: {missing}"
                )
        elif isinstance(field, NumberExtraction):
            _require_grounded_numbers(field.field_name, excerpts, (field.value,))
            _require_grounded_unit(field.field_name, excerpts, field.unit)
        elif isinstance(field, NumericRangeExtraction):
            _require_grounded_numbers(
                field.field_name,
                excerpts,
                (field.minimum, field.maximum),
            )
            _require_grounded_unit(field.field_name, excerpts, field.unit)
        elif isinstance(field, TextExtraction):
            source_hits = tuple(
                hit_by_chunk[source.chunk_id] for source in field.sources
            )
            if not any(field.value in hit.text for hit in source_hits):
                raise ValueError(
                    f"text field {field.field_name!r} is not grounded in its chunks"
                )


def _require_grounded_numbers(
    field_name: str,
    excerpts: str,
    expected_values: tuple[float, ...],
) -> None:
    parsed: set[Decimal] = set()
    for token in re.findall(r"(?<![\w.])[-+]?\d[\d,]*(?:\.\d+)?", excerpts):
        try:
            parsed.add(Decimal(token.replace(",", "")))
        except InvalidOperation:
            continue
    missing = tuple(
        value for value in expected_values if Decimal(str(value)) not in parsed
    )
    if missing:
        raise ValueError(
            f"numeric field {field_name!r} has ungrounded values: {missing}"
        )


def _require_grounded_unit(
    field_name: str,
    excerpts: str,
    unit: str | None,
) -> None:
    if unit is not None and unit.casefold() not in excerpts.casefold():
        raise ValueError(f"numeric field {field_name!r} has an ungrounded unit")


def _resolve_sources(
    sources: tuple[ExtractionSource, ...],
    hits: tuple[SearchHit, ...],
) -> tuple[EvidenceReference, ...]:
    hit_by_chunk = {hit.chunk_id: hit for hit in hits}
    references: list[EvidenceReference] = []
    identities: set[tuple[str, str, int, str]] = set()
    for source in sources:
        hit = hit_by_chunk.get(source.chunk_id)
        if hit is None:
            raise ValueError(f"source {source.source_id!r} references an unknown chunk")
        if hit.document_id != source.document_id:
            raise ValueError(f"source {source.source_id!r} has the wrong document")
        if hit.page_start is None or hit.page_end is None:
            raise ValueError(f"source {source.source_id!r} has no page provenance")
        if not hit.page_start <= source.page_number <= hit.page_end:
            raise ValueError(f"source {source.source_id!r} page is outside its chunk")
        if source.section_path != hit.section_path:
            raise ValueError(f"source {source.source_id!r} has the wrong section")
        if source.excerpt not in hit.text:
            raise ValueError(f"source {source.source_id!r} excerpt is not in its chunk")
        identity = (
            source.document_id,
            source.chunk_id,
            source.page_number,
            source.excerpt,
        )
        if identity in identities:
            continue
        identities.add(identity)
        references.append(
            EvidenceReference(
                document_id=source.document_id,
                chunk_id=source.chunk_id,
                page_number=source.page_number,
                section_path=source.section_path,
                excerpt=source.excerpt,
                text_sha256=hashlib.sha256(source.excerpt.encode("utf-8")).hexdigest(),
            )
        )
    if not references:
        raise ValueError("a conclusion needs at least one valid evidence reference")
    return tuple(references)


def _validate_finding_evidence(
    finding: FindingSummary, hits: tuple[SearchHit, ...]
) -> None:
    if finding.conclusion not in {"compliant", "noncompliant"}:
        raise ValueError("workflow findings must have a deterministic conclusion")
    if not finding.evidence:
        raise ValueError("a deterministic conclusion needs evidence")
    hit_by_chunk = {hit.chunk_id: hit for hit in hits}
    for reference in finding.evidence:
        expected_hash = hashlib.sha256(reference.excerpt.encode("utf-8")).hexdigest()
        if reference.text_sha256 != expected_hash:
            raise ValueError("finding evidence text hash does not match its excerpt")
        hit = hit_by_chunk.get(reference.chunk_id)
        if hit is None or hit.document_id != reference.document_id:
            raise ValueError("finding evidence does not resolve to a retrieved chunk")
        if hit.page_start is None or hit.page_end is None:
            raise ValueError("finding evidence has no page provenance")
        if not hit.page_start <= reference.page_number <= hit.page_end:
            raise ValueError("finding evidence page is outside its chunk")
        if reference.section_path != hit.section_path:
            raise ValueError("finding evidence section does not match its chunk")
        if reference.excerpt not in hit.text:
            raise ValueError("finding evidence excerpt is not in its chunk")


def _append_records(
    state: ReviewGraphState, records: tuple[ExternalCallRecord, ...]
) -> ReviewGraphState:
    if not records:
        return state
    payload = state.model_dump()
    payload["call_records"] = (*state.call_records, *records)
    return ReviewGraphState.model_validate(payload)


def _review_failure(failure: _InvocationFailure) -> ReviewFailure:
    return ReviewFailure(
        code=failure.code,
        message=failure.message,
        category=failure.category,
        retryable=failure.retryable,
        call_id=failure.call_id,
    )


def _invoke_with_retries(
    *,
    operation: str,
    root_call: CallContext,
    invoke: Callable[[CallContext], T],
    validate: Callable[[T], R],
    invalid_output_retryable: bool,
) -> tuple[R, tuple[ExternalCallRecord, ...]]:
    records: list[ExternalCallRecord] = []
    for attempt in range(1, root_call.max_attempts + 1):
        call = CallContext(
            call_id=_child_call_id(root_call.call_id, operation, attempt),
            timeout_seconds=root_call.timeout_seconds,
            max_attempts=1,
            cancelled=root_call.cancelled,
        )
        started = time.perf_counter()
        try:
            raw_value = _execute_with_timeout(
                lambda: invoke(call), timeout_seconds=call.timeout_seconds
            )
        except FutureTimeoutError:
            record = _call_record(
                operation=operation,
                call=call,
                attempt=attempt,
                started=started,
                outcome="timeout",
                retryable=True,
                error_code="external_call_timeout",
            )
            records.append(record)
            if attempt < root_call.max_attempts:
                continue
            raise _InvocationFailure(
                code="external_call_timeout",
                message=f"External operation {operation!r} exceeded its timeout.",
                category=ErrorCategory.RETRYABLE,
                retryable=True,
                call_id=call.call_id,
                records=tuple(records),
            ) from None
        except ServiceError as exc:
            retryable = exc.retryable
            records.append(
                _call_record(
                    operation=operation,
                    call=call,
                    attempt=attempt,
                    started=started,
                    outcome="retryable_error" if retryable else "permanent_error",
                    retryable=retryable,
                    error_code=exc.code,
                )
            )
            if retryable and attempt < root_call.max_attempts:
                continue
            raise _InvocationFailure(
                code=exc.code,
                message=exc.message,
                category=exc.category,
                retryable=retryable,
                call_id=call.call_id,
                records=tuple(records),
            ) from exc
        except Exception as exc:
            records.append(
                _call_record(
                    operation=operation,
                    call=call,
                    attempt=attempt,
                    started=started,
                    outcome="permanent_error",
                    retryable=False,
                    error_code="external_call_failed",
                )
            )
            raise _InvocationFailure(
                code="external_call_failed",
                message=f"External operation {operation!r} failed: {exc}",
                category=ErrorCategory.PERMANENT,
                retryable=False,
                call_id=call.call_id,
                records=tuple(records),
            ) from exc

        try:
            validated = validate(raw_value)
        except (ValidationError, ValueError, TypeError) as exc:
            records.append(
                _call_record(
                    operation=operation,
                    call=call,
                    attempt=attempt,
                    started=started,
                    outcome="invalid_output",
                    retryable=invalid_output_retryable,
                    error_code="invalid_external_output",
                )
            )
            if invalid_output_retryable and attempt < root_call.max_attempts:
                continue
            raise _InvocationFailure(
                code="invalid_external_output",
                message=f"External operation {operation!r} returned invalid output: {exc}",
                category=(
                    ErrorCategory.RETRYABLE
                    if invalid_output_retryable
                    else ErrorCategory.PERMANENT
                ),
                retryable=invalid_output_retryable,
                call_id=call.call_id,
                records=tuple(records),
            ) from exc
        records.append(
            _call_record(
                operation=operation,
                call=call,
                attempt=attempt,
                started=started,
                outcome="success",
                retryable=False,
                error_code=None,
            )
        )
        return validated, tuple(records)
    raise AssertionError("max_attempts validation guarantees at least one attempt")


def _execute_with_timeout(operation: Callable[[], T], *, timeout_seconds: float) -> T:
    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="review-call")
    future = executor.submit(operation)
    try:
        value = future.result(timeout=timeout_seconds)
    except BaseException:
        future.cancel()
        executor.shutdown(wait=False, cancel_futures=True)
        raise
    executor.shutdown(wait=True)
    return value


def _child_call_id(root_call_id: str, operation: str, attempt: int) -> str:
    suffix = f":{operation}:{attempt}"
    prefix = root_call_id[: 128 - len(suffix)]
    return f"{prefix}{suffix}"


def _call_record(
    *,
    operation: str,
    call: CallContext,
    attempt: int,
    started: float,
    outcome: str,
    retryable: bool,
    error_code: str | None,
) -> ExternalCallRecord:
    return ExternalCallRecord(
        operation=operation,
        call_id=call.call_id,
        attempt=attempt,
        timeout_seconds=call.timeout_seconds,
        outcome=outcome,
        retryable=retryable,
        error_code=error_code,
        duration_ms=max(0.0, (time.perf_counter() - started) * 1000.0),
    )
