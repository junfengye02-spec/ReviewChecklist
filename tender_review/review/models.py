from __future__ import annotations

from datetime import date
from enum import Enum
from typing import Annotated, Literal, Self, TypeAlias

from pydantic import Field, StrictInt, StrictStr, field_validator, model_validator

from tender_review.findings.public import EvidenceReference, FindingSummary
from tender_review.retrieval.public import SearchHit, SearchResult
from tender_review.shared.contracts import CallContext, ContractModel
from tender_review.shared.errors import ErrorCategory


SHA256_PATTERN = r"^[0-9a-f]{64}$"
MAX_ABSOLUTE_NUMBER = 1_000_000_000_000.0
StrictFiniteNumber = Annotated[
    float,
    Field(
        strict=True,
        allow_inf_nan=False,
        ge=-MAX_ABSOLUTE_NUMBER,
        le=MAX_ABSOLUTE_NUMBER,
    ),
]


def _require_non_blank(value: str) -> str:
    if not value.strip():
        raise ValueError("value must not be blank")
    return value


def _validate_unique_non_blank(
    values: tuple[str, ...], *, name: str
) -> tuple[str, ...]:
    if any(not value.strip() for value in values):
        raise ValueError(f"{name} must not contain blank values")
    if len(values) != len(set(values)):
        raise ValueError(f"{name} must contain unique values")
    return values


class LlmMessage(ContractModel):
    role: Literal["system", "user", "assistant"]
    content: str


class LlmRequest(ContractModel):
    messages: tuple[LlmMessage, ...]
    response_schema_name: str | None = None
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    call: CallContext


class LlmResponse(ContractModel):
    model: str
    content: str
    finish_reason: str = "stop"
    prompt_tokens: int = Field(default=0, ge=0)
    completion_tokens: int = Field(default=0, ge=0)


class ToolRequest(ContractModel):
    tool_name: str = Field(min_length=1)
    input_json: str
    call: CallContext


class ToolResult(ContractModel):
    tool_name: str
    tool_version: str
    output_json: str


class ExtractionSource(ContractModel):
    """A model-proposed citation that must resolve against retrieved evidence."""

    schema_version: Literal[1] = 1
    source_id: StrictStr = Field(min_length=1, max_length=128)
    document_id: StrictStr = Field(min_length=1, max_length=256)
    chunk_id: StrictStr = Field(min_length=1, max_length=256)
    page_number: StrictInt = Field(ge=1)
    section_path: tuple[StrictStr, ...] = Field(min_length=1, max_length=32)
    excerpt: StrictStr = Field(min_length=1, max_length=4000)

    @field_validator("source_id", "document_id", "chunk_id", "excerpt")
    @classmethod
    def values_are_not_blank(cls, value: str) -> str:
        return _require_non_blank(value)

    @field_validator("section_path")
    @classmethod
    def sections_are_not_blank(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        if any(not value.strip() for value in values):
            raise ValueError("section_path must not contain blank values")
        return values


class ExtractedFieldBase(ContractModel):
    schema_version: Literal[1] = 1
    field_name: StrictStr = Field(min_length=1, max_length=128)
    sources: tuple[ExtractionSource, ...] = Field(min_length=1, max_length=20)

    @field_validator("field_name")
    @classmethod
    def field_name_is_not_blank(cls, value: str) -> str:
        return _require_non_blank(value)

    @field_validator("sources")
    @classmethod
    def source_ids_are_unique(
        cls, values: tuple[ExtractionSource, ...]
    ) -> tuple[ExtractionSource, ...]:
        source_ids = tuple(value.source_id for value in values)
        if len(source_ids) != len(set(source_ids)):
            raise ValueError("extraction source_id values must be unique")
        return values


class DateExtraction(ExtractedFieldBase):
    value_type: Literal["date"] = "date"
    value: date


class SetExtraction(ExtractedFieldBase):
    value_type: Literal["set"] = "set"
    values: tuple[StrictStr, ...] = Field(min_length=1, max_length=100)

    @field_validator("values")
    @classmethod
    def values_are_valid(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        return _validate_unique_non_blank(values, name="set values")


class NumberExtraction(ExtractedFieldBase):
    value_type: Literal["number"] = "number"
    value: StrictFiniteNumber
    unit: StrictStr | None = Field(default=None, min_length=1, max_length=32)

    @field_validator("unit")
    @classmethod
    def unit_is_not_blank(cls, value: str | None) -> str | None:
        return None if value is None else _require_non_blank(value)


class NumericRangeExtraction(ExtractedFieldBase):
    value_type: Literal["numeric_range"] = "numeric_range"
    minimum: StrictFiniteNumber
    maximum: StrictFiniteNumber
    unit: StrictStr | None = Field(default=None, min_length=1, max_length=32)

    @field_validator("unit")
    @classmethod
    def unit_is_not_blank(cls, value: str | None) -> str | None:
        return None if value is None else _require_non_blank(value)

    @model_validator(mode="after")
    def range_is_ordered(self) -> Self:
        if self.maximum < self.minimum:
            raise ValueError("numeric range maximum must not be less than minimum")
        return self


class TextExtraction(ExtractedFieldBase):
    value_type: Literal["text"] = "text"
    value: StrictStr = Field(min_length=1, max_length=20000)

    @field_validator("value")
    @classmethod
    def value_is_not_blank(cls, value: str) -> str:
        return _require_non_blank(value)


ExtractedField: TypeAlias = Annotated[
    DateExtraction
    | SetExtraction
    | NumberExtraction
    | NumericRangeExtraction
    | TextExtraction,
    Field(discriminator="value_type"),
]


class StructuredExtraction(ContractModel):
    """The only accepted LLM extraction shape; free-form text is never a finding."""

    schema_version: Literal[1] = 1
    review_item_id: StrictStr = Field(min_length=1, max_length=128)
    fields: tuple[ExtractedField, ...] = Field(min_length=1, max_length=64)

    @field_validator("review_item_id")
    @classmethod
    def review_item_is_not_blank(cls, value: str) -> str:
        return _require_non_blank(value)

    @model_validator(mode="after")
    def field_names_are_unique(self) -> Self:
        names = tuple(field.field_name for field in self.fields)
        if len(names) != len(set(names)):
            raise ValueError("structured extraction field_name values must be unique")
        return self


class ReviewRuleBase(ContractModel):
    schema_version: Literal[1] = 1
    review_item_id: str = Field(min_length=1, max_length=128)
    field_name: str = Field(min_length=1, max_length=128)

    @field_validator("review_item_id", "field_name")
    @classmethod
    def identifiers_are_not_blank(cls, value: str) -> str:
        return _require_non_blank(value)


class DateRule(ReviewRuleBase):
    tool_name: Literal["date_compare"] = "date_compare"
    operator: Literal["before", "on_or_before", "equal", "on_or_after", "after"]
    expected: date


class SetRule(ReviewRuleBase):
    tool_name: Literal["set_compare"] = "set_compare"
    mode: Literal["contains_all", "exact"] = "contains_all"
    expected_values: tuple[str, ...] = Field(min_length=1, max_length=100)

    @field_validator("expected_values")
    @classmethod
    def expected_values_are_valid(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        return _validate_unique_non_blank(values, name="expected_values")


class NumericRangeRule(ReviewRuleBase):
    tool_name: Literal["numeric_range"] = "numeric_range"
    expected_minimum: StrictFiniteNumber | None = None
    expected_maximum: StrictFiniteNumber | None = None
    unit: str | None = Field(default=None, min_length=1, max_length=32)

    @field_validator("unit")
    @classmethod
    def unit_is_not_blank(cls, value: str | None) -> str | None:
        return None if value is None else _require_non_blank(value)

    @model_validator(mode="after")
    def expected_range_is_valid(self) -> Self:
        if self.expected_minimum is None and self.expected_maximum is None:
            raise ValueError("numeric rule needs at least one expected bound")
        if (
            self.expected_minimum is not None
            and self.expected_maximum is not None
            and self.expected_maximum < self.expected_minimum
        ):
            raise ValueError("expected maximum must not be less than expected minimum")
        return self


class TextPresenceRule(ReviewRuleBase):
    tool_name: Literal["text_presence"] = "text_presence"
    required_terms: tuple[str, ...] = Field(min_length=1, max_length=100)
    mode: Literal["all", "any"] = "all"
    case_sensitive: bool = False

    @field_validator("required_terms")
    @classmethod
    def required_terms_are_valid(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        return _validate_unique_non_blank(values, name="required_terms")


ReviewRule: TypeAlias = Annotated[
    DateRule | SetRule | NumericRangeRule | TextPresenceRule,
    Field(discriminator="tool_name"),
]


class ComparisonToolInput(ContractModel):
    schema_version: Literal[1] = 1
    rule: ReviewRule
    extraction: StructuredExtraction


class ComparisonResult(ContractModel):
    schema_version: Literal[1] = 1
    review_item_id: str = Field(min_length=1, max_length=128)
    tool_name: str = Field(min_length=1, max_length=128)
    tool_version: str = Field(min_length=1, max_length=64)
    passed: bool
    code: str = Field(min_length=1, max_length=128)
    message: str = Field(min_length=1, max_length=4000)
    sources: tuple[ExtractionSource, ...] = Field(min_length=1, max_length=20)

    @model_validator(mode="after")
    def source_ids_are_unique(self) -> Self:
        source_ids = tuple(source.source_id for source in self.sources)
        if len(source_ids) != len(set(source_ids)):
            raise ValueError("comparison source_id values must be unique")
        return self


class ReviewInputProvenance(ContractModel):
    """Provenance travels unchanged through the graph and into its final state."""

    schema_version: Literal[1] = 1
    source_kind: Literal["provisional_retrieval", "verified_retrieval"]
    status: Literal["provisional", "verified"]
    claims_allowed: bool
    dataset_version_id: str = Field(min_length=1, max_length=128)
    input_sha256: str = Field(pattern=SHA256_PATTERN)
    results_sha256: str = Field(pattern=SHA256_PATTERN)
    variant: str = Field(min_length=1, max_length=128)

    @model_validator(mode="after")
    def source_claims_are_consistent(self) -> Self:
        if self.source_kind == "provisional_retrieval":
            if self.status != "provisional" or self.claims_allowed:
                raise ValueError(
                    "provisional retrieval must retain status=provisional and "
                    "claims_allowed=false"
                )
        elif self.status != "verified":
            raise ValueError("verified retrieval must have status=verified")
        return self


class ReviewRequest(ContractModel):
    schema_version: Literal[1] = 1
    review_job_id: str = Field(min_length=1, max_length=128)
    query: str = Field(min_length=1, max_length=4000)
    document_ids: tuple[str, ...] = Field(min_length=1, max_length=100)
    rule: ReviewRule
    provenance: ReviewInputProvenance
    call: CallContext
    retrieval_result: SearchResult | None = None

    @field_validator("review_job_id", "query")
    @classmethod
    def values_are_not_blank(cls, value: str) -> str:
        return _require_non_blank(value)

    @field_validator("document_ids")
    @classmethod
    def document_ids_are_valid(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        return _validate_unique_non_blank(values, name="document_ids")

    @model_validator(mode="after")
    def retrieval_result_is_scoped(self) -> Self:
        if self.retrieval_result is not None:
            scope = set(self.document_ids)
            if any(hit.document_id not in scope for hit in self.retrieval_result.hits):
                raise ValueError(
                    "retrieval result contains a document outside request scope"
                )
        return self


class ReviewLifecycle(str, Enum):
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    NEED_MORE_EVIDENCE = "NEED_MORE_EVIDENCE"
    WAITING_HUMAN = "WAITING_HUMAN"
    FAILED = "FAILED"


class ReviewProcessingStage(str, Enum):
    RETRIEVING = "RETRIEVING"
    VERIFYING_EVIDENCE = "VERIFYING_EVIDENCE"
    EXTRACTING = "EXTRACTING"
    COMPARING = "COMPARING"
    REPORTING = "REPORTING"


class ReviewGraphNode(str, Enum):
    INPUT = "INPUT"
    RETRIEVAL = "RETRIEVAL"
    EVIDENCE_VALIDATION = "EVIDENCE_VALIDATION"
    EXTRACTION = "EXTRACTION"
    COMPARISON = "COMPARISON"
    CONCLUSION = "CONCLUSION"
    EVIDENCE_INTEGRITY = "EVIDENCE_INTEGRITY"
    DONE = "DONE"
    NEED_MORE_EVIDENCE = "NEED_MORE_EVIDENCE"
    HUMAN_HANDOFF = "HUMAN_HANDOFF"
    FAILED = "FAILED"


class ExternalCallRecord(ContractModel):
    schema_version: Literal[1] = 1
    operation: str = Field(min_length=1, max_length=128)
    call_id: str = Field(min_length=1, max_length=128)
    attempt: int = Field(ge=1)
    timeout_seconds: float = Field(gt=0)
    outcome: Literal[
        "success", "timeout", "retryable_error", "permanent_error", "invalid_output"
    ]
    retryable: bool
    error_code: str | None = Field(default=None, min_length=1, max_length=128)
    duration_ms: float = Field(ge=0)


class ReviewFailure(ContractModel):
    schema_version: Literal[1] = 1
    code: str = Field(min_length=1, max_length=128)
    message: str = Field(min_length=1, max_length=4000)
    category: ErrorCategory
    retryable: bool
    call_id: str | None = Field(default=None, min_length=1, max_length=128)


class ReviewGraphState(ContractModel):
    """Immutable graph data; control node, processing stage, and lifecycle differ."""

    schema_version: Literal[1] = 1
    review_job_id: str = Field(min_length=1, max_length=128)
    rule: ReviewRule
    provenance: ReviewInputProvenance
    node: ReviewGraphNode = ReviewGraphNode.INPUT
    stage: ReviewProcessingStage | None = None
    lifecycle: ReviewLifecycle = ReviewLifecycle.RUNNING
    visited_nodes: tuple[ReviewGraphNode, ...] = (ReviewGraphNode.INPUT,)
    retrieval_result: SearchResult | None = None
    eligible_hits: tuple[SearchHit, ...] = ()
    extraction: StructuredExtraction | None = None
    comparison: ComparisonResult | None = None
    validated_evidence: tuple[EvidenceReference, ...] = ()
    finding: FindingSummary | None = None
    call_records: tuple[ExternalCallRecord, ...] = ()
    failure: ReviewFailure | None = None
    reason: str | None = Field(default=None, min_length=1, max_length=4000)

    @model_validator(mode="after")
    def terminal_state_is_consistent(self) -> Self:
        expected = {
            ReviewGraphNode.DONE: ReviewLifecycle.COMPLETED,
            ReviewGraphNode.NEED_MORE_EVIDENCE: ReviewLifecycle.NEED_MORE_EVIDENCE,
            ReviewGraphNode.HUMAN_HANDOFF: ReviewLifecycle.WAITING_HUMAN,
            ReviewGraphNode.FAILED: ReviewLifecycle.FAILED,
        }
        terminal_lifecycle = expected.get(self.node)
        if terminal_lifecycle is not None and self.lifecycle is not terminal_lifecycle:
            raise ValueError("terminal graph node and lifecycle do not match")
        if terminal_lifecycle is None and self.lifecycle is not ReviewLifecycle.RUNNING:
            raise ValueError("non-terminal graph nodes must have RUNNING lifecycle")
        if self.lifecycle is ReviewLifecycle.COMPLETED:
            if self.finding is None or not self.finding.evidence:
                raise ValueError("completed review needs a finding with evidence")
            if self.finding.conclusion == "needs_more_evidence":
                raise ValueError("completed review cannot claim insufficient evidence")
        elif terminal_lifecycle is not None and self.finding is not None:
            raise ValueError("non-completed terminal states must not expose a finding")
        if self.failure is not None and self.node not in {
            ReviewGraphNode.FAILED,
            ReviewGraphNode.HUMAN_HANDOFF,
            ReviewGraphNode.NEED_MORE_EVIDENCE,
        }:
            raise ValueError("failure details belong only to a terminal branch")
        if not self.visited_nodes or self.visited_nodes[-1] is not self.node:
            raise ValueError("visited_nodes must end at the current graph node")
        return self
