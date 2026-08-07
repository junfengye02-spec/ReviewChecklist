from __future__ import annotations

import hashlib
import json
from enum import StrEnum
from typing import Self

from pydantic import Field, model_validator

from tender_review.shared.contracts import CallContext, ContractModel


SHA256_PATTERN = r"^[0-9a-f]{64}$"


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def stable_sha256(value: object) -> str:
    """Hash JSON-compatible DTO content with a stable representation."""
    if isinstance(value, ContractModel):
        payload = value.model_dump(mode="json")
    else:
        payload = value
    rendered = json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return sha256_text(rendered)


class ParserDescriptor(ContractModel):
    name: str = Field(min_length=1, max_length=128)
    version: str = Field(min_length=1, max_length=128)
    config_sha256: str = Field(pattern=SHA256_PATTERN)


class StrategyDescriptor(ContractModel):
    name: str = Field(min_length=1, max_length=128)
    version: str = Field(min_length=1, max_length=128)
    config_sha256: str = Field(pattern=SHA256_PATTERN)


class BoundingBox(ContractModel):
    x0: float
    y0: float
    x1: float
    y1: float

    @model_validator(mode="after")
    def has_positive_area(self) -> Self:
        if self.x1 <= self.x0 or self.y1 <= self.y0:
            raise ValueError("bounding box must have positive area")
        return self


class BlockSource(StrEnum):
    PDF_TEXT = "pdf_text"
    OCR = "ocr"


class PageParseStatus(StrEnum):
    OK = "ok"
    EMPTY_TEXT = "empty_text"
    LOW_TEXT = "low_text"
    ANOMALOUS_TEXT = "anomalous_text"
    EXTRACTION_FAILED = "extraction_failed"
    OCR_APPLIED = "ocr_applied"
    OCR_FAILED = "ocr_failed"


class OcrOutcome(StrEnum):
    NOT_REQUESTED = "not_requested"
    SUCCESS = "success"
    EMPTY_RESULT = "empty_result"
    PROVIDER_FAILED = "provider_failed"
    PAGE_RENDER_FAILED = "page_render_failed"


class ParsedBlock(ContractModel):
    block_id: str = Field(min_length=1, max_length=256)
    page_number: int = Field(ge=1)
    reading_order: int = Field(ge=0)
    text: str = Field(min_length=1)
    text_sha256: str = Field(pattern=SHA256_PATTERN)
    bbox: BoundingBox | None = None
    source: BlockSource
    font_name: str | None = Field(default=None, max_length=256)
    font_size: float | None = Field(default=None, gt=0)
    title_candidate: bool = False
    heading_level: int | None = Field(default=None, ge=1, le=6)
    table_id: str | None = Field(default=None, max_length=256)

    @model_validator(mode="after")
    def has_matching_text_hash(self) -> Self:
        if self.text_sha256 != sha256_text(self.text):
            raise ValueError("text_sha256 does not match block text")
        return self


class TableCell(ContractModel):
    row_index: int = Field(ge=0)
    column_index: int = Field(ge=0)
    text: str = ""
    text_sha256: str = Field(pattern=SHA256_PATTERN)
    bbox: BoundingBox | None = None

    @model_validator(mode="after")
    def has_matching_text_hash(self) -> Self:
        if self.text_sha256 != sha256_text(self.text):
            raise ValueError("text_sha256 does not match table cell text")
        return self


class TableRegion(ContractModel):
    table_id: str = Field(min_length=1, max_length=256)
    page_number: int = Field(ge=1)
    bbox: BoundingBox
    cells: tuple[TableCell, ...]
    source_block_ids: tuple[str, ...] = ()
    extraction_method: str = Field(min_length=1, max_length=128)
    text_sha256: str = Field(pattern=SHA256_PATTERN)

    @model_validator(mode="after")
    def has_matching_text_hash(self) -> Self:
        text = "\n".join(cell.text for cell in self.cells)
        if self.text_sha256 != sha256_text(text):
            raise ValueError("text_sha256 does not match table cells")
        return self


class ParseFailure(ContractModel):
    stage: str = Field(min_length=1, max_length=64)
    error_type: str = Field(min_length=1, max_length=128)
    message: str = Field(min_length=1, max_length=2048)
    retryable: bool = False


class PageExtraction(ContractModel):
    page_number: int = Field(ge=1)
    width: float = Field(gt=0)
    height: float = Field(gt=0)
    blocks: tuple[ParsedBlock, ...] = ()
    tables: tuple[TableRegion, ...] = ()
    text_sha256: str = Field(pattern=SHA256_PATTERN)
    text_characters: int = Field(ge=0)
    status: PageParseStatus
    candidate_reason: str | None = Field(default=None, max_length=256)
    extraction_failure: ParseFailure | None = None
    table_detection_attempted: bool = False
    table_warnings: tuple[str, ...] = ()

    @model_validator(mode="after")
    def is_consistent(self) -> Self:
        text = "\n".join(block.text for block in self.blocks)
        if self.text_sha256 != sha256_text(text):
            raise ValueError("text_sha256 does not match page blocks")
        if self.text_characters != sum(not char.isspace() for char in text):
            raise ValueError("text_characters does not match page blocks")
        if any(block.page_number != self.page_number for block in self.blocks):
            raise ValueError("all blocks must belong to the page")
        if any(table.page_number != self.page_number for table in self.tables):
            raise ValueError("all tables must belong to the page")
        return self


class ParsedDocument(ContractModel):
    document_id: str = Field(min_length=1, max_length=256)
    document_sha256: str = Field(pattern=SHA256_PATTERN)
    parser: ParserDescriptor
    pages: tuple[PageExtraction, ...]
    page_count: int = Field(ge=0)
    content_sha256: str = Field(pattern=SHA256_PATTERN)

    @model_validator(mode="after")
    def is_consistent(self) -> Self:
        if self.page_count != len(self.pages):
            raise ValueError("page_count does not match pages")
        expected_pages = tuple(range(1, self.page_count + 1))
        if tuple(page.page_number for page in self.pages) != expected_pages:
            raise ValueError("pages must be ordered and one-based")
        payload = self.model_dump(mode="json", exclude={"content_sha256"})
        if self.content_sha256 != stable_sha256(payload):
            raise ValueError("content_sha256 does not match parsed document")
        return self


class ParseRequest(ContractModel):
    document_id: str = Field(min_length=1, max_length=256)
    pdf_bytes: bytes = Field(min_length=1)
    document_sha256: str = Field(pattern=SHA256_PATTERN)
    call: CallContext

    @model_validator(mode="after")
    def has_matching_document_hash(self) -> Self:
        if self.document_sha256 != sha256_bytes(self.pdf_bytes):
            raise ValueError("document_sha256 does not match pdf_bytes")
        return self


class PageImageRequest(ContractModel):
    document_id: str = Field(min_length=1, max_length=256)
    document_sha256: str = Field(pattern=SHA256_PATTERN)
    page_number: int = Field(ge=1)
    pdf_bytes: bytes = Field(min_length=1)
    dpi: int = Field(default=200, ge=72, le=600)
    call: CallContext

    @model_validator(mode="after")
    def has_matching_document_hash(self) -> Self:
        if self.document_sha256 != sha256_bytes(self.pdf_bytes):
            raise ValueError("document_sha256 does not match pdf_bytes")
        return self


class RenderedPage(ContractModel):
    document_id: str = Field(min_length=1, max_length=256)
    document_sha256: str = Field(pattern=SHA256_PATTERN)
    page_number: int = Field(ge=1)
    image_png: bytes = Field(min_length=1)
    image_sha256: str = Field(pattern=SHA256_PATTERN)
    renderer: ParserDescriptor

    @model_validator(mode="after")
    def has_matching_image_hash(self) -> Self:
        if self.image_sha256 != sha256_bytes(self.image_png):
            raise ValueError("image_sha256 does not match image_png")
        return self


class OcrRequest(ContractModel):
    document_id: str = Field(min_length=1, max_length=256)
    document_sha256: str = Field(pattern=SHA256_PATTERN)
    page_number: int = Field(ge=1)
    image_png: bytes = Field(min_length=1)
    image_sha256: str = Field(pattern=SHA256_PATTERN)
    call: CallContext

    @model_validator(mode="after")
    def has_matching_image_hash(self) -> Self:
        if self.image_sha256 != sha256_bytes(self.image_png):
            raise ValueError("image_sha256 does not match image_png")
        return self


class OcrResult(ContractModel):
    provider: str = Field(min_length=1, max_length=128)
    provider_version: str = Field(min_length=1, max_length=128)
    text: str
    text_sha256: str = Field(pattern=SHA256_PATTERN)
    confidence_lower: float | None = Field(default=None, ge=0, le=1)
    confidence_upper: float | None = Field(default=None, ge=0, le=1)

    @model_validator(mode="after")
    def is_consistent(self) -> Self:
        if self.text_sha256 != sha256_text(self.text):
            raise ValueError("text_sha256 does not match OCR text")
        if self.confidence_lower is None and self.confidence_upper is not None:
            raise ValueError("confidence_upper requires confidence_lower")
        if self.confidence_lower is not None and self.confidence_upper is None:
            raise ValueError("confidence_lower requires confidence_upper")
        if (
            self.confidence_lower is not None
            and self.confidence_upper is not None
            and self.confidence_lower > self.confidence_upper
        ):
            raise ValueError("OCR confidence bounds are invalid")
        return self


class OcrAttempt(ContractModel):
    page_number: int = Field(ge=1)
    candidate_reason: str = Field(min_length=1, max_length=256)
    outcome: OcrOutcome
    provider_called: bool
    provider: str | None = Field(default=None, max_length=128)
    provider_version: str | None = Field(default=None, max_length=128)
    confidence_lower: float | None = Field(default=None, ge=0, le=1)
    confidence_upper: float | None = Field(default=None, ge=0, le=1)
    text_sha256: str | None = Field(default=None, pattern=SHA256_PATTERN)
    failure: ParseFailure | None = None

    @model_validator(mode="after")
    def is_consistent(self) -> Self:
        if self.outcome is OcrOutcome.SUCCESS and self.text_sha256 is None:
            raise ValueError("successful OCR must include text_sha256")
        if self.outcome in {
            OcrOutcome.EMPTY_RESULT,
            OcrOutcome.PROVIDER_FAILED,
            OcrOutcome.PAGE_RENDER_FAILED,
        } and self.failure is None:
            raise ValueError("failed OCR outcome must include a failure record")
        called_outcomes = {
            OcrOutcome.SUCCESS,
            OcrOutcome.EMPTY_RESULT,
            OcrOutcome.PROVIDER_FAILED,
        }
        if self.provider_called != (self.outcome in called_outcomes):
            raise ValueError("provider_called does not match OCR outcome")
        return self


class PageQuality(ContractModel):
    page_number: int = Field(ge=1)
    status: PageParseStatus
    initial_status: PageParseStatus
    text_characters: int = Field(ge=0)
    ocr_candidate: bool
    candidate_reason: str | None = Field(default=None, max_length=256)
    ocr_attempt: OcrAttempt | None = None
    table_warnings: tuple[str, ...] = ()
    extraction_failure: ParseFailure | None = None


class ParseQualityStatistics(ContractModel):
    page_count: int = Field(ge=0)
    text_page_count: int = Field(ge=0)
    ocr_candidate_count: int = Field(ge=0)
    ocr_called_count: int = Field(ge=0)
    ocr_success_count: int = Field(ge=0)
    ocr_failure_count: int = Field(ge=0)
    table_warning_count: int = Field(ge=0)
    extraction_failure_count: int = Field(ge=0)


class ParseQualityReport(ContractModel):
    document_id: str = Field(min_length=1, max_length=256)
    document_sha256: str = Field(pattern=SHA256_PATTERN)
    parser: ParserDescriptor
    pages: tuple[PageQuality, ...]
    statistics: ParseQualityStatistics
    report_sha256: str = Field(pattern=SHA256_PATTERN)

    @model_validator(mode="after")
    def has_matching_report_hash(self) -> Self:
        payload = self.model_dump(mode="json", exclude={"report_sha256"})
        if self.report_sha256 != stable_sha256(payload):
            raise ValueError("report_sha256 does not match quality report")
        return self


class ParseArtifact(ContractModel):
    document: ParsedDocument
    quality_report: ParseQualityReport
    artifact_sha256: str = Field(pattern=SHA256_PATTERN)

    @model_validator(mode="after")
    def is_consistent(self) -> Self:
        if self.document.document_id != self.quality_report.document_id:
            raise ValueError("document and quality report document IDs differ")
        if self.document.document_sha256 != self.quality_report.document_sha256:
            raise ValueError("document and quality report hashes differ")
        payload = self.model_dump(mode="json", exclude={"artifact_sha256"})
        if self.artifact_sha256 != stable_sha256(payload):
            raise ValueError("artifact_sha256 does not match parse artifact")
        return self


class ChunkBlockReference(ContractModel):
    block_id: str = Field(min_length=1, max_length=256)
    page_number: int = Field(ge=1)
    text_start: int = Field(ge=0)
    text_end: int = Field(gt=0)
    bbox: BoundingBox | None = None

    @model_validator(mode="after")
    def has_valid_text_range(self) -> Self:
        if self.text_end <= self.text_start:
            raise ValueError("text range must have positive length")
        return self


class DocumentChunk(ContractModel):
    chunk_id: str = Field(min_length=1, max_length=256)
    document_id: str = Field(min_length=1, max_length=256)
    document_sha256: str = Field(pattern=SHA256_PATTERN)
    parser: ParserDescriptor
    strategy: StrategyDescriptor
    page_start: int = Field(ge=1)
    page_end: int = Field(ge=1)
    section_path: tuple[str, ...] = ()
    raw_text: str = Field(min_length=1)
    text_sha256: str = Field(pattern=SHA256_PATTERN)
    source_blocks: tuple[ChunkBlockReference, ...] = ()
    table_ids: tuple[str, ...] = ()

    @model_validator(mode="after")
    def is_consistent(self) -> Self:
        if self.page_end < self.page_start:
            raise ValueError("page_end must not precede page_start")
        if self.text_sha256 != sha256_text(self.raw_text):
            raise ValueError("text_sha256 does not match chunk text")
        if any(
            reference.page_number < self.page_start
            or reference.page_number > self.page_end
            for reference in self.source_blocks
        ):
            raise ValueError("chunk block reference falls outside page range")
        return self


class ChunkSet(ContractModel):
    document_id: str = Field(min_length=1, max_length=256)
    document_sha256: str = Field(pattern=SHA256_PATTERN)
    parser: ParserDescriptor
    strategy: StrategyDescriptor
    chunks: tuple[DocumentChunk, ...]
    chunk_set_sha256: str = Field(pattern=SHA256_PATTERN)

    @model_validator(mode="after")
    def is_consistent(self) -> Self:
        if any(chunk.document_id != self.document_id for chunk in self.chunks):
            raise ValueError("all chunks must belong to the document")
        payload = self.model_dump(mode="json", exclude={"chunk_set_sha256"})
        if self.chunk_set_sha256 != stable_sha256(payload):
            raise ValueError("chunk_set_sha256 does not match chunk set")
        return self


class EvidenceReference(ContractModel):
    document_id: str = Field(min_length=1, max_length=256)
    document_sha256: str = Field(pattern=SHA256_PATTERN)
    parser: ParserDescriptor
    chunk_id: str = Field(min_length=1, max_length=256)
    source_block_id: str = Field(min_length=1, max_length=256)
    page_number: int = Field(ge=1)
    section_path: tuple[str, ...] = ()
    excerpt: str = Field(min_length=1)
    excerpt_sha256: str = Field(pattern=SHA256_PATTERN)
    source_block_text_sha256: str = Field(pattern=SHA256_PATTERN)
    bbox: BoundingBox | None = None

    @model_validator(mode="after")
    def has_matching_excerpt_hash(self) -> Self:
        if self.excerpt_sha256 != sha256_text(self.excerpt):
            raise ValueError("excerpt_sha256 does not match excerpt")
        return self


def build_parsed_document(
    *,
    document_id: str,
    document_sha256: str,
    parser: ParserDescriptor,
    pages: tuple[PageExtraction, ...],
) -> ParsedDocument:
    payload = {
        "schema_version": 1,
        "document_id": document_id,
        "document_sha256": document_sha256,
        "parser": parser.model_dump(mode="json"),
        "pages": [page.model_dump(mode="json") for page in pages],
        "page_count": len(pages),
    }
    return ParsedDocument(
        **payload,
        content_sha256=stable_sha256(payload),
    )


def build_parse_artifact(
    document: ParsedDocument,
    page_qualities: tuple[PageQuality, ...],
) -> ParseArtifact:
    if tuple(page.page_number for page in page_qualities) != tuple(
        page.page_number for page in document.pages
    ):
        raise ValueError("quality records must match document pages")
    attempts = tuple(
        page.ocr_attempt for page in page_qualities if page.ocr_attempt is not None
    )
    statistics = ParseQualityStatistics(
        page_count=len(document.pages),
        text_page_count=sum(page.text_characters > 0 for page in document.pages),
        ocr_candidate_count=sum(page.ocr_candidate for page in page_qualities),
        ocr_called_count=sum(attempt.provider_called for attempt in attempts),
        ocr_success_count=sum(
            attempt.outcome is OcrOutcome.SUCCESS for attempt in attempts
        ),
        ocr_failure_count=sum(
            attempt.outcome
            in {
                OcrOutcome.EMPTY_RESULT,
                OcrOutcome.PROVIDER_FAILED,
                OcrOutcome.PAGE_RENDER_FAILED,
            }
            for attempt in attempts
        ),
        table_warning_count=sum(
            len(page.table_warnings) for page in page_qualities
        ),
        extraction_failure_count=sum(
            page.extraction_failure is not None for page in page_qualities
        ),
    )
    report_payload = {
        "schema_version": 1,
        "document_id": document.document_id,
        "document_sha256": document.document_sha256,
        "parser": document.parser.model_dump(mode="json"),
        "pages": [page.model_dump(mode="json") for page in page_qualities],
        "statistics": statistics.model_dump(mode="json"),
    }
    report = ParseQualityReport(
        **report_payload,
        report_sha256=stable_sha256(report_payload),
    )
    artifact_payload = {
        "schema_version": 1,
        "document": document.model_dump(mode="json"),
        "quality_report": report.model_dump(mode="json"),
    }
    return ParseArtifact(
        document=document,
        quality_report=report,
        artifact_sha256=stable_sha256(artifact_payload),
    )
