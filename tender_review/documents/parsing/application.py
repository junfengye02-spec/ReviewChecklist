from __future__ import annotations

from collections.abc import Iterable

from tender_review.shared.contracts import ensure_call_active

from .models import (
    BlockSource,
    ChunkSet,
    EvidenceReference,
    OcrAttempt,
    OcrOutcome,
    OcrRequest,
    PageExtraction,
    PageImageRequest,
    PageParseStatus,
    PageQuality,
    ParseArtifact,
    ParseFailure,
    ParseRequest,
    ParsedBlock,
    build_parse_artifact,
    build_parsed_document,
    sha256_text,
)
from .ports import OcrProvider, PdfPageRenderer, StructuredPdfParser


class DocumentParsingService:
    """Apply OCR only to pages that the structural parser marks as candidates."""

    def __init__(
        self,
        *,
        parser: StructuredPdfParser,
        renderer: PdfPageRenderer,
        ocr_provider: OcrProvider,
    ) -> None:
        self._parser = parser
        self._renderer = renderer
        self._ocr_provider = ocr_provider

    def parse(self, request: ParseRequest) -> ParseArtifact:
        ensure_call_active(request.call)
        initial = self._parser.parse(request)
        if initial.document.document_id != request.document_id:
            raise ValueError("parser returned a result for a different document ID")
        if initial.document.document_sha256 != request.document_sha256:
            raise ValueError("parser returned a result for a different document hash")
        pages: list[PageExtraction] = []
        attempts: dict[int, OcrAttempt] = {}
        for page in initial.document.pages:
            ensure_call_active(request.call)
            if page.candidate_reason is None:
                pages.append(page)
                continue
            updated_page, attempt = self._apply_ocr(request, page)
            pages.append(updated_page)
            attempts[page.page_number] = attempt

        document = build_parsed_document(
            document_id=initial.document.document_id,
            document_sha256=initial.document.document_sha256,
            parser=initial.document.parser,
            pages=tuple(pages),
        )
        qualities = tuple(
            PageQuality(
                page_number=page.page_number,
                status=page.status,
                initial_status=initial_page.status,
                text_characters=page.text_characters,
                ocr_candidate=initial_page.candidate_reason is not None,
                candidate_reason=initial_page.candidate_reason,
                ocr_attempt=attempts.get(page.page_number),
                table_warnings=page.table_warnings,
                extraction_failure=page.extraction_failure,
            )
            for page, initial_page in zip(pages, initial.document.pages, strict=True)
        )
        return build_parse_artifact(document, qualities)

    def _apply_ocr(
        self, request: ParseRequest, page: PageExtraction
    ) -> tuple[PageExtraction, OcrAttempt]:
        reason = page.candidate_reason
        if reason is None:
            raise ValueError("OCR was requested for a non-candidate page")
        try:
            rendered = self._renderer.render_page(
                PageImageRequest(
                    document_id=request.document_id,
                    document_sha256=request.document_sha256,
                    page_number=page.page_number,
                    pdf_bytes=request.pdf_bytes,
                    dpi=getattr(self._renderer, "render_dpi", 200),
                    call=request.call,
                )
            )
            if (
                rendered.document_id != request.document_id
                or rendered.document_sha256 != request.document_sha256
                or rendered.page_number != page.page_number
            ):
                raise ValueError("renderer returned an image for another source page")
        except Exception as exc:
            failure = _failure("page_render", exc)
            return self._failed_page(page), OcrAttempt(
                page_number=page.page_number,
                candidate_reason=reason,
                outcome=OcrOutcome.PAGE_RENDER_FAILED,
                provider_called=False,
                provider=_provider_name(self._ocr_provider),
                provider_version=_provider_version(self._ocr_provider),
                failure=failure,
            )

        try:
            result = self._ocr_provider.recognize(
                OcrRequest(
                    document_id=request.document_id,
                    document_sha256=request.document_sha256,
                    page_number=page.page_number,
                    image_png=rendered.image_png,
                    image_sha256=rendered.image_sha256,
                    call=request.call,
                )
            )
        except Exception as exc:
            failure = _failure("ocr", exc)
            return self._failed_page(page), OcrAttempt(
                page_number=page.page_number,
                candidate_reason=reason,
                outcome=OcrOutcome.PROVIDER_FAILED,
                provider_called=True,
                provider=_provider_name(self._ocr_provider),
                provider_version=_provider_version(self._ocr_provider),
                failure=failure,
            )

        if not result.text.strip():
            failure = ParseFailure(
                stage="ocr",
                error_type="empty_ocr_result",
                message="OCR provider returned no usable text",
            )
            return self._failed_page(page), OcrAttempt(
                page_number=page.page_number,
                candidate_reason=reason,
                outcome=OcrOutcome.EMPTY_RESULT,
                provider_called=True,
                provider=result.provider,
                provider_version=result.provider_version,
                confidence_lower=result.confidence_lower,
                confidence_upper=result.confidence_upper,
                failure=failure,
            )

        digest = sha256_text(result.text)
        ocr_block = ParsedBlock(
            block_id=f"p{page.page_number}:ocr:{digest[:16]}",
            page_number=page.page_number,
            reading_order=max((block.reading_order for block in page.blocks), default=-1)
            + 1,
            text=result.text,
            text_sha256=digest,
            source=BlockSource.OCR,
        )
        blocks = (*page.blocks, ocr_block)
        text = "\n".join(block.text for block in blocks)
        updated_page = page.model_copy(
            update={
                "blocks": blocks,
                "text_sha256": sha256_text(text),
                "text_characters": sum(not character.isspace() for character in text),
                "status": PageParseStatus.OCR_APPLIED,
            }
        )
        return updated_page, OcrAttempt(
            page_number=page.page_number,
            candidate_reason=reason,
            outcome=OcrOutcome.SUCCESS,
            provider_called=True,
            provider=result.provider,
            provider_version=result.provider_version,
            confidence_lower=result.confidence_lower,
            confidence_upper=result.confidence_upper,
            text_sha256=digest,
        )

    @staticmethod
    def _failed_page(page: PageExtraction) -> PageExtraction:
        return page.model_copy(update={"status": PageParseStatus.OCR_FAILED})


class EvidenceValidator:
    """Pre-indexed strict validator for one parsed document and chunk set."""

    def __init__(self, *, artifact: ParseArtifact, chunks: ChunkSet) -> None:
        self.artifact = artifact
        self.chunks = chunks
        all_blocks = tuple(
            block for page in artifact.document.pages for block in page.blocks
        )
        self._blocks = {block.block_id: block for block in all_blocks}
        self._chunks = {chunk.chunk_id: chunk for chunk in chunks.chunks}
        if len(self._blocks) != len(all_blocks):
            raise ValueError("source block IDs must be unique within a document")
        if len(self._chunks) != len(chunks.chunks):
            raise ValueError("chunk IDs must be unique within a chunk set")
        self._validate_shared_identity()

    def create(
        self,
        *,
        chunk_id: str,
        source_block_id: str,
        excerpt: str,
    ) -> EvidenceReference:
        block = self._source_block(source_block_id)
        chunk = self._chunk(chunk_id)
        reference = EvidenceReference(
            document_id=self.artifact.document.document_id,
            document_sha256=self.artifact.document.document_sha256,
            parser=self.artifact.document.parser,
            chunk_id=chunk.chunk_id,
            source_block_id=block.block_id,
            page_number=block.page_number,
            section_path=chunk.section_path,
            excerpt=excerpt,
            excerpt_sha256=sha256_text(excerpt),
            source_block_text_sha256=block.text_sha256,
            bbox=block.bbox,
        )
        self.validate(reference)
        return reference

    def validate(self, reference: EvidenceReference) -> None:
        """Reject citations that cannot be located in the parsed source block."""
        document = self.artifact.document
        if reference.document_id != document.document_id:
            raise ValueError("evidence document_id does not match parsed document")
        if reference.document_sha256 != document.document_sha256:
            raise ValueError("evidence document hash does not match parsed document")
        if reference.parser != document.parser:
            raise ValueError("evidence parser metadata does not match parsed document")
        if reference.excerpt_sha256 != sha256_text(reference.excerpt):
            raise ValueError("evidence excerpt hash does not match excerpt")
        block = self._source_block(reference.source_block_id)
        chunk = self._chunk(reference.chunk_id)
        if block.page_number != reference.page_number:
            raise ValueError("evidence page number does not match source block")
        if block.bbox != reference.bbox:
            raise ValueError("evidence bounding box does not match source block")
        if block.text_sha256 != reference.source_block_text_sha256:
            raise ValueError("evidence source block hash does not match")
        if reference.section_path != chunk.section_path:
            raise ValueError("evidence section path does not match chunk")
        if not chunk.page_start <= reference.page_number <= chunk.page_end:
            raise ValueError("evidence page falls outside chunk page range")
        matching_references = tuple(
            item for item in chunk.source_blocks if item.block_id == block.block_id
        )
        if not matching_references:
            raise ValueError("evidence source block does not belong to the chunk")
        if not _excerpt_is_in_references(
            reference.excerpt,
            block.text,
            matching_references,
        ):
            raise ValueError(
                "evidence excerpt is outside the source block range in the chunk"
            )

    def _validate_shared_identity(self) -> None:
        document = self.artifact.document
        if self.chunks.document_id != document.document_id:
            raise ValueError("chunk set belongs to another document")
        if self.chunks.document_sha256 != document.document_sha256:
            raise ValueError("chunk set document hash does not match parsed document")
        if self.chunks.parser != document.parser:
            raise ValueError("chunk set parser metadata does not match parsed document")

    def _source_block(self, block_id: str) -> ParsedBlock:
        try:
            return self._blocks[block_id]
        except KeyError as exc:
            raise ValueError(
                "source block ID must identify exactly one parsed block"
            ) from exc

    def _chunk(self, chunk_id: str):
        try:
            return self._chunks[chunk_id]
        except KeyError as exc:
            raise ValueError("chunk ID must identify exactly one chunk") from exc


def create_evidence_reference(
    *,
    artifact: ParseArtifact,
    chunks: ChunkSet,
    chunk_id: str,
    source_block_id: str,
    excerpt: str,
) -> EvidenceReference:
    return EvidenceValidator(artifact=artifact, chunks=chunks).create(
        chunk_id=chunk_id,
        source_block_id=source_block_id,
        excerpt=excerpt,
    )


def validate_evidence_reference(
    reference: EvidenceReference,
    *,
    artifact: ParseArtifact,
    chunks: ChunkSet,
) -> None:
    EvidenceValidator(artifact=artifact, chunks=chunks).validate(reference)


def _excerpt_is_in_references(
    excerpt: str,
    source_text: str,
    references: Iterable[object],
) -> bool:
    offset = source_text.find(excerpt)
    while offset >= 0:
        end = offset + len(excerpt)
        if any(
            getattr(reference, "text_start") <= offset
            and end <= getattr(reference, "text_end")
            for reference in references
        ):
            return True
        offset = source_text.find(excerpt, offset + 1)
    return False


def _failure(stage: str, exc: Exception) -> ParseFailure:
    return ParseFailure(
        stage=stage,
        error_type=type(exc).__name__,
        message=str(exc)[:2048] or f"{stage} failed without an error message",
    )


def _provider_name(provider: OcrProvider) -> str:
    return str(getattr(provider, "name", type(provider).__name__))


def _provider_version(provider: OcrProvider) -> str:
    return str(getattr(provider, "version", "unknown"))
