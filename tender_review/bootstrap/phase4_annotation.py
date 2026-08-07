"""Composition-root assembly for the offline Stage 4 annotation export."""

from tender_review.documents.parsing.adapters.pymupdf import PyMuPDFStructuredParser
from tender_review.documents.parsing.application import DocumentParsingService
from tender_review.documents.parsing.chunking import StructuralChunker
from tender_review.documents.parsing.fakes import UnavailableOcrProvider
from tender_review.documents.parsing.models import ParseRequest, sha256_bytes
from tender_review.evaluation.public import CandidateChunk, RebuiltChunkDocument
from tender_review.shared.contracts import CallContext


def rebuild_chunk_document(
    document_id: str,
    pdf_bytes: bytes,
    call: CallContext,
) -> RebuiltChunkDocument:
    """Run the concrete Phase 3 parser/chunker behind the composition boundary."""

    parser = PyMuPDFStructuredParser()
    artifact = DocumentParsingService(
        parser=parser,
        renderer=parser,
        ocr_provider=UnavailableOcrProvider(),
    ).parse(
        ParseRequest(
            document_id=document_id,
            pdf_bytes=pdf_bytes,
            document_sha256=sha256_bytes(pdf_bytes),
            call=call,
        )
    )
    chunk_set = StructuralChunker().chunk(artifact)
    return RebuiltChunkDocument(
        document_id=document_id,
        document_sha256=artifact.document.document_sha256,
        parse_artifact_sha256=artifact.artifact_sha256,
        chunk_set_sha256=chunk_set.chunk_set_sha256,
        chunks=tuple(
            CandidateChunk(
                chunk_id=chunk.chunk_id,
                document_id=chunk.document_id,
                document_sha256=chunk.document_sha256,
                page_start=chunk.page_start,
                page_end=chunk.page_end,
                section_path=chunk.section_path,
                text=chunk.raw_text,
                text_sha256=chunk.text_sha256,
            )
            for chunk in chunk_set.chunks
        ),
    )
