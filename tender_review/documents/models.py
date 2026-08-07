from __future__ import annotations

from pydantic import Field

from tender_review.shared.contracts import CallContext, ContractModel


class ArtifactWrite(ContractModel):
    key: str = Field(min_length=1, max_length=1024)
    content: bytes
    media_type: str = "application/octet-stream"


class ArtifactReference(ContractModel):
    key: str
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    size_bytes: int = Field(ge=0)
    media_type: str


class ParseDocumentRequest(ContractModel):
    document_id: str
    content: bytes
    media_type: str = "application/pdf"
    call: CallContext


class ParsedPage(ContractModel):
    page_number: int = Field(ge=1)
    text: str


class ParseDocumentResult(ContractModel):
    document_id: str
    parser_name: str
    parser_version: str
    pages: tuple[ParsedPage, ...]


class OcrRequest(ContractModel):
    document_id: str
    page_number: int = Field(ge=1)
    image: bytes
    call: CallContext


class OcrResult(ContractModel):
    text: str
    provider: str


class ChunkDocumentRequest(ContractModel):
    document_id: str
    pages: tuple[ParsedPage, ...]


class DocumentChunk(ContractModel):
    chunk_id: str
    document_id: str
    page_start: int = Field(ge=1)
    page_end: int = Field(ge=1)
    section_path: tuple[str, ...] = ()
    raw_text: str
    text_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class ChunkDocumentResult(ContractModel):
    strategy_name: str
    strategy_version: str
    chunks: tuple[DocumentChunk, ...]
