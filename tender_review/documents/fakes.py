from __future__ import annotations

import hashlib
from threading import Lock

from tender_review.shared.contracts import CallContext, ensure_call_active
from tender_review.shared.errors import NotFoundError

from .models import (
    ArtifactReference,
    ArtifactWrite,
    ChunkDocumentRequest,
    ChunkDocumentResult,
    DocumentChunk,
    OcrRequest,
    OcrResult,
    ParseDocumentRequest,
    ParseDocumentResult,
    ParsedPage,
)


class InMemoryArtifactStore:
    def __init__(self) -> None:
        self._content: dict[str, bytes] = {}
        self._media_types: dict[str, str] = {}
        self._lock = Lock()

    @staticmethod
    def _key_for(content: bytes) -> str:
        digest = hashlib.sha256(content).hexdigest()
        return f"sha256/{digest[:2]}/{digest}"

    def put(
        self, artifact: ArtifactWrite, *, call: CallContext | None = None
    ) -> ArtifactReference:
        if call is not None:
            ensure_call_active(call)
        content = bytes(artifact.content)
        key = self._key_for(content)
        with self._lock:
            self._content.setdefault(key, content)
            self._media_types.setdefault(key, artifact.media_type)
        return ArtifactReference(
            key=key,
            sha256=hashlib.sha256(content).hexdigest(),
            size_bytes=len(content),
            media_type=self._media_types[key],
            schema_version=artifact.schema_version,
        )

    def get(self, key: str, *, call: CallContext | None = None) -> bytes:
        if call is not None:
            ensure_call_active(call)
        with self._lock:
            try:
                return self._content[key]
            except KeyError as exc:
                raise NotFoundError(
                    f"Artifact {key!r} does not exist", code="artifact_not_found"
                ) from exc

    def exists(self, key: str, *, call: CallContext | None = None) -> bool:
        if call is not None:
            ensure_call_active(call)
        with self._lock:
            return key in self._content

    def delete(self, key: str, *, call: CallContext | None = None) -> None:
        if call is not None:
            ensure_call_active(call)
        with self._lock:
            self._content.pop(key, None)
            self._media_types.pop(key, None)


class FakeDocumentParser:
    name = "fake-document-parser"
    version = "1"

    def __init__(self, result: ParseDocumentResult | None = None) -> None:
        self.result = result
        self.calls: list[ParseDocumentRequest] = []

    def parse(self, request: ParseDocumentRequest) -> ParseDocumentResult:
        ensure_call_active(request.call)
        self.calls.append(request)
        if self.result is not None:
            return self.result
        text = request.content.decode("utf-8", errors="replace")
        return ParseDocumentResult(
            document_id=request.document_id,
            parser_name=self.name,
            parser_version=self.version,
            pages=(ParsedPage(page_number=1, text=text),),
        )


class FakeOcrProvider:
    def __init__(self, text: str = "") -> None:
        self.text = text
        self.calls: list[OcrRequest] = []

    def recognize(self, request: OcrRequest) -> OcrResult:
        ensure_call_active(request.call)
        self.calls.append(request)
        return OcrResult(text=self.text, provider="fake-ocr")


class FakeChunkingStrategy:
    name = "fake-page-chunker"
    version = "1"

    def __init__(self) -> None:
        self.calls: list[ChunkDocumentRequest] = []

    def chunk(self, request: ChunkDocumentRequest) -> ChunkDocumentResult:
        self.calls.append(request)
        chunks = tuple(
            DocumentChunk(
                chunk_id=f"{request.document_id}:page:{page.page_number}",
                document_id=request.document_id,
                page_start=page.page_number,
                page_end=page.page_number,
                raw_text=page.text,
                text_sha256=hashlib.sha256(page.text.encode("utf-8")).hexdigest(),
            )
            for page in request.pages
        )
        return ChunkDocumentResult(
            strategy_name=self.name,
            strategy_version=self.version,
            chunks=chunks,
        )
