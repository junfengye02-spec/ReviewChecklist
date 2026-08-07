from __future__ import annotations

from typing import Protocol, runtime_checkable

from tender_review.shared.contracts import CallContext

from .models import (
    ArtifactReference,
    ArtifactWrite,
    ChunkDocumentRequest,
    ChunkDocumentResult,
    OcrRequest,
    OcrResult,
    ParseDocumentRequest,
    ParseDocumentResult,
)


@runtime_checkable
class ArtifactStore(Protocol):
    def put(
        self, artifact: ArtifactWrite, *, call: CallContext | None = None
    ) -> ArtifactReference: ...

    def get(self, key: str, *, call: CallContext | None = None) -> bytes: ...

    def exists(self, key: str, *, call: CallContext | None = None) -> bool: ...

    def delete(self, key: str, *, call: CallContext | None = None) -> None: ...


@runtime_checkable
class DocumentParser(Protocol):
    def parse(self, request: ParseDocumentRequest) -> ParseDocumentResult: ...


@runtime_checkable
class OcrProvider(Protocol):
    def recognize(self, request: OcrRequest) -> OcrResult: ...


@runtime_checkable
class ChunkingStrategy(Protocol):
    def chunk(self, request: ChunkDocumentRequest) -> ChunkDocumentResult: ...
