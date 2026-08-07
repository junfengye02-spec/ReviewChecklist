from __future__ import annotations

from typing import Protocol, runtime_checkable

from .models import (
    ChunkSet,
    OcrRequest,
    OcrResult,
    PageImageRequest,
    ParseArtifact,
    ParseRequest,
    RenderedPage,
    ParserDescriptor,
    StrategyDescriptor,
)


@runtime_checkable
class StructuredPdfParser(Protocol):
    descriptor: ParserDescriptor

    def parse(self, request: ParseRequest) -> ParseArtifact: ...


@runtime_checkable
class PdfPageRenderer(Protocol):
    descriptor: ParserDescriptor

    def render_page(self, request: PageImageRequest) -> RenderedPage: ...


@runtime_checkable
class OcrProvider(Protocol):
    name: str
    version: str

    def recognize(self, request: OcrRequest) -> OcrResult: ...


@runtime_checkable
class ChunkingStrategy(Protocol):
    descriptor: StrategyDescriptor

    def chunk(self, artifact: ParseArtifact) -> ChunkSet: ...
