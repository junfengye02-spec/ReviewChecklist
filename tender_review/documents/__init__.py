"""Document ingestion ports and deterministic offline adapters."""

from .fakes import (
    FakeChunkingStrategy,
    FakeDocumentParser,
    FakeOcrProvider,
    InMemoryArtifactStore,
)
from .ports import ArtifactStore, ChunkingStrategy, DocumentParser, OcrProvider

__all__ = [
    "ArtifactStore",
    "ChunkingStrategy",
    "DocumentParser",
    "FakeChunkingStrategy",
    "FakeDocumentParser",
    "FakeOcrProvider",
    "InMemoryArtifactStore",
    "OcrProvider",
]
"""Document snapshots, immutable artifacts, and PDF evidence modeling."""
