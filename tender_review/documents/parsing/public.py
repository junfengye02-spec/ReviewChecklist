"""Stable DTO and service entry points for the PDF evidence layer."""

from .application import (
    DocumentParsingService,
    EvidenceValidator,
    create_evidence_reference,
    validate_evidence_reference,
)
from .chunking import StructuralChunker
from .models import (
    ChunkSet,
    DocumentChunk,
    EvidenceReference,
    ParseArtifact,
    ParseQualityReport,
    ParseRequest,
)

__all__ = [
    "ChunkSet",
    "DocumentChunk",
    "DocumentParsingService",
    "EvidenceReference",
    "EvidenceValidator",
    "ParseArtifact",
    "ParseQualityReport",
    "ParseRequest",
    "StructuralChunker",
    "create_evidence_reference",
    "validate_evidence_reference",
]
