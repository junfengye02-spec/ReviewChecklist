from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Mapping

from tender_review.shared.errors import PermanentError

from tender_review.documents.storage import ContentAddressedObject


class ArtifactType(str, Enum):
    PARSED_JSON = "parsed_json"
    TABLES = "tables"
    INDEX = "index"
    REPORT = "report"


_TABLE_MEDIA_TYPES = frozenset(
    {
        "application/json",
        "text/csv",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    }
)
_ARTIFACT_MEDIA_TYPES: dict[ArtifactType, frozenset[str]] = {
    ArtifactType.PARSED_JSON: frozenset({"application/json"}),
    ArtifactType.TABLES: _TABLE_MEDIA_TYPES,
    ArtifactType.INDEX: frozenset({"application/json"}),
    ArtifactType.REPORT: frozenset(
        {"application/json", "application/pdf", "text/markdown"}
    ),
}


class ArtifactValidationError(PermanentError):
    default_code = "artifact_validation_failed"


def _require_nonempty(value: str, field_name: str, *, limit: int) -> str:
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{field_name} must not be empty")
    if len(normalized) > limit:
        raise ValueError(f"{field_name} must be at most {limit} characters")
    return normalized


@dataclass(frozen=True, slots=True)
class SourceDocument:
    source_system: str
    source_document_id: str
    file_name: str
    content: bytes
    media_type: str = "application/pdf"
    schema_version: str = "1"

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "source_system",
            _require_nonempty(self.source_system, "source_system", limit=64),
        )
        object.__setattr__(
            self,
            "source_document_id",
            _require_nonempty(
                self.source_document_id, "source_document_id", limit=255
            ),
        )
        object.__setattr__(
            self, "file_name", _require_nonempty(self.file_name, "file_name", limit=512)
        )
        object.__setattr__(
            self, "media_type", _require_nonempty(self.media_type, "media_type", limit=255)
        )
        object.__setattr__(
            self,
            "schema_version",
            _require_nonempty(self.schema_version, "schema_version", limit=16),
        )
        if not isinstance(self.content, bytes):
            raise TypeError("content must be bytes")


@dataclass(frozen=True, slots=True)
class SnapshotRecord:
    id: str
    source_system: str
    source_document_id: str
    file_name: str
    object: ContentAddressedObject
    parse_status: str
    parser_name: str | None
    parser_version: str | None


@dataclass(frozen=True, slots=True)
class SnapshotSaveResult:
    snapshot: SnapshotRecord
    created: bool


@dataclass(frozen=True, slots=True)
class ArtifactSubmission:
    document_snapshot_id: str
    artifact_type: ArtifactType
    content: bytes
    media_type: str
    schema_version: str
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "document_snapshot_id",
            _require_nonempty(
                self.document_snapshot_id, "document_snapshot_id", limit=36
            ),
        )
        if not isinstance(self.artifact_type, ArtifactType):
            object.__setattr__(self, "artifact_type", ArtifactType(self.artifact_type))
        if not isinstance(self.content, bytes):
            raise TypeError("content must be bytes")
        object.__setattr__(
            self, "media_type", _require_nonempty(self.media_type, "media_type", limit=255)
        )
        object.__setattr__(
            self,
            "schema_version",
            _require_nonempty(self.schema_version, "schema_version", limit=16),
        )
        object.__setattr__(self, "metadata", dict(self.metadata))
        self.validate_content_contract()

    def validate_content_contract(self) -> None:
        allowed_media_types = _ARTIFACT_MEDIA_TYPES[self.artifact_type]
        if self.media_type not in allowed_media_types:
            raise ArtifactValidationError(
                f"{self.artifact_type.value} artifacts require one of "
                f"{sorted(allowed_media_types)!r}, not {self.media_type!r}"
            )
        if self.media_type != "application/json":
            return
        try:
            payload = json.loads(self.content.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ArtifactValidationError(
                f"{self.artifact_type.value} JSON artifact is invalid"
            ) from exc
        if not isinstance(payload, dict) or not str(payload.get("schema_version", "")).strip():
            raise ArtifactValidationError(
                "JSON artifacts must contain a non-empty schema_version"
            )
        if str(payload["schema_version"]) != self.schema_version:
            raise ArtifactValidationError(
                "JSON artifact schema_version must match its object metadata"
            )


@dataclass(frozen=True, slots=True)
class ArtifactRecord:
    id: str
    document_snapshot_id: str
    artifact_type: ArtifactType
    object: ContentAddressedObject
    metadata: Mapping[str, object]


@dataclass(frozen=True, slots=True)
class ArtifactSaveResult:
    artifact: ArtifactRecord
    created: bool
