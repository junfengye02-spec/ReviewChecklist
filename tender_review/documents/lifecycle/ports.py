from __future__ import annotations

from typing import Protocol, runtime_checkable

from tender_review.documents.storage import ContentAddressedObject

from .models import (
    ArtifactRecord,
    ArtifactSaveResult,
    ArtifactSubmission,
    SnapshotRecord,
    SnapshotSaveResult,
    SourceDocument,
)


@runtime_checkable
class DocumentLifecycleRepository(Protocol):
    def create_or_get_snapshot(
        self, document: SourceDocument, object_reference: ContentAddressedObject
    ) -> SnapshotSaveResult: ...

    def get_snapshot(self, snapshot_id: str) -> SnapshotRecord: ...

    def create_or_get_artifact(
        self, submission: ArtifactSubmission, object_reference: ContentAddressedObject
    ) -> ArtifactSaveResult: ...

    def get_artifact(self, artifact_id: str) -> ArtifactRecord: ...

    def list_artifacts(self, document_snapshot_id: str) -> tuple[ArtifactRecord, ...]: ...

    def set_parse_status(
        self,
        snapshot_id: str,
        status: str,
        *,
        parser_name: str | None = None,
        parser_version: str | None = None,
    ) -> SnapshotRecord: ...

    def is_object_referenced(self, reference: ContentAddressedObject) -> bool: ...
