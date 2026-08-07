from __future__ import annotations

from threading import RLock

from tender_review.documents.storage import ContentAddressedObject
from tender_review.shared.errors import ConflictError, NotFoundError
from tender_review.shared.ids import IdGenerator, UuidGenerator

from .models import (
    ArtifactRecord,
    ArtifactSaveResult,
    ArtifactSubmission,
    SnapshotRecord,
    SnapshotSaveResult,
    SourceDocument,
)


class InMemoryDocumentLifecycleRepository:
    """Thread-safe lifecycle repository used by the offline application graph."""

    def __init__(self, ids: IdGenerator | None = None) -> None:
        self._ids = ids or UuidGenerator()
        self._snapshots: dict[str, SnapshotRecord] = {}
        self._snapshot_by_source: dict[tuple[str, str], str] = {}
        self._snapshot_by_hash: dict[str, str] = {}
        self._artifacts: dict[str, ArtifactRecord] = {}
        self._artifact_keys: dict[tuple[str, str, str], str] = {}
        self._lock = RLock()

    def create_or_get_snapshot(
        self, document: SourceDocument, object_reference: ContentAddressedObject
    ) -> SnapshotSaveResult:
        source_key = (document.source_system, document.source_document_id)
        with self._lock:
            existing_id = self._snapshot_by_source.get(source_key)
            if existing_id is not None:
                existing = self._snapshots[existing_id]
                if existing.object.sha256 != object_reference.sha256:
                    raise ConflictError(
                        "source document already refers to different immutable content",
                        code="source_document_hash_conflict",
                    )
                return SnapshotSaveResult(snapshot=existing, created=False)
            hash_owner = self._snapshot_by_hash.get(object_reference.sha256)
            if hash_owner is not None:
                raise ConflictError(
                    "content hash already belongs to a different source document",
                    code="document_hash_source_conflict",
                )
            snapshot = SnapshotRecord(
                id=self._ids.new(),
                source_system=document.source_system,
                source_document_id=document.source_document_id,
                file_name=document.file_name,
                object=object_reference,
                parse_status="UPLOADED",
                parser_name=None,
                parser_version=None,
            )
            self._snapshots[snapshot.id] = snapshot
            self._snapshot_by_source[source_key] = snapshot.id
            self._snapshot_by_hash[object_reference.sha256] = snapshot.id
            return SnapshotSaveResult(snapshot=snapshot, created=True)

    def get_snapshot(self, snapshot_id: str) -> SnapshotRecord:
        with self._lock:
            try:
                return self._snapshots[snapshot_id]
            except KeyError as exc:
                raise NotFoundError(
                    f"document snapshot {snapshot_id!r} does not exist",
                    code="document_snapshot_not_found",
                ) from exc

    def create_or_get_artifact(
        self, submission: ArtifactSubmission, object_reference: ContentAddressedObject
    ) -> ArtifactSaveResult:
        key = (
            submission.document_snapshot_id,
            submission.artifact_type.value,
            object_reference.object_key,
        )
        with self._lock:
            if submission.document_snapshot_id not in self._snapshots:
                raise NotFoundError(
                    f"document snapshot {submission.document_snapshot_id!r} does not exist",
                    code="document_snapshot_not_found",
                )
            existing_id = self._artifact_keys.get(key)
            if existing_id is not None:
                existing = self._artifacts[existing_id]
                if dict(existing.metadata) != dict(submission.metadata):
                    raise ConflictError(
                        "an immutable artifact exists with incompatible metadata",
                        code="document_artifact_conflict",
                    )
                return ArtifactSaveResult(artifact=existing, created=False)
            artifact = ArtifactRecord(
                id=self._ids.new(),
                document_snapshot_id=submission.document_snapshot_id,
                artifact_type=submission.artifact_type,
                object=object_reference,
                metadata=dict(submission.metadata),
            )
            self._artifacts[artifact.id] = artifact
            self._artifact_keys[key] = artifact.id
            return ArtifactSaveResult(artifact=artifact, created=True)

    def get_artifact(self, artifact_id: str) -> ArtifactRecord:
        with self._lock:
            try:
                return self._artifacts[artifact_id]
            except KeyError as exc:
                raise NotFoundError(
                    f"document artifact {artifact_id!r} does not exist",
                    code="document_artifact_not_found",
                ) from exc

    def list_artifacts(self, document_snapshot_id: str) -> tuple[ArtifactRecord, ...]:
        with self._lock:
            return tuple(
                item
                for item in self._artifacts.values()
                if item.document_snapshot_id == document_snapshot_id
            )

    def set_parse_status(
        self,
        snapshot_id: str,
        status: str,
        *,
        parser_name: str | None = None,
        parser_version: str | None = None,
    ) -> SnapshotRecord:
        allowed = {"UPLOADED", "PARSING", "PARSED", "FAILED"}
        normalized = status.strip().upper()
        if normalized not in allowed:
            raise ValueError(f"unsupported parse status: {status}")
        with self._lock:
            current = self.get_snapshot(snapshot_id)
            updated = SnapshotRecord(
                id=current.id,
                source_system=current.source_system,
                source_document_id=current.source_document_id,
                file_name=current.file_name,
                object=current.object,
                parse_status=normalized,
                parser_name=parser_name,
                parser_version=parser_version,
            )
            self._snapshots[snapshot_id] = updated
            return updated

    def is_object_referenced(self, reference: ContentAddressedObject) -> bool:
        with self._lock:
            return any(
                item.object.bucket == reference.bucket
                and item.object.object_key == reference.object_key
                for item in (*self._snapshots.values(), *self._artifacts.values())
            )
