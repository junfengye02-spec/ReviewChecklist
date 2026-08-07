from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

from datetime import datetime, timezone

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session, sessionmaker

from tender_review.documents.lifecycle.models import (
    ArtifactRecord,
    ArtifactSaveResult,
    ArtifactSubmission,
    ArtifactType,
    SnapshotRecord,
    SnapshotSaveResult,
    SourceDocument,
)
from tender_review.documents.storage import ContentAddressedObject
from tender_review.infrastructure.database.models import (
    DocumentArtifact,
    DocumentSnapshot,
)
from tender_review.shared.errors import ConflictError, NotFoundError


class SqlAlchemyDocumentLifecycleRepository:
    """Persistence adapter for immutable document snapshots and artifacts."""

    def __init__(
        self,
        sessions: sessionmaker[Session],
        *,
        snapshot_bucket: str,
    ) -> None:
        if not snapshot_bucket.strip():
            raise ValueError("snapshot_bucket must not be empty")
        self._sessions = sessions
        self._snapshot_bucket = snapshot_bucket

    @contextmanager
    def _transaction(self) -> Iterator[Session]:
        session = self._sessions()
        try:
            with session.begin():
                yield session
        finally:
            session.close()

    def create_or_get_snapshot(
        self, document: SourceDocument, object_reference: ContentAddressedObject
    ) -> SnapshotSaveResult:
        try:
            with self._transaction() as session:
                existing = self._find_snapshot_conflict(
                    session, document=document, object_reference=object_reference
                )
                if existing is not None:
                    return SnapshotSaveResult(
                        snapshot=self._snapshot_record(existing), created=False
                    )
                row = DocumentSnapshot(
                    sha256=object_reference.sha256,
                    object_key=object_reference.object_key,
                    source_system=document.source_system,
                    source_document_id=document.source_document_id,
                    file_name=document.file_name,
                    media_type=document.media_type,
                    size_bytes=object_reference.size_bytes,
                    schema_version=document.schema_version,
                )
                session.add(row)
                session.flush()
                return SnapshotSaveResult(
                    snapshot=self._snapshot_record(row), created=True
                )
        except IntegrityError as exc:
            # Concurrent writers may reach the two unique constraints together.
            with self._sessions() as session:
                existing = self._find_snapshot_conflict(
                    session, document=document, object_reference=object_reference
                )
                if existing is not None:
                    return SnapshotSaveResult(
                        snapshot=self._snapshot_record(existing), created=False
                    )
            raise ConflictError(
                "document snapshot could not be stored due to a conflicting source or hash"
            ) from exc

    def get_snapshot(self, snapshot_id: str) -> SnapshotRecord:
        with self._sessions() as session:
            row = session.get(DocumentSnapshot, snapshot_id)
            if row is None:
                raise NotFoundError(
                    f"document snapshot {snapshot_id!r} does not exist",
                    code="document_snapshot_not_found",
                )
            return self._snapshot_record(row)

    def create_or_get_artifact(
        self, submission: ArtifactSubmission, object_reference: ContentAddressedObject
    ) -> ArtifactSaveResult:
        try:
            with self._transaction() as session:
                if session.get(DocumentSnapshot, submission.document_snapshot_id) is None:
                    raise NotFoundError(
                        f"document snapshot {submission.document_snapshot_id!r} does not exist",
                        code="document_snapshot_not_found",
                    )
                existing = session.scalar(
                    select(DocumentArtifact).where(
                        DocumentArtifact.document_snapshot_id
                        == submission.document_snapshot_id,
                        DocumentArtifact.artifact_type == submission.artifact_type.value,
                        DocumentArtifact.object_key == object_reference.object_key,
                    )
                )
                if existing is not None:
                    self._assert_matching_artifact(existing, submission, object_reference)
                    return ArtifactSaveResult(
                        artifact=self._artifact_record(existing), created=False
                    )
                row = DocumentArtifact(
                    document_snapshot_id=submission.document_snapshot_id,
                    artifact_type=submission.artifact_type.value,
                    bucket=object_reference.bucket,
                    object_key=object_reference.object_key,
                    sha256=object_reference.sha256,
                    size_bytes=object_reference.size_bytes,
                    media_type=object_reference.media_type,
                    schema_version=object_reference.schema_version,
                    metadata_json=dict(submission.metadata),
                )
                session.add(row)
                session.flush()
                return ArtifactSaveResult(
                    artifact=self._artifact_record(row), created=True
                )
        except IntegrityError as exc:
            with self._sessions() as session:
                existing = session.scalar(
                    select(DocumentArtifact).where(
                        DocumentArtifact.document_snapshot_id
                        == submission.document_snapshot_id,
                        DocumentArtifact.artifact_type == submission.artifact_type.value,
                        DocumentArtifact.object_key == object_reference.object_key,
                    )
                )
                if existing is not None:
                    self._assert_matching_artifact(existing, submission, object_reference)
                    return ArtifactSaveResult(
                        artifact=self._artifact_record(existing), created=False
                    )
            raise ConflictError("document artifact could not be stored") from exc

    def get_artifact(self, artifact_id: str) -> ArtifactRecord:
        with self._sessions() as session:
            row = session.get(DocumentArtifact, artifact_id)
            if row is None:
                raise NotFoundError(
                    f"document artifact {artifact_id!r} does not exist",
                    code="document_artifact_not_found",
                )
            return self._artifact_record(row)

    def list_artifacts(self, document_snapshot_id: str) -> tuple[ArtifactRecord, ...]:
        with self._sessions() as session:
            rows = session.scalars(
                select(DocumentArtifact)
                .where(DocumentArtifact.document_snapshot_id == document_snapshot_id)
                .order_by(DocumentArtifact.created_at, DocumentArtifact.id)
            )
            return tuple(self._artifact_record(row) for row in rows)

    def set_parse_status(
        self,
        snapshot_id: str,
        status: str,
        *,
        parser_name: str | None = None,
        parser_version: str | None = None,
    ) -> SnapshotRecord:
        normalized = status.strip().upper()
        if normalized not in {"UPLOADED", "PARSING", "PARSED", "FAILED"}:
            raise ValueError(f"unsupported parse status: {status}")
        with self._transaction() as session:
            row = session.get(DocumentSnapshot, snapshot_id)
            if row is None:
                raise NotFoundError(
                    f"document snapshot {snapshot_id!r} does not exist",
                    code="document_snapshot_not_found",
                )
            row.parse_status = normalized
            row.parser_name = parser_name
            row.parser_version = parser_version
            session.flush()
            return self._snapshot_record(row)

    def is_object_referenced(self, reference: ContentAddressedObject) -> bool:
        with self._sessions() as session:
            artifact_reference = session.scalar(
                select(DocumentArtifact.id).where(
                    DocumentArtifact.bucket == reference.bucket,
                    DocumentArtifact.object_key == reference.object_key,
                )
            )
            if artifact_reference is not None:
                return True
            if reference.bucket != self._snapshot_bucket:
                return False
            snapshot_reference = session.scalar(
                select(DocumentSnapshot.id).where(
                    DocumentSnapshot.object_key == reference.object_key
                )
            )
            return snapshot_reference is not None

    def _find_snapshot_conflict(
        self,
        session: Session,
        *,
        document: SourceDocument,
        object_reference: ContentAddressedObject,
    ) -> DocumentSnapshot | None:
        source_match = session.scalar(
            select(DocumentSnapshot).where(
                DocumentSnapshot.source_system == document.source_system,
                DocumentSnapshot.source_document_id == document.source_document_id,
            )
        )
        if source_match is not None:
            if source_match.sha256 == object_reference.sha256:
                return source_match
            raise ConflictError(
                "source document already refers to different immutable content",
                code="source_document_hash_conflict",
            )
        hash_match = session.scalar(
            select(DocumentSnapshot).where(
                DocumentSnapshot.sha256 == object_reference.sha256
            )
        )
        if hash_match is not None:
            raise ConflictError(
                "content hash already belongs to a different source document",
                code="document_hash_source_conflict",
            )
        return None

    def _snapshot_record(self, row: DocumentSnapshot) -> SnapshotRecord:
        return SnapshotRecord(
            id=row.id,
            source_system=row.source_system,
            source_document_id=row.source_document_id,
            file_name=row.file_name,
            object=ContentAddressedObject(
                bucket=self._snapshot_bucket,
                object_key=row.object_key,
                sha256=row.sha256,
                size_bytes=row.size_bytes,
                media_type=row.media_type,
            schema_version=row.schema_version,
                created_at=_as_utc(row.created_at),
            ),
            parse_status=row.parse_status,
            parser_name=row.parser_name,
            parser_version=row.parser_version,
        )

    @staticmethod
    def _artifact_record(row: DocumentArtifact) -> ArtifactRecord:
        return ArtifactRecord(
            id=row.id,
            document_snapshot_id=row.document_snapshot_id,
            artifact_type=ArtifactType(row.artifact_type),
            object=ContentAddressedObject(
                bucket=row.bucket,
                object_key=row.object_key,
                sha256=row.sha256,
                size_bytes=row.size_bytes,
                media_type=row.media_type,
                schema_version=row.schema_version,
                created_at=_as_utc(row.created_at),
            ),
            metadata=dict(row.metadata_json),
        )

    @staticmethod
    def _assert_matching_artifact(
        row: DocumentArtifact,
        submission: ArtifactSubmission,
        object_reference: ContentAddressedObject,
    ) -> None:
        if (
            row.sha256 != object_reference.sha256
            or row.size_bytes != object_reference.size_bytes
            or row.media_type != object_reference.media_type
            or row.schema_version != object_reference.schema_version
            or dict(row.metadata_json) != dict(submission.metadata)
        ):
            raise ConflictError(
                "an immutable artifact exists with incompatible metadata",
                code="document_artifact_conflict",
            )


def _as_utc(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)
