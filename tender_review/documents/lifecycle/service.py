from __future__ import annotations

import logging
import time
from datetime import datetime, timedelta, timezone

from tender_review.documents.storage import (
    CONTENT_ADDRESS_PREFIX,
    ContentAddressedObjectStore,
)

from .models import (
    ArtifactSaveResult,
    ArtifactSubmission,
    SnapshotSaveResult,
    SourceDocument,
)
from .ports import DocumentLifecycleRepository
from tender_review.shared.observability import (
    CorrelationContext,
    log_event,
    record_metric,
)


class DocumentLifecycleService:
    """Coordinates storage-before-database commits for immutable document facts."""

    def __init__(
        self,
        *,
        storage: ContentAddressedObjectStore,
        repository: DocumentLifecycleRepository,
    ) -> None:
        self._storage = storage
        self._repository = repository
        self._logger = logging.getLogger("tender_review.documents.lifecycle")

    def upload_document(self, document: SourceDocument) -> SnapshotSaveResult:
        object_reference = self._storage.put_content(
            document.content,
            media_type=document.media_type,
            schema_version=document.schema_version,
        )
        # The repository transaction happens only after a full object read-back.
        self._storage.read_content(object_reference)
        try:
            return self._repository.create_or_get_snapshot(document, object_reference)
        except Exception as exc:
            self._record_persistence_failure(
                object_reference,
                resource_type="document_snapshot",
                error=exc,
            )
            raise

    def write_artifact(self, submission: ArtifactSubmission) -> ArtifactSaveResult:
        object_reference = self._storage.put_content(
            submission.content,
            media_type=submission.media_type,
            schema_version=submission.schema_version,
        )
        self._storage.read_content(object_reference)
        try:
            return self._repository.create_or_get_artifact(
                submission, object_reference
            )
        except Exception as exc:
            self._record_persistence_failure(
                object_reference,
                resource_type="document_artifact",
                error=exc,
            )
            raise

    def read_snapshot(self, snapshot_id: str) -> bytes:
        snapshot = self._repository.get_snapshot(snapshot_id)
        return self._storage.read_content(snapshot.object)

    def read_artifact(self, artifact_id: str) -> bytes:
        artifact = self._repository.get_artifact(artifact_id)
        return self._storage.read_content(artifact.object)

    def _record_persistence_failure(
        self,
        reference,
        *,
        resource_type: str,
        error: Exception,
    ) -> None:
        try:
            referenced = self._repository.is_object_referenced(reference)
        except Exception:
            referenced = None
        decision = (
            "retain_existing_reference"
            if referenced is True
            else "deferred_reference_check_unavailable"
            if referenced is None
            else "deferred_retention_cleanup"
        )
        correlation = CorrelationContext(
            call_id=f"object:{reference.sha256[:64]}"
        )
        log_event(
            self._logger,
            logging.WARNING,
            event="object.database_commit_failed",
            message="Object upload succeeded but database persistence failed",
            context=correlation,
            resource_type=resource_type,
            object_sha256=reference.sha256,
            cleanup_decision=decision,
            referenced=referenced,
            error_type=type(error).__name__,
            error_code=getattr(error, "code", None),
        )


class OrphanCleanupResult:
    def __init__(
        self,
        *,
        scanned: int,
        deleted: int,
        retained: int,
        too_recent: int,
        next_cursor: str | None,
    ) -> None:
        self.scanned = scanned
        self.deleted = deleted
        self.retained = retained
        self.too_recent = too_recent
        self.next_cursor = next_cursor


class OrphanCleanupService:
    """Deletes only old, content-addressed objects that remain unreferenced twice."""

    def __init__(
        self,
        *,
        storage: ContentAddressedObjectStore,
        repository: DocumentLifecycleRepository,
        now_provider=None,
    ) -> None:
        self._storage = storage
        self._repository = repository
        self._now_provider = now_provider or (lambda: datetime.now(timezone.utc))
        self._logger = logging.getLogger("tender_review.documents.cleanup")

    def clean_page(
        self,
        *,
        retention: timedelta,
        page_size: int = 100,
        delete_limit: int = 100,
        cursor: str | None = None,
    ) -> OrphanCleanupResult:
        if retention <= timedelta(0):
            raise ValueError("retention must be positive")
        if not 1 <= page_size <= 1000:
            raise ValueError("page_size must be between 1 and 1000")
        if not 1 <= delete_limit <= 1000:
            raise ValueError("delete_limit must be between 1 and 1000")
        now = self._now_provider()
        if now.tzinfo is None or now.utcoffset() is None:
            raise ValueError("now_provider must return a timezone-aware datetime")
        cutoff = now - retention
        started = time.perf_counter()
        page = self._storage.list_content_addressed(
            prefix=CONTENT_ADDRESS_PREFIX,
            limit=page_size,
            cursor=cursor,
        )
        deleted = retained = too_recent = 0
        for reference in page.objects:
            if reference.created_at is None or reference.created_at > cutoff:
                too_recent += 1
                self._log_decision(reference.sha256, "retained_too_recent")
                continue
            if self._repository.is_object_referenced(reference):
                retained += 1
                self._log_decision(reference.sha256, "retained_referenced")
                continue
            if deleted >= delete_limit:
                retained += 1
                self._log_decision(reference.sha256, "retained_delete_limit")
                continue
            # A reference can appear after the first lookup; test again at deletion.
            if self._repository.is_object_referenced(reference):
                retained += 1
                self._log_decision(
                    reference.sha256, "retained_concurrent_reference"
                )
                continue
            self._storage.delete_content(reference)
            deleted += 1
            self._log_decision(reference.sha256, "deleted_orphan")
        duration_ms = max(0.0, (time.perf_counter() - started) * 1000.0)
        record_metric(
            self._logger,
            name="orphan_cleanup_duration",
            value=duration_ms,
            unit="ms",
            source="process_monotonic",
            scanned=len(page.objects),
            deleted=deleted,
            retained=retained,
            too_recent=too_recent,
        )
        log_event(
            self._logger,
            logging.INFO,
            event="object.cleanup_page_completed",
            message="Object orphan cleanup page completed",
            scanned=len(page.objects),
            deleted=deleted,
            retained=retained,
            too_recent=too_recent,
            has_next_page=page.next_cursor is not None,
        )
        return OrphanCleanupResult(
            scanned=len(page.objects),
            deleted=deleted,
            retained=retained,
            too_recent=too_recent,
            next_cursor=page.next_cursor,
        )

    def _log_decision(self, sha256: str, decision: str) -> None:
        log_event(
            self._logger,
            logging.INFO,
            event="object.cleanup_decision",
            message="Object cleanup decision recorded",
            context=CorrelationContext(call_id=f"object:{sha256[:64]}"),
            object_sha256=sha256,
            cleanup_decision=decision,
        )
