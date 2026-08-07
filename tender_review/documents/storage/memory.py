from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from threading import Lock

from tender_review.shared.errors import PermanentError

from .contracts import (
    CONTENT_ADDRESS_PREFIX,
    ContentAddressedObject,
    ContentAddressedPage,
    content_reference,
    object_key_for,
)


class StorageIntegrityError(PermanentError):
    default_code = "storage_integrity_error"


class InMemoryContentAddressedStore:
    """Offline implementation of the immutable object-store lifecycle contract."""

    def __init__(self, *, bucket: str = "artifacts", now_provider=None) -> None:
        if not bucket.strip():
            raise ValueError("bucket must not be empty")
        self._bucket = bucket
        self._now_provider = now_provider or (lambda: datetime.now(timezone.utc))
        self._objects: dict[str, tuple[bytes, ContentAddressedObject]] = {}
        self._lock = Lock()

    @property
    def bucket(self) -> str:
        return self._bucket

    def put_content(
        self,
        content: bytes,
        *,
        media_type: str,
        schema_version: str,
        created_at: datetime | None = None,
    ) -> ContentAddressedObject:
        if not isinstance(content, bytes):
            raise TypeError("content must be bytes")
        timestamp = created_at or self._now_provider()
        reference = content_reference(
            bucket=self._bucket,
            content=content,
            media_type=media_type,
            schema_version=schema_version,
            created_at=timestamp,
        )
        with self._lock:
            existing = self._objects.get(reference.object_key)
            if existing is not None:
                existing_content, existing_reference = existing
                if existing_content != content:
                    raise StorageIntegrityError(
                        f"content-addressed object {reference.object_key} was overwritten"
                    )
                return existing_reference
            self._objects[reference.object_key] = (content, reference)
        return reference

    def read_content(self, reference: ContentAddressedObject) -> bytes:
        self._validate_reference(reference)
        with self._lock:
            try:
                content, stored_reference = self._objects[reference.object_key]
            except KeyError as exc:
                raise StorageIntegrityError(
                    f"object {reference.object_key!r} does not exist",
                    code="storage_object_not_found",
                ) from exc
        if (
            stored_reference.bucket != reference.bucket
            or stored_reference.object_key != reference.object_key
            or stored_reference.sha256 != reference.sha256
            or stored_reference.size_bytes != reference.size_bytes
            or stored_reference.media_type != reference.media_type
            or stored_reference.schema_version != reference.schema_version
        ):
            raise StorageIntegrityError(
                f"object metadata changed for {reference.object_key}"
            )
        actual_hash = hashlib.sha256(content).hexdigest()
        if actual_hash != reference.sha256 or len(content) != reference.size_bytes:
            raise StorageIntegrityError(
                f"object content failed integrity verification for {reference.object_key}"
            )
        return content

    def list_content_addressed(
        self,
        *,
        prefix: str = CONTENT_ADDRESS_PREFIX,
        limit: int = 100,
        cursor: str | None = None,
    ) -> ContentAddressedPage:
        self._validate_listing(prefix=prefix, limit=limit, cursor=cursor)
        with self._lock:
            keys = [
                key
                for key in sorted(self._objects)
                if key.startswith(prefix) and (cursor is None or key > cursor)
            ]
            selected = keys[: limit + 1]
            objects = tuple(self._objects[key][1] for key in selected[:limit])
        return ContentAddressedPage(
            objects=objects,
            next_cursor=objects[-1].object_key if len(selected) > limit else None,
        )

    def delete_content(self, reference: ContentAddressedObject) -> None:
        self._validate_reference(reference)
        with self._lock:
            existing = self._objects.get(reference.object_key)
            if existing is None:
                return
            stored_reference = existing[1]
            if (
                stored_reference.bucket != reference.bucket
                or stored_reference.object_key != reference.object_key
                or stored_reference.sha256 != reference.sha256
                or stored_reference.size_bytes != reference.size_bytes
                or stored_reference.media_type != reference.media_type
                or stored_reference.schema_version != reference.schema_version
            ):
                raise StorageIntegrityError(
                    f"object metadata changed for {reference.object_key}"
                )
            del self._objects[reference.object_key]

    def corrupt_for_test(self, reference: ContentAddressedObject, content: bytes) -> None:
        """Deliberately violate stored bytes to exercise read-time verification."""

        with self._lock:
            _, stored_reference = self._objects[reference.object_key]
            self._objects[reference.object_key] = (content, stored_reference)

    def _validate_reference(self, reference: ContentAddressedObject) -> None:
        if reference.bucket != self._bucket:
            raise StorageIntegrityError(
                f"object bucket {reference.bucket!r} does not match {self._bucket!r}"
            )
        if reference.object_key != object_key_for(reference.sha256):
            raise StorageIntegrityError("object reference is not content-addressed")

    @staticmethod
    def _validate_listing(*, prefix: str, limit: int, cursor: str | None) -> None:
        if not prefix.startswith(CONTENT_ADDRESS_PREFIX):
            raise ValueError("only the sha256/ content-addressed prefix may be listed")
        if not 1 <= limit <= 1000:
            raise ValueError("limit must be between 1 and 1000")
        if cursor is not None and not cursor.startswith(prefix):
            raise ValueError("cursor must be within the requested prefix")
