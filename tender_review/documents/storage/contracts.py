from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import datetime
from typing import Protocol, runtime_checkable


CONTENT_ADDRESS_PREFIX = "sha256/"


def object_key_for(sha256: str) -> str:
    digest = sha256.strip().lower()
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise ValueError("sha256 must be a 64-character lowercase hexadecimal digest")
    return f"{CONTENT_ADDRESS_PREFIX}{digest[:2]}/{digest}"


@dataclass(frozen=True, slots=True)
class ContentAddressedObject:
    """A verified immutable object reference, never a logical file name."""

    bucket: str
    object_key: str
    sha256: str
    size_bytes: int
    media_type: str
    schema_version: str
    created_at: datetime | None = None

    def __post_init__(self) -> None:
        if not self.bucket.strip():
            raise ValueError("bucket must not be empty")
        if self.object_key != object_key_for(self.sha256):
            raise ValueError("object_key must be the SHA-256 content-addressed key")
        if self.size_bytes < 0:
            raise ValueError("size_bytes must not be negative")
        if not self.media_type.strip():
            raise ValueError("media_type must not be empty")
        if not self.schema_version.strip():
            raise ValueError("schema_version must not be empty")
        if self.created_at is not None and (
            self.created_at.tzinfo is None
            or self.created_at.utcoffset() is None
        ):
            raise ValueError("created_at must be timezone-aware")


@dataclass(frozen=True, slots=True)
class ContentAddressedPage:
    objects: tuple[ContentAddressedObject, ...]
    next_cursor: str | None


@runtime_checkable
class ContentAddressedObjectStore(Protocol):
    def put_content(
        self,
        content: bytes,
        *,
        media_type: str,
        schema_version: str,
        created_at: datetime | None = None,
    ) -> ContentAddressedObject: ...

    def read_content(self, reference: ContentAddressedObject) -> bytes: ...

    def list_content_addressed(
        self,
        *,
        prefix: str = CONTENT_ADDRESS_PREFIX,
        limit: int = 100,
        cursor: str | None = None,
    ) -> ContentAddressedPage: ...

    def delete_content(self, reference: ContentAddressedObject) -> None: ...


def content_reference(
    *,
    bucket: str,
    content: bytes,
    media_type: str,
    schema_version: str,
    created_at: datetime | None,
) -> ContentAddressedObject:
    digest = hashlib.sha256(content).hexdigest()
    return ContentAddressedObject(
        bucket=bucket,
        object_key=object_key_for(digest),
        sha256=digest,
        size_bytes=len(content),
        media_type=media_type,
        schema_version=schema_version,
        created_at=created_at,
    )
