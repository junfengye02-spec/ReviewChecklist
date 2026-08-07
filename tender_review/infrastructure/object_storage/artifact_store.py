from __future__ import annotations

import hashlib
import io
import json
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Mapping

from tender_review.documents.models import (
    ArtifactReference as ContractArtifactReference,
    ArtifactWrite,
)
from tender_review.documents.storage import (
    CONTENT_ADDRESS_PREFIX,
    ContentAddressedObject,
    ContentAddressedPage,
    StorageIntegrityError,
)
from tender_review.infrastructure.health import HealthStatus
from tender_review.shared.contracts import CallContext, ensure_call_active
from tender_review.shared.errors import NotFoundError, RetryableError


class ArtifactStoreError(RetryableError):
    pass


class ArtifactNotFoundError(NotFoundError):
    pass


class ArtifactIntegrityError(StorageIntegrityError):
    pass


@dataclass(frozen=True, slots=True)
class ArtifactReference:
    bucket: str
    object_key: str
    sha256: str
    size_bytes: int
    media_type: str
    schema_version: str = "1"
    created_at: datetime | None = None


class MinioArtifactStore:
    """Immutable, SHA-256-addressed object storage backed by MinIO."""

    def __init__(
        self,
        *,
        endpoint: str,
        access_key: str,
        secret_key: str,
        bucket: str,
        secure: bool = False,
        region: str | None = None,
        timeout_seconds: float = 5.0,
        max_attempts: int = 3,
        client: Any | None = None,
    ) -> None:
        if not endpoint.strip():
            raise ValueError("endpoint must not be empty")
        if not bucket.strip():
            raise ValueError("bucket must not be empty")
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if max_attempts < 1:
            raise ValueError("max_attempts must be at least 1")
        if client is None:
            try:
                from minio import Minio
                import urllib3
            except ImportError as exc:
                raise ArtifactStoreError(
                    "The 'minio' package is required for MinioArtifactStore"
                ) from exc
            http_client = urllib3.PoolManager(
                timeout=urllib3.Timeout(
                    connect=timeout_seconds,
                    read=timeout_seconds,
                ),
                retries=urllib3.Retry(
                    total=max_attempts - 1,
                    backoff_factor=0.2,
                    status_forcelist=(429, 500, 502, 503, 504),
                ),
            )
            client = Minio(
                endpoint,
                access_key=access_key,
                secret_key=secret_key,
                secure=secure,
                region=region,
                http_client=http_client,
            )
        self._client = client
        self._bucket = bucket

    @property
    def bucket(self) -> str:
        return self._bucket

    @property
    def name(self) -> str:
        return "object_storage"

    @staticmethod
    def object_key_for(sha256: str) -> str:
        normalized = sha256.lower()
        if len(normalized) != 64 or any(
            char not in "0123456789abcdef" for char in normalized
        ):
            raise ValueError("sha256 must be a 64-character hexadecimal digest")
        return f"sha256/{normalized[:2]}/{normalized}"

    def ensure_bucket(self) -> None:
        try:
            if not self._client.bucket_exists(self._bucket):
                self._client.make_bucket(self._bucket)
        except Exception as exc:
            raise ArtifactStoreError(
                f"could not ensure artifact bucket: {type(exc).__name__}"
            ) from exc

    def put_bytes(
        self,
        content: bytes,
        *,
        media_type: str = "application/octet-stream",
        schema_version: str = "1",
        created_at: datetime | None = None,
        call_id: str | None = None,
        call: CallContext | None = None,
    ) -> ArtifactReference:
        if call is not None:
            ensure_call_active(call)
            call_id = call.call_id
        if not isinstance(content, bytes):
            raise TypeError("content must be bytes")
        if not schema_version.strip():
            raise ValueError("schema_version must not be empty")
        if created_at is not None and (
            created_at.tzinfo is None or created_at.utcoffset() is None
        ):
            raise ValueError("created_at must be timezone-aware")

        digest = hashlib.sha256(content).hexdigest()
        object_key = self.object_key_for(digest)
        timestamp = created_at or datetime.now(timezone.utc)
        metadata = {
            "sha256": digest,
            "schema-version": schema_version,
            "created-at": timestamp.astimezone(timezone.utc).isoformat(),
        }
        if call_id:
            metadata["call-id"] = call_id

        try:
            try:
                stat = self._client.stat_object(self._bucket, object_key)
            except Exception as exc:
                if not self._is_not_found(exc):
                    raise
            else:
                reference = self._validated_reference(
                    stat, object_key, digest, len(content), media_type, schema_version
                )
                self.get_bytes(reference, call=call)
                return reference

            self._client.put_object(
                self._bucket,
                object_key,
                io.BytesIO(content),
                len(content),
                content_type=media_type,
                metadata=metadata,
            )
            stat = self._client.stat_object(self._bucket, object_key)
            reference = self._validated_reference(
                stat, object_key, digest, len(content), media_type, schema_version
            )
            self.get_bytes(reference, call=call)
            return reference
        except ArtifactIntegrityError:
            raise
        except Exception as exc:
            raise ArtifactStoreError(
                f"artifact upload failed: {type(exc).__name__}"
            ) from exc

    def put_json(
        self,
        payload: Mapping[str, Any],
        *,
        call_id: str | None = None,
        call: CallContext | None = None,
    ) -> ArtifactReference:
        if (
            "schema_version" not in payload
            or not str(payload["schema_version"]).strip()
        ):
            raise ValueError("JSON artifacts must contain schema_version")
        encoded = json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return self.put_bytes(
            encoded,
            media_type="application/json",
            schema_version=str(payload["schema_version"]),
            call_id=call_id,
            call=call,
        )

    def put_content(
        self,
        content: bytes,
        *,
        media_type: str,
        schema_version: str,
        created_at: datetime | None = None,
    ) -> ContentAddressedObject:
        reference = self.put_bytes(
            content,
            media_type=media_type,
            schema_version=schema_version,
            created_at=created_at,
        )
        return ContentAddressedObject(
            bucket=reference.bucket,
            object_key=reference.object_key,
            sha256=reference.sha256,
            size_bytes=reference.size_bytes,
            media_type=reference.media_type,
            schema_version=reference.schema_version,
            created_at=reference.created_at,
        )

    def read_content(self, reference: ContentAddressedObject) -> bytes:
        self._validate_content_reference(reference)
        try:
            stored = self._reference_for_key(reference.object_key)
        except ArtifactNotFoundError as exc:
            raise ArtifactIntegrityError(
                f"content-addressed object {reference.object_key!r} does not exist"
            ) from exc
        if (
            stored.sha256 != reference.sha256
            or stored.size_bytes != reference.size_bytes
            or stored.media_type != reference.media_type
            or stored.schema_version != reference.schema_version
        ):
            raise ArtifactIntegrityError(
                f"stored metadata mismatch for {reference.object_key}"
            )
        return self.get_bytes(
            ArtifactReference(
                bucket=reference.bucket,
                object_key=reference.object_key,
                sha256=reference.sha256,
                size_bytes=reference.size_bytes,
                media_type=reference.media_type,
                schema_version=reference.schema_version,
                created_at=reference.created_at,
            )
        )

    def list_content_addressed(
        self,
        *,
        prefix: str = CONTENT_ADDRESS_PREFIX,
        limit: int = 100,
        cursor: str | None = None,
    ) -> ContentAddressedPage:
        if not prefix.startswith(CONTENT_ADDRESS_PREFIX):
            raise ValueError("only the sha256/ content-addressed prefix may be listed")
        if not 1 <= limit <= 1000:
            raise ValueError("limit must be between 1 and 1000")
        if cursor is not None and not cursor.startswith(prefix):
            raise ValueError("cursor must be within the requested prefix")
        try:
            try:
                listing = self._client.list_objects(
                    self._bucket,
                    prefix=prefix,
                    recursive=True,
                    start_after=cursor,
                    include_user_meta=True,
                )
            except TypeError:
                listing = self._client.list_objects(
                    self._bucket,
                    prefix=prefix,
                    recursive=True,
                    start_after=cursor,
                )
            objects: list[ContentAddressedObject] = []
            has_more = False
            for entry in listing:
                object_key = str(getattr(entry, "object_name", ""))
                try:
                    digest = object_key.rsplit("/", 1)[-1]
                    if self.object_key_for(digest) != object_key:
                        continue
                except ValueError:
                    continue
                if len(objects) == limit:
                    has_more = True
                    break
                metadata = {
                    str(key).lower(): str(value)
                    for key, value in (
                        getattr(entry, "metadata", None)
                        or getattr(entry, "user_metadata", None)
                        or {}
                    ).items()
                }
                objects.append(
                    ContentAddressedObject(
                        bucket=self._bucket,
                        object_key=object_key,
                        sha256=digest,
                        size_bytes=int(getattr(entry, "size", 0)),
                        media_type=str(
                            getattr(entry, "content_type", None)
                            or "application/octet-stream"
                        ),
                        schema_version=(
                            metadata.get("x-amz-meta-schema-version")
                            or metadata.get("schema-version")
                            or "1"
                        ),
                        created_at=self._created_at(entry, metadata),
                    )
                )
        except ArtifactIntegrityError:
            raise
        except Exception as exc:
            raise ArtifactStoreError(
                f"artifact listing failed: {type(exc).__name__}"
            ) from exc
        return ContentAddressedPage(
            objects=tuple(objects),
            next_cursor=objects[-1].object_key if has_more else None,
        )

    def delete_content(self, reference: ContentAddressedObject) -> None:
        self._validate_content_reference(reference)
        self.delete(reference.object_key)

    def put(
        self, artifact: ArtifactWrite, *, call: CallContext | None = None
    ) -> ContractArtifactReference:
        reference = self.put_bytes(
            bytes(artifact.content),
            media_type=artifact.media_type,
            schema_version=str(artifact.schema_version),
            call=call,
        )
        return ContractArtifactReference(
            key=reference.object_key,
            sha256=reference.sha256,
            size_bytes=reference.size_bytes,
            media_type=reference.media_type,
            schema_version=artifact.schema_version,
        )

    def get(self, key: str, *, call: CallContext | None = None) -> bytes:
        if call is not None:
            ensure_call_active(call)
        return self.get_bytes(self._reference_for_key(key), call=call)

    def exists(self, key: str, *, call: CallContext | None = None) -> bool:
        if call is not None:
            ensure_call_active(call)
        try:
            self._client.stat_object(self._bucket, key)
        except Exception as exc:
            if self._is_not_found(exc):
                return False
            raise ArtifactStoreError(
                f"artifact existence check failed: {type(exc).__name__}"
            ) from exc
        return True

    def delete(self, key: str, *, call: CallContext | None = None) -> None:
        if call is not None:
            ensure_call_active(call)
        try:
            self._client.remove_object(self._bucket, key)
        except Exception as exc:
            if self._is_not_found(exc):
                return
            raise ArtifactStoreError(
                f"artifact deletion failed: {type(exc).__name__}"
            ) from exc

    def get_bytes(
        self,
        reference: ArtifactReference,
        *,
        call: CallContext | None = None,
    ) -> bytes:
        if call is not None:
            ensure_call_active(call)
        self._validate_reference_bucket(reference)
        if self.object_key_for(reference.sha256) != reference.object_key:
            raise ArtifactIntegrityError(
                f"artifact key does not match its SHA-256 digest: {reference.object_key}"
            )
        response: Any | None = None
        try:
            response = self._client.get_object(self._bucket, reference.object_key)
            content = response.read()
        except Exception as exc:
            if self._is_not_found(exc):
                raise ArtifactNotFoundError(
                    f"artifact {reference.object_key!r} does not exist",
                    code="artifact_not_found",
                ) from exc
            raise ArtifactStoreError(
                f"artifact download failed: {type(exc).__name__}"
            ) from exc
        finally:
            if response is not None:
                close = getattr(response, "close", None)
                if close is not None:
                    close()
                release_conn = getattr(response, "release_conn", None)
                if release_conn is not None:
                    release_conn()

        actual_hash = hashlib.sha256(content).hexdigest()
        if len(content) != reference.size_bytes or actual_hash != reference.sha256:
            raise ArtifactIntegrityError(
                f"artifact integrity check failed for {reference.object_key}"
            )
        return content

    def presigned_get_url(
        self,
        reference: ArtifactReference,
        *,
        expires: timedelta = timedelta(minutes=15),
        call: CallContext | None = None,
    ) -> str:
        if call is not None:
            ensure_call_active(call)
        self._validate_reference_bucket(reference)
        if expires <= timedelta(0) or expires > timedelta(days=7):
            raise ValueError("expires must be between 0 seconds and 7 days")
        try:
            return str(
                self._client.presigned_get_object(
                    self._bucket, reference.object_key, expires=expires
                )
            )
        except Exception as exc:
            raise ArtifactStoreError(
                f"could not create presigned URL: {type(exc).__name__}"
            ) from exc

    def check(self) -> HealthStatus:
        try:
            healthy = bool(self._client.bucket_exists(self._bucket))
        except Exception as exc:
            return HealthStatus(
                service=self.name,
                healthy=False,
                detail=f"object storage unavailable: {type(exc).__name__}",
            )
        return HealthStatus(
            service=self.name,
            healthy=healthy,
            detail="ok" if healthy else "artifact bucket does not exist",
        )

    def _validate_reference_bucket(self, reference: ArtifactReference) -> None:
        if reference.bucket != self._bucket:
            raise ArtifactStoreError(
                f"artifact reference belongs to bucket {reference.bucket!r}, "
                f"not {self._bucket!r}"
            )

    def _validate_content_reference(self, reference: ContentAddressedObject) -> None:
        if reference.bucket != self._bucket:
            raise ArtifactIntegrityError(
                f"artifact reference belongs to bucket {reference.bucket!r}, "
                f"not {self._bucket!r}"
            )
        if self.object_key_for(reference.sha256) != reference.object_key:
            raise ArtifactIntegrityError("artifact reference is not content-addressed")

    def _reference_for_key(self, key: str) -> ArtifactReference:
        try:
            digest = key.rsplit("/", 1)[-1]
            self.object_key_for(digest)
        except ValueError as exc:
            raise ArtifactIntegrityError(
                f"artifact key is not content-addressed: {key!r}",
                code="artifact_key_invalid",
            ) from exc
        try:
            stat = self._client.stat_object(self._bucket, key)
        except Exception as exc:
            if self._is_not_found(exc):
                raise ArtifactNotFoundError(
                    f"artifact {key!r} does not exist", code="artifact_not_found"
                ) from exc
            raise ArtifactStoreError(
                f"artifact stat failed: {type(exc).__name__}"
            ) from exc
        metadata = {
            str(name).lower(): str(value)
            for name, value in (getattr(stat, "metadata", None) or {}).items()
        }
        schema_version = (
            metadata.get("x-amz-meta-schema-version")
            or metadata.get("schema-version")
            or "1"
        )
        return self._validated_reference(
            stat,
            key,
            digest,
            int(getattr(stat, "size", -1)),
            str(getattr(stat, "content_type", None) or "application/octet-stream"),
            schema_version,
        )

    @staticmethod
    def _is_not_found(exc: Exception) -> bool:
        return getattr(exc, "code", None) in {
            "NoSuchBucket",
            "NoSuchKey",
            "NoSuchObject",
            "NotFound",
        }

    def _validated_reference(
        self,
        stat: Any,
        object_key: str,
        expected_hash: str,
        expected_size: int,
        media_type: str,
        schema_version: str,
    ) -> ArtifactReference:
        metadata = {
            str(key).lower(): str(value)
            for key, value in (getattr(stat, "metadata", None) or {}).items()
        }
        stored_hash = metadata.get("x-amz-meta-sha256") or metadata.get("sha256")
        stored_version = metadata.get("x-amz-meta-schema-version") or metadata.get(
            "schema-version"
        )
        if getattr(stat, "size", None) != expected_size:
            raise ArtifactIntegrityError(f"stored size mismatch for {object_key}")
        if stored_hash != expected_hash:
            raise ArtifactIntegrityError(
                f"stored hash metadata mismatch for {object_key}"
            )
        if stored_version != schema_version:
            raise ArtifactIntegrityError(
                f"stored schema version metadata mismatch for {object_key}"
            )
        return ArtifactReference(
            bucket=self._bucket,
            object_key=object_key,
            sha256=expected_hash,
            size_bytes=expected_size,
            media_type=str(getattr(stat, "content_type", None) or media_type),
            schema_version=schema_version,
            created_at=self._created_at(stat, metadata),
        )

    @staticmethod
    def _created_at(value: Any, metadata: Mapping[str, str]) -> datetime | None:
        encoded = metadata.get("x-amz-meta-created-at") or metadata.get("created-at")
        if encoded:
            try:
                parsed = datetime.fromisoformat(encoded.replace("Z", "+00:00"))
            except ValueError:
                parsed = None
            else:
                if parsed.tzinfo is not None and parsed.utcoffset() is not None:
                    return parsed.astimezone(timezone.utc)
        last_modified = getattr(value, "last_modified", None)
        if isinstance(last_modified, datetime):
            if last_modified.tzinfo is None or last_modified.utcoffset() is None:
                return last_modified.replace(tzinfo=timezone.utc)
            return last_modified.astimezone(timezone.utc)
        return None
