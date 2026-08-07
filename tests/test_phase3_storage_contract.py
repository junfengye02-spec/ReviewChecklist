from __future__ import annotations

import hashlib
import io
import unittest
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

from tender_review.documents.storage import (
    ContentAddressedObjectStore,
    InMemoryContentAddressedStore,
    StorageIntegrityError,
)
from tender_review.infrastructure.object_storage import MinioArtifactStore


NOW = datetime(2026, 7, 27, 12, 0, tzinfo=timezone.utc)


class FakeS3Error(Exception):
    def __init__(self, code: str):
        super().__init__(code)
        self.code = code


class FakeResponse(io.BytesIO):
    def release_conn(self) -> None:
        pass


class FakeMinioClient:
    def __init__(self) -> None:
        self.buckets: set[str] = set()
        self.objects: dict[tuple[str, str], dict[str, object]] = {}
        self.put_count = 0

    def bucket_exists(self, bucket: str) -> bool:
        return bucket in self.buckets

    def make_bucket(self, bucket: str) -> None:
        self.buckets.add(bucket)

    def stat_object(self, bucket: str, key: str):
        try:
            value = self.objects[(bucket, key)]
        except KeyError as exc:
            raise FakeS3Error("NoSuchKey") from exc
        return SimpleNamespace(
            size=len(value["content"]),
            content_type=value["content_type"],
            metadata=value["metadata"],
            last_modified=value["last_modified"],
        )

    def put_object(
        self, bucket: str, key: str, stream, length: int, *, content_type: str, metadata
    ) -> None:
        content = stream.read()
        if len(content) != length:
            raise AssertionError("length mismatch")
        self.put_count += 1
        self.objects[(bucket, key)] = {
            "content": content,
            "content_type": content_type,
            "metadata": metadata,
            "last_modified": NOW,
        }

    def get_object(self, bucket: str, key: str) -> FakeResponse:
        try:
            return FakeResponse(self.objects[(bucket, key)]["content"])
        except KeyError as exc:
            raise FakeS3Error("NoSuchKey") from exc

    def list_objects(self, bucket: str, *, prefix: str, recursive: bool, start_after=None, include_user_meta=False):
        del recursive, include_user_meta
        values = []
        for (object_bucket, key), value in sorted(self.objects.items()):
            if object_bucket == bucket and key.startswith(prefix) and (
                start_after is None or key > start_after
            ):
                values.append(
                    SimpleNamespace(
                        object_name=key,
                        size=len(value["content"]),
                        content_type=value["content_type"],
                        metadata=value["metadata"],
                        last_modified=value["last_modified"],
                    )
                )
        return values

    def remove_object(self, bucket: str, key: str) -> None:
        self.objects.pop((bucket, key), None)


class ContentAddressedStoreContract:
    def run(self, store: ContentAddressedObjectStore) -> None:
        first = store.put_content(
            b"same payload", media_type="application/pdf", schema_version="1", created_at=NOW - timedelta(days=2)
        )
        second = store.put_content(
            b"same payload", media_type="application/pdf", schema_version="1", created_at=NOW
        )
        if first != second:
            raise AssertionError("same content must deduplicate to one reference")
        if first.object_key != f"sha256/{first.sha256[:2]}/{first.sha256}":
            raise AssertionError("object key must be content addressed")
        if store.read_content(first) != b"same payload":
            raise AssertionError("read content mismatch")
        page = store.list_content_addressed(limit=10)
        if page.objects != (first,):
            raise AssertionError("listing mismatch")
        store.delete_content(first)
        try:
            store.read_content(first)
        except StorageIntegrityError:
            pass
        else:
            raise AssertionError("deleted object must not be readable")


class StorageContractTests(unittest.TestCase):
    def assertEqual(self, first, second, msg=None):  # type: ignore[no-untyped-def]
        return super().assertEqual(first, second, msg)

    def test_memory_store_contract(self):
        ContentAddressedStoreContract().run(InMemoryContentAddressedStore(now_provider=lambda: NOW))

    def test_minio_store_contract_without_network(self):
        client = FakeMinioClient()
        store = MinioArtifactStore(
            endpoint="minio.invalid:9000",
            access_key="access",
            secret_key="secret",
            bucket="artifacts",
            client=client,
        )
        store.ensure_bucket()
        ContentAddressedStoreContract().run(store)
        self.assertEqual(client.put_count, 1)

    def test_read_detects_corruption_even_when_hash_metadata_is_unchanged(self):
        store = InMemoryContentAddressedStore(now_provider=lambda: NOW)
        reference = store.put_content(b"original", media_type="application/pdf", schema_version="1")
        store.corrupt_for_test(reference, b"tampered")
        with self.assertRaises(StorageIntegrityError):
            store.read_content(reference)

    def test_minio_existing_metadata_is_immutable(self):
        client = FakeMinioClient()
        store = MinioArtifactStore(
            endpoint="minio.invalid:9000", access_key="a", secret_key="b", bucket="artifacts", client=client
        )
        store.ensure_bucket()
        payload = b"immutable"
        digest = hashlib.sha256(payload).hexdigest()
        store.put_content(payload, media_type="application/pdf", schema_version="1", created_at=NOW)
        client.objects[("artifacts", f"sha256/{digest[:2]}/{digest}")]["content"] = b"changed"
        with self.assertRaises(StorageIntegrityError):
            store.put_content(payload, media_type="application/pdf", schema_version="1", created_at=NOW)


if __name__ == "__main__":
    unittest.main()
