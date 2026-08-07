from __future__ import annotations

import io
import unittest
from types import SimpleNamespace

from sqlalchemy import create_engine

from tender_review.documents import ArtifactStore, InMemoryArtifactStore
from tender_review.documents.models import ArtifactWrite
from tender_review.infrastructure.database import DatabaseHealthAdapter
from tender_review.infrastructure.object_storage import MinioArtifactStore
from tender_review.shared.contracts import CallContext
from tender_review.shared.errors import CancelledError, NotFoundError
from tender_review.shared.health import (
    ReadinessCheck,
    ReadinessResult,
    StaticReadinessCheck,
)


class FakeS3Error(Exception):
    def __init__(self, code: str):
        super().__init__(code)
        self.code = code


class FakeResponse(io.BytesIO):
    def release_conn(self) -> None:
        return None


class FakeMinioClient:
    def __init__(self):
        self.buckets: set[str] = set()
        self.objects: dict[tuple[str, str], dict[str, object]] = {}

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
        )

    def put_object(
        self,
        bucket: str,
        key: str,
        stream,
        length: int,
        *,
        content_type: str,
        metadata: dict[str, str],
    ) -> None:
        content = stream.read()
        if len(content) != length:
            raise AssertionError("length mismatch")
        self.objects[(bucket, key)] = {
            "content": content,
            "content_type": content_type,
            "metadata": metadata,
        }

    def get_object(self, bucket: str, key: str) -> FakeResponse:
        try:
            value = self.objects[(bucket, key)]
        except KeyError as exc:
            raise FakeS3Error("NoSuchKey") from exc
        return FakeResponse(value["content"])

    def remove_object(self, bucket: str, key: str) -> None:
        self.objects.pop((bucket, key), None)


def build_real_store() -> MinioArtifactStore:
    client = FakeMinioClient()
    store = MinioArtifactStore(
        endpoint="minio.invalid:9000",
        access_key="access",
        secret_key="secret",
        bucket="artifacts",
        client=client,
    )
    store.ensure_bucket()
    return store


class SharedAdapterContractTests(unittest.TestCase):
    def test_fake_and_real_artifact_stores_share_the_same_contract(self):
        stores: tuple[tuple[str, ArtifactStore], ...] = (
            ("fake", InMemoryArtifactStore()),
            ("minio", build_real_store()),
        )
        for name, store in stores:
            with self.subTest(adapter=name):
                artifact = ArtifactWrite(
                    key="logical-hint",
                    content=b"contract payload",
                    media_type="application/octet-stream",
                )
                first = store.put(artifact)
                second = store.put(artifact)
                self.assertEqual(first, second)
                self.assertTrue(first.key.startswith("sha256/"))
                self.assertTrue(store.exists(first.key))
                self.assertEqual(store.get(first.key), artifact.content)
                store.delete(first.key)
                self.assertFalse(store.exists(first.key))
                with self.assertRaises(NotFoundError):
                    store.get(first.key)

    def test_fake_and_real_artifact_stores_honor_cancellation(self):
        cancelled = CallContext(call_id="cancelled", cancelled=True)
        for name, store in (
            ("fake", InMemoryArtifactStore()),
            ("minio", build_real_store()),
        ):
            with self.subTest(adapter=name), self.assertRaises(CancelledError):
                store.put(
                    ArtifactWrite(key="ignored", content=b"payload"),
                    call=cancelled,
                )

    def test_fake_and_real_readiness_adapters_share_the_same_contract(self):
        engine = create_engine("sqlite:///:memory:")
        minio = build_real_store()
        checks = (
            StaticReadinessCheck("fake"),
            DatabaseHealthAdapter(engine),
            minio,
        )
        try:
            for check in checks:
                with self.subTest(adapter=check.name):
                    self.assertIsInstance(check, ReadinessCheck)
                    result = check.check()
                    self.assertIsInstance(result, ReadinessResult)
                    self.assertTrue(result.ready)
                    self.assertEqual(result.name, check.name)
        finally:
            engine.dispose()


if __name__ == "__main__":
    unittest.main()
