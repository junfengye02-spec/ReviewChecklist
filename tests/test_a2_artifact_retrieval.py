from __future__ import annotations

import hashlib
import json
import unittest
from collections.abc import Mapping, Sequence
from typing import Any

from tender_review.documents import InMemoryArtifactStore
from tender_review.documents.models import ArtifactWrite
from tender_review.retrieval import (
    ArtifactBackedHybridRetriever,
    FakeEmbeddingProvider,
    RetrievalIndexLoadError,
    RetrievalIndexLoader,
    Retriever,
    SearchRequest,
)
from tender_review.shared.contracts import CallContext
from tender_review.shared.errors import CancelledError


CALL = CallContext(call_id="a2-artifact-retrieval")
CHUNK_CONFIG = {
    "strategy_name": "structural-pdf-chunker",
    "strategy_version": "1",
    "config_sha256": "1" * 64,
}
EMBEDDING_MODEL = "embedding-v1"


def _chunk(
    chunk_id: str,
    document_id: str,
    text: str,
    *,
    page: int,
    section: str,
) -> dict[str, Any]:
    return {
        "chunk_id": chunk_id,
        "document_id": document_id,
        "page_start": page,
        "page_end": page,
        "section_path": [section],
        "text": text,
        "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
    }


CHUNKS = (
    _chunk("a", "document-1", "bond bond bond", page=2, section="Bond"),
    _chunk("b", "document-1", "bond", page=4, section="Security"),
    _chunk("c", "document-2", "schedule", page=7, section="Schedule"),
)
VECTORS = (
    {"chunk_id": "a", "document_id": "document-1", "values": [0.0, 1.0]},
    {"chunk_id": "b", "document_id": "document-1", "values": [1.0, 0.0]},
    {"chunk_id": "c", "document_id": "document-2", "values": [0.8, 0.2]},
)


def _encoded(payload: Mapping[str, Any]) -> bytes:
    return json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _put_bytes(store: InMemoryArtifactStore, content: bytes) -> dict[str, Any]:
    reference = store.put(
        ArtifactWrite(
            key="content-addressed-by-store",
            content=content,
            media_type="application/json",
        )
    )
    return {
        "key": reference.key,
        "sha256": reference.sha256,
        "size_bytes": reference.size_bytes,
        "media_type": reference.media_type,
    }


def _put_json(
    store: InMemoryArtifactStore, payload: Mapping[str, Any]
) -> dict[str, Any]:
    return _put_bytes(store, _encoded(payload))


def _catalog_payload(
    *,
    chunks: Sequence[Mapping[str, Any]] = CHUNKS,
    chunk_config: Mapping[str, Any] = CHUNK_CONFIG,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "artifact_type": "retrieval_chunk_catalog",
        "format_version": "retrieval-chunk-catalog-v1",
        "chunk_config": dict(chunk_config),
        "chunk_count": len(chunks),
        "chunks": list(chunks),
    }


def _vector_payload(
    catalog_sha256: str,
    *,
    vectors: Sequence[Mapping[str, Any]] = VECTORS,
    dimensions: int = 2,
    model: str = EMBEDDING_MODEL,
    chunk_config: Mapping[str, Any] = CHUNK_CONFIG,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "artifact_type": "retrieval_vector_index",
        "format_version": "retrieval-vector-index-v1",
        "chunk_catalog_sha256": catalog_sha256,
        "chunk_config": dict(chunk_config),
        "embedding_model": model,
        "dimensions": dimensions,
        "vector_count": len(vectors),
        "vectors": list(vectors),
    }


def _manifest_payload(
    catalog_reference: Mapping[str, Any],
    vector_reference: Mapping[str, Any],
    *,
    dimensions: int = 2,
    model: str = EMBEDDING_MODEL,
    top_k: int = 2,
    chunk_config: Mapping[str, Any] = CHUNK_CONFIG,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "artifact_type": "hybrid_retrieval_index",
        "format_version": "retrieval-index-manifest-v1",
        "retriever_version": "artifact-backed-hybrid-v1",
        "chunk_catalog": dict(catalog_reference),
        "vector_index": dict(vector_reference),
        "chunk_config": dict(chunk_config),
        "embedding_model": model,
        "embedding_dimensions": dimensions,
        "top_k": top_k,
        "candidate_limit": 3,
        "bm25": {
            "kind": "bm25",
            "k1": 1.2,
            "b": 0.75,
            "domain_terms": [],
        },
        "rrf": {"k": 60},
        "status": "provisional",
        "claims_allowed": False,
    }


def _build_artifacts(
    store: InMemoryArtifactStore,
    *,
    chunks: Sequence[Mapping[str, Any]] = CHUNKS,
    vectors: Sequence[Mapping[str, Any]] = VECTORS,
    vector_dimensions: int = 2,
    manifest_dimensions: int = 2,
    vector_model: str = EMBEDDING_MODEL,
    manifest_model: str = EMBEDDING_MODEL,
    catalog_chunk_config: Mapping[str, Any] = CHUNK_CONFIG,
    vector_chunk_config: Mapping[str, Any] = CHUNK_CONFIG,
    manifest_chunk_config: Mapping[str, Any] = CHUNK_CONFIG,
) -> tuple[str, dict[str, Any], dict[str, Any]]:
    catalog_reference = _put_json(
        store,
        _catalog_payload(chunks=chunks, chunk_config=catalog_chunk_config),
    )
    vector_reference = _put_json(
        store,
        _vector_payload(
            catalog_reference["sha256"],
            vectors=vectors,
            dimensions=vector_dimensions,
            model=vector_model,
            chunk_config=vector_chunk_config,
        ),
    )
    manifest_reference = _put_json(
        store,
        _manifest_payload(
            catalog_reference,
            vector_reference,
            dimensions=manifest_dimensions,
            model=manifest_model,
            chunk_config=manifest_chunk_config,
        ),
    )
    return manifest_reference["key"], catalog_reference, vector_reference


class _OverridingStore:
    def __init__(
        self,
        delegate: InMemoryArtifactStore,
        overrides: Mapping[str, bytes] | None = None,
    ) -> None:
        self.delegate = delegate
        self.overrides = dict(overrides or {})
        self.get_calls: list[tuple[str, CallContext | None]] = []

    def get(self, key: str, *, call: CallContext | None = None) -> bytes:
        self.get_calls.append((key, call))
        if key in self.overrides:
            return self.overrides[key]
        return self.delegate.get(key, call=call)


class ArtifactBackedHybridRetrieverTests(unittest.TestCase):
    def setUp(self) -> None:
        self.store = InMemoryArtifactStore()
        self.manifest_key, self.catalog_reference, self.vector_reference = (
            _build_artifacts(self.store)
        )
        self.provider = FakeEmbeddingProvider(
            dimensions=2,
            model=EMBEDDING_MODEL,
            vectors={"bond": (1.0, 0.0)},
        )

    def _retriever(self) -> ArtifactBackedHybridRetriever:
        return ArtifactBackedHybridRetriever.from_manifest(
            artifact_store=self.store,
            manifest_key=self.manifest_key,
            embedding_provider=self.provider,
            call=CALL,
        )

    def test_bm25_vector_and_rrf_ordering_use_the_loaded_index(self) -> None:
        retriever = self._retriever()
        request = SearchRequest(query="bond", limit=3, call=CALL)

        bm25 = retriever.bm25.search(request)
        vector = retriever.vector.search(request)
        hybrid = retriever.search(request)

        self.assertEqual([hit.chunk_id for hit in bm25.hits], ["a", "b"])
        self.assertEqual([hit.chunk_id for hit in vector.hits], ["b", "c", "a"])
        self.assertEqual([hit.chunk_id for hit in hybrid.hits], ["b", "a"])
        self.assertIsInstance(retriever, Retriever)

    def test_document_filter_top_k_evidence_and_provisional_provenance(self) -> None:
        retriever = self._retriever()
        result = retriever.search(
            SearchRequest(
                query="bond",
                document_ids=("document-2",),
                limit=100,
                call=CALL,
            )
        )

        self.assertEqual([hit.chunk_id for hit in result.hits], ["c"])
        hit = result.hits[0]
        self.assertEqual(hit.document_id, "document-2")
        self.assertEqual((hit.page_start, hit.page_end), (7, 7))
        self.assertEqual(hit.section_path, ("Schedule",))
        self.assertEqual(hit.text, "schedule")
        provenance = result.provenance
        self.assertIsNotNone(provenance)
        assert provenance is not None
        self.assertEqual(provenance.status, "provisional")
        self.assertFalse(provenance.claims_allowed)
        self.assertEqual(provenance.retriever_version, "artifact-backed-hybrid-v1")
        self.assertEqual(provenance.embedding_model, EMBEDDING_MODEL)
        self.assertEqual(provenance.embedding_dimensions, 2)
        self.assertEqual(provenance.chunk_config.strategy_name, CHUNK_CONFIG["strategy_name"])
        self.assertEqual(provenance.top_k, 2)
        self.assertEqual(provenance.candidate_limit, 3)
        self.assertEqual(provenance.chunk_catalog_sha256, self.catalog_reference["sha256"])
        self.assertEqual(provenance.index_sha256, self.vector_reference["sha256"])
        self.assertGreaterEqual(provenance.latency_ms, 0)

    def test_provider_model_and_dimensions_are_pinned_to_manifest(self) -> None:
        loaded = RetrievalIndexLoader(self.store).load(self.manifest_key, call=CALL)
        for provider in (
            FakeEmbeddingProvider(dimensions=2, model="other-model"),
            FakeEmbeddingProvider(dimensions=3, model=EMBEDDING_MODEL),
        ):
            with self.subTest(provider=provider), self.assertRaisesRegex(
                RetrievalIndexLoadError, "provider"
            ):
                ArtifactBackedHybridRetriever(loaded, provider)


class RetrievalIndexIntegrityTests(unittest.TestCase):
    def test_corrupt_json_and_sha256_mismatch_are_rejected(self) -> None:
        corrupt_store = InMemoryArtifactStore()
        catalog_reference = _put_json(corrupt_store, _catalog_payload())
        corrupt_vector_reference = _put_bytes(corrupt_store, b"{")
        manifest_key = _put_json(
            corrupt_store,
            _manifest_payload(catalog_reference, corrupt_vector_reference),
        )["key"]
        with self.assertRaisesRegex(RetrievalIndexLoadError, "invalid vector index"):
            RetrievalIndexLoader(corrupt_store).load(manifest_key, call=CALL)

        hash_store = InMemoryArtifactStore()
        manifest_key, catalog_reference, _ = _build_artifacts(hash_store)
        catalog_bytes = hash_store.get(catalog_reference["key"])
        same_size_corruption = bytes((catalog_bytes[0] ^ 1,)) + catalog_bytes[1:]
        overriding = _OverridingStore(
            hash_store,
            {catalog_reference["key"]: same_size_corruption},
        )
        with self.assertRaisesRegex(RetrievalIndexLoadError, "SHA-256 mismatch"):
            RetrievalIndexLoader(overriding).load(manifest_key, call=CALL)

    def test_dimension_model_and_chunk_config_mismatches_are_rejected(self) -> None:
        cases = (
            ({"vector_dimensions": 3, "manifest_dimensions": 3}, "dimensions"),
            ({"vector_model": "other-model"}, "embedding model"),
            (
                {
                    "vector_chunk_config": {
                        **CHUNK_CONFIG,
                        "config_sha256": "2" * 64,
                    }
                },
                "chunk config",
            ),
        )
        for kwargs, message in cases:
            with self.subTest(message=message):
                store = InMemoryArtifactStore()
                manifest_key, _, _ = _build_artifacts(store, **kwargs)
                with self.assertRaisesRegex(RetrievalIndexLoadError, message):
                    RetrievalIndexLoader(store).load(manifest_key, call=CALL)

    def test_unknown_chunks_and_cross_document_vectors_are_rejected(self) -> None:
        unknown = (*VECTORS[:-1], {"chunk_id": "outside", "document_id": "document-2", "values": [0.8, 0.2]})
        cross_document = (
            VECTORS[0],
            {**VECTORS[1], "document_id": "document-2"},
            VECTORS[2],
        )
        for vectors, message in (
            (unknown, "chunk IDs"),
            (cross_document, "outside its catalog chunk"),
        ):
            with self.subTest(message=message):
                store = InMemoryArtifactStore()
                manifest_key, _, _ = _build_artifacts(store, vectors=vectors)
                with self.assertRaisesRegex(RetrievalIndexLoadError, message):
                    RetrievalIndexLoader(store).load(manifest_key, call=CALL)

    def test_manifest_schema_is_required_and_extra_fields_are_forbidden(self) -> None:
        store = InMemoryArtifactStore()
        catalog_reference = _put_json(store, _catalog_payload())
        vector_reference = _put_json(
            store,
            _vector_payload(catalog_reference["sha256"]),
        )
        for mutation in ("missing", "extra"):
            payload = _manifest_payload(catalog_reference, vector_reference)
            if mutation == "missing":
                payload.pop("schema_version")
            else:
                payload["unversioned_extension"] = True
            manifest_key = _put_json(store, payload)["key"]
            with self.subTest(mutation=mutation), self.assertRaisesRegex(
                RetrievalIndexLoadError, "invalid retrieval manifest"
            ):
                RetrievalIndexLoader(store).load(manifest_key, call=CALL)

    def test_content_addressed_cache_is_bounded_and_never_bypasses_call_state(self) -> None:
        store = InMemoryArtifactStore()
        manifest_key, _, _ = _build_artifacts(store)
        counting_store = _OverridingStore(store)
        loader = RetrievalIndexLoader(counting_store, max_cache_entries=1)

        first = loader.load(manifest_key, call=CALL)
        second = loader.load(manifest_key, call=CALL)

        self.assertIs(first, second)
        self.assertEqual(len(counting_store.get_calls), 3)
        with self.assertRaises(CancelledError):
            loader.load(
                manifest_key,
                call=CallContext(call_id="cancelled-cache", cancelled=True),
            )


if __name__ == "__main__":
    unittest.main()
