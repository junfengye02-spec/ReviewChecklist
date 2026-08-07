"""Versioned, content-addressed artifacts for in-memory hybrid retrieval."""

from __future__ import annotations

import hashlib
import json
import math
import re
import time
from collections import OrderedDict
from collections.abc import Mapping
from dataclasses import dataclass
from threading import Lock
from types import MappingProxyType
from typing import Any, Literal, Self

from pydantic import Field, ValidationError, field_validator, model_validator

from tender_review.documents import ArtifactStore
from tender_review.shared.contracts import CallContext, ContractModel, ensure_call_active

from .bm25 import BM25Retriever
from .config import BM25Config, RRFConfig
from .fusion import RrfFusionStrategy
from .hybrid import HybridRetriever
from .models import (
    ArtifactSearchResult,
    EmbeddingRequest,
    EmbeddingResult,
    RetrievalChunkConfig,
    RetrievalDocument,
    RetrievalProvenance,
    SearchRequest,
    SearchResult,
)
from .ports import EmbeddingProvider
from .vector import VectorRetriever


RETRIEVAL_MANIFEST_VERSION = "retrieval-index-manifest-v1"
CHUNK_CATALOG_VERSION = "retrieval-chunk-catalog-v1"
VECTOR_INDEX_VERSION = "retrieval-vector-index-v1"
ARTIFACT_RETRIEVER_VERSION = "artifact-backed-hybrid-v1"

_SHA256_PATTERN = r"^[0-9a-f]{64}$"
_CONTENT_KEY = re.compile(r"^sha256/([0-9a-f]{2})/([0-9a-f]{64})$")


class RetrievalIndexLoadError(ValueError):
    """An immutable retrieval artifact failed validation."""


class RetrievalArtifactReference(ContractModel):
    key: str = Field(min_length=1, max_length=1024)
    sha256: str = Field(pattern=_SHA256_PATTERN)
    size_bytes: int = Field(ge=1)
    media_type: Literal["application/json"] = "application/json"

    @model_validator(mode="after")
    def is_content_addressed(self) -> Self:
        if _digest_from_key(self.key) != self.sha256:
            raise ValueError("artifact key does not match its declared SHA-256")
        return self


class RetrievalIndexChunk(ContractModel):
    chunk_id: str = Field(min_length=1, max_length=256)
    document_id: str = Field(min_length=1, max_length=256)
    page_start: int = Field(ge=1)
    page_end: int = Field(ge=1)
    section_path: tuple[str, ...]
    text: str = Field(min_length=1)
    text_sha256: str = Field(pattern=_SHA256_PATTERN)

    @field_validator("chunk_id", "document_id", "text")
    @classmethod
    def values_are_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("retrieval chunk values must not be blank")
        return value

    @field_validator("section_path")
    @classmethod
    def sections_are_locatable(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        if not values or any(not value.strip() for value in values):
            raise ValueError("section_path must contain locatable non-blank values")
        return values

    @model_validator(mode="after")
    def has_valid_location_and_text(self) -> Self:
        if self.page_end < self.page_start:
            raise ValueError("page_end must not precede page_start")
        actual = hashlib.sha256(self.text.encode("utf-8")).hexdigest()
        if actual != self.text_sha256:
            raise ValueError("text_sha256 does not match chunk text")
        return self

    def as_document(self) -> RetrievalDocument:
        return RetrievalDocument(
            chunk_id=self.chunk_id,
            document_id=self.document_id,
            text=self.text,
            section_path=self.section_path,
            page_start=self.page_start,
            page_end=self.page_end,
        )


class RetrievalChunkCatalog(ContractModel):
    schema_version: Literal[1]
    artifact_type: Literal["retrieval_chunk_catalog"] = "retrieval_chunk_catalog"
    format_version: Literal["retrieval-chunk-catalog-v1"] = CHUNK_CATALOG_VERSION
    chunk_config: RetrievalChunkConfig
    chunk_count: int = Field(ge=1)
    chunks: tuple[RetrievalIndexChunk, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def has_consistent_chunks(self) -> Self:
        if self.chunk_count != len(self.chunks):
            raise ValueError("chunk_count does not match chunk catalog")
        chunk_ids = tuple(chunk.chunk_id for chunk in self.chunks)
        if len(chunk_ids) != len(set(chunk_ids)):
            raise ValueError("chunk catalog contains duplicate chunk_id values")
        return self


class RetrievalVector(ContractModel):
    chunk_id: str = Field(min_length=1, max_length=256)
    document_id: str = Field(min_length=1, max_length=256)
    values: tuple[float, ...] = Field(min_length=1)

    @field_validator("chunk_id", "document_id")
    @classmethod
    def vector_identity_is_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("vector identity values must not be blank")
        return value

    @field_validator("values")
    @classmethod
    def vector_is_finite_and_non_zero(
        cls, values: tuple[float, ...]
    ) -> tuple[float, ...]:
        if any(not math.isfinite(value) for value in values):
            raise ValueError("vectors must contain only finite values")
        if not any(value != 0 for value in values):
            raise ValueError("vectors must have non-zero norms")
        return values


class RetrievalVectorIndex(ContractModel):
    schema_version: Literal[1]
    artifact_type: Literal["retrieval_vector_index"] = "retrieval_vector_index"
    format_version: Literal["retrieval-vector-index-v1"] = VECTOR_INDEX_VERSION
    chunk_catalog_sha256: str = Field(pattern=_SHA256_PATTERN)
    chunk_config: RetrievalChunkConfig
    embedding_model: str = Field(min_length=1, max_length=256)
    dimensions: int = Field(ge=1)
    vector_count: int = Field(ge=1)
    vectors: tuple[RetrievalVector, ...] = Field(min_length=1)

    @field_validator("embedding_model")
    @classmethod
    def model_is_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("embedding_model must not be blank")
        return value

    @model_validator(mode="after")
    def has_consistent_vectors(self) -> Self:
        if self.vector_count != len(self.vectors):
            raise ValueError("vector_count does not match vector index")
        chunk_ids = tuple(vector.chunk_id for vector in self.vectors)
        if len(chunk_ids) != len(set(chunk_ids)):
            raise ValueError("vector index contains duplicate chunk_id values")
        if any(len(vector.values) != self.dimensions for vector in self.vectors):
            raise ValueError("vector dimensions do not match index dimensions")
        return self


class RetrievalIndexManifest(ContractModel):
    schema_version: Literal[1]
    artifact_type: Literal["hybrid_retrieval_index"] = "hybrid_retrieval_index"
    format_version: Literal["retrieval-index-manifest-v1"] = (
        RETRIEVAL_MANIFEST_VERSION
    )
    retriever_version: Literal["artifact-backed-hybrid-v1"] = (
        ARTIFACT_RETRIEVER_VERSION
    )
    chunk_catalog: RetrievalArtifactReference
    vector_index: RetrievalArtifactReference
    chunk_config: RetrievalChunkConfig
    embedding_model: str = Field(min_length=1, max_length=256)
    embedding_dimensions: int = Field(ge=1)
    top_k: int = Field(default=10, ge=1, le=100)
    candidate_limit: int = Field(default=100, ge=1, le=100)
    bm25: BM25Config = Field(default_factory=BM25Config)
    rrf: RRFConfig = Field(default_factory=RRFConfig)
    status: Literal["provisional"] = "provisional"
    claims_allowed: Literal[False] = False

    @field_validator("embedding_model")
    @classmethod
    def embedding_model_is_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("embedding_model must not be blank")
        return value

    @model_validator(mode="after")
    def candidate_pool_covers_top_k(self) -> Self:
        if self.candidate_limit < self.top_k:
            raise ValueError("candidate_limit must not be smaller than top_k")
        return self


@dataclass(frozen=True, slots=True)
class LoadedRetrievalIndex:
    manifest_key: str
    manifest_sha256: str
    manifest: RetrievalIndexManifest
    documents: tuple[RetrievalDocument, ...]
    vectors: Mapping[str, tuple[float, ...]]


class RetrievalIndexLoader:
    """Load and validate an immutable index, with a bounded key-only cache."""

    def __init__(
        self,
        artifact_store: ArtifactStore,
        *,
        max_cache_entries: int = 4,
        max_manifest_bytes: int = 1 * 1024 * 1024,
        max_catalog_bytes: int = 128 * 1024 * 1024,
        max_vector_index_bytes: int = 512 * 1024 * 1024,
        max_chunks: int = 50_000,
        max_dimensions: int = 8_192,
    ) -> None:
        for name, value in (
            ("max_cache_entries", max_cache_entries),
            ("max_manifest_bytes", max_manifest_bytes),
            ("max_catalog_bytes", max_catalog_bytes),
            ("max_vector_index_bytes", max_vector_index_bytes),
            ("max_chunks", max_chunks),
            ("max_dimensions", max_dimensions),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if min(
            max_manifest_bytes,
            max_catalog_bytes,
            max_vector_index_bytes,
            max_chunks,
            max_dimensions,
        ) < 1:
            raise ValueError("artifact and index bounds must be positive")
        self.artifact_store = artifact_store
        self.max_cache_entries = max_cache_entries
        self.max_manifest_bytes = max_manifest_bytes
        self.max_catalog_bytes = max_catalog_bytes
        self.max_vector_index_bytes = max_vector_index_bytes
        self.max_chunks = max_chunks
        self.max_dimensions = max_dimensions
        self._cache: OrderedDict[str, LoadedRetrievalIndex] = OrderedDict()
        self._cache_lock = Lock()

    def load(
        self,
        manifest_key: str,
        *,
        call: CallContext | None = None,
    ) -> LoadedRetrievalIndex:
        if call is not None:
            ensure_call_active(call)
        manifest_sha256 = _digest_from_key(manifest_key)
        cached = self._get_cached(manifest_key)
        if cached is not None:
            return cached

        manifest_bytes = self._read(
            manifest_key,
            expected_sha256=manifest_sha256,
            expected_size=None,
            max_bytes=self.max_manifest_bytes,
            call=call,
        )
        manifest = _parse_model(
            manifest_bytes,
            RetrievalIndexManifest,
            artifact_name="retrieval manifest",
        )
        catalog_bytes = self._read_reference(
            manifest.chunk_catalog,
            max_bytes=self.max_catalog_bytes,
            call=call,
        )
        catalog = _parse_model(
            catalog_bytes,
            RetrievalChunkCatalog,
            artifact_name="chunk catalog",
        )
        vector_bytes = self._read_reference(
            manifest.vector_index,
            max_bytes=self.max_vector_index_bytes,
            call=call,
        )
        vector_index = _parse_model(
            vector_bytes,
            RetrievalVectorIndex,
            artifact_name="vector index",
        )
        self._validate_index(manifest, catalog, vector_index)

        documents = tuple(chunk.as_document() for chunk in catalog.chunks)
        vectors = MappingProxyType(
            {vector.chunk_id: vector.values for vector in vector_index.vectors}
        )
        loaded = LoadedRetrievalIndex(
            manifest_key=manifest_key,
            manifest_sha256=manifest_sha256,
            manifest=manifest,
            documents=documents,
            vectors=vectors,
        )
        self._put_cached(manifest_key, loaded)
        return loaded

    def clear_cache(self) -> None:
        with self._cache_lock:
            self._cache.clear()

    def _read_reference(
        self,
        reference: RetrievalArtifactReference,
        *,
        max_bytes: int,
        call: CallContext | None,
    ) -> bytes:
        return self._read(
            reference.key,
            expected_sha256=reference.sha256,
            expected_size=reference.size_bytes,
            max_bytes=max_bytes,
            call=call,
        )

    def _read(
        self,
        key: str,
        *,
        expected_sha256: str,
        expected_size: int | None,
        max_bytes: int,
        call: CallContext | None,
    ) -> bytes:
        if _digest_from_key(key) != expected_sha256:
            raise RetrievalIndexLoadError(
                f"artifact key does not match expected SHA-256: {key}"
            )
        content = self.artifact_store.get(key, call=call)
        if not isinstance(content, bytes):
            raise RetrievalIndexLoadError("ArtifactStore.get must return bytes")
        if len(content) > max_bytes:
            raise RetrievalIndexLoadError(f"retrieval artifact exceeds size bound: {key}")
        if expected_size is not None and len(content) != expected_size:
            raise RetrievalIndexLoadError(f"artifact size mismatch: {key}")
        actual_sha256 = hashlib.sha256(content).hexdigest()
        if actual_sha256 != expected_sha256:
            raise RetrievalIndexLoadError(f"artifact SHA-256 mismatch: {key}")
        return content

    def _validate_index(
        self,
        manifest: RetrievalIndexManifest,
        catalog: RetrievalChunkCatalog,
        vector_index: RetrievalVectorIndex,
    ) -> None:
        if catalog.chunk_count > self.max_chunks:
            raise RetrievalIndexLoadError("chunk catalog exceeds configured bound")
        if vector_index.dimensions > self.max_dimensions:
            raise RetrievalIndexLoadError("vector dimensions exceed configured bound")
        if catalog.chunk_config != manifest.chunk_config:
            raise RetrievalIndexLoadError("chunk catalog config does not match manifest")
        if vector_index.chunk_config != manifest.chunk_config:
            raise RetrievalIndexLoadError("vector index chunk config does not match manifest")
        if vector_index.chunk_catalog_sha256 != manifest.chunk_catalog.sha256:
            raise RetrievalIndexLoadError("vector index targets another chunk catalog")
        if vector_index.embedding_model != manifest.embedding_model:
            raise RetrievalIndexLoadError("embedding model does not match manifest")
        if vector_index.dimensions != manifest.embedding_dimensions:
            raise RetrievalIndexLoadError("embedding dimensions do not match manifest")
        if vector_index.vector_count != catalog.chunk_count:
            raise RetrievalIndexLoadError("vector count does not match chunk catalog")

        chunks = {chunk.chunk_id: chunk for chunk in catalog.chunks}
        indexed_ids = {vector.chunk_id for vector in vector_index.vectors}
        if indexed_ids != set(chunks):
            raise RetrievalIndexLoadError(
                "vector index chunk IDs do not match chunk catalog"
            )
        for vector in vector_index.vectors:
            if chunks[vector.chunk_id].document_id != vector.document_id:
                raise RetrievalIndexLoadError(
                    f"vector document is outside its catalog chunk: {vector.chunk_id}"
                )

    def _get_cached(self, key: str) -> LoadedRetrievalIndex | None:
        with self._cache_lock:
            cached = self._cache.pop(key, None)
            if cached is not None:
                self._cache[key] = cached
            return cached

    def _put_cached(self, key: str, index: LoadedRetrievalIndex) -> None:
        if self.max_cache_entries == 0:
            return
        with self._cache_lock:
            self._cache[key] = index
            self._cache.move_to_end(key)
            while len(self._cache) > self.max_cache_entries:
                self._cache.popitem(last=False)


class ArtifactBackedHybridRetriever:
    """BM25 + dense vector + RRF over one validated immutable artifact index."""

    name = "artifact-hybrid:rrf"

    def __init__(
        self,
        index: LoadedRetrievalIndex,
        embedding_provider: EmbeddingProvider,
    ) -> None:
        self.index = index
        self.embedding_provider = _PinnedEmbeddingProvider(
            embedding_provider,
            model=index.manifest.embedding_model,
            dimensions=index.manifest.embedding_dimensions,
        )
        self._validate_provider_configuration(embedding_provider)
        self.bm25 = BM25Retriever(
            index.documents,
            k1=index.manifest.bm25.k1,
            b=index.manifest.bm25.b,
            domain_terms=index.manifest.bm25.domain_terms,
        )
        self.vector = VectorRetriever(
            index.documents,
            self.embedding_provider,
            index.vectors,
        )
        self.hybrid = HybridRetriever(
            (self.bm25, self.vector),
            RrfFusionStrategy(k=index.manifest.rrf.k),
            candidate_limit=index.manifest.candidate_limit,
        )

    @classmethod
    def from_manifest(
        cls,
        *,
        artifact_store: ArtifactStore,
        manifest_key: str,
        embedding_provider: EmbeddingProvider,
        call: CallContext | None = None,
        loader: RetrievalIndexLoader | None = None,
    ) -> "ArtifactBackedHybridRetriever":
        active_loader = loader or RetrievalIndexLoader(artifact_store)
        if active_loader.artifact_store is not artifact_store:
            raise ValueError("loader and artifact_store must reference the same store")
        return cls(
            active_loader.load(manifest_key, call=call),
            embedding_provider,
        )

    @property
    def version(self) -> str:
        return self.index.manifest.retriever_version

    def search(self, request: SearchRequest) -> SearchResult:
        ensure_call_active(request.call)
        started_ns = time.perf_counter_ns()
        top_k = min(request.limit, self.index.manifest.top_k)
        result = self.hybrid.search(request.model_copy(update={"limit": top_k}))
        latency_ms = (time.perf_counter_ns() - started_ns) / 1_000_000
        manifest = self.index.manifest
        return ArtifactSearchResult(
            retriever=self.name,
            hits=result.hits,
            provenance=RetrievalProvenance(
                retriever_version=manifest.retriever_version,
                embedding_model=manifest.embedding_model,
                embedding_dimensions=manifest.embedding_dimensions,
                chunk_config=manifest.chunk_config,
                top_k=top_k,
                candidate_limit=manifest.candidate_limit,
                manifest_sha256=self.index.manifest_sha256,
                chunk_catalog_sha256=manifest.chunk_catalog.sha256,
                index_sha256=manifest.vector_index.sha256,
                latency_ms=latency_ms,
            ),
        )

    def _validate_provider_configuration(
        self, embedding_provider: EmbeddingProvider
    ) -> None:
        manifest = self.index.manifest
        provider_model = getattr(embedding_provider, "model", None)
        if provider_model is not None and provider_model != manifest.embedding_model:
            raise RetrievalIndexLoadError(
                "embedding provider model does not match retrieval index"
            )
        provider_dimensions = getattr(embedding_provider, "dimensions", None)
        if (
            provider_dimensions is not None
            and provider_dimensions != manifest.embedding_dimensions
        ):
            raise RetrievalIndexLoadError(
                "embedding provider dimensions do not match retrieval index"
            )


class _PinnedEmbeddingProvider:
    def __init__(
        self,
        provider: EmbeddingProvider,
        *,
        model: str,
        dimensions: int,
    ) -> None:
        self.provider = provider
        self.model = model
        self.dimensions = dimensions

    def embed(self, request: EmbeddingRequest) -> EmbeddingResult:
        result = self.provider.embed(request)
        if not isinstance(result, EmbeddingResult):
            raise TypeError("embedding provider must return EmbeddingResult")
        if result.model != self.model:
            raise RetrievalIndexLoadError(
                "embedding response model does not match retrieval index"
            )
        if result.dimensions != self.dimensions:
            raise RetrievalIndexLoadError(
                "embedding response dimensions do not match retrieval index"
            )
        return result


def _digest_from_key(key: str) -> str:
    if not isinstance(key, str):
        raise RetrievalIndexLoadError("artifact key must be a string")
    match = _CONTENT_KEY.fullmatch(key)
    if match is None or match.group(1) != match.group(2)[:2]:
        raise RetrievalIndexLoadError(
            f"retrieval artifact key is not content-addressed: {key!r}"
        )
    return match.group(2)


def _parse_model(
    content: bytes,
    model_type: type[ContractModel],
    *,
    artifact_name: str,
) -> Any:
    try:
        payload = json.loads(
            content.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
        )
        if not isinstance(payload, dict):
            raise ValueError("artifact root must be an object")
        return model_type.model_validate(payload)
    except (UnicodeDecodeError, json.JSONDecodeError, ValidationError, ValueError) as exc:
        raise RetrievalIndexLoadError(f"invalid {artifact_name}: {exc}") from exc


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result
