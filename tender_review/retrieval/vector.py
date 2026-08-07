"""Dense cosine-similarity retriever backed by an EmbeddingProvider."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence

from tender_review.shared.contracts import CallContext, ensure_call_active

from ._common import IndexedDocument, coerce_documents, vector_for
from .models import (
    EmbeddingRequest,
    EmbeddingResult,
    SearchHit,
    SearchRequest,
    SearchResult,
)
from .ports import EmbeddingProvider


class VectorRetriever:
    """Search precomputed chunk vectors with cosine similarity."""

    name = "vector"

    def __init__(
        self,
        documents: Iterable[object] = (),
        embedding_provider: EmbeddingProvider | None = None,
        vectors: Mapping[str, Sequence[float]]
        | Sequence[Sequence[float]]
        | None = None,
        *,
        provider: EmbeddingProvider | None = None,
    ) -> None:
        if (
            embedding_provider is not None
            and provider is not None
            and embedding_provider is not provider
        ):
            raise ValueError("provide only one embedding provider")
        self.embedding_provider = embedding_provider or provider
        if self.embedding_provider is None:
            raise ValueError("embedding_provider is required")
        self.documents: tuple[IndexedDocument, ...] = coerce_documents(documents)
        self.vectors = vector_for(vectors, self.documents)
        self.dimensions = len(next(iter(self.vectors.values())))
        self._norms = {
            chunk_id: math.sqrt(sum(value * value for value in vector))
            for chunk_id, vector in self.vectors.items()
        }
        zero_norm_chunks = sorted(
            chunk_id for chunk_id, norm in self._norms.items() if norm == 0
        )
        if zero_norm_chunks:
            raise ValueError(
                "document vectors must have non-zero norms: "
                + ", ".join(zero_norm_chunks)
            )

    @classmethod
    def from_documents(
        cls,
        documents: Iterable[object],
        embedding_provider: EmbeddingProvider,
        *,
        call: CallContext,
    ) -> "VectorRetriever":
        if not isinstance(call, CallContext):
            raise TypeError("call must be a CallContext")
        ensure_call_active(call)
        normalized = coerce_documents(documents)
        if not normalized:
            raise ValueError("vector retriever requires at least one document")
        result = embedding_provider.embed(
            EmbeddingRequest(
                texts=tuple(document.text for document in normalized),
                call=call,
            )
        )
        vectors = _validated_vectors(result, expected_count=len(normalized))
        return cls(normalized, embedding_provider, vectors)

    def search(self, request: SearchRequest) -> SearchResult:
        ensure_call_active(request.call)
        result = self.embedding_provider.embed(
            EmbeddingRequest(texts=(request.query,), call=request.call)
        )
        query_vector = _validated_vectors(
            result,
            expected_count=1,
            expected_dimensions=self.dimensions,
        )[0]
        query_norm = math.sqrt(sum(value * value for value in query_vector))
        if query_norm == 0:
            raise ValueError("query embedding must have a non-zero norm")
        allowed = set(request.document_ids)
        ranked: list[tuple[float, IndexedDocument]] = []
        for document in self.documents:
            if allowed and document.document_id not in allowed:
                continue
            document_norm = self._norms[document.chunk_id]
            raw_score = sum(
                left * right
                for left, right in zip(
                    query_vector,
                    self.vectors[document.chunk_id],
                    strict=True,
                )
            ) / (query_norm * document_norm)
            score = max(-1.0, min(1.0, raw_score))
            ranked.append((score, document))
        ranked.sort(key=lambda item: (-item[0], item[1].chunk_id))
        hits = tuple(
            SearchHit(
                chunk_id=document.chunk_id,
                document_id=document.document_id,
                text=document.text,
                section_path=document.section_path,
                page_start=document.page_start,
                page_end=document.page_end,
                score=score,
                source=self.name,
                rank=rank,
            )
            for rank, (score, document) in enumerate(ranked[: request.limit], start=1)
        )
        return SearchResult(retriever=self.name, hits=hits)


DenseVectorRetriever = VectorRetriever


def _validated_vectors(
    result: EmbeddingResult,
    *,
    expected_count: int,
    expected_dimensions: int | None = None,
) -> tuple[tuple[float, ...], ...]:
    if not isinstance(result, EmbeddingResult):
        raise TypeError("embedding provider must return EmbeddingResult")
    if len(result.vectors) != expected_count:
        raise ValueError("embedding provider returned an unexpected number of vectors")
    if expected_dimensions is not None and result.dimensions != expected_dimensions:
        raise ValueError("embedding dimensions do not match the index")
    vectors = tuple(
        tuple(float(value) for value in vector) for vector in result.vectors
    )
    if any(len(vector) != result.dimensions for vector in vectors):
        raise ValueError("embedding vector dimensions do not match dimensions")
    if any(not math.isfinite(value) for vector in vectors for value in vector):
        raise ValueError("embedding vectors must contain only finite values")
    return vectors
