"""Dense cosine-similarity retriever backed by an EmbeddingProvider."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence

from tender_review.shared.contracts import ensure_call_active

from ._common import IndexedDocument, coerce_documents, vector_for
from .models import EmbeddingRequest, SearchHit, SearchRequest, SearchResult
from .ports import EmbeddingProvider


class VectorRetriever:
    """Search precomputed chunk vectors with cosine similarity."""

    name = "vector"

    def __init__(
        self,
        documents: Iterable[object] = (),
        embedding_provider: EmbeddingProvider | None = None,
        vectors: Mapping[str, Sequence[float]] | Sequence[Sequence[float]] | None = None,
        *,
        provider: EmbeddingProvider | None = None,
    ) -> None:
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

    def search(self, request: SearchRequest) -> SearchResult:
        ensure_call_active(request.call)
        result = self.embedding_provider.embed(
            EmbeddingRequest(texts=(request.query,), call=request.call)
        )
        if len(result.vectors) != 1:
            raise ValueError("embedding provider must return one query vector")
        query_vector = tuple(float(value) for value in result.vectors[0])
        if len(query_vector) != self.dimensions or result.dimensions != self.dimensions:
            raise ValueError("query vector dimensions do not match the index")
        query_norm = math.sqrt(sum(value * value for value in query_vector))
        allowed = set(request.document_ids)
        ranked: list[tuple[float, IndexedDocument]] = []
        for document in self.documents:
            if allowed and document.document_id not in allowed:
                continue
            document_norm = self._norms[document.chunk_id]
            score = (
                sum(
                    left * right
                    for left, right in zip(
                        query_vector,
                        self.vectors[document.chunk_id],
                        strict=True,
                    )
                )
                / (query_norm * document_norm)
                if query_norm and document_norm
                else 0.0
            )
            ranked.append((score, document))
        ranked.sort(key=lambda item: (-item[0], item[1].chunk_id))
        hits = tuple(
            SearchHit(
                chunk_id=document.chunk_id,
                document_id=document.document_id,
                text=document.text,
                score=score,
                source=self.name,
                rank=rank,
            )
            for rank, (score, document) in enumerate(ranked[: request.limit], start=1)
        )
        return SearchResult(retriever=self.name, hits=hits)


DenseVectorRetriever = VectorRetriever
