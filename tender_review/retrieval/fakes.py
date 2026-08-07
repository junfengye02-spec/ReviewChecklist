from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping, Sequence

from tender_review.shared.contracts import ensure_call_active

from .models import (
    EmbeddingRequest,
    EmbeddingResult,
    FusionRequest,
    FusionResult,
    SearchHit,
    SearchRequest,
    SearchResult,
)


class FakeEmbeddingProvider:
    def __init__(
        self,
        dimensions: int = 4,
        *,
        vectors: Mapping[str, Sequence[float]] | None = None,
        model: str = "fake-embedding",
    ) -> None:
        if dimensions < 1:
            raise ValueError("dimensions must be positive")
        if not model.strip():
            raise ValueError("model must not be blank")
        self.dimensions = dimensions
        self.model = model
        self.vectors = {
            text: tuple(float(value) for value in vector)
            for text, vector in (vectors or {}).items()
        }
        if any(len(vector) != dimensions for vector in self.vectors.values()):
            raise ValueError("configured fake vectors must match dimensions")
        if any(
            not math.isfinite(value)
            for vector in self.vectors.values()
            for value in vector
        ):
            raise ValueError("configured fake vectors must contain finite values")
        self.calls: list[EmbeddingRequest] = []

    def embed(self, request: EmbeddingRequest) -> EmbeddingResult:
        ensure_call_active(request.call)
        self.calls.append(request)
        vectors = tuple(self._vector(text) for text in request.texts)
        return EmbeddingResult(
            model=self.model,
            dimensions=self.dimensions,
            vectors=vectors,
        )

    def _vector(self, text: str) -> tuple[float, ...]:
        configured = self.vectors.get(text)
        if configured is not None:
            return configured
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        return tuple(
            digest[index % len(digest)] / 255.0 for index in range(self.dimensions)
        )


class FakeRetriever:
    def __init__(self, hits: tuple[SearchHit, ...] = ()) -> None:
        self.hits = hits
        self.calls: list[SearchRequest] = []

    def search(self, request: SearchRequest) -> SearchResult:
        ensure_call_active(request.call)
        self.calls.append(request)
        allowed = set(request.document_ids)
        hits = [hit for hit in self.hits if not allowed or hit.document_id in allowed][
            : request.limit
        ]
        normalized = tuple(
            hit.model_copy(update={"rank": index})
            for index, hit in enumerate(hits, start=1)
        )
        return SearchResult(retriever="fake-retriever", hits=normalized)


class FakeFusionStrategy:
    name = "fake-rrf"

    def fuse(self, request: FusionRequest) -> FusionResult:
        by_chunk: dict[str, tuple[float, SearchHit]] = {}
        for result in request.result_sets:
            for hit in result.hits:
                fused_score = 1.0 / (60 + hit.rank)
                current_score, _ = by_chunk.get(hit.chunk_id, (0.0, hit))
                by_chunk[hit.chunk_id] = (current_score + fused_score, hit)
        ordered = sorted(
            by_chunk.values(), key=lambda item: (-item[0], item[1].chunk_id)
        )
        hits = tuple(
            hit.model_copy(update={"score": score, "source": self.name, "rank": rank})
            for rank, (score, hit) in enumerate(ordered[: request.limit], start=1)
        )
        return FusionResult(strategy=self.name, hits=hits)
