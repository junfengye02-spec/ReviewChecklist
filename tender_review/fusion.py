"""Rank fusion strategies for independent retrievers."""

from __future__ import annotations

from collections import defaultdict

from .models import FusionRequest, FusionResult, SearchHit


class RrfFusionStrategy:
    """Reciprocal Rank Fusion with deterministic ranking and provenance."""

    name = "rrf"

    def __init__(self, *, k: int = 60) -> None:
        if k < 1:
            raise ValueError("k must be positive")
        self.k = k

    def fuse(self, request: FusionRequest) -> FusionResult:
        scores: defaultdict[str, float] = defaultdict(float)
        representatives: dict[str, SearchHit] = {}
        sources: defaultdict[str, set[str]] = defaultdict(set)
        for result in request.result_sets:
            for position, hit in enumerate(result.hits, start=1):
                scores[hit.chunk_id] += 1.0 / (self.k + position)
                representatives.setdefault(hit.chunk_id, hit)
                sources[hit.chunk_id].add(hit.source or result.retriever)
        ordered = sorted(
            scores,
            key=lambda chunk_id: (-scores[chunk_id], chunk_id),
        )
        hits: list[SearchHit] = []
        for rank, chunk_id in enumerate(ordered[: request.limit], start=1):
            representative = representatives[chunk_id]
            provenance = "+".join(sorted(sources[chunk_id]))
            hits.append(
                representative.model_copy(
                    update={
                        "score": scores[chunk_id],
                        "source": f"{self.name}:{provenance}",
                        "rank": rank,
                    }
                )
            )
        return FusionResult(strategy=self.name, hits=tuple(hits))


RRFFusionStrategy = RrfFusionStrategy
