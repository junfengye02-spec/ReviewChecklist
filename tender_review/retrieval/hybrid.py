"""Hybrid retrieval orchestration over stable retriever and fusion ports."""

from __future__ import annotations

from collections.abc import Iterable

from tender_review.shared.contracts import ensure_call_active

from .models import FusionRequest, SearchRequest, SearchResult
from .ports import FusionStrategy, Retriever


class HybridRetriever:
    """Run independent candidate retrievers and fuse their ranked results."""

    name = "hybrid"

    def __init__(
        self,
        retrievers: Iterable[Retriever],
        fusion_strategy: FusionStrategy,
        *,
        candidate_limit: int = 100,
    ) -> None:
        normalized = tuple(retrievers)
        if len(normalized) < 2:
            raise ValueError("hybrid retrieval requires at least two retrievers")
        if (
            isinstance(candidate_limit, bool)
            or not isinstance(candidate_limit, int)
            or not 1 <= candidate_limit <= 100
        ):
            raise ValueError("candidate_limit must be between 1 and 100")
        self.retrievers = normalized
        self.fusion_strategy = fusion_strategy
        self.candidate_limit = candidate_limit

    def search(self, request: SearchRequest) -> SearchResult:
        ensure_call_active(request.call)
        candidate_request = request.model_copy(
            update={"limit": max(request.limit, self.candidate_limit)}
        )
        result_sets = tuple(
            retriever.search(candidate_request) for retriever in self.retrievers
        )
        result_names = [result.retriever for result in result_sets]
        if len(set(result_names)) != len(result_names):
            raise ValueError("hybrid child retrievers must have unique result names")
        fused = self.fusion_strategy.fuse(
            FusionRequest(result_sets=result_sets, limit=request.limit)
        )
        return SearchResult(
            retriever=f"{self.name}:{fused.strategy}",
            hits=fused.hits,
        )


HybridRrfRetriever = HybridRetriever
