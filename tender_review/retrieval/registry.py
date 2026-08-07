"""Explicit retriever registration and construction."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType

from tender_review.shared.contracts import CallContext

from .bm25 import BM25Retriever
from .config import BM25Config, HybridConfig, RetrieverConfig, VectorConfig
from .hybrid import HybridRetriever
from .ports import EmbeddingProvider, Retriever
from .fusion import RrfFusionStrategy
from .vector import VectorRetriever


VectorMap = Mapping[str, Sequence[float]] | Sequence[Sequence[float]]


@dataclass(frozen=True, slots=True)
class _BuildContext:
    config: RetrieverConfig
    documents: tuple[object, ...]
    embedding_provider: EmbeddingProvider | None
    vectors: VectorMap | None
    index_call: CallContext | None


RetrieverBuilder = Callable[[_BuildContext], Retriever]


def _build_bm25(context: _BuildContext) -> Retriever:
    if not isinstance(context.config, BM25Config):
        raise TypeError("bm25 builder requires BM25Config")
    return BM25Retriever(
        context.documents,
        k1=context.config.k1,
        b=context.config.b,
        domain_terms=context.config.domain_terms,
    )


def _build_vector(context: _BuildContext) -> Retriever:
    if not isinstance(context.config, VectorConfig):
        raise TypeError("vector builder requires VectorConfig")
    if context.embedding_provider is None:
        raise ValueError("embedding_provider is required for vector retrieval")
    if context.vectors is not None:
        return VectorRetriever(
            context.documents,
            context.embedding_provider,
            context.vectors,
        )
    if context.index_call is None:
        raise ValueError(
            "index_call is required when document vectors are not precomputed"
        )
    return VectorRetriever.from_documents(
        context.documents,
        context.embedding_provider,
        call=context.index_call,
    )


def _build_hybrid(context: _BuildContext) -> Retriever:
    if not isinstance(context.config, HybridConfig):
        raise TypeError("hybrid builder requires HybridConfig")
    bm25 = _build_bm25(
        _BuildContext(
            config=context.config.bm25,
            documents=context.documents,
            embedding_provider=None,
            vectors=None,
            index_call=None,
        )
    )
    vector = _build_vector(
        _BuildContext(
            config=context.config.vector,
            documents=context.documents,
            embedding_provider=context.embedding_provider,
            vectors=context.vectors,
            index_call=context.index_call,
        )
    )
    return HybridRetriever(
        (bm25, vector),
        RrfFusionStrategy(k=context.config.fusion.k),
        candidate_limit=context.config.candidate_limit,
    )


_RETRIEVER_BUILDERS: Mapping[str, RetrieverBuilder] = MappingProxyType(
    {
        "bm25": _build_bm25,
        "vector": _build_vector,
        "hybrid": _build_hybrid,
    }
)


def registered_retriever_kinds() -> tuple[str, ...]:
    """Return the stable, explicitly wired retriever names."""

    return tuple(_RETRIEVER_BUILDERS)


def build_retriever(
    config: RetrieverConfig,
    documents: Iterable[object],
    *,
    embedding_provider: EmbeddingProvider | None = None,
    vectors: VectorMap | None = None,
    index_call: CallContext | None = None,
) -> Retriever:
    """Build one registered strategy without dynamic imports or global state."""

    if not isinstance(config, (BM25Config, VectorConfig, HybridConfig)):
        raise TypeError("config must be a validated retriever configuration")
    builder = _RETRIEVER_BUILDERS[config.kind]
    return builder(
        _BuildContext(
            config=config,
            documents=tuple(documents),
            embedding_provider=embedding_provider,
            vectors=vectors,
            index_call=index_call,
        )
    )
