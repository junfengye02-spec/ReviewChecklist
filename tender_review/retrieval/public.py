"""Stable cross-module retrieval API."""

from .artifacts import (
    ARTIFACT_RETRIEVER_VERSION,
    CHUNK_CATALOG_VERSION,
    RETRIEVAL_MANIFEST_VERSION,
    VECTOR_INDEX_VERSION,
    ArtifactBackedHybridRetriever,
    LoadedRetrievalIndex,
    RetrievalArtifactReference,
    RetrievalChunkCatalog,
    RetrievalIndexChunk,
    RetrievalIndexLoadError,
    RetrievalIndexLoader,
    RetrievalIndexManifest,
    RetrievalVector,
    RetrievalVectorIndex,
)
from .bm25 import BM25Retriever, Bm25Retriever
from .config import BM25Config, HybridConfig, RRFConfig, RetrieverConfig, VectorConfig
from .fusion import RRFFusionStrategy, RrfFusionStrategy
from .hybrid import HybridRetriever, HybridRrfRetriever
from .models import (
    ArtifactSearchResult,
    EmbeddingRequest,
    EmbeddingResult,
    FusionRequest,
    FusionResult,
    RetrievalChunkConfig,
    RetrievalDocument,
    RetrievalProvenance,
    SearchDocument,
    SearchHit,
    SearchRequest,
    SearchResult,
)
from .ports import EmbeddingProvider, FusionStrategy, Retriever
from .registry import build_retriever, registered_retriever_kinds
from .tokenization import tokenize
from .vector import DenseVectorRetriever, VectorRetriever

__all__ = [
    "ARTIFACT_RETRIEVER_VERSION",
    "ArtifactBackedHybridRetriever",
    "ArtifactSearchResult",
    "BM25Retriever",
    "BM25Config",
    "Bm25Retriever",
    "CHUNK_CATALOG_VERSION",
    "DenseVectorRetriever",
    "EmbeddingProvider",
    "EmbeddingRequest",
    "EmbeddingResult",
    "FusionRequest",
    "FusionResult",
    "FusionStrategy",
    "HybridConfig",
    "HybridRetriever",
    "HybridRrfRetriever",
    "LoadedRetrievalIndex",
    "RETRIEVAL_MANIFEST_VERSION",
    "RRFConfig",
    "RRFFusionStrategy",
    "RetrievalArtifactReference",
    "RetrievalChunkCatalog",
    "RetrievalChunkConfig",
    "RetrievalDocument",
    "RetrievalIndexChunk",
    "RetrievalIndexLoadError",
    "RetrievalIndexLoader",
    "RetrievalIndexManifest",
    "RetrievalProvenance",
    "Retriever",
    "RetrieverConfig",
    "RrfFusionStrategy",
    "SearchDocument",
    "SearchHit",
    "SearchRequest",
    "SearchResult",
    "VectorConfig",
    "VECTOR_INDEX_VERSION",
    "VectorRetriever",
    "RetrievalVector",
    "RetrievalVectorIndex",
    "build_retriever",
    "registered_retriever_kinds",
    "tokenize",
]
