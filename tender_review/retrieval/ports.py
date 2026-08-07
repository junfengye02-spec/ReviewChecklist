from __future__ import annotations

from typing import Protocol, runtime_checkable

from .models import (
    EmbeddingRequest,
    EmbeddingResult,
    FusionRequest,
    FusionResult,
    SearchRequest,
    SearchResult,
)


@runtime_checkable
class EmbeddingProvider(Protocol):
    def embed(self, request: EmbeddingRequest) -> EmbeddingResult: ...


@runtime_checkable
class Retriever(Protocol):
    def search(self, request: SearchRequest) -> SearchResult: ...


@runtime_checkable
class FusionStrategy(Protocol):
    def fuse(self, request: FusionRequest) -> FusionResult: ...
