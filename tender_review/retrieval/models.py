from __future__ import annotations

import math
from typing import Literal, Self

from pydantic import Field, field_validator, model_validator

from tender_review.shared.contracts import CallContext, ContractModel


class EmbeddingRequest(ContractModel):
    texts: tuple[str, ...] = Field(min_length=1)
    call: CallContext

    @field_validator("texts")
    @classmethod
    def validate_texts(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        if any(not value.strip() for value in values):
            raise ValueError("embedding texts must not be blank")
        return values


class EmbeddingResult(ContractModel):
    model: str = Field(min_length=1)
    dimensions: int = Field(ge=1)
    vectors: tuple[tuple[float, ...], ...] = Field(min_length=1)

    @field_validator("model")
    @classmethod
    def validate_model(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("model must not be blank")
        return value

    @model_validator(mode="after")
    def validate_vectors(self) -> Self:
        if any(len(vector) != self.dimensions for vector in self.vectors):
            raise ValueError("embedding vector dimensions do not match dimensions")
        if any(not math.isfinite(value) for vector in self.vectors for value in vector):
            raise ValueError("embedding vectors must contain only finite values")
        return self


class SearchRequest(ContractModel):
    query: str = Field(min_length=1)
    document_ids: tuple[str, ...] = ()
    limit: int = Field(default=10, ge=1, le=100)
    call: CallContext

    @field_validator("query")
    @classmethod
    def validate_query(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("query must not be blank")
        return value

    @field_validator("document_ids")
    @classmethod
    def validate_document_ids(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        if any(not value.strip() for value in values):
            raise ValueError("document_ids must not contain blank values")
        if len(set(values)) != len(values):
            raise ValueError("document_ids must not contain duplicates")
        return values


class SearchHit(ContractModel):
    chunk_id: str = Field(min_length=1)
    document_id: str = Field(min_length=1)
    text: str = Field(min_length=1)
    section_path: tuple[str, ...] = ()
    page_start: int | None = Field(default=None, ge=1)
    page_end: int | None = Field(default=None, ge=1)
    score: float
    source: str = Field(min_length=1)
    rank: int = Field(ge=1)

    @field_validator("chunk_id", "document_id", "text", "source")
    @classmethod
    def validate_non_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("value must not be blank")
        return value

    @field_validator("section_path")
    @classmethod
    def validate_section_path(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        if any(not value.strip() for value in values):
            raise ValueError("section_path must not contain blank values")
        return values

    @field_validator("score")
    @classmethod
    def validate_score(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("score must be finite")
        return value

    @model_validator(mode="after")
    def validate_page_range(self) -> Self:
        if (self.page_start is None) != (self.page_end is None):
            raise ValueError("page_start and page_end must be provided together")
        if (
            self.page_start is not None
            and self.page_end is not None
            and self.page_end < self.page_start
        ):
            raise ValueError("page_end must not precede page_start")
        return self


class RetrievalChunkConfig(ContractModel):
    """Identity of the chunking strategy used to build an immutable index."""

    strategy_name: str = Field(min_length=1, max_length=128)
    strategy_version: str = Field(min_length=1, max_length=128)
    config_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @field_validator("strategy_name", "strategy_version")
    @classmethod
    def validate_chunk_config_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("chunk config values must not be blank")
        return value


class RetrievalProvenance(ContractModel):
    """Non-claimable runtime provenance for artifact-backed retrieval."""

    status: Literal["provisional"] = "provisional"
    claims_allowed: Literal[False] = False
    retriever_version: str = Field(min_length=1, max_length=128)
    embedding_model: str = Field(min_length=1, max_length=256)
    embedding_dimensions: int = Field(ge=1)
    chunk_config: RetrievalChunkConfig
    top_k: int = Field(ge=1, le=100)
    candidate_limit: int = Field(ge=1, le=100)
    manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    chunk_catalog_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    index_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    latency_ms: float = Field(ge=0)

    @field_validator("retriever_version", "embedding_model")
    @classmethod
    def validate_provenance_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("retrieval provenance values must not be blank")
        return value

    @field_validator("latency_ms")
    @classmethod
    def validate_latency(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("retrieval latency must be finite")
        return value


class SearchResult(ContractModel):
    retriever: str = Field(min_length=1)
    hits: tuple[SearchHit, ...]

    @field_validator("retriever")
    @classmethod
    def validate_retriever(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("retriever must not be blank")
        return value

    @model_validator(mode="after")
    def validate_ranking(self) -> Self:
        _validate_ranked_hits(self.hits)
        return self


class ArtifactSearchResult(SearchResult):
    """SearchResult extension emitted only by the artifact-backed retriever."""

    provenance: RetrievalProvenance


class RetrievalDocument(ContractModel):
    """Small immutable document record consumed by retrieval adapters.

    Parsed document chunks deliberately do not depend on this module.  The
    adapters also accept objects exposing ``chunk_id``, ``document_id`` and
    ``raw_text``/``text`` so callers can pass parsed chunks without copying
    them into this DTO first.
    """

    chunk_id: str = Field(min_length=1)
    document_id: str = Field(min_length=1)
    text: str = Field(min_length=1)
    section_path: tuple[str, ...] = ()
    page_start: int | None = Field(default=None, ge=1)
    page_end: int | None = Field(default=None, ge=1)

    @field_validator("chunk_id", "document_id", "text")
    @classmethod
    def validate_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("text must not be blank")
        return value

    @field_validator("section_path")
    @classmethod
    def validate_document_section_path(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        if any(not value.strip() for value in values):
            raise ValueError("section_path must not contain blank values")
        return values

    @model_validator(mode="after")
    def validate_document_page_range(self) -> Self:
        if (self.page_start is None) != (self.page_end is None):
            raise ValueError("page_start and page_end must be provided together")
        if (
            self.page_start is not None
            and self.page_end is not None
            and self.page_end < self.page_start
        ):
            raise ValueError("page_end must not precede page_start")
        return self


# ``SearchDocument`` is a useful compatibility spelling for application code.
SearchDocument = RetrievalDocument


class FusionRequest(ContractModel):
    result_sets: tuple[SearchResult, ...]
    limit: int = Field(default=10, ge=1, le=100)


class FusionResult(ContractModel):
    strategy: str = Field(min_length=1)
    hits: tuple[SearchHit, ...]

    @field_validator("strategy")
    @classmethod
    def validate_strategy(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("strategy must not be blank")
        return value

    @model_validator(mode="after")
    def validate_ranking(self) -> Self:
        _validate_ranked_hits(self.hits)
        return self


def _validate_ranked_hits(hits: tuple[SearchHit, ...]) -> None:
    chunk_ids = [hit.chunk_id for hit in hits]
    if len(set(chunk_ids)) != len(chunk_ids):
        raise ValueError("ranked hits must not contain duplicate chunk_ids")
    expected_ranks = list(range(1, len(hits) + 1))
    if [hit.rank for hit in hits] != expected_ranks:
        raise ValueError("hit ranks must be contiguous and match result order")
