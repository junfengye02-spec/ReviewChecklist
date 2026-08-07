"""Validated configuration for explicitly registered retrieval strategies."""

from __future__ import annotations

import math
from typing import Literal

from pydantic import Field, field_validator

from tender_review.shared.contracts import ContractModel


class BM25Config(ContractModel):
    kind: Literal["bm25"] = "bm25"
    k1: float = Field(default=1.2, gt=0)
    b: float = Field(default=0.75, ge=0, le=1)
    domain_terms: tuple[str, ...] = ()

    @field_validator("k1", "b")
    @classmethod
    def validate_finite_number(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("retrieval numeric settings must be finite")
        return value

    @field_validator("domain_terms")
    @classmethod
    def normalize_domain_terms(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        normalized = tuple(value.strip().casefold() for value in values)
        if any(not value for value in normalized):
            raise ValueError("domain_terms must not contain blank values")
        if len(set(normalized)) != len(normalized):
            raise ValueError("domain_terms must not contain duplicates")
        return normalized


class VectorConfig(ContractModel):
    kind: Literal["vector"] = "vector"
    similarity: Literal["cosine"] = "cosine"


class RRFConfig(ContractModel):
    k: int = Field(default=60, ge=1, le=10_000)


class HybridConfig(ContractModel):
    kind: Literal["hybrid"] = "hybrid"
    bm25: BM25Config = Field(default_factory=BM25Config)
    vector: VectorConfig = Field(default_factory=VectorConfig)
    fusion: RRFConfig = Field(default_factory=RRFConfig)
    candidate_limit: int = Field(default=100, ge=1, le=100)


RetrieverConfig = BM25Config | VectorConfig | HybridConfig
