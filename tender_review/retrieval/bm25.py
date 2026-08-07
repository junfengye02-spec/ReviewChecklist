"""Deterministic in-memory BM25 retriever."""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Iterable

from tender_review.shared.contracts import ensure_call_active

from ._common import IndexedDocument, coerce_documents
from .models import SearchHit, SearchRequest, SearchResult
from .tokenization import tokenize


class BM25Retriever:
    """A dependency-free BM25 index over immutable chunk text."""

    name = "bm25"

    def __init__(
        self,
        documents: Iterable[object] = (),
        *,
        k1: float = 1.2,
        b: float = 0.75,
        domain_terms: Iterable[str] = (),
    ) -> None:
        if not math.isfinite(k1) or k1 <= 0:
            raise ValueError("k1 must be a finite positive number")
        if not math.isfinite(b) or not 0 <= b <= 1:
            raise ValueError("b must be a finite number between 0 and 1")
        normalized_terms = tuple(
            sorted({term.strip().casefold() for term in domain_terms if term.strip()})
        )
        self.k1 = float(k1)
        self.b = float(b)
        self.domain_terms = normalized_terms
        self.documents: tuple[IndexedDocument, ...] = coerce_documents(documents)
        self._tokens = {
            document.chunk_id: tokenize(
                document.text,
                domain_terms=self.domain_terms,
            )
            for document in self.documents
        }
        self._term_frequencies = {
            chunk_id: Counter(tokens) for chunk_id, tokens in self._tokens.items()
        }
        self._document_frequencies: Counter[str] = Counter()
        for tokens in self._tokens.values():
            self._document_frequencies.update(set(tokens))
        self._average_length = (
            sum(len(tokens) for tokens in self._tokens.values()) / len(self.documents)
            if self.documents
            else 0.0
        )

    def search(self, request: SearchRequest) -> SearchResult:
        ensure_call_active(request.call)
        allowed = set(request.document_ids)
        query_terms = tokenize(request.query, domain_terms=self.domain_terms)
        ranked: list[tuple[float, IndexedDocument]] = []
        for document in self.documents:
            if allowed and document.document_id not in allowed:
                continue
            score = self._score(query_terms, document)
            if score > 0:
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
            for rank, (score, document) in enumerate(
                ranked[: request.limit],
                start=1,
            )
        )
        return SearchResult(retriever=self.name, hits=hits)

    def _score(
        self,
        query_terms: tuple[str, ...],
        document: IndexedDocument,
    ) -> float:
        document_count = len(self.documents)
        if not query_terms or document_count == 0:
            return 0.0
        frequencies = self._term_frequencies[document.chunk_id]
        length = len(self._tokens[document.chunk_id])
        score = 0.0
        for term, query_frequency in Counter(query_terms).items():
            term_frequency = frequencies.get(term, 0)
            if not term_frequency:
                continue
            document_frequency = self._document_frequencies[term]
            inverse_document_frequency = math.log(
                1.0
                + (document_count - document_frequency + 0.5)
                / (document_frequency + 0.5)
            )
            length_normalization = (
                1.0 - self.b + self.b * length / self._average_length
                if self._average_length
                else 1.0
            )
            denominator = term_frequency + self.k1 * length_normalization
            score += (
                query_frequency
                * inverse_document_frequency
                * (term_frequency * (self.k1 + 1.0) / denominator)
            )
        return score


Bm25Retriever = BM25Retriever
