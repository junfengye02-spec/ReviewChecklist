"""Private normalization helpers shared by retrieval adapters."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

from .models import RetrievalDocument, SearchHit


@dataclass(frozen=True, slots=True)
class IndexedDocument:
    chunk_id: str
    document_id: str
    text: str
    section_path: tuple[str, ...] = ()
    page_start: int | None = None
    page_end: int | None = None


def coerce_documents(documents: Iterable[object]) -> tuple[IndexedDocument, ...]:
    result: list[IndexedDocument] = []
    seen: set[str] = set()
    for value in documents:
        document = _coerce_document(value)
        if document.chunk_id in seen:
            raise ValueError(f"duplicate chunk_id: {document.chunk_id}")
        seen.add(document.chunk_id)
        result.append(document)
    return tuple(result)


def _coerce_document(value: object) -> IndexedDocument:
    if isinstance(value, RetrievalDocument):
        return IndexedDocument(
            value.chunk_id,
            value.document_id,
            value.text,
            value.section_path,
            value.page_start,
            value.page_end,
        )
    if isinstance(value, SearchHit):
        return IndexedDocument(
            value.chunk_id,
            value.document_id,
            value.text,
            value.section_path,
            value.page_start,
            value.page_end,
        )
    if isinstance(value, Mapping):
        chunk_id = value.get("chunk_id")
        document_id = value.get("document_id")
        text = value.get("text")
        if text is None:
            text = value.get("raw_text")
        section_path = value.get("section_path", ())
        page_start = value.get("page_start")
        page_end = value.get("page_end")
    else:
        chunk_id = getattr(value, "chunk_id", None)
        document_id = getattr(value, "document_id", None)
        text = getattr(value, "text", None)
        if text is None:
            text = getattr(value, "raw_text", None)
        section_path = getattr(value, "section_path", ())
        page_start = getattr(value, "page_start", None)
        page_end = getattr(value, "page_end", None)
    if not isinstance(chunk_id, str) or not chunk_id.strip():
        raise TypeError("retrieval documents require a non-empty chunk_id")
    if not isinstance(document_id, str) or not document_id.strip():
        raise TypeError("retrieval documents require a non-empty document_id")
    if not isinstance(text, str) or not text.strip():
        raise TypeError("retrieval documents require non-empty text/raw_text")
    if isinstance(section_path, str) or not isinstance(section_path, Sequence):
        raise TypeError("section_path must be a sequence of strings")
    normalized_path = tuple(section_path)
    if any(not isinstance(item, str) or not item.strip() for item in normalized_path):
        raise TypeError("section_path must contain only non-empty strings")
    if (page_start is None) != (page_end is None):
        raise ValueError("page_start and page_end must be provided together")
    if page_start is not None:
        if (
            not isinstance(page_start, int)
            or isinstance(page_start, bool)
            or not isinstance(page_end, int)
            or isinstance(page_end, bool)
            or page_start < 1
            or page_end < page_start
        ):
            raise ValueError("retrieval document page range is invalid")
    return IndexedDocument(
        chunk_id,
        document_id,
        text,
        normalized_path,
        page_start,
        page_end,
    )


def vector_for(
    vectors: Mapping[str, Sequence[float]] | Sequence[Sequence[float]] | None,
    documents: tuple[IndexedDocument, ...],
) -> dict[str, tuple[float, ...]]:
    if not documents:
        raise ValueError("vector retriever requires at least one document")
    if vectors is None:
        raise ValueError("vectors are required for a vector retriever")
    try:
        if isinstance(vectors, Mapping):
            values = {
                str(key): tuple(float(item) for item in value)
                for key, value in vectors.items()
            }
        else:
            if len(vectors) != len(documents):
                raise ValueError("vector sequence must match document count")
            values = {
                document.chunk_id: tuple(float(item) for item in vector)
                for document, vector in zip(documents, vectors, strict=True)
            }
    except (TypeError, ValueError) as exc:
        raise ValueError("vectors must contain numeric sequences") from exc
    document_ids = {document.chunk_id for document in documents}
    missing = [
        document.chunk_id for document in documents if document.chunk_id not in values
    ]
    if missing:
        raise ValueError(f"missing vectors for chunks: {', '.join(missing)}")
    unexpected = sorted(set(values) - document_ids)
    if unexpected:
        raise ValueError(f"vectors contain unknown chunks: {', '.join(unexpected)}")
    dimensions = {len(value) for value in values.values()}
    if not dimensions or dimensions == {0}:
        raise ValueError("vectors must have at least one dimension")
    if len(dimensions) != 1 or 0 in dimensions:
        raise ValueError("all vectors must have the same positive dimension")
    if any(not math.isfinite(item) for value in values.values() for item in value):
        raise ValueError("vectors must contain only finite values")
    return values
