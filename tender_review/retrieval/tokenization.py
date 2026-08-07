"""Deterministic, dependency-free tokenization for retrieval indexes."""

from __future__ import annotations

import re
from collections.abc import Iterable


_TOKEN_PATTERN = re.compile(r"[A-Za-z0-9]+|[\u3400-\u9fff]+")


def tokenize(text: str, *, domain_terms: Iterable[str] = ()) -> tuple[str, ...]:
    """Return normalized tokens suitable for BM25.

    Latin words and numbers are kept as runs while CJK runs are split into
    characters.  This makes Chinese queries useful without relying on a
    mutable external dictionary and keeps indexing reproducible offline.
    """

    normalized_text = text.casefold()
    tokens: list[str] = []
    for match in _TOKEN_PATTERN.finditer(normalized_text):
        value = match.group(0)
        if value[0] >= "\u3400":
            tokens.extend(value)
        else:
            tokens.append(value)
    normalized_terms = sorted(
        {term.strip().casefold() for term in domain_terms if term.strip()},
        key=lambda term: (-len(term), term),
    )
    for term in normalized_terms:
        start = 0
        while True:
            index = normalized_text.find(term, start)
            if index < 0:
                break
            tokens.append(term)
            start = index + len(term)
    return tuple(tokens)
