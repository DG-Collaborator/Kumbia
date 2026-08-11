"""
   A tiny in-memory vector store: holds texts alongside their embedding
   vectors, and retrieves the texts most similar to a query vector by cosine
   similarity.

   "In-memory first" by design — proves the RAG retrieval loop with zero
   dependencies. A persistent store (e.g. SQLite + a vector extension) can
   replace it later behind the same add/search shape.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import sqrt


@dataclass(frozen=True)
class _Entry:
    text: str
    vector: list[float]


class InMemoryVectorIndex:
    def __init__(self) -> None:
        self._entries: list[_Entry] = []

    @property
    def size(self) -> int:
        return len(self._entries)

    def add(self, text: str, vector: list[float]) -> None:
        self._entries.append(_Entry(text, vector))

    def search(self, query_vector: list[float], top_k: int = 4) -> list[str]:
        """
           Returns the top_k texts whose vectors are most similar to the
           query vector, most-similar first. With fewer than top_k entries it
           returns everything it has, ranked.
        """
        scored = [
            (entry.text, _cosine_similarity(query_vector, entry.vector))
            for entry in self._entries
        ]
        scored.sort(key=lambda pair: pair[1], reverse=True)
        return [text for text, _ in scored[:top_k]]


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """
       Cosine similarity = how aligned two vectors are, ignoring length. 1.0
       means identical direction, 0.0 means unrelated (orthogonal). It's the
       standard relevance measure for text embeddings.
    """
    if len(a) != len(b):
        raise ValueError(f"vector dimensions differ: {len(a)} vs {len(b)}")

    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sqrt(sum(x * x for x in a))
    norm_b = sqrt(sum(y * y for y in b))

    denominator = norm_a * norm_b
    return dot / denominator if denominator != 0.0 else 0.0