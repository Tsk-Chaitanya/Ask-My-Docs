"""
BM25 sparse retrieval using rank_bm25.

BM25 (Best Matching 25) excels at exact keyword matching — technical
terms, acronyms, proper nouns — where dense embeddings sometimes
struggle. By combining BM25 with vector search (hybrid retrieval),
we get the best of both worlds.

HOW BM25 WORKS:
  1. Tokenize all documents (bag of words)
  2. For each query term, score documents based on:
     - Term frequency in the document (more = better)
     - Inverse document frequency (rare terms matter more)
     - Document length normalization (don't favor long docs)
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from rank_bm25 import BM25Okapi

from src.ingestion.chunker import Chunk
from src.retrieval.vector import RetrievalResult


class BM25Index:
    """BM25 sparse retrieval index over document chunks."""

    def __init__(self):
        self._chunks: list[dict] = []       # [{chunk_id, content, metadata}]
        self._tokenized: list[list[str]] = []
        self._index: BM25Okapi | None = None

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """Lowercase and split on word boundaries."""
        return re.findall(r"\b\w+\b", text.lower())

    def build(self, chunks: list[Chunk]) -> None:
        """Tokenize chunk content and build the BM25Okapi index."""
        self._chunks = [
            {"chunk_id": c.chunk_id, "content": c.content, "metadata": c.metadata}
            for c in chunks
        ]
        self._tokenized = [self._tokenize(c.content) for c in chunks]
        self._index = BM25Okapi(self._tokenized)

    def query(self, query_text: str, top_k: int = 10) -> list[RetrievalResult]:
        """Score all chunks against the query, return top-k with score > 0."""
        if self._index is None:
            raise RuntimeError("BM25 index has not been built. Call build() first.")

        tokens = self._tokenize(query_text)
        scores = self._index.get_scores(tokens)

        # Pair each chunk with its score, filter score > 0, sort descending
        scored = [
            (self._chunks[i], float(scores[i]))
            for i in range(len(self._chunks))
            if scores[i] > 0
        ]
        scored.sort(key=lambda x: x[1], reverse=True)

        return [
            RetrievalResult(
                chunk_id=chunk["chunk_id"],
                content=chunk["content"],
                score=score,
                metadata=chunk["metadata"],
            )
            for chunk, score in scored[:top_k]
        ]

    def save(self, path: str | Path) -> None:
        """Save chunk metadata to JSON (the BM25 index is rebuilt on load)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self._chunks, f, indent=2)

    def load(self, path: str | Path) -> None:
        """Load chunk metadata from JSON and rebuild the BM25 index."""
        with open(path, "r", encoding="utf-8") as f:
            self._chunks = json.load(f)
        self._tokenized = [self._tokenize(c["content"]) for c in self._chunks]
        self._index = BM25Okapi(self._tokenized)
