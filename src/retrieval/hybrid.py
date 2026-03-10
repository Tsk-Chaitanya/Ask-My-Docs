"""
Hybrid retriever combining BM25 + vector search with Reciprocal Rank Fusion.

WHY HYBRID:
  Dense (vector) retrieval captures semantic similarity but can miss exact
  keyword matches. BM25 nails exact terms but misses paraphrases. Combining
  them covers both cases.

WHY RRF (not score averaging):
  BM25 scores range 0–15, cosine similarity ranges 0–1. Simply averaging
  would let BM25 dominate. RRF uses only rank positions, so no score
  calibration is needed.

RRF FORMULA:
  For each document d, fused_score(d) = sum over retrievers R of:
      weight_R / (k + rank_R(d))
  where k=60 is a smoothing constant and rank starts at 1.
"""

from __future__ import annotations

from src.retrieval.bm25 import BM25Index
from src.retrieval.vector import VectorStore, RetrievalResult


class HybridRetriever:
    """Fuses BM25 and vector retrieval results using Reciprocal Rank Fusion."""

    def __init__(
        self,
        vector_store: VectorStore,
        bm25_index: BM25Index,
        rrf_k: int = 60,
        vector_weight: float = 1.0,
        bm25_weight: float = 1.0,
    ):
        self.vector_store = vector_store
        self.bm25_index = bm25_index
        self.rrf_k = rrf_k
        self.vector_weight = vector_weight
        self.bm25_weight = bm25_weight

    def retrieve(
        self, query: str, top_k: int = 10, fetch_k: int = 20
    ) -> list[RetrievalResult]:
        """
        Fetch candidates from both retrievers, fuse with RRF, return top_k.

        Args:
            query: The search query.
            top_k: Number of final results to return.
            fetch_k: Number of candidates to fetch from each retriever.
        """
        # Step 1: Fetch candidates from both retrievers
        vector_results = self.vector_store.query(query, top_k=fetch_k)
        bm25_results = self.bm25_index.query(query, top_k=fetch_k)

        # Step 2: Build a lookup dict of chunk_id → RetrievalResult
        lookup: dict[str, RetrievalResult] = {}
        for r in vector_results:
            lookup[r.chunk_id] = r
        for r in bm25_results:
            if r.chunk_id not in lookup:
                lookup[r.chunk_id] = r

        # Step 3: Compute RRF scores
        fused_scores: dict[str, float] = {}

        # Vector retriever contribution (rank starts at 1)
        for rank, r in enumerate(vector_results, start=1):
            fused_scores[r.chunk_id] = fused_scores.get(r.chunk_id, 0.0) + (
                self.vector_weight / (self.rrf_k + rank)
            )

        # BM25 retriever contribution
        for rank, r in enumerate(bm25_results, start=1):
            fused_scores[r.chunk_id] = fused_scores.get(r.chunk_id, 0.0) + (
                self.bm25_weight / (self.rrf_k + rank)
            )

        # Step 4: Sort by fused score descending, return top_k
        sorted_ids = sorted(fused_scores, key=fused_scores.get, reverse=True)

        return [
            RetrievalResult(
                chunk_id=cid,
                content=lookup[cid].content,
                score=fused_scores[cid],
                metadata=lookup[cid].metadata,
            )
            for cid in sorted_ids[:top_k]
        ]
