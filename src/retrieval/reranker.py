"""
Cross-encoder reranker for second-stage relevance filtering.

WHY RERANK:
  First-stage retrievers (BM25, vector) are fast but approximate. A
  cross-encoder processes (query, document) pairs *jointly*, attending
  to fine-grained interactions between query and document tokens. This
  is much more accurate but too slow to run on the full corpus — so we
  use it only on the top candidates from hybrid retrieval.

MODEL:
  ms-marco-MiniLM-L-6-v2 is a compact cross-encoder trained on the
  MS MARCO passage ranking dataset. ~80MB download on first use.

NOTE:
  sentence-transformers and torch are imported lazily so the module
  can be imported even when those packages are not installed (e.g.
  in lightweight test environments). The actual model is loaded at
  Reranker construction time.
"""

from __future__ import annotations

from src.retrieval.vector import RetrievalResult


class Reranker:
    """Cross-encoder reranker using sentence-transformers."""

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        relevance_threshold: float = 0.0,
    ):
        from sentence_transformers import CrossEncoder
        self.model = CrossEncoder(model_name, max_length=512)
        self.relevance_threshold = relevance_threshold

    def rerank(
        self,
        query: str,
        candidates: list[RetrievalResult],
        top_k: int = 5,
    ) -> list[RetrievalResult]:
        """
        Re-score candidates with the cross-encoder and return the top_k.

        Args:
            query: The user's question.
            candidates: First-stage retrieval results to rerank.
            top_k: How many to keep after reranking.
        """
        if not candidates:
            return []

        # Build (query, document) pairs for the cross-encoder
        pairs = [(query, c.content) for c in candidates]

        # Score all pairs
        scores = self.model.predict(pairs)

        # Create new RetrievalResults with cross-encoder scores
        reranked = [
            RetrievalResult(
                chunk_id=candidates[i].chunk_id,
                content=candidates[i].content,
                score=float(scores[i]),
                metadata=candidates[i].metadata,
            )
            for i in range(len(candidates))
            if float(scores[i]) >= self.relevance_threshold
        ]

        # Sort by score descending, return top_k
        reranked.sort(key=lambda r: r.score, reverse=True)
        return reranked[:top_k]
