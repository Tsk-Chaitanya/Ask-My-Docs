"""
Phase 2 Tests — BM25, hybrid retrieval, and reranker.

Run with:  pytest tests/test_phase2.py -v

Note: Reranker tests that need the real CrossEncoder model require
      sentence-transformers and PyTorch. Those tests are marked with
      @requires_cross_encoder and will be skipped if unavailable.
      Mock-based reranker tests always run.
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.ingestion.chunker import Chunk
from src.retrieval.vector import VectorStore, RetrievalResult
from src.retrieval.bm25 import BM25Index
from src.retrieval.hybrid import HybridRetriever
from src.retrieval.reranker import Reranker

# Check if sentence-transformers + torch are available for live reranker tests
try:
    from sentence_transformers import CrossEncoder
    _HAS_CROSS_ENCODER = True
except (ImportError, OSError, ValueError):
    _HAS_CROSS_ENCODER = False

requires_cross_encoder = pytest.mark.skipif(
    not _HAS_CROSS_ENCODER,
    reason="sentence-transformers / PyTorch not available",
)


# ── Helper ───────────────────────────────────────────────────


def _make_chunks() -> list[Chunk]:
    """Three test chunks on different topics."""
    return [
        Chunk(
            chunk_id="c1",
            content="Python is a high-level programming language used for web development and data science",
            token_count=14,
            metadata={"source": "python.txt"},
        ),
        Chunk(
            chunk_id="c2",
            content="Retrieval augmented generation combines search with language models to answer questions from documents",
            token_count=15,
            metadata={"source": "rag.md"},
        ),
        Chunk(
            chunk_id="c3",
            content="BM25 is a ranking function used in information retrieval to score document relevance for search queries",
            token_count=17,
            metadata={"source": "bm25.txt"},
        ),
    ]


# ── BM25 Tests ───────────────────────────────────────────────


class TestBM25:

    def test_build_and_query(self):
        """BM25 should rank the BM25 chunk first for a BM25-related query."""
        bm25 = BM25Index()
        bm25.build(_make_chunks())
        results = bm25.query("BM25 ranking", top_k=3)
        assert len(results) > 0
        assert results[0].chunk_id == "c3"

    def test_save_and_load(self, tmp_path):
        """Save, load into a new instance, verify query still works."""
        bm25 = BM25Index()
        bm25.build(_make_chunks())

        save_path = tmp_path / "bm25_index.json"
        bm25.save(save_path)

        bm25_loaded = BM25Index()
        bm25_loaded.load(save_path)

        results = bm25_loaded.query("BM25 ranking", top_k=3)
        assert len(results) > 0
        assert results[0].chunk_id == "c3"

    def test_query_before_build_raises(self):
        """Querying before build() should raise RuntimeError."""
        bm25 = BM25Index()
        with pytest.raises(RuntimeError):
            bm25.query("test")

    def test_no_results_for_nonsense(self):
        """Gibberish query should return empty list."""
        bm25 = BM25Index()
        bm25.build(_make_chunks())
        results = bm25.query("xyzzyplugh42 qwfpgjluy")
        assert results == []


# ── Hybrid Retriever Tests ───────────────────────────────────


class TestHybridRetriever:

    def test_fuses_results(self):
        """Hybrid retriever should return results from both sources."""
        chunks = _make_chunks()

        store = VectorStore()
        store.add_chunks(chunks)

        bm25 = BM25Index()
        bm25.build(chunks)

        hybrid = HybridRetriever(vector_store=store, bm25_index=bm25)
        results = hybrid.retrieve("BM25 ranking search", top_k=3)
        assert len(results) > 0
        valid_ids = {c.chunk_id for c in chunks}
        for r in results:
            assert r.chunk_id in valid_ids

    def test_rrf_scoring(self):
        """A chunk ranked highly by both retrievers should score higher
        than one ranked highly by only one."""
        chunks = _make_chunks()

        store = VectorStore()
        store.add_chunks(chunks)

        bm25 = BM25Index()
        bm25.build(chunks)

        hybrid = HybridRetriever(vector_store=store, bm25_index=bm25)
        results = hybrid.retrieve("BM25 ranking function for search", top_k=3)

        assert len(results) > 0
        # c3 (BM25 chunk) should have the highest fused score
        assert results[0].chunk_id == "c3"


# ── Reranker Tests ───────────────────────────────────────────


class TestReranker:

    @requires_cross_encoder
    def test_rerank_returns_results(self):
        """Reranker should return sorted results (live model)."""
        candidates = [
            RetrievalResult(chunk_id="c1", content="Python is a programming language", score=0.5, metadata={}),
            RetrievalResult(chunk_id="c2", content="BM25 is a ranking function for search", score=0.8, metadata={}),
            RetrievalResult(chunk_id="c3", content="Machine learning uses neural networks", score=0.3, metadata={}),
        ]
        reranker = Reranker()
        results = reranker.rerank("What is BM25?", candidates, top_k=3)
        assert len(results) > 0
        for i in range(len(results) - 1):
            assert results[i].score >= results[i + 1].score

    @requires_cross_encoder
    def test_rerank_empty_candidates(self):
        """Reranking empty list returns empty list (live model)."""
        reranker = Reranker()
        results = reranker.rerank("test query", [], top_k=5)
        assert results == []

    @requires_cross_encoder
    def test_rerank_respects_top_k(self):
        """Pass 5 candidates with top_k=2, verify only 2 returned (live model)."""
        candidates = [
            RetrievalResult(chunk_id=f"c{i}", content=f"Document number {i} about various topics", score=0.5, metadata={})
            for i in range(5)
        ]
        reranker = Reranker()
        results = reranker.rerank("topics", candidates, top_k=2)
        assert len(results) <= 2

    # ── Mock-based tests (always run, no PyTorch needed) ─────

    def test_rerank_empty_with_mock(self):
        """Reranking empty list works via mocked model."""
        with patch.object(Reranker, "__init__", lambda self, **kw: None):
            reranker = Reranker()
            reranker.relevance_threshold = 0.0
            results = reranker.rerank("test query", [], top_k=5)
            assert results == []

    def test_rerank_logic_with_mock(self):
        """Test rerank scoring and sorting with a mocked CrossEncoder."""
        with patch.object(Reranker, "__init__", lambda self, **kw: None):
            reranker = Reranker()
            reranker.relevance_threshold = 0.0
            reranker.model = MagicMock()
            reranker.model.predict.return_value = np.array([0.1, 0.9, 0.5])

            candidates = [
                RetrievalResult(chunk_id="c0", content="Low relevance", score=0.8, metadata={}),
                RetrievalResult(chunk_id="c1", content="High relevance", score=0.3, metadata={}),
                RetrievalResult(chunk_id="c2", content="Medium relevance", score=0.5, metadata={}),
            ]
            results = reranker.rerank("test", candidates, top_k=2)

            assert len(results) == 2
            assert results[0].chunk_id == "c1"   # score 0.9
            assert results[1].chunk_id == "c2"   # score 0.5
            assert results[0].score > results[1].score

    def test_rerank_top_k_with_mock(self):
        """Verify top_k is respected with mocked model."""
        with patch.object(Reranker, "__init__", lambda self, **kw: None):
            reranker = Reranker()
            reranker.relevance_threshold = 0.0
            reranker.model = MagicMock()
            reranker.model.predict.return_value = np.array([0.9, 0.8, 0.7, 0.6, 0.5])

            candidates = [
                RetrievalResult(chunk_id=f"c{i}", content=f"Doc {i}", score=0.5, metadata={})
                for i in range(5)
            ]
            results = reranker.rerank("test", candidates, top_k=2)
            assert len(results) == 2
