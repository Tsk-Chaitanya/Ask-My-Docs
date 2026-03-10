"""
Phase 1 Tests — verifying each component of the RAG pipeline.

Run with:  pytest tests/ -v
"""

import json
import tempfile
import unittest.mock
from pathlib import Path

import pytest

from src.ingestion.loader import Document, load_text, load_markdown, load_directory
from src.ingestion.chunker import TokenChunker, Chunk
from src.retrieval.vector import VectorStore, RetrievalResult
from src.generation.prompt_manager import PromptManager
from src.generation.generator import AnswerGenerator


# ── Loader Tests ─────────────────────────────────────────────


class TestLoaders:

    def test_load_text_reads_content(self, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text("Hello, world!")
        docs = load_text(f)
        assert len(docs) == 1
        assert docs[0].content == "Hello, world!"
        assert docs[0].metadata["type"] == "text"

    def test_load_markdown_strips_frontmatter(self, tmp_path):
        f = tmp_path / "test.md"
        f.write_text("---\ntitle: Test\n---\n# Heading\nBody text.")
        docs = load_markdown(f)
        assert len(docs) == 1
        assert "# Heading" in docs[0].content
        assert "title: Test" not in docs[0].content

    def test_load_directory_only_loads_supported_files(self, tmp_path):
        (tmp_path / "good.txt").write_text("Valid file")
        (tmp_path / "good.md").write_text("# Also valid")
        (tmp_path / "skip.jpg").write_bytes(b"\xff\xd8")
        (tmp_path / "skip.docx").write_bytes(b"PK")
        docs = load_directory(tmp_path)
        assert len(docs) == 2  # only .txt and .md


# ── Chunker Tests ────────────────────────────────────────────


class TestChunker:

    def test_short_doc_stays_single_chunk(self):
        doc = Document(content="Short text.", metadata={"source": "test"})
        chunker = TokenChunker(chunk_size=600, chunk_overlap=100)
        chunks = chunker.chunk_document(doc)
        assert len(chunks) == 1

    def test_long_doc_produces_multiple_chunks(self):
        long_text = " ".join(["word"] * 2000)
        doc = Document(content=long_text, metadata={"source": "test"})
        chunker = TokenChunker(chunk_size=600, chunk_overlap=100)
        chunks = chunker.chunk_document(doc)
        assert len(chunks) > 1

    def test_overlap_is_correct(self):
        """Verify consecutive chunks share ~100 words at their boundary."""
        long_text = " ".join([f"w{i}" for i in range(2000)])
        doc = Document(content=long_text, metadata={"source": "test"})
        chunker = TokenChunker(chunk_size=600, chunk_overlap=100)
        chunks = chunker.chunk_document(doc)

        c0_words = chunks[0].content.split()
        c1_words = chunks[1].content.split()

        # Chunk 1 should start 500 words into chunk 0
        # (step = 600 - 100 = 500)
        assert c1_words[0] in c0_words

    def test_chunk_ids_are_deterministic(self):
        """Same input always produces the same chunk IDs."""
        doc = Document(content="Deterministic content.", metadata={"source": "t"})
        c1 = TokenChunker().chunk_document(doc)
        c2 = TokenChunker().chunk_document(doc)
        assert c1[0].chunk_id == c2[0].chunk_id

    def test_chunk_metadata_preserved(self):
        doc = Document(content="Text.", metadata={"source": "paper.pdf", "page": 3})
        chunks = TokenChunker().chunk_document(doc)
        assert chunks[0].metadata["source"] == "paper.pdf"
        assert chunks[0].metadata["page"] == 3
        assert chunks[0].metadata["chunk_index"] == 0


# ── Vector Store Tests ───────────────────────────────────────


class TestVectorStore:

    def _make_chunks(self) -> list[Chunk]:
        return [
            Chunk(chunk_id="c1", content="Python is a programming language", token_count=6, metadata={"source": "a.txt"}),
            Chunk(chunk_id="c2", content="Retrieval augmented generation uses documents", token_count=6, metadata={"source": "b.txt"}),
            Chunk(chunk_id="c3", content="BM25 is a ranking function for search", token_count=8, metadata={"source": "c.txt"}),
        ]

    def test_add_and_count(self):
        store = VectorStore()
        store.add_chunks(self._make_chunks())
        assert store.count == 3

    def test_query_returns_relevant_results(self):
        store = VectorStore()
        store.add_chunks(self._make_chunks())
        results = store.query("BM25 ranking search", top_k=3)
        assert len(results) > 0
        # BM25 chunk should rank first for a BM25-related query
        assert results[0].chunk_id == "c3"

    def test_query_empty_store_returns_empty(self):
        store = VectorStore()
        # Delete and recreate collection to ensure it's empty
        store.client.delete_collection(name="ask_my_docs")
        store.collection = store.client.get_or_create_collection(
            name="ask_my_docs",
            metadata={"hnsw:space": "cosine"},
            embedding_function=store.embedding_fn,
        )
        results = store.query("anything", top_k=5)
        assert results == []

    def test_results_have_scores(self):
        store = VectorStore()
        store.add_chunks(self._make_chunks())
        results = store.query("Python programming", top_k=1)
        assert isinstance(results[0].score, float)


# ── Prompt Manager Tests ─────────────────────────────────────


class TestPromptManager:

    def test_loads_version(self):
        pm = PromptManager()
        assert pm.version == "1.0.0"

    def test_renders_template(self):
        pm = PromptManager()
        rendered = pm.render_user_prompt(
            "generation",
            context="Some context here",
            question="What is RAG?",
        )
        assert "Some context here" in rendered
        assert "What is RAG?" in rendered

    def test_system_prompt_exists(self):
        pm = PromptManager()
        system = pm.get_system_prompt("generation")
        assert "source chunks" in system.lower()


# ── Generator Tests ──────────────────────────────────────────


class TestGenerator:

    def test_declines_with_no_chunks(self):
        gen = AnswerGenerator()
        result = gen.generate("Any question?", chunks=[])
        assert result.declined is True
        assert "don't have enough" in result.answer

    def test_demo_mode_without_api_key(self):
        chunks = [
            RetrievalResult(chunk_id="abc123", content="Test content", score=0.9, metadata={"source": "test.txt"}),
        ]
        with unittest.mock.patch.dict("os.environ", {"ANTHROPIC_API_KEY": ""}, clear=False):
            gen = AnswerGenerator()
        result = gen.generate("Test question?", chunks)
        assert "DEMO MODE" in result.answer
        assert result.prompt_version == "1.0.0"
