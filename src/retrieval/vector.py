"""
Dense vector retrieval using ChromaDB with local embeddings.

Since this environment has restricted network access, we provide
our own embedding function. In production, you'd swap this for
sentence-transformers or an API-based embedder.

THE CONCEPT IS THE SAME:
  1. Convert text → numbers (embeddings)
  2. Store in a vector database
  3. At query time, embed the query and find nearest neighbors
"""

from __future__ import annotations

import hashlib
import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import chromadb
from chromadb.api.types import EmbeddingFunction, Documents, Embeddings

from src.ingestion.chunker import Chunk


COLLECTION_NAME = "ask_my_docs"


# ── Local Embedding Function ────────────────────────────────


class LocalEmbedding(EmbeddingFunction):
    """
    Bag-of-words embedding that runs entirely locally.

    HOW IT WORKS:
      1. Tokenize text into words
      2. Hash each word to a fixed position in a 384-dim vector
      3. Apply log(1 + count) smoothing
      4. L2-normalize

    SWAPPING FOR PRODUCTION:
      Replace with SentenceTransformerEmbeddingFunction or OpenAI embeddings.
    """

    def __init__(self, dim: int = 384):
        self.dim = dim

    def _tokenize(self, text: str) -> list[str]:
        return re.findall(r"\b\w+\b", text.lower())

    def _embed_one(self, text: str) -> list[float]:
        tokens = self._tokenize(text)
        counts = Counter(tokens)
        vec = [0.0] * self.dim

        for word, count in counts.items():
            idx = int(hashlib.md5(word.encode()).hexdigest(), 16) % self.dim
            vec[idx] += math.log(1 + count)

        # L2 normalize
        norm = math.sqrt(sum(v * v for v in vec))
        if norm > 0:
            vec = [v / norm for v in vec]
        return vec

    def __call__(self, input: Documents) -> Embeddings:
        return [self._embed_one(doc) for doc in input]


# ── Data Types ───────────────────────────────────────────────


@dataclass
class RetrievalResult:
    """A single retrieval hit with score and metadata."""
    chunk_id: str
    content: str
    score: float
    metadata: dict


# ── Vector Store ─────────────────────────────────────────────


class VectorStore:
    """ChromaDB-backed dense retrieval with local embeddings."""

    def __init__(self, persist_directory: str = "./chroma_store"):
        Path(persist_directory).mkdir(parents=True, exist_ok=True)
        self.client = chromadb.Client()
        self.embedding_fn = LocalEmbedding()
        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
            embedding_function=self.embedding_fn,
        )

    def add_chunks(self, chunks: list[Chunk]) -> None:
        """Embed and upsert chunks."""
        if not chunks:
            return
        batch_size = 100
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i : i + batch_size]
            self.collection.upsert(
                ids=[c.chunk_id for c in batch],
                documents=[c.content for c in batch],
                metadatas=[c.metadata for c in batch],
            )

    def query(self, query_text: str, top_k: int = 10) -> list[RetrievalResult]:
        """Find top-k most similar chunks."""
        count = self.collection.count()
        if count == 0:
            return []

        results = self.collection.query(
            query_texts=[query_text],
            n_results=min(top_k, count),
            include=["documents", "metadatas", "distances"],
        )

        hits: list[RetrievalResult] = []
        for idx in range(len(results["ids"][0])):
            distance = results["distances"][0][idx]
            similarity = 1.0 - distance
            hits.append(RetrievalResult(
                chunk_id=results["ids"][0][idx],
                content=results["documents"][0][idx],
                score=similarity,
                metadata=results["metadatas"][0][idx] if results["metadatas"] else {},
            ))
        return hits

    @property
    def count(self) -> int:
        return self.collection.count()
