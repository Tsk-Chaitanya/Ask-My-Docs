"""
Token-aware document chunker with overlap.

WHY TOKEN-BASED (not character-based):
  LLMs process tokens, not characters. "Hello" is 1 token, but
  "Backpropagation" is 3 tokens. A 2000-character chunk could be
  anywhere from 300 to 600 tokens — unpredictable. Token-based
  chunking gives us precise control over context window usage.

WHY OVERLAP:
  Without overlap, a sentence that spans a chunk boundary gets
  split in half. The first chunk has the beginning, the second has
  the end, and neither has the full meaning. A 100-token overlap
  ensures that boundary sentences appear complete in at least one
  chunk.

WHY DETERMINISTIC IDs:
  We hash the chunk content to create IDs. If you re-ingest the
  same document, the same chunks get the same IDs, so ChromaDB
  upserts instead of creating duplicates.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field

from src.ingestion.loader import Document


@dataclass
class Chunk:
    """A single chunk ready for embedding and indexing."""

    chunk_id: str
    content: str
    token_count: int
    metadata: dict = field(default_factory=dict)

    def __repr__(self) -> str:
        return f"Chunk(id={self.chunk_id[:16]}..., tokens={self.token_count})"


class TokenChunker:
    """
    Splits documents into token-bounded chunks with overlap.

    Default: 600 tokens per chunk, 100 token overlap.
    This targets the 500-800 range while leaving room for
    the overlap tokens.
    """

    def __init__(
        self,
        chunk_size: int = 600,
        chunk_overlap: int = 100,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """Simple word-level tokenizer. ~1.3 words per token is a common ratio."""
        return re.findall(r"\S+", text)

    @staticmethod
    def _detokenize(tokens: list[str]) -> str:
        return " ".join(tokens)

    def _make_chunk_id(self, source: str, index: int, text: str) -> str:
        """
        Deterministic ID = hash of content + index.
        Same content always produces the same ID.
        """
        raw = f"{source}::{index}::{text}"
        return hashlib.sha256(raw.encode()).hexdigest()[:16]

    def chunk_document(self, doc: Document) -> list[Chunk]:
        """Split a single Document into overlapping token-bounded chunks."""
        tokens = self._tokenize(doc.content)
        source = doc.metadata.get("source", "unknown")

        # Short document? Return as a single chunk.
        if len(tokens) <= self.chunk_size:
            return [
                Chunk(
                    chunk_id=self._make_chunk_id(source, 0, doc.content),
                    content=doc.content,
                    token_count=len(tokens),
                    metadata={**doc.metadata, "chunk_index": 0},
                )
            ]

        # Slide a window with step = chunk_size - overlap
        chunks: list[Chunk] = []
        step = self.chunk_size - self.chunk_overlap  # 600 - 100 = 500

        for idx, start in enumerate(range(0, len(tokens), step)):
            end = min(start + self.chunk_size, len(tokens))
            chunk_tokens = tokens[start:end]
            text = self._detokenize(chunk_tokens)

            chunks.append(
                Chunk(
                    chunk_id=self._make_chunk_id(source, idx, text),
                    content=text,
                    token_count=len(chunk_tokens),
                    metadata={**doc.metadata, "chunk_index": idx},
                )
            )

            # Don't create a tiny trailing chunk
            if end >= len(tokens):
                break

        return chunks

    def chunk_documents(self, docs: list[Document]) -> list[Chunk]:
        """Chunk a batch of documents."""
        all_chunks: list[Chunk] = []
        for doc in docs:
            all_chunks.extend(self.chunk_document(doc))
        return all_chunks
