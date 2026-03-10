"""
End-to-end ingestion pipeline.

This is the entry point for Phase 1: load docs, chunk them,
and index into ChromaDB for retrieval.

Usage:
    python -m src.ingestion.pipeline --source data/documents/
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from src.ingestion.loader import load_directory
from src.ingestion.chunker import TokenChunker
from src.retrieval.vector import VectorStore
from src.retrieval.bm25 import BM25Index


def run_ingestion(
    source_dir: str,
    persist_dir: str = "./chroma_store",
    chunk_size: int = 600,
    chunk_overlap: int = 100,
) -> dict:
    """
    Full ingestion pipeline: load → chunk → index.

    Returns a summary dict with counts.
    """
    print(f"\n{'═' * 55}")
    print(f"  Ask My Docs — Ingestion Pipeline")
    print(f"{'═' * 55}\n")

    # ── Step 1: Load documents ───────────────────────────────
    print(f"  📂 Loading documents from: {source_dir}")
    docs = load_directory(source_dir)
    print(f"     Found {len(docs)} document(s)\n")

    if not docs:
        print("  ⚠  No documents found. Add files to the source directory.")
        return {"documents": 0, "chunks": 0}

    # ── Step 2: Chunk ────────────────────────────────────────
    print(f"  ✂️  Chunking (size={chunk_size}, overlap={chunk_overlap}) ...")
    chunker = TokenChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = chunker.chunk_documents(docs)
    print(f"     Produced {len(chunks)} chunk(s)\n")

    # ── Step 3: Index into vector store ──────────────────────
    print(f"  🔢 Indexing into ChromaDB ...")
    t0 = time.time()
    store = VectorStore(persist_directory=persist_dir)
    store.add_chunks(chunks)
    elapsed = time.time() - t0
    print(f"     Indexed {store.count} chunks in {elapsed:.1f}s\n")

    # ── Step 4: Build BM25 index ─────────────────────────────
    print(f"  📊 Building BM25 index ...")
    bm25 = BM25Index()
    bm25.build(chunks)
    bm25.save(Path(persist_dir) / "bm25_index.json")
    print(f"     Done\n")

    # ── Summary ──────────────────────────────────────────────
    summary = {
        "documents": len(docs),
        "chunks": len(chunks),
        "indexed": store.count,
    }

    print(f"{'═' * 55}")
    print(f"  ✅ Done: {len(docs)} docs → {len(chunks)} chunks → indexed")
    print(f"{'═' * 55}\n")

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest documents")
    parser.add_argument("--source", required=True, help="Directory with source documents")
    parser.add_argument("--persist", default="./chroma_store", help="ChromaDB directory")
    args = parser.parse_args()

    run_ingestion(args.source, args.persist)
