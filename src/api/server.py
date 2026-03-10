"""
FastAPI server for Ask My Docs.

Phase 3 pipeline:
  - hybrid retrieval (BM25 + vector + RRF) -> reranker -> generation
  - conversation memory (session-based follow-up questions)
  - enriched API responses (chunk content for UI previews)

Usage:
    python -m src.api.server
    Open http://localhost:8000
"""

from __future__ import annotations

import uuid
from collections import OrderedDict
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from src.ingestion.loader import load_directory
from src.ingestion.chunker import TokenChunker
from src.retrieval.vector import VectorStore
from src.retrieval.bm25 import BM25Index
from src.retrieval.hybrid import HybridRetriever
from src.retrieval.reranker import Reranker
from src.generation.generator import AnswerGenerator


# ── App & State ──────────────────────────────────────────────

app = FastAPI(title="Ask My Docs", version="3.0.0")

_state: dict = {}


# ── Conversation Memory ─────────────────────────────────────
# Simple in-memory store: session_id -> list of (question, answer) pairs.
# Capped at MAX_SESSIONS to prevent unbounded memory growth.

MAX_SESSIONS = 100
MAX_HISTORY_PER_SESSION = 20


class SessionStore:
    """LRU-bounded conversation memory."""

    def __init__(self, max_sessions: int = MAX_SESSIONS):
        self._sessions: OrderedDict[str, list[dict]] = OrderedDict()
        self._max = max_sessions

    def get_history(self, session_id: str) -> list[dict]:
        if session_id in self._sessions:
            self._sessions.move_to_end(session_id)
            return self._sessions[session_id]
        return []

    def add_turn(self, session_id: str, question: str, answer: str) -> None:
        if session_id not in self._sessions:
            if len(self._sessions) >= self._max:
                self._sessions.popitem(last=False)  # evict oldest
            self._sessions[session_id] = []
        self._sessions.move_to_end(session_id)
        history = self._sessions[session_id]
        history.append({"question": question, "answer": answer})
        # Trim to keep memory bounded
        if len(history) > MAX_HISTORY_PER_SESSION:
            self._sessions[session_id] = history[-MAX_HISTORY_PER_SESSION:]

    def clear(self, session_id: str) -> None:
        self._sessions.pop(session_id, None)


sessions = SessionStore()


# ── Request / Response Models ────────────────────────────────


class QueryRequest(BaseModel):
    question: str
    top_k: int = 5
    session_id: str = ""


class ChunkInfo(BaseModel):
    chunk_id: str
    source: str
    score: float
    content: str = ""  # Phase 3: chunk text for UI previews


class QueryResponse(BaseModel):
    answer: str
    citations: list[str]
    chunks_used: list[ChunkInfo]
    declined: bool
    prompt_version: str


class StatsResponse(BaseModel):
    documents_loaded: int
    chunks_indexed: int
    retrieval_mode: str


class SessionClearRequest(BaseModel):
    session_id: str


# ── Pipeline Bootstrap ───────────────────────────────────────


def get_pipeline() -> dict:
    """Lazy-initialize the full RAG pipeline (once)."""
    if _state:
        return _state

    source_dir = "data/documents"
    print(f"\n  Loading documents from {source_dir} ...")
    docs = load_directory(source_dir)
    print(f"  Found {len(docs)} document(s)")

    chunker = TokenChunker(chunk_size=600, chunk_overlap=100)
    chunks = chunker.chunk_documents(docs)
    print(f"  Produced {len(chunks)} chunk(s)")

    # Vector store
    store = VectorStore()
    store.add_chunks(chunks)
    print(f"  Indexed {store.count} chunks into ChromaDB")

    # BM25 index
    bm25 = BM25Index()
    bm25.build(chunks)
    print(f"  Built BM25 index over {len(chunks)} chunks")

    # Hybrid retriever
    hybrid = HybridRetriever(vector_store=store, bm25_index=bm25)

    # Cross-encoder reranker
    print("  Loading cross-encoder reranker ...")
    reranker = Reranker()
    print("  Reranker ready")

    # Generator
    generator = AnswerGenerator()

    _state["store"] = store
    _state["bm25"] = bm25
    _state["hybrid"] = hybrid
    _state["reranker"] = reranker
    _state["generator"] = generator
    _state["docs_count"] = len(docs)
    _state["chunks_count"] = len(chunks)

    print(f"  Pipeline ready (hybrid + reranker)\n")
    return _state


# ── Helper: build context-aware query ────────────────────────


def _build_contextual_query(question: str, history: list[dict]) -> str:
    """
    If there's conversation history, prepend a brief summary so the
    retriever can find relevant chunks even for vague follow-ups like
    'Tell me more about that' or 'What are the downsides?'.
    """
    if not history:
        return question

    # Take the last 3 turns as context
    recent = history[-3:]
    context_parts = [f"Q: {t['question']}" for t in recent]
    context_summary = " | ".join(context_parts)

    return f"[Conversation context: {context_summary}] {question}"


# ── Endpoints ────────────────────────────────────────────────


@app.get("/", response_class=HTMLResponse)
async def serve_ui():
    """Serve the web interface."""
    ui_path = Path(__file__).parent / "ui.html"
    return HTMLResponse(content=ui_path.read_text(encoding="utf-8"))


@app.post("/api/query", response_model=QueryResponse)
async def query(req: QueryRequest):
    """Full Phase 3 pipeline: context-aware hybrid retrieval -> reranking -> generation."""
    pipeline = get_pipeline()

    hybrid: HybridRetriever = pipeline["hybrid"]
    reranker: Reranker = pipeline["reranker"]
    generator: AnswerGenerator = pipeline["generator"]

    # Get conversation history for context
    session_id = req.session_id or str(uuid.uuid4())
    history = sessions.get_history(session_id)

    # Build a context-aware query for retrieval
    retrieval_query = _build_contextual_query(req.question, history)

    # Step 1: Hybrid retrieval (BM25 + vector, fused with RRF)
    candidates = hybrid.retrieve(retrieval_query, top_k=req.top_k * 2, fetch_k=20)

    # Step 2: Cross-encoder reranking
    top_chunks = reranker.rerank(req.question, candidates, top_k=req.top_k)

    # Step 3: Generate cited answer
    answer = generator.generate(req.question, top_chunks)

    # Save turn to session memory
    sessions.add_turn(session_id, req.question, answer.answer)

    return QueryResponse(
        answer=answer.answer,
        citations=answer.citations,
        chunks_used=[
            ChunkInfo(
                chunk_id=c["chunk_id"],
                source=c["source"],
                score=c["score"],
                content=next(
                    (ch.content for ch in top_chunks if ch.chunk_id == c["chunk_id"]),
                    "",
                ),
            )
            for c in answer.chunks_used
        ],
        declined=answer.declined,
        prompt_version=answer.prompt_version,
    )


@app.get("/api/stats", response_model=StatsResponse)
async def get_stats():
    """Return pipeline statistics."""
    pipeline = get_pipeline()
    return StatsResponse(
        documents_loaded=pipeline["docs_count"],
        chunks_indexed=pipeline["chunks_count"],
        retrieval_mode="hybrid (BM25 + vector + reranker)",
    )


@app.post("/api/session/clear")
async def clear_session(req: SessionClearRequest):
    """Clear conversation memory for a session."""
    sessions.clear(req.session_id)
    return {"status": "cleared"}


# ── Run ──────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api.server:app", host="0.0.0.0", port=8000, reload=True)
