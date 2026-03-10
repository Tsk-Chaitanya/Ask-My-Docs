# Ask My Docs

A production-quality Retrieval Augmented Generation (RAG) system that lets you ask natural language questions about your documents and get accurate, cited answers powered by Claude.

Built as a learning project to understand every layer of the RAG stack — from document chunking to hybrid retrieval to cross-encoder reranking — with no shortcuts or black-box abstractions.

![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue)
![License MIT](https://img.shields.io/badge/license-MIT-green)

---

## What It Does

Drop in your documents (PDF, Markdown, plain text), run the ingestion pipeline, start the server, and ask questions through a clean web UI. The system:

1. **Chunks** your documents into token-bounded overlapping pieces (600 tokens, 100 overlap)
2. **Indexes** them in both a vector store (ChromaDB) and a BM25 sparse index
3. **Retrieves** candidates using hybrid search (BM25 + vector similarity, fused with Reciprocal Rank Fusion)
4. **Reranks** the top candidates with a cross-encoder model for precise relevance scoring
5. **Generates** a cited answer using Claude, with strict citation enforcement — it refuses to answer rather than hallucinate
6. **Remembers** your conversation so you can ask follow-up questions naturally

---

## Architecture

```
User Question
     |
     v
+--------------------+
| Conversation Memory | -- adds context from prior turns for follow-ups
+--------------------+
     |
     v
+-----------+     +-------------+
| BM25      |     | Vector      |
| (keyword  |     | (semantic   |
|  matching)|     |  similarity)|
+-----------+     +-------------+
     \                 /
      v               v
  +-------------------------+
  | Reciprocal Rank Fusion  | -- combines rankings without score calibration
  +-------------------------+
           |
           v
  +-------------------+
  | Cross-Encoder     | -- re-scores (query, chunk) pairs jointly
  | Reranker          |
  +-------------------+
           |
           v
  +-------------------+
  | Claude (LLM)      | -- generates answer with [chunk_id] citations
  | Citation Enforced  |
  +-------------------+
           |
           v
     Cited Answer
```

---

## Project Structure

```
Ask-My-Docs/
├── config/
│   └── prompts.yaml              # Versioned prompt templates (v1.0.0)
├── data/
│   ├── documents/                # Your source documents go here
│   │   ├── bm25.txt
│   │   ├── embeddings.md
│   │   └── rag_survey.md
│   └── eval/
│       └── test_questionnaire.md # 23 evaluation questions
├── src/
│   ├── ingestion/
│   │   ├── loader.py             # PDF/MD/TXT loaders → list[Document]
│   │   ├── chunker.py            # Token-aware chunker with overlap
│   │   └── pipeline.py           # Orchestrates load → chunk → index
│   ├── retrieval/
│   │   ├── vector.py             # ChromaDB vector store + local embeddings
│   │   ├── bm25.py               # BM25Okapi sparse retrieval index
│   │   ├── hybrid.py             # RRF fusion of BM25 + vector results
│   │   └── reranker.py           # Cross-encoder reranker (ms-marco-MiniLM)
│   ├── generation/
│   │   ├── prompt_manager.py     # Loads versioned prompts from YAML
│   │   └── generator.py          # Claude-powered answer generation
│   └── api/
│       ├── server.py             # FastAPI server with conversation memory
│       └── ui.html               # Chat-style web interface
├── tests/
│   ├── test_phase1.py            # 17 tests — loaders, chunker, vector, generator
│   ├── test_phase2.py            # 12 tests — BM25, hybrid, reranker
│   └── test_phase3.py            # 12 tests — session memory, contextual queries
├── requirements.txt
└── README.md
```

---

## How We Built It

### Phase 1: Foundations

Built the core RAG pipeline from scratch:

- **Document Loaders** (`loader.py`): Uniform `Document` dataclass for all file types. PDF loader splits per-page for better citation granularity. Markdown loader strips YAML front-matter.
- **Token-Aware Chunker** (`chunker.py`): Uses 600-token chunks with 100-token overlap. Token-based (not character-based) because LLMs process tokens — a 2000-character chunk could be anywhere from 300 to 600 tokens. Deterministic chunk IDs via content hashing so re-ingestion upserts instead of duplicating.
- **Vector Store** (`vector.py`): ChromaDB with a local bag-of-words embedding function. Stores chunks with metadata for source tracking. Returns `RetrievalResult` dataclass used across all retrieval modules.
- **Answer Generator** (`generator.py`): Sends retrieved chunks to Claude with a system prompt that enforces citation using `[chunk_id]` references. Refuses to answer when chunks are insufficient — the key safety feature. Falls back to demo mode when no API key is set.
- **Prompt Management** (`prompt_manager.py` + `prompts.yaml`): Prompts stored in versioned YAML, not hardcoded strings. Supports A/B testing and rollback via git history.
- **17 passing tests** covering all components.

### Phase 2: Production Retrieval

Upgraded from vector-only search to a three-stage retrieval pipeline:

- **BM25 Index** (`bm25.py`): Complements vector search by excelling at exact keyword matching — technical terms, acronyms, proper nouns. Uses the `rank_bm25` library with word-boundary tokenization. Supports save/load for persistence.
- **Hybrid Retriever** (`hybrid.py`): Fuses BM25 and vector results using Reciprocal Rank Fusion (RRF). Why RRF instead of score averaging? BM25 scores range 0–15, cosine similarity ranges 0–1 — averaging would let BM25 dominate. RRF uses only rank positions, so no calibration needed. Formula: for each chunk, sum `weight / (k + rank)` across retrievers, where k=60.
- **Cross-Encoder Reranker** (`reranker.py`): Uses `cross-encoder/ms-marco-MiniLM-L-6-v2` to jointly process (query, chunk) pairs. Much more accurate than bi-encoder similarity but too slow for the full corpus — so we use it only on the top candidates from hybrid retrieval. Filters by configurable relevance threshold.
- **FastAPI Server** (`server.py`): REST API serving the full pipeline with a web UI.
- **12 additional tests** (including mock-based tests for environments without PyTorch).

### Phase 3: UI & Conversation Memory

Made the system feel polished and conversational:

- **Chat Interface** (`ui.html`): Soft warm design with teal accents. Chat-style conversation history, animated loading dots, copy-to-clipboard, hover previews on source cards showing chunk text, "New Chat" button to reset.
- **Conversation Memory** (`server.py`): LRU-bounded `SessionStore` (max 100 sessions, 20 turns each). Follow-up questions are context-aware — asking "tell me more about that" works because the retriever gets conversation context prepended to the query.
- **Context-Aware Retrieval**: The last 3 conversation turns are summarized and prepended to the retrieval query, so vague follow-ups still find relevant chunks.
- **12 more tests** covering session store (add, retrieve, clear, LRU eviction, history trimming) and contextual query building.

---

## Getting Started

### Prerequisites

- Python 3.10+
- An Anthropic API key (for Claude-powered generation)

### Installation

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/Ask-My-Docs.git
cd Ask-My-Docs

# Create and activate virtual environment
python -m venv venv

# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Set your API key

```bash
# Windows PowerShell
$env:ANTHROPIC_API_KEY = "your-key-here"

# macOS/Linux
export ANTHROPIC_API_KEY="your-key-here"
```

### Run

```bash
# 1. Index your documents
python -m src.ingestion.pipeline --source data/documents/

# 2. Start the server
python -m src.api.server

# 3. Open http://localhost:8000 and start asking questions
```

### Run Tests

```bash
python -m pytest tests/ -v
# Expected: 41 passed (3 skipped if PyTorch not installed for live reranker tests)
```

---

## Adding Your Own Documents

1. Drop `.txt`, `.md`, or `.pdf` files into `data/documents/`
2. Re-run ingestion: `python -m src.ingestion.pipeline --source data/documents/`
3. Restart the server: `python -m src.api.server`

Tip: Use descriptive filenames (e.g., `neural-networks-intro.md` instead of `notes.txt`) — the filename shows up in source citations.

---

## Test Questionnaire

A 23-question evaluation suite is included at `data/eval/test_questionnaire.md`, covering six categories:

### A. Factual Recall (single-document retrieval)
1. What is BM25?
2. What are the two tunable parameters in BM25?
3. What is a cross-encoder?
4. Name three vector databases mentioned in the documents.
5. What is the typical chunk size recommended for production RAG systems?

### B. Comparison & Synthesis (multi-document retrieval)
6. How does BM25 differ from dense retrieval using embeddings?
7. What are the tradeoffs between bi-encoders and cross-encoders?
8. Why is hybrid retrieval better than using BM25 or embeddings alone?
9. Compare cosine similarity and dot product for vector search.

### C. Follow-Up Conversations (conversation memory)
10. Ask "What is RAG?" → "What are the three stages you mentioned?" → "Tell me more about the second stage."
11. Ask "Explain BM25 scoring." → "What are its limitations?" → "How do embeddings solve those limitations?"

### D. Edge Cases & Safety (citation enforcement)
12. What is the capital of France? *(should decline — not in docs)*
13. Who invented the transformer architecture? *(should hedge or decline)*
14. Explain quantum computing. *(should decline)*
15. What are embeddings used for in cooking? *(should decline)*

### E. Keyword-Heavy Queries (BM25 strength)
16. HNSW graph
17. RAGAS framework evaluation
18. k1 parameter b parameter BM25
19. Sentence-BERT contrastive learning

### F. Semantic Queries (embedding strength)
20. How can I make my search system find similar meanings, not just matching words?
21. What happens when a language model makes up facts?
22. How do I split long documents for an AI system?
23. What's a good way to check if an AI's answer is actually correct?

**Scoring:** Rate each answer 1–5 (5 = perfect, accurate, well-cited). Target: average 4+.

---

## How to Scale This

### Short-Term Improvements

| Area | What to Do | Effort |
|------|-----------|--------|
| **More doc types** | Add loaders for `.docx`, `.csv`, `.html` in `loader.py` | Low |
| **Better embeddings** | Replace `LocalEmbedding` in `vector.py` with `sentence-transformers` bi-encoder (e.g., `all-MiniLM-L6-v2`) | Medium |
| **Query expansion** | Rephrase user questions multiple ways before retrieval (improves recall) | Medium |
| **Streaming answers** | Use Claude's streaming API + Server-Sent Events for real-time response display | Medium |

### Medium-Term Scaling

| Area | What to Do | Effort |
|------|-----------|--------|
| **Multi-hop reasoning** | Break complex questions into sub-questions, retrieve for each, synthesize | High |
| **Persistent sessions** | Move `SessionStore` from in-memory to SQLite or Redis | Medium |
| **Document management UI** | Upload/delete documents through the web interface instead of file system | Medium |
| **Evaluation automation** | Run the test questionnaire programmatically and auto-score with an LLM judge | Medium |
| **Metadata filtering** | Filter retrieval by source, date, or document type before ranking | Low |

### Production-Grade Scaling

| Area | What to Do | Effort |
|------|-----------|--------|
| **Vector DB upgrade** | Migrate from ChromaDB to Weaviate or Pinecone for millions of documents | High |
| **GPU reranking** | Run the cross-encoder on GPU for faster inference at scale | Medium |
| **Async ingestion** | Background job queue (Celery/RQ) for document processing without blocking | High |
| **Auth & multi-tenancy** | User accounts with isolated document collections | High |
| **Caching** | Cache frequent queries and their answers (TTL-based invalidation on re-ingestion) | Medium |
| **Observability** | Add logging, latency tracking, and retrieval quality metrics (RAGAS) | Medium |
| **Containerization** | Dockerfile + docker-compose for one-command deployment | Low |

### Architecture Evolution Path

```
Current (Phase 3)              Near-term                    Production
─────────────────              ─────────                    ──────────
ChromaDB (in-memory)    →   ChromaDB (persistent)    →   Weaviate / Pinecone
Local embeddings        →   Sentence-BERT            →   OpenAI / Cohere embeddings
In-memory sessions      →   SQLite sessions          →   Redis sessions
Single-user             →   Auth + multi-tenant      →   SSO + RBAC
Manual ingestion        →   Watch folder + auto      →   API-driven ingestion
3 doc types             →   10+ doc types            →   Any format (Unstructured.io)
FastAPI (dev server)    →   Gunicorn + Nginx         →   Kubernetes + autoscaling
```

---

## Tech Stack

| Component | Technology | Why |
|-----------|-----------|-----|
| Language | Python 3.10+ | Ecosystem support for ML/NLP |
| Vector Store | ChromaDB | Lightweight, easy setup, HNSW-based |
| Sparse Retrieval | rank-bm25 | Simple, fast BM25Okapi implementation |
| Reranker | sentence-transformers (CrossEncoder) | ms-marco-MiniLM-L-6-v2, accurate relevance scoring |
| LLM | Claude (Anthropic API) | Strong instruction following, citation support |
| API Framework | FastAPI | Async, auto-docs, Pydantic validation |
| Prompt Config | YAML | Versioned, git-trackable prompt management |
| Testing | pytest | 41 tests across 3 phases |

---

## Key Design Decisions

**Why token-based chunking?** LLMs process tokens, not characters. A 2000-character chunk could be 300–600 tokens depending on content. Token-based chunking gives precise control over context window usage.

**Why overlap between chunks?** Without overlap, sentences at chunk boundaries get split in half. A 100-token overlap ensures boundary sentences appear complete in at least one chunk.

**Why RRF instead of score averaging?** BM25 scores range 0–15, cosine similarity ranges 0–1. Linear combination would let BM25 dominate. RRF uses only rank positions, requiring no score calibration.

**Why a cross-encoder reranker?** Bi-encoders encode query and document independently — fast but approximate. Cross-encoders process the pair jointly, catching nuanced relevance signals. We use the cross-encoder only on the top candidates (10–20) from hybrid retrieval to balance accuracy and speed.

**Why citation enforcement?** The generator refuses to answer when chunks are insufficient rather than hallucinating. Every claim must cite a `[chunk_id]`. This is the core safety feature — grounded answers or no answers.

**Why conversation memory?** Follow-up questions like "tell me more" or "what about the downsides?" are natural in conversation. The session store keeps the last 20 turns and prepends context to retrieval queries so vague follow-ups still find relevant chunks.

---

## License

MIT

---

## Acknowledgments

Built by [Sai Krishna Chaitanya Thommandru](https://github.com/Tsk-Chaitanya) as a hands-on deep dive into production RAG systems. Each phase was designed to understand a specific layer of the stack, from document processing fundamentals to hybrid retrieval to conversational AI.
