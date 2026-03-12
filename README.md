# Ask My Docs

**Live demo:** [web-production-1983.up.railway.app](https://web-production-1983.up.railway.app)

A production-quality Retrieval Augmented Generation (RAG) system that lets you ask natural language questions about your documents and get accurate, cited answers powered by Claude.

Built to understand every layer of the RAG stack — from document chunking to hybrid retrieval to cross-encoder reranking — with no shortcuts or black-box abstractions.

---

## Why This Project

Most RAG tutorials stop at "embed chunks → cosine search → send to LLM." That's a demo, not a system. This project implements what production RAG actually requires:

- **Hybrid retrieval** that catches both semantic meaning *and* exact keyword matches
- **Cross-encoder reranking** that dramatically improves precision on the narrowed candidate set
- **Citation enforcement** where the system refuses to answer rather than hallucinate
- **Versioned prompts** tracked in config files, not buried in source code
- **An evaluation pipeline** with golden datasets and CI gating that fails the build if quality drops

---

## Architecture

```
                    ┌──────────────────────────────────┐
                    │         Document Ingestion        │
                    │  PDF / Markdown / Text → Chunks   │
                    │  (600 tokens, 100 overlap)        │
                    └──────────┬───────────────────────┘
                               │
                 ┌─────────────┼─────────────┐
                 ▼                           ▼
        ┌────────────────┐          ┌────────────────┐
        │   ChromaDB     │          │   BM25 Index   │
        │ Vector Search  │          │ Keyword Search  │
        └───────┬────────┘          └───────┬────────┘
                │                           │
                └─────────┬─────────────────┘
                          ▼
                ┌──────────────────┐
                │  Reciprocal Rank │
                │  Fusion (RRF)    │
                └────────┬─────────┘
                         ▼
                ┌──────────────────┐
                │  Cross-Encoder   │
                │  Re-ranker       │
                └────────┬─────────┘
                         ▼
                ┌──────────────────┐
                │  Claude LLM      │
                │  + Citations     │
                │  + Decline if    │
                │    unsupported   │
                └──────────────────┘
```

---

## Key Features

**Hybrid Retrieval** — Combines dense vector search (semantic similarity) with BM25 sparse search (exact term matching) via Reciprocal Rank Fusion. This means queries like "BERT fine-tuning learning rate" find documents by both meaning and exact keywords.

**Cross-Encoder Reranking** — After hybrid retrieval narrows candidates to ~20, a cross-encoder model (`ms-marco-MiniLM-L-6-v2`) re-scores each (query, chunk) pair jointly. This is significantly more accurate than bi-encoder similarity alone.

**Citation Enforcement** — Every claim in the generated answer must cite a `[chunk_id]`. If the retrieved chunks don't support an answer, the system explicitly declines rather than hallucinating.

**Versioned Prompts** — All LLM prompts live in `config/prompts.yaml` with version tracking. No prompts are hardcoded in source files. You can diff prompt changes in git, A/B test versions, and roll back if quality drops.

**Evaluation Pipeline** — A golden dataset of verified Q&A pairs, automated faithfulness/relevance/citation metrics, and a CI gate that blocks deployment when quality falls below threshold.

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Vector Store | ChromaDB |
| Sparse Retrieval | rank-bm25 (BM25Okapi) |
| Re-ranker | sentence-transformers CrossEncoder |
| LLM | Anthropic Claude (via API) |
| Prompt Config | YAML with version tracking |
| API Server | FastAPI + Uvicorn |
| Evaluation | Custom metrics (faithfulness, citation accuracy, retrieval recall) |
| Testing | pytest |
| CI | GitHub Actions (`eval_gate.yml`) |

---

## Project Structure

```
ask-my-docs/
├── config/
│   └── prompts.yaml              # Versioned prompt templates (v1.0.0)
├── data/
│   ├── documents/                # Source documents (PDF, MD, TXT)
│   └── eval/
│       ├── golden_dataset.json   # Curated evaluation Q&A pairs
│       └── test_questionnaire.md # Manual testing guide (23 questions)
├── src/
│   ├── ingestion/
│   │   ├── loader.py             # Multi-format document loading
│   │   ├── chunker.py            # Token-aware chunking with overlap
│   │   └── pipeline.py           # Ingestion orchestration
│   ├── retrieval/
│   │   ├── vector.py             # ChromaDB dense retrieval
│   │   ├── bm25.py               # BM25 sparse retrieval
│   │   ├── hybrid.py             # RRF fusion of both retrievers
│   │   └── reranker.py           # Cross-encoder re-ranking
│   ├── generation/
│   │   ├── prompt_manager.py     # YAML prompt loading & rendering
│   │   └── generator.py          # Citation-enforced answer generation
│   ├── evaluation/
│   │   ├── golden_dataset.py     # Eval dataset loader & validator
│   │   ├── metrics.py            # Coverage, citation & decline metrics
│   │   └── run_eval.py           # Eval runner with CI gating
│   └── api/
│       ├── server.py             # FastAPI endpoints
│       └── ui.html               # Web interface
├── tests/
│   ├── test_phase1.py            # 17 tests — loaders, chunker, vector, generator
│   ├── test_phase2.py            # 12 tests — BM25, hybrid, reranker
│   ├── test_phase3.py            # 12 tests — session memory, contextual queries
│   └── test_phase4.py            # 14 tests — golden dataset, metrics, aggregation
├── .github/
│   └── workflows/
│       └── eval_gate.yml         # GitHub Actions CI pipeline
├── requirements.txt
└── README.md
```

---

## Quick Start

### Prerequisites

- Python 3.10+
- An [Anthropic API key](https://console.anthropic.com/) (optional — system runs in demo mode without it)

### Setup

```bash
# Clone the repository
git clone https://github.com/Tsk-Chaitanya/Ask-My-Docs.git
cd Ask-My-Docs

# Create and activate virtual environment
python -m venv venv
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Set API key (optional — enables real Claude answers)
# Windows:
set ANTHROPIC_API_KEY=your-key-here
# macOS/Linux:
export ANTHROPIC_API_KEY=your-key-here
```

### Ingest Documents

```bash
# Index the sample CS/AI documents
python -m src.ingestion.pipeline --source data/documents/
```

### Run the Server

```bash
python -m src.api.server
# Open http://localhost:8000 in your browser
```

### Run Tests

```bash
python -m pytest tests/ -v
```

### Run Evaluation

```bash
# Interactive report
python -m src.evaluation.run_eval --dataset data/eval/golden_dataset.json

# CI mode — exits non-zero if quality drops below threshold
python -m src.evaluation.run_eval --dataset data/eval/golden_dataset.json --ci --threshold 0.85
```

---

## How It Works

### Phase 1: Ingestion

Documents (PDF, Markdown, plain text) are loaded, split into **600-word chunks with 100-word overlap**, and indexed into ChromaDB. The overlap ensures sentences at chunk boundaries aren't lost. Chunk IDs are deterministic (content-hashed), so re-ingesting the same document updates rather than duplicates.

### Phase 2: Retrieval & Reranking

When you ask a question:

1. **Hybrid retrieval** runs BM25 and vector search in parallel, then fuses results using Reciprocal Rank Fusion (RRF). RRF uses only rank positions, not raw scores, which elegantly sidesteps the problem of BM25 scores (0–15 range) being incompatible with cosine similarity (0–1 range).

2. **Cross-encoder reranking** takes the top ~20 fused results and re-scores them by processing each (query, chunk) pair jointly through a transformer model. This catches nuanced relevance that bi-encoders miss.

### Phase 3: Generation & Evaluation

The top-ranked chunks are sent to Claude with a prompt requiring `[chunk_id]` citations for every claim. If the chunks don't support an answer, the system responds with "I don't have enough information" instead of guessing.

The evaluation pipeline measures faithfulness, answer relevance, and citation accuracy against a golden dataset. The CI pipeline automatically blocks merges if quality metrics fall below the configured threshold.

---

## Adding Your Own Documents

Drop PDF, Markdown, or text files into `data/documents/` and re-run ingestion:

```bash
python -m src.ingestion.pipeline --source data/documents/
```

The system works with any domain — legal contracts, medical papers, internal wikis, technical documentation.

---

## Deployment

The project ships with a `railway.toml` config for free deployment to [Railway](https://railway.app) — no credit card required, $5 free credit/month.

### Deploy to Railway (Free)

1. Push this repository to GitHub (already done).
2. Go to [railway.app](https://railway.app) and sign in with GitHub.
3. Click **New Project** → **Deploy from GitHub repo** → select `Ask-My-Docs`.
4. Railway auto-detects Python and uses `railway.toml` for the start command.
5. Open the service → **Variables** tab → add:
   - Key: `ANTHROPIC_API_KEY`
   - Value: your key from [console.anthropic.com](https://console.anthropic.com)
6. Click **Deploy**. Railway builds and starts the server automatically.

Once live, go to **Settings → Networking → Generate Domain** to get a public URL (e.g. `https://ask-my-docs.up.railway.app`).

### Per-User API Key (UI)

Users can supply their **own** Anthropic API key directly in the interface — no server-side key required. Click the **🔑 API Key** button in the top-right corner, paste a key, and save. The key is stored in browser `localStorage` and sent with every request. The server uses it instead of its own environment variable, so each user's queries are billed to their own account.

### Notes for Self-Hosting

- The server reads `PORT` from the environment automatically, so it works on any platform that injects a port (Railway, Fly.io, etc.).
- ChromaDB uses an in-memory store by default. For persistent storage across deploys, mount a persistent volume and point `CHROMA_PATH` to it.
- Cold starts after inactivity may take 20–30 seconds while the reranker model loads — this is normal on free tiers.

---

## License

MIT
