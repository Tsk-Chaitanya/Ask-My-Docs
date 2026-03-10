# Retrieval Augmented Generation: A Survey

## Introduction

Retrieval Augmented Generation (RAG) is a paradigm that combines information retrieval with
text generation to produce more accurate, grounded, and verifiable outputs from large language
models. The core idea is simple: instead of relying solely on the knowledge stored in a model's
parameters during pre-training, RAG systems fetch relevant documents from an external knowledge
base at inference time and include them as context in the prompt.

This approach addresses several fundamental limitations of standalone LLMs. First, LLMs have a
knowledge cutoff date — they cannot access information published after their training data was
collected. Second, LLMs are prone to hallucination, generating plausible-sounding but factually
incorrect statements. Third, LLMs cannot easily cite their sources, making it difficult to verify
their claims.

## Architecture

A typical RAG system consists of three stages:

1. **Indexing**: Documents are split into chunks, converted to vector embeddings, and stored in
   a vector database. This is a one-time (or periodic) offline process.

2. **Retrieval**: When a user submits a query, the system converts it to an embedding and
   performs a similarity search against the vector database to find the most relevant chunks.
   Common retrieval strategies include dense retrieval (using embeddings), sparse retrieval
   (using BM25 or TF-IDF), and hybrid approaches that combine both.

3. **Generation**: The retrieved chunks are inserted into the LLM's prompt as context, and the
   model generates an answer grounded in this evidence. Well-designed systems enforce citations
   back to specific source chunks.

## Chunking Strategies

Document chunking is a critical but often underestimated step. Chunks that are too small lose
context; chunks that are too large dilute relevance and waste the model's context window.
Production systems typically use chunks of 500-800 tokens with 50-100 token overlap between
consecutive chunks to preserve context at boundaries.

Token-based chunking is preferred over character-based chunking because LLMs process tokens, not
characters. A 500-token chunk maps predictably to context window usage, while a 2000-character
chunk may vary significantly in token count depending on the content.

## Evaluation

Evaluating RAG systems requires measuring multiple dimensions:

- **Faithfulness**: Are the generated claims actually supported by the retrieved chunks?
- **Answer Relevance**: Does the answer address the user's question?
- **Context Relevance**: Are the retrieved chunks relevant to the query?
- **Citation Accuracy**: Do the cited sources actually support the claims they're attached to?

The RAGAS framework provides automated metrics for these dimensions, but human evaluation
remains the gold standard for assessing output quality.
