# Vector Embeddings in Information Retrieval

## What Are Embeddings?

Vector embeddings are dense numerical representations of text in a high-dimensional space,
typically ranging from 384 to 1536 dimensions. The key property of a good embedding model is
that semantically similar texts are mapped to nearby points in this space, allowing similarity
to be measured using distance metrics like cosine similarity or dot product.

Modern embedding models like OpenAI's text-embedding-3-small, Sentence-BERT, and Cohere's
embed-v3 are trained on large datasets of text pairs (query, relevant passage) using contrastive
learning objectives. The model learns to pull matching pairs closer together in embedding space
while pushing non-matching pairs apart.

## Bi-Encoders vs Cross-Encoders

There are two main architectures for computing text similarity:

**Bi-encoders** encode the query and document independently into separate vectors, then compute
similarity via dot product or cosine similarity. This is fast because document embeddings can be
pre-computed and cached. However, because the query and document never "see" each other during
encoding, bi-encoders may miss nuanced relevance signals.

**Cross-encoders** concatenate the query and document into a single input and produce a relevance
score directly. This is much more accurate because the model can attend to fine-grained
interactions between query and document tokens. The tradeoff is speed: cross-encoders must
process every (query, document) pair at query time, making them impractical for searching large
collections. They are typically used as re-rankers on a small candidate set (10-50 documents)
retrieved by a faster bi-encoder.

## Vector Databases

Vector databases are specialized storage systems optimized for similarity search over embeddings.
Popular options include:

- **ChromaDB**: Lightweight, easy to set up, good for prototyping and small-to-medium collections.
  Uses HNSW (Hierarchical Navigable Small World) graphs for approximate nearest neighbor search.
- **Weaviate**: Full-featured vector database with hybrid search capabilities, multi-tenancy,
  and production-grade scalability.
- **Pinecone**: Managed cloud service with automatic scaling and high availability.
- **FAISS**: Facebook's library for efficient similarity search, often used as the underlying
  engine in other systems.

## Similarity Metrics

The choice of similarity metric matters:

- **Cosine Similarity**: Measures the angle between vectors, ignoring magnitude. Ranges from -1
  to 1. Most commonly used because it's robust to differences in text length.
- **Dot Product**: Faster to compute but sensitive to vector magnitude. Works best when vectors
  are normalized.
- **Euclidean Distance**: Measures straight-line distance. Less commonly used for text but
  standard in other domains.
