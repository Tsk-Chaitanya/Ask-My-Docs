# Ask My Docs — Test Questionnaire

Use these questions to evaluate your RAG system after adding documents.
Run through each category and check the quality of answers and citations.

---

## A. Factual Recall (direct answers from a single document)

1. What is BM25?
2. What are the two tunable parameters in BM25?
3. What is a cross-encoder?
4. Name three vector databases mentioned in the documents.
5. What is the typical chunk size recommended for production RAG systems?

**What to check:** Answer should directly quote or closely paraphrase a specific chunk. Source card should show the correct file with a high score.

---

## B. Comparison & Synthesis (pulls from multiple documents)

6. How does BM25 differ from dense retrieval using embeddings?
7. What are the tradeoffs between bi-encoders and cross-encoders?
8. Why is hybrid retrieval better than using BM25 or embeddings alone?
9. Compare cosine similarity and dot product for vector search.

**What to check:** Answer should reference multiple source files. Look for source cards from 2+ documents.

---

## C. Follow-Up Conversations (tests conversation memory)

10. Ask: "What is RAG?"
    Then follow up: "What are the three stages you mentioned?"
    Then: "Tell me more about the second stage."

11. Ask: "Explain BM25 scoring."
    Then: "What are its limitations?"
    Then: "How do embeddings solve those limitations?"

**What to check:** Follow-up answers should be coherent and contextual, not generic. The system should retrieve relevant chunks even for vague questions like "tell me more."

---

## D. Edge Cases & Safety (tests citation enforcement)

12. What is the capital of France?
    *(Not in documents — should decline to answer)*

13. Who invented the transformer architecture?
    *(Might be partially in docs — check if it hedges properly)*

14. Explain quantum computing.
    *(Not in documents — should decline)*

15. What are embeddings used for in cooking?
    *(Trick question — should decline or clarify scope)*

**What to check:** System should say "I don't have enough information" rather than hallucinating. The `declined: true` flag should appear.

---

## E. Keyword-Heavy Queries (tests BM25 contribution)

16. HNSW graph
17. RAGAS framework evaluation
18. k1 parameter b parameter BM25
19. Sentence-BERT contrastive learning

**What to check:** These exact-match queries should return highly relevant results. BM25 should shine here since these are specific technical terms.

---

## F. Semantic Queries (tests embedding contribution)

20. How can I make my search system find similar meanings, not just matching words?
21. What happens when a language model makes up facts?
22. How do I split long documents for an AI system?
23. What's a good way to check if an AI's answer is actually correct?

**What to check:** These paraphrased queries don't use the exact terminology from the docs. The vector store should still find relevant chunks through semantic similarity.

---

## Scoring Guide

For each answer, rate on a 1-5 scale:

| Score | Meaning |
|-------|---------|
| 5 | Perfect — accurate, well-cited, uses correct sources |
| 4 | Good — mostly accurate, minor citation gaps |
| 3 | Acceptable — relevant but incomplete or weak citations |
| 2 | Poor — partially wrong or missing key information |
| 1 | Fail — hallucinated, wrong sources, or should have declined |

**Target:** Average score of 4+ across all applicable questions.
