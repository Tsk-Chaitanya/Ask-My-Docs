"""
Evaluation runner for the Ask My Docs RAG pipeline.

Usage:
    # Interactive report
    python -m src.evaluation.run_eval --dataset data/eval/golden_dataset.json

    # CI mode — exits non-zero if quality drops below threshold
    python -m src.evaluation.run_eval --dataset data/eval/golden_dataset.json --ci --threshold 0.85
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from src.evaluation.golden_dataset import load_golden_dataset
from src.evaluation.metrics import score_sample, aggregate_results
from src.ingestion.loader import load_directory
from src.ingestion.chunker import TokenChunker
from src.retrieval.vector import VectorStore
from src.retrieval.bm25 import BM25Index
from src.retrieval.hybrid import HybridRetriever
from src.retrieval.reranker import Reranker
from src.generation.generator import AnswerGenerator


# ── Pipeline Bootstrap ───────────────────────────────────────


def build_pipeline(source_dir: str = "data/documents") -> dict:
    """Build the full RAG pipeline for evaluation."""
    print("  Loading documents ...")
    docs = load_directory(source_dir)

    chunker = TokenChunker(chunk_size=600, chunk_overlap=100)
    chunks = chunker.chunk_documents(docs)
    print(f"  Indexed {len(chunks)} chunks")

    store = VectorStore()
    store.add_chunks(chunks)

    bm25 = BM25Index()
    bm25.build(chunks)

    hybrid = HybridRetriever(vector_store=store, bm25_index=bm25)

    print("  Loading reranker ...")
    reranker = Reranker()

    generator = AnswerGenerator()

    return {
        "hybrid": hybrid,
        "reranker": reranker,
        "generator": generator,
    }


# ── Runner ───────────────────────────────────────────────────


def run_evaluation(
    dataset_path: str,
    source_dir: str = "data/documents",
    ci_mode: bool = False,
    threshold: float = 0.75,
) -> int:
    """
    Run evaluation against the golden dataset.

    Returns:
        0 if evaluation passes (or not in CI mode), 1 if CI gate fails.
    """
    print("\n" + "═" * 56)
    print("  Ask My Docs — Evaluation Pipeline")
    print("═" * 56 + "\n")

    # Load dataset
    print("  Loading golden dataset ...")
    try:
        samples = load_golden_dataset(dataset_path)
    except FileNotFoundError as e:
        print(f"  ✗ {e}")
        return 1
    print(f"  Loaded {len(samples)} samples\n")

    # Build pipeline
    pipeline = build_pipeline(source_dir)
    hybrid = pipeline["hybrid"]
    reranker = pipeline["reranker"]
    generator = pipeline["generator"]

    # Run each sample
    print(f"  Running {len(samples)} evaluations ...\n")
    sample_results = []

    for sample in samples:
        candidates = hybrid.retrieve(sample.question, top_k=10, fetch_k=20)
        top_chunks = reranker.rerank(sample.question, candidates, top_k=5)
        answer = generator.generate(sample.question, top_chunks)

        result = score_sample(sample, answer.answer, answer.declined)
        sample_results.append(result)

        status = "✓" if result.passed else "✗"
        print(
            f"  {status} [{sample.id}] {sample.question[:55]:<55} "
            f"coverage={result.answer_coverage:.2f}"
        )

    # Aggregate
    report = aggregate_results(sample_results)

    # Print report
    print("\n" + "─" * 56)
    print("  Results Summary")
    print("─" * 56)
    print(f"  Pass rate:        {report.pass_rate:.1%}  ({report.passed}/{report.total})")
    print(f"  Answer coverage:  {report.avg_answer_coverage:.1%}")
    print(f"  Citation rate:    {report.citation_rate:.1%}")
    print(f"  Decline accuracy: {report.decline_accuracy:.1%}")

    print("\n  By Category:")
    for cat, stats in report.by_category().items():
        print(f"    {cat:<20} {stats['passed']}/{stats['total']}  ({stats['pass_rate']:.1%})")

    # Failed samples detail
    failed = [r for r in sample_results if not r.passed]
    if failed:
        print(f"\n  Failed Samples ({len(failed)}):")
        for r in failed:
            print(f"    • [{r.sample_id}] {r.question[:60]}")
            print(f"      coverage={r.answer_coverage:.2f}, "
                  f"citation={r.citation_present}, "
                  f"decline_ok={r.decline_correct}")

    print("\n" + "═" * 56)

    # CI gate
    if ci_mode:
        if report.pass_rate >= threshold:
            print(f"  ✅ CI PASSED — pass rate {report.pass_rate:.1%} >= threshold {threshold:.1%}")
            print("═" * 56 + "\n")
            return 0
        else:
            print(f"  ❌ CI FAILED — pass rate {report.pass_rate:.1%} < threshold {threshold:.1%}")
            print("═" * 56 + "\n")
            return 1

    print("═" * 56 + "\n")
    return 0


# ── CLI Entry Point ──────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="Ask My Docs — Evaluation Runner")
    parser.add_argument(
        "--dataset",
        default="data/eval/golden_dataset.json",
        help="Path to golden dataset JSON file",
    )
    parser.add_argument(
        "--source",
        default="data/documents",
        help="Source documents directory",
    )
    parser.add_argument(
        "--ci",
        action="store_true",
        help="CI mode — exit non-zero if quality below threshold",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.75,
        help="Pass rate threshold for CI gate (default: 0.75)",
    )
    args = parser.parse_args()

    exit_code = run_evaluation(
        dataset_path=args.dataset,
        source_dir=args.source,
        ci_mode=args.ci,
        threshold=args.threshold,
    )
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
