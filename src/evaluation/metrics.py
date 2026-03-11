"""
Evaluation metrics for the RAG pipeline.

Three core metrics:

  ANSWER COVERAGE
    Does the generated answer contain the key phrases we expect?
    Measures whether the LLM actually addressed the question.

  CITATION ACCURACY
    Does the answer include [chunk_id] citations?
    Measures whether citation enforcement is working.

  DECLINE ACCURACY
    For out-of-scope questions, does the system correctly refuse?
    Measures hallucination prevention.

These are lightweight heuristic metrics that run without a second LLM
call. For production, complement with RAGAS-style LLM-judge metrics.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from src.evaluation.golden_dataset import GoldenSample


@dataclass
class SampleResult:
    """Evaluation result for a single golden sample."""

    sample_id: str
    question: str
    category: str
    difficulty: str
    generated_answer: str
    answer_coverage: float        # 0.0–1.0: fraction of expected phrases found
    citation_present: bool        # True if answer contains at least one [citation]
    decline_correct: bool         # True if decline behavior matched expectation
    passed: bool                  # Overall pass/fail for this sample


@dataclass
class EvalReport:
    """Aggregated evaluation report across all samples."""

    total: int = 0
    passed: int = 0
    failed: int = 0
    avg_answer_coverage: float = 0.0
    citation_rate: float = 0.0
    decline_accuracy: float = 0.0
    results: list[SampleResult] = field(default_factory=list)

    @property
    def pass_rate(self) -> float:
        return self.passed / self.total if self.total > 0 else 0.0

    def by_category(self) -> dict[str, dict]:
        """Break down pass rate by question category."""
        cats: dict[str, list[bool]] = {}
        for r in self.results:
            cats.setdefault(r.category, []).append(r.passed)
        return {
            cat: {
                "total": len(vals),
                "passed": sum(vals),
                "pass_rate": round(sum(vals) / len(vals), 3),
            }
            for cat, vals in cats.items()
        }


def score_sample(sample: GoldenSample, generated_answer: str, declined: bool) -> SampleResult:
    """
    Score a single generated answer against a golden sample.

    Args:
        sample: The golden sample with expected outputs.
        generated_answer: The answer produced by the RAG pipeline.
        declined: Whether the pipeline explicitly declined to answer.
    """
    answer_lower = generated_answer.lower()

    # ── Answer Coverage ───────────────────────────────────────
    # For declined samples, if we expect a decline, coverage is perfect.
    # Otherwise measure how many expected phrases appear in the answer.
    if sample.should_decline:
        coverage = 1.0 if declined else 0.0
    else:
        if declined:
            coverage = 0.0
        else:
            matches = sum(
                1 for phrase in sample.expected_answer_contains
                if phrase.lower() in answer_lower
            )
            coverage = matches / len(sample.expected_answer_contains) if sample.expected_answer_contains else 1.0

    # ── Citation Presence ─────────────────────────────────────
    import re
    has_citation = bool(re.search(r"\[[a-f0-9]{8,}\]", generated_answer))
    # For declined answers, citations aren't required
    citation_ok = declined or has_citation

    # ── Decline Accuracy ──────────────────────────────────────
    decline_correct = (sample.should_decline == declined)

    # ── Overall Pass ─────────────────────────────────────────
    # Pass if: coverage >= 0.5, decline behavior correct, citation present (if not declined)
    passed = coverage >= 0.5 and decline_correct and citation_ok

    return SampleResult(
        sample_id=sample.id,
        question=sample.question,
        category=sample.category,
        difficulty=sample.difficulty,
        generated_answer=generated_answer,
        answer_coverage=round(coverage, 3),
        citation_present=has_citation,
        decline_correct=decline_correct,
        passed=passed,
    )


def aggregate_results(results: list[SampleResult]) -> EvalReport:
    """Aggregate individual sample results into a report."""
    if not results:
        return EvalReport()

    report = EvalReport(
        total=len(results),
        passed=sum(1 for r in results if r.passed),
        failed=sum(1 for r in results if not r.passed),
        avg_answer_coverage=round(
            sum(r.answer_coverage for r in results) / len(results), 3
        ),
        citation_rate=round(
            sum(1 for r in results if r.citation_present) / len(results), 3
        ),
        decline_accuracy=round(
            sum(1 for r in results if r.decline_correct) / len(results), 3
        ),
        results=results,
    )
    return report
