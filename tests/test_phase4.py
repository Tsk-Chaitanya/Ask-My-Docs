"""
Phase 4 Tests — Evaluation pipeline.

Run with:  pytest tests/test_phase4.py -v
"""

import json
import pytest
from pathlib import Path

from src.evaluation.golden_dataset import load_golden_dataset, GoldenSample
from src.evaluation.metrics import score_sample, aggregate_results, SampleResult


# ── Golden Dataset Tests ──────────────────────────────────────


class TestGoldenDataset:

    def test_load_valid_dataset(self, tmp_path):
        """Loads a valid dataset file successfully."""
        data = [
            {
                "id": "t1",
                "question": "What is BM25?",
                "expected_answer_contains": ["ranking", "retrieval"],
                "source_document": "bm25.txt",
                "category": "factual_recall",
                "difficulty": "easy",
            }
        ]
        path = tmp_path / "dataset.json"
        path.write_text(json.dumps(data))

        samples = load_golden_dataset(path)
        assert len(samples) == 1
        assert samples[0].id == "t1"
        assert samples[0].question == "What is BM25?"
        assert samples[0].should_decline is False

    def test_load_sample_with_decline(self, tmp_path):
        """Loads a sample with should_decline=True correctly."""
        data = [
            {
                "id": "t2",
                "question": "What is quantum computing?",
                "expected_answer_contains": ["don't have enough information"],
                "source_document": None,
                "category": "out_of_scope",
                "difficulty": "easy",
                "should_decline": True,
            }
        ]
        path = tmp_path / "dataset.json"
        path.write_text(json.dumps(data))

        samples = load_golden_dataset(path)
        assert samples[0].should_decline is True

    def test_load_missing_file_raises(self, tmp_path):
        """Raises FileNotFoundError for missing file."""
        with pytest.raises(FileNotFoundError):
            load_golden_dataset(tmp_path / "nonexistent.json")

    def test_load_missing_field_raises(self, tmp_path):
        """Raises ValueError when a required field is missing."""
        data = [{"id": "t1", "question": "What is BM25?"}]  # missing fields
        path = tmp_path / "dataset.json"
        path.write_text(json.dumps(data))

        with pytest.raises(ValueError):
            load_golden_dataset(path)

    def test_loads_real_dataset(self):
        """Loads the actual golden dataset from the project."""
        path = Path("data/eval/golden_dataset.json")
        if not path.exists():
            pytest.skip("Golden dataset not found")
        samples = load_golden_dataset(path)
        assert len(samples) > 0
        for s in samples:
            assert s.id
            assert s.question
            assert s.category


# ── Metrics Tests ─────────────────────────────────────────────


class TestMetrics:

    def _make_sample(self, should_decline=False) -> GoldenSample:
        return GoldenSample(
            id="t1",
            question="What is BM25?",
            expected_answer_contains=["ranking function", "information retrieval"],
            source_document="bm25.txt",
            category="factual_recall",
            difficulty="easy",
            should_decline=should_decline,
        )

    def test_full_coverage_passes(self):
        """Answer containing all expected phrases gets full coverage."""
        sample = self._make_sample()
        answer = "BM25 is a ranking function used in information retrieval. [abc12345]"
        result = score_sample(sample, answer, declined=False)
        assert result.answer_coverage == 1.0
        assert result.passed is True

    def test_partial_coverage(self):
        """Answer containing some but not all expected phrases gets partial coverage."""
        sample = self._make_sample()
        answer = "BM25 is a ranking function. [abc12345]"
        result = score_sample(sample, answer, declined=False)
        assert 0.0 < result.answer_coverage < 1.0

    def test_no_coverage_fails(self):
        """Answer with no expected phrases gets zero coverage and fails."""
        sample = self._make_sample()
        answer = "I'm not sure about that."
        result = score_sample(sample, answer, declined=False)
        assert result.answer_coverage == 0.0
        assert result.passed is False

    def test_correct_decline_passes(self):
        """System correctly declining an out-of-scope question passes."""
        sample = self._make_sample(should_decline=True)
        answer = "I don't have enough information in the provided documents."
        result = score_sample(sample, answer, declined=True)
        assert result.decline_correct is True
        assert result.passed is True

    def test_wrong_decline_fails(self):
        """System declining an in-scope question fails."""
        sample = self._make_sample(should_decline=False)
        answer = "I don't have enough information."
        result = score_sample(sample, answer, declined=True)
        assert result.decline_correct is False
        assert result.passed is False

    def test_citation_detection(self):
        """Citation regex correctly detects [chunk_id] patterns."""
        sample = self._make_sample()
        answer_with_cite = "BM25 is a ranking function [a1b2c3d4] used in information retrieval."
        answer_without_cite = "BM25 is a ranking function used in information retrieval."
        result_with = score_sample(sample, answer_with_cite, declined=False)
        result_without = score_sample(sample, answer_without_cite, declined=False)
        assert result_with.citation_present is True
        assert result_without.citation_present is False

    def test_aggregate_results(self):
        """Aggregation correctly computes pass rate and averages."""
        results = [
            SampleResult("t1", "Q1", "factual_recall", "easy", "A1", 1.0, True, True, True),
            SampleResult("t2", "Q2", "factual_recall", "easy", "A2", 0.5, True, True, True),
            SampleResult("t3", "Q3", "out_of_scope", "easy", "A3", 0.0, False, False, False),
        ]
        report = aggregate_results(results)
        assert report.total == 3
        assert report.passed == 2
        assert report.failed == 1
        assert round(report.pass_rate, 2) == 0.67

    def test_aggregate_by_category(self):
        """Category breakdown is computed correctly."""
        results = [
            SampleResult("t1", "Q1", "factual_recall", "easy", "A1", 1.0, True, True, True),
            SampleResult("t2", "Q2", "factual_recall", "easy", "A2", 0.0, False, False, False),
            SampleResult("t3", "Q3", "out_of_scope", "easy", "A3", 1.0, False, True, True),
        ]
        report = aggregate_results(results)
        cats = report.by_category()
        assert cats["factual_recall"]["total"] == 2
        assert cats["factual_recall"]["passed"] == 1
        assert cats["out_of_scope"]["passed"] == 1
