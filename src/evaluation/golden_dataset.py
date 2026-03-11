"""
Golden dataset loader and validator for RAG evaluation.

A golden dataset is a curated set of (question, expected_answer_contains,
source_document) triples that represent known-good test cases. It lets you
measure system quality objectively and catch regressions when you change
retrieval parameters, prompts, or models.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class GoldenSample:
    """A single evaluation sample with expected outputs."""

    id: str
    question: str
    expected_answer_contains: list[str]
    source_document: str | None
    category: str
    difficulty: str
    should_decline: bool = False


def load_golden_dataset(path: str | Path) -> list[GoldenSample]:
    """
    Load and validate the golden dataset from a JSON file.

    Raises:
        FileNotFoundError: If the dataset file doesn't exist.
        ValueError: If any sample is missing required fields.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Golden dataset not found: {path}")

    with open(path, encoding="utf-8") as f:
        raw = json.load(f)

    required_fields = {"id", "question", "expected_answer_contains", "category", "difficulty"}
    samples: list[GoldenSample] = []

    for i, item in enumerate(raw):
        missing = required_fields - set(item.keys())
        if missing:
            raise ValueError(f"Sample {i} missing required fields: {missing}")

        samples.append(
            GoldenSample(
                id=item["id"],
                question=item["question"],
                expected_answer_contains=item["expected_answer_contains"],
                source_document=item.get("source_document"),
                category=item["category"],
                difficulty=item["difficulty"],
                should_decline=item.get("should_decline", False),
            )
        )

    return samples
