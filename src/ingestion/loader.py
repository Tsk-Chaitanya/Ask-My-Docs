"""
Document loaders for PDF, Markdown, and plain text files.

WHY THIS DESIGN:
  Every loader returns a list[Document] — a uniform format regardless
  of source type. This means the chunker doesn't need to care where
  the text came from. The metadata travels with the content so we
  can cite sources later (e.g. "page 3 of paper.pdf").
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Document:
    """A single document unit before chunking."""

    content: str
    metadata: dict = field(default_factory=dict)

    def __repr__(self) -> str:
        src = self.metadata.get("source", "?")
        return f"Document(source={src}, chars={len(self.content)})"


# ── Individual Loaders ───────────────────────────────────────


def load_text(path: str | Path) -> list[Document]:
    """Load a plain .txt file as a single Document."""
    path = Path(path)
    text = path.read_text(encoding="utf-8", errors="replace")
    return [Document(content=text, metadata={"source": str(path), "type": "text"})]


def load_markdown(path: str | Path) -> list[Document]:
    """
    Load a .md file, stripping optional YAML front-matter.

    Many technical papers and blog posts use front-matter (---...---).
    We strip it because it's metadata, not content you'd want to
    search through or cite.
    """
    path = Path(path)
    text = path.read_text(encoding="utf-8", errors="replace")

    if text.startswith("---"):
        parts = text.split("---", 2)
        if len(parts) >= 3:
            text = parts[2].strip()

    return [Document(content=text, metadata={"source": str(path), "type": "markdown"})]


def load_pdf(path: str | Path) -> list[Document]:
    """
    Load a PDF, returning one Document per page.

    WHY PER-PAGE: PDFs often have distinct sections per page.
    Splitting here gives the chunker better boundaries to work with
    and lets us track page numbers for precise citations.
    """
    from pypdf import PdfReader

    path = Path(path)
    reader = PdfReader(str(path))
    docs: list[Document] = []

    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        if text.strip():
            docs.append(
                Document(
                    content=text,
                    metadata={"source": str(path), "type": "pdf", "page": i + 1},
                )
            )

    return docs


# ── Directory Scanner ────────────────────────────────────────

LOADERS = {
    ".txt": load_text,
    ".md": load_markdown,
    ".pdf": load_pdf,
}


def load_directory(directory: str | Path) -> list[Document]:
    """
    Recursively load all supported files from a directory.
    Unsupported file types are silently skipped.
    """
    directory = Path(directory)
    documents: list[Document] = []

    for root, _, files in os.walk(directory):
        for fname in sorted(files):
            ext = Path(fname).suffix.lower()
            if ext in LOADERS:
                full_path = Path(root) / fname
                try:
                    docs = LOADERS[ext](full_path)
                    documents.extend(docs)
                    print(f"    ✓ {full_path.name} → {len(docs)} document(s)")
                except Exception as exc:
                    print(f"    ✗ {full_path.name} — skipped: {exc}")

    return documents
