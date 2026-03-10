"""
Answer generator with citation enforcement.

This is the final stage of the RAG pipeline:
  Retrieved chunks → LLM → Cited answer (or "I don't know")

CITATION ENFORCEMENT means two things:
  1. The prompt instructs the LLM to cite [chunk_id] for every claim
  2. If chunks are insufficient, the system REFUSES to answer
     rather than hallucinating — this is the key production safety feature
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass

import anthropic
from dotenv import load_dotenv

from src.generation.prompt_manager import PromptManager
from src.retrieval.vector import RetrievalResult

load_dotenv()


@dataclass
class GeneratedAnswer:
    """The complete output of the RAG pipeline."""

    answer: str
    citations: list[str]       # chunk_ids found in the answer
    chunks_used: list[dict]    # metadata about chunks that were provided
    declined: bool = False     # True if system refused to answer
    prompt_version: str = ""


class AnswerGenerator:
    """Generates cited answers from retrieved chunks using Claude."""

    def __init__(
        self,
        prompt_manager: PromptManager | None = None,
        model: str = "claude-sonnet-4-20250514",
        max_tokens: int = 1024,
    ):
        api_key = os.environ.get("ANTHROPIC_API_KEY", "")
        self.client = anthropic.Anthropic(api_key=api_key) if api_key else None
        self.prompts = prompt_manager or PromptManager()
        self.model = model
        self.max_tokens = max_tokens

    def _format_context(self, chunks: list[RetrievalResult]) -> str:
        """Format chunks into a context block the LLM can reference."""
        parts: list[str] = []
        for chunk in chunks:
            source = chunk.metadata.get("source", "unknown")
            parts.append(f"[{chunk.chunk_id}] Source: {source}\n{chunk.content}")
        return "\n\n---\n\n".join(parts)

    def _extract_citations(self, text: str, valid_ids: set[str]) -> list[str]:
        """Pull [chunk_id] references from the answer, keeping only valid ones."""
        cited = re.findall(r"\[([^\]]+)\]", text)
        return [cid for cid in cited if cid in valid_ids]

    def generate(self, question: str, chunks: list[RetrievalResult]) -> GeneratedAnswer:
        """
        Generate a cited answer.

        If no chunks are provided → decline.
        If API key is missing → return a demo response showing what would happen.
        """
        # Citation enforcement: no chunks = no answer
        if not chunks:
            return GeneratedAnswer(
                answer="I don't have enough information in the provided documents to answer this question.",
                citations=[],
                chunks_used=[],
                declined=True,
                prompt_version=self.prompts.version,
            )

        context = self._format_context(chunks)
        system_prompt = self.prompts.get_system_prompt("generation")
        user_prompt = self.prompts.render_user_prompt(
            "generation", context=context, question=question,
        )

        # If no API key, return a demo showing the pipeline works
        if not self.client:
            chunk_ids = [c.chunk_id for c in chunks]
            return GeneratedAnswer(
                answer=(
                    f"[DEMO MODE — no ANTHROPIC_API_KEY set]\n\n"
                    f"The system would send {len(chunks)} chunk(s) to Claude "
                    f"and generate a cited answer.\n\n"
                    f"Chunks provided: {', '.join(c.chunk_id[:12] for c in chunks)}\n"
                    f"Prompt version: {self.prompts.version}"
                ),
                citations=chunk_ids,
                chunks_used=[
                    {"chunk_id": c.chunk_id, "source": c.metadata.get("source", ""), "score": c.score}
                    for c in chunks
                ],
                declined=False,
                prompt_version=self.prompts.version,
            )

        # ── Real API call ────────────────────────────────────
        response = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        )

        answer_text = response.content[0].text
        declined = "I don't have enough information" in answer_text

        valid_ids = {c.chunk_id for c in chunks}
        citations = self._extract_citations(answer_text, valid_ids)

        return GeneratedAnswer(
            answer=answer_text,
            citations=citations,
            chunks_used=[
                {"chunk_id": c.chunk_id, "source": c.metadata.get("source", ""), "score": c.score}
                for c in chunks
            ],
            declined=declined,
            prompt_version=self.prompts.version,
        )
