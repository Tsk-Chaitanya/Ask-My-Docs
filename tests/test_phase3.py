"""
Phase 3 Tests — Conversation memory and server enhancements.

Run with:  pytest tests/test_phase3.py -v
"""

import pytest

from src.api.server import SessionStore, _build_contextual_query


# ── Session Store Tests ──────────────────────────────────────


class TestSessionStore:

    def test_empty_history(self):
        """New session returns empty history."""
        store = SessionStore()
        assert store.get_history("new-session") == []

    def test_add_and_retrieve(self):
        """Adding a turn stores question and answer."""
        store = SessionStore()
        store.add_turn("s1", "What is BM25?", "BM25 is a ranking function.")
        history = store.get_history("s1")
        assert len(history) == 1
        assert history[0]["question"] == "What is BM25?"
        assert history[0]["answer"] == "BM25 is a ranking function."

    def test_multiple_turns(self):
        """Multiple turns accumulate in order."""
        store = SessionStore()
        store.add_turn("s1", "Q1", "A1")
        store.add_turn("s1", "Q2", "A2")
        store.add_turn("s1", "Q3", "A3")
        history = store.get_history("s1")
        assert len(history) == 3
        assert history[0]["question"] == "Q1"
        assert history[2]["question"] == "Q3"

    def test_separate_sessions(self):
        """Different sessions are independent."""
        store = SessionStore()
        store.add_turn("s1", "Q1", "A1")
        store.add_turn("s2", "Q2", "A2")
        assert len(store.get_history("s1")) == 1
        assert len(store.get_history("s2")) == 1
        assert store.get_history("s1")[0]["question"] == "Q1"
        assert store.get_history("s2")[0]["question"] == "Q2"

    def test_clear_session(self):
        """Clearing a session removes its history."""
        store = SessionStore()
        store.add_turn("s1", "Q1", "A1")
        store.clear("s1")
        assert store.get_history("s1") == []

    def test_clear_nonexistent_session(self):
        """Clearing a non-existent session doesn't raise."""
        store = SessionStore()
        store.clear("nonexistent")  # should not raise

    def test_lru_eviction(self):
        """Oldest session is evicted when max is reached."""
        store = SessionStore(max_sessions=3)
        store.add_turn("s1", "Q1", "A1")
        store.add_turn("s2", "Q2", "A2")
        store.add_turn("s3", "Q3", "A3")
        store.add_turn("s4", "Q4", "A4")  # should evict s1
        assert store.get_history("s1") == []
        assert len(store.get_history("s4")) == 1

    def test_lru_access_refreshes(self):
        """Accessing a session moves it to end (prevents eviction)."""
        store = SessionStore(max_sessions=3)
        store.add_turn("s1", "Q1", "A1")
        store.add_turn("s2", "Q2", "A2")
        store.add_turn("s3", "Q3", "A3")
        # Access s1 to refresh it
        store.get_history("s1")
        # Adding s4 should evict s2 (oldest untouched), not s1
        store.add_turn("s4", "Q4", "A4")
        assert len(store.get_history("s1")) == 1  # still alive
        assert store.get_history("s2") == []  # evicted

    def test_history_trimming(self):
        """History is trimmed when it exceeds max per session."""
        store = SessionStore()
        # Add more turns than MAX_HISTORY_PER_SESSION (20)
        for i in range(25):
            store.add_turn("s1", f"Q{i}", f"A{i}")
        history = store.get_history("s1")
        assert len(history) <= 20
        # Should keep the most recent turns
        assert history[-1]["question"] == "Q24"


# ── Contextual Query Tests ───────────────────────────────────


class TestContextualQuery:

    def test_no_history(self):
        """Without history, query passes through unchanged."""
        result = _build_contextual_query("What is BM25?", [])
        assert result == "What is BM25?"

    def test_with_history(self):
        """With history, context is prepended."""
        history = [
            {"question": "What is BM25?", "answer": "A ranking function."},
        ]
        result = _build_contextual_query("Tell me more", history)
        assert "What is BM25?" in result
        assert "Tell me more" in result
        assert result.startswith("[Conversation context:")

    def test_only_recent_turns(self):
        """Only last 3 turns are included in context."""
        history = [
            {"question": f"Q{i}", "answer": f"A{i}"}
            for i in range(10)
        ]
        result = _build_contextual_query("Follow up", history)
        # Should include Q7, Q8, Q9 (last 3) but not Q0-Q6
        assert "Q9" in result
        assert "Q8" in result
        assert "Q7" in result
        assert "Q0" not in result
