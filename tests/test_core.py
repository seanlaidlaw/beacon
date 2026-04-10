"""
Fast unit tests for Beacon core logic — no ML model loading, no disk I/O.
These run in CI with `pytest -m "not slow"`.
"""

import sqlite3
import tempfile
from pathlib import Path

from beacon.benchmark import (
    check_baseline_recall,
    check_recall,
    count_tokens_approx,
    summary_stats,
)


# ── count_tokens_approx ───────────────────────────────────────────────────────

class TestCountTokensApprox:
    def test_empty_string_returns_one(self):
        assert count_tokens_approx("") == 1

    def test_four_chars_is_one_token(self):
        assert count_tokens_approx("abcd") == 1

    def test_scales_linearly(self):
        assert count_tokens_approx("a" * 400) == 100

    def test_never_zero(self):
        assert count_tokens_approx("x") >= 1


# ── check_recall ──────────────────────────────────────────────────────────────

class TestCheckRecall:
    def test_finds_file_path_hint(self):
        output = "django/middleware/csrf.py  CsrfViewMiddleware"
        assert check_recall(output, "django/middleware/csrf.py")

    def test_case_insensitive(self):
        output = "CSRFVIEWMIDDLEWARE processes the token"
        assert check_recall(output, "django/middleware/csrf.py")

    def test_returns_false_when_no_match(self):
        output = "completely unrelated output about something else"
        assert not check_recall(output, "django/db/migrations/executor.py")

    def test_partial_match_enough(self):
        # hint has 3 meaningful parts; matching ≥2 should pass
        output = "executor migration"
        assert check_recall(output, "django/db/migrations/executor.py")

    def test_empty_output_returns_false(self):
        assert not check_recall("", "django/middleware/csrf.py")


# ── check_baseline_recall ─────────────────────────────────────────────────────

class TestCheckBaselineRecall:
    def test_matching_file_returns_true(self):
        files = [{"file": "django/middleware/csrf.py", "tokens": 500}]
        assert check_baseline_recall(files, "django/middleware/csrf.py")

    def test_empty_files_returns_false(self):
        assert not check_baseline_recall([], "django/middleware/csrf.py")

    def test_no_matching_file(self):
        files = [{"file": "django/db/models/base.py", "tokens": 800}]
        assert not check_baseline_recall(files, "django/middleware/csrf.py")


# ── summary_stats ─────────────────────────────────────────────────────────────

class TestSummaryStats:
    def _make_result(self, beacon_tokens, baseline_tokens, recall=True):
        ratio = baseline_tokens / max(1, beacon_tokens)
        pct = round((1 - beacon_tokens / max(1, baseline_tokens)) * 100, 1)
        return {
            "beacon_tokens": beacon_tokens,
            "baseline_tokens": baseline_tokens,
            "savings_ratio": ratio,
            "pct_saved": pct,
            "beacon_recall": recall,
        }

    def test_empty_results(self):
        assert summary_stats([]) == {}

    def test_overall_pct_saved(self):
        results = [
            self._make_result(1000, 10000),   # 90% saved
            self._make_result(2000, 4000),    # 50% saved
        ]
        stats = summary_stats(results)
        # Total: 3000 beacon / 14000 baseline → ~78.6%
        assert stats["overall_pct_saved"] == round((1 - 3000 / 14000) * 100, 1)

    def test_beacon_wins_counts_better_than_one(self):
        results = [
            self._make_result(100, 1000),   # wins
            self._make_result(1000, 100),   # loses
        ]
        assert summary_stats(results)["beacon_wins"] == 1

    def test_recall_count(self):
        results = [
            self._make_result(100, 1000, recall=True),
            self._make_result(100, 1000, recall=False),
            self._make_result(100, 1000, recall=True),
        ]
        assert summary_stats(results)["beacon_recall_count"] == 2


# ── schema: open_db creates a valid database ─────────────────────────────────

class TestSchema:
    def test_open_db_creates_tables(self):
        from beacon.schema import open_db
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "index.db"
            conn = open_db(db_path)
            tables = {
                row[0]
                for row in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                ).fetchall()
            }
        assert "nodes" in tables
        assert "edges" in tables
        assert "node_embeddings_dense" in tables
        assert "import_refs" in tables
