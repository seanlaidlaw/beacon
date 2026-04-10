"""Integration and smoke tests for beacon.

These tests exercise the full pipeline — files on disk → indexer → DB → search —
using temporary directories created by pytest's tmp_path fixture.

Dense embeddings are skipped gracefully when the model isn't cached locally
(the indexer degrades to TF-IDF only), so no ML model download is required.

Run with:
    pytest tests/test_integration.py -v
"""

import subprocess
import sys
from pathlib import Path

import pytest

from beacon.schema import open_db
from beacon.indexer.indexer import check_and_reindex, index
from beacon.search.query import search
from beacon.search.capsule import get_capsule


# ── Shared fixtures ───────────────────────────────────────────────────────────

MATH_PY = """\
import os


def add(a, b):
    \"\"\"Add two numbers and return the result.\"\"\"
    return a + b


def subtract(a, b):
    \"\"\"Subtract b from a.\"\"\"
    return a - b


class Calculator:
    \"\"\"A simple arithmetic calculator.\"\"\"

    def multiply(self, x, y):
        \"\"\"Multiply x by y using repeated addition.\"\"\"
        total = add(x, 0)
        return x * y

    def divide(self, x, y):
        \"\"\"Divide x by y, raising ValueError on zero divisor.\"\"\"
        if y == 0:
            raise ValueError("Cannot divide by zero")
        return x / y
"""

UTILS_PY = """\
from math_ops import add, subtract


def double(n):
    \"\"\"Return n doubled.\"\"\"
    return add(n, n)


def negate(n):
    \"\"\"Return the negation of n.\"\"\"
    return subtract(0, n)
"""

TESTS_PY = """\
from math_ops import add


def test_add_positive():
    assert add(1, 2) == 3


def test_add_zero():
    assert add(0, 0) == 0
"""


@pytest.fixture()
def project(tmp_path):
    """Small Python project on disk, ready to index."""
    (tmp_path / "math_ops.py").write_text(MATH_PY)
    (tmp_path / "utils.py").write_text(UTILS_PY)
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_math.py").write_text(TESTS_PY)
    return tmp_path


@pytest.fixture()
def indexed_db(project, tmp_path):
    """Indexed in-memory DB (or real file DB) seeded from the project fixture."""
    db_path = tmp_path / ".beacon" / "index.db"
    conn = index(project, db_path=db_path, skip_coupling=True)
    return conn, project


# ── Indexing pipeline ─────────────────────────────────────────────────────────

class TestIndexingPipeline:
    def test_indexes_expected_symbols(self, indexed_db):
        conn, _ = indexed_db
        names = {r[0] for r in conn.execute("SELECT name FROM nodes").fetchall()}
        assert "add" in names
        assert "subtract" in names
        assert "Calculator" in names
        assert "multiply" in names
        assert "double" in names

    def test_function_kinds_correct(self, indexed_db):
        conn, _ = indexed_db
        row = conn.execute("SELECT kind FROM nodes WHERE name='add'").fetchone()
        assert row is not None
        assert row[0] == "function"

    def test_class_kind_correct(self, indexed_db):
        conn, _ = indexed_db
        row = conn.execute("SELECT kind FROM nodes WHERE name='Calculator'").fetchone()
        assert row is not None
        assert row[0] == "class"

    def test_method_kind_correct(self, indexed_db):
        conn, _ = indexed_db
        row = conn.execute("SELECT kind FROM nodes WHERE name='multiply'").fetchone()
        assert row is not None
        assert row[0] == "method"

    def test_docstrings_stored(self, indexed_db):
        conn, _ = indexed_db
        row = conn.execute("SELECT docstring FROM nodes WHERE name='add'").fetchone()
        assert row is not None
        assert "two numbers" in row[0]

    def test_file_cache_populated(self, indexed_db):
        conn, _ = indexed_db
        count = conn.execute("SELECT COUNT(*) FROM file_cache").fetchone()[0]
        assert count >= 3  # math_ops.py, utils.py, tests/test_math.py

    def test_import_refs_stored(self, indexed_db):
        conn, _ = indexed_db
        rows = conn.execute("SELECT target_module FROM import_refs").fetchall()
        modules = {r[0] for r in rows}
        # utils.py imports math_ops
        assert any("math_ops" in m for m in modules)

    def test_test_nodes_flagged(self, indexed_db):
        conn, _ = indexed_db
        row = conn.execute(
            "SELECT is_test FROM nodes WHERE name='test_add_positive'"
        ).fetchone()
        assert row is not None
        assert row[0] == 1

    def test_non_test_nodes_not_flagged(self, indexed_db):
        conn, _ = indexed_db
        row = conn.execute(
            "SELECT is_test FROM nodes WHERE name='add'"
        ).fetchone()
        assert row is not None
        assert row[0] == 0

    def test_exported_public_symbols(self, indexed_db):
        conn, _ = indexed_db
        row = conn.execute(
            "SELECT is_exported FROM nodes WHERE name='add'"
        ).fetchone()
        assert row is not None
        assert row[0] == 1

    def test_fts5_table_populated(self, indexed_db):
        conn, _ = indexed_db
        count = conn.execute("SELECT COUNT(*) FROM nodes_fts").fetchone()[0]
        assert count > 0

    def test_fts5_search_finds_symbol(self, indexed_db):
        conn, _ = indexed_db
        rows = conn.execute(
            "SELECT rowid FROM nodes_fts WHERE nodes_fts MATCH 'Calculator'"
        ).fetchall()
        assert len(rows) > 0

    def test_contains_edges_created(self, indexed_db):
        conn, _ = indexed_db
        # Calculator class should have CONTAINS edges to its methods
        edges = conn.execute(
            """SELECT e.type FROM edges e
               JOIN nodes src ON src.id = e.source_id
               JOIN nodes tgt ON tgt.id = e.target_id
               WHERE src.name = 'Calculator' AND e.type = 'CONTAINS'"""
        ).fetchall()
        assert len(edges) >= 2  # multiply and divide


class TestIncrementalReindex:
    def test_unchanged_files_return_zero(self, indexed_db, tmp_path):
        conn, project = indexed_db
        result = check_and_reindex(conn, project, silent=True)
        assert result == 0

    def test_new_file_gets_indexed(self, indexed_db, tmp_path):
        conn, project = indexed_db
        (project / "new_module.py").write_text("def greet(name):\n    return f'Hello {name}'\n")
        result = check_and_reindex(conn, project, silent=True)
        assert result >= 1
        row = conn.execute("SELECT name FROM nodes WHERE name='greet'").fetchone()
        assert row is not None

    def test_modified_file_reindexed(self, indexed_db):
        conn, project = indexed_db
        # Rewrite utils.py with a new function
        (project / "utils.py").write_text(
            "def triple(n):\n    \"\"\"Return n tripled.\"\"\"\n    return n * 3\n"
        )
        check_and_reindex(conn, project, silent=True)
        row = conn.execute("SELECT name FROM nodes WHERE name='triple'").fetchone()
        assert row is not None

    def test_deleted_file_removes_nodes(self, indexed_db):
        conn, project = indexed_db
        before = conn.execute(
            "SELECT COUNT(*) FROM nodes WHERE file_path LIKE '%utils%'"
        ).fetchone()[0]
        assert before > 0
        (project / "utils.py").unlink()
        check_and_reindex(conn, project, silent=True)
        after = conn.execute(
            "SELECT COUNT(*) FROM nodes WHERE file_path LIKE '%utils%'"
        ).fetchone()[0]
        assert after == 0

    def test_deleted_file_clears_cache(self, indexed_db):
        conn, project = indexed_db
        (project / "utils.py").unlink()
        check_and_reindex(conn, project, silent=True)
        row = conn.execute(
            "SELECT * FROM file_cache WHERE file_path LIKE '%utils%'"
        ).fetchone()
        assert row is None


# ── Search pipeline ───────────────────────────────────────────────────────────

class TestSearchPipeline:
    def test_search_returns_results(self, indexed_db):
        conn, _ = indexed_db
        results = search(conn, "add numbers")
        assert len(results) > 0

    def test_search_exact_name_ranks_high(self, indexed_db):
        conn, _ = indexed_db
        results = search(conn, "add", limit=10)
        names = [r.name for r in results]
        # "add" should appear in the top results
        assert "add" in names[:5]

    def test_search_by_docstring_concept(self, indexed_db):
        conn, _ = indexed_db
        results = search(conn, "arithmetic calculator")
        names = [r.name for r in results]
        assert "Calculator" in names

    def test_search_kind_filter_class(self, indexed_db):
        conn, _ = indexed_db
        results = search(conn, "calculator", kind_filter="class")
        assert all(r.kind == "class" for r in results)
        assert any(r.name == "Calculator" for r in results)

    def test_search_kind_filter_function(self, indexed_db):
        conn, _ = indexed_db
        results = search(conn, "add subtract", kind_filter="function")
        assert all(r.kind == "function" for r in results)

    def test_search_result_fields_populated(self, indexed_db):
        conn, _ = indexed_db
        results = search(conn, "add")
        assert len(results) > 0
        r = results[0]
        assert r.name
        assert r.fqn
        assert r.file_path
        assert r.kind
        assert r.score > 0

    def test_search_results_sorted_by_score(self, indexed_db):
        conn, _ = indexed_db
        results = search(conn, "divide zero")
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_search_no_match_returns_empty(self, indexed_db):
        conn, _ = indexed_db
        results = search(conn, "xyzzy_nonexistent_symbol_zzz")
        assert results == []

    def test_search_limit_respected(self, indexed_db):
        conn, _ = indexed_db
        results = search(conn, "function", limit=2)
        assert len(results) <= 2

    def test_search_division_finds_divide(self, indexed_db):
        conn, _ = indexed_db
        results = search(conn, "divide by zero error")
        names = [r.name for r in results]
        assert "divide" in names


# ── Capsule pipeline ──────────────────────────────────────────────────────────

class TestCapsulePipeline:
    def test_capsule_returns_nodes_after_indexing(self, indexed_db):
        conn, _ = indexed_db
        cap = get_capsule(conn, "arithmetic operations add subtract", max_tokens=8000)
        assert len(cap.nodes) > 0

    def test_capsule_budget_respected(self, indexed_db):
        conn, _ = indexed_db
        cap = get_capsule(conn, "calculator", max_tokens=100)
        assert cap.token_estimate <= 100

    def test_capsule_empty_db_returns_empty(self, tmp_path):
        conn = open_db(tmp_path / "empty.db")
        cap = get_capsule(conn, "anything", max_tokens=8000)
        assert cap.nodes == []
        assert cap.token_estimate == 0

    def test_capsule_nodes_have_scores(self, indexed_db):
        conn, _ = indexed_db
        cap = get_capsule(conn, "add numbers", max_tokens=8000)
        assert all(n.score > 0 for n in cap.nodes)

    def test_capsule_test_nodes_score_lower_than_prod(self, indexed_db):
        conn, _ = indexed_db
        cap = get_capsule(conn, "add numbers arithmetic", max_tokens=8000)
        # Look up is_test from the DB for each capsule node
        test_fqns = {
            r[0] for r in conn.execute("SELECT fqn FROM nodes WHERE is_test=1").fetchall()
        }
        prod_nodes = [n for n in cap.nodes if n.fqn not in test_fqns]
        test_nodes = [n for n in cap.nodes if n.fqn in test_fqns]
        if prod_nodes and test_nodes:
            avg_prod = sum(n.score for n in prod_nodes) / len(prod_nodes)
            avg_test = sum(n.score for n in test_nodes) / len(test_nodes)
            assert avg_prod > avg_test


# ── Ignored directories ───────────────────────────────────────────────────────

class TestIgnoredDirectories:
    def test_pycache_not_indexed(self, project, tmp_path):
        pycache = project / "__pycache__"
        pycache.mkdir()
        (pycache / "math_ops.cpython-312.pyc").write_bytes(b"fake pyc")
        # Even a .py file inside __pycache__ should be excluded
        (pycache / "hidden.py").write_text("def secret(): pass\n")
        db_path = tmp_path / ".beacon" / "index.db"
        conn = index(project, db_path=db_path, skip_coupling=True)
        row = conn.execute("SELECT name FROM nodes WHERE name='secret'").fetchone()
        assert row is None

    def test_node_modules_not_indexed(self, project, tmp_path):
        nm = project / "node_modules"
        nm.mkdir()
        (nm / "lodash.py").write_text("def cloneDeep(): pass\n")
        db_path = tmp_path / ".beacon" / "index.db"
        conn = index(project, db_path=db_path, skip_coupling=True)
        row = conn.execute("SELECT name FROM nodes WHERE name='cloneDeep'").fetchone()
        assert row is None

    def test_gitignored_file_not_indexed(self, project, tmp_path):
        (project / ".gitignore").write_text("secret_module.py\n")
        (project / "secret_module.py").write_text("def hidden(): pass\n")
        db_path = tmp_path / ".beacon" / "index.db"
        conn = index(project, db_path=db_path, skip_coupling=True)
        row = conn.execute("SELECT name FROM nodes WHERE name='hidden'").fetchone()
        assert row is None


# ── CLI smoke tests ───────────────────────────────────────────────────────────

class TestCLISmoke:
    """Invoke beacon CLI as a subprocess — verifies the installed entry point works."""

    def _run(self, *args, cwd=None):
        return subprocess.run(
            [sys.executable, "-m", "beacon.cli", *args],
            capture_output=True, text=True, cwd=cwd,
        )

    def test_help_exits_zero(self):
        result = self._run("--help")
        assert result.returncode == 0
        assert "beacon" in result.stdout.lower()

    def test_index_command_creates_db(self, project, tmp_path):
        db_path = tmp_path / "test_index.db"
        result = self._run("index", str(project), "--db", str(db_path))
        assert result.returncode == 0, result.stderr
        assert db_path.exists()

    def test_index_command_output_mentions_files(self, project, tmp_path):
        db_path = tmp_path / "test_index.db"
        result = self._run("index", str(project), "--db", str(db_path))
        assert result.returncode == 0, result.stderr
        # Should mention scanning/indexing files
        combined = result.stdout + result.stderr
        assert any(word in combined.lower() for word in ("scan", "index", "file", "node"))

    def test_search_command_after_index(self, project, tmp_path):
        db_path = tmp_path / "search_test.db"
        self._run("index", str(project), "--db", str(db_path))
        result = self._run("search", "add numbers", "--db", str(db_path))
        assert result.returncode == 0, result.stderr

    def test_search_outputs_results(self, project, tmp_path):
        db_path = tmp_path / "search_test.db"
        self._run("index", str(project), "--db", str(db_path))
        result = self._run("search", "add", "--db", str(db_path))
        assert result.returncode == 0, result.stderr
        assert "add" in result.stdout.lower()

    def test_show_config_exits_zero(self):
        result = self._run("show-config")
        assert result.returncode == 0

    def test_search_empty_db_does_not_crash(self, tmp_path):
        db_path = tmp_path / "empty.db"
        open_db(db_path).close()
        result = self._run("search", "anything", "--db", str(db_path))
        assert result.returncode == 0, result.stderr
