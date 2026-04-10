"""Tests for beacon/schema.py — SQLite schema and migration logic."""
import sqlite3
import tempfile
from pathlib import Path

import pytest

from beacon.schema import open_db, _needs_rebuild, SCHEMA_VERSION


# ── _needs_rebuild ────────────────────────────────────────────────────────────

class TestNeedsRebuild:
    def test_fresh_db_with_no_meta_table_returns_true(self):
        conn = sqlite3.connect(":memory:")
        assert _needs_rebuild(conn) is True

    def test_correct_version_returns_false(self):
        conn = sqlite3.connect(":memory:")
        conn.execute("CREATE TABLE meta (key TEXT, value TEXT)")
        conn.execute("INSERT INTO meta VALUES ('schema_version', ?)", (SCHEMA_VERSION,))
        conn.commit()
        assert _needs_rebuild(conn) is False

    def test_old_version_with_missing_critical_columns_returns_true(self):
        conn = sqlite3.connect(":memory:")
        conn.execute("CREATE TABLE meta (key TEXT, value TEXT)")
        conn.execute("INSERT INTO meta VALUES ('schema_version', '0')")
        # nodes table without 'kind' or 'repo_alias'
        conn.execute("CREATE TABLE nodes (id INTEGER PRIMARY KEY, name TEXT, fqn TEXT)")
        conn.commit()
        assert _needs_rebuild(conn) is True

    def test_old_version_missing_only_repo_alias_returns_true(self):
        conn = sqlite3.connect(":memory:")
        conn.execute("CREATE TABLE meta (key TEXT, value TEXT)")
        conn.execute("INSERT INTO meta VALUES ('schema_version', '0')")
        conn.execute("CREATE TABLE nodes (id INTEGER PRIMARY KEY, kind TEXT)")  # has kind, no repo_alias
        conn.commit()
        assert _needs_rebuild(conn) is True

    def test_old_version_with_all_critical_columns_returns_false(self):
        # Old version but both 'kind' and 'repo_alias' exist — considered compatible
        conn = sqlite3.connect(":memory:")
        conn.execute("CREATE TABLE meta (key TEXT, value TEXT)")
        conn.execute("INSERT INTO meta VALUES ('schema_version', '0')")
        conn.execute("CREATE TABLE nodes (id INTEGER PRIMARY KEY, kind TEXT, repo_alias TEXT)")
        conn.commit()
        assert _needs_rebuild(conn) is False

    def test_empty_meta_table_with_critical_columns_returns_false(self):
        # meta table exists but no schema_version row; nodes has required columns
        conn = sqlite3.connect(":memory:")
        conn.execute("CREATE TABLE meta (key TEXT, value TEXT)")
        conn.execute("CREATE TABLE nodes (id INTEGER PRIMARY KEY, kind TEXT, repo_alias TEXT)")
        conn.commit()
        # row is None → falls through to column check → columns exist → False
        assert _needs_rebuild(conn) is False


# ── open_db ───────────────────────────────────────────────────────────────────

class TestOpenDb:
    def test_creates_all_required_tables(self):
        conn = open_db(":memory:")
        tables = {
            r[0] for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        for required in ("nodes", "edges", "node_embeddings_dense", "import_refs",
                         "co_change_edges", "file_cache", "meta"):
            assert required in tables, f"Table '{required}' not found"

    def test_creates_fts5_virtual_tables(self):
        conn = open_db(":memory:")
        tables = {
            r[0] for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        assert "nodes_fts" in tables
        assert "observations_fts" in tables

    def test_schema_version_stored_in_meta(self):
        conn = open_db(":memory:")
        row = conn.execute(
            "SELECT value FROM meta WHERE key='schema_version'"
        ).fetchone()
        assert row is not None
        assert row[0] == SCHEMA_VERSION

    def test_row_factory_is_sqlite_row(self):
        conn = open_db(":memory:")
        # Insert and retrieve using dict-style access
        conn.execute(
            "INSERT INTO nodes (name, fqn, file_path, kind, is_exported, is_test, repo_alias)"
            " VALUES ('f', 'a::f', 'a.py', 'function', 1, 0, 'primary')"
        )
        row = conn.execute("SELECT name, kind FROM nodes WHERE fqn='a::f'").fetchone()
        assert row["name"] == "f"
        assert row["kind"] == "function"

    def test_fts_trigger_syncs_on_insert(self):
        conn = open_db(":memory:")
        conn.execute(
            "INSERT INTO nodes (name, fqn, file_path, kind, is_exported, is_test, repo_alias)"
            " VALUES ('my_function', 'a::my_function', 'a.py', 'function', 1, 0, 'primary')"
        )
        conn.commit()
        # FTS5 should have been updated by trigger
        results = conn.execute(
            "SELECT rowid FROM nodes_fts WHERE nodes_fts MATCH 'my_function'"
        ).fetchall()
        assert len(results) > 0

    def test_idempotent_on_second_open(self, tmp_path):
        db_path = tmp_path / "test.db"
        conn1 = open_db(db_path)
        conn1.execute(
            "INSERT INTO nodes (name, fqn, file_path, kind, is_exported, is_test, repo_alias)"
            " VALUES ('preserved', 'a::preserved', 'a.py', 'function', 1, 0, 'primary')"
        )
        conn1.commit()
        conn1.close()

        # Re-opening should NOT wipe the data (no rebuild needed for same version)
        conn2 = open_db(db_path)
        row = conn2.execute("SELECT name FROM nodes WHERE fqn='a::preserved'").fetchone()
        assert row is not None
        assert row[0] == "preserved"

    def test_creates_parent_directories(self, tmp_path):
        deep_path = tmp_path / "a" / "b" / "c" / "index.db"
        conn = open_db(deep_path)
        assert deep_path.exists()
        conn.close()
