"""
SQLite schema for pyvexp — mirrors vexp-core schema version 5.
"""

import sqlite3
from pathlib import Path

SCHEMA_VERSION = "5"

DDL = """
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;

-- ── Core graph ────────────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS nodes (
    id          INTEGER PRIMARY KEY,
    name        TEXT NOT NULL,
    fqn         TEXT NOT NULL UNIQUE,
    file_path   TEXT NOT NULL,
    kind        TEXT NOT NULL,   -- function, method, class, interface, ...
    start_line  INTEGER,
    end_line    INTEGER,
    signature   TEXT,
    docstring   TEXT,
    is_exported INTEGER DEFAULT 0,
    is_test     INTEGER DEFAULT 0,
    repo_alias  TEXT NOT NULL DEFAULT 'primary'
);

CREATE INDEX IF NOT EXISTS idx_nodes_file ON nodes(file_path);
CREATE INDEX IF NOT EXISTS idx_nodes_fqn  ON nodes(fqn);

CREATE TABLE IF NOT EXISTS edges (
    id            INTEGER PRIMARY KEY,
    source_id     INTEGER NOT NULL REFERENCES nodes(id) ON DELETE CASCADE,
    target_id     INTEGER NOT NULL REFERENCES nodes(id) ON DELETE CASCADE,
    type          TEXT NOT NULL DEFAULT 'CALLS',  -- CALLS, IMPORTS, CONTAINS
    call_site_line INTEGER,
    confidence    REAL DEFAULT 1.0
);

CREATE INDEX IF NOT EXISTS idx_edges_source ON edges(source_id);
CREATE INDEX IF NOT EXISTS idx_edges_target ON edges(target_id);

-- High-confidence edges from LSP (type-resolved)
CREATE TABLE IF NOT EXISTS lsp_edges (
    id          INTEGER PRIMARY KEY,
    source_fqn  TEXT NOT NULL,
    target_fqn  TEXT NOT NULL,
    edge_type   TEXT NOT NULL DEFAULT 'CALLS'
);

-- Cross-repo edges (multi-repo workspaces)
CREATE TABLE IF NOT EXISTS cross_repo_edges (
    id           INTEGER PRIMARY KEY,
    workspace_id TEXT,
    source_repo  TEXT NOT NULL,
    source_fqn   TEXT NOT NULL,
    target_repo  TEXT NOT NULL,
    target_fqn   TEXT NOT NULL,
    type         TEXT NOT NULL DEFAULT 'CALLS',
    confidence   REAL DEFAULT 1.0
);

-- Change coupling: files that commit together
CREATE TABLE IF NOT EXISTS co_change_edges (
    id             INTEGER PRIMARY KEY,
    file_a         TEXT NOT NULL,
    file_b         TEXT NOT NULL,
    coupling_score REAL NOT NULL DEFAULT 0.0,
    shared_commits INTEGER NOT NULL DEFAULT 0,
    updated_at     TEXT NOT NULL,
    UNIQUE(file_a, file_b)
);

CREATE INDEX IF NOT EXISTS idx_cochange_a ON co_change_edges(file_a);
CREATE INDEX IF NOT EXISTS idx_cochange_b ON co_change_edges(file_b);

-- ── Embeddings ────────────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS node_embeddings (
    node_id     INTEGER PRIMARY KEY REFERENCES nodes(id) ON DELETE CASCADE,
    vector_json TEXT NOT NULL,   -- sparse TF-IDF: {"term": weight, ...}
    updated_at  TEXT NOT NULL
);

-- ── File tracking ─────────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS file_cache (
    file_path       TEXT PRIMARY KEY,
    blake3_hash     TEXT NOT NULL,
    last_indexed_at TEXT NOT NULL,
    node_count      INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS file_changes (
    id          INTEGER PRIMARY KEY,
    file_path   TEXT NOT NULL,
    change_type TEXT NOT NULL,  -- added, modified, deleted
    old_hash    TEXT,
    new_hash    TEXT,
    detected_at TEXT NOT NULL,
    session_id  TEXT,
    repo_alias  TEXT NOT NULL DEFAULT 'primary'
);

CREATE TABLE IF NOT EXISTS file_lineage (
    file_path     TEXT PRIMARY KEY,
    commit_count  INTEGER DEFAULT 0,
    churn_score   REAL NOT NULL DEFAULT 0.0,
    last_author   TEXT,
    last_commit_ts TEXT,
    updated_at    TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS ast_diffs (
    id             INTEGER PRIMARY KEY,
    file_change_id INTEGER REFERENCES file_changes(id),
    symbol_fqn     TEXT NOT NULL,
    diff_type      TEXT NOT NULL,   -- added, removed, modified
    summary        TEXT,
    old_snippet    TEXT,
    new_snippet    TEXT,
    detected_at    TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS stale_refs (
    node_id     INTEGER PRIMARY KEY REFERENCES nodes(id) ON DELETE CASCADE,
    stale_since TEXT NOT NULL,
    reason      TEXT
);

-- ── Memory / sessions ─────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS sessions (
    id         TEXT PRIMARY KEY,
    agent_id   TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    status     TEXT DEFAULT 'active',  -- active, compressed
    summary    TEXT,
    summary_embedding TEXT
);

CREATE TABLE IF NOT EXISTS observations (
    id             INTEGER PRIMARY KEY,
    session_id     TEXT REFERENCES sessions(id),
    type           TEXT NOT NULL,  -- insight, decision, tool_call, ...
    content        TEXT NOT NULL,
    embedding      TEXT,           -- TF-IDF JSON for semantic search
    file_paths     TEXT,           -- JSON array
    created_at     TEXT NOT NULL,
    source         TEXT DEFAULT 'agent',
    confidence     REAL DEFAULT 1.0,
    category       TEXT,
    file_change_id INTEGER,
    stale          INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS observation_node_links (
    observation_id INTEGER NOT NULL REFERENCES observations(id) ON DELETE CASCADE,
    node_id        INTEGER NOT NULL REFERENCES nodes(id) ON DELETE CASCADE,
    repo_alias     TEXT DEFAULT 'primary',
    PRIMARY KEY (observation_id, node_id)
);

CREATE TABLE IF NOT EXISTS project_rules (
    id                   INTEGER PRIMARY KEY,
    scope_pattern        TEXT,
    rule_text            TEXT NOT NULL,
    confidence           REAL DEFAULT 1.0,
    status               TEXT DEFAULT 'candidate',  -- candidate, active, invalidated
    category             TEXT,
    source_observation_ids TEXT,
    observation_count    INTEGER DEFAULT 1,
    created_at           TEXT NOT NULL
);

-- ── Feedback / adaptive capsule ───────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS capsule_feedback (
    query_hash    TEXT PRIMARY KEY,
    feedback_hint TEXT,
    relationship  TEXT,
    token_count   INTEGER,
    relevance_score REAL,
    depth         INTEGER,
    sample_count  INTEGER DEFAULT 1,
    avg_usage_pct REAL DEFAULT 0.0,
    avg_follow_ups REAL DEFAULT 0.0,
    updated_at    TEXT NOT NULL
);

-- ── Usage tracking ────────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS daily_usage (
    usage_date   TEXT NOT NULL,
    tool_category TEXT NOT NULL,
    call_count   INTEGER DEFAULT 0,
    PRIMARY KEY (usage_date, tool_category)
);

CREATE TABLE IF NOT EXISTS token_savings_log (
    id           INTEGER PRIMARY KEY,
    tool_name    TEXT,
    tokens_budget INTEGER,
    tokens_used  INTEGER,
    tokens_saved INTEGER,
    saving_pct   REAL,
    session_id   TEXT,
    created_at   TEXT NOT NULL
);

-- ── Workspaces (multi-repo) ───────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS workspaces (
    id         TEXT PRIMARY KEY,
    root_path  TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS workspace_repos (
    workspace_id TEXT NOT NULL REFERENCES workspaces(id),
    repo_alias   TEXT NOT NULL,
    repo_path    TEXT NOT NULL,
    PRIMARY KEY (workspace_id, repo_alias)
);

-- ── Meta ──────────────────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

-- ── FTS5 virtual tables ───────────────────────────────────────────────────────
-- These are kept in sync via triggers below.

CREATE VIRTUAL TABLE IF NOT EXISTS nodes_fts USING fts5(
    name, fqn, docstring, signature,
    content='nodes',
    content_rowid='id'
);

CREATE VIRTUAL TABLE IF NOT EXISTS observations_fts USING fts5(
    content,
    content='observations',
    content_rowid='id'
);

-- ── FTS5 sync triggers ────────────────────────────────────────────────────────

CREATE TRIGGER IF NOT EXISTS nodes_ai AFTER INSERT ON nodes BEGIN
    INSERT INTO nodes_fts(rowid, name, fqn, docstring, signature)
    VALUES (new.id, new.name, new.fqn, new.docstring, new.signature);
END;

CREATE TRIGGER IF NOT EXISTS nodes_au AFTER UPDATE ON nodes BEGIN
    INSERT INTO nodes_fts(nodes_fts, rowid, name, fqn, docstring, signature)
    VALUES ('delete', old.id, old.name, old.fqn, old.docstring, old.signature);
    INSERT INTO nodes_fts(rowid, name, fqn, docstring, signature)
    VALUES (new.id, new.name, new.fqn, new.docstring, new.signature);
END;

CREATE TRIGGER IF NOT EXISTS nodes_ad AFTER DELETE ON nodes BEGIN
    INSERT INTO nodes_fts(nodes_fts, rowid, name, fqn, docstring, signature)
    VALUES ('delete', old.id, old.name, old.fqn, old.docstring, old.signature);
END;

CREATE TRIGGER IF NOT EXISTS obs_ai AFTER INSERT ON observations BEGIN
    INSERT INTO observations_fts(rowid, content) VALUES (new.id, new.content);
END;

CREATE TRIGGER IF NOT EXISTS obs_au AFTER UPDATE ON observations BEGIN
    INSERT INTO observations_fts(observations_fts, rowid, content)
    VALUES ('delete', old.id, old.content);
    INSERT INTO observations_fts(rowid, content) VALUES (new.id, new.content);
END;

CREATE TRIGGER IF NOT EXISTS obs_ad AFTER DELETE ON observations BEGIN
    INSERT INTO observations_fts(observations_fts, rowid, content)
    VALUES ('delete', old.id, old.content);
END;
"""


def _needs_rebuild(conn: sqlite3.Connection) -> bool:
    """Return True if the existing schema is missing required columns."""
    # Check schema version first
    try:
        row = conn.execute("SELECT value FROM meta WHERE key='schema_version'").fetchone()
        if row and row[0] == SCHEMA_VERSION:
            return False
    except sqlite3.OperationalError:
        return True  # meta table doesn't exist yet

    # Version mismatch — also verify critical columns exist
    try:
        cols = {r[1] for r in conn.execute("PRAGMA table_info(nodes)").fetchall()}
        return "kind" not in cols or "repo_alias" not in cols
    except sqlite3.OperationalError:
        return True


def _drop_all(conn: sqlite3.Connection) -> None:
    """Drop all user tables so DDL can recreate them cleanly."""
    tables = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
    ).fetchall()
    triggers = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='trigger'"
    ).fetchall()
    for (t,) in triggers:
        conn.execute(f"DROP TRIGGER IF EXISTS [{t}]")
    for (t,) in tables:
        conn.execute(f"DROP TABLE IF EXISTS [{t}]")
    conn.commit()


def open_db(db_path: str | Path) -> sqlite3.Connection:
    """Open (or create) the index database and apply the schema."""
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    conn.row_factory = sqlite3.Row

    if _needs_rebuild(conn):
        print(f"Schema migration required — rebuilding index at {db_path}")
        _drop_all(conn)

    # executescript handles multi-statement DDL (triggers with BEGIN/END)
    conn.executescript(DDL)
    conn.execute(
        "INSERT OR IGNORE INTO meta (key, value) VALUES ('schema_version', ?)",
        (SCHEMA_VERSION,),
    )
    conn.commit()
    return conn
