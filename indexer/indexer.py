"""
Main incremental indexer — orchestrates scanner → symbols → embedder → coupling.

Incremental strategy:
  - Hash each file with blake3
  - Skip files whose hash hasn't changed (file_cache)
  - For changed/new files: delete old nodes+edges, re-extract, re-embed
  - After all files: rebuild TF-IDF (IDF must be recomputed over full corpus)
  - Run change coupling once at the end
"""

import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import blake3

from . import scanner, symbols, embedder, coupling


def _hash_file(path: Path) -> str:
    return blake3.blake3(path.read_bytes()).hexdigest()


def _upsert_node(conn: sqlite3.Connection, sym: symbols.Symbol, repo_alias: str = "primary") -> int:
    """Insert or replace a node, return its rowid."""
    cur = conn.execute(
        """INSERT INTO nodes
           (name, fqn, file_path, kind, start_line, end_line,
            signature, docstring, is_exported, is_test, repo_alias)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
           ON CONFLICT(fqn) DO UPDATE SET
             name=excluded.name, file_path=excluded.file_path,
             kind=excluded.kind, start_line=excluded.start_line,
             end_line=excluded.end_line, signature=excluded.signature,
             docstring=excluded.docstring, is_exported=excluded.is_exported,
             is_test=excluded.is_test
           RETURNING id""",
        (sym.name, sym.fqn, sym.file_path, sym.kind,
         sym.start_line, sym.end_line, sym.signature, sym.docstring,
         int(sym.is_exported), int(sym.is_test), repo_alias),
    )
    return cur.fetchone()[0]


def _resolve_call_edges(conn: sqlite3.Connection, edges: list[symbols.CallEdge]) -> None:
    """
    Resolve unqualified target names to node IDs and insert edges.
    For each call edge, look up the target by name (best-effort).
    """
    for edge in edges:
        if edge.edge_type == "CONTAINS":
            # CONTAINS edges have FQNs on both sides
            src = conn.execute("SELECT id FROM nodes WHERE fqn=?", (edge.source_fqn,)).fetchone()
            tgt = conn.execute("SELECT id FROM nodes WHERE fqn=?", (edge.target_name,)).fetchone()
        else:
            src = conn.execute("SELECT id FROM nodes WHERE fqn=?", (edge.source_fqn,)).fetchone()
            # Try FQN first, then name
            tgt = (conn.execute("SELECT id FROM nodes WHERE fqn=?", (edge.target_name,)).fetchone()
                   or conn.execute("SELECT id FROM nodes WHERE name=? LIMIT 1", (edge.target_name,)).fetchone())

        if src and tgt and src[0] != tgt[0]:
            conn.execute(
                """INSERT OR IGNORE INTO edges
                   (source_id, target_id, type, call_site_line, confidence)
                   VALUES (?, ?, ?, ?, ?)""",
                (src[0], tgt[0], edge.edge_type, edge.call_site_line, edge.confidence),
            )


def _delete_file_nodes(conn: sqlite3.Connection, rel_path: str) -> None:
    """Remove all nodes (and cascaded edges/embeddings) for a file."""
    conn.execute("DELETE FROM nodes WHERE file_path=?", (rel_path,))


def index(
    root: str | Path,
    db_path: str | Path | None = None,
    repo_alias: str = "primary",
    skip_coupling: bool = False,
) -> sqlite3.Connection:
    """
    Index the repository at *root*.  Returns the open database connection.

    Parameters
    ----------
    root         : repository root directory
    db_path      : explicit path to index.db (default: <root>/.vexp/index.db)
    repo_alias   : label for this repo in multi-repo setups
    skip_coupling: skip git change coupling (faster, useful for testing)
    """
    from pyvexp.schema import open_db  # avoid circular at module level

    root = Path(root).resolve()
    if db_path is None:
        db_path = root / ".vexp" / "index.db"

    conn = open_db(db_path)

    # ── Scan files ─────────────────────────────────────────────────────────
    files = scanner.scan(root)
    print(f"Scanning {len(files)} files in {root}")

    changed_node_ids: list[int] = []
    now = datetime.now(timezone.utc).isoformat()
    all_edges: list[symbols.CallEdge] = []

    for file_path, lang in files:
        rel = str(file_path.relative_to(root))
        try:
            h = _hash_file(file_path)
        except OSError:
            continue

        # Check cache
        cached = conn.execute(
            "SELECT blake3_hash FROM file_cache WHERE file_path=?", (rel,)
        ).fetchone()

        if cached and cached[0] == h:
            continue  # unchanged

        # File changed — remove old data
        _delete_file_nodes(conn, rel)

        # Extract symbols
        try:
            file_syms = symbols.extract(file_path, lang, root)
        except Exception as e:
            print(f"  Warning: parse error in {rel}: {e}")
            continue

        # Insert nodes
        for sym in file_syms.symbols:
            nid = _upsert_node(conn, sym, repo_alias)
            changed_node_ids.append(nid)

        all_edges.extend(file_syms.edges)

        # Update cache
        conn.execute(
            """INSERT OR REPLACE INTO file_cache
               (file_path, blake3_hash, last_indexed_at, node_count)
               VALUES (?, ?, ?, ?)""",
            (rel, h, now, len(file_syms.symbols)),
        )

    conn.commit()
    print(f"Indexed {len(changed_node_ids)} new/changed nodes")

    # ── Resolve and insert call edges ──────────────────────────────────────
    if all_edges:
        _resolve_call_edges(conn, all_edges)
        conn.commit()
        print(f"Resolved {len(all_edges)} candidate edges")

    # ── Rebuild TF-IDF vectors ─────────────────────────────────────────────
    if changed_node_ids:
        embedder.build_incremental(conn, changed_node_ids)
    else:
        print("No changes — embeddings up to date")

    # ── Git change coupling ────────────────────────────────────────────────
    if not skip_coupling:
        coupling.compute(conn, root)

    return conn
