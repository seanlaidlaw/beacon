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
            signature, docstring, body_preview, is_exported, is_test, repo_alias)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
           ON CONFLICT(fqn) DO UPDATE SET
             name=excluded.name, file_path=excluded.file_path,
             kind=excluded.kind, start_line=excluded.start_line,
             end_line=excluded.end_line, signature=excluded.signature,
             docstring=excluded.docstring, body_preview=excluded.body_preview,
             is_exported=excluded.is_exported, is_test=excluded.is_test
           RETURNING id""",
        (sym.name, sym.fqn, sym.file_path, sym.kind,
         sym.start_line, sym.end_line, sym.signature, sym.docstring, sym.body_preview,
         int(sym.is_exported), int(sym.is_test), repo_alias),
    )
    return cur.fetchone()[0]


def _resolve_call_edges(conn: sqlite3.Connection, edges: list[symbols.CallEdge]) -> None:
    """
    Resolve unqualified target names to node IDs and insert edges.
    For each call edge, look up the target by name (best-effort).

    IMPORTS edges are also stored verbatim in import_refs so that "what files
    import module X?" queries work even when the target has no indexed node.
    """
    for edge in edges:
        if edge.edge_type == "IMPORTS":
            # Store raw import ref regardless of whether target resolves to a node
            source_file = edge.source_fqn.split("::")[0]
            conn.execute(
                "INSERT OR IGNORE INTO import_refs (source_file, target_module, call_site_line) "
                "VALUES (?, ?, ?)",
                (source_file, edge.target_name, edge.call_site_line),
            )

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
    conn.execute("DELETE FROM import_refs WHERE source_file=?", (rel_path,))


def check_and_reindex(
    conn: sqlite3.Connection,
    root: str | Path,
    repo_alias: str = "primary",
    silent: bool = False,
) -> int:
    """
    Scan *root* for changed/new/deleted files and incrementally update the index.
    Returns the number of files that were re-indexed (0 = nothing changed).

    Designed to be called cheaply at query time — the file scan is fast (os.walk
    with directory pruning) and only changed files trigger re-parsing.
    """
    root = Path(root).resolve()
    now = datetime.now(timezone.utc).isoformat()

    # Current state on disk
    current_files: dict[str, tuple[Path, str]] = {
        str(p.relative_to(root)): (p, lang)
        for p, lang in scanner.scan(root)
    }

    # Cached state in the DB
    cached_hashes: dict[str, str] = {
        r[0]: r[1]
        for r in conn.execute("SELECT file_path, blake3_hash FROM file_cache").fetchall()
    }

    changed: list[tuple[str, Path, str]] = []   # (rel, path, lang) — new or modified
    deleted: list[str] = []                      # rel paths no longer on disk

    # Detect modified / new files
    for rel, (path, lang) in current_files.items():
        try:
            h = _hash_file(path)
        except OSError:
            continue
        if cached_hashes.get(rel) != h:
            changed.append((rel, path, lang))

    # Detect deleted files (in cache but gone from disk)
    for rel in cached_hashes:
        if rel not in current_files:
            deleted.append(rel)

    if not changed and not deleted:
        return 0

    if not silent:
        parts = []
        if changed:
            parts.append(f"{len(changed)} changed/new")
        if deleted:
            parts.append(f"{len(deleted)} deleted")
        print(f"[beacon] auto-reindex: {', '.join(parts)}", flush=True)

    # Remove deleted files
    for rel in deleted:
        _delete_file_nodes(conn, rel)
        conn.execute("DELETE FROM file_cache WHERE file_path=?", (rel,))

    # Re-index changed/new files
    changed_node_ids: list[int] = []
    all_edges: list[symbols.CallEdge] = []

    for rel, path, lang in changed:
        _delete_file_nodes(conn, rel)
        try:
            file_syms = symbols.extract(path, lang, root)
        except Exception:
            continue

        for sym in file_syms.symbols:
            nid = _upsert_node(conn, sym, repo_alias)
            changed_node_ids.append(nid)

        all_edges.extend(file_syms.edges)

        h = _hash_file(path)
        conn.execute(
            "INSERT OR REPLACE INTO file_cache (file_path, blake3_hash, last_indexed_at, node_count) "
            "VALUES (?, ?, ?, ?)",
            (rel, h, now, len(file_syms.symbols)),
        )

    conn.commit()

    if all_edges:
        _resolve_call_edges(conn, all_edges)
        conn.commit()

    if changed_node_ids:
        embedder.build_incremental(conn, changed_node_ids)
        embedder.build_dense_incremental(conn, changed_node_ids)
        # Rebuild FTS5 so new nodes are searchable immediately
        conn.execute("INSERT INTO nodes_fts(nodes_fts) VALUES('rebuild')")
        conn.commit()

    return len(changed) + len(deleted)


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
    db_path      : explicit path to index.db (default: <root>/.beacon/index.db)
    repo_alias   : label for this repo in multi-repo setups
    skip_coupling: skip git change coupling (faster, useful for testing)
    """
    from beacon.schema import open_db  # avoid circular at module level

    root = Path(root).resolve()
    if db_path is None:
        db_path = root / ".beacon" / "index.db"

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

    # ── Rebuild TF-IDF + dense vectors ────────────────────────────────────
    if changed_node_ids:
        embedder.build_incremental(conn, changed_node_ids)
        embedder.build_dense_incremental(conn, changed_node_ids)
    else:
        print("No changes — embeddings up to date")

    # ── Git change coupling ────────────────────────────────────────────────
    if not skip_coupling:
        coupling.compute(conn, root)

    return conn
