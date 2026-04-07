"""
pyvexp CLI

Commands:
    index     [<dir>]   — scan + index a codebase
    search    <query>   — search the index
    capsule   <query>   — generate a context capsule
    mcp                 — start MCP stdio server
    show-config         — print MCP config for AI agents
"""

import argparse
import json
import os
import sys
from pathlib import Path


# ── index ─────────────────────────────────────────────────────────────────────

def cmd_index(args):
    import sqlite3
    from datetime import datetime, timezone

    import blake3
    import tqdm

    from pyvexp.indexer import scanner, symbols, embedder, coupling
    from pyvexp.schema import open_db

    root = Path(args.dir or os.getcwd()).resolve()
    db_path = Path(args.db) if args.db else root / ".vexp" / "index.db"

    print(f"pyvexp index — {root}")
    print()

    # ── Phase 1: scan (fast) ──────────────────────────────────────────────────
    print("Scanning files...", end=" ", flush=True)
    sc = scanner.Scanner(root)
    all_files = sc.collect()

    # Apply --langs filter if given
    if args.langs:
        keep = {l.strip().lower() for l in args.langs.split(",")}
        all_files = [(p, l) for p, l in all_files if l in keep]

    print(f"{len(all_files)} indexable files found")
    if not all_files:
        print("Nothing to index.")
        return

    # Show language breakdown
    from collections import Counter
    lang_counts = Counter(lang for _, lang in all_files)
    print("  " + "  ".join(f"{lang}:{n}" for lang, n in lang_counts.most_common()))
    print()

    # Prompt before starting a large index (unless --yes)
    if len(all_files) > 5000 and not args.yes:
        print(f"  This will parse {len(all_files):,} files.")
        print(f"  Use --langs python  (or comma-separated list) to restrict languages.")
        ans = input("  Proceed? [y/N] ").strip().lower()
        if ans != "y":
            print("Aborted.")
            return
        print()

    # ── Phase 2: hash-check (determine what's changed) ────────────────────────
    conn = open_db(db_path)

    cached = {
        r[0]: r[1]
        for r in conn.execute("SELECT file_path, blake3_hash FROM file_cache").fetchall()
    }

    to_index: list[tuple[Path, str, str]] = []   # (path, lang, rel)
    for file_path, lang in all_files:
        rel = str(file_path.relative_to(root))
        try:
            h = blake3.blake3(file_path.read_bytes()).hexdigest()
        except OSError:
            continue
        if cached.get(rel) != h:
            to_index.append((file_path, lang, rel))

    skipped = len(all_files) - len(to_index)
    if skipped:
        print(f"  {skipped} files unchanged — skipping")
    if not to_index:
        print("Index is already up to date.")
        _finish(conn, root, args)
        return
    print(f"  {len(to_index)} files to index")
    print()

    # ── Phase 3: parse + insert (with progress bar) ───────────────────────────
    now = datetime.now(timezone.utc).isoformat()
    changed_node_ids: list[int] = []
    all_edges: list[symbols.CallEdge] = []
    parse_errors = 0

    bar = tqdm.tqdm(
        to_index,
        desc="Indexing",
        unit="file",
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
        dynamic_ncols=True,
    )

    for file_path, lang, rel in bar:
        bar.set_postfix_str(rel[-40:] if len(rel) > 40 else rel, refresh=False)

        # Remove stale nodes for this file
        conn.execute("DELETE FROM nodes WHERE file_path=?", (rel,))

        try:
            file_syms = symbols.extract(file_path, lang, root)
        except Exception:
            parse_errors += 1
            continue

        for sym in file_syms.symbols:
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
                 int(sym.is_exported), int(sym.is_test), "primary"),
            )
            nid = cur.fetchone()[0]
            changed_node_ids.append(nid)

        all_edges.extend(file_syms.edges)

        h = blake3.blake3(file_path.read_bytes()).hexdigest()
        conn.execute(
            "INSERT OR REPLACE INTO file_cache (file_path, blake3_hash, last_indexed_at, node_count) "
            "VALUES (?, ?, ?, ?)",
            (rel, h, now, len(file_syms.symbols)),
        )

        # Commit every 50 files to avoid one giant transaction
        if len(changed_node_ids) % 50 == 0:
            conn.commit()

    conn.commit()
    bar.close()

    node_total = conn.execute("SELECT COUNT(*) FROM nodes").fetchone()[0]
    print(f"\n  {len(changed_node_ids)} nodes indexed  ({node_total} total)")
    if parse_errors:
        print(f"  {parse_errors} files skipped (parse errors)")

    # ── Phase 4: resolve call edges ───────────────────────────────────────────
    if all_edges:
        print()
        _resolve_edges(conn, all_edges)

    # ── Phase 5: TF-IDF embeddings ────────────────────────────────────────────
    if changed_node_ids:
        print()
        print("Building TF-IDF embeddings...", end=" ", flush=True)
        embedder.build_incremental(conn, changed_node_ids)

    # ── Phase 6: git change coupling ─────────────────────────────────────────
    if not args.no_coupling:
        print()
        coupling.compute(conn, root)

    # ── Phase 7: rebuild FTS5 index to guarantee sync ─────────────────────────
    print()
    print("Rebuilding FTS5 index...", end=" ", flush=True)
    conn.execute("INSERT INTO nodes_fts(nodes_fts) VALUES('rebuild')")
    conn.execute("INSERT INTO observations_fts(observations_fts) VALUES('rebuild')")
    conn.commit()
    fts_count = conn.execute("SELECT COUNT(*) FROM nodes_fts").fetchone()[0]
    print(f"done ({fts_count:,} entries)")

    _finish(conn, root, args)


def _resolve_edges(conn, all_edges):
    import tqdm
    from pyvexp.indexer.symbols import CallEdge

    print(f"Resolving {len(all_edges)} call edges...", end=" ", flush=True)
    inserted = 0
    for edge in all_edges:
        if edge.edge_type == "CONTAINS":
            src = conn.execute("SELECT id FROM nodes WHERE fqn=?", (edge.source_fqn,)).fetchone()
            tgt = conn.execute("SELECT id FROM nodes WHERE fqn=?", (edge.target_name,)).fetchone()
        else:
            src = conn.execute("SELECT id FROM nodes WHERE fqn=?", (edge.source_fqn,)).fetchone()
            tgt = (conn.execute("SELECT id FROM nodes WHERE fqn=?", (edge.target_name,)).fetchone()
                   or conn.execute("SELECT id FROM nodes WHERE name=? LIMIT 1", (edge.target_name,)).fetchone())
        if src and tgt and src[0] != tgt[0]:
            conn.execute(
                "INSERT OR IGNORE INTO edges (source_id, target_id, type, call_site_line, confidence) "
                "VALUES (?, ?, ?, ?, ?)",
                (src[0], tgt[0], edge.edge_type, edge.call_site_line, edge.confidence),
            )
            inserted += 1
    conn.commit()
    edge_total = conn.execute("SELECT COUNT(*) FROM edges").fetchone()[0]
    print(f"{inserted} new  ({edge_total} total)")


def _finish(conn, root, args):
    from datetime import datetime, timezone
    n = conn.execute("SELECT COUNT(*) FROM nodes").fetchone()[0]
    e = conn.execute("SELECT COUNT(*) FROM edges").fetchone()[0]
    emb = conn.execute("SELECT COUNT(*) FROM node_embeddings").fetchone()[0]
    db_path = Path(args.db) if args.db else root / ".vexp" / "index.db"

    # Write healthy marker — read by the Claude Code hook to know the index is ready
    healthy = db_path.parent / "healthy"
    healthy.write_text(
        f"nodes={n}\nedges={e}\nembeddings={emb}\n"
        f"indexed_at={datetime.now(timezone.utc).isoformat()}\n"
    )

    print()
    print("─" * 50)
    print(f"  nodes:      {n:,}")
    print(f"  edges:      {e:,}")
    print(f"  embeddings: {emb:,}")
    print(f"  index:      {db_path}")
    print("─" * 50)


# ── search ────────────────────────────────────────────────────────────────────

def cmd_search(args):
    from pyvexp.schema import open_db
    from pyvexp.search.query import search
    db = args.db or str(Path(os.getcwd()) / ".vexp" / "index.db")
    conn = open_db(db)
    results = search(conn, args.query, limit=args.limit)
    if not results:
        print("No results.")
        return
    for r in results:
        print(f"[{r.score:.3f}] {r.kind:8s} {r.fqn}")
        print(f"         {r.file_path}:{r.start_line}  — {r.reason}")
        if r.signature:
            print(f"         sig: {r.signature[:80]}")
        print()


# ── capsule ───────────────────────────────────────────────────────────────────

def cmd_capsule(args):
    from pyvexp.schema import open_db
    from pyvexp.search.capsule import get_capsule, render_capsule
    db = args.db or str(Path(os.getcwd()) / ".vexp" / "index.db")
    conn = open_db(db)
    cap = get_capsule(conn, args.query, max_tokens=args.max_tokens)
    print(render_capsule(cap))


# ── mcp ───────────────────────────────────────────────────────────────────────

def cmd_mcp(args):
    from pyvexp.mcp import McpServer
    workspace = Path(args.workspace or os.getcwd()).resolve()
    db = Path(args.db) if args.db else None
    McpServer(workspace=workspace, db_path=db).run()


# ── setup ────────────────────────────────────────────────────────────────────

GUARD_SCRIPT = """\
#!/bin/bash
# pyvexp-guard: redirect Grep/Glob/Read to pyvexp MCP tools when index is ready.
# Checks for .vexp/index.db and .vexp/healthy marker written by `pyvexp index`.
VEXP_DIR="${CLAUDE_PROJECT_DIR:-.}/.vexp"
HEALTHY="$VEXP_DIR/healthy"
DB="$VEXP_DIR/index.db"

if [ -f "$DB" ] && [ -f "$HEALTHY" ]; then
  printf '{"hookSpecificOutput":{"hookEventName":"PreToolUse","permissionDecision":"deny","permissionDecisionReason":"pyvexp index is ready. Use run_pipeline or get_context_capsule instead of Grep/Glob/Read — it searches semantically and saves tokens."}}'
else
  printf '{"hookSpecificOutput":{"hookEventName":"PreToolUse","permissionDecision":"allow","permissionDecisionReason":"pyvexp index not ready, allowing direct search fallback."}}'
fi
exit 0
"""

HOOK_CONFIG = {
    "PreToolUse": [
        {
            "matcher": "Grep|Glob|Read",
            "hooks": [
                {
                    "type": "command",
                    "command": "$CLAUDE_PROJECT_DIR/.claude/hooks/pyvexp-guard.sh",
                    "timeout": 3000,
                }
            ],
        }
    ]
}


def cmd_setup(args):
    workspace = Path(args.workspace or os.getcwd()).resolve()
    python = sys.executable
    pyvexp_dir = str(Path(__file__).resolve().parent.parent)
    db_path = workspace / ".vexp" / "index.db"

    # ── 1. Write guard script ─────────────────────────────────────────────────
    hooks_dir = workspace / ".claude" / "hooks"
    hooks_dir.mkdir(parents=True, exist_ok=True)
    guard = hooks_dir / "pyvexp-guard.sh"
    guard.write_text(GUARD_SCRIPT)
    guard.chmod(0o755)
    print(f"  hook script → {guard}")

    # ── 2. Hooks → project .claude/settings.json  ────────────────────────────
    # Hooks are project-scoped (they reference CLAUDE_PROJECT_DIR).
    settings_path = workspace / ".claude" / "settings.json"
    settings = json.loads(settings_path.read_text()) if settings_path.exists() else {}

    hooks = settings.setdefault("hooks", {})
    existing = hooks.get("PreToolUse", [])
    # Remove any prior pyvexp-guard entry before re-adding
    existing = [e for e in existing
                if not any("pyvexp-guard" in str(h.get("command", ""))
                           for h in e.get("hooks", []))]
    existing.extend(HOOK_CONFIG["PreToolUse"])
    hooks["PreToolUse"] = existing

    # Remove any stale mcpServers key that may have been written here previously
    settings.pop("mcpServers", None)

    settings_path.write_text(json.dumps(settings, indent=2))
    print(f"  hooks       → {settings_path}")

    # ── 3. MCP server → ~/.claude.json  (user scope — shown in /mcp) ─────────
    user_cfg_path = Path.home() / ".claude.json"
    user_cfg = json.loads(user_cfg_path.read_text()) if user_cfg_path.exists() else {}

    user_cfg.setdefault("mcpServers", {})["pyvexp"] = {
        "command": python,
        "args": ["-m", "pyvexp.mcp",
                 "--workspace", str(workspace),
                 "--db", str(db_path)],
        "env": {"PYTHONPATH": pyvexp_dir},
    }

    user_cfg_path.write_text(json.dumps(user_cfg, indent=2))
    print(f"  MCP server  → {user_cfg_path}  (user scope)")

    # ── 4. Add .vexp/ to .gitignore ───────────────────────────────────────────
    gitignore = workspace / ".gitignore"
    entry = ".vexp/\n"
    if gitignore.exists():
        content = gitignore.read_text()
        if ".vexp" not in content:
            gitignore.write_text(content.rstrip("\n") + "\n" + entry)
            print(f"  .gitignore  → added .vexp/")
    else:
        gitignore.write_text(entry)
        print(f"  .gitignore  → created with .vexp/")

    print()
    print("Setup complete. Now run:")
    print(f"  python -m pyvexp.cli index {workspace} --no-coupling")
    print("Then restart Claude Code to activate the MCP and hook.")


# ── show-config ───────────────────────────────────────────────────────────────

def cmd_show_config(args):
    workspace = str(Path(args.workspace or os.getcwd()).resolve())
    python = sys.executable
    pyvexp_dir = str(Path(__file__).resolve().parent.parent)
    db = str(Path(workspace) / ".vexp" / "index.db")

    config = {
        "command": python,
        "args": ["-m", "pyvexp.mcp", "--workspace", workspace, "--db", db],
        "env": {"PYTHONPATH": pyvexp_dir},
    }
    print(json.dumps({"mcpServers": {"pyvexp": config}}, indent=2))


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(prog="python -m pyvexp.cli")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("index", help="Index a directory")
    p.add_argument("dir", nargs="?", help="Root directory (default: cwd)")
    p.add_argument("--db", help="Path to index.db")
    p.add_argument("--no-coupling", action="store_true", help="Skip git change coupling")
    p.add_argument("--langs", help="Comma-separated languages to index, e.g. python,typescript")
    p.add_argument("--yes", "-y", action="store_true", help="Skip confirmation prompt")

    p = sub.add_parser("search", help="Search the index")
    p.add_argument("query")
    p.add_argument("--db")
    p.add_argument("--limit", type=int, default=10)

    p = sub.add_parser("capsule", help="Generate a context capsule")
    p.add_argument("query")
    p.add_argument("--db")
    p.add_argument("--max-tokens", type=int, default=8000)

    p = sub.add_parser("mcp", help="Start MCP stdio server")
    p.add_argument("--workspace")
    p.add_argument("--db")

    p = sub.add_parser("show-config", help="Print MCP config for AI agents")
    p.add_argument("--workspace")

    p = sub.add_parser("setup", help="Install hook + MCP config for a project")
    p.add_argument("workspace", nargs="?", help="Project root (default: cwd)")

    args = parser.parse_args()
    {"index": cmd_index, "search": cmd_search, "capsule": cmd_capsule,
     "mcp": cmd_mcp, "show-config": cmd_show_config,
     "setup": cmd_setup}[args.cmd](args)


if __name__ == "__main__":
    main()
