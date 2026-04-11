"""
MCP stdio server — JSON-RPC 2.0 over stdin/stdout.

Implements the same 11 tools as vexp-core:
  run_pipeline, get_context_capsule, get_impact_graph, search_logic_flow,
  get_skeleton, index_status, get_session_context, save_observation,
  search_memory, submit_lsp_edges, workspace_setup

Start with:
    beacon mcp --workspace /path/to/repo
"""

import json
import os
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

# ── Tool definitions (inputSchema matches real vexp MCP tools) ────────────────

TOOLS = [
    {
        "name": "run_pipeline",
        "description": (
            "PRIMARY TOOL — full analysis pipeline (context search + impact + memory). "
            "Use for ANY codebase task before making changes. "
            "For natural-language questions, also pass `hypothetical_code` — a short "
            "code snippet in the target language that resembles what you're looking for. "
            "This is embedded directly against the index (HyDE technique) and dramatically "
            "improves semantic retrieval. BM25 keyword search still uses `task`, so both "
            "signals work together."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "task": {"type": "string", "description": "Description of the task"},
                "hypothetical_code": {
                    "type": "string",
                    "description": (
                        "RECOMMENDED for natural-language queries. A short code snippet "
                        "(5-20 lines) in the target language resembling the code you want "
                        "to find — invent plausible identifier names. Example: to find "
                        "encounter filtering by test date in Swift, write "
                        "`let filtered = encounters.filter { $0.date > lastTestDate }`. "
                        "Used for semantic (dense) search only; BM25 still uses `task`."
                    ),
                },
                "preset": {
                    "type": "string",
                    "enum": ["auto", "explore", "modify", "debug", "refactor"],
                    "description": "Analysis preset (default: auto)",
                },
                "max_tokens": {"type": "number", "description": "Token budget (default: 10000)"},
                "repos": {"type": "array", "items": {"type": "string"}},
                "include_file_content": {"type": "boolean"},
                "include_tests": {"type": "boolean"},
                "observation": {"type": "string", "description": "Auto-save this observation"},
            },
            "required": ["task"],
        },
    },
    {
        "name": "get_context_capsule",
        "description": (
            "Lightweight context search via semantic + graph search. "
            "Pass `hypothetical_code` alongside `query` for better semantic matching "
            "on natural-language questions (HyDE technique)."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "hypothetical_code": {
                    "type": "string",
                    "description": (
                        "RECOMMENDED for natural-language queries. A short code snippet "
                        "in the target language that resembles what you want to find. "
                        "Used for semantic (dense) search; BM25 still uses `query`."
                    ),
                },
                "max_tokens": {"type": "number"},
                "pivot_depth": {"type": "number"},
                "include_tests": {"type": "boolean"},
                "skeleton_detail": {"type": "string", "enum": ["minimal", "standard", "detailed"]},
                "repos": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["query"],
        },
    },
    {
        "name": "get_impact_graph",
        "description": "All code that would be affected if a symbol changes.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "symbol_fqn": {"type": "string"},
                "depth": {"type": "number"},
                "format": {"type": "string", "enum": ["list", "tree", "mermaid"]},
                "cross_repo": {"type": "boolean"},
            },
            "required": ["symbol_fqn"],
        },
    },
    {
        "name": "search_logic_flow",
        "description": "Find execution paths between two symbols through the call graph.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "start": {"type": "string", "description": "FQN of start symbol"},
                "end": {"type": "string", "description": "FQN of end symbol"},
                "max_paths": {"type": "number"},
                "cross_repo": {"type": "boolean"},
            },
            "required": ["start", "end"],
        },
    },
    {
        "name": "get_skeleton",
        "description": "Token-efficient file summary — signatures only, no bodies (70-90% token savings).",
        "inputSchema": {
            "type": "object",
            "properties": {
                "files": {"type": "array", "items": {"type": "string"}},
                "detail": {"type": "string", "enum": ["minimal", "standard", "detailed"]},
                "repo": {"type": "string"},
            },
            "required": ["files"],
        },
    },
    {
        "name": "index_status",
        "description": "Current index stats: node/edge counts, file coverage, last indexed.",
        "inputSchema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "get_session_context",
        "description": "Observations from the current (and optionally previous) sessions.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "include_previous": {"type": "boolean"},
                "max_results": {"type": "number"},
                "types": {
                    "type": "array",
                    "items": {"type": "string", "enum": ["tool_call", "insight", "decision", "error", "manual"]},
                },
            },
            "required": [],
        },
    },
    {
        "name": "save_observation",
        "description": "Persist an insight or decision to session memory with optional FQN links.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "content": {"type": "string"},
                "type": {"type": "string", "enum": ["insight", "decision", "error"]},
                "linked_symbols": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["content"],
        },
    },
    {
        "name": "search_memory",
        "description": "Search observations by keyword + semantic similarity across all sessions.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "max_results": {"type": "number"},
                "include_stale": {"type": "boolean"},
                "time_range_days": {"type": "number"},
                "session_filter": {"type": "string"},
            },
            "required": ["query"],
        },
    },
    {
        "name": "submit_lsp_edges",
        "description": "Submit type-resolved call edges from a Language Server.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "edges": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "source_fqn": {"type": "string"},
                            "target_fqn": {"type": "string"},
                            "edge_type": {"type": "string"},
                            "language": {"type": "string"},
                        },
                        "required": ["source_fqn", "target_fqn"],
                    },
                }
            },
            "required": ["edges"],
        },
    },
    {
        "name": "workspace_setup",
        "description": "Set up beacon for the current workspace.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "workspace_root": {"type": "string"},
                "detect_agents": {"type": "boolean"},
            },
            "required": [],
        },
    },
    {
        "name": "resolve_symbols",
        "description": (
            "Resolve symbol names or partial FQNs against the indexed symbol table. "
            "Call this at the start of a task to get precise FQN anchors before "
            "calling run_pipeline — dramatically improves retrieval for queries that "
            "mention specific function or class names."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "names": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Symbol names or partial FQNs to look up",
                },
                "file_hint": {
                    "type": "string",
                    "description": "Optional: restrict search to files matching this substring",
                },
                "kind": {
                    "type": "string",
                    "description": "Optional: restrict to this kind (function, class, method, ...)",
                },
            },
            "required": ["names"],
        },
    },
]


# ── Handler ────────────────────────────────────────────────────────────────────

class McpServer:
    def __init__(self, workspace: str | Path, db_path: str | Path | None = None):
        self.workspace = Path(workspace).resolve()
        if db_path is None:
            db_path = self.workspace / ".beacon" / "index.db"
        self.db_path = Path(db_path)
        self._conn = None
        self.session_id = str(uuid.uuid4())
        # Session-level dedup: track FQNs already sent to avoid resending identical chunks
        self._sent_fqns: set[str] = set()
        # Throttle auto-reindex checks to at most once every 30 seconds
        self._last_reindex_check: float = 0.0
        self._reindex_interval: float = 30.0
        # Request log file — one JSONL file per session in .beacon/logs/
        log_dir = self.db_path.parent / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._log_path = log_dir / f"mcp_{ts}_{self.session_id[:8]}.jsonl"
        self._log_fh = open(self._log_path, "a", buffering=1)  # line-buffered

    def _log(self, entry: dict) -> None:
        """Append a JSON entry to the session log file. Never raises."""
        try:
            self._log_fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except Exception:
            pass

    def _conn_lazy(self):
        """Open DB connection on first use (lazy so startup is fast)."""
        if self._conn is None:
            from beacon.schema import open_db
            self._conn = open_db(self.db_path)
            self._ensure_session()
        return self._conn

    def _ensure_session(self):
        now = datetime.now(timezone.utc).isoformat()
        self._conn.execute(
            "INSERT OR IGNORE INTO sessions (id, created_at, updated_at) VALUES (?, ?, ?)",
            (self.session_id, now, now),
        )
        self._conn.commit()

    def _maybe_reindex(self) -> None:
        """
        Check for changed/new/deleted files and update the index if needed.
        Throttled to run at most once every _reindex_interval seconds.
        Runs silently — any output goes to stderr so it doesn't pollute MCP JSON.
        """
        import time
        now = time.monotonic()
        if now - self._last_reindex_check < self._reindex_interval:
            return
        self._last_reindex_check = now

        try:
            from beacon.indexer.indexer import check_and_reindex
            conn = self._conn_lazy()
            n = check_and_reindex(conn, self.workspace, silent=True)
            if n > 0:
                # Reset sent-FQN cache so updated symbols aren't suppressed by dedup
                self._sent_fqns.clear()
                import sys
                print(f"[beacon] auto-reindexed {n} file(s)", file=sys.stderr, flush=True)
        except Exception:
            pass  # never let a reindex failure break a query

    def _auto_observe(self, tool: str, params: dict, result_summary: str):
        """Auto-save a tool_call observation (mirrors vexp-core behaviour)."""
        conn = self._conn_lazy()
        now = datetime.now(timezone.utc).isoformat()
        summary = f"[{tool}] {result_summary[:200]}"
        conn.execute(
            "INSERT INTO observations (session_id, type, content, created_at, source) VALUES (?, ?, ?, ?, ?)",
            (self.session_id, "tool_call", summary, now, "agent"),
        )
        conn.commit()

    # ── Tool handlers ──────────────────────────────────────────────────────────

    def handle_run_pipeline(self, params: dict) -> str:
        from beacon.search.capsule import get_capsule, render_capsule
        from beacon.search.graph import get_impact_graph, format_impact_tree, get_skeleton, search_logic_flow, format_flow
        from beacon.search.query import expand_query

        task = params["task"]
        hypothetical_code = params.get("hypothetical_code") or None
        max_tokens = int(params.get("max_tokens", 10_000))
        preset = params.get("preset", "auto")
        explicit_steps = params.get("steps")
        conn = self._conn_lazy()
        parts: list[str] = []

        self._maybe_reindex()
        # Query expansion: resolve symbol names to FQN anchors
        _, anchor_fqns = expand_query(conn, task)

        if explicit_steps:
            # Advanced: user-specified pipeline steps
            for step in explicit_steps:
                tool = step.get("tool")
                sp = step.get("params") or {}
                if tool == "capsule":
                    cap = get_capsule(conn, task, max_tokens=sp.get("max_tokens", max_tokens // 2),
                                      hypothetical_code=hypothetical_code)
                    parts.append(render_capsule(cap))
                elif tool == "impact":
                    fqn = sp.get("symbol_fqn") or task
                    ig = get_impact_graph(conn, fqn, depth=sp.get("depth", 3))
                    parts.append(format_impact_tree(ig))
                elif tool == "flow":
                    paths = search_logic_flow(conn, sp.get("start", ""), sp.get("end", ""))
                    parts.append(format_flow(paths, sp.get("start",""), sp.get("end","")))
                elif tool == "skeleton":
                    files = sp.get("files") or []
                    parts.append(get_skeleton(conn, files, detail=sp.get("detail","standard"), root=str(self.workspace)))
                elif tool == "memory_search":
                    mem = self.handle_search_memory({"query": task, "max_results": 5})
                    parts.append(mem)
                elif tool == "save_observation":
                    obs = sp.get("content") or task
                    parts.append(self.handle_save_observation({"content": obs, "type": "insight"}))
        else:
            # Default pipeline based on preset
            cap = get_capsule(conn, task, max_tokens=max_tokens // 2, pivot_depth=2,
                              exclude_fqns=self._sent_fqns, anchor_fqns=anchor_fqns or None,
                              hypothetical_code=hypothetical_code)
            # Track what was sent for session-level dedup
            self._sent_fqns.update(n.fqn for n in cap.nodes)
            parts.append(render_capsule(cap))

            if preset in ("auto", "modify", "debug", "refactor") and cap.nodes:
                top_fqn = cap.nodes[0].fqn
                ig = get_impact_graph(conn, top_fqn, depth=3)
                if ig.nodes:
                    parts.append(format_impact_tree(ig))

            if preset in ("explore", "debug", "refactor"):
                mem = self.handle_search_memory({"query": task, "max_results": 5})
                if "No memory found" not in mem:
                    parts.append(mem)

        # Auto-save observation if provided
        if params.get("observation"):
            self.handle_save_observation({"content": params["observation"], "type": "insight"})

        output = "\n\n---\n\n".join(p for p in parts if p)
        self._auto_observe("run_pipeline", params, f"task={task!r}")
        return output

    def handle_get_context_capsule(self, params: dict) -> str:
        from beacon.search.capsule import get_capsule, render_capsule
        from beacon.search.query import expand_query
        self._maybe_reindex()
        conn = self._conn_lazy()
        query = params["query"]
        hypothetical_code = params.get("hypothetical_code") or None
        _, anchor_fqns = expand_query(conn, query)
        cap = get_capsule(
            conn,
            query,
            max_tokens=int(params.get("max_tokens", 8_000)),
            pivot_depth=int(params.get("pivot_depth", 2)),
            exclude_fqns=self._sent_fqns,
            anchor_fqns=anchor_fqns or None,
            hypothetical_code=hypothetical_code,
        )
        self._sent_fqns.update(n.fqn for n in cap.nodes)
        result = render_capsule(cap)
        self._auto_observe("get_context_capsule", params, f"query={query!r}, nodes={len(cap.nodes)}")
        return result

    def handle_get_impact_graph(self, params: dict) -> str:
        from beacon.search.graph import get_impact_graph, format_impact_tree, format_impact_list, format_impact_mermaid
        conn = self._conn_lazy()
        ig = get_impact_graph(
            conn,
            params["symbol_fqn"],
            depth=int(params.get("depth", 5)),
            cross_repo=bool(params.get("cross_repo", False)),
        )
        fmt = params.get("format", "tree")
        if fmt == "list":
            result = format_impact_list(ig)
        elif fmt == "mermaid":
            result = format_impact_mermaid(ig)
        else:
            result = format_impact_tree(ig)
        self._auto_observe("get_impact_graph", params, f"fqn={params['symbol_fqn']!r}, deps={len(ig.nodes)}")
        return result

    def handle_search_logic_flow(self, params: dict) -> str:
        from beacon.search.graph import search_logic_flow, format_flow
        conn = self._conn_lazy()
        paths = search_logic_flow(
            conn,
            params["start"],
            params["end"],
            max_paths=int(params.get("max_paths", 3)),
        )
        result = format_flow(paths, params["start"], params["end"])
        self._auto_observe("search_logic_flow", params, f"start={params['start']!r}, paths={len(paths)}")
        return result

    def handle_get_skeleton(self, params: dict) -> str:
        from beacon.search.graph import get_skeleton
        from beacon.indexer.scanner import LANG_MAP
        conn = self._conn_lazy()
        files = params["files"]
        detail = params.get("detail", "standard")

        # Claude sometimes sends files as a JSON string rather than a list
        if isinstance(files, str):
            try:
                files = json.loads(files)
            except json.JSONDecodeError:
                files = [files]  # treat as single path

        # Resolve absolute paths relative to workspace if needed
        resolved = []
        for f in files:
            p = Path(f)
            if not p.is_absolute():
                p = self.workspace / f
            resolved.append(str(p.resolve()))

        result = get_skeleton(conn, files, detail=detail, root=str(self.workspace))

        # For any file that came back "(not indexed)", parse it from disk on the fly
        lines = result.split("\n")
        needs_disk: list[tuple[int, str, str]] = []   # (header_line_idx, original_path, abs_path)
        for i, line in enumerate(lines):
            if line.startswith("## ") and i + 1 < len(lines) and lines[i + 1].strip() == "(not indexed)":
                orig = line[3:].strip()
                abs_p = self.workspace / orig
                if abs_p.exists():
                    needs_disk.append((i, orig, str(abs_p)))

        if needs_disk:
            from beacon.indexer.symbols import extract
            for header_idx, orig, abs_p in needs_disk:
                lang = LANG_MAP.get(Path(abs_p).suffix.lower())
                if not lang:
                    continue
                try:
                    file_syms = extract(Path(abs_p), lang, self.workspace)
                except Exception:
                    continue
                new_lines = [f"## {orig}  (from disk, not indexed)"]
                for sym in file_syms.symbols:
                    if detail == "minimal" and not sym.is_exported:
                        continue
                    new_lines.append(f"\n  {sym.kind} `{sym.name}`  (line {sym.start_line})")
                    if sym.signature:
                        new_lines.append(f"    sig:  {sym.signature[:200].strip()}")
                    if sym.docstring and detail != "minimal":
                        doc = sym.docstring[:120] + "…" if len(sym.docstring) > 120 else sym.docstring
                        new_lines.append(f"    doc:  {doc}")
                # Replace the (not indexed) section
                lines[header_idx] = "\n".join(new_lines)
                lines[header_idx + 1] = ""
            result = "\n".join(lines)

        self._auto_observe("get_skeleton", params, f"files={files}")
        return result

    def handle_index_status(self, _params: dict) -> str:
        conn = self._conn_lazy()
        node_count = conn.execute("SELECT COUNT(*) FROM nodes").fetchone()[0]
        edge_count = conn.execute("SELECT COUNT(*) FROM edges").fetchone()[0]
        file_count = conn.execute("SELECT COUNT(*) FROM file_cache").fetchone()[0]
        embed_count = conn.execute("SELECT COUNT(*) FROM node_embeddings").fetchone()[0]
        schema_ver = conn.execute("SELECT value FROM meta WHERE key='schema_version'").fetchone()
        kinds = conn.execute("SELECT kind, COUNT(*) FROM nodes GROUP BY kind ORDER BY 2 DESC").fetchall()
        kind_str = ", ".join(f"{r[0]}:{r[1]}" for r in kinds)

        return (
            f"# beacon index status\n\n"
            f"- Workspace: {self.workspace}\n"
            f"- DB: {self.db_path}\n"
            f"- Schema version: {schema_ver[0] if schema_ver else 'unknown'}\n"
            f"- Files indexed: {file_count}\n"
            f"- Nodes: {node_count}  ({kind_str})\n"
            f"- Edges: {edge_count}\n"
            f"- Embeddings: {embed_count}\n"
            f"- Session: {self.session_id}\n"
        )

    def handle_get_session_context(self, params: dict) -> str:
        conn = self._conn_lazy()
        max_results = int(params.get("max_results", 20))
        include_previous = bool(params.get("include_previous", False))
        types_filter = params.get("types")

        if include_previous:
            session_clause = "1=1"
            binds: list = []
        else:
            session_clause = "o.session_id = ?"
            binds = [self.session_id]

        type_clause = ""
        if types_filter:
            ph = ",".join("?" * len(types_filter))
            type_clause = f"AND o.type IN ({ph})"
            binds.extend(types_filter)

        rows = conn.execute(
            f"""SELECT o.content, o.type, o.created_at, o.stale, o.confidence
                FROM observations o
                WHERE {session_clause} {type_clause}
                ORDER BY o.created_at DESC
                LIMIT ?""",
            (*binds, max_results),
        ).fetchall()

        if not rows:
            return "No observations found for this session."

        lines = [f"# Session context ({len(rows)} observations)\n"]
        for r in rows:
            stale = " [STALE]" if r["stale"] else ""
            lines.append(f"- [{r['type']}]{stale} {r['content']}")
            lines.append(f"  at={r['created_at']}  confidence={r['confidence']:.1f}\n")
        return "\n".join(lines)

    def handle_save_observation(self, params: dict) -> str:
        conn = self._conn_lazy()
        now = datetime.now(timezone.utc).isoformat()
        cur = conn.execute(
            "INSERT INTO observations (session_id, type, content, created_at, source, confidence) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (self.session_id, params.get("type", "insight"),
             params["content"], now, "agent", 1.0),
        )
        obs_id = cur.lastrowid

        # Link to symbols if provided
        linked = params.get("linked_symbols") or []
        for fqn in linked:
            node = conn.execute("SELECT id FROM nodes WHERE fqn=?", (fqn,)).fetchone()
            if node:
                conn.execute(
                    "INSERT OR IGNORE INTO observation_node_links (observation_id, node_id) VALUES (?, ?)",
                    (obs_id, node[0]),
                )
        conn.commit()
        return f"Observation saved (id={obs_id}, type={params.get('type','insight')}, linked={len(linked)} symbols)"

    def handle_search_memory(self, params: dict) -> str:
        conn = self._conn_lazy()
        query = params["query"]
        max_results = int(params.get("max_results", 10))
        include_stale = bool(params.get("include_stale", True))
        time_days = params.get("time_range_days")
        session_filter = params.get("session_filter")

        stale_clause = "" if include_stale else "AND o.stale = 0"
        session_clause = f"AND o.session_id = '{session_filter}'" if session_filter else ""
        time_clause = ""
        binds: list = [query]
        if time_days:
            time_clause = "AND o.created_at >= datetime('now', ?)"
            binds.append(f"-{int(time_days)} days")
        binds.append(max_results * 2)

        from beacon.search.query import _fts5_query
        fts_query = _fts5_query(query)
        binds[0] = fts_query
        try:
            rows = conn.execute(
                f"""SELECT o.content, o.type, o.created_at, o.stale, o.confidence,
                           -bm25(observations_fts) AS score
                    FROM observations o
                    JOIN observations_fts ON observations_fts.rowid = o.id
                    WHERE observations_fts MATCH ? {stale_clause} {session_clause} {time_clause}
                    ORDER BY o.stale ASC, score DESC
                    LIMIT ?""",
                binds,
            ).fetchall()
        except Exception:
            rows = []

        if not rows:
            return f"No memory found matching {query!r}"

        lines = [f"# Memory search: {query!r}\n"]
        for r in rows[:max_results]:
            stale = " [STALE — code changed]" if r["stale"] else ""
            lines.append(f"- [{r['type']}]{stale} {r['content']}")
            lines.append(f"  score={r['score']:.3f}  at={r['created_at']}\n")
        return "\n".join(lines)

    def handle_submit_lsp_edges(self, params: dict) -> str:
        conn = self._conn_lazy()
        edges = params.get("edges", [])
        inserted = 0
        merged = 0
        for e in edges:
            # Store in lsp_edges for reference
            conn.execute(
                "INSERT OR IGNORE INTO lsp_edges (source_fqn, target_fqn, edge_type) VALUES (?, ?, ?)",
                (e["source_fqn"], e["target_fqn"], e.get("edge_type", "CALLS")),
            )
            inserted += 1
            # Also merge into main edges table so search/impact/flow can use them
            src = conn.execute("SELECT id FROM nodes WHERE fqn=?", (e["source_fqn"],)).fetchone()
            tgt = conn.execute("SELECT id FROM nodes WHERE fqn=?", (e["target_fqn"],)).fetchone()
            if src and tgt:
                conn.execute(
                    "INSERT OR IGNORE INTO edges (source_id, target_id, type, confidence) VALUES (?, ?, ?, ?)",
                    (src[0], tgt[0], e.get("edge_type", "CALLS"), 1.0),  # LSP = highest confidence
                )
                merged += 1
        conn.commit()
        return f"Submitted {inserted} LSP edge(s), {merged} merged into search graph"

    def handle_resolve_symbols(self, params: dict) -> str:
        self._maybe_reindex()
        conn = self._conn_lazy()
        names = params.get("names", [])
        file_hint = params.get("file_hint", "")
        kind_filter = params.get("kind", "")

        if not names:
            return "No names provided."

        lines = ["# Symbol resolution\n"]
        for name in names:
            # Exact name match
            query = "SELECT fqn, file_path, kind, start_line, signature FROM nodes WHERE name=?"
            binds: list = [name]
            if file_hint:
                query += " AND file_path LIKE ?"
                binds.append(f"%{file_hint}%")
            if kind_filter:
                query += " AND kind=?"
                binds.append(kind_filter)
            query += " LIMIT 5"
            rows = conn.execute(query, binds).fetchall()

            if not rows:
                # Partial FQN match
                rows = conn.execute(
                    "SELECT fqn, file_path, kind, start_line, signature FROM nodes "
                    "WHERE fqn LIKE ? LIMIT 5",
                    (f"%{name}%",),
                ).fetchall()

            lines.append(f"## `{name}`")
            if not rows:
                lines.append("  not found in index\n")
            else:
                for r in rows:
                    lines.append(f"  FQN:  {r[0]}")
                    lines.append(f"  File: {r[1]}:{r[3]}  ({r[2]})")
                    if r[4]:
                        lines.append(f"  Sig:  {r[4][:120]}")
                lines.append("")

        lines.append(
            "\nUse these FQNs directly in `run_pipeline` or `get_impact_graph` "
            "for precise graph-traversal based retrieval."
        )
        return "\n".join(lines)

    def handle_workspace_setup(self, params: dict) -> str:
        root = Path(params.get("workspace_root") or self.workspace)
        vexp_dir = root / ".beacon"
        vexp_dir.mkdir(exist_ok=True)
        gitignore = vexp_dir / ".gitignore"
        if not gitignore.exists():
            gitignore.write_text("index.db\nindex.db-shm\nindex.db-wal\n")
        return (
            f"# beacon workspace setup\n\n"
            f"- Workspace: {root}\n"
            f"- Index DB: {root / '.beacon' / 'index.db'}\n"
            f"- .gitignore: created\n\n"
            f"Run indexing:\n"
            f"    beacon index {root}\n\n"
            f"Start MCP server:\n"
            f"    beacon mcp --workspace {root}\n"
        )

    # ── Dispatch ───────────────────────────────────────────────────────────────

    HANDLERS = {
        "run_pipeline":         handle_run_pipeline,
        "get_context_capsule":  handle_get_context_capsule,
        "get_impact_graph":     handle_get_impact_graph,
        "search_logic_flow":    handle_search_logic_flow,
        "get_skeleton":         handle_get_skeleton,
        "index_status":         handle_index_status,
        "get_session_context":  handle_get_session_context,
        "save_observation":     handle_save_observation,
        "search_memory":        handle_search_memory,
        "submit_lsp_edges":     handle_submit_lsp_edges,
        "workspace_setup":      handle_workspace_setup,
        "resolve_symbols":      handle_resolve_symbols,
    }

    def call_tool(self, name: str, params: dict) -> str:
        handler = self.HANDLERS.get(name)
        if handler is None:
            raise ValueError(f"Unknown tool: {name}")
        return handler(self, params)

    # ── JSON-RPC loop ──────────────────────────────────────────────────────────

    def _send(self, msg: dict):
        line = json.dumps(msg, ensure_ascii=False)
        sys.stdout.write(line + "\n")
        sys.stdout.flush()

    def _error(self, req_id, code: int, message: str):
        self._send({"jsonrpc": "2.0", "id": req_id, "error": {"code": code, "message": message}})

    def run(self):
        for raw_line in sys.stdin:
            raw_line = raw_line.strip()
            if not raw_line:
                continue
            try:
                msg = json.loads(raw_line)
            except json.JSONDecodeError as e:
                self._send({"jsonrpc": "2.0", "id": None,
                            "error": {"code": -32700, "message": f"Parse error: {e}"}})
                continue

            method = msg.get("method", "")
            req_id = msg.get("id")
            params = msg.get("params") or {}

            # Notifications (no id) — just handle silently
            if req_id is None:
                if method == "notifications/initialized":
                    pass  # nothing to do
                continue

            if method == "initialize":
                self._log({
                    "ts": datetime.now(timezone.utc).isoformat(),
                    "event": "session_start",
                    "session": self.session_id[:8],
                    "workspace": str(self.workspace),
                    "db": str(self.db_path),
                    "client": params.get("clientInfo", {}),
                })
                self._send({
                    "jsonrpc": "2.0", "id": req_id,
                    "result": {
                        "protocolVersion": "2024-11-05",
                        "capabilities": {"tools": {}},
                        "serverInfo": {"name": "beacon", "version": "1.0.0"},
                    },
                })

            elif method == "tools/list":
                self._send({"jsonrpc": "2.0", "id": req_id, "result": {"tools": TOOLS}})

            elif method == "tools/call":
                tool_name = params.get("name")
                tool_args = params.get("arguments") or {}
                t0 = time.monotonic()
                error_msg: str | None = None
                result_text: str = ""
                try:
                    result_text = self.call_tool(tool_name, tool_args)
                    self._send({
                        "jsonrpc": "2.0", "id": req_id,
                        "result": {"content": [{"type": "text", "text": result_text}]},
                    })
                except Exception as e:
                    error_msg = str(e)
                    self._error(req_id, -32603, error_msg)
                finally:
                    elapsed_ms = round((time.monotonic() - t0) * 1000)
                    self._log({
                        "ts": datetime.now(timezone.utc).isoformat(),
                        "session": self.session_id[:8],
                        "tool": tool_name,
                        "args": tool_args,
                        "elapsed_ms": elapsed_ms,
                        "result_chars": len(result_text),
                        "result_tokens_approx": len(result_text) // 4,
                        # First 300 chars of result — enough to judge relevance
                        "result_preview": result_text[:300] if result_text else None,
                        "error": error_msg,
                    })

            elif method == "ping":
                self._send({"jsonrpc": "2.0", "id": req_id, "result": {}})

            else:
                self._error(req_id, -32601, f"Method not found: {method}")


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Beacon MCP stdio server")
    parser.add_argument("--workspace", default=os.getcwd(), help="Workspace root (default: cwd)")
    parser.add_argument("--db", default=None, help="Explicit path to index.db")
    args = parser.parse_args()

    server = McpServer(workspace=args.workspace, db_path=args.db)
    server.run()


if __name__ == "__main__":
    main()
