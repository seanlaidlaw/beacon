"""
Context capsule — mirrors vexp-core's mcp/capsule.rs.

Given a query and a token budget, produces a ranked, budget-bounded
context package containing:
  - Seed nodes (top search hits)
  - Their callers and callees (graph expansion)
  - Co-changing files (coupling expansion)
  - Linked observations from memory
  - Stale observation warnings

Token counting is approximate: 1 token ≈ 4 chars (GPT/Claude rough estimate).
Per-node budget cap: 1024 tokens (from vexp-core constants).
"""

import sqlite3
from dataclasses import dataclass, field
from textwrap import shorten

from .query import search, SearchResult, _graph_scores

DEFAULT_MAX_TOKENS = 8_000
MAX_NODE_TOKENS    = 1_024
CHARS_PER_TOKEN    = 4      # rough approximation


def _approx_tokens(text: str) -> int:
    return max(1, len(text) // CHARS_PER_TOKEN)


@dataclass
class CapsuleNode:
    fqn: str
    file_path: str
    kind: str
    start_line: int
    signature: str
    docstring: str
    score: float
    reason: str
    role: str = "seed"          # seed | caller | callee | co_change | importer
    token_estimate: int = 0
    body_preview: str = ""      # rendered for seeds so the answer is IN the capsule


@dataclass
class CapsuleObservation:
    content: str
    created_at: str
    stale: bool = False
    confidence: float = 1.0


@dataclass
class Capsule:
    query: str
    nodes: list[CapsuleNode] = field(default_factory=list)
    observations: list[CapsuleObservation] = field(default_factory=list)
    token_estimate: int = 0
    token_budget: int = DEFAULT_MAX_TOKENS


def _node_to_capsule(r: SearchResult, role: str, include_body: bool = False) -> CapsuleNode:
    body = (r.body_preview or "") if include_body else ""
    text = f"{r.kind} {r.fqn}\n{r.signature}\n{r.docstring}"
    text = shorten(text, width=MAX_NODE_TOKENS * CHARS_PER_TOKEN, placeholder="…")
    if body:
        # Body is rendered verbatim (code), so count it as-is up to the node cap
        body = body[: MAX_NODE_TOKENS * CHARS_PER_TOKEN - len(text)]
    return CapsuleNode(
        fqn=r.fqn,
        file_path=r.file_path,
        kind=r.kind,
        start_line=r.start_line,
        signature=r.signature,
        docstring=r.docstring[:512],
        score=r.score,
        reason=r.reason,
        role=role,
        token_estimate=_approx_tokens(text + body),
        body_preview=body,
    )


# Minimum edge confidence to follow during BFS expansion.
# Regex-detected edges have confidence=0.7; tree-sitter AST = 1.0.
# Filtering at 0.75 keeps AST-detected and LSP edges while dropping
# low-confidence regex matches that add noise.
MIN_EDGE_CONFIDENCE = 0.75

# Maximum number of callers/callees to expand from a single node.
MAX_BFS_FANOUT = 10

# Hard cap on total nodes in a capsule, regardless of token budget.
# Prevents hundreds of tiny-signature nodes from accumulating — each node adds
# ~20 tokens of rendering overhead (headers, labels) not counted in token_estimate.
MAX_CAPSULE_NODES = 50

# Per-node rendering overhead not tracked by token_estimate:
# "### kind `fqn`", "  File: …", "  Score: …" etc. ≈ 80 chars = 20 tokens.
RENDER_OVERHEAD_PER_NODE = 20

# Score scaling for expansion roles so all capsule nodes share one scale.
# Seeds carry hybrid search scores (~0.2-0.8); expansion context must rank
# below comparable seeds, never above them.
W_NEIGHBOR  = 0.5    # × graph score (0-1) → 0-0.5
W_CO_CHANGE = 0.4    # × coupling score (0-1) → 0-0.4
IMPORTER_SCORE = 0.3 # flat


def _expand_neighbors(
    conn: sqlite3.Connection,
    seed_ids: list[int],
    depth: int = 2,
) -> dict[int, str]:
    """
    BFS expansion from seed node IDs.
    Returns {node_id: role} for callers and callees within *depth* hops.

    Edges with confidence < MIN_EDGE_CONFIDENCE are skipped (P3).
    Each node contributes at most MAX_BFS_FANOUT callers and MAX_BFS_FANOUT
    callees (P4) — high-degree hubs are capped, not dropped entirely, so the
    most central symbols still get graph context.
    """
    visited: dict[int, str] = {}
    frontier = list(seed_ids)
    prev_frontier: set[int] = set(seed_ids)

    def _capped_neighbors(from_col: str, to_col: str, role: str) -> None:
        ph = ",".join("?" * len(frontier))
        rows = conn.execute(
            f"""SELECT {from_col}, {to_col}
                FROM edges
                WHERE {from_col} IN ({ph})
                  AND type='CALLS'
                  AND confidence >= {MIN_EDGE_CONFIDENCE}
                ORDER BY id""",
            frontier,
        ).fetchall()
        per_node: dict[int, int] = {}
        for src, dst in rows:
            if per_node.get(src, 0) >= MAX_BFS_FANOUT:
                continue
            per_node[src] = per_node.get(src, 0) + 1
            if dst not in visited and dst not in seed_ids:
                visited[dst] = role

    for _ in range(depth):
        if not frontier:
            break
        # Callees: frontier node is the edge source
        _capped_neighbors("source_id", "target_id", "callee")
        # Callers: frontier node is the edge target
        _capped_neighbors("target_id", "source_id", "caller")

        # Advance to only the newly discovered nodes (not all visited)
        frontier = [nid for nid in visited if nid not in seed_ids and nid not in prev_frontier]
        prev_frontier = set(visited.keys())

    return visited


def _file_path_to_module(file_path: str) -> list[str]:
    """Convert a relative file path to candidate Python module dot-paths.

    e.g. "django/db/models/signals.py" → ["django.db.models.signals",
                                           "db.models.signals", "models.signals"]
    Returns multiple candidates (package subsets) to handle partial imports.
    """
    p = file_path.replace("\\", "/")
    if p.endswith(".py"):
        p = p[:-3]
    parts = p.split("/")
    # Remove common package root prefixes (src/, lib/, etc.)
    if parts[0] in ("src", "lib", "pkg"):
        parts = parts[1:]
    # Generate suffix candidates: "a.b.c", "b.c", "c"
    candidates = []
    for i in range(len(parts)):
        candidates.append(".".join(parts[i:]))
    return candidates


def _importer_nodes(
    conn: sqlite3.Connection,
    seed_file_paths: list[str],
    budget_remaining: int,
) -> list[CapsuleNode]:
    """Find files that import any of the seed files and return their top nodes.

    Uses the import_refs table which stores raw import target text regardless
    of whether the target resolved to an indexed node.
    """
    if not seed_file_paths:
        return []

    # Build all module candidates for all seed files
    candidates: list[str] = []
    for fp in seed_file_paths:
        candidates.extend(_file_path_to_module(fp))
    if not candidates:
        return []

    ph = ",".join("?" * len(candidates))
    importer_files: set[str] = set()
    for row in conn.execute(
        f"SELECT DISTINCT source_file FROM import_refs WHERE target_module IN ({ph})",
        candidates,
    ).fetchall():
        importer_files.add(row[0])

    # Remove self-imports (the seed files themselves)
    importer_files -= set(seed_file_paths)
    if not importer_files:
        return []

    result_nodes: list[CapsuleNode] = []
    used = 0
    for fp in sorted(importer_files):
        nodes = conn.execute(
            """SELECT id, name, fqn, file_path, kind, start_line, signature, docstring
               FROM nodes WHERE file_path=? AND is_exported=1 LIMIT 3""",
            (fp,),
        ).fetchall()
        for n in nodes:
            text = f"{n['kind']} {n['fqn']}\n{n['signature'] or ''}"
            tok = _approx_tokens(text)
            if used + tok > budget_remaining:
                return result_nodes
            from .query import SearchResult
            sr = SearchResult(
                node_id=n["id"], name=n["name"], fqn=n["fqn"],
                file_path=n["file_path"], kind=n["kind"],
                start_line=n["start_line"] or 0,
                signature=n["signature"] or "",
                docstring=n["docstring"] or "",
                score=IMPORTER_SCORE,
                reason=f"IMPORTS ({fp})",
            )
            cn = _node_to_capsule(sr, "importer")
            result_nodes.append(cn)
            used += tok

    return result_nodes


def _co_change_nodes(
    conn: sqlite3.Connection,
    file_paths: list[str],
    budget_remaining: int,
) -> list[CapsuleNode]:
    """Fetch nodes from co-changing files, respecting token budget."""
    if not file_paths:
        return []

    result_nodes: list[CapsuleNode] = []
    used = 0

    for fp in file_paths:
        coupled = conn.execute(
            """SELECT file_b AS coupled, coupling_score FROM co_change_edges WHERE file_a=?
               UNION
               SELECT file_a AS coupled, coupling_score FROM co_change_edges WHERE file_b=?
               ORDER BY coupling_score DESC LIMIT 3""",
            (fp, fp),
        ).fetchall()

        for row in coupled:
            coupled_file, score = row[0], row[1]
            # Prefer exported, non-test symbols — these are the file's API,
            # not arbitrary rows. Score is scaled below seed range (W_CO_CHANGE)
            # so co-change context never outranks direct matches.
            nodes = conn.execute(
                """SELECT id, name, fqn, file_path, kind, start_line, signature, docstring
                   FROM nodes WHERE file_path=?
                   ORDER BY is_test ASC, is_exported DESC, start_line ASC LIMIT 5""",
                (coupled_file,),
            ).fetchall()
            for n in nodes:
                text = f"{n['kind']} {n['fqn']}\n{n['signature'] or ''}"
                tok = _approx_tokens(text)
                if used + tok > budget_remaining:
                    return result_nodes
                from .query import SearchResult
                sr = SearchResult(
                    node_id=n["id"], name=n["name"], fqn=n["fqn"],
                    file_path=n["file_path"], kind=n["kind"],
                    start_line=n["start_line"] or 0,
                    signature=n["signature"] or "",
                    docstring=n["docstring"] or "",
                    score=W_CO_CHANGE * float(score),
                    reason=f"CO_CHANGES_WITH (score {score:.2f})",
                )
                cn = _node_to_capsule(sr, "co_change")
                result_nodes.append(cn)
                used += tok

    return result_nodes


def _fetch_observations(
    conn: sqlite3.Connection,
    node_ids: list[int],
    limit: int = 5,
) -> list[CapsuleObservation]:
    """Fetch observations linked to the given node IDs, demoting stale ones."""
    if not node_ids:
        return []
    ph = ",".join("?" * len(node_ids))
    rows = conn.execute(
        f"""SELECT DISTINCT o.content, o.created_at, o.stale, o.confidence
            FROM observations o
            JOIN observation_node_links l ON l.observation_id = o.id
            WHERE l.node_id IN ({ph})
            ORDER BY o.stale ASC, o.confidence DESC, o.created_at DESC
            LIMIT ?""",
        (*node_ids, limit),
    ).fetchall()
    return [
        CapsuleObservation(
            content=r["content"],
            created_at=r["created_at"],
            stale=bool(r["stale"]),
            confidence=float(r["confidence"]),
        )
        for r in rows
    ]


def get_capsule(
    conn: sqlite3.Connection,
    query: str,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    pivot_depth: int = 1,
    include_observations: bool = True,
    exclude_fqns: set[str] | None = None,
    anchor_fqns: list[str] | None = None,
    hypothetical_code: str | None = None,
) -> Capsule:
    """
    Build a context capsule for *query* within *max_tokens* budget.

    Steps:
      1. Hybrid search → seed nodes
      2. BFS graph expansion → callers/callees
      3. Co-change expansion
      4. Linked observation retrieval
      5. Budget trim (highest-scoring nodes first)

    hypothetical_code, if provided, is used for the dense (semantic) search
    pass instead of *query*. Write a short code snippet in the target
    language resembling what you're looking for (HyDE technique). BM25 still
    uses *query* so both signals work together.
    """
    cap = Capsule(query=query, token_budget=max_tokens)
    budget = max_tokens
    all_nodes: list[CapsuleNode] = []
    exclude = exclude_fqns or set()

    # ── Step 1: seed nodes ────────────────────────────────────────────────
    # Seeds are NEVER excluded by session dedup (exclude_fqns): a refined
    # follow-up query must return the most relevant symbols again, otherwise
    # the second query is strictly worse than the first.
    # Seeds carry their body preview so the capsule contains the answer,
    # not just a pointer to it.
    seed_results = search(conn, query, limit=8, anchor_fqns=anchor_fqns,
                          dense_query=hypothetical_code)
    seed_ids = [r.node_id for r in seed_results]
    for r in seed_results:
        cn = _node_to_capsule(r, "seed", include_body=True)
        all_nodes.append(cn)

    # ── Step 2: graph expansion ───────────────────────────────────────────
    neighbor_ids = _expand_neighbors(conn, seed_ids, depth=pivot_depth)
    if neighbor_ids:
        ph = ",".join("?" * len(neighbor_ids))
        neighbor_rows = conn.execute(
            f"SELECT id, name, fqn, file_path, kind, start_line, signature, docstring "
            f"FROM nodes WHERE id IN ({ph})",
            list(neighbor_ids.keys()),
        ).fetchall()
        graph_scores = _graph_scores(conn, list(neighbor_ids.keys()))
        for row in neighbor_rows:
            role = neighbor_ids[row["id"]]
            from .query import SearchResult
            sr = SearchResult(
                node_id=row["id"], name=row["name"], fqn=row["fqn"],
                file_path=row["file_path"], kind=row["kind"],
                start_line=row["start_line"] or 0,
                signature=row["signature"] or "",
                docstring=row["docstring"] or "",
                # Graph score is normalised within the neighbor set; scale it
                # below seed range so expansion never outranks direct matches.
                score=W_NEIGHBOR * graph_scores.get(row["id"], 0.0),
                reason=role,
            )
            cn = _node_to_capsule(sr, role)
            all_nodes.append(cn)

    # ── Step 3a: co-change expansion ──────────────────────────────────────
    seed_files = list({r.file_path for r in seed_results})
    # Pass the actual remaining budget (after seeds + neighbors already tallied)
    seed_tokens = sum(min(n.token_estimate, MAX_NODE_TOKENS) for n in all_nodes)
    co_nodes = _co_change_nodes(conn, seed_files, max(0, budget - seed_tokens))
    all_nodes.extend(cn for cn in co_nodes if cn.fqn not in exclude)

    # ── Step 3b: importer expansion (P1) ─────────────────────────────────
    # Find files that import the seed files so "what depends on X?" works.
    # Cap at 20% of total budget so importers don't crowd out direct context.
    importer_budget = budget // 5
    if importer_budget > 0:
        imp_nodes = _importer_nodes(conn, seed_files, importer_budget)
        all_nodes.extend(cn for cn in imp_nodes if cn.fqn not in exclude)

    # ── Step 4: deduplicate and sort by score ─────────────────────────────
    # Session dedup (exclude_fqns) applies to expansion roles only — seeds
    # were intentionally kept in Step 1.
    seen_fqns: set[str] = set()
    deduped: list[CapsuleNode] = []
    for cn in sorted(all_nodes, key=lambda x: x.score, reverse=True):
        if cn.role != "seed" and cn.fqn in exclude:
            continue
        if cn.fqn not in seen_fqns:
            seen_fqns.add(cn.fqn)
            deduped.append(cn)

    # P2: Penalise test nodes and unexported symbols when the query is not
    # test-focused. Seeds are already penalised inside search() (before the
    # seed cut-off), so only expansion nodes are adjusted here.
    query_is_test = any(t in query.lower() for t in ("test", "spec", "fixture", "mock"))
    expansion = [cn for cn in deduped if cn.role != "seed"]
    if not query_is_test and expansion:
        fqns = [cn.fqn for cn in expansion]
        ph2 = ",".join("?" * len(fqns))
        flags = {
            row[0]: (bool(row[1]), bool(row[2]))
            for row in conn.execute(
                f"SELECT fqn, is_test, is_exported FROM nodes WHERE fqn IN ({ph2})", fqns
            ).fetchall()
        }
        for cn in expansion:
            is_test, is_exported = flags.get(cn.fqn, (False, True))
            if is_test:
                cn.score *= 0.3      # strongly demote test symbols
            elif not is_exported:
                cn.score *= 0.85     # mildly demote private symbols

    unique_nodes = sorted(deduped, key=lambda x: x.score, reverse=True)

    # ── Step 5: budget trim ───────────────────────────────────────────────
    used = 0
    for cn in unique_nodes:
        if len(cap.nodes) >= MAX_CAPSULE_NODES:
            break
        tok = min(cn.token_estimate, MAX_NODE_TOKENS) + RENDER_OVERHEAD_PER_NODE
        if used + tok > budget:
            break
        cap.nodes.append(cn)
        used += tok

    # ── Step 6: observations ──────────────────────────────────────────────
    if include_observations and cap.nodes:
        # Batch FQN → id lookup (single query)
        fqns = [cn.fqn for cn in cap.nodes]
        ph = ",".join("?" * len(fqns))
        node_id_list = [
            r[0] for r in conn.execute(
                f"SELECT id FROM nodes WHERE fqn IN ({ph})", fqns
            ).fetchall()
        ]
        cap.observations = _fetch_observations(conn, node_id_list)

    cap.token_estimate = used
    return cap


def render_capsule(cap: Capsule) -> str:
    """Format a capsule as a human/agent-readable string."""
    lines = [
        f"# Context Capsule: {cap.query!r}",
        f"# {len(cap.nodes)} nodes, ~{cap.token_estimate} tokens (budget {cap.token_budget})",
        "",
    ]

    # Group by role
    roles = ["seed", "caller", "callee", "co_change", "importer"]
    role_labels = {
        "seed": "## Seed — direct matches",
        "caller": "## Callers",
        "callee": "## Callees",
        "co_change": "## Co-changing context",
        "importer": "## Importers — files that depend on this module",
    }

    for role in roles:
        nodes = [n for n in cap.nodes if n.role == role]
        if not nodes:
            continue
        lines.append(role_labels[role])
        for n in nodes:
            lines.append(f"\n### {n.kind} `{n.fqn}`")
            lines.append(f"  File: {n.file_path}:{n.start_line}")
            if n.signature:
                lines.append(f"  Signature: {n.signature}")
            if n.docstring:
                lines.append(f"  Docstring: {n.docstring[:200]}")
            lines.append(f"  Score: {n.score:.3f}  ({n.reason})")
            if n.body_preview:
                lines.append("  ```")
                lines.append(n.body_preview.rstrip())
                lines.append("  ```")

    if cap.observations:
        lines.append("\n## Memory observations")
        for obs in cap.observations:
            stale_tag = " [STALE]" if obs.stale else ""
            lines.append(f"\n- {obs.content[:300]}{stale_tag}")
            lines.append(f"  (confidence={obs.confidence:.1f}, at={obs.created_at})")

    return "\n".join(lines)
