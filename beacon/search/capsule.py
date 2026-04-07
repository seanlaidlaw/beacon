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
    role: str = "seed"          # seed | caller | callee | co_change
    token_estimate: int = 0


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


def _node_to_capsule(r: SearchResult, role: str) -> CapsuleNode:
    text = f"{r.kind} {r.fqn}\n{r.signature}\n{r.docstring}"
    text = shorten(text, width=MAX_NODE_TOKENS * CHARS_PER_TOKEN, placeholder="…")
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
        token_estimate=_approx_tokens(text),
    )


def _expand_neighbors(
    conn: sqlite3.Connection,
    seed_ids: list[int],
    depth: int = 2,
) -> dict[int, str]:
    """
    BFS expansion from seed node IDs.
    Returns {node_id: role} for callers and callees within *depth* hops.
    """
    visited: dict[int, str] = {}
    frontier = list(seed_ids)
    prev_frontier: set[int] = set(seed_ids)

    for _ in range(depth):
        if not frontier:
            break
        ph = ",".join("?" * len(frontier))
        # Callees (source → target)
        for row in conn.execute(
            f"SELECT DISTINCT target_id FROM edges WHERE source_id IN ({ph}) AND type='CALLS'",
            frontier,
        ).fetchall():
            nid = row[0]
            if nid not in visited and nid not in seed_ids:
                visited[nid] = "callee"

        # Callers (target ← source)
        for row in conn.execute(
            f"SELECT DISTINCT source_id FROM edges WHERE target_id IN ({ph}) AND type='CALLS'",
            frontier,
        ).fetchall():
            nid = row[0]
            if nid not in visited and nid not in seed_ids:
                visited[nid] = "caller"

        # Advance to only the newly discovered nodes (not all visited)
        frontier = [nid for nid in visited if nid not in seed_ids and nid not in prev_frontier]
        prev_frontier = set(visited.keys())

    return visited


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
            nodes = conn.execute(
                """SELECT id, name, fqn, file_path, kind, start_line, signature, docstring
                   FROM nodes WHERE file_path=? LIMIT 5""",
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
                    score=float(score),
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
    pivot_depth: int = 2,
    include_observations: bool = True,
    exclude_fqns: set[str] | None = None,
    anchor_fqns: list[str] | None = None,
) -> Capsule:
    """
    Build a context capsule for *query* within *max_tokens* budget.

    Steps:
      1. Hybrid search → seed nodes
      2. BFS graph expansion → callers/callees
      3. Co-change expansion
      4. Linked observation retrieval
      5. Budget trim (highest-scoring nodes first)
    """
    cap = Capsule(query=query, token_budget=max_tokens)
    budget = max_tokens
    all_nodes: list[CapsuleNode] = []
    exclude = exclude_fqns or set()

    # ── Step 1: seed nodes ────────────────────────────────────────────────
    seed_results = search(conn, query, limit=15, anchor_fqns=anchor_fqns)
    seed_ids = [r.node_id for r in seed_results]
    for r in seed_results:
        if r.fqn in exclude:
            continue
        cn = _node_to_capsule(r, "seed")
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
                score=graph_scores.get(row["id"], 0.0),
                reason=role,
            )
            cn = _node_to_capsule(sr, role)
            all_nodes.append(cn)

    # ── Step 3: co-change expansion ───────────────────────────────────────
    seed_files = list({r.file_path for r in seed_results})
    # Pass the actual remaining budget (after seeds + neighbors already tallied)
    seed_tokens = sum(min(n.token_estimate, MAX_NODE_TOKENS) for n in all_nodes)
    co_nodes = _co_change_nodes(conn, seed_files, max(0, budget - seed_tokens))
    all_nodes.extend(cn for cn in co_nodes if cn.fqn not in exclude)

    # ── Step 4: deduplicate and sort by score ─────────────────────────────
    seen_fqns: set[str] = set()
    unique_nodes: list[CapsuleNode] = []
    for cn in sorted(all_nodes, key=lambda x: x.score, reverse=True):
        if cn.fqn not in seen_fqns:
            seen_fqns.add(cn.fqn)
            unique_nodes.append(cn)

    # ── Step 5: budget trim ───────────────────────────────────────────────
    used = 0
    for cn in unique_nodes:
        tok = min(cn.token_estimate, MAX_NODE_TOKENS)
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
    roles = ["seed", "caller", "callee", "co_change"]
    role_labels = {
        "seed": "## Seed — direct matches",
        "caller": "## Callers",
        "callee": "## Callees",
        "co_change": "## Co-changing context",
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

    if cap.observations:
        lines.append("\n## Memory observations")
        for obs in cap.observations:
            stale_tag = " [STALE]" if obs.stale else ""
            lines.append(f"\n- {obs.content[:300]}{stale_tag}")
            lines.append(f"  (confidence={obs.confidence:.1f}, at={obs.created_at})")

    return "\n".join(lines)
