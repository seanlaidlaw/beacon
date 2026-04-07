"""
Graph operations: impact graph, logic flow (path search), skeleton.

impact_graph  — BFS outward from a symbol: all callers/importers up to depth N
search_flow   — find execution paths between two FQNs through the call graph
get_skeleton  — token-efficient file summary (signatures only, no bodies)
"""

import sqlite3
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path


# ── Impact graph ──────────────────────────────────────────────────────────────

@dataclass
class ImpactNode:
    fqn: str
    file_path: str
    kind: str
    depth: int
    edge_type: str   # CALLS, IMPORTS, CONTAINS


@dataclass
class ImpactGraph:
    root_fqn: str
    nodes: list[ImpactNode] = field(default_factory=list)
    # edges as (source_fqn, target_fqn, type)
    edges: list[tuple[str, str, str]] = field(default_factory=list)


def get_impact_graph(
    conn: sqlite3.Connection,
    symbol_fqn: str,
    depth: int = 5,
    cross_repo: bool = False,
) -> ImpactGraph:
    """
    All code that depends on *symbol_fqn* — callers, importers, transitively.
    BFS traversal walking edges in reverse (target → source).
    """
    graph = ImpactGraph(root_fqn=symbol_fqn)
    root = conn.execute("SELECT id, fqn FROM nodes WHERE fqn=?", (symbol_fqn,)).fetchone()
    if root is None:
        # Try partial match on name
        name = symbol_fqn.split("::")[-1]
        root = conn.execute("SELECT id, fqn FROM nodes WHERE name=? LIMIT 1", (name,)).fetchone()
    if root is None:
        return graph

    visited: dict[int, tuple[str, int]] = {root["id"]: (root["fqn"], 0)}  # id → (fqn, depth)
    frontier: list[int] = [root["id"]]

    for current_depth in range(depth):
        if not frontier:
            break
        ph = ",".join("?" * len(frontier))

        # Batch: all callers of every node in the current frontier
        caller_rows = conn.execute(
            f"""SELECT n.id, n.fqn, n.file_path, n.kind, e.type, e.target_id
                FROM edges e
                JOIN nodes n ON n.id = e.source_id
                WHERE e.target_id IN ({ph})""",
            frontier,
        ).fetchall()

        # Also include cross-repo edges if requested
        if cross_repo:
            xr_rows = conn.execute(
                f"""SELECT 0 AS id, cr.source_fqn AS fqn, cr.source_repo AS file_path,
                           'function' AS kind, cr.type AS type, 0 AS target_id
                    FROM cross_repo_edges cr
                    WHERE cr.target_fqn IN (
                        SELECT fqn FROM nodes WHERE id IN ({ph})
                    )""",
                frontier,
            ).fetchall()
            caller_rows = list(caller_rows) + list(xr_rows)

        next_frontier: list[int] = []
        target_fqns = {nid: visited[nid][0] for nid in frontier}

        for row in caller_rows:
            caller_id = row[0]
            caller_fqn = row[1]
            target_fqn = target_fqns.get(row[5], root["fqn"])
            if caller_id in visited:
                continue
            visited[caller_id] = (caller_fqn, current_depth + 1)
            graph.nodes.append(ImpactNode(
                fqn=caller_fqn,
                file_path=row[2],
                kind=row[3],
                depth=current_depth + 1,
                edge_type=row[4],
            ))
            graph.edges.append((caller_fqn, target_fqn, row[4]))
            next_frontier.append(caller_id)

        frontier = next_frontier

    return graph


def format_impact_tree(graph: ImpactGraph) -> str:
    if not graph.nodes:
        return f"No dependents found for `{graph.root_fqn}`"

    lines = [f"Impact graph for `{graph.root_fqn}`", ""]
    by_depth: dict[int, list[ImpactNode]] = {}
    for n in graph.nodes:
        by_depth.setdefault(n.depth, []).append(n)

    def _tree(fqn: str, d: int, prefix: str = ""):
        children = [n for n in graph.nodes if n.depth == d and
                    any(e[0] == n.fqn and e[1] == fqn for e in graph.edges)]
        for i, node in enumerate(children):
            connector = "└─ " if i == len(children) - 1 else "├─ "
            lines.append(f"{prefix}{connector}[{node.edge_type}] {node.kind} `{node.fqn}`")
            child_prefix = prefix + ("   " if i == len(children) - 1 else "│  ")
            _tree(node.fqn, d + 1, child_prefix)

    _tree(graph.root_fqn, 1)
    lines.append(f"\n{len(graph.nodes)} dependent(s) found.")
    return "\n".join(lines)


def format_impact_list(graph: ImpactGraph) -> str:
    if not graph.nodes:
        return f"No dependents found for `{graph.root_fqn}`"
    lines = [f"# Dependents of `{graph.root_fqn}`\n"]
    for n in sorted(graph.nodes, key=lambda x: x.depth):
        lines.append(f"- depth={n.depth}  [{n.edge_type}]  {n.kind} `{n.fqn}`  ({n.file_path})")
    return "\n".join(lines)


def format_impact_mermaid(graph: ImpactGraph) -> str:
    lines = ["```mermaid", "graph TD"]
    def safe(s: str) -> str:
        return s.replace("::", "__").replace(".", "_").replace("/", "_").replace("-", "_")

    lines.append(f'    ROOT["{graph.root_fqn}"]')
    for src, tgt, etype in graph.edges:
        lines.append(f'    {safe(src)}["{src}"] -->|{etype}| {safe(tgt)}["{tgt}"]')
    lines.append("```")
    return "\n".join(lines)


# ── Logic flow (path search) ──────────────────────────────────────────────────

def search_logic_flow(
    conn: sqlite3.Connection,
    start_fqn: str,
    end_fqn: str,
    max_paths: int = 3,
) -> list[list[str]]:
    """
    Find execution paths from *start_fqn* to *end_fqn* through CALLS edges.
    Returns up to *max_paths* paths, each as a list of FQNs.
    Uses iterative DFS with path length limit.
    """
    start = conn.execute("SELECT id FROM nodes WHERE fqn=?", (start_fqn,)).fetchone()
    end   = conn.execute("SELECT id FROM nodes WHERE fqn=?", (end_fqn,)).fetchone()

    if not start or not end:
        return []

    start_id = start[0]
    end_id   = end[0]
    max_depth = 10
    paths: list[list[str]] = []

    # Build a forward adjacency dict (source_id → [target_id, ...])
    # Only load nodes reachable from start within max_depth — avoid full scan
    reachable: set[int] = set()
    frontier = {start_id}
    for _ in range(max_depth):
        if not frontier:
            break
        ph = ",".join("?" * len(frontier))
        nexts = conn.execute(
            f"SELECT DISTINCT target_id FROM edges WHERE source_id IN ({ph}) AND type='CALLS'",
            list(frontier),
        ).fetchall()
        new_frontier = {r[0] for r in nexts} - reachable - frontier
        reachable |= frontier
        frontier = new_frontier
    reachable |= frontier

    if end_id not in reachable:
        return []

    # Load adjacency for reachable subgraph
    ph = ",".join("?" * len(reachable))
    adj_rows = conn.execute(
        f"SELECT source_id, target_id FROM edges WHERE source_id IN ({ph}) AND type='CALLS'",
        list(reachable),
    ).fetchall()
    adj: dict[int, list[int]] = {}
    for row in adj_rows:
        adj.setdefault(row[0], []).append(row[1])

    # id → fqn lookup
    ph = ",".join("?" * len(reachable))
    id_fqn = {
        r[0]: r[1]
        for r in conn.execute(
            f"SELECT id, fqn FROM nodes WHERE id IN ({ph})", list(reachable)
        ).fetchall()
    }

    # Iterative DFS
    stack: list[tuple[int, list[int]]] = [(start_id, [start_id])]
    while stack and len(paths) < max_paths:
        node, path = stack.pop()
        if node == end_id:
            paths.append([id_fqn.get(nid, str(nid)) for nid in path])
            continue
        if len(path) >= max_depth:
            continue
        path_set = set(path)
        for neighbor in adj.get(node, []):
            if neighbor not in path_set:
                stack.append((neighbor, path + [neighbor]))

    return paths


def format_flow(paths: list[list[str]], start: str, end: str) -> str:
    if not paths:
        return f"No execution path found between `{start}` and `{end}`"
    lines = [f"# Execution paths: `{start}` → `{end}`\n"]
    for i, path in enumerate(paths, 1):
        lines.append(f"## Path {i}")
        for j, fqn in enumerate(path):
            prefix = "  " * j
            arrow = "→ " if j > 0 else "  "
            lines.append(f"{prefix}{arrow}`{fqn}`")
        lines.append("")
    return "\n".join(lines)


# ── Skeleton ──────────────────────────────────────────────────────────────────

def get_skeleton(
    conn: sqlite3.Connection,
    file_paths: list[str],
    detail: str = "standard",
    root: str | None = None,
) -> str:
    """
    Token-efficient file summary at three detail levels:
      minimal  — exported symbols, names + signatures only
      standard — all symbols + signatures + first 120 chars of docstring
      detailed — all symbols + full signature + full docstring
    """
    lines = []
    for fp in file_paths:
        # Normalise: try as-is, then strip leading root
        nodes = conn.execute(
            "SELECT name, fqn, kind, start_line, signature, docstring, is_exported "
            "FROM nodes WHERE file_path=? ORDER BY start_line",
            (fp,),
        ).fetchall()
        if not nodes and root:
            # Try relative path
            try:
                rel = str(Path(fp).relative_to(Path(root)))
                nodes = conn.execute(
                    "SELECT name, fqn, kind, start_line, signature, docstring, is_exported "
                    "FROM nodes WHERE file_path=? ORDER BY start_line",
                    (rel,),
                ).fetchall()
            except ValueError:
                pass

        lines.append(f"## {fp}")
        if not nodes:
            lines.append("  (not indexed)")
            continue

        for n in nodes:
            if detail == "minimal" and not n["is_exported"]:
                continue
            sig = (n["signature"] or "").strip().replace("\n", " ")
            doc = (n["docstring"] or "").strip()
            if detail == "standard":
                doc = doc[:120] + "…" if len(doc) > 120 else doc
            elif detail == "minimal":
                doc = ""

            lines.append(f"\n  {n['kind']} `{n['name']}`  (line {n['start_line']})")
            if sig:
                lines.append(f"    sig:  {sig[:200]}")
            if doc:
                lines.append(f"    doc:  {doc}")
        lines.append("")

    return "\n".join(lines)
