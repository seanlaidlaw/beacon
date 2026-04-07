"""
3-layer hybrid search — mirrors vexp-core's capsule/pipeline search.

Layer 1: FTS5 BM25 keyword search
  Column weights: name=10, fqn=5, signature=2, docstring=1
  (exact values from vexp-core binary strings)

Layer 2: TF-IDF cosine similarity
  Query is vectorised with the same tokeniser, cosine similarity computed
  against all node_embeddings rows in the candidate set.

Layer 3: Graph signals
  - degree centrality (in + out edge count)
  - change coupling score (co_change_edges)
  - churn score (file_lineage)

Final score = w_bm25*bm25 + w_tfidf*cosine + w_graph*graph_signal
"""

import json
import math
import sqlite3
from dataclasses import dataclass, field

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

# ── Score weights ─────────────────────────────────────────────────────────────
W_BM25   = 0.50
W_TFIDF  = 0.35
W_GRAPH  = 0.15

# BM25 FTS5 column weights (positional — must match FTS5 column order)
# nodes_fts columns: name, fqn, docstring, signature
BM25_WEIGHTS = (10.0, 5.0, 1.0, 2.0)


@dataclass
class SearchResult:
    node_id: int
    name: str
    fqn: str
    file_path: str
    kind: str
    start_line: int
    signature: str
    docstring: str
    score: float
    score_breakdown: dict = field(default_factory=dict)
    reason: str = ""


def _fts5_query(query: str) -> str:
    """
    Convert a natural language query to a safe FTS5 OR expression.

    FTS5 is sensitive to special characters:
    - Hyphens are parsed as NOT operators (methylation-expression → NOT expression)
    - Quotes, parens, * have special meaning
    - Purely numeric tokens are dropped by the tokenizer

    Strategy: extract only alphanumeric+underscore tokens of length >= 3,
    then join with OR. Hyphenated terms like "methylation-expression" are
    split into both halves ("methylation", "expression") so neither is lost.
    """
    import re
    # Split on any non-alphanumeric-underscore character to handle hyphens,
    # slashes, dots, etc. — this also naturally splits camelCase won't help
    # but snake_case stays intact (underscores kept)
    tokens = re.findall(r'[A-Za-z][A-Za-z0-9_]*', query)
    # Filter: minimum 3 chars, skip pure stopwords
    _STOP = {'the','and','for','with','from','that','this','are','was',
             'not','but','all','can','its','use','via','per','than','into'}
    tokens = [t for t in tokens if len(t) >= 3 and t.lower() not in _STOP]
    if not tokens:
        return query[:50]   # last resort — pass raw (may error, caught upstream)
    # Deduplicate while preserving order
    seen: set[str] = set()
    unique = []
    for t in tokens:
        if t.lower() not in seen:
            seen.add(t.lower())
            unique.append(t)
    return " OR ".join(unique)


def _bm25_search(conn: sqlite3.Connection, query: str, limit: int = 100) -> list[tuple[int, float]]:
    """Layer 1: FTS5 BM25 search. Returns [(node_id, bm25_score), ...]."""
    weights_str = ", ".join(str(w) for w in BM25_WEIGHTS)
    fts_query = _fts5_query(query)
    sql = f"""
        SELECT n.id,
               -bm25(nodes_fts, {weights_str}) AS score
        FROM nodes n
        JOIN nodes_fts ON nodes_fts.rowid = n.id
        WHERE nodes_fts MATCH ?
        ORDER BY bm25(nodes_fts, {weights_str})
        LIMIT ?
    """
    try:
        rows = conn.execute(sql, (fts_query, limit)).fetchall()
        return [(r[0], float(r[1])) for r in rows]
    except sqlite3.OperationalError:
        # FTS5 query syntax error — fall back to LIKE search
        return _fallback_like_search(conn, query, limit)


def _fallback_like_search(conn: sqlite3.Connection, query: str, limit: int) -> list[tuple[int, float]]:
    pattern = f"%{query}%"
    rows = conn.execute(
        "SELECT id, 1.0 FROM nodes WHERE name LIKE ? OR fqn LIKE ? LIMIT ?",
        (pattern, pattern, limit),
    ).fetchall()
    return [(r[0], float(r[1])) for r in rows]


def _tfidf_scores(
    conn: sqlite3.Connection,
    query: str,
    node_ids: list[int],
) -> dict[int, float]:
    """
    Layer 2: TF-IDF cosine similarity.
    Fetches stored sparse vectors, builds query vector, computes cosine similarity.
    Returns {node_id: cosine_score}.
    """
    if not node_ids:
        return {}

    placeholders = ",".join("?" * len(node_ids))
    rows = conn.execute(
        f"SELECT node_id, vector_json FROM node_embeddings WHERE node_id IN ({placeholders})",
        node_ids,
    ).fetchall()

    if not rows:
        return {}

    # Collect vocabulary from stored vectors
    vocab: set[str] = set()
    node_vecs: dict[int, dict[str, float]] = {}
    for row in rows:
        vec = json.loads(row[1])
        node_vecs[row[0]] = vec
        vocab.update(vec.keys())

    if not vocab:
        return {}

    vocab_list = sorted(vocab)
    vocab_idx = {t: i for i, t in enumerate(vocab_list)}
    n = len(vocab_list)

    # Build query vector using same tokeniser as embedder
    vectorizer = TfidfVectorizer(
        analyzer="word",
        token_pattern=r"(?u)\b[a-zA-Z_]\w{2,}\b",
        vocabulary=vocab_list,
        sublinear_tf=True,
    )
    try:
        q_vec = vectorizer.fit_transform([query]).toarray()[0]
    except ValueError:
        return {}

    q_norm = np.linalg.norm(q_vec)
    if q_norm == 0:
        return {}
    q_unit = q_vec / q_norm

    scores: dict[int, float] = {}
    for nid, vec_dict in node_vecs.items():
        # Build dense vector for this node
        d_vec = np.zeros(n)
        for term, weight in vec_dict.items():
            idx = vocab_idx.get(term)
            if idx is not None:
                d_vec[idx] = weight
        d_norm = np.linalg.norm(d_vec)
        if d_norm == 0:
            scores[nid] = 0.0
        else:
            scores[nid] = float(np.dot(q_unit, d_vec / d_norm))

    return scores


def _graph_scores(conn: sqlite3.Connection, node_ids: list[int]) -> dict[int, float]:
    """
    Layer 3: Graph signals for a set of node IDs.
    Combines:
      - normalised degree centrality (in + out)
      - max co-change coupling score for the file
      - churn score from file_lineage
    Returns {node_id: graph_score} in [0, 1].
    """
    if not node_ids:
        return {}

    placeholders = ",".join("?" * len(node_ids))

    # Degree: in + out
    degree: dict[int, int] = {nid: 0 for nid in node_ids}
    for row in conn.execute(
        f"SELECT source_id, COUNT(*) FROM edges WHERE source_id IN ({placeholders}) GROUP BY source_id",
        node_ids,
    ).fetchall():
        degree[row[0]] = degree.get(row[0], 0) + row[1]
    for row in conn.execute(
        f"SELECT target_id, COUNT(*) FROM edges WHERE target_id IN ({placeholders}) GROUP BY target_id",
        node_ids,
    ).fetchall():
        degree[row[0]] = degree.get(row[0], 0) + row[1]

    max_degree = max(degree.values(), default=1) or 1

    # File-level signals per node
    node_files = {
        r[0]: r[1]
        for r in conn.execute(
            f"SELECT id, file_path FROM nodes WHERE id IN ({placeholders})",
            node_ids,
        ).fetchall()
    }

    # Churn scores
    unique_files = list(set(node_files.values()))
    churn: dict[str, float] = {}
    if unique_files:
        fp_ph = ",".join("?" * len(unique_files))
        for row in conn.execute(
            f"SELECT file_path, churn_score FROM file_lineage WHERE file_path IN ({fp_ph})",
            unique_files,
        ).fetchall():
            churn[row[0]] = float(row[1])

    # Co-change coupling: max coupling score for each file
    couple: dict[str, float] = {}
    if unique_files:
        for fp in unique_files:
            row = conn.execute(
                """SELECT MAX(coupling_score) FROM co_change_edges
                   WHERE file_a=? OR file_b=?""",
                (fp, fp),
            ).fetchone()
            couple[fp] = float(row[0] or 0.0)

    scores: dict[int, float] = {}
    for nid in node_ids:
        fp = node_files.get(nid, "")
        deg_score = degree.get(nid, 0) / max_degree
        churn_score = churn.get(fp, 0.0)
        couple_score = couple.get(fp, 0.0)
        # Weighted blend: degree most important, coupling and churn secondary
        scores[nid] = 0.6 * deg_score + 0.25 * couple_score + 0.15 * churn_score

    return scores


def search(
    conn: sqlite3.Connection,
    query: str,
    limit: int = 20,
    kind_filter: str | None = None,
) -> list[SearchResult]:
    """
    Run 3-layer hybrid search and return ranked results.

    Parameters
    ----------
    conn        : open database connection
    query       : natural language or symbol query
    limit       : max results to return
    kind_filter : optional filter by node kind (function, class, method, ...)
    """
    # Layer 1: BM25
    bm25_hits = _bm25_search(conn, query, limit=limit * 5)

    # Expand with a second pass on exact name match (catches symbols not in FTS top-N)
    name_hits = conn.execute(
        "SELECT id, 5.0 FROM nodes WHERE name=? LIMIT 10", (query,)
    ).fetchall()
    extra_ids = {r[0] for r in name_hits} - {r[0] for r in bm25_hits}
    bm25_dict = dict(bm25_hits)
    for nid, score in name_hits:
        if nid not in bm25_dict:
            bm25_dict[nid] = score

    all_ids = list(bm25_dict.keys())
    if not all_ids:
        return []

    # Layer 2: TF-IDF
    tfidf_dict = _tfidf_scores(conn, query, all_ids)

    # Layer 3: Graph
    graph_dict = _graph_scores(conn, all_ids)

    # Normalise BM25 scores to [0, 1]
    max_bm25 = max(bm25_dict.values(), default=1.0) or 1.0

    # Fuse
    fused: list[tuple[int, float, dict]] = []
    for nid in all_ids:
        b = bm25_dict.get(nid, 0.0) / max_bm25
        t = tfidf_dict.get(nid, 0.0)
        g = graph_dict.get(nid, 0.0)
        total = W_BM25 * b + W_TFIDF * t + W_GRAPH * g
        fused.append((nid, total, {"bm25": round(b, 3), "tfidf": round(t, 3), "graph": round(g, 3)}))

    fused.sort(key=lambda x: x[1], reverse=True)

    # Fetch node details
    top_ids = [x[0] for x in fused[:limit * 2]]
    placeholders = ",".join("?" * len(top_ids))
    node_rows = {
        r["id"]: r
        for r in conn.execute(
            f"SELECT id, name, fqn, file_path, kind, start_line, signature, docstring "
            f"FROM nodes WHERE id IN ({placeholders})",
            top_ids,
        ).fetchall()
    }

    results: list[SearchResult] = []
    for nid, score, breakdown in fused:
        row = node_rows.get(nid)
        if row is None:
            continue
        if kind_filter and row["kind"] != kind_filter:
            continue

        reason = _make_reason(breakdown)
        results.append(SearchResult(
            node_id=nid,
            name=row["name"],
            fqn=row["fqn"],
            file_path=row["file_path"],
            kind=row["kind"],
            start_line=row["start_line"] or 0,
            signature=row["signature"] or "",
            docstring=(row["docstring"] or "")[:256],
            score=round(score, 4),
            score_breakdown=breakdown,
            reason=reason,
        ))
        if len(results) >= limit:
            break

    return results


def _make_reason(breakdown: dict) -> str:
    parts = []
    if breakdown["bm25"] > 0.5:
        parts.append("high text relevance")
    elif breakdown["bm25"] > 0.1:
        parts.append("keyword match")
    if breakdown["tfidf"] > 0.3:
        parts.append("semantic similarity")
    if breakdown["graph"] > 0.3:
        parts.append("highly connected")
    return " + ".join(parts) if parts else "weak match"
