"""
3-layer hybrid search.

Layer 1: FTS5 BM25 keyword search
  Column weights: name=10, fqn=5, signature=2, docstring=1

Layer 2: Dense cosine similarity (primary) OR TF-IDF cosine (fallback)
  Dense: loads stored float32 vectors from node_embeddings_dense,
         encodes query with same model, cosine similarity in numpy.
  TF-IDF fallback: loads pickled vectorizer from .beacon/tfidf.pkl,
         transforms query with corpus IDF, cosine similarity against
         stored sparse vectors. (Previously broken: now uses saved IDF.)

Layer 3: Graph signals
  - degree centrality (in + out edge count)
  - change coupling score (co_change_edges)
  - churn score (file_lineage)

Final score = W_BM25*bm25 + W_DENSE*dense + W_GRAPH*graph
"""

import json
import re
import sqlite3
from dataclasses import dataclass, field

import numpy as np

# ── Score weights ─────────────────────────────────────────────────────────────
W_BM25  = 0.45
W_DENSE = 0.40
W_GRAPH = 0.15

# BM25 FTS5 column weights — positional, must match FTS5 column order:
# nodes_fts columns: name, fqn, docstring, signature, body_preview
BM25_WEIGHTS = (10.0, 5.0, 1.0, 2.0, 0.5)


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


# ── FTS5 query sanitisation ───────────────────────────────────────────────────

def _fts5_query(query: str) -> str:
    """
    Convert a natural language query to a safe FTS5 OR expression.

    FTS5 hazards:
    - Hyphens → NOT operator  (methylation-expression → NOT expression)
    - Quotes, parens, * have special meaning
    - Pure-numeric tokens ignored by tokenizer

    Strategy: extract alphanumeric+underscore tokens >= 3 chars, skip stopwords,
    join with OR. Hyphenated terms split into both halves.
    """
    tokens = re.findall(r'[A-Za-z][A-Za-z0-9_]*', query)
    _STOP = {'the', 'and', 'for', 'with', 'from', 'that', 'this', 'are', 'was',
             'not', 'but', 'all', 'can', 'its', 'use', 'via', 'per', 'than', 'into'}
    tokens = [t for t in tokens if len(t) >= 3 and t.lower() not in _STOP]
    if not tokens:
        return query[:50]
    seen: set[str] = set()
    unique = []
    for t in tokens:
        if t.lower() not in seen:
            seen.add(t.lower())
            unique.append(t)
    return " OR ".join(unique)


# ── Layer 1: BM25 ─────────────────────────────────────────────────────────────

def _bm25_search(conn: sqlite3.Connection, query: str, limit: int = 100) -> list[tuple[int, float]]:
    weights_str = ", ".join(str(w) for w in BM25_WEIGHTS)
    fts_query = _fts5_query(query)
    sql = f"""
        SELECT n.id, -bm25(nodes_fts, {weights_str}) AS score
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
        return _fallback_like_search(conn, query, limit)


def _fallback_like_search(conn: sqlite3.Connection, query: str, limit: int) -> list[tuple[int, float]]:
    """Token-by-token LIKE fallback when FTS5 fails."""
    tokens = re.findall(r'[A-Za-z][A-Za-z0-9_]{2,}', query)[:5]
    if not tokens:
        return []
    hits: dict[int, float] = {}
    for tok in tokens:
        pattern = f"%{tok}%"
        for row in conn.execute(
            "SELECT id FROM nodes WHERE name LIKE ? OR fqn LIKE ?"
            " OR body_preview LIKE ? OR docstring LIKE ? LIMIT ?",
            (pattern, pattern, pattern, pattern, limit),
        ).fetchall():
            hits[row[0]] = hits.get(row[0], 0.0) + 1.0
    return sorted(hits.items(), key=lambda x: x[1], reverse=True)[:limit]


# ── Layer 2a: Dense cosine ────────────────────────────────────────────────────

def _dense_scores(
    conn: sqlite3.Connection,
    query: str,
    node_ids: list[int],
) -> dict[int, float]:
    """
    Cosine similarity using stored dense neural embeddings.
    Loads the encoder singleton (cached per process), encodes the query,
    fetches stored float32 blobs, computes dot product (vectors are L2-normalised).
    Returns {} if dense embeddings not available.
    """
    if not node_ids:
        return {}

    # Check if any dense embeddings exist
    ph = ",".join("?" * len(node_ids))
    rows = conn.execute(
        f"SELECT node_id, vector FROM node_embeddings_dense WHERE node_id IN ({ph})",
        node_ids,
    ).fetchall()
    if not rows:
        return {}

    try:
        from beacon.indexer.embedder import get_encoder
        encoder = get_encoder()
        q_vecs = encoder.encode([query])
        if q_vecs is None:
            return {}
        q_vec = q_vecs[0]  # (dim,)

        scores: dict[int, float] = {}
        for row in rows:
            nid = row[0]
            vec = np.frombuffer(row[1], dtype=np.float32)
            if vec.shape != q_vec.shape:
                continue
            # Both are L2-normalised at index time → dot product = cosine
            scores[nid] = float(np.dot(q_vec, vec))
        return scores
    except Exception:
        return {}


# ── Layer 2b: TF-IDF cosine fallback ─────────────────────────────────────────

def _tfidf_scores(
    conn: sqlite3.Connection,
    query: str,
    node_ids: list[int],
) -> dict[int, float]:
    """
    Cosine similarity using stored sparse TF-IDF vectors.
    Uses the corpus-fitted vectorizer (saved to tfidf.pkl at index time)
    so query-time IDF matches index-time IDF.
    """
    if not node_ids:
        return {}

    from beacon.indexer.embedder import load_vectorizer
    vectorizer = load_vectorizer(conn)
    if vectorizer is None:
        return {}

    ph = ",".join("?" * len(node_ids))
    rows = conn.execute(
        f"SELECT node_id, vector_json FROM node_embeddings WHERE node_id IN ({ph})",
        node_ids,
    ).fetchall()
    if not rows:
        return {}

    try:
        feature_names = vectorizer.get_feature_names_out()
        vocab_idx = {t: i for i, t in enumerate(feature_names)}
        n = len(feature_names)

        q_vec = vectorizer.transform([query]).toarray()[0]
        q_norm = np.linalg.norm(q_vec)
        if q_norm == 0:
            return {}
        q_unit = q_vec / q_norm

        scores: dict[int, float] = {}
        for row in rows:
            vec_dict = json.loads(row[1])
            d_vec = np.zeros(n)
            for term, weight in vec_dict.items():
                idx = vocab_idx.get(term)
                if idx is not None:
                    d_vec[idx] = weight
            d_norm = np.linalg.norm(d_vec)
            scores[row[0]] = float(np.dot(q_unit, d_vec / d_norm)) if d_norm > 0 else 0.0
        return scores
    except Exception:
        return {}


# ── Layer 3: Graph signals ────────────────────────────────────────────────────

def _graph_scores(conn: sqlite3.Connection, node_ids: list[int]) -> dict[int, float]:
    if not node_ids:
        return {}

    ph = ",".join("?" * len(node_ids))

    # In + out degree
    degree: dict[int, int] = {nid: 0 for nid in node_ids}
    for row in conn.execute(
        f"SELECT source_id, COUNT(*) FROM edges WHERE source_id IN ({ph}) GROUP BY source_id",
        node_ids,
    ).fetchall():
        degree[row[0]] = degree.get(row[0], 0) + row[1]
    for row in conn.execute(
        f"SELECT target_id, COUNT(*) FROM edges WHERE target_id IN ({ph}) GROUP BY target_id",
        node_ids,
    ).fetchall():
        degree[row[0]] = degree.get(row[0], 0) + row[1]

    max_degree = max(degree.values(), default=1) or 1

    # File paths per node (one query)
    node_files = {
        r[0]: r[1]
        for r in conn.execute(
            f"SELECT id, file_path FROM nodes WHERE id IN ({ph})", node_ids
        ).fetchall()
    }

    unique_files = list(set(node_files.values()))
    churn: dict[str, float] = {}
    couple: dict[str, float] = {}

    if unique_files:
        fp_ph = ",".join("?" * len(unique_files))
        for row in conn.execute(
            f"SELECT file_path, churn_score FROM file_lineage WHERE file_path IN ({fp_ph})",
            unique_files,
        ).fetchall():
            churn[row[0]] = float(row[1])

        # Batch co-change lookup
        for fp in unique_files:
            row = conn.execute(
                "SELECT MAX(coupling_score) FROM co_change_edges WHERE file_a=? OR file_b=?",
                (fp, fp),
            ).fetchone()
            couple[fp] = float(row[0] or 0.0)

    scores: dict[int, float] = {}
    for nid in node_ids:
        fp = node_files.get(nid, "")
        deg_score = degree.get(nid, 0) / max_degree
        scores[nid] = 0.6 * deg_score + 0.25 * couple.get(fp, 0.0) + 0.15 * churn.get(fp, 0.0)

    return scores


# ── Query expansion ───────────────────────────────────────────────────────────

def expand_query(
    conn: sqlite3.Connection,
    query: str,
) -> tuple[str, list[str]]:
    """
    Resolve identifier-shaped tokens in the query against the symbol table.

    Returns (fts5_query_str, anchor_fqns) where:
    - fts5_query_str is the improved FTS5 OR expression
    - anchor_fqns is a list of FQNs for symbols whose exact name matched,
      to be used as graph traversal seeds in addition to search results.
    """
    # Extract potential symbol names (CamelCase, snake_case, ALLCAPS)
    candidates = re.findall(r'[A-Za-z][A-Za-z0-9_]{2,}', query)

    anchor_fqns: list[str] = []
    extra_tokens: list[str] = []

    for name in candidates:
        # Exact name match in the symbol table
        rows = conn.execute(
            "SELECT fqn FROM nodes WHERE name=? LIMIT 3", (name,)
        ).fetchall()
        if rows:
            for row in rows:
                anchor_fqns.append(row[0])
            # Add the file path tokens as extra FTS5 context
            file_rows = conn.execute(
                "SELECT DISTINCT file_path FROM nodes WHERE name=? LIMIT 1", (name,)
            ).fetchall()
            for fr in file_rows:
                # Extract meaningful parts of the file path
                stem = re.sub(r'\W+', ' ', fr[0]).strip()
                extra_tokens.extend(t for t in stem.split() if len(t) >= 3)

    base_fts = _fts5_query(query)
    if extra_tokens:
        # Deduplicate against base tokens
        base_set = set(base_fts.split(" OR "))
        new_tokens = [t for t in extra_tokens if t not in base_set][:10]
        if new_tokens:
            base_fts = base_fts + " OR " + " OR ".join(new_tokens)

    return base_fts, list(dict.fromkeys(anchor_fqns))  # deduplicated


# ── Main search ───────────────────────────────────────────────────────────────

def search(
    conn: sqlite3.Connection,
    query: str,
    limit: int = 20,
    kind_filter: str | None = None,
    anchor_fqns: list[str] | None = None,
    dense_query: str | None = None,
) -> list[SearchResult]:
    """
    3-layer hybrid search. anchor_fqns (from expand_query) are added as
    additional candidates with a baseline graph score.

    dense_query overrides the text sent to the semantic (dense/TF-IDF) layer.
    Pass a hypothetical code snippet (HyDE) for far better semantic matching on
    natural-language queries — BM25 still uses the original *query*.
    """
    # Layer 1: BM25
    bm25_hits = _bm25_search(conn, query, limit=limit * 5)

    # Exact name match (catches symbols below FTS top-N)
    name_hits = conn.execute(
        "SELECT id, 5.0 FROM nodes WHERE name=? LIMIT 10", (query,)
    ).fetchall()
    bm25_dict = dict(bm25_hits)
    for nid, score in name_hits:
        if nid not in bm25_dict:
            bm25_dict[nid] = score

    # Anchor FQNs from query expansion
    if anchor_fqns:
        for fqn in anchor_fqns:
            row = conn.execute("SELECT id FROM nodes WHERE fqn=?", (fqn,)).fetchone()
            if row and row[0] not in bm25_dict:
                bm25_dict[row[0]] = 3.0  # baseline anchor score

    all_ids = list(bm25_dict.keys())
    if not all_ids:
        return []

    # Layer 2: Dense (preferred) or TF-IDF fallback.
    # Use HyDE snippet if provided — code→code similarity is much stronger
    # than NL→code in the embedding space.
    dense_input = dense_query if dense_query else query
    semantic_dict = _dense_scores(conn, dense_input, all_ids)
    if not semantic_dict:
        semantic_dict = _tfidf_scores(conn, dense_input, all_ids)

    # Layer 3: Graph
    graph_dict = _graph_scores(conn, all_ids)

    # Normalise BM25 to [0, 1]
    max_bm25 = max(bm25_dict.values(), default=1.0) or 1.0

    # Fuse
    fused: list[tuple[int, float, dict]] = []
    for nid in all_ids:
        b = bm25_dict.get(nid, 0.0) / max_bm25
        s = semantic_dict.get(nid, 0.0)
        g = graph_dict.get(nid, 0.0)
        total = W_BM25 * b + W_DENSE * s + W_GRAPH * g
        fused.append((nid, total, {"bm25": round(b, 3), "semantic": round(s, 3), "graph": round(g, 3)}))

    fused.sort(key=lambda x: x[1], reverse=True)

    # Fetch node details in one query
    top_ids = [x[0] for x in fused[:limit * 2]]
    ph = ",".join("?" * len(top_ids))
    node_rows = {
        r["id"]: r
        for r in conn.execute(
            f"SELECT id, name, fqn, file_path, kind, start_line, signature, docstring "
            f"FROM nodes WHERE id IN ({ph})",
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
            reason=_make_reason(breakdown),
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
    if breakdown["semantic"] > 0.5:
        parts.append("strong semantic match")
    elif breakdown["semantic"] > 0.2:
        parts.append("semantic similarity")
    if breakdown["graph"] > 0.3:
        parts.append("highly connected")
    return " + ".join(parts) if parts else "weak match"
