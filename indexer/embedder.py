"""
TF-IDF embedder — mirrors vexp-core's indexer/embedder.rs.

Builds a corpus from all indexed node text fields, fits a TfidfVectorizer,
then stores each node's sparse vector as a JSON dict {term: weight} in
node_embeddings.vector_json.

The sparse dict format matches what vexp-core stores, which lets the search
layer do cosine similarity without loading a full dense matrix.
"""

import json
import sqlite3
from datetime import datetime, timezone

from sklearn.feature_extraction.text import TfidfVectorizer


def _node_text(row: sqlite3.Row) -> str:
    """Concatenate text fields in the same order vexp-core uses for TF-IDF."""
    parts = [
        row["name"] or "",
        row["fqn"] or "",
        row["signature"] or "",
        row["docstring"] or "",
    ]
    return " ".join(p for p in parts if p)


def build(conn: sqlite3.Connection) -> None:
    """
    (Re-)build TF-IDF vectors for all nodes in the database.
    Safe to call repeatedly — uses INSERT OR REPLACE.
    """
    rows = conn.execute(
        "SELECT id, name, fqn, signature, docstring FROM nodes"
    ).fetchall()

    if not rows:
        return

    ids = [r["id"] for r in rows]
    texts = [_node_text(r) for r in rows]

    vectorizer = TfidfVectorizer(
        analyzer="word",
        token_pattern=r"(?u)\b[a-zA-Z_]\w{2,}\b",  # min 3-char tokens
        sublinear_tf=True,   # log(1+tf) — standard for code
        max_features=20_000, # vocabulary cap to keep vectors manageable
        min_df=1,
    )
    tfidf_matrix = vectorizer.fit_transform(texts)
    feature_names = vectorizer.get_feature_names_out()

    now = datetime.now(timezone.utc).isoformat()
    batch: list[tuple[int, str, str]] = []

    cx = tfidf_matrix.tocsr()
    for i, node_id in enumerate(ids):
        row_data = cx.getrow(i)
        indices = row_data.indices
        data = row_data.data
        # Store only non-zero entries (sparse)
        vec = {feature_names[j]: float(data[k]) for k, j in enumerate(indices)}
        batch.append((node_id, json.dumps(vec), now))

    conn.executemany(
        "INSERT OR REPLACE INTO node_embeddings (node_id, vector_json, updated_at) VALUES (?, ?, ?)",
        batch,
    )
    conn.commit()
    print(f"TF-IDF embeddings: {len(batch)} vectors stored")


def build_incremental(conn: sqlite3.Connection, node_ids: list[int]) -> None:
    """
    Update vectors for a specific set of nodes (incremental re-index).
    Re-fits the vectorizer over the whole corpus so IDF stays correct,
    but only writes rows for the changed nodes.
    """
    if not node_ids:
        return

    all_rows = conn.execute(
        "SELECT id, name, fqn, signature, docstring FROM nodes"
    ).fetchall()
    if not all_rows:
        return

    all_ids = [r["id"] for r in all_rows]
    all_texts = [_node_text(r) for r in all_rows]
    changed_set = set(node_ids)

    vectorizer = TfidfVectorizer(
        analyzer="word",
        token_pattern=r"(?u)\b[a-zA-Z_]\w{2,}\b",
        sublinear_tf=True,
        max_features=20_000,
        min_df=1,
    )
    tfidf_matrix = vectorizer.fit_transform(all_texts)
    feature_names = vectorizer.get_feature_names_out()

    now = datetime.now(timezone.utc).isoformat()
    batch = []
    cx = tfidf_matrix.tocsr()

    for i, node_id in enumerate(all_ids):
        if node_id not in changed_set:
            continue
        row_data = cx.getrow(i)
        vec = {feature_names[j]: float(row_data.data[k])
               for k, j in enumerate(row_data.indices)}
        batch.append((node_id, json.dumps(vec), now))

    conn.executemany(
        "INSERT OR REPLACE INTO node_embeddings (node_id, vector_json, updated_at) VALUES (?, ?, ?)",
        batch,
    )
    conn.commit()
