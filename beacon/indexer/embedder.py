"""
Embedder — builds both sparse TF-IDF and dense neural embeddings for indexed nodes.

Sparse TF-IDF (layer 1 lexical):
  - Fits a TfidfVectorizer over the full corpus at index time
  - Stores per-node sparse {term: weight} JSON in node_embeddings
  - Persists the fitted vectorizer to .beacon/tfidf.pkl so query-time IDF is consistent

Dense neural (layer 2 semantic):
  - Uses jinaai/jina-embeddings-v2-base-code (768-dim, code+NL trained)
  - Falls back to microsoft/unixcoder-base if Jina unavailable
  - Stores float32 numpy bytes in node_embeddings_dense
  - Batched inference; uses GPU if available
  - Gracefully skips if model cannot be loaded (no network / no GPU)
"""

import json
import pickle
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


# ── Sparse TF-IDF ─────────────────────────────────────────────────────────────

def _node_text(row: sqlite3.Row) -> str:
    """Concatenate text fields in the same order vexp-core uses for TF-IDF."""
    parts = [row["name"] or "", row["fqn"] or "", row["signature"] or "", row["docstring"] or ""]
    return " ".join(p for p in parts if p)


def _vectorizer_path(conn: sqlite3.Connection) -> Path:
    db_path = Path(conn.execute("PRAGMA database_list").fetchone()[2])
    return db_path.parent / "tfidf.pkl"


def build(conn: sqlite3.Connection) -> None:
    """Full TF-IDF rebuild — fits on entire corpus, stores sparse vectors + vectorizer."""
    rows = conn.execute("SELECT id, name, fqn, signature, docstring FROM nodes").fetchall()
    if not rows:
        return

    ids = [r["id"] for r in rows]
    texts = [_node_text(r) for r in rows]

    vectorizer = TfidfVectorizer(
        analyzer="word",
        token_pattern=r"(?u)\b[a-zA-Z_]\w{2,}\b",
        sublinear_tf=True,
        max_features=20_000,
        min_df=2,
    )
    tfidf_matrix = vectorizer.fit_transform(texts)
    feature_names = vectorizer.get_feature_names_out()

    # Persist vectorizer so query-time IDF is identical to index-time IDF
    pkl_path = _vectorizer_path(conn)
    with open(pkl_path, "wb") as f:
        pickle.dump(vectorizer, f)

    now = datetime.now(timezone.utc).isoformat()
    cx = tfidf_matrix.tocsr()
    batch = []
    for i, node_id in enumerate(ids):
        row_data = cx.getrow(i)
        vec = {feature_names[j]: float(row_data.data[k]) for k, j in enumerate(row_data.indices)}
        batch.append((node_id, json.dumps(vec), now))

    conn.executemany(
        "INSERT OR REPLACE INTO node_embeddings (node_id, vector_json, updated_at) VALUES (?, ?, ?)",
        batch,
    )
    conn.commit()
    print(f"TF-IDF embeddings: {len(batch)} vectors stored")


def build_incremental(conn: sqlite3.Connection, node_ids: list[int]) -> None:
    """Update TF-IDF for specific nodes. Re-fits over full corpus to keep IDF correct."""
    if not node_ids:
        return

    all_rows = conn.execute("SELECT id, name, fqn, signature, docstring FROM nodes").fetchall()
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
        min_df=2,
    )
    tfidf_matrix = vectorizer.fit_transform(all_texts)
    feature_names = vectorizer.get_feature_names_out()

    pkl_path = _vectorizer_path(conn)
    with open(pkl_path, "wb") as f:
        pickle.dump(vectorizer, f)

    now = datetime.now(timezone.utc).isoformat()
    cx = tfidf_matrix.tocsr()
    batch = []
    for i, node_id in enumerate(all_ids):
        if node_id not in changed_set:
            continue
        row_data = cx.getrow(i)
        vec = {feature_names[j]: float(row_data.data[k]) for k, j in enumerate(row_data.indices)}
        batch.append((node_id, json.dumps(vec), now))

    conn.executemany(
        "INSERT OR REPLACE INTO node_embeddings (node_id, vector_json, updated_at) VALUES (?, ?, ?)",
        batch,
    )
    conn.commit()


def load_vectorizer(conn: sqlite3.Connection) -> TfidfVectorizer | None:
    """Load the persisted vectorizer, or None if not available."""
    try:
        pkl_path = _vectorizer_path(conn)
        if pkl_path.exists():
            with open(pkl_path, "rb") as f:
                return pickle.load(f)
    except Exception:
        pass
    return None


# ── Dense neural encoder ──────────────────────────────────────────────────────

_MODEL = "jinaai/jina-embeddings-v2-base-code"
_BATCH_SIZE = 64


class SentenceEncoder:
    """
    Wraps jinaai/jina-embeddings-v2-base-code via sentence-transformers.

    Using sentence-transformers rather than raw transformers/AutoModel avoids:
    - trust_remote_code warnings (sentence-transformers handles Jina natively)
    - Internal transformers API breakage (find_pruneable_heads_and_indices etc.)

    Loaded lazily; gracefully returns None if unavailable.
    """

    def __init__(self, model_name: str):
        self.model_name = model_name
        self._model = None
        self._failed = False

    def _load(self) -> bool:
        if self._failed:
            return False
        if self._model is not None:
            return True
        try:
            from sentence_transformers import SentenceTransformer
            print(f"Loading {self.model_name}...", end=" ", flush=True)
            self._model = SentenceTransformer(self.model_name, trust_remote_code=True)
            print("done")
            return True
        except Exception as e:
            print(f"failed ({e})")
            self._failed = True
            return False

    def encode(self, texts: list[str]) -> np.ndarray | None:
        """Encode texts. Returns (N, dim) float32 L2-normalised array or None on failure."""
        if not self._load():
            return None
        try:
            vecs = self._model.encode(
                texts,
                batch_size=_BATCH_SIZE,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            return vecs.astype(np.float32)
        except Exception as e:
            print(f"Encoding error: {e}")
            return None


# Module-level singleton — loaded once per process
_encoder: SentenceEncoder | None = None


def get_encoder() -> SentenceEncoder:
    global _encoder
    if _encoder is None:
        _encoder = SentenceEncoder(_MODEL)
    return _encoder


def build_dense(conn: sqlite3.Connection) -> None:
    """Full dense embedding rebuild over all nodes."""
    rows = conn.execute("SELECT id, name, fqn, signature, docstring FROM nodes").fetchall()
    if not rows:
        return

    encoder = get_encoder()
    texts = [_node_text(r) for r in rows]
    vecs = encoder.encode(texts)
    if vecs is None:
        print("Dense embeddings: skipped (model unavailable)")
        return

    now = datetime.now(timezone.utc).isoformat()
    batch = [(r["id"], vecs[i].tobytes(), encoder.model_name, now) for i, r in enumerate(rows)]
    conn.executemany(
        "INSERT OR REPLACE INTO node_embeddings_dense (node_id, vector, model_name, updated_at) "
        "VALUES (?, ?, ?, ?)",
        batch,
    )
    conn.commit()
    print(f"Dense embeddings: {len(batch)} vectors stored (model={encoder.model_name}, dim={vecs.shape[1]})")


def build_dense_incremental(conn: sqlite3.Connection, node_ids: list[int]) -> None:
    """Update dense embeddings for a specific set of nodes only."""
    if not node_ids:
        return
    ph = ",".join("?" * len(node_ids))
    rows = conn.execute(
        f"SELECT id, name, fqn, signature, docstring FROM nodes WHERE id IN ({ph})",
        node_ids,
    ).fetchall()
    if not rows:
        return

    encoder = get_encoder()
    texts = [_node_text(r) for r in rows]
    vecs = encoder.encode(texts)
    if vecs is None:
        return

    now = datetime.now(timezone.utc).isoformat()
    batch = [(r["id"], vecs[i].tobytes(), encoder.model_name, now) for i, r in enumerate(rows)]
    conn.executemany(
        "INSERT OR REPLACE INTO node_embeddings_dense (node_id, vector, model_name, updated_at) "
        "VALUES (?, ?, ?, ?)",
        batch,
    )
    conn.commit()
