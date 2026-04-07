"""
User configuration for Beacon.

Persisted to ~/.config/beacon/config.yaml as plain key: value pairs.
No external YAML library required.
"""
from pathlib import Path

_CONFIG_PATH = Path.home() / ".config" / "beacon" / "config.yaml"

# ── Model registry ─────────────────────────────────────────────────────────────

MODELS: list[dict] = [
    {
        "id": "jinaai/jina-embeddings-v2-base-code",
        "short": "jina-embeddings-v2-base-code",
        "tag": "Fast",
        "desc": "137M params · 768-dim · recommended for most codebases",
        "dim": 768,
    },
    {
        "id": "jinaai/jina-code-embeddings-1.5b",
        "short": "jina-code-embeddings-1.5b",
        "tag": "Best",
        "desc": "1.5B params · 1536-dim · highest quality (needs ~6 GB RAM/VRAM)",
        "dim": 1536,
    },
]

DEFAULT_MODEL_ID: str = MODELS[0]["id"]


# ── Low-level I/O (no pyyaml dependency) ──────────────────────────────────────

def _read(path: Path) -> dict:
    result: dict = {}
    for line in path.read_text(errors="replace").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if ":" in line:
            k, _, v = line.partition(":")
            result[k.strip()] = v.strip()
    return result


def _write(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f"{k}: {v}\n" for k, v in data.items()))


# ── Public API ─────────────────────────────────────────────────────────────────

def exists() -> bool:
    return _CONFIG_PATH.exists()


def load() -> dict:
    if not _CONFIG_PATH.exists():
        return {}
    try:
        return _read(_CONFIG_PATH)
    except Exception:
        return {}


def save(cfg: dict) -> None:
    try:
        _write(_CONFIG_PATH, cfg)
    except Exception:
        pass


def get_dense_model() -> str:
    return load().get("dense_model", DEFAULT_MODEL_ID)


def set_dense_model(model_id: str) -> None:
    cfg = load()
    cfg["dense_model"] = model_id
    save(cfg)


def config_path() -> Path:
    return _CONFIG_PATH
