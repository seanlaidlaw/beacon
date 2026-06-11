"""
beacon.benchmark — token-savings benchmark logic.

Measures how many tokens an AI agent would consume to answer each query via:
  A) Beacon path:   get_context_capsule (single structured call, Python API)
  B) Baseline path: grep -rl + read top 3 matching files (what an agent does without Beacon)

Used by the `beacon run-benchmark` CLI command.
"""

import json
import re
import subprocess
import time
from pathlib import Path
from typing import Any, Callable


# ── Query sets ───────────────────────────────────────────────────────────────

# Django query set — use with beacon run-benchmark --root ~/repos/django
QUERIES = [
    {
        "id": 1,
        "query": "Where is database connection pooling configured?",
        "type": "keyword_easy",
        "expected_file_hint": "django/db/backends/base/base.py",
        "grep_strategy": ["connection pool", "persistent", "CONN_MAX_AGE"],
    },
    {
        "id": 2,
        "query": "How does Django prevent CSRF attacks?",
        "type": "semantic_paraphrase",
        "expected_file_hint": "django/middleware/csrf.py",
        "grep_strategy": ["csrf", "CsrfViewMiddleware", "csrftoken"],
    },
    {
        "id": 3,
        "query": "What happens when a migration is applied?",
        "type": "multi_hop_call_chain",
        "expected_file_hint": "django/db/migrations/executor.py",
        "grep_strategy": ["apply_migration", "MigrationExecutor", "migration apply"],
    },
    {
        "id": 4,
        "query": "Find where session cookies are signed and verified",
        "type": "cross_subsystem",
        "expected_file_hint": "django/contrib/sessions/backends/signed_cookies.py",
        "grep_strategy": ["session.*sign", "signing.dumps", "signing.loads", "SessionStore"],
    },
    {
        "id": 5,
        "query": "What calls the ORM SQL compiler?",
        "type": "graph_traversal_callers",
        "expected_file_hint": "django/db/models/sql/compiler.py",
        "grep_strategy": ["SQLCompiler", "as_sql", "execute_sql"],
    },
    {
        "id": 6,
        "query": "How does the template engine parse variables?",
        "type": "semantic_no_obvious_keyword",
        "expected_file_hint": "django/template/base.py",
        "grep_strategy": ["template.*variable", "Variable", "resolve_lookup"],
    },
    {
        "id": 7,
        "query": "What code runs during model .save()?",
        "type": "execution_path",
        "expected_file_hint": "django/db/models/base.py",
        "grep_strategy": ["def save", "pre_save", "post_save", "Model.save"],
    },
    {
        "id": 8,
        "query": "Where is content type detection done?",
        "type": "needle_in_haystack",
        "expected_file_hint": "django/utils/encoding.py",
        "grep_strategy": ["content.type", "mime", "detect.*type", "content_type"],
    },
    {
        "id": 9,
        "query": "What depends on django.db.models.signals?",
        "type": "import_graph_query",
        "expected_file_hint": "any file importing signals",
        "grep_strategy": ["from django.db.models import signals",
                          "from django.db.models.signals import",
                          "import signals"],
    },
    {
        "id": 10,
        "query": "How does Django cache template fragments?",
        "type": "multi_module_feature",
        "expected_file_hint": "django/template/context.py or django/core/cache/",
        "grep_strategy": ["cache.*template", "CacheNode", "cache_page", "fragment_cache"],
    },
]


# Beacon-code query set — used automatically when run against this repo
BEACON_CODE_QUERIES = [
    {
        "id": 1,
        "query": "Where is the FTS5 search query string built?",
        "type": "keyword_easy",
        "expected_file_hint": "beacon/search/query.py",
        "grep_strategy": ["_fts5_query", "FTS5", "fts5"],
    },
    {
        "id": 2,
        "query": "How does beacon limit results to a token budget?",
        "type": "semantic_paraphrase",
        "expected_file_hint": "beacon/search/capsule.py",
        "grep_strategy": ["max_tokens", "_approx_tokens", "token_budget"],
    },
    {
        "id": 3,
        "query": "What happens when a source file changes and needs reindexing?",
        "type": "multi_hop_call_chain",
        "expected_file_hint": "beacon/indexer/indexer.py",
        "grep_strategy": ["check_and_reindex", "_hash_file", "reindex"],
    },
    {
        "id": 4,
        "query": "Where is the neural sentence embedding model loaded?",
        "type": "cross_subsystem",
        "expected_file_hint": "beacon/indexer/embedder.py",
        "grep_strategy": ["SentenceEncoder", "SentenceTransformer", "sentence_transformers"],
    },
    {
        "id": 5,
        "query": "What performs BFS graph expansion to find callers and callees?",
        "type": "graph_traversal_callers",
        "expected_file_hint": "beacon/search/capsule.py",
        "grep_strategy": ["_expand_neighbors", "_capped_neighbors", "CALLS"],
    },
    {
        "id": 6,
        "query": "How does beacon boost exported symbols over test code in ranking?",
        "type": "semantic_no_obvious_keyword",
        "expected_file_hint": "beacon/search/query.py",
        "grep_strategy": ["is_exported", "is_test", "_graph_scores"],
    },
    {
        "id": 7,
        "query": "What runs when the MCP run_pipeline tool is called?",
        "type": "execution_path",
        "expected_file_hint": "beacon/mcp.py",
        "grep_strategy": ["handle_run_pipeline", "run_pipeline", "json_rpc"],
    },
    {
        "id": 8,
        "query": "Where is tree-sitter configured per programming language?",
        "type": "needle_in_haystack",
        "expected_file_hint": "beacon/lang_map.py",
        "grep_strategy": ["tree_sitter", "get_language", "LANG_MAP"],
    },
    {
        "id": 9,
        "query": "What imports from the capsule module?",
        "type": "import_graph_query",
        "expected_file_hint": "any file importing capsule",
        "grep_strategy": [
            "from beacon.search.capsule import",
            "from beacon.search import capsule",
            "search.capsule",
        ],
    },
    {
        "id": 10,
        "query": "How does session deduplication prevent returning the same symbols twice?",
        "type": "multi_module_feature",
        "expected_file_hint": "beacon/search/capsule.py",
        "grep_strategy": ["exclude_fqns", "_sent_fqns", "session.*dedup"],
    },
]


def _select_queries(root: str) -> list[dict]:
    """Pick the right query set based on what's indexed at *root*."""
    r = Path(root)
    if (r / "django" / "__init__.py").exists():
        return QUERIES
    if (r / "beacon" / "mcp.py").exists():
        return BEACON_CODE_QUERIES
    return QUERIES  # default to Django set for external codebases


# ── Token counting ────────────────────────────────────────────────────────────

def count_tokens_approx(text: str) -> int:
    """Rough BPE token count: ~4 chars per token for code."""
    return max(1, len(text) // 4)


# ── Beacon path (direct Python API) ──────────────────────────────────────────

def beacon_capsule_direct(query: str, root: str, max_tokens: int = 8000) -> dict[str, Any]:
    """Run the full capsule pipeline (semantic search + graph expansion + budget trim)
    via the Python API directly — no subprocess overhead.
    """
    from beacon.schema import open_db
    from beacon.search.capsule import get_capsule, render_capsule

    db_path = str(Path(root) / ".beacon" / "index.db")
    t0 = time.time()
    try:
        conn = open_db(db_path)
        cap = get_capsule(conn, query, max_tokens=max_tokens)
        output = render_capsule(cap)
        elapsed = time.time() - t0
        tokens = count_tokens_approx(output)
        return {
            "tokens": tokens,
            "elapsed_s": round(elapsed, 2),
            "output": output,
            "returncode": 0,
        }
    except Exception as e:
        return {"tokens": 0, "elapsed_s": round(time.time() - t0, 2), "error": str(e), "output": ""}


# ── Baseline path ─────────────────────────────────────────────────────────────

_GREP_EXCLUDE_DIRS = [
    ".venv", "venv", "__pycache__", ".git", "node_modules", "build", "dist",
    ".tox", ".eggs", "*.egg-info",
]


def grep_search(patterns: list[str], root: str) -> dict[str, Any]:
    """Simulate what an AI agent does without Beacon:
    1. grep -rl to find candidate files
    2. Read the top 3 matching files in full

    Uses /usr/bin/grep directly (rg is wrapped by Claude Code on this machine).
    Excludes virtualenvs and build artifacts (mirrors what rg/.gitignore would do).
    Total tokens = grep listing output + file contents read.
    """
    grep_bin = "/usr/bin/grep"
    exclude_args = []
    for d in _GREP_EXCLUDE_DIRS:
        exclude_args += [f"--exclude-dir={d}"]

    total_tokens = 0
    steps = []
    file_hit_count: dict[str, int] = {}

    for pattern in patterns:
        try:
            result = subprocess.run(
                [grep_bin, "-rl", "--include=*.py"] + exclude_args + [pattern, "."],
                capture_output=True, text=True, timeout=30,
                cwd=root,
            )
            matching_files = [f.lstrip("./") for f in result.stdout.strip().split("\n") if f]
            grep_tokens = count_tokens_approx(result.stdout)
            total_tokens += grep_tokens
            steps.append({
                "pattern": pattern,
                "matching_files": len(matching_files),
                "grep_output_tokens": grep_tokens,
            })
            for f in matching_files:
                file_hit_count[f] = file_hit_count.get(f, 0) + 1
        except Exception as e:
            steps.append({"pattern": pattern, "error": str(e)})

    # Rank files by how many patterns matched — most-mentioned = most likely relevant.
    # This mirrors what an agent would do: pick the files grep mentions most often.
    ranked_files = sorted(file_hit_count, key=lambda f: -file_hit_count[f])

    # Simulate reading the top 3 unique matching files fully
    files_read_tokens = 0
    files_read = []
    for fp in ranked_files[:3]:
        try:
            content = (Path(root) / fp).read_text(encoding="utf-8", errors="replace")
            tok = count_tokens_approx(content)
            files_read_tokens += tok
            files_read.append({"file": fp, "tokens": tok})
        except Exception:
            pass

    total_tokens += files_read_tokens
    return {
        "tokens": total_tokens,
        "grep_steps": steps,
        "files_read": files_read,
        "files_read_tokens": files_read_tokens,
    }


# ── Recall check ─────────────────────────────────────────────────────────────

def check_recall(beacon_output: str, expected_hint: str) -> bool:
    """Heuristic: did the beacon output mention the expected file or key symbol?
    Checks the full output (not truncated) against key terms from the hint.
    """
    skip = {"any", "file", "importing", "or", "similar"}
    hint_parts = [
        p for p in re.split(r'[/\\._ ]', expected_hint)
        if len(p) >= 3 and p.lower() not in skip
    ]
    if not hint_parts:
        return False
    output_lower = beacon_output.lower()
    matches = sum(1 for p in hint_parts if p.lower() in output_lower)
    return matches >= max(1, len(hint_parts) // 2)


def check_baseline_recall(files_read: list[dict], expected_hint: str) -> bool:
    """Did the baseline land on the right file?

    Uses the filename stem (e.g. 'csrf' from 'csrf.py') as the discriminator
    to avoid false positives from shared path components like 'django'.
    """
    stem = Path(expected_hint.split(" ")[0]).stem.lower()  # e.g. "csrf", "executor"
    for f in files_read:
        if stem in f["file"].lower():
            return True
    return False


# ── Core benchmark runner ─────────────────────────────────────────────────────

def run_benchmark(
    root: str,
    output_path: Path | None = None,
    on_result: Callable[[dict], None] | None = None,
    max_tokens: int = 8000,
) -> list[dict]:
    """Run all QUERIES and return list of per-query result dicts.

    Args:
        root:         Path to the indexed codebase (must have .beacon/index.db)
        output_path:  If provided, write JSON results to this path
        on_result:    Optional callback called after each query completes
        max_tokens:   Token budget passed to get_capsule (default 8000)

    Returns:
        List of result dicts, one per query, each containing:
          id, query, type, beacon_tokens, baseline_tokens,
          savings_ratio, pct_saved, beacon_recall, baseline_recall,
          beacon_elapsed_s, beacon_detail, baseline_detail
    """
    queries = _select_queries(root)
    results = []

    for q in queries:
        beacon = beacon_capsule_direct(q["query"], root, max_tokens=max_tokens)
        baseline = grep_search(q["grep_strategy"], root)

        beacon_recall = check_recall(beacon.get("output", ""), q["expected_file_hint"])
        baseline_recall = check_baseline_recall(baseline.get("files_read", []), q["expected_file_hint"])

        b_tok = beacon.get("tokens", 0)
        bl_tok = baseline["tokens"]
        savings_ratio = (bl_tok / b_tok) if b_tok > 0 else 0.0
        pct_saved = round((1 - b_tok / max(1, bl_tok)) * 100, 1) if bl_tok > 0 else 0.0

        result = {
            "id": q["id"],
            "query": q["query"],
            "type": q["type"],
            "beacon_tokens": b_tok,
            "baseline_tokens": bl_tok,
            "savings_ratio": round(savings_ratio, 2),
            "pct_saved": pct_saved,
            "beacon_recall": beacon_recall,
            "baseline_recall": baseline_recall,
            "beacon_elapsed_s": beacon.get("elapsed_s", 0),
            "beacon_detail": beacon,
            "baseline_detail": baseline,
        }
        results.append(result)

        if on_result:
            on_result(result)

    if output_path:
        output_path.write_text(json.dumps(results, indent=2))

    return results


# ── Summary helpers ───────────────────────────────────────────────────────────

def summary_stats(results: list[dict]) -> dict:
    """Compute aggregate stats from a results list."""
    n = len(results)
    if n == 0:
        return {}
    avg_savings = sum(r["savings_ratio"] for r in results) / n
    avg_pct_saved = sum(r["pct_saved"] for r in results) / n
    beacon_wins = sum(1 for r in results if r["savings_ratio"] > 1.0)
    beacon_recall_count = sum(1 for r in results if r["beacon_recall"])
    total_beacon = sum(r["beacon_tokens"] for r in results)
    total_baseline = sum(r["baseline_tokens"] for r in results)
    overall_pct_saved = round((1 - total_beacon / max(1, total_baseline)) * 100, 1)
    return {
        "avg_savings_ratio": round(avg_savings, 2),
        "avg_pct_saved": round(avg_pct_saved, 1),
        "overall_pct_saved": overall_pct_saved,
        "beacon_wins": beacon_wins,
        "total_queries": n,
        "beacon_recall_count": beacon_recall_count,
        "total_beacon_tokens": total_beacon,
        "total_baseline_tokens": total_baseline,
    }
