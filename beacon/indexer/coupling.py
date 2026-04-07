"""
Git-based change coupling — mirrors vexp-core's indexer/change_coupling.rs.

Algorithm:
  1. Run `git log --format=%H --no-merges` to get commit SHAs
  2. For each commit, get the list of changed files
  3. Count how many commits each (file_a, file_b) pair co-appears in
  4. coupling_score = shared_commits / max(commits_a, commits_b)  [Jaccard-like]
  5. Only persist pairs with shared_commits >= MIN_SHARED_COMMITS

Also computes file_lineage (churn_score = commit_count normalised to [0, 1]).
"""

import sqlite3
from collections import defaultdict
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path

import git

MIN_SHARED_COMMITS = 4   # from vexp-core strings: "HAVING cnt >= 4"


def _safe_repo(root: Path) -> git.Repo | None:
    try:
        return git.Repo(str(root), search_parent_directories=True)
    except git.InvalidGitRepositoryError:
        return None


def compute(conn: sqlite3.Connection, root: Path) -> None:
    """
    Compute co_change_edges and file_lineage from git history.
    Safe to call repeatedly — uses INSERT OR REPLACE / UPSERT.
    """
    repo = _safe_repo(root)
    if repo is None:
        print("Skipping change coupling: not a git repository")
        return

    # --- gather per-commit file lists ---
    file_commits: dict[str, int] = defaultdict(int)   # file → commit count
    pair_commits: dict[tuple[str, str], int] = defaultdict(int)
    last_author: dict[str, str] = {}
    last_ts: dict[str, int] = {}
    total_commits = 0

    for commit in repo.iter_commits(no_merges=True):
        total_commits += 1
        try:
            changed = list(commit.stats.files.keys())
        except Exception:
            continue

        for f in changed:
            file_commits[f] += 1
            if commit.authored_date > last_ts.get(f, 0):
                last_ts[f] = commit.authored_date
                last_author[f] = commit.author.name or ""

        # All pairs in this commit
        for a, b in combinations(sorted(changed), 2):
            pair_commits[(a, b)] += 1

    # --- churn scores: normalise commit count to [0, 1] ---
    max_commits = max(file_commits.values(), default=1)
    now = datetime.now(timezone.utc).isoformat()

    lineage_rows = []
    for fp, cnt in file_commits.items():
        churn = cnt / max_commits
        ts = datetime.fromtimestamp(last_ts.get(fp, 0), tz=timezone.utc).isoformat()
        lineage_rows.append((fp, cnt, churn, last_author.get(fp, ""), ts, now))

    conn.executemany(
        """INSERT OR REPLACE INTO file_lineage
           (file_path, commit_count, churn_score, last_author, last_commit_ts, updated_at)
           VALUES (?, ?, ?, ?, ?, ?)""",
        lineage_rows,
    )

    # --- co-change edges ---
    edge_rows = []
    for (a, b), shared in pair_commits.items():
        if shared < MIN_SHARED_COMMITS:
            continue
        denom = max(file_commits[a], file_commits[b], 1)
        score = shared / denom
        edge_rows.append((a, b, score, shared, now))

    conn.executemany(
        """INSERT OR REPLACE INTO co_change_edges
           (file_a, file_b, coupling_score, shared_commits, updated_at)
           VALUES (?, ?, ?, ?, ?)""",
        edge_rows,
    )
    conn.commit()
    print(f"Change coupling: {len(edge_rows)} edges, "
          f"{len(lineage_rows)} file lineage entries from {total_commits} commits")
