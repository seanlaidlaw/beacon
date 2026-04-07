"""
File discovery with layered ignore support.

Priority order (first match wins):
  1. Hardcoded always-exclude dirs (node_modules, .git, __pycache__, etc.)
  2. .beaconignore  (beacon-specific overrides)
  3. .claudeignore (Claude Code ignore file)
  4. .gitignore    (standard git ignore, per-directory)

Only files with a recognised tree-sitter language extension are returned.
"""

import fnmatch
import re
from pathlib import Path

# Always excluded — never descend into these directories
ALWAYS_EXCLUDE_DIRS = {
    # VCS
    ".git", ".hg", ".svn",
    # Python environments
    "venv", ".venv", "env", ".env", "site-packages", "__pycache__",
    ".tox", ".mypy_cache", ".pytest_cache", ".ruff_cache", ".hypothesis",
    # Build outputs
    "dist", "build", ".next", ".nuxt", "target", "obj",
    "bin/Debug", "bin/Release", "Pods", "DerivedData", ".expo",
    # Package managers
    "node_modules", ".npm", ".yarn", ".pnpm-store",
    ".gradle", ".m2",
    # beacon own index
    ".beacon",
    # Conda/system environment dirs — not source code
    "conda-meta",           # conda package metadata
    "include",              # C/C++ system headers
    "go",                   # Go SDK bundled in conda
    "jre",                  # Java runtime
    "fonts",                # font files
    "ssl",                  # SSL certificates
    "man",                  # man pages
    "share",                # shared data (locale, icons, etc.)
    "var",                  # variable data
    "sbin",                 # system binaries
    "libexec",              # internal executables
    "pkgconfig",            # pkg-config files
    "cmake",                # CMake find-modules
    "girepository-1.0",     # GObject introspection
    "gdalplugins",          # GDAL plugins
    "gdk-pixbuf-2.0",       # GTK image loaders
    "dbus-1.0",             # D-Bus interfaces
    "gnome-settings-daemon-3.0",
    "bfd-plugins",
    "x86_64-conda-linux-gnu",
    "x86_64-conda_cos6-linux-gnu",
    # Common vendor/generated dirs
    "vendor", "third_party", "third-party", "extern", "external",
    "generated", "pb", "protobuf",
    # Misc large non-code dirs
    "duckdb_build",
}

# Extension → tree-sitter language tag.
# Populated lazily from Helix languages.toml on first use.
# Falls back to a minimal hardcoded set if the download fails.
_LANG_MAP_CACHE: dict[str, str] | None = None


def _get_lang_map() -> dict[str, str]:
    global _LANG_MAP_CACHE
    if _LANG_MAP_CACHE is None:
        from beacon.lang_map import get_lang_map
        _LANG_MAP_CACHE = get_lang_map()
    return _LANG_MAP_CACHE


# Keep a module-level alias for callers that import LANG_MAP directly
# (resolved on first access via the Scanner)
LANG_MAP: dict[str, str] = {}   # populated on first Scanner instantiation


# ── Gitignore parser ──────────────────────────────────────────────────────────

def _load_patterns(path: Path) -> list[str]:
    """Return non-empty, non-comment lines from an ignore file."""
    if not path.exists():
        return []
    lines = []
    for line in path.read_text(errors="replace").splitlines():
        line = line.rstrip()
        if line and not line.startswith("#"):
            lines.append(line)
    return lines


def _gitignore_match(rel: str, patterns: list[str]) -> bool:
    """
    Very close approximation to gitignore matching.
    Handles leading /, trailing /, ** globs, and simple wildcards.
    """
    parts = rel.replace("\\", "/")
    for pattern in patterns:
        neg = pattern.startswith("!")
        p = pattern.lstrip("!")

        # Directory-only pattern
        dir_only = p.endswith("/")
        p = p.rstrip("/")

        # Anchor to root if pattern contains a slash (other than trailing)
        anchored = "/" in p

        # Convert gitignore glob to fnmatch glob
        if "**" in p:
            # ** matches any path segment
            regex = re.escape(p).replace(r"\*\*", ".*").replace(r"\*", "[^/]*").replace(r"\?", "[^/]")
            matched = bool(re.fullmatch(regex, parts)) or bool(re.search(r"(^|/)" + regex + r"($|/)", parts))
        elif anchored:
            p_clean = p.lstrip("/")
            matched = fnmatch.fnmatch(parts, p_clean) or parts.startswith(p_clean + "/")
        else:
            # Match against any path component or the full path
            name = parts.split("/")[-1]
            matched = fnmatch.fnmatch(name, p) or fnmatch.fnmatch(parts, p) or any(
                fnmatch.fnmatch(seg, p) for seg in parts.split("/")
            )

        if matched:
            return not neg  # negation = "don't exclude"

    return False


# ── Scanner ───────────────────────────────────────────────────────────────────

class Scanner:
    def __init__(self, root: Path):
        self.root = root.resolve()
        # Populate the module-level alias on first instantiation
        global LANG_MAP
        LANG_MAP = _get_lang_map()
        # Root-level ignore patterns loaded once
        self._root_patterns: list[str] = (
            _load_patterns(self.root / ".beaconignore") +
            _load_patterns(self.root / ".claudeignore") +
            _load_patterns(self.root / ".gitignore")
        )
        # Per-directory gitignore cache
        self._dir_patterns: dict[Path, list[str]] = {}

    def _patterns_for_dir(self, d: Path) -> list[str]:
        """Load and cache gitignore patterns for a directory."""
        if d not in self._dir_patterns:
            self._dir_patterns[d] = _load_patterns(d / ".gitignore")
        return self._dir_patterns[d]

    def _is_ignored(self, path: Path) -> bool:
        rel = str(path.relative_to(self.root))

        # Hardcoded dir exclusions — check only path components relative to root
        # (not ancestors of root, which may legitimately be named node_modules etc.)
        for part in Path(rel).parts:
            if part in ALWAYS_EXCLUDE_DIRS:
                return True

        # Root-level patterns
        if _gitignore_match(rel, self._root_patterns):
            return True

        # Per-directory gitignore patterns (walk up from file's parent)
        cur = path.parent
        while cur != self.root and cur != cur.parent:
            local = self._patterns_for_dir(cur)
            if local:
                # Relative to the directory that owns the gitignore
                try:
                    local_rel = str(path.relative_to(cur))
                    if _gitignore_match(local_rel, local):
                        return True
                except ValueError:
                    pass
            cur = cur.parent

        return False

    def collect(self) -> list[tuple[Path, str]]:
        """
        Walk root and return [(path, language), ...] for all indexable files.
        Uses os.walk so excluded directories are pruned before descending —
        critical for large trees like conda environments.
        """
        import os
        results: list[tuple[Path, str]] = []

        for dirpath, dirnames, filenames in os.walk(self.root, followlinks=False):
            cur = Path(dirpath)

            # Prune excluded directories in-place (prevents os.walk from descending)
            dirnames[:] = sorted(
                d for d in dirnames
                if d not in ALWAYS_EXCLUDE_DIRS
                and not self._is_ignored(cur / d)
            )

            rel_dir = str(cur.relative_to(self.root)) if cur != self.root else ""

            for filename in sorted(filenames):
                lang = _get_lang_map().get(Path(filename).suffix.lower())
                if not lang:
                    continue
                full = cur / filename
                if self._is_ignored(full):
                    continue
                results.append((full, lang))

        return results


def scan(root: str | Path) -> list[tuple[Path, str]]:
    """Convenience wrapper — returns list of (path, language) tuples."""
    return Scanner(Path(root)).collect()
