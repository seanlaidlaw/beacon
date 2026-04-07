"""
Build a file-extension → (language_name, tree_sitter_module) map from the
Helix editor's languages.toml, cross-referenced against the tree-sitter
grammar packages we actually have installed.

The Helix file is the most comprehensive, community-maintained catalogue of
language → file-type mappings available. We use it as ground truth for which
extensions belong to which language, then filter down to only the languages
we can actually parse.

Cached to .beacon/languages.toml after first download.
"""

import importlib
import tomllib
import urllib.request
from pathlib import Path

HELIX_LANGUAGES_URL = (
    "https://raw.githubusercontent.com/helix-editor/helix/"
    "refs/heads/master/languages.toml"
)

# Map from Helix language name → tree-sitter Python package name.
# Only languages listed here will be indexed.
# Add more as you install additional tree-sitter-* packages.
GRAMMAR_PACKAGES: dict[str, str] = {
    "python":         "tree_sitter_python",
    "javascript":     "tree_sitter_javascript",
    "jsx":            "tree_sitter_javascript",
    "typescript":     "tree_sitter_typescript",
    "tsx":            "tree_sitter_typescript",
    "go":             "tree_sitter_go",
    "rust":           "tree_sitter_rust",
    "java":           "tree_sitter_java",
    "c":              "tree_sitter_c",
    "cpp":            "tree_sitter_cpp",
    "c++":            "tree_sitter_cpp",
    "bash":           "tree_sitter_bash",
    "lua":            "tree_sitter_lua",
    "swift":          "tree_sitter_swift",
    "markdown":       "tree_sitter_markdown",
    # R is not on PyPI — loaded from a compiled .so built at first use
    "r":              "_r_grammar",
}

# Canonical name to use when multiple Helix names share a grammar package
_CANONICAL: dict[str, str] = {
    "c++":  "cpp",
    "jsx":  "javascript",
    "tsx":  "typescript",
}


def _download(cache_path: Path) -> bytes:
    """Download languages.toml, caching to disk."""
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if cache_path.exists():
        return cache_path.read_bytes()
    print(f"Downloading Helix languages.toml → {cache_path} ...", end=" ", flush=True)
    with urllib.request.urlopen(HELIX_LANGUAGES_URL, timeout=15) as resp:
        data = resp.read()
    cache_path.write_bytes(data)
    print("done")
    return data


def _build_r_grammar(cache_dir: Path) -> bool:
    """
    Clone tree-sitter-r and compile parser.c → r.so using system gcc.
    Returns True on success.
    """
    import subprocess
    r_so = cache_dir / "grammars" / "r.so"
    if r_so.exists():
        return True

    r_so.parent.mkdir(parents=True, exist_ok=True)
    r_src = cache_dir / "grammars" / "tree-sitter-r"

    print("Building R grammar from source...", end=" ", flush=True)

    if not r_src.exists():
        result = subprocess.run(
            ["git", "clone", "--depth=1",
             "https://github.com/r-lib/tree-sitter-r", str(r_src)],
            capture_output=True,
        )
        if result.returncode != 0:
            print(f"git clone failed: {result.stderr.decode()[:200]}")
            return False

    srcs = [str(r_src / "src" / "parser.c")]
    scanner = r_src / "src" / "scanner.c"
    if scanner.exists():
        srcs.append(str(scanner))

    result = subprocess.run(
        ["gcc", "-shared", "-fPIC", "-O2", "-o", str(r_so)]
        + srcs + ["-I", str(r_src / "src")],
        capture_output=True,
    )
    if result.returncode != 0:
        print(f"gcc failed: {result.stderr.decode()[:200]}")
        return False

    print("done")
    return True


def _installed_packages(cache_dir: Path) -> set[str]:
    """Return grammar identifiers that are available (pip packages + compiled .so)."""
    installed = set()
    for pkg in set(GRAMMAR_PACKAGES.values()):
        if pkg == "_r_grammar":
            if _build_r_grammar(cache_dir):
                installed.add(pkg)
        else:
            try:
                importlib.import_module(pkg)
                installed.add(pkg)
            except ImportError:
                pass
    return installed


def build_lang_map(cache_dir: str | Path | None = None) -> dict[str, str]:
    """
    Return {file_extension: canonical_language_name} for all extensions
    whose language has an installed tree-sitter grammar.

    cache_dir : directory to store the downloaded languages.toml and compiled grammars
                (default: ~/.cache/beacon)
    """
    if cache_dir is None:
        cache_dir = Path.home() / ".cache" / "beacon"
    cache_dir = Path(cache_dir)
    cache_path = cache_dir / "languages.toml"

    raw = _download(cache_path)
    data = tomllib.loads(raw.decode("utf-8", errors="replace"))

    if cache_dir is None:
        cache_dir = Path.home() / ".cache" / "beacon"
    cache_dir = Path(cache_dir)

    installed = _installed_packages(cache_dir)
    # Which Helix language names have an installed grammar?
    supported_names = {
        name
        for name, pkg in GRAMMAR_PACKAGES.items()
        if pkg in installed
    }
    # rmarkdown / quarto use the markdown or r grammar — treat them as "r"
    # so .rmd and .qmd get picked up if R is available
    if "r" in supported_names:
        supported_names.update({"rmarkdown", "quarto"})


    lang_map: dict[str, str] = {}

    for lang in data.get("language", []):
        name: str = lang.get("name", "").lower()
        if name not in supported_names:
            continue
        canonical = _CANONICAL.get(name, name)
        file_types = lang.get("file-types", [])

        for ft in file_types:
            if isinstance(ft, str):
                # Plain extension — normalise to ".ext"
                ext = ft if ft.startswith(".") else f".{ft}"
                # Don't overwrite a higher-priority entry (e.g. .h → c already set)
                if ext not in lang_map:
                    lang_map[ext] = canonical
            elif isinstance(ft, dict):
                glob = ft.get("glob", "")
                # Only handle simple "*.ext" globs, not full filename matches
                if glob.startswith("*."):
                    ext = glob[1:]   # "*.py" → ".py"
                    if ext not in lang_map:
                        lang_map[ext] = canonical
                # Full-filename globs (e.g. "Makefile") are skipped —
                # we only index by extension for simplicity

    return lang_map


# Pre-built fallback used when there is no network / cache miss
_FALLBACK: dict[str, str] = {
    ".py":   "python",
    ".js":   "javascript",
    ".jsx":  "javascript",
    ".ts":   "typescript",
    ".tsx":  "typescript",
    ".go":   "go",
    ".rs":   "rust",
    ".java": "java",
    ".c":    "c",
    ".h":    "c",
    ".cc":   "cpp",
    ".cpp":  "cpp",
    ".cxx":  "cpp",
    ".hpp":  "cpp",
    ".hh":   "cpp",
}


def get_lang_map(cache_dir: str | Path | None = None) -> dict[str, str]:
    """
    Return the lang map, falling back to the built-in minimal set on error.
    """
    try:
        return build_lang_map(cache_dir)
    except Exception as e:
        print(f"Warning: could not build lang map from Helix ({e}), using fallback")
        return _FALLBACK.copy()
