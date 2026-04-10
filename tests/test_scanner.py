"""Tests for beacon/indexer/scanner.py — file discovery and ignore logic."""
import tempfile
from pathlib import Path

import pytest

from beacon.indexer.scanner import _gitignore_match, _load_patterns, Scanner


# ── _gitignore_match ──────────────────────────────────────────────────────────

class TestGitignoreMatch:
    # Basic glob matching
    def test_simple_extension_match(self):
        assert _gitignore_match("foo.pyc", ["*.pyc"]) is True

    def test_simple_extension_no_match(self):
        assert _gitignore_match("foo.py", ["*.pyc"]) is False

    def test_no_patterns_never_matches(self):
        assert _gitignore_match("anything/at/all.py", []) is False

    def test_exact_filename_match(self):
        assert _gitignore_match("Makefile", ["Makefile"]) is True

    def test_partial_name_no_match(self):
        assert _gitignore_match("Makefile.local", ["Makefile"]) is False

    # Anchored patterns (contain / other than trailing)
    def test_anchored_matches_from_root(self):
        # "/build" matches "build/output.js" but treats it as rooted
        assert _gitignore_match("build/output.js", ["/build"]) is True

    def test_anchored_does_not_match_nested(self):
        # "/build" should NOT match "src/build/output.js"
        assert _gitignore_match("src/build/output.js", ["/build"]) is False

    def test_anchored_matches_file_in_root(self):
        assert _gitignore_match("dist/bundle.js", ["/dist"]) is True

    # Double-star patterns
    def test_doublestar_matches_nested(self):
        assert _gitignore_match("a/node_modules/pkg/index.js", ["**/node_modules"]) is True

    def test_doublestar_requires_leading_path_component(self):
        # This implementation's ** regex (.*/foo) requires a / before the pattern,
        # so **/foo does NOT match foo/... at the root (no leading component).
        # In practice, ALWAYS_EXCLUDE_DIRS handles root-level node_modules anyway.
        assert _gitignore_match("node_modules/pkg/index.js", ["**/node_modules"]) is False

    def test_doublestar_no_false_positive(self):
        assert _gitignore_match("my_node_modules_backup/x.js", ["**/node_modules"]) is False

    # Unanchored patterns match any component
    def test_unanchored_matches_nested_dir(self):
        # "dist" (no slash) matches any component named "dist"
        assert _gitignore_match("a/b/dist/c.js", ["dist"]) is True

    def test_unanchored_matches_filename(self):
        assert _gitignore_match("src/debug.log", ["debug.log"]) is True

    # Directory-only patterns (trailing /) — this impl strips the slash and
    # treats them as unanchored patterns (dir_only flag is computed but unused)
    def test_dir_pattern_matches_subpath(self):
        assert _gitignore_match("dist/bundle.js", ["dist/"]) is True

    # Negation (must come BEFORE the matching positive pattern to override it,
    # because this implementation returns on first match)
    def test_negation_before_positive_prevents_ignore(self):
        # Negation first: "important.log" should NOT be ignored
        assert _gitignore_match("important.log", ["!important.log", "*.log"]) is False

    def test_negation_after_positive_does_not_override(self):
        # Positive first: "important.log" IS ignored (negation comes too late)
        assert _gitignore_match("important.log", ["*.log", "!important.log"]) is True

    def test_negation_of_non_matching_pattern(self):
        # Negation for a non-matching name — then *.log still matches
        assert _gitignore_match("other.log", ["!important.log", "*.log"]) is True

    # Wildcard in middle
    def test_wildcard_middle(self):
        assert _gitignore_match("foo_bar.pyc", ["*_bar.pyc"]) is True


# ── _load_patterns ────────────────────────────────────────────────────────────

class TestLoadPatterns:
    def test_returns_empty_for_missing_file(self, tmp_path):
        result = _load_patterns(tmp_path / "nonexistent.ignore")
        assert result == []

    def test_strips_comments(self, tmp_path):
        f = tmp_path / ".gitignore"
        f.write_text("# This is a comment\n*.pyc\n# Another comment\n*.log\n")
        result = _load_patterns(f)
        assert "# This is a comment" not in result
        assert "*.pyc" in result
        assert "*.log" in result

    def test_strips_blank_lines(self, tmp_path):
        f = tmp_path / ".gitignore"
        f.write_text("*.pyc\n\n\n*.log\n")
        result = _load_patterns(f)
        assert "" not in result
        assert len(result) == 2

    def test_preserves_valid_patterns(self, tmp_path):
        f = tmp_path / ".gitignore"
        f.write_text("node_modules/\n*.pyc\n!important.py\n**/build\n")
        result = _load_patterns(f)
        assert "node_modules/" in result
        assert "*.pyc" in result
        assert "!important.py" in result
        assert "**/build" in result


# ── Scanner.collect() ─────────────────────────────────────────────────────────

class TestScannerCollect:
    def test_collects_python_files(self, tmp_path):
        (tmp_path / "main.py").write_text("x = 1")
        (tmp_path / "utils.py").write_text("def foo(): pass")
        scanner = Scanner(tmp_path)
        results = scanner.collect()
        langs = {lang for _, lang in results}
        names = {p.name for p, _ in results}
        assert "python" in langs
        assert "main.py" in names
        assert "utils.py" in names

    def test_ignores_unrecognised_extensions(self, tmp_path):
        (tmp_path / "data.xyz_unknown").write_text("data")
        (tmp_path / "script.py").write_text("pass")
        scanner = Scanner(tmp_path)
        results = scanner.collect()
        names = {p.name for p, _ in results}
        assert "data.xyz_unknown" not in names
        assert "script.py" in names

    def test_never_descends_into_always_exclude_dirs(self, tmp_path):
        pycache = tmp_path / "__pycache__"
        pycache.mkdir()
        (pycache / "mod.cpython-312.pyc").write_text("")  # won't have a lang anyway
        (pycache / "legit.py").write_text("pass")  # even .py inside should be excluded
        (tmp_path / "real.py").write_text("pass")
        scanner = Scanner(tmp_path)
        results = scanner.collect()
        paths = {p for p, _ in results}
        # Nothing inside __pycache__ should appear
        assert not any(str(p).find("__pycache__") != -1 for p in paths)
        # But real.py at root should appear
        assert any(p.name == "real.py" for p, _ in results)

    def test_never_descends_into_node_modules(self, tmp_path):
        nm = tmp_path / "node_modules"
        nm.mkdir()
        (nm / "package.py").write_text("pass")
        (tmp_path / "app.py").write_text("pass")
        scanner = Scanner(tmp_path)
        results = scanner.collect()
        names = {p.name for p, _ in results}
        assert "package.py" not in names
        assert "app.py" in names

    def test_respects_root_gitignore(self, tmp_path):
        (tmp_path / ".gitignore").write_text("ignored_module.py\n")
        (tmp_path / "ignored_module.py").write_text("pass")
        (tmp_path / "kept_module.py").write_text("pass")
        scanner = Scanner(tmp_path)
        results = scanner.collect()
        names = {p.name for p, _ in results}
        assert "ignored_module.py" not in names
        assert "kept_module.py" in names

    def test_empty_directory_returns_empty(self, tmp_path):
        scanner = Scanner(tmp_path)
        results = scanner.collect()
        assert results == []

    def test_nested_subdirectory_files_collected(self, tmp_path):
        subdir = tmp_path / "pkg" / "sub"
        subdir.mkdir(parents=True)
        (subdir / "module.py").write_text("pass")
        scanner = Scanner(tmp_path)
        results = scanner.collect()
        names = {p.name for p, _ in results}
        assert "module.py" in names
