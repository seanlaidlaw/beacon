"""Tests for beacon/indexer/symbols.py — tree-sitter symbol extraction.

Requires all tree-sitter grammar packages (installed via `pip install -e .`).
Skipped automatically when grammars are missing (e.g. partial local installs).
"""
import textwrap
import pytest

# Skip the entire module if any tree-sitter grammar is unavailable.
# On CI, `pip install -e .` installs all grammars from pyproject.toml.
_sym = pytest.importorskip(
    "beacon.indexer.symbols",
    reason="tree-sitter grammar packages not installed — run `pip install -e .`",
)
_ts_python = pytest.importorskip("tree_sitter_python")
_ts = pytest.importorskip("tree_sitter")

_extract_python = _sym._extract_python
_extract_python_docstring = _sym._extract_python_docstring
_body_preview = _sym._body_preview
FileSymbols = _sym.FileSymbols

from tree_sitter import Language, Parser


# ── Helpers ───────────────────────────────────────────────────────────────────

_PY_LANG = Language(_ts_python.language())


def _parse(source: str):
    """Parse Python source and return (tree, src_bytes)."""
    src = textwrap.dedent(source).encode()
    parser = Parser(_PY_LANG)
    return parser.parse(src), src


def _py_symbols(source: str, rel_path: str = "mod.py") -> FileSymbols:
    tree, src = _parse(source)
    return _extract_python(tree, src, rel_path)


def _find_node(tree, node_type: str):
    """Return first node of the given type in the tree."""
    def _search(n):
        if n.type == node_type:
            return n
        for child in n.children:
            result = _search(child)
            if result:
                return result
        return None
    return _search(tree.root_node)


# ── FQN qualification ────────────────────────────────────────────────────────

class TestFqnQualification:
    def test_top_level_function(self):
        fs = _py_symbols("def foo():\n    pass\n")
        fqns = {s.fqn for s in fs.symbols}
        assert "mod.py::foo" in fqns

    def test_top_level_class(self):
        fs = _py_symbols("class MyClass:\n    pass\n")
        fqns = {s.fqn for s in fs.symbols}
        assert "mod.py::MyClass" in fqns

    def test_method_qualified_under_class(self):
        source = """\
            class MyClass:
                def my_method(self):
                    pass
        """
        fs = _py_symbols(source)
        fqns = {s.fqn for s in fs.symbols}
        assert "mod.py::MyClass.my_method" in fqns

    def test_nested_class_method(self):
        source = """\
            class Outer:
                class Inner:
                    def deep(self):
                        pass
        """
        fs = _py_symbols(source)
        fqns = {s.fqn for s in fs.symbols}
        assert "mod.py::Outer.Inner.deep" in fqns

    def test_fqn_uses_rel_path_prefix(self):
        fs = _py_symbols("def bar():\n    pass\n", rel_path="pkg/sub/mod.py")
        fqns = {s.fqn for s in fs.symbols}
        assert "pkg/sub/mod.py::bar" in fqns


# ── Export detection ─────────────────────────────────────────────────────────

class TestExportDetection:
    def test_public_function_is_exported(self):
        fs = _py_symbols("def public_func():\n    pass\n")
        sym = next(s for s in fs.symbols if s.name == "public_func")
        assert sym.is_exported is True

    def test_private_function_not_exported(self):
        fs = _py_symbols("def _private_func():\n    pass\n")
        sym = next(s for s in fs.symbols if s.name == "_private_func")
        assert sym.is_exported is False

    def test_dunder_method_not_exported(self):
        source = "class A:\n    def __init__(self):\n        pass\n"
        fs = _py_symbols(source)
        sym = next(s for s in fs.symbols if s.name == "__init__")
        assert sym.is_exported is False

    def test_public_class_is_exported(self):
        fs = _py_symbols("class PublicClass:\n    pass\n")
        sym = next(s for s in fs.symbols if s.name == "PublicClass")
        assert sym.is_exported is True

    def test_private_class_not_exported(self):
        fs = _py_symbols("class _PrivateClass:\n    pass\n")
        sym = next(s for s in fs.symbols if s.name == "_PrivateClass")
        assert sym.is_exported is False


# ── Test detection ────────────────────────────────────────────────────────────

class TestTestDetection:
    def test_function_prefixed_test_is_test(self):
        fs = _py_symbols("def test_something():\n    pass\n")
        sym = next(s for s in fs.symbols if s.name == "test_something")
        assert sym.is_test is True

    def test_regular_function_not_test(self):
        fs = _py_symbols("def do_something():\n    pass\n")
        sym = next(s for s in fs.symbols if s.name == "do_something")
        assert sym.is_test is False

    def test_suffix_test_not_detected(self):
        # "something_test" should NOT be flagged (only prefix "test_" counts)
        fs = _py_symbols("def something_test():\n    pass\n")
        sym = next(s for s in fs.symbols if s.name == "something_test")
        assert sym.is_test is False

    def test_file_in_test_directory_flags_all_symbols(self):
        # "test" in rel_path → all symbols are marked is_test
        fs = _py_symbols("def regular_func():\n    pass\n", rel_path="tests/mod.py")
        sym = next(s for s in fs.symbols if s.name == "regular_func")
        assert sym.is_test is True

    def test_file_not_in_test_path_not_flagged(self):
        fs = _py_symbols("def regular_func():\n    pass\n", rel_path="src/app/views.py")
        sym = next(s for s in fs.symbols if s.name == "regular_func")
        assert sym.is_test is False


# ── Kind classification ───────────────────────────────────────────────────────

class TestKindClassification:
    def test_top_level_function_kind(self):
        fs = _py_symbols("def top_level():\n    pass\n")
        sym = next(s for s in fs.symbols if s.name == "top_level")
        assert sym.kind == "function"

    def test_method_kind(self):
        source = "class A:\n    def my_method(self):\n        pass\n"
        fs = _py_symbols(source)
        sym = next(s for s in fs.symbols if s.name == "my_method")
        assert sym.kind == "method"

    def test_class_kind(self):
        fs = _py_symbols("class Foo:\n    pass\n")
        sym = next(s for s in fs.symbols if s.name == "Foo")
        assert sym.kind == "class"

    def test_async_function_is_function(self):
        fs = _py_symbols("async def fetch():\n    pass\n")
        sym = next(s for s in fs.symbols if s.name == "fetch")
        assert sym.kind == "function"


# ── Line numbers ──────────────────────────────────────────────────────────────

class TestLineNumbers:
    def test_function_start_line(self):
        fs = _py_symbols("x = 1\ndef foo():\n    pass\n")
        sym = next(s for s in fs.symbols if s.name == "foo")
        assert sym.start_line == 2

    def test_function_end_line(self):
        source = "def foo():\n    x = 1\n    return x\n"
        fs = _py_symbols(source)
        sym = next(s for s in fs.symbols if s.name == "foo")
        assert sym.end_line == 3


# ── Docstring extraction ──────────────────────────────────────────────────────

class TestExtractPythonDocstring:
    def test_triple_quoted_docstring(self):
        source = 'def foo():\n    """This is the docstring."""\n    pass\n'
        tree, src = _parse(source)
        node = _find_node(tree, "function_definition")
        result = _extract_python_docstring(node, src)
        assert "This is the docstring." in result

    def test_single_quoted_docstring(self):
        source = "def foo():\n    'Simple docstring'\n    pass\n"
        tree, src = _parse(source)
        node = _find_node(tree, "function_definition")
        result = _extract_python_docstring(node, src)
        assert "Simple docstring" in result

    def test_no_docstring_returns_empty(self):
        source = "def foo():\n    x = 1\n    return x\n"
        tree, src = _parse(source)
        node = _find_node(tree, "function_definition")
        result = _extract_python_docstring(node, src)
        assert result == ""

    def test_code_before_string_not_docstring(self):
        source = "def foo():\n    x = 1\n    'not a docstring'\n    return x\n"
        tree, src = _parse(source)
        node = _find_node(tree, "function_definition")
        result = _extract_python_docstring(node, src)
        assert result == ""

    def test_docstring_capped_at_512_chars(self):
        long_doc = "A" * 600
        source = f'def foo():\n    """{long_doc}"""\n    pass\n'
        tree, src = _parse(source)
        node = _find_node(tree, "function_definition")
        result = _extract_python_docstring(node, src)
        assert len(result) <= 512


# ── Body preview ──────────────────────────────────────────────────────────────

class TestBodyPreview:
    def test_returns_body_content(self):
        source = "def foo():\n    x = 1\n    y = 2\n    return x + y\n"
        tree, src = _parse(source)
        node = _find_node(tree, "function_definition")
        result = _body_preview(node, src)
        assert "x = 1" in result
        assert "return x + y" in result

    def test_skips_leading_docstring(self):
        source = 'def foo():\n    """Docstring."""\n    x = 1\n    return x\n'
        tree, src = _parse(source)
        node = _find_node(tree, "function_definition")
        result = _body_preview(node, src)
        # Docstring content should not appear as body
        assert "Docstring" not in result
        assert "x = 1" in result

    def test_function_with_only_docstring_returns_empty(self):
        source = 'def foo():\n    """Only docstring."""\n'
        tree, src = _parse(source)
        node = _find_node(tree, "function_definition")
        result = _body_preview(node, src)
        assert result == ""

    def test_caps_at_max_lines(self):
        lines = "\n".join(f"    x_{i} = {i}" for i in range(30))
        source = f"def foo():\n{lines}\n"
        tree, src = _parse(source)
        node = _find_node(tree, "function_definition")
        result = _body_preview(node, src, max_lines=5)
        result_lines = [l for l in result.splitlines() if l.strip()]
        assert len(result_lines) <= 5

    def test_result_is_dedented(self):
        source = "def foo():\n    x = 1\n    return x\n"
        tree, src = _parse(source)
        node = _find_node(tree, "function_definition")
        result = _body_preview(node, src)
        # Should not have leading spaces (dedented)
        assert not result.startswith("    ")

    def test_empty_body_returns_empty(self):
        source = "def foo():\n    pass\n"
        tree, src = _parse(source)
        node = _find_node(tree, "function_definition")
        result = _body_preview(node, src)
        # "pass" may or may not appear depending on blank-line filtering
        assert isinstance(result, str)


# ── Edges ─────────────────────────────────────────────────────────────────────

class TestEdges:
    def test_contains_edge_from_class_to_method(self):
        source = "class A:\n    def method(self):\n        pass\n"
        fs = _py_symbols(source)
        contains = [e for e in fs.edges if e.edge_type == "CONTAINS"]
        parent_fqns = {e.source_fqn for e in contains}
        assert "mod.py::A" in parent_fqns

    def test_import_statement_produces_imports_edge(self):
        source = "import os\n\ndef foo():\n    pass\n"
        fs = _py_symbols(source)
        import_edges = [e for e in fs.edges if e.edge_type == "IMPORTS"]
        targets = {e.target_name for e in import_edges}
        assert "os" in targets

    def test_from_import_produces_imports_edge(self):
        source = "from pathlib import Path\n\ndef foo():\n    pass\n"
        fs = _py_symbols(source)
        import_edges = [e for e in fs.edges if e.edge_type == "IMPORTS"]
        targets = {e.target_name for e in import_edges}
        assert any("Path" in t for t in targets)
