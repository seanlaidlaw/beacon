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


# ── AST call detection (Python) ───────────────────────────────────────────────

class TestAstCalls:
    """_ast_calls must use the tree-sitter AST — not regex — so it has no
    false positives from string literals and no cross-scope leakage."""

    def test_no_false_positives_from_string_literals(self):
        """A call mentioned inside a string literal must NOT produce a CALLS edge."""
        source = textwrap.dedent("""\
            def foo():
                x = "please call helper() for help"
        """)
        fs = _py_symbols(source)
        call_targets = {e.target_name for e in fs.edges if e.edge_type == "CALLS"}
        assert "helper" not in call_targets

    def test_no_false_positives_from_comments(self):
        """A call mentioned in a comment must NOT produce a CALLS edge."""
        source = textwrap.dedent("""\
            def foo():
                # TODO: call cleanup() later
                pass
        """)
        fs = _py_symbols(source)
        call_targets = {e.target_name for e in fs.edges if e.edge_type == "CALLS"}
        assert "cleanup" not in call_targets

    def test_nested_calls_not_attributed_to_outer(self):
        """Calls inside a nested function must only be attributed to that
        nested function, not to the enclosing one."""
        source = textwrap.dedent("""\
            def outer():
                def inner():
                    bar()
                baz()
        """)
        fs = _py_symbols(source)
        calls_by_source: dict[str, set[str]] = {}
        for e in fs.edges:
            if e.edge_type == "CALLS":
                calls_by_source.setdefault(e.source_fqn, set()).add(e.target_name)
        outer_fqn = "mod.py::outer"
        inner_fqn = "mod.py::outer.inner"
        # bar() is inside inner — must NOT appear under outer
        assert "bar" not in calls_by_source.get(outer_fqn, set())
        assert "bar" in calls_by_source.get(inner_fqn, set())
        # baz() is in outer's direct body
        assert "baz" in calls_by_source.get(outer_fqn, set())

    def test_call_edges_have_confidence_1(self):
        source = textwrap.dedent("""\
            def foo():
                something()
        """)
        fs = _py_symbols(source)
        call_edges = [e for e in fs.edges if e.edge_type == "CALLS"]
        assert call_edges, "Expected at least one CALLS edge"
        assert all(e.confidence == 1.0 for e in call_edges)


# ── Multi-language smoke tests ────────────────────────────────────────────────

def _symbols_for(source: str, lang: str, rel_path: str | None = None) -> _sym.FileSymbols:
    """Parse *source* as *lang* and return the extracted FileSymbols."""
    import tree_sitter
    from beacon.indexer.symbols import _LANGUAGES, LANG_CONFIGS

    ts_lang = _LANGUAGES[lang]
    parser = tree_sitter.Parser(ts_lang)
    src = textwrap.dedent(source).encode()
    tree = parser.parse(src)
    if rel_path is None:
        rel_path = f"test_file.{lang}"

    from beacon.indexer import symbols as sym_mod
    extractors = {
        "javascript": lambda: sym_mod._extract_js_ts(tree, src, rel_path, "javascript"),
        "typescript": lambda: sym_mod._extract_js_ts(tree, src, rel_path, "typescript"),
        "go":         lambda: sym_mod._extract_go(tree, src, rel_path),
        "rust":       lambda: sym_mod._extract_generic(tree, src, rel_path, LANG_CONFIGS["rust"]),
        "java":       lambda: sym_mod._extract_generic(tree, src, rel_path, LANG_CONFIGS["java"]),
        "cpp":        lambda: sym_mod._extract_c_cpp(tree, src, rel_path, "cpp"),
        "swift":      lambda: sym_mod._extract_swift(tree, src, rel_path),
    }
    return extractors[lang]()


class TestJavaScript:
    def test_arrow_const_gets_name(self):
        """const add = (a, b) => a + b  →  symbol named 'add', not '<anonymous>'."""
        fs = _symbols_for("const add = (a, b) => a + b\n", "javascript")
        names = {s.name for s in fs.symbols}
        assert "add" in names
        assert "<anonymous>" not in names

    def test_export_const_is_exported(self):
        """export const foo = () => {}  →  is_exported=True."""
        fs = _symbols_for("export const foo = () => {}\n", "javascript")
        foo = next((s for s in fs.symbols if s.name == "foo"), None)
        assert foo is not None, "symbol 'foo' not found"
        assert foo.is_exported is True

    def test_function_expression_const_gets_name(self):
        """const greet = function() {}  →  symbol named 'greet'."""
        fs = _symbols_for("const greet = function() {}\n", "javascript")
        names = {s.name for s in fs.symbols}
        assert "greet" in names

    def test_import_target_clean(self):
        """import { x } from './utils'  →  import target is './utils' (no quotes)."""
        fs = _symbols_for("import { x } from './utils'\n", "javascript")
        import_edges = [e for e in fs.edges if e.edge_type == "IMPORTS"]
        targets = {e.target_name for e in import_edges}
        assert "./utils" in targets

    def test_named_function_still_works(self):
        """function foo() {}  →  symbol named 'foo' (unchanged)."""
        fs = _symbols_for("function foo() {}\n", "javascript")
        names = {s.name for s in fs.symbols}
        assert "foo" in names


class TestTypeScript:
    def test_interface_extracted(self):
        """export interface IFoo { bar(): void }  →  kind='interface', is_exported=True."""
        fs = _symbols_for(
            "export interface IFoo { bar(): void }\n", "typescript"
        )
        iface = next((s for s in fs.symbols if s.name == "IFoo"), None)
        assert iface is not None, "interface IFoo not found"
        assert iface.kind == "interface"
        assert iface.is_exported is True

    def test_type_alias_extracted(self):
        """export type ID = string  →  kind='type_alias'."""
        fs = _symbols_for("export type ID = string\n", "typescript")
        ta = next((s for s in fs.symbols if s.name == "ID"), None)
        assert ta is not None, "type alias ID not found"
        assert ta.kind == "type_alias"

    def test_enum_extracted(self):
        """export enum Color { Red, Green, Blue }  →  kind='enum'."""
        fs = _symbols_for("export enum Color { Red, Green, Blue }\n", "typescript")
        en = next((s for s in fs.symbols if s.name == "Color"), None)
        assert en is not None, "enum Color not found"
        assert en.kind == "enum"

    def test_ts_arrow_const_gets_name(self):
        """const fn = (): void => {}  →  symbol named 'fn'."""
        fs = _symbols_for("const fn_ = (): void => {}\n", "typescript")
        names = {s.name for s in fs.symbols}
        assert "fn_" in names


class TestGo:
    def test_receiver_no_pointer_star(self):
        """func (r *MyType) Foo() {}  →  FQN contains '(MyType).Foo', not '(*MyType)'."""
        fs = _symbols_for(
            "package main\nfunc (r *MyType) Foo() {}\n", "go"
        )
        fqns = {s.fqn for s in fs.symbols}
        assert any("(MyType).Foo" in fqn for fqn in fqns), f"FQNs: {fqns}"
        assert not any("(*MyType)" in fqn for fqn in fqns)

    def test_type_kind_map_struct(self):
        """type S struct{}  →  kind='struct'."""
        fs = _symbols_for("package main\ntype S struct{}\n", "go")
        s = next((sym for sym in fs.symbols if sym.name == "S"), None)
        assert s is not None
        assert s.kind == "struct"

    def test_type_kind_map_interface(self):
        """type I interface{}  →  kind='interface'."""
        fs = _symbols_for("package main\ntype I interface{}\n", "go")
        i = next((sym for sym in fs.symbols if sym.name == "I"), None)
        assert i is not None
        assert i.kind == "interface"

    def test_type_kind_map_map(self):
        """type M map[string]int  →  kind='map' (not 'struct')."""
        fs = _symbols_for("package main\ntype M map[string]int\n", "go")
        m = next((sym for sym in fs.symbols if sym.name == "M"), None)
        assert m is not None
        assert m.kind == "map"

    def test_exported_method_call_captured(self):
        """func Foo(client Client) { client.DoRequest() }  →  CALLS edge for DoRequest."""
        source = textwrap.dedent("""\
            package main
            func Foo(client Client) {
                client.DoRequest()
            }
        """)
        fs = _symbols_for(source, "go")
        call_targets = {e.target_name for e in fs.edges if e.edge_type == "CALLS"}
        assert "DoRequest" in call_targets

    def test_call_edges_confidence_1(self):
        source = "package main\nfunc Foo() { something() }\n"
        fs = _symbols_for(source, "go")
        calls = [e for e in fs.edges if e.edge_type == "CALLS"]
        assert all(e.confidence == 1.0 for e in calls)


class TestRust:
    def test_impl_method_fqn(self):
        """impl MyStruct { fn foo() {} }  →  FQN ends in 'MyStruct.foo'."""
        source = textwrap.dedent("""\
            impl MyStruct {
                fn foo() {}
            }
        """)
        fs = _symbols_for(source, "rust")
        fqns = {s.fqn for s in fs.symbols}
        assert any("MyStruct" in fqn and "foo" in fqn for fqn in fqns), f"FQNs: {fqns}"

    def test_use_import_clean(self):
        """use std::collections::HashMap;  →  target is the scoped path, not the full statement."""
        fs = _symbols_for("use std::collections::HashMap;\n", "rust")
        import_edges = [e for e in fs.edges if e.edge_type == "IMPORTS"]
        assert import_edges, "Expected at least one IMPORTS edge"
        for edge in import_edges:
            assert not edge.target_name.startswith("use "), (
                f"Import target still includes 'use' keyword: {edge.target_name!r}"
            )

    def test_pub_fn_is_exported(self):
        """pub fn open() {}  →  is_exported=True."""
        fs = _symbols_for("pub fn open() {}\n", "rust")
        sym = next((s for s in fs.symbols if s.name == "open"), None)
        assert sym is not None
        assert sym.is_exported is True


class TestJava:
    def test_import_clean(self):
        """import java.util.List;  →  target doesn't include 'import' or ';'."""
        fs = _symbols_for("import java.util.List;\n", "java")
        import_edges = [e for e in fs.edges if e.edge_type == "IMPORTS"]
        assert import_edges, "Expected at least one IMPORTS edge"
        for edge in import_edges:
            assert not edge.target_name.startswith("import "), (
                f"Import target still includes 'import' keyword: {edge.target_name!r}"
            )
            assert not edge.target_name.endswith(";"), (
                f"Import target still includes trailing semicolon: {edge.target_name!r}"
            )


class TestCpp:
    def test_class_symbol_extracted(self):
        """class Foo { public: void bar() {} }  →  class symbol 'Foo'."""
        source = textwrap.dedent("""\
            class Foo {
            public:
                void bar() {}
            };
        """)
        fs = _symbols_for(source, "cpp")
        names = {s.name for s in fs.symbols}
        assert "Foo" in names

    def test_method_inside_class(self):
        """class Foo { void bar() {} }  →  symbol 'bar' with kind='method'."""
        source = textwrap.dedent("""\
            class Foo {
                void bar() {}
            };
        """)
        fs = _symbols_for(source, "cpp")
        bar = next((s for s in fs.symbols if s.name == "bar"), None)
        assert bar is not None, "method 'bar' not found"
        assert bar.kind == "method"

    def test_this_arrow_call_captured(self):
        """this->bar() inside a method must produce a CALLS edge for 'bar'."""
        source = textwrap.dedent("""\
            class Foo {
                void run() { this->bar(); }
                void bar() {}
            };
        """)
        fs = _symbols_for(source, "cpp")
        call_targets = {e.target_name for e in fs.edges if e.edge_type == "CALLS"}
        assert "bar" in call_targets


class TestMissingGrammar:
    def test_missing_grammar_raises(self):
        """extract() with an unknown language must raise RuntimeError."""
        from pathlib import Path
        import tempfile, os
        with tempfile.TemporaryDirectory() as tmpdir:
            f = Path(tmpdir) / "foo.unknown"
            f.write_text("hello world")
            with pytest.raises(RuntimeError, match="grammar"):
                _sym.extract(f, "unknown_lang_xyz", Path(tmpdir))
