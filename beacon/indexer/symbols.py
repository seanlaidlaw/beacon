"""
Symbol and call-edge extraction via tree-sitter.

Extracts:
  - Symbol nodes (functions, methods, classes, etc.) with FQN, signature, docstring
  - CALLS edges (tree-sitter AST + regex fallback)
  - IMPORTS edges
  - CONTAINS edges (file → symbol, class → method)

FQN format: <relative_file_path>::<qualified_name>
e.g. src/foo.py::MyClass.my_method
"""

import re
from dataclasses import dataclass, field
from pathlib import Path

import tree_sitter_python as tspython
import tree_sitter_javascript as tsjs
import tree_sitter_typescript as tsts
import tree_sitter_go as tsgo
import tree_sitter_rust as tsrust
import tree_sitter_java as tsjava
import tree_sitter_c as tsc
import tree_sitter_cpp as tscpp
import tree_sitter_bash as tsbash
import tree_sitter_lua as tslua
import tree_sitter_swift as tsswift
import tree_sitter_markdown as tsmarkdown
from tree_sitter import Language, Parser, Node

# ── Language setup ────────────────────────────────────────────────────────────

def _r_language() -> Language | None:
    """Load the R grammar compiled from source, or return None if unavailable."""
    import ctypes
    from pathlib import Path
    r_so = Path.home() / ".cache" / "beacon" / "grammars" / "r.so"
    if not r_so.exists():
        return None
    try:
        lib = ctypes.cdll.LoadLibrary(str(r_so))
        lib.tree_sitter_r.restype = ctypes.c_void_p
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            return Language(lib.tree_sitter_r())
    except Exception:
        return None


_LANGUAGES: dict[str, Language] = {
    "python":     Language(tspython.language()),
    "javascript": Language(tsjs.language()),
    "typescript": Language(tsts.language_typescript()),
    "go":         Language(tsgo.language()),
    "rust":       Language(tsrust.language()),
    "java":       Language(tsjava.language()),
    "c":          Language(tsc.language()),
    "cpp":        Language(tscpp.language()),
    "bash":       Language(tsbash.language()),
    "lua":        Language(tslua.language()),
    "swift":      Language(tsswift.language()),
    "markdown":   Language(tsmarkdown.language()),
}

# Aliases
_LANGUAGES["tsx"] = _LANGUAGES["typescript"]
_LANGUAGES["jsx"] = _LANGUAGES["javascript"]

# R — loaded lazily (requires compiled .so)
_r_lang = _r_language()
if _r_lang:
    _LANGUAGES["r"] = _r_lang
    _LANGUAGES["rmarkdown"] = _r_lang   # extract R blocks from Rmd
    _LANGUAGES["quarto"]    = _r_lang   # same for .qmd


def _parser(lang: str) -> Parser:
    p = Parser(_LANGUAGES[lang])
    return p


# ── Data types ────────────────────────────────────────────────────────────────

@dataclass
class Symbol:
    name: str
    fqn: str
    file_path: str
    kind: str          # function, method, class, interface, ...
    start_line: int
    end_line: int
    signature: str = ""
    docstring: str = ""
    body_preview: str = ""   # first ~20 lines of function body for semantic embedding
    is_exported: bool = False
    is_test: bool = False


@dataclass
class CallEdge:
    source_fqn: str
    target_name: str   # unresolved name; resolved to FQN later
    call_site_line: int
    edge_type: str = "CALLS"   # CALLS, IMPORTS, CONTAINS
    confidence: float = 0.8    # regex-derived; tree-sitter = 1.0


@dataclass
class FileSymbols:
    symbols: list[Symbol] = field(default_factory=list)
    edges: list[CallEdge] = field(default_factory=list)


# ── Regex fallback patterns (from vexp-core strings) ─────────────────────────

_RE_METHOD_CALL    = re.compile(r"(?:self|this)\.(\w+)\s*\(")
_RE_FUNC_CALL      = re.compile(r"(?:^|[^.\w])([a-z_]\w{2,})\s*\(")
_RE_OBJ_METHOD     = re.compile(r"(?:^|[^.\w])(\w+)\.([a-z_]\w*)\s*\(")

# Stdlib/builtin names to skip as call targets
_SKIP_CALLS = frozenset({
    "print", "len", "range", "isinstance", "type", "str", "int", "float",
    "list", "dict", "set", "tuple", "bool", "open", "super", "object",
    "max", "min", "sum", "abs", "zip", "map", "filter", "enumerate",
    "hasattr", "getattr", "setattr", "format", "repr", "id",
    "console", "Promise", "Array", "Object", "String", "Number", "Math",
    "printf", "fprintf", "sprintf", "malloc", "calloc", "free", "sizeof",
    "assert", "panic", "make", "append", "copy", "delete", "new",
})


def _node_text(node: Node, src: bytes) -> str:
    return src[node.start_byte:node.end_byte].decode("utf-8", errors="replace")


def _first_child_text(node: Node, src: bytes, *kinds: str) -> str:
    for child in node.children:
        if child.type in kinds:
            return _node_text(child, src)
    return ""


# ── Language-specific extractors ─────────────────────────────────────────────

def _extract_python(tree, src: bytes, rel_path: str) -> FileSymbols:
    result = FileSymbols()
    scope_stack: list[str] = []

    def qualified(name: str) -> str:
        parts = scope_stack + [name]
        return f"{rel_path}::{'.'.join(parts)}"

    def visit(node: Node, parent_fqn: str | None = None):
        if node.type in ("function_definition", "async_function_definition"):
            name_node = node.child_by_field_name("name")
            if not name_node:
                return
            name = _node_text(name_node, src)
            kind = "method" if scope_stack else "function"
            fqn = qualified(name)
            sig = _node_text(node.child_by_field_name("parameters") or name_node, src)
            doc = _extract_python_docstring(node, src)
            preview = _body_preview(node, src)
            exported = not name.startswith("_")
            is_test = name.startswith("test_") or "test" in rel_path.lower()
            sym = Symbol(name, fqn, rel_path, kind,
                         node.start_point[0] + 1, node.end_point[0] + 1,
                         signature=sig, docstring=doc, body_preview=preview,
                         is_exported=exported, is_test=is_test)
            result.symbols.append(sym)
            if parent_fqn:
                result.edges.append(CallEdge(parent_fqn, fqn, node.start_point[0] + 1, "CONTAINS", 1.0))
            scope_stack.append(name)
            for child in node.children:
                visit(child, fqn)
            scope_stack.pop()
            _regex_calls(node, src, fqn, result)

        elif node.type == "class_definition":
            name_node = node.child_by_field_name("name")
            if not name_node:
                return
            name = _node_text(name_node, src)
            fqn = qualified(name)
            doc = _extract_python_docstring(node, src)
            sym = Symbol(name, fqn, rel_path, "class",
                         node.start_point[0] + 1, node.end_point[0] + 1,
                         docstring=doc, is_exported=not name.startswith("_"))
            result.symbols.append(sym)
            if parent_fqn:
                result.edges.append(CallEdge(parent_fqn, fqn, node.start_point[0] + 1, "CONTAINS", 1.0))
            scope_stack.append(name)
            for child in node.children:
                visit(child, fqn)
            scope_stack.pop()

        elif node.type == "import_statement":
            for child in node.children:
                if child.type in ("dotted_name", "identifier"):
                    target = _node_text(child, src)
                    container = qualified(scope_stack[-1]) if scope_stack else f"{rel_path}::__module__"
                    result.edges.append(CallEdge(container, target, node.start_point[0] + 1, "IMPORTS", 0.9))
        elif node.type == "import_from_statement":
            module_node = node.child_by_field_name("module_name")
            module = _node_text(module_node, src) if module_node else ""
            for child in node.children:
                if child.type == "import_from_statement":
                    continue
                if child.type in ("dotted_name", "identifier") and child != module_node:
                    target = f"{module}.{_node_text(child, src)}" if module else _node_text(child, src)
                    container = qualified(scope_stack[-1]) if scope_stack else f"{rel_path}::__module__"
                    result.edges.append(CallEdge(container, target, node.start_point[0] + 1, "IMPORTS", 0.9))
        else:
            for child in node.children:
                visit(child, parent_fqn)

    visit(tree.root_node)
    return result


def _extract_python_docstring(node: Node, src: bytes) -> str:
    # First statement in body that is a string literal
    body = node.child_by_field_name("body")
    if not body:
        return ""
    for child in body.children:
        if child.type == "expression_statement":
            for grandchild in child.children:
                if grandchild.type == "string":
                    raw = _node_text(grandchild, src).strip("\"'").strip()
                    return raw[:512]
    return ""


def _body_preview(node: Node, src: bytes, max_lines: int = 20, max_chars: int = 800) -> str:
    """Extract the first *max_lines* lines of a function/method body for embedding.

    Uses tree-sitter to skip the leading docstring node (if present), then
    takes raw source text of the remaining statements. Strips common
    indentation and caps at *max_chars* to keep embedding text compact.
    """
    body = node.child_by_field_name("body")
    if not body:
        return ""

    # Find the byte offset past the docstring (first expression_statement
    # containing a string literal), if one exists.
    start_byte = body.start_byte
    for child in body.children:
        if child.type == "expression_statement":
            for gc in child.children:
                if gc.type == "string":
                    start_byte = child.end_byte  # skip past the docstring
                    break
            break  # only check the first statement

    body_text = src[start_byte:body.end_byte].decode("utf-8", errors="replace")
    lines = [l for l in body_text.splitlines() if l.strip()]  # drop blank lines
    preview = "\n".join(lines[:max_lines])
    if preview:
        import textwrap
        preview = textwrap.dedent(preview).strip()
    return preview[:max_chars]


def _extract_js_ts(tree, src: bytes, rel_path: str, lang: str) -> FileSymbols:
    result = FileSymbols()
    scope_stack: list[str] = []

    def qualified(name: str) -> str:
        return f"{rel_path}::{'.'.join(scope_stack + [name])}"

    FUNC_NODES = {
        "function_declaration", "function_expression", "arrow_function",
        "method_definition", "generator_function_declaration",
    }
    CLASS_NODES = {"class_declaration", "class_expression"}

    def visit(node: Node, parent_fqn: str | None = None):
        if node.type in FUNC_NODES:
            name_node = node.child_by_field_name("name")
            name = _node_text(name_node, src) if name_node else "<anonymous>"
            kind = "method" if scope_stack else "function"
            fqn = qualified(name)
            params = node.child_by_field_name("parameters")
            sig = _node_text(params, src) if params else ""
            exported = _is_js_exported(node)
            sym = Symbol(name, fqn, rel_path, kind,
                         node.start_point[0] + 1, node.end_point[0] + 1,
                         signature=sig, is_exported=exported,
                         is_test="test" in rel_path.lower() or "spec" in rel_path.lower())
            result.symbols.append(sym)
            if parent_fqn:
                result.edges.append(CallEdge(parent_fqn, fqn, node.start_point[0] + 1, "CONTAINS", 1.0))
            scope_stack.append(name)
            for child in node.children:
                visit(child, fqn)
            scope_stack.pop()
            _regex_calls(node, src, fqn, result)

        elif node.type in CLASS_NODES:
            name_node = node.child_by_field_name("name")
            name = _node_text(name_node, src) if name_node else "<anonymous>"
            fqn = qualified(name)
            exported = _is_js_exported(node)
            sym = Symbol(name, fqn, rel_path, "class",
                         node.start_point[0] + 1, node.end_point[0] + 1,
                         is_exported=exported)
            result.symbols.append(sym)
            if parent_fqn:
                result.edges.append(CallEdge(parent_fqn, fqn, node.start_point[0] + 1, "CONTAINS", 1.0))
            scope_stack.append(name)
            for child in node.children:
                visit(child, fqn)
            scope_stack.pop()

        elif node.type == "import_statement":
            _js_imports(node, src, rel_path, scope_stack, result)
        else:
            for child in node.children:
                visit(child, parent_fqn)

    visit(tree.root_node)
    return result


def _is_js_exported(node: Node) -> bool:
    p = node.parent
    return p is not None and p.type in ("export_statement", "export_named_declaration")


def _js_imports(node: Node, src: bytes, rel_path: str, scope_stack: list, result: FileSymbols):
    container = f"{rel_path}::__module__"
    for child in node.children:
        if child.type == "string":
            target = _node_text(child, src).strip("'\"")
            result.edges.append(CallEdge(container, target, node.start_point[0] + 1, "IMPORTS", 0.9))


def _extract_go(tree, src: bytes, rel_path: str) -> FileSymbols:
    result = FileSymbols()

    def fqn(name: str, receiver: str = "") -> str:
        if receiver:
            return f"{rel_path}::({receiver}).{name}"
        return f"{rel_path}::{name}"

    def visit(node: Node):
        if node.type == "function_declaration":
            name_node = node.child_by_field_name("name")
            if not name_node:
                return
            name = _node_text(name_node, src)
            params = node.child_by_field_name("parameters")
            sig = _node_text(params, src) if params else ""
            f = fqn(name)
            sym = Symbol(name, f, rel_path, "function",
                         node.start_point[0] + 1, node.end_point[0] + 1, signature=sig)
            result.symbols.append(sym)
            _regex_calls(node, src, f, result)

        elif node.type == "method_declaration":
            name_node = node.child_by_field_name("name")
            recv_node = node.child_by_field_name("receiver")
            if not name_node:
                return
            name = _node_text(name_node, src)
            recv = _node_text(recv_node, src).strip("()").split()[-1] if recv_node else ""
            f = fqn(name, recv)
            params = node.child_by_field_name("parameters")
            sig = _node_text(params, src) if params else ""
            sym = Symbol(name, f, rel_path, "method",
                         node.start_point[0] + 1, node.end_point[0] + 1, signature=sig)
            result.symbols.append(sym)
            _regex_calls(node, src, f, result)

        elif node.type == "type_declaration":
            for child in node.children:
                if child.type == "type_spec":
                    name_node = child.child_by_field_name("name")
                    type_node = child.child_by_field_name("type")
                    if name_node:
                        name = _node_text(name_node, src)
                        kind = "interface" if type_node and "interface" in type_node.type else "struct"
                        f = fqn(name)
                        result.symbols.append(Symbol(name, f, rel_path, kind,
                                                     node.start_point[0] + 1, node.end_point[0] + 1))

        else:
            for child in node.children:
                visit(child)

    visit(tree.root_node)
    return result


def _extract_rust(tree, src: bytes, rel_path: str) -> FileSymbols:
    result = FileSymbols()
    scope_stack: list[str] = []

    def qualified(name: str) -> str:
        return f"{rel_path}::{'.'.join(scope_stack + [name])}"

    def visit(node: Node, parent_fqn: str | None = None):
        if node.type == "function_item":
            name_node = node.child_by_field_name("name")
            if not name_node:
                return
            name = _node_text(name_node, src)
            fqn = qualified(name)
            params = node.child_by_field_name("parameters")
            sig = _node_text(params, src) if params else ""
            sym = Symbol(name, fqn, rel_path, "function",
                         node.start_point[0] + 1, node.end_point[0] + 1, signature=sig)
            result.symbols.append(sym)
            scope_stack.append(name)
            for child in node.children:
                visit(child, fqn)
            scope_stack.pop()
            _regex_calls(node, src, fqn, result)

        elif node.type in ("impl_item", "trait_item", "struct_item", "enum_item"):
            name_node = node.child_by_field_name("name")
            if not name_node:
                for child in node.children:
                    visit(child, parent_fqn)
                return
            name = _node_text(name_node, src)
            fqn = qualified(name)
            kind = node.type.replace("_item", "")
            result.symbols.append(Symbol(name, fqn, rel_path, kind,
                                         node.start_point[0] + 1, node.end_point[0] + 1))
            scope_stack.append(name)
            for child in node.children:
                visit(child, fqn)
            scope_stack.pop()

        elif node.type == "use_declaration":
            target = _node_text(node, src).replace("use ", "").rstrip(";").strip()
            container = parent_fqn or f"{rel_path}::__module__"
            result.edges.append(CallEdge(container, target, node.start_point[0] + 1, "IMPORTS", 0.9))
        else:
            for child in node.children:
                visit(child, parent_fqn)

    visit(tree.root_node)
    return result


def _extract_java(tree, src: bytes, rel_path: str) -> FileSymbols:
    result = FileSymbols()
    scope_stack: list[str] = []

    def qualified(name: str) -> str:
        return f"{rel_path}::{'.'.join(scope_stack + [name])}"

    def visit(node: Node, parent_fqn: str | None = None):
        if node.type in ("class_declaration", "interface_declaration", "enum_declaration"):
            name_node = node.child_by_field_name("name")
            if not name_node:
                return
            name = _node_text(name_node, src)
            fqn = qualified(name)
            kind = node.type.replace("_declaration", "")
            result.symbols.append(Symbol(name, fqn, rel_path, kind,
                                         node.start_point[0] + 1, node.end_point[0] + 1))
            scope_stack.append(name)
            for child in node.children:
                visit(child, fqn)
            scope_stack.pop()

        elif node.type == "method_declaration":
            name_node = node.child_by_field_name("name")
            if not name_node:
                return
            name = _node_text(name_node, src)
            fqn = qualified(name)
            params = node.child_by_field_name("formal_parameters")
            sig = _node_text(params, src) if params else ""
            result.symbols.append(Symbol(name, fqn, rel_path, "method",
                                         node.start_point[0] + 1, node.end_point[0] + 1,
                                         signature=sig))
            scope_stack.append(name)
            for child in node.children:
                visit(child, fqn)
            scope_stack.pop()
            _regex_calls(node, src, fqn, result)
        else:
            for child in node.children:
                visit(child, parent_fqn)

    visit(tree.root_node)
    return result


def _extract_c_cpp(tree, src: bytes, rel_path: str, lang: str) -> FileSymbols:
    result = FileSymbols()

    def visit(node: Node):
        if node.type == "function_definition":
            decl = node.child_by_field_name("declarator")
            if not decl:
                return
            # Drill into pointer_declarator / function_declarator
            while decl and decl.type in ("pointer_declarator", "reference_declarator"):
                decl = decl.child_by_field_name("declarator")
            if decl and decl.type == "function_declarator":
                name_node = decl.child_by_field_name("declarator")
                name = _node_text(name_node, src) if name_node else "<unknown>"
                fqn = f"{rel_path}::{name}"
                params = decl.child_by_field_name("parameters")
                sig = _node_text(params, src) if params else ""
                result.symbols.append(Symbol(name, fqn, rel_path, "function",
                                             node.start_point[0] + 1, node.end_point[0] + 1,
                                             signature=sig))
                _regex_calls(node, src, fqn, result)
        else:
            for child in node.children:
                visit(child)

    visit(tree.root_node)
    return result


# ── Regex-based call detection (fallback, confidence 0.7) ────────────────────

def _regex_calls(node: Node, src: bytes, source_fqn: str, result: FileSymbols):
    text = _node_text(node, src)
    line_base = node.start_point[0] + 1

    for m in _RE_METHOD_CALL.finditer(text):
        name = m.group(1)
        if name not in _SKIP_CALLS:
            line = line_base + text[:m.start()].count("\n")
            result.edges.append(CallEdge(source_fqn, name, line, "CALLS", 0.7))

    for m in _RE_OBJ_METHOD.finditer(text):
        name = m.group(2)
        if name not in _SKIP_CALLS:
            line = line_base + text[:m.start()].count("\n")
            result.edges.append(CallEdge(source_fqn, name, line, "CALLS", 0.7))


# ── Bash extractor ────────────────────────────────────────────────────────────

def _extract_bash(tree, src: bytes, rel_path: str) -> FileSymbols:
    result = FileSymbols()

    def visit(node: Node):
        if node.type == "function_definition":
            name_node = node.child_by_field_name("name")
            if not name_node:
                return
            name = _node_text(name_node, src)
            fqn = f"{rel_path}::{name}"
            body = node.child_by_field_name("body")
            sig = f"{name}()"
            result.symbols.append(Symbol(name, fqn, rel_path, "function",
                                         node.start_point[0] + 1, node.end_point[0] + 1,
                                         signature=sig))
            _regex_calls(node, src, fqn, result)
        else:
            for child in node.children:
                visit(child)

    visit(tree.root_node)
    return result


# ── Lua extractor ─────────────────────────────────────────────────────────────

def _extract_lua(tree, src: bytes, rel_path: str) -> FileSymbols:
    result = FileSymbols()
    scope_stack: list[str] = []

    def qualified(name: str) -> str:
        return f"{rel_path}::{'.'.join(scope_stack + [name])}"

    def visit(node: Node):
        if node.type in ("function_declaration", "local_function"):
            name_node = node.child_by_field_name("name")
            if not name_node:
                return
            name = _node_text(name_node, src)
            fqn = qualified(name)
            params = node.child_by_field_name("parameters")
            sig = _node_text(params, src) if params else ""
            result.symbols.append(Symbol(name, fqn, rel_path, "function",
                                         node.start_point[0] + 1, node.end_point[0] + 1,
                                         signature=sig))
            scope_stack.append(name)
            for child in node.children:
                visit(child)
            scope_stack.pop()
            _regex_calls(node, src, fqn, result)
        elif node.type == "function_statement":
            # method-style: MyClass.myMethod = function(...)
            name_node = node.child_by_field_name("name")
            name = _node_text(name_node, src) if name_node else "<anon>"
            fqn = qualified(name)
            params = node.child_by_field_name("parameters")
            sig = _node_text(params, src) if params else ""
            result.symbols.append(Symbol(name, fqn, rel_path, "method",
                                         node.start_point[0] + 1, node.end_point[0] + 1,
                                         signature=sig))
            _regex_calls(node, src, fqn, result)
        else:
            for child in node.children:
                visit(child)

    visit(tree.root_node)
    return result


# ── Swift extractor ───────────────────────────────────────────────────────────

def _extract_swift(tree, src: bytes, rel_path: str) -> FileSymbols:
    result = FileSymbols()
    scope_stack: list[str] = []

    def qualified(name: str) -> str:
        return f"{rel_path}::{'.'.join(scope_stack + [name])}"

    FUNC_NODES = {"function_declaration", "init_declaration", "deinit_declaration"}
    TYPE_NODES = {"class_declaration", "struct_declaration", "enum_declaration",
                  "protocol_declaration", "extension_declaration"}

    def visit(node: Node, parent_fqn: str | None = None):
        if node.type in FUNC_NODES:
            name_node = node.child_by_field_name("name")
            name = _node_text(name_node, src) if name_node else "<init>"
            fqn = qualified(name)
            params = node.child_by_field_name("params")
            sig = _node_text(params, src) if params else ""
            kind = "method" if scope_stack else "function"
            result.symbols.append(Symbol(name, fqn, rel_path, kind,
                                         node.start_point[0] + 1, node.end_point[0] + 1,
                                         signature=sig))
            scope_stack.append(name)
            for child in node.children:
                visit(child, fqn)
            scope_stack.pop()
            _regex_calls(node, src, fqn, result)
        elif node.type in TYPE_NODES:
            name_node = node.child_by_field_name("name")
            if not name_node:
                return
            name = _node_text(name_node, src)
            fqn = qualified(name)
            kind = node.type.replace("_declaration", "")
            result.symbols.append(Symbol(name, fqn, rel_path, kind,
                                         node.start_point[0] + 1, node.end_point[0] + 1))
            scope_stack.append(name)
            for child in node.children:
                visit(child, fqn)
            scope_stack.pop()
        else:
            for child in node.children:
                visit(child, parent_fqn)

    visit(tree.root_node)
    return result


# ── R extractor ───────────────────────────────────────────────────────────────

def _extract_r(tree, src: bytes, rel_path: str) -> FileSymbols:
    """
    Extract R function assignments: name <- function(...) { ... }
    Also handles: name = function(...), setGeneric, setMethod, R6Class.
    """
    result = FileSymbols()

    def visit(node: Node):
        # R uses binary_operator for <- and =
        if node.type == "binary_operator":
            children = node.children
            # Find the operator
            ops = [c for c in children if c.type in ("<-", "=", "<<-")]
            if not ops:
                for child in node.children:
                    visit(child)
                return
            lhs = node.child_by_field_name("lhs") or (children[0] if children else None)
            rhs = node.child_by_field_name("rhs") or (children[-1] if children else None)
            if lhs and rhs and rhs.type == "function_definition":
                name = _node_text(lhs, src).strip()
                if re.match(r'^[A-Za-z._][A-Za-z0-9._]*$', name):
                    fqn = f"{rel_path}::{name}"
                    params_node = rhs.child_by_field_name("parameters")
                    sig = _node_text(params_node, src) if params_node else ""
                    result.symbols.append(Symbol(name, fqn, rel_path, "function",
                                                 node.start_point[0] + 1, node.end_point[0] + 1,
                                                 signature=sig, is_exported=not name.startswith(".")))
                    _regex_calls(rhs, src, fqn, result)
                    return
        for child in node.children:
            visit(child)

    visit(tree.root_node)
    return result


# ── Markdown / Rmd / Qmd extractor ───────────────────────────────────────────

def _extract_markdown(tree, src: bytes, rel_path: str) -> FileSymbols:
    """
    Extract headings and named code chunks from Markdown / Rmd / Qmd files.

    Headings are only extracted from the Markdown body — lines inside fenced
    code blocks (``` ... ```) are skipped so that R/Python comments starting
    with `#` are not mistakenly treated as headings.
    """
    result = FileSymbols()
    text = src.decode("utf-8", errors="replace")
    lines = text.splitlines()

    # Build a set of line numbers (0-based) that are inside fenced code blocks
    inside_fence: set[int] = set()
    in_block = False
    fence_char = ""
    for i, line in enumerate(lines):
        stripped = line.strip()
        if not in_block:
            # Opening fence: ``` or ~~~, optionally with language tag
            if stripped.startswith("```") or stripped.startswith("~~~"):
                fence_char = stripped[:3]
                in_block = True
                inside_fence.add(i)
        else:
            inside_fence.add(i)
            # Closing fence: line is exactly the fence chars (no language tag)
            if stripped.startswith(fence_char) and stripped == stripped[:3]:
                in_block = False

    # Extract ATX headings only from lines outside code blocks
    for i, line in enumerate(lines):
        if i in inside_fence:
            continue
        m = re.match(r'^(#{1,6})\s+(.+)$', line)
        if not m:
            continue
        level = len(m.group(1))
        title = m.group(2).strip()
        name = re.sub(r'[^A-Za-z0-9_\s]', '', title)[:60].strip().replace(' ', '_')
        if not name:
            continue
        fqn = f"{rel_path}::{name}"
        result.symbols.append(Symbol(title, fqn, rel_path, f"heading{level}",
                                     i + 1, i + 1, signature=f"{'#' * level} {title}"))

    # Extract named code chunks: ```{r chunk_name} or ```{python chunk_name}
    for m in re.finditer(r'^```\{(\w+)\s+([A-Za-z0-9_.-]+)', text, re.MULTILINE):
        lang_tag = m.group(1)
        chunk_name = m.group(2)
        fqn = f"{rel_path}::{lang_tag}_{chunk_name}"
        line = text[:m.start()].count('\n') + 1
        result.symbols.append(Symbol(chunk_name, fqn, rel_path, "chunk",
                                     line, line, signature=f"```{{{lang_tag} {chunk_name}}}"))

    return result


# ── Public API ────────────────────────────────────────────────────────────────

def extract(file_path: Path, lang: str, root: Path) -> FileSymbols:
    """
    Parse *file_path* and return its symbols and edges.
    *root* is the repo root used to compute relative paths for FQNs.
    """
    if lang not in _LANGUAGES:
        return FileSymbols()

    src = file_path.read_bytes()
    rel_path = str(file_path.relative_to(root))
    parser = _parser(lang)
    tree = parser.parse(src)

    if lang == "python":
        return _extract_python(tree, src, rel_path)
    elif lang in ("javascript", "typescript", "tsx", "jsx"):
        return _extract_js_ts(tree, src, rel_path, lang)
    elif lang == "go":
        return _extract_go(tree, src, rel_path)
    elif lang == "rust":
        return _extract_rust(tree, src, rel_path)
    elif lang == "java":
        return _extract_java(tree, src, rel_path)
    elif lang in ("c", "cpp"):
        return _extract_c_cpp(tree, src, rel_path, lang)
    elif lang == "bash":
        return _extract_bash(tree, src, rel_path)
    elif lang == "lua":
        return _extract_lua(tree, src, rel_path)
    elif lang == "swift":
        return _extract_swift(tree, src, rel_path)
    elif lang == "r":
        return _extract_r(tree, src, rel_path)
    elif lang in ("markdown", "rmarkdown", "quarto"):
        # Rmd/Qmd are markdown-first: extract headings + named code chunks
        return _extract_markdown(tree, src, rel_path)
    else:
        return FileSymbols()
