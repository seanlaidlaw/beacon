"""
Symbol and call-edge extraction via tree-sitter.

Extracts:
  - Symbol nodes (functions, methods, classes, etc.) with FQN, signature, docstring
  - CALLS edges (tree-sitter AST only — no regex fallbacks)
  - IMPORTS edges
  - CONTAINS edges (file → symbol, class → method)

FQN format: <relative_file_path>::<qualified_name>
e.g. src/foo.py::MyClass.my_method
"""

import re
import textwrap
from dataclasses import dataclass, field, field as dc_field
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
    confidence: float = 1.0    # tree-sitter AST = 1.0


@dataclass
class FileSymbols:
    symbols: list[Symbol] = field(default_factory=list)
    edges: list[CallEdge] = field(default_factory=list)


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


# ── LangConfig dataclass ──────────────────────────────────────────────────────

@dataclass
class LangConfig:
    # Tree-sitter node types that map to functions/methods
    func_types: frozenset
    # Tree-sitter node types that map to classes/structs/enums/etc
    type_types: frozenset
    # Field names to try for extracting the symbol's name (in priority order)
    name_fields: tuple = ("name",)
    # Field names to try for parameters
    param_fields: tuple = ("parameters", "params", "formal_parameters")
    # Field names to try for function/method body (in priority order)
    body_fields: tuple = ("body", "block", "compound_statement", "function_body", "statements")
    # Field name for return type annotation (None = try arrow detection)
    return_type_field: str | None = None
    # Comment node types that precede declarations as doc comments
    comment_types: frozenset = dc_field(default_factory=lambda: frozenset({
        "line_comment", "block_comment", "comment", "multiline_comment", "doc_comment"
    }))
    # Only accept comments that start with these strings as docstrings
    # (empty tuple = accept all comment types)
    doc_comment_prefixes: tuple = ()
    # sibling/parent keyword tokens that mean this symbol is exported
    export_keywords: frozenset = dc_field(default_factory=frozenset)
    # If True, export is determined by whether the name starts with uppercase (Go)
    export_by_capitalization: bool = False
    # Name prefixes that mark test functions (in addition to "test" in file path)
    test_name_prefixes: tuple = ("test_", "Test")
    # How to map node.type → kind string (applied after stripping _declaration/_item/_definition)
    kind_overrides: dict = dc_field(default_factory=dict)
    # Whether method_kind is "method" when inside a type scope (True for most languages)
    use_scope_for_method_kind: bool = True
    # Import node types to detect (maps node_type -> field name or None for full text)
    import_nodes: dict = dc_field(default_factory=dict)
    # ── Call extraction via tree-sitter AST ───────────────────────────────────
    # Node types that represent a function/method call in this language's AST
    call_node_types: frozenset = dc_field(default_factory=lambda: frozenset({"call_expression"}))
    # Ordered list of field names to try for the callee of a call node
    call_function_fields: tuple = ("function",)
    # Extra node types (beyond func_types) that should stop recursion during the
    # call walk — prevents attributing inner-lambda calls to the outer function
    extra_nested_fn_types: frozenset = dc_field(default_factory=frozenset)
    # ── Declarator hoisting (e.g. const foo = () => {}) ──────────────────────
    # Node types where an anonymous function value should inherit the declarator name
    declarator_node_types: frozenset = dc_field(default_factory=frozenset)
    declarator_name_field: str = "name"
    declarator_value_field: str = "value"
    # ── Export via ancestor walk ──────────────────────────────────────────────
    # Ancestor node types that indicate the symbol is exported (walked up to
    # export_walk_depth hops from the symbol node)
    export_ancestor_types: frozenset = dc_field(default_factory=frozenset)
    export_walk_depth: int = 4
    # ── Import target extraction ──────────────────────────────────────────────
    # When resolving the target of an import node, walk into children whose type
    # is in this set to extract the clean module string (avoids returning the
    # full raw statement text including keywords and semicolons)
    import_target_child_types: frozenset = dc_field(default_factory=lambda: frozenset({
        "string_fragment", "scoped_identifier", "scoped_use_list",
        "dotted_name", "identifier",
    }))


# ── Per-language configs ──────────────────────────────────────────────────────

LANG_CONFIGS: dict[str, LangConfig] = {
    "python": LangConfig(
        func_types=frozenset({"function_definition", "async_function_definition"}),
        type_types=frozenset({"class_definition"}),
        body_fields=("body",),
        return_type_field="return_type",
        test_name_prefixes=("test_",),
        comment_types=frozenset({"comment"}),
        call_node_types=frozenset({"call"}),
        call_function_fields=("function",),
        extra_nested_fn_types=frozenset({"lambda"}),
    ),
    "javascript": LangConfig(
        func_types=frozenset({
            "function_declaration", "function_expression", "arrow_function",
            "method_definition", "generator_function_declaration",
        }),
        type_types=frozenset({"class_declaration", "class_expression"}),
        body_fields=("body",),
        return_type_field="return_type",
        export_keywords=frozenset({"export"}),
        test_name_prefixes=("test", "it", "describe"),
        doc_comment_prefixes=("/**", "//"),
        comment_types=frozenset({"comment"}),
        import_nodes={"import_statement": None},
        call_node_types=frozenset({"call_expression", "new_expression"}),
        call_function_fields=("function", "constructor"),
        declarator_node_types=frozenset({"variable_declarator"}),
        export_ancestor_types=frozenset({"export_statement"}),
    ),
    "typescript": LangConfig(
        func_types=frozenset({
            "function_declaration", "function_expression", "arrow_function",
            "method_definition", "generator_function_declaration",
        }),
        type_types=frozenset({
            "class_declaration", "class_expression",
            "interface_declaration", "type_alias_declaration", "enum_declaration",
        }),
        body_fields=("body",),
        return_type_field="return_type",
        export_keywords=frozenset({"export"}),
        test_name_prefixes=("test", "it", "describe"),
        doc_comment_prefixes=("/**", "//"),
        comment_types=frozenset({"comment"}),
        import_nodes={"import_statement": None},
        kind_overrides={
            "interface_declaration": "interface",
            "type_alias_declaration": "type_alias",
            "enum_declaration": "enum",
        },
        call_node_types=frozenset({"call_expression", "new_expression"}),
        call_function_fields=("function", "constructor"),
        declarator_node_types=frozenset({"variable_declarator"}),
        export_ancestor_types=frozenset({"export_statement"}),
    ),
    "go": LangConfig(
        func_types=frozenset({"function_declaration", "method_declaration"}),
        type_types=frozenset({"type_declaration", "type_spec"}),
        name_fields=("name",),
        param_fields=("parameters",),
        body_fields=("body",),
        return_type_field="result",
        export_by_capitalization=True,
        test_name_prefixes=("Test", "Benchmark", "Example"),
        doc_comment_prefixes=("//",),
        comment_types=frozenset({"comment"}),
        import_nodes={"import_declaration": "path", "import_spec": "path"},
        call_node_types=frozenset({"call_expression"}),
        call_function_fields=("function",),
        extra_nested_fn_types=frozenset({"func_literal"}),
    ),
    "rust": LangConfig(
        func_types=frozenset({"function_item"}),
        type_types=frozenset({"struct_item", "enum_item", "impl_item", "trait_item", "type_item"}),
        # Try "name" first (works for struct_item/enum_item/trait_item/function_item/type_item),
        # then "type" (works for impl_item which has no "name" field but has a "type" field).
        name_fields=("name", "type"),
        body_fields=("body",),
        return_type_field="return_type",
        export_keywords=frozenset({"pub"}),
        test_name_prefixes=("test_",),
        doc_comment_prefixes=("///", "//!"),
        comment_types=frozenset({"line_comment", "block_comment", "doc_comment"}),
        import_nodes={"use_declaration": None},
        kind_overrides={
            "impl_item": "impl",
            "trait_item": "trait",
            "struct_item": "struct",
            "enum_item": "enum",
            "type_item": "type",
        },
        call_node_types=frozenset({"call_expression", "macro_invocation"}),
        call_function_fields=("function", "macro"),
        extra_nested_fn_types=frozenset({"closure_expression"}),
    ),
    "java": LangConfig(
        func_types=frozenset({"method_declaration", "constructor_declaration"}),
        type_types=frozenset({"class_declaration", "interface_declaration", "enum_declaration"}),
        param_fields=("formal_parameters",),
        body_fields=("body",),
        return_type_field=None,
        export_keywords=frozenset({"public", "protected"}),
        test_name_prefixes=("test", "should"),
        doc_comment_prefixes=("/**",),
        comment_types=frozenset({"block_comment", "line_comment"}),
        import_nodes={"import_declaration": None},
        call_node_types=frozenset({"method_invocation", "object_creation_expression"}),
        call_function_fields=("name", "type"),
    ),
    "bash": LangConfig(
        func_types=frozenset({"function_definition"}),
        type_types=frozenset(),
        body_fields=("body", "compound_statement"),
        export_keywords=frozenset(),
        test_name_prefixes=("test_", "@test"),
        comment_types=frozenset({"comment"}),
        call_node_types=frozenset({"command"}),
        call_function_fields=("name",),
    ),
    "lua": LangConfig(
        func_types=frozenset({"function_declaration", "local_function", "function_statement"}),
        type_types=frozenset(),
        body_fields=("body",),
        test_name_prefixes=("test_", "it", "describe"),
        comment_types=frozenset({"comment"}),
        call_node_types=frozenset({"function_call"}),
        call_function_fields=("name",),
    ),
    "swift": LangConfig(
        func_types=frozenset({
            "function_declaration", "init_declaration", "deinit_declaration",
            "subscript_declaration",
        }),
        type_types=frozenset({
            "class_declaration", "struct_declaration", "enum_declaration",
            "protocol_declaration", "extension_declaration", "actor_declaration",
        }),
        body_fields=("body", "computed_body", "statements"),
        comment_types=frozenset({"line_comment", "multiline_comment", "comment"}),
        test_name_prefixes=("test",),
        call_node_types=frozenset({"call_expression"}),
        call_function_fields=("function",),
        extra_nested_fn_types=frozenset({"lambda_literal", "closure_expression"}),
    ),
}

# Aliases
LANG_CONFIGS["tsx"] = LANG_CONFIGS["typescript"]
LANG_CONFIGS["jsx"] = LANG_CONFIGS["javascript"]

# C/C++ config for docstring/body helpers only (extractor remains hand-written)
_C_CFG = LangConfig(
    func_types=frozenset({"function_definition"}),
    type_types=frozenset({"class_specifier", "struct_specifier"}),
    body_fields=("body", "compound_statement"),
    comment_types=frozenset({"comment"}),
    call_node_types=frozenset({"call_expression"}),
    call_function_fields=("function",),
    extra_nested_fn_types=frozenset({"lambda_expression"}),
)


# ── Generic helper functions ──────────────────────────────────────────────────

def _strip_comment_markers(raw: str, node_type: str) -> str:
    """Strip comment delimiters from a raw comment string.

    Uses tree-sitter's node type — which the grammar already classifies
    language-specifically — rather than per-language regex configs.

    Node type → expected syntax:
      doc_comment    → /// or //! (Rust)
      line_comment   → // or # or -- (per language, detect from text)
      block_comment  → /* … */ or /** … */
      comment        → generic; detect from first chars (Go, JS, Python, Bash, Lua)
    """
    if not raw:
        return ""

    if node_type == "doc_comment":
        # Rust ///... or //!... — strip the leading marker from every line
        lines = [re.sub(r'^\s*//[/!]\s?', '', ln) for ln in raw.splitlines()]
        return "\n".join(lines).strip()

    if node_type == "line_comment":
        # Covers // (C-family), # (Python/Bash/Ruby), -- (Lua/SQL/Haskell)
        # Check the first non-whitespace chars to decide which marker to strip
        stripped_head = raw.lstrip()
        if stripped_head.startswith("//"):
            marker = r'^\s*///?\s?'
        elif stripped_head.startswith("#"):
            marker = r'^\s*#\s?'
        elif stripped_head.startswith("--"):
            marker = r'^\s*--\s?'
        else:
            marker = r'^\s*\S{1,3}\s?'   # best-effort for unknown markers
        lines = [re.sub(marker, '', ln) for ln in raw.splitlines()]
        return "\n".join(lines).strip()

    if node_type == "block_comment":
        # /** ... */ or /* ... */ — strip opening/closing delimiters and
        # the leading * on interior lines (standard JavaDoc / C-style).
        text = re.sub(r'^/\*+\s?', '', raw)          # strip opening /* or /**
        text = re.sub(r'\s*\*+/\s*$', '', text)      # strip closing */
        lines = [re.sub(r'^\s*\*\s?', '', ln) for ln in text.splitlines()]
        return "\n".join(lines).strip()

    # node_type == "comment" (Go, JavaScript, Python, Bash, Lua all use this)
    # or any unrecognised comment type: detect from text content
    stripped_head = raw.lstrip()
    if stripped_head.startswith("/**") or stripped_head.startswith("/*"):
        text = re.sub(r'^/\*+\s?', '', raw)
        text = re.sub(r'\s*\*+/\s*$', '', text)
        lines = [re.sub(r'^\s*\*\s?', '', ln) for ln in text.splitlines()]
        return "\n".join(lines).strip()
    if stripped_head.startswith("//"):
        lines = [re.sub(r'^\s*///?\s?', '', ln) for ln in raw.splitlines()]
        return "\n".join(lines).strip()
    if stripped_head.startswith("#"):
        lines = [re.sub(r'^\s*#\s?', '', ln) for ln in raw.splitlines()]
        return "\n".join(lines).strip()
    if stripped_head.startswith("--"):
        lines = [re.sub(r'^\s*--\s?', '', ln) for ln in raw.splitlines()]
        return "\n".join(lines).strip()

    return raw.strip()


def _generic_body(node: Node, src: bytes, body_fields: tuple,
                  max_lines: int = 20, max_chars: int = 800) -> str:
    """Extract body preview from any node, trying body_fields in order,
    then falling back to finding child nodes whose type is in body_fields."""
    body = None
    for fname in body_fields:
        body = node.child_by_field_name(fname)
        if body:
            break
    if body is None:
        body_type_set = set(body_fields)
        for child in node.children:
            if child.type in body_type_set:
                body = child
                break
    if body is None:
        return ""
    body_text = src[body.start_byte:body.end_byte].decode("utf-8", errors="replace")
    lines = [ln for ln in body_text.splitlines() if ln.strip()]
    preview = "\n".join(lines[:max_lines])
    if preview:
        preview = textwrap.dedent(preview).strip()
    return preview[:max_chars]


def _generic_docstring(node: Node, src: bytes, cfg: LangConfig) -> str:
    """Extract a doc comment from siblings immediately preceding this node.
    Works for /// (Swift/Rust), /** */ (Java/JS), # (Python already handled), // (Go)."""
    parent = node.parent
    if parent is None:
        return ""
    siblings = list(parent.children)
    try:
        idx = siblings.index(node)
    except ValueError:
        return ""
    parts: list[str] = []
    for sib in reversed(siblings[:idx]):
        if sib.type in cfg.comment_types:
            raw = _node_text(sib, src).strip()
            # Filter to doc-style prefixes if configured
            if cfg.doc_comment_prefixes and not any(raw.startswith(p) for p in cfg.doc_comment_prefixes):
                break
            # Use tree-sitter node type to drive stripping — no per-language regex needed
            cleaned = _strip_comment_markers(raw, sib.type)
            if cleaned:
                parts.insert(0, cleaned)
        elif sib.type in ("\n", "newline", ""):
            continue
        else:
            break
    return " ".join(parts)[:512]


def _generic_return_type(node: Node, src: bytes, return_type_field: str | None) -> str:
    """Extract return type annotation. Tries the configured field name first,
    then looks for a -> token followed by a type annotation node."""
    if return_type_field:
        rt = node.child_by_field_name(return_type_field)
        if rt:
            return " -> " + _node_text(rt, src).strip()
    # Arrow detection: scan children for -> token
    found_arrow = False
    for child in node.children:
        if child.type in ("->", "arrow", "=>"):
            found_arrow = True
            continue
        if found_arrow:
            # Skip into the body/block
            if child.type in ("block", "compound_statement", "function_body", "{"):
                break
            return " -> " + _node_text(child, src).strip()
    return ""


def _generic_is_exported(node: Node, src: bytes, name: str, cfg: LangConfig) -> bool:
    """Determine if a symbol is exported/public."""
    if cfg.export_by_capitalization:
        return bool(name) and name[0].isupper()

    # Ancestor walk for languages like JS/TS where `export const foo = …` wraps
    # the declaration in an export_statement ancestor.
    if cfg.export_ancestor_types:
        p = node.parent
        for _ in range(cfg.export_walk_depth):
            if p is None:
                break
            if p.type in cfg.export_ancestor_types:
                return True
            p = p.parent

    if cfg.export_keywords:
        # Check parent/grandparent for export wrapper nodes (legacy path)
        p = node.parent
        if p and "export" in p.type:
            return True
        # Check preceding siblings for pub/public/export keywords
        if p:
            for sib in p.children:
                if sib == node:
                    break
                if _node_text(sib, src).strip() in cfg.export_keywords:
                    return True
        # Check node's own modifier/visibility children (handles Rust `pub`, `pub(crate)`)
        for child in node.children:
            ct = child.type
            if ct == "visibility_modifier":
                return True
            if _node_text(child, src).strip() in cfg.export_keywords:
                return True
    # Default: not underscore-prefixed = exported
    return not (name or "").startswith("_")


def _generic_is_test(name: str, rel_path: str, cfg: LangConfig) -> bool:
    """Return True if this symbol is a test function."""
    return (
        "test" in rel_path.lower()
        or any(name.startswith(p) for p in cfg.test_name_prefixes)
    )


def _import_target(node: Node, src: bytes, child_types: frozenset) -> str:
    """Extract a clean module string from an import node by finding the first
    descendant whose type is in child_types.  Falls back to the full node text."""
    for child in node.children:
        if child.type in child_types:
            return _node_text(child, src)
    return _node_text(node, src)


def _callee_name(node: Node, src: bytes) -> str:
    """Extract the bare function/method name from a call's callee node.

    Handles the main callee shapes:
      identifier / simple_identifier         → direct name
      member_expression / selector_expression → take the rightmost field (property/field)
      attribute (Python)                      → take the last identifier (obj.method)
      scoped_identifier / qualified_identifier→ take the last identifier child
      anything else                           → walk to the rightmost identifier leaf
    """
    t = node.type

    # Direct identifier forms
    if t in ("identifier", "simple_identifier", "field_identifier",
             "name_identifier", "type_identifier", "property_identifier"):
        return _node_text(node, src)

    # member_expression (JS/Java): obj.method, this.method
    # field_expression (Rust): obj.field()
    # selector_expression (Go): obj.Method
    # attribute (Python): obj.method
    if t in ("member_expression", "field_expression", "selector_expression", "attribute"):
        for fname in ("property", "field", "attribute"):
            child = node.child_by_field_name(fname)
            if child:
                return _node_text(child, src)
        # Fallback: last identifier-like child
        for child in reversed(node.children):
            if child.type in ("identifier", "simple_identifier", "field_identifier",
                              "property_identifier", "name_identifier"):
                return _node_text(child, src)

    # Rust/Java/C++ scoped identifiers: std::foo::bar, Foo::bar
    if t in ("scoped_identifier", "qualified_identifier", "scoped_type_identifier"):
        name_field = node.child_by_field_name("name")
        if name_field:
            return _node_text(name_field, src)
        for child in reversed(node.children):
            if child.type in ("identifier", "type_identifier"):
                return _node_text(child, src)

    # Generic fallback: walk children right-to-left for first identifier-like node
    for child in reversed(node.children):
        if child.type in ("identifier", "simple_identifier", "field_identifier",
                          "property_identifier", "name_identifier", "type_identifier"):
            return _node_text(child, src)
        if child.type in ("member_expression", "field_expression", "selector_expression",
                          "scoped_identifier", "qualified_identifier", "attribute"):
            name = _callee_name(child, src)
            if name:
                return name

    return ""


def _ast_calls(func_node: Node, src: bytes, source_fqn: str,
               cfg: LangConfig, result: FileSymbols) -> None:
    """Walk the function/method body and emit CALLS edges using the tree-sitter AST.

    Advantages over regex scanning:
    - No false positives from string literals or comments
    - Calls in nested functions/lambdas are NOT attributed to the enclosing scope
    - All edges have confidence=1.0 (AST-derived)
    """
    stop_types = cfg.func_types | cfg.extra_nested_fn_types

    def walk(node: Node) -> None:
        if node.type in cfg.call_node_types:
            callee_node = None
            for field_name in cfg.call_function_fields:
                callee_node = node.child_by_field_name(field_name)
                if callee_node is not None:
                    break
            if callee_node is not None:
                name = _callee_name(callee_node, src)
                if name and name not in _SKIP_CALLS and len(name) > 1:
                    result.edges.append(
                        CallEdge(source_fqn, name, node.start_point[0] + 1, "CALLS", 1.0)
                    )
        # Recurse into children — stop at nested function/lambda definitions
        for child in node.children:
            if child.type not in stop_types:
                walk(child)

    # Start from the function body (not the whole function node) so we don't
    # accidentally scan the parameter list or return type annotation.
    body: Node | None = None
    for field in cfg.body_fields:
        body = func_node.child_by_field_name(field)
        if body is not None:
            break

    if body is not None:
        walk(body)
    else:
        # Arrow function with an expression body (no block), or language where
        # the body is not named — walk the function node directly.
        walk(func_node)


def _extract_generic(tree, src: bytes, rel_path: str, cfg: LangConfig) -> FileSymbols:
    """Universal tree-sitter extractor driven by LangConfig.
    Handles functions, methods, types (classes/structs/enums),
    body previews, docstrings, return types, export/test detection,
    and CONTAINS edges — for any language with a tree-sitter grammar.
    """
    result = FileSymbols()
    scope_stack: list[str] = []

    def qualified(name: str) -> str:
        return f"{rel_path}::{'.'.join(scope_stack + [name])}"

    def get_name(node: Node) -> str | None:
        for fname in cfg.name_fields:
            n = node.child_by_field_name(fname)
            if n:
                # The "type" field (used for Rust impl_item) returns a complex type node;
                # extract just the base type name — the first type_identifier or identifier.
                if fname == "type":
                    for desc in n.children:
                        if desc.type in ("type_identifier", "identifier", "generic_type"):
                            # For generic_type, take the first type_identifier child
                            if desc.type == "generic_type":
                                for gc in desc.children:
                                    if gc.type in ("type_identifier", "identifier"):
                                        return _node_text(gc, src)
                            return _node_text(desc, src)
                    return _node_text(n, src)  # fallback: full type text
                return _node_text(n, src)
        # Fallback: first identifier/simple_identifier child
        for child in node.children:
            if child.type in ("identifier", "simple_identifier", "name_identifier"):
                return _node_text(child, src)
        return None

    def get_params(node: Node) -> str:
        for fname in cfg.param_fields:
            p = node.child_by_field_name(fname)
            if p:
                return _node_text(p, src)
        return ""

    def node_kind(node: Node) -> str:
        raw = node.type
        for suffix in ("_declaration", "_definition", "_item", "_statement", "_expression"):
            if raw.endswith(suffix):
                raw = raw[:-len(suffix)]
                break
        return cfg.kind_overrides.get(node.type, raw)

    def visit(node: Node, parent_fqn: str | None = None):
        if node.type in cfg.func_types:
            name = get_name(node)
            if not name:
                for child in node.children:
                    visit(child, parent_fqn)
                return
            fqn = qualified(name)
            params = get_params(node)
            ret = _generic_return_type(node, src, cfg.return_type_field)
            sig = params + ret
            doc = _generic_docstring(node, src, cfg)
            preview = _generic_body(node, src, cfg.body_fields)
            kind = "method" if (cfg.use_scope_for_method_kind and scope_stack) else "function"
            exp = _generic_is_exported(node, src, name, cfg)
            tst = _generic_is_test(name, rel_path, cfg)

            result.symbols.append(Symbol(
                name, fqn, rel_path, kind,
                node.start_point[0] + 1, node.end_point[0] + 1,
                signature=sig, docstring=doc, body_preview=preview,
                is_exported=exp, is_test=tst,
            ))
            if parent_fqn:
                result.edges.append(CallEdge(parent_fqn, fqn, node.start_point[0] + 1, "CONTAINS", 1.0))

            scope_stack.append(name)
            for child in node.children:
                visit(child, fqn)
            scope_stack.pop()
            _ast_calls(node, src, fqn, cfg, result)

        elif node.type in cfg.type_types:
            name = get_name(node)
            if not name:
                for child in node.children:
                    visit(child, parent_fqn)
                return
            fqn = qualified(name)
            doc = _generic_docstring(node, src, cfg)
            kind = node_kind(node)
            exp = _generic_is_exported(node, src, name, cfg)
            tst = _generic_is_test(name, rel_path, cfg)

            result.symbols.append(Symbol(
                name, fqn, rel_path, kind,
                node.start_point[0] + 1, node.end_point[0] + 1,
                docstring=doc, is_exported=exp, is_test=tst,
            ))
            if parent_fqn:
                result.edges.append(CallEdge(parent_fqn, fqn, node.start_point[0] + 1, "CONTAINS", 1.0))

            scope_stack.append(name)
            for child in node.children:
                visit(child, fqn)
            scope_stack.pop()

        elif node.type in cfg.import_nodes:
            field_name = cfg.import_nodes[node.type]
            if field_name:
                mod_node = node.child_by_field_name(field_name)
                target = _node_text(mod_node, src) if mod_node else _node_text(node, src)
            else:
                # Walk children to find the module name, avoiding raw statement text
                # (e.g. "use std::foo;" → find the scoped_identifier child)
                target = _import_target(node, src, cfg.import_target_child_types)
            container = parent_fqn or f"{rel_path}::__module__"
            result.edges.append(CallEdge(container, target.strip(), node.start_point[0] + 1, "IMPORTS", 0.9))

        else:
            for child in node.children:
                visit(child, parent_fqn)

    visit(tree.root_node)
    return result


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
            _ast_calls(node, src, fqn, LANG_CONFIGS["python"], result)

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
    """Only the FIRST statement in the body counts as a docstring.

    Uses tree-sitter's `string_content` child of the `string` node to get
    clean text — no manual quote-stripping regex needed.
    """
    body = node.child_by_field_name("body")
    if not body:
        return ""
    for child in body.children:
        if child.type in ("newline", "comment", "indent", "dedent"):
            continue
        # First real statement: if it's an expression_statement holding a
        # string literal then it's the docstring; otherwise there is none.
        if child.type == "expression_statement":
            for grandchild in child.children:
                if grandchild.type == "string":
                    # tree-sitter-python splits string nodes into:
                    #   string_start  (the opening """ / ''' / " / ')
                    #   string_content (the raw text between delimiters)
                    #   string_end    (the closing delimiter)
                    # Use string_content directly — no quote-stripping needed.
                    for gc in grandchild.children:
                        if gc.type == "string_content":
                            return _node_text(gc, src).strip()[:512]
                    # Fallback for grammars that don't split (shouldn't happen
                    # with current tree-sitter-python, but keeps us safe).
                    raw = _node_text(grandchild, src).strip()
                    for q in ('"""', "'''", '"', "'"):
                        if raw.startswith(q) and raw.endswith(q) and len(raw) >= 2 * len(q):
                            return raw[len(q):-len(q)].strip()[:512]
        return ""
    return ""


def _body_preview(node: Node, src: bytes, max_lines: int = 20, max_chars: int = 800) -> str:
    """Extract the first *max_lines* lines of a function/method body for embedding.

    Uses tree-sitter to skip the leading docstring node (if present), then
    takes raw source text of the remaining statements. Strips common
    indentation and caps at *max_chars* to keep embedding text compact.

    For Python: uses the "body" field and skips the docstring.
    For other languages: delegates to _generic_body.
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
        preview = textwrap.dedent(preview).strip()
    return preview[:max_chars]


def _js_is_exported(node: Node, cfg: LangConfig) -> bool:
    """Walk up ancestors (bounded) to find an export_statement ancestor."""
    p = node.parent
    for _ in range(cfg.export_walk_depth):
        if p is None:
            break
        if p.type in cfg.export_ancestor_types:
            return True
        p = p.parent
    return False


def _js_imports(node: Node, src: bytes, rel_path: str, result: FileSymbols):
    """Emit an IMPORTS edge for each JS/TS import statement.

    Uses the `string_fragment` grandchild of the `string` node — which the
    tree-sitter grammar provides delimiter-free — rather than stripping quotes.
    """
    container = f"{rel_path}::__module__"
    for child in node.children:
        if child.type == "string":
            # tree-sitter-javascript splits `"./foo"` into:
            #   string → string_fragment  (the delimiter-free content)
            for gc in child.children:
                if gc.type == "string_fragment":
                    target = _node_text(gc, src)
                    result.edges.append(
                        CallEdge(container, target, node.start_point[0] + 1, "IMPORTS", 0.9)
                    )
                    break
            else:
                # Grammar doesn't split the string — fall back to stripping quotes
                raw = _node_text(child, src)
                if len(raw) >= 2:
                    raw = raw[1:-1]
                result.edges.append(
                    CallEdge(container, raw, node.start_point[0] + 1, "IMPORTS", 0.9)
                )


def _extract_js_ts(tree, src: bytes, rel_path: str, lang: str) -> FileSymbols:
    cfg = LANG_CONFIGS.get(lang, LANG_CONFIGS["javascript"])
    result = FileSymbols()
    scope_stack: list[str] = []

    def qualified(name: str) -> str:
        return f"{rel_path}::{'.'.join(scope_stack + [name])}"

    FUNC_NODES = cfg.func_types
    CLASS_NODES = cfg.type_types
    DECL_NODES = cfg.declarator_node_types  # e.g. {"variable_declarator"}

    def _sym_name_and_anchor(node: Node) -> tuple[str, Node]:
        """Return (name, anchor_node) for a function/class node.

        For named declarations (`function foo() {}`), the name comes from the
        `name` field of the node itself.  For anonymous functions/classes assigned
        to a variable (`const foo = () => {}`), we hoist the name from the
        enclosing `variable_declarator` so the symbol is stored under `foo`
        rather than `<anonymous>`.  The anchor node is used for line-number,
        docstring, and export detection.
        """
        name_node = node.child_by_field_name("name")
        if name_node:
            return _node_text(name_node, src), node

        # Attempt declarator hoisting
        p = node.parent
        if p is not None and p.type in DECL_NODES:
            name_field = p.child_by_field_name(cfg.declarator_name_field)
            if name_field:
                return _node_text(name_field, src), p

        return "<anonymous>", node

    def visit(node: Node, parent_fqn: str | None = None):
        if node.type in FUNC_NODES:
            name, anchor = _sym_name_and_anchor(node)
            kind = "method" if scope_stack else "function"
            fqn = qualified(name)
            params = node.child_by_field_name("parameters") or node.child_by_field_name("params")
            sig = _node_text(params, src) if params else ""
            ret = _generic_return_type(node, src, cfg.return_type_field)
            if ret:
                sig = sig + ret
            doc = _generic_docstring(anchor, src, cfg)
            preview = _generic_body(node, src, cfg.body_fields)
            exported = _js_is_exported(anchor, cfg)
            is_test = (
                "test" in rel_path.lower()
                or "spec" in rel_path.lower()
                or _generic_is_test(name, rel_path, cfg)
            )
            sym = Symbol(name, fqn, rel_path, kind,
                         anchor.start_point[0] + 1, node.end_point[0] + 1,
                         signature=sig, docstring=doc, body_preview=preview,
                         is_exported=exported, is_test=is_test)
            result.symbols.append(sym)
            if parent_fqn:
                result.edges.append(CallEdge(parent_fqn, fqn, anchor.start_point[0] + 1, "CONTAINS", 1.0))
            scope_stack.append(name)
            for child in node.children:
                visit(child, fqn)
            scope_stack.pop()
            _ast_calls(node, src, fqn, cfg, result)

        elif node.type in CLASS_NODES:
            name, anchor = _sym_name_and_anchor(node)
            fqn = qualified(name)
            doc = _generic_docstring(anchor, src, cfg)
            exported = _js_is_exported(anchor, cfg)
            kind = cfg.kind_overrides.get(node.type, "class")
            sym = Symbol(name, fqn, rel_path, kind,
                         anchor.start_point[0] + 1, node.end_point[0] + 1,
                         docstring=doc, is_exported=exported)
            result.symbols.append(sym)
            if parent_fqn:
                result.edges.append(CallEdge(parent_fqn, fqn, anchor.start_point[0] + 1, "CONTAINS", 1.0))
            scope_stack.append(name)
            for child in node.children:
                visit(child, fqn)
            scope_stack.pop()

        elif node.type == "import_statement":
            _js_imports(node, src, rel_path, result)
        else:
            for child in node.children:
                visit(child, parent_fqn)

    visit(tree.root_node)
    return result


# Go type kind mapping — keyed on the type field's node type
_GO_TYPE_KIND: dict[str, str] = {
    "struct_type":    "struct",
    "interface_type": "interface",
    "function_type":  "func_type",
    "map_type":       "map",
    "channel_type":   "chan",
    "type_alias":     "alias",
}


def _go_receiver_type(recv_node: Node, src: bytes) -> str:
    """Extract the bare type name from a Go method receiver node.

    The receiver is a `parameter_list` containing a `parameter_declaration`.
    That declaration's `type` field is a `type_identifier` (value receiver)
    or a `pointer_type` whose child is the `type_identifier` (pointer receiver).
    We return just the bare type name — no stars, no generics.
    """
    if recv_node is None:
        return ""
    for child in recv_node.children:
        if child.type == "parameter_declaration":
            type_node = child.child_by_field_name("type")
            if type_node is None:
                continue
            if type_node.type == "pointer_type":
                # *MyType — step into the pointer to get the type_identifier
                for gc in type_node.children:
                    if gc.type in ("type_identifier", "generic_type"):
                        if gc.type == "generic_type":
                            # MyType[T] — return just the base name
                            for ggc in gc.children:
                                if ggc.type == "type_identifier":
                                    return _node_text(ggc, src)
                        return _node_text(gc, src)
            elif type_node.type in ("type_identifier", "generic_type"):
                if type_node.type == "generic_type":
                    for gc in type_node.children:
                        if gc.type == "type_identifier":
                            return _node_text(gc, src)
                return _node_text(type_node, src)
    return ""


def _extract_go(tree, src: bytes, rel_path: str) -> FileSymbols:
    cfg = LANG_CONFIGS["go"]
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
            ret = _generic_return_type(node, src, cfg.return_type_field)
            if ret:
                sig = sig + ret
            doc = _generic_docstring(node, src, cfg)
            preview = _generic_body(node, src, cfg.body_fields)
            is_exp = _generic_is_exported(node, src, name, cfg)
            is_tst = _generic_is_test(name, rel_path, cfg)
            f = fqn(name)
            sym = Symbol(name, f, rel_path, "function",
                         node.start_point[0] + 1, node.end_point[0] + 1,
                         signature=sig, docstring=doc, body_preview=preview,
                         is_exported=is_exp, is_test=is_tst)
            result.symbols.append(sym)
            _ast_calls(node, src, f, cfg, result)

        elif node.type == "method_declaration":
            name_node = node.child_by_field_name("name")
            recv_node = node.child_by_field_name("receiver")
            if not name_node:
                return
            name = _node_text(name_node, src)
            recv = _go_receiver_type(recv_node, src)
            f = fqn(name, recv)
            params = node.child_by_field_name("parameters")
            sig = _node_text(params, src) if params else ""
            ret = _generic_return_type(node, src, cfg.return_type_field)
            if ret:
                sig = sig + ret
            doc = _generic_docstring(node, src, cfg)
            preview = _generic_body(node, src, cfg.body_fields)
            is_exp = _generic_is_exported(node, src, name, cfg)
            is_tst = _generic_is_test(name, rel_path, cfg)
            sym = Symbol(name, f, rel_path, "method",
                         node.start_point[0] + 1, node.end_point[0] + 1,
                         signature=sig, docstring=doc, body_preview=preview,
                         is_exported=is_exp, is_test=is_tst)
            result.symbols.append(sym)
            _ast_calls(node, src, f, cfg, result)

        elif node.type == "type_declaration":
            for child in node.children:
                if child.type == "type_spec":
                    name_node = child.child_by_field_name("name")
                    type_node = child.child_by_field_name("type")
                    if name_node:
                        name = _node_text(name_node, src)
                        kind = _GO_TYPE_KIND.get(
                            type_node.type if type_node else "", "type"
                        )
                        f = fqn(name)
                        # Use the type_spec node for docstring so each spec in a
                        # parenthesised type block gets its own preceding comment.
                        doc = _generic_docstring(child, src, cfg)
                        is_exp = _generic_is_exported(child, src, name, cfg)
                        is_tst = _generic_is_test(name, rel_path, cfg)
                        result.symbols.append(Symbol(name, f, rel_path, kind,
                                                     child.start_point[0] + 1,
                                                     child.end_point[0] + 1,
                                                     docstring=doc,
                                                     is_exported=is_exp,
                                                     is_test=is_tst))

        elif node.type in ("import_declaration", "import_spec"):
            field_name = cfg.import_nodes.get(node.type)
            if field_name:
                mod_node = node.child_by_field_name(field_name)
                target = _node_text(mod_node, src) if mod_node else _node_text(node, src)
            else:
                target = _node_text(node, src)
            container = f"{rel_path}::__module__"
            result.edges.append(CallEdge(container, target.strip(), node.start_point[0] + 1, "IMPORTS", 0.9))

        else:
            for child in node.children:
                visit(child)

    visit(tree.root_node)
    return result


def _cpp_drill_declarator(decl: Node | None) -> Node | None:
    """Drill through pointer/reference/parenthesised declarators to reach
    a function_declarator."""
    while decl and decl.type in ("pointer_declarator", "reference_declarator",
                                 "abstract_pointer_declarator", "parenthesized_declarator"):
        decl = decl.child_by_field_name("declarator")
    return decl


def _cpp_declarator_name(decl: Node, src: bytes) -> str:
    """Extract the symbol name from a function_declarator's inner declarator.

    Handles:
      plain identifier          → foo
      qualified_identifier      → Foo::bar  (out-of-line method definition)
      destructor_name           → ~Foo
    """
    inner = decl.child_by_field_name("declarator")
    if inner is None:
        return "<unknown>"
    if inner.type == "qualified_identifier":
        return _node_text(inner, src)   # e.g. "Foo::bar" — keep for FQN
    return _node_text(inner, src)


def _extract_c_cpp(tree, src: bytes, rel_path: str, lang: str) -> FileSymbols:
    """Extract symbols from C and C++ source.

    Handles:
    - Top-level and static functions
    - Class/struct declarations (C++ only) with nested methods
    - Out-of-line method definitions: `void Foo::bar() {}`
    - Call detection via tree-sitter call_expression nodes
    """
    result = FileSymbols()
    scope_stack: list[str] = []   # class names currently in scope

    def qualified(name: str) -> str:
        if scope_stack:
            return f"{rel_path}::{scope_stack[-1]}::{name}"
        return f"{rel_path}::{name}"

    def visit(node: Node, parent_fqn: str | None = None):
        if node.type == "function_definition":
            decl = node.child_by_field_name("declarator")
            decl = _cpp_drill_declarator(decl)
            if decl and decl.type == "function_declarator":
                raw_name = _cpp_declarator_name(decl, src)
                params = decl.child_by_field_name("parameters")
                sig = _node_text(params, src) if params else ""
                doc = _generic_docstring(node, src, _C_CFG)
                preview = _generic_body(node, src, _C_CFG.body_fields)

                if "::" in raw_name:
                    # Out-of-line method: Foo::bar — reconstruct the FQN
                    parts = raw_name.split("::")
                    class_name = "::".join(parts[:-1])
                    method_name = parts[-1]
                    fqn = f"{rel_path}::{class_name}::{method_name}"
                    kind = "method"
                    exp = not method_name.startswith("_")
                else:
                    kind = "method" if scope_stack else "function"
                    fqn = qualified(raw_name)
                    exp = not raw_name.startswith("_")

                sym = Symbol(raw_name, fqn, rel_path, kind,
                             node.start_point[0] + 1, node.end_point[0] + 1,
                             signature=sig, docstring=doc, body_preview=preview,
                             is_exported=exp)
                result.symbols.append(sym)
                if parent_fqn:
                    result.edges.append(
                        CallEdge(parent_fqn, fqn, node.start_point[0] + 1, "CONTAINS", 1.0)
                    )
                _ast_calls(node, src, fqn, _C_CFG, result)
            else:
                for child in node.children:
                    visit(child, parent_fqn)

        elif node.type in ("class_specifier", "struct_specifier"):
            name_node = node.child_by_field_name("name")
            if name_node:
                class_name = _node_text(name_node, src)
                fqn = qualified(class_name)
                doc = _generic_docstring(node, src, _C_CFG)
                kind = "class" if node.type == "class_specifier" else "struct"
                sym = Symbol(class_name, fqn, rel_path, kind,
                             node.start_point[0] + 1, node.end_point[0] + 1,
                             docstring=doc, is_exported=not class_name.startswith("_"))
                result.symbols.append(sym)
                if parent_fqn:
                    result.edges.append(
                        CallEdge(parent_fqn, fqn, node.start_point[0] + 1, "CONTAINS", 1.0)
                    )
                scope_stack.append(class_name)
                for child in node.children:
                    visit(child, fqn)
                scope_stack.pop()
            else:
                for child in node.children:
                    visit(child, parent_fqn)

        else:
            for child in node.children:
                visit(child, parent_fqn)

    visit(tree.root_node)
    return result


# ── Swift extractor ───────────────────────────────────────────────────────────

def _extract_swift_docstring(node: Node, src: bytes) -> str:
    """
    Extract Swift doc comment from the `///` or `/** */` lines immediately
    preceding a declaration.  tree-sitter-swift surfaces these as
    `multiline_comment` or `line_comment` sibling nodes before the declaration.
    Uses _strip_comment_markers (keyed on node.type) — no hardcoded regexes.
    """
    parent = node.parent
    if parent is None:
        return ""
    siblings = list(parent.children)
    try:
        idx = siblings.index(node)
    except ValueError:
        return ""
    # Walk backwards collecting contiguous doc comment lines
    lines: list[str] = []
    for sib in reversed(siblings[:idx]):
        t = sib.type
        if t in ("line_comment", "multiline_comment", "comment"):
            raw = _node_text(sib, src).strip()
            # Only accept doc-style comments (/// or /** ... */)
            if raw.startswith("///") or raw.startswith("/**"):
                lines.insert(0, _strip_comment_markers(raw, t))
            else:
                break  # non-doc comment breaks the run
        elif t in ("newline", "\n"):
            continue
        else:
            break
    return " ".join(lines)[:512]


def _swift_signature(node: Node, src: bytes) -> str:
    """
    Build a full Swift function signature including parameters and return type.
    e.g.  (sinceDate: Date?, limit: Int) -> [RiskEntry]
    """
    params = node.child_by_field_name("params")
    ret = None
    # Return type lives in various field names depending on grammar version
    for fname in ("return_type", "result"):
        ret = node.child_by_field_name(fname)
        if ret:
            break
    # Fallback: find the first `type_annotation` or `->` sibling after params
    if ret is None:
        found_arrow = False
        for child in node.children:
            if child.type in ("->", "arrow"):
                found_arrow = True
                continue
            if found_arrow and child.type not in ("{", "function_body", "computed_property"):
                ret = child
                break

    sig = _node_text(params, src) if params else ""
    if ret:
        sig = sig + " -> " + _node_text(ret, src).strip()
    return sig


def _extract_swift(tree, src: bytes, rel_path: str) -> FileSymbols:
    result = FileSymbols()
    scope_stack: list[str] = []

    def qualified(name: str) -> str:
        return f"{rel_path}::{'.'.join(scope_stack + [name])}"

    FUNC_NODES = {
        "function_declaration",
        "init_declaration",
        "deinit_declaration",
        "subscript_declaration",   # subscript(index:) -> T
    }
    TYPE_NODES = {
        "class_declaration",
        "struct_declaration",
        "enum_declaration",
        "protocol_declaration",
        "extension_declaration",
        "actor_declaration",       # Swift 5.5+ actors
    }

    def visit(node: Node, parent_fqn: str | None = None):
        if node.type in FUNC_NODES:
            name_node = node.child_by_field_name("name")
            name = _node_text(name_node, src) if name_node else (
                "subscript" if node.type == "subscript_declaration" else "<init>"
            )
            fqn = qualified(name)
            sig = _swift_signature(node, src)
            doc = _extract_swift_docstring(node, src)
            _swift_cfg = LANG_CONFIGS["swift"]
            preview = _generic_body(node, src, _swift_cfg.body_fields)
            kind = "method" if scope_stack else "function"
            is_exp = not name.startswith("_")
            is_tst = "test" in rel_path.lower() or name.startswith("test")
            result.symbols.append(Symbol(
                name, fqn, rel_path, kind,
                node.start_point[0] + 1, node.end_point[0] + 1,
                signature=sig, docstring=doc, body_preview=preview,
                is_exported=is_exp, is_test=is_tst,
            ))
            if parent_fqn:
                result.edges.append(CallEdge(parent_fqn, fqn,
                                             node.start_point[0] + 1, "CONTAINS", 1.0))
            scope_stack.append(name)
            for child in node.children:
                visit(child, fqn)
            scope_stack.pop()
            _ast_calls(node, src, fqn, LANG_CONFIGS["swift"], result)

        elif node.type == "computed_property":
            # var foo: Bar { get { … } set { … } }
            # Name lives in the enclosing property_declaration
            # tree-sitter-swift wraps as: property_declaration > computed_property
            pass  # handled inside property_declaration below

        elif node.type == "property_declaration":
            # var/let stored and computed properties
            name_node = node.child_by_field_name("name")
            if name_node:
                name = _node_text(name_node, src)
                fqn = qualified(name)
                # Check for computed body
                comp = node.child_by_field_name("computed_value") or \
                       next((c for c in node.children
                             if c.type == "computed_property"), None)
                if comp:
                    preview = _generic_body(node, src, LANG_CONFIGS["swift"].body_fields)
                    doc = _extract_swift_docstring(node, src)
                    sig = ""
                    # Try to get type annotation for signature
                    for child in node.children:
                        if child.type == "type_annotation":
                            sig = _node_text(child, src)
                            break
                    is_exp = not name.startswith("_")
                    is_tst = "test" in rel_path.lower()
                    result.symbols.append(Symbol(
                        name, fqn, rel_path, "property",
                        node.start_point[0] + 1, node.end_point[0] + 1,
                        signature=sig, docstring=doc, body_preview=preview,
                        is_exported=is_exp, is_test=is_tst,
                    ))
                    if parent_fqn:
                        result.edges.append(CallEdge(parent_fqn, fqn,
                                                     node.start_point[0] + 1, "CONTAINS", 1.0))
                    # Recurse into computed body for nested calls
                    for child in node.children:
                        visit(child, fqn)
                    return
            for child in node.children:
                visit(child, parent_fqn)

        elif node.type == "enum_entry":
            # enum cases: case foo, case bar(Int)
            for child in node.children:
                if child.type == "simple_identifier":
                    name = _node_text(child, src)
                    fqn = qualified(name)
                    doc = _extract_swift_docstring(node, src)
                    result.symbols.append(Symbol(
                        name, fqn, rel_path, "enum_case",
                        node.start_point[0] + 1, node.end_point[0] + 1,
                        docstring=doc, is_exported=not name.startswith("_"),
                    ))
                    if parent_fqn:
                        result.edges.append(CallEdge(parent_fqn, fqn,
                                                     node.start_point[0] + 1, "CONTAINS", 1.0))

        elif node.type in TYPE_NODES:
            name_node = node.child_by_field_name("name")
            if not name_node:
                for child in node.children:
                    visit(child, parent_fqn)
                return
            name = _node_text(name_node, src)
            fqn = qualified(name)
            doc = _extract_swift_docstring(node, src)
            kind = node.type.replace("_declaration", "")
            is_exp = not name.startswith("_")
            is_tst = "test" in rel_path.lower() or "Test" in name
            result.symbols.append(Symbol(
                name, fqn, rel_path, kind,
                node.start_point[0] + 1, node.end_point[0] + 1,
                docstring=doc, is_exported=is_exp, is_test=is_tst,
            ))
            if parent_fqn:
                result.edges.append(CallEdge(parent_fqn, fqn,
                                             node.start_point[0] + 1, "CONTAINS", 1.0))
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
                    # Build a minimal config for R call extraction
                    _r_cfg = LangConfig(
                        func_types=frozenset({"function_definition"}),
                        type_types=frozenset(),
                        body_fields=("body",),
                        call_node_types=frozenset({"call"}),
                        call_function_fields=("function",),
                    )
                    result.symbols.append(Symbol(name, fqn, rel_path, "function",
                                                 node.start_point[0] + 1, node.end_point[0] + 1,
                                                 signature=sig, is_exported=not name.startswith(".")))
                    _ast_calls(rhs, src, fqn, _r_cfg, result)
                    return
        for child in node.children:
            visit(child)

    visit(tree.root_node)
    return result


# ── Markdown / Rmd / Qmd extractor ───────────────────────────────────────────

def _extract_markdown(tree, src: bytes, rel_path: str) -> FileSymbols:
    """
    Extract headings and named code chunks from Markdown / Rmd / Qmd files.

    Uses the tree-sitter AST — headings are `atx_heading` nodes (children
    include the level marker and `inline` content), code blocks are
    `fenced_code_block` nodes with an `info_string` child.  Because the AST
    already separates headings from fenced-code content, no manual fence
    tracking is needed.
    """
    result = FileSymbols()

    # tree-sitter-markdown uses atx_h{1-6}_marker to encode heading level
    _MARKER_LEVEL = {
        "atx_h1_marker": 1, "atx_h2_marker": 2, "atx_h3_marker": 3,
        "atx_h4_marker": 4, "atx_h5_marker": 5, "atx_h6_marker": 6,
    }

    def visit(node: Node):
        if node.type == "atx_heading":
            level = None
            title = None
            for child in node.children:
                if child.type in _MARKER_LEVEL:
                    level = _MARKER_LEVEL[child.type]
                elif child.type == "inline":
                    title = _node_text(child, src).strip()
            if level and title:
                name = re.sub(r'[^A-Za-z0-9_\s]', '', title)[:60].strip().replace(' ', '_')
                if name:
                    fqn = f"{rel_path}::{name}"
                    result.symbols.append(Symbol(
                        title, fqn, rel_path, f"heading{level}",
                        node.start_point[0] + 1, node.start_point[0] + 1,
                        signature=f"{'#' * level} {title}",
                    ))
            return  # don't recurse — headings don't nest

        if node.type == "fenced_code_block":
            # Named Rmd/Qmd chunks: ```{r chunk_name} or ```{python chunk_name}
            for child in node.children:
                if child.type == "info_string":
                    info = _node_text(child, src).strip()
                    m = re.match(r'^\{(\w+)\s+([A-Za-z0-9_.-]+)', info)
                    if m:
                        lang_tag, chunk_name = m.group(1), m.group(2)
                        fqn = f"{rel_path}::{lang_tag}_{chunk_name}"
                        line = node.start_point[0] + 1
                        result.symbols.append(Symbol(
                            chunk_name, fqn, rel_path, "chunk",
                            line, line, signature=f"```{{{lang_tag} {chunk_name}}}",
                        ))
            return  # don't recurse into code block content

        for child in node.children:
            visit(child)

    visit(tree.root_node)
    return result


# ── Public API ────────────────────────────────────────────────────────────────

def extract(file_path: Path, lang: str, root: Path) -> FileSymbols:
    """
    Parse *file_path* and return its symbols and edges.
    *root* is the repo root used to compute relative paths for FQNs.
    """
    if lang not in _LANGUAGES:
        raise RuntimeError(
            f"Tree-sitter grammar for {lang!r} is not available. "
            f"Install the tree_sitter_{lang} package or exclude files of this "
            f"language from indexing (see beacon/lang_map.py)."
        )

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
        return _extract_generic(tree, src, rel_path, LANG_CONFIGS["rust"])
    elif lang == "java":
        return _extract_generic(tree, src, rel_path, LANG_CONFIGS["java"])
    elif lang in ("c", "cpp"):
        return _extract_c_cpp(tree, src, rel_path, lang)
    elif lang == "bash":
        return _extract_generic(tree, src, rel_path, LANG_CONFIGS["bash"])
    elif lang == "lua":
        return _extract_generic(tree, src, rel_path, LANG_CONFIGS["lua"])
    elif lang == "swift":
        return _extract_swift(tree, src, rel_path)
    elif lang == "r":
        return _extract_r(tree, src, rel_path)
    elif lang in ("markdown", "rmarkdown", "quarto"):
        # Rmd/Qmd are markdown-first: extract headings + named code chunks
        return _extract_markdown(tree, src, rel_path)
    else:
        return FileSymbols()
