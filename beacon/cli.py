"""
beacon CLI

Commands:
    index     [<dir>]   — scan + index a codebase
    search    <query>   — search the index
    capsule   <query>   — generate a context capsule
    mcp                 — start MCP stdio server
    setup               — install hook + MCP config for a project
    show-config         — print MCP config for AI agents
"""

import argparse
import json
import os
import sys
from pathlib import Path


# ── Rich helpers ──────────────────────────────────────────────────────────────

def _make_console():
    from rich.console import Console
    return Console(highlight=False)


def _header(console):
    from rich.panel import Panel
    from rich.text import Text
    t = Text()
    t.append("◆ Beacon", style="bold cyan")
    t.append("  semantic code intelligence", style="dim")
    console.print(Panel(t, border_style="dim cyan", padding=(0, 2)))
    console.print()


# ── Model selection TUI ───────────────────────────────────────────────────────

def _prompt_model(console, current_id: str) -> str:
    """
    Interactive model selection panel.
    Returns the chosen model ID (same as current_id if user presses Enter).
    """
    from rich.panel import Panel
    from rich.table import Table
    from rich.prompt import Prompt
    from rich import box
    from beacon.config import MODELS

    table = Table(box=None, show_header=False, padding=(0, 1), expand=False)
    table.add_column("num", style="bold cyan", width=4, no_wrap=True)
    table.add_column("tag", width=6, no_wrap=True)
    table.add_column("model", no_wrap=True)
    table.add_column("description", style="dim")

    default_choice = "1"
    for i, m in enumerate(MODELS, 1):
        active = m["id"] == current_id
        marker = " [green]◆[/green]" if active else ""
        if active:
            default_choice = str(i)
        table.add_row(
            f"[{i}]",
            f"[bold {'green' if active else ''}]{m['tag']}[/bold {'green' if active else ''}]",
            m["short"] + marker,
            m["desc"],
        )

    console.print(Panel(table, title="[bold]Embedding model[/bold]",
                        border_style="blue", padding=(1, 2)))

    choice = Prompt.ask(
        "  [cyan]Select model[/cyan]",
        choices=[str(i) for i in range(1, len(MODELS) + 1)],
        default=default_choice,
        console=console,
        show_choices=False,
    )
    return MODELS[int(choice) - 1]["id"]


# ── index ─────────────────────────────────────────────────────────────────────

def cmd_index(args):
    import sqlite3
    from collections import Counter
    from datetime import datetime, timezone

    import blake3
    from rich.console import Console
    from rich.panel import Panel
    from rich.progress import (
        Progress, SpinnerColumn, BarColumn, TextColumn,
        MofNCompleteColumn, TimeRemainingColumn, TaskProgressColumn,
    )
    from rich.table import Table
    from rich.text import Text
    from rich import box
    from rich.prompt import Confirm

    _suppress_ml_noise()
    from beacon.indexer import scanner, symbols, embedder, coupling
    from beacon.schema import open_db
    import beacon.config as cfg

    console = Console(highlight=False)
    _header(console)

    root = Path(args.dir or os.getcwd()).resolve()
    db_path = Path(args.db) if args.db else root / ".beacon" / "index.db"

    console.print(f"  [dim]root[/dim]  [bold]{root}[/bold]")
    console.print(f"  [dim]db  [/dim]  [dim]{db_path}[/dim]")
    console.print()

    # ── Phase 1: Scan ─────────────────────────────────────────────────────────
    with console.status("[cyan]Scanning files…[/cyan]", spinner="dots"):
        sc = scanner.Scanner(root)
        all_files = sc.collect()

    if args.langs:
        keep = {l.strip().lower() for l in args.langs.split(",")}
        all_files = [(p, l) for p, l in all_files if l in keep]

    if not all_files:
        console.print("[yellow]Nothing to index.[/yellow]")
        return

    lang_counts = Counter(lang for _, lang in all_files)
    lang_str = "  ".join(
        f"[cyan]{lang}[/cyan] [dim]{n}[/dim]"
        for lang, n in lang_counts.most_common(8)
    )
    console.print(f"  [bold]{len(all_files):,}[/bold] indexable files  {lang_str}")
    console.print()

    # Large-repo confirmation
    if len(all_files) > 5000 and not args.yes:
        console.print(f"  [yellow]⚠[/yellow]  {len(all_files):,} files found. "
                      "Use [bold]--langs python[/bold] to restrict languages.")
        if not Confirm.ask("  Proceed?", console=console, default=False):
            console.print("[dim]Aborted.[/dim]")
            return
        console.print()

    # ── Model selection ───────────────────────────────────────────────────────
    current_model = cfg.get_dense_model()

    if not cfg.exists() or getattr(args, "configure", False):
        chosen_model = _prompt_model(console, current_model)
        if chosen_model != current_model:
            cfg.set_dense_model(chosen_model)
            current_model = chosen_model
            console.print(
                f"  [green]✓[/green] Model saved to [dim]{cfg.config_path()}[/dim]"
            )
        console.print()
    else:
        model_short = current_model.split("/")[-1]
        console.print(
            f"  [dim]model[/dim]  [cyan]{model_short}[/cyan]  "
            f"[dim](use --configure to change)[/dim]"
        )
        console.print()

    # ── Phase 2: Hash-check ───────────────────────────────────────────────────
    conn = open_db(db_path)
    cached = {
        r[0]: r[1]
        for r in conn.execute("SELECT file_path, blake3_hash FROM file_cache").fetchall()
    }

    to_index: list[tuple[Path, str, str]] = []
    for file_path, lang in all_files:
        rel = str(file_path.relative_to(root))
        try:
            h = blake3.blake3(file_path.read_bytes()).hexdigest()
        except OSError:
            continue
        if cached.get(rel) != h:
            to_index.append((file_path, lang, rel))

    skipped = len(all_files) - len(to_index)
    if skipped:
        console.print(f"  [dim]{skipped:,} files unchanged — skipping[/dim]")
    if not to_index:
        console.print("  [green]✓[/green] Index is already up to date.")
        _finish(conn, root, db_path, console)
        return
    console.print(f"  [bold]{len(to_index):,}[/bold] files to parse")
    console.print()

    # ── Phase 3: Parse + insert ───────────────────────────────────────────────
    now = datetime.now(timezone.utc).isoformat()
    changed_node_ids: list[int] = []
    all_edges: list[symbols.CallEdge] = []
    parse_errors = 0

    with Progress(
        SpinnerColumn(style="cyan"),
        TextColumn("[progress.description]{task.description}", style="bold"),
        BarColumn(bar_width=32, style="cyan", complete_style="green"),
        MofNCompleteColumn(),
        TaskProgressColumn(),
        TimeRemainingColumn(),
        TextColumn("[dim]{task.fields[current_file]}[/dim]"),
        console=console,
        transient=False,
    ) as progress:
        task = progress.add_task("Parsing", total=len(to_index), current_file="")

        for file_path, lang, rel in to_index:
            display = rel[-52:] if len(rel) > 52 else rel
            progress.update(task, current_file=display)

            conn.execute("DELETE FROM nodes WHERE file_path=?", (rel,))

            try:
                file_syms = symbols.extract(file_path, lang, root)
            except Exception:
                parse_errors += 1
                progress.advance(task)
                continue

            for sym in file_syms.symbols:
                cur = conn.execute(
                    """INSERT INTO nodes
                       (name, fqn, file_path, kind, start_line, end_line,
                        signature, docstring, is_exported, is_test, repo_alias)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                       ON CONFLICT(fqn) DO UPDATE SET
                         name=excluded.name, file_path=excluded.file_path,
                         kind=excluded.kind, start_line=excluded.start_line,
                         end_line=excluded.end_line, signature=excluded.signature,
                         docstring=excluded.docstring, is_exported=excluded.is_exported,
                         is_test=excluded.is_test
                       RETURNING id""",
                    (sym.name, sym.fqn, sym.file_path, sym.kind,
                     sym.start_line, sym.end_line, sym.signature, sym.docstring,
                     int(sym.is_exported), int(sym.is_test), "primary"),
                )
                nid = cur.fetchone()[0]
                changed_node_ids.append(nid)

            all_edges.extend(file_syms.edges)

            h = blake3.blake3(file_path.read_bytes()).hexdigest()
            conn.execute(
                "INSERT OR REPLACE INTO file_cache "
                "(file_path, blake3_hash, last_indexed_at, node_count) VALUES (?, ?, ?, ?)",
                (rel, h, now, len(file_syms.symbols)),
            )

            if len(changed_node_ids) % 50 == 0:
                conn.commit()

            progress.advance(task)

    conn.commit()

    node_total = conn.execute("SELECT COUNT(*) FROM nodes").fetchone()[0]
    console.print(
        f"  [green]✓[/green] [bold]{len(changed_node_ids):,}[/bold] nodes indexed  "
        f"[dim]({node_total:,} total)[/dim]"
    )
    if parse_errors:
        console.print(f"  [yellow]⚠[/yellow]  {parse_errors} files skipped (parse errors)")
    console.print()

    # ── Phase 4: Resolve call edges ───────────────────────────────────────────
    if all_edges:
        with console.status(
            f"[cyan]Resolving {len(all_edges):,} call edges…[/cyan]", spinner="dots"
        ):
            inserted = _resolve_edges_silent(conn, all_edges)
        edge_total = conn.execute("SELECT COUNT(*) FROM edges").fetchone()[0]
        console.print(
            f"  [green]✓[/green] Edges  [bold]{inserted:,}[/bold] new  "
            f"[dim]({edge_total:,} total)[/dim]"
        )
        console.print()

    # ── Phase 5: TF-IDF + dense embeddings ───────────────────────────────────
    if changed_node_ids:
        with console.status("[cyan]Building TF-IDF embeddings…[/cyan]", spinner="dots"):
            embedder.build_incremental(conn, changed_node_ids)
        console.print("  [green]✓[/green] TF-IDF embeddings")

        model_short = current_model.split("/")[-1]
        with console.status(
            f"[cyan]Building dense embeddings ({model_short})…[/cyan]", spinner="dots"
        ):
            embedder.build_dense_incremental(conn, changed_node_ids)
        dense_count = conn.execute("SELECT COUNT(*) FROM node_embeddings_dense").fetchone()[0]
        console.print(
            f"  [green]✓[/green] Dense embeddings  "
            f"[dim]model={model_short}  n={dense_count:,}[/dim]"
        )
        console.print()

    # ── Phase 6: Git change coupling ─────────────────────────────────────────
    if not args.no_coupling:
        with console.status("[cyan]Computing git change coupling…[/cyan]", spinner="dots"):
            # Suppress coupling's own print() output
            import io, contextlib
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                coupling.compute(conn, root)
        msg = buf.getvalue().strip()
        if "Skipping" in msg:
            console.print(f"  [dim]{msg}[/dim]")
        else:
            console.print(f"  [green]✓[/green] Change coupling  [dim]{msg}[/dim]")
        console.print()

    # ── Phase 7: Rebuild FTS5 ─────────────────────────────────────────────────
    with console.status("[cyan]Rebuilding FTS5 index…[/cyan]", spinner="dots"):
        conn.execute("INSERT INTO nodes_fts(nodes_fts) VALUES('rebuild')")
        conn.execute("INSERT INTO observations_fts(observations_fts) VALUES('rebuild')")
        conn.commit()
    fts_count = conn.execute("SELECT COUNT(*) FROM nodes_fts").fetchone()[0]
    console.print(f"  [green]✓[/green] FTS5 index  [dim]{fts_count:,} entries[/dim]")
    console.print()

    _finish(conn, root, db_path, console)


def _resolve_edges_silent(conn, all_edges) -> int:
    """Resolve call edges without any print output. Returns inserted count."""
    from beacon.indexer.symbols import CallEdge
    inserted = 0
    for edge in all_edges:
        if edge.edge_type == "CONTAINS":
            src = conn.execute("SELECT id FROM nodes WHERE fqn=?", (edge.source_fqn,)).fetchone()
            tgt = conn.execute("SELECT id FROM nodes WHERE fqn=?", (edge.target_name,)).fetchone()
        else:
            src = conn.execute("SELECT id FROM nodes WHERE fqn=?", (edge.source_fqn,)).fetchone()
            tgt = (conn.execute("SELECT id FROM nodes WHERE fqn=?", (edge.target_name,)).fetchone()
                   or conn.execute("SELECT id FROM nodes WHERE name=? LIMIT 1", (edge.target_name,)).fetchone())
        if src and tgt and src[0] != tgt[0]:
            conn.execute(
                "INSERT OR IGNORE INTO edges (source_id, target_id, type, call_site_line, confidence) "
                "VALUES (?, ?, ?, ?, ?)",
                (src[0], tgt[0], edge.edge_type, edge.call_site_line, edge.confidence),
            )
            inserted += 1
    conn.commit()
    return inserted


def _finish(conn, root: Path, db_path: Path, console):
    from datetime import datetime, timezone
    from rich.panel import Panel
    from rich.table import Table
    from rich import box

    n = conn.execute("SELECT COUNT(*) FROM nodes").fetchone()[0]
    e = conn.execute("SELECT COUNT(*) FROM edges").fetchone()[0]
    emb = conn.execute("SELECT COUNT(*) FROM node_embeddings").fetchone()[0]
    dense = conn.execute("SELECT COUNT(*) FROM node_embeddings_dense").fetchone()[0]

    # Write healthy marker
    healthy = db_path.parent / "healthy"
    healthy.write_text(
        f"nodes={n}\nedges={e}\nembeddings={emb}\ndense={dense}\n"
        f"indexed_at={datetime.now(timezone.utc).isoformat()}\n"
    )

    table = Table(box=None, show_header=False, padding=(0, 2), expand=False)
    table.add_column("key", style="dim", width=20)
    table.add_column("val", style="bold cyan", justify="right")

    table.add_row("Nodes", f"{n:,}")
    table.add_row("Edges", f"{e:,}")
    table.add_row("TF-IDF embeddings", f"{emb:,}")
    table.add_row("Dense embeddings", f"{dense:,}")
    table.add_row("Index", str(db_path))

    console.print(Panel(
        table,
        title="[bold green]✓ Index complete[/bold green]",
        border_style="green",
        padding=(1, 2),
    ))


# ── ask ───────────────────────────────────────────────────────────────────────

_KIND_STYLE: dict[str, tuple[str, str]] = {
    "function":  ("fn",     "blue"),
    "method":    ("mth",    "cyan"),
    "class":     ("cls",    "magenta"),
    "struct":    ("struct", "magenta"),
    "interface": ("iface",  "green"),
    "type":      ("type",   "green"),
    "variable":  ("var",    "yellow"),
    "constant":  ("const",  "yellow"),
    "module":    ("mod",    "dim"),
    "heading":   ("h",      "dim"),
}

_EXT_LEXER: dict[str, str] = {
    ".py": "python", ".pyi": "python",
    ".js": "javascript", ".jsx": "javascript",
    ".ts": "typescript", ".tsx": "tsx",
    ".go": "go", ".rs": "rust", ".java": "java",
    ".c": "c", ".h": "c", ".cpp": "cpp", ".hpp": "cpp",
    ".sh": "bash", ".bash": "bash", ".zsh": "bash",
    ".r": "r", ".R": "r", ".lua": "lua", ".swift": "swift",
    ".rb": "ruby", ".php": "php", ".cs": "csharp",
}


def _lexer(file_path: str) -> str:
    return _EXT_LEXER.get(Path(file_path).suffix.lower(), "text")


def _suppress_ml_noise() -> None:
    """Silence noisy transformers/sentence-transformers log output."""
    import logging
    import warnings
    import os
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    warnings.filterwarnings("ignore", category=FutureWarning, module="transformers")
    for name in ("transformers", "sentence_transformers", "huggingface_hub"):
        logging.getLogger(name).setLevel(logging.ERROR)


def _render_seed(console, node, idx: int) -> None:
    from rich.panel import Panel
    from rich.syntax import Syntax
    from rich.text import Text
    from rich.console import Group

    abbr, color = _KIND_STYLE.get(node.kind, (node.kind[:4], "white"))

    # Short name: split on :: (Beacon FQN separator)
    short_name = node.fqn.split("::")[-1] if "::" in node.fqn else node.fqn

    title = Text()
    title.append(f" {abbr} ", style=f"bold reverse {color}")
    title.append(f"  {short_name}", style="bold white")

    subtitle = Text(f"{node.file_path}:{node.start_line}", style="dim")

    parts: list = []

    # Full FQN as dim context line (only if different from short name)
    if node.fqn != short_name:
        parts.append(Text(node.fqn, style="dim"))
        parts.append(Text(""))

    # Syntax-highlighted signature
    if node.signature and node.signature.strip():
        sig = node.signature.strip()
        parts.append(Syntax(
            sig,
            _lexer(node.file_path),
            theme="nord",
            background_color="default",
            word_wrap=True,
            padding=(0, 0),
        ))

    # Docstring
    if node.docstring and node.docstring.strip():
        doc = node.docstring.strip()[:400]
        if len(node.docstring.strip()) > 400:
            doc += "…"
        parts.append(Text(""))
        parts.append(Text(doc, style="dim italic"))

    # Score + reason
    score_line = Text()
    score_line.append("\n")
    score_line.append(f"  {node.score:.3f}", style="bold cyan")
    score_line.append("  ", style="")
    score_line.append(node.reason, style="dim")
    parts.append(score_line)

    console.print(Panel(
        Group(*parts) if parts else Text("(no signature)"),
        title=title,
        subtitle=subtitle,
        subtitle_align="right",
        border_style=color,
        padding=(1, 2),
    ))


def _render_neighbors(console, callers, callees, co_changes) -> None:
    from rich.table import Table
    from rich.text import Text
    from rich import box

    role_groups = [
        ("Callers",          "◄", "blue",   callers),
        ("Callees",          "►", "cyan",   callees),
        ("Co-changing",      "↔", "yellow", co_changes),
    ]

    for label, arrow, color, nodes in role_groups:
        if not nodes:
            continue
        table = Table(
            box=None, show_header=False, padding=(0, 2),
            expand=False, show_edge=False,
        )
        table.add_column("arrow", style=f"bold {color}", width=3, no_wrap=True)
        table.add_column("fqn", style="bold", no_wrap=True)
        table.add_column("loc", style="dim")
        table.add_column("kind", style="dim", width=8, no_wrap=True)

        for n in nodes:
            abbr, _ = _KIND_STYLE.get(n.kind, (n.kind[:4], "white"))
            table.add_row(
                arrow,
                n.fqn.split(".")[-1] if "." in n.fqn else n.fqn,
                f"{n.file_path}:{n.start_line}",
                abbr,
            )

        console.print(f"  [bold {color}]{label}[/bold {color}]")
        console.print(table)
        console.print()


def _render_observations(console, observations) -> None:
    from rich.panel import Panel
    from rich.text import Text

    lines = Text()
    for obs in observations:
        prefix = "[yellow]⚠ STALE[/yellow]  " if obs.stale else "[green]●[/green]  "
        lines.append_text(Text.from_markup(prefix))
        lines.append(obs.content[:300])
        if obs.stale:
            lines.append(" (stale)", style="dim")
        lines.append("\n")

    console.print(Panel(
        lines,
        title="[bold]Memory observations[/bold]",
        border_style="dim",
        padding=(1, 2),
    ))


def cmd_ask(args):
    import time
    _suppress_ml_noise()
    from rich.console import Console
    from rich.rule import Rule
    from rich.text import Text
    from beacon.schema import open_db
    from beacon.search.capsule import get_capsule
    from beacon.search.query import expand_query

    console = Console(highlight=False)

    db = args.db or str(Path(os.getcwd()) / ".beacon" / "index.db")
    conn = open_db(db)

    t0 = time.monotonic()
    with console.status(
        f"[cyan]Searching[/cyan] [bold]{args.query!r}[/bold]…", spinner="dots"
    ):
        fts_query, anchor_fqns = expand_query(conn, args.query)
        cap = get_capsule(conn, args.query, max_tokens=args.max_tokens,
                          anchor_fqns=anchor_fqns)
    elapsed = time.monotonic() - t0

    console.print()
    console.print(Rule(
        Text.assemble(("◆ ", "cyan"), (args.query, "bold white")),
        style="cyan",
    ))
    console.print()

    if not cap.nodes:
        console.print("  [yellow]No results found.[/yellow]  "
                      "[dim]Try running[/dim] [bold]beacon index[/bold] [dim]first.[/dim]")
        return

    seeds     = [n for n in cap.nodes if n.role == "seed"]
    callers   = [n for n in cap.nodes if n.role == "caller"]
    callees   = [n for n in cap.nodes if n.role == "callee"]
    co_change = [n for n in cap.nodes if n.role == "co_change"]

    limit = getattr(args, "limit", 5)

    for i, node in enumerate(seeds[:limit]):
        _render_seed(console, node, i + 1)

    if len(seeds) > limit:
        console.print(
            f"  [dim]… {len(seeds) - limit} more seed result(s) — "
            "use [bold]--limit[/bold] to show more[/dim]"
        )
        console.print()

    if callers or callees or co_change:
        _render_neighbors(console, callers, callees, co_change)

    if cap.observations:
        _render_observations(console, cap.observations)

    # Footer
    anchor_note = f" · {len(anchor_fqns)} symbol anchor(s)" if anchor_fqns else ""
    console.print(Rule(style="dim"))
    console.print(
        f"  [dim]{len(cap.nodes)} nodes · ~{cap.token_estimate:,} tokens · "
        f"{elapsed:.2f}s{anchor_note}[/dim]"
    )
    console.print()


# ── search ────────────────────────────────────────────────────────────────────

def cmd_search(args):
    from rich.console import Console
    from rich.table import Table
    from rich import box
    from beacon.schema import open_db
    from beacon.search.query import search

    console = Console(highlight=False)
    db = args.db or str(Path(os.getcwd()) / ".beacon" / "index.db")
    conn = open_db(db)
    results = search(conn, args.query, limit=args.limit)

    if not results:
        console.print("[yellow]No results.[/yellow]")
        return

    table = Table(box=box.SIMPLE, show_header=True, header_style="bold dim",
                  border_style="dim", expand=True)
    table.add_column("Score", width=7, justify="right")
    table.add_column("Kind", width=10)
    table.add_column("FQN")
    table.add_column("Location", style="dim")
    table.add_column("Reason", style="dim")

    for r in results:
        table.add_row(
            f"[cyan]{r.score:.3f}[/cyan]",
            r.kind,
            r.fqn,
            f"{r.file_path}:{r.start_line}",
            r.reason,
        )

    console.print(table)


# ── capsule ───────────────────────────────────────────────────────────────────

def cmd_capsule(args):
    from beacon.schema import open_db
    from beacon.search.capsule import get_capsule, render_capsule
    db = args.db or str(Path(os.getcwd()) / ".beacon" / "index.db")
    conn = open_db(db)
    cap = get_capsule(conn, args.query, max_tokens=args.max_tokens)
    print(render_capsule(cap))


# ── mcp ───────────────────────────────────────────────────────────────────────

def cmd_mcp(args):
    from beacon.mcp import McpServer
    workspace = Path(args.workspace or os.getcwd()).resolve()
    db = Path(args.db) if args.db else None
    McpServer(workspace=workspace, db_path=db).run()


# ── setup ────────────────────────────────────────────────────────────────────

GUARD_SCRIPT = """\
#!/bin/bash
# beacon-guard: redirect Grep/Glob/Read to Beacon MCP tools when index is ready.
# Checks for .beacon/index.db and .beacon/healthy marker written by `beacon index`.
BEACON_DIR="${CLAUDE_PROJECT_DIR:-.}/.beacon"
HEALTHY="$BEACON_DIR/healthy"
DB="$BEACON_DIR/index.db"

if [ -f "$DB" ] && [ -f "$HEALTHY" ]; then
  printf '{"hookSpecificOutput":{"hookEventName":"PreToolUse","permissionDecision":"deny","permissionDecisionReason":"beacon index is ready. Use run_pipeline or get_context_capsule instead of Grep/Glob/Read — it searches semantically and saves tokens."}}'
else
  printf '{"hookSpecificOutput":{"hookEventName":"PreToolUse","permissionDecision":"allow","permissionDecisionReason":"beacon index not ready, allowing direct search fallback."}}'
fi
exit 0
"""

HOOK_CONFIG = {
    "PreToolUse": [
        {
            "matcher": "Grep|Glob|Read",
            "hooks": [
                {
                    "type": "command",
                    "command": "$CLAUDE_PROJECT_DIR/.claude/hooks/beacon-guard.sh",
                    "timeout": 3000,
                }
            ],
        }
    ]
}


def cmd_setup(args):
    from rich.console import Console
    from rich.panel import Panel

    console = Console(highlight=False)
    _header(console)

    workspace = Path(args.workspace or os.getcwd()).resolve()
    python = sys.executable
    beacon_dir = str(Path(__file__).resolve().parent.parent)
    db_path = workspace / ".beacon" / "index.db"

    # ── 1. Write guard script ─────────────────────────────────────────────────
    hooks_dir = workspace / ".claude" / "hooks"
    hooks_dir.mkdir(parents=True, exist_ok=True)
    guard = hooks_dir / "beacon-guard.sh"
    guard.write_text(GUARD_SCRIPT)
    guard.chmod(0o755)
    console.print(f"  [green]✓[/green] Hook script  [dim]{guard}[/dim]")

    # ── 2. Hooks → project .claude/settings.json ─────────────────────────────
    settings_path = workspace / ".claude" / "settings.json"
    settings = json.loads(settings_path.read_text()) if settings_path.exists() else {}
    hooks = settings.setdefault("hooks", {})
    existing = hooks.get("PreToolUse", [])
    existing = [e for e in existing
                if not any("beacon-guard" in str(h.get("command", ""))
                           for h in e.get("hooks", []))]
    existing.extend(HOOK_CONFIG["PreToolUse"])
    hooks["PreToolUse"] = existing
    settings.pop("mcpServers", None)
    settings_path.write_text(json.dumps(settings, indent=2))
    console.print(f"  [green]✓[/green] Hooks        [dim]{settings_path}[/dim]")

    # ── 3. MCP server → ~/.claude.json ───────────────────────────────────────
    user_cfg_path = Path.home() / ".claude.json"
    user_cfg = json.loads(user_cfg_path.read_text()) if user_cfg_path.exists() else {}
    user_cfg.setdefault("mcpServers", {})["beacon"] = {
        "command": python,
        "args": ["-m", "beacon.mcp",
                 "--workspace", str(workspace),
                 "--db", str(db_path)],
        "env": {"PYTHONPATH": beacon_dir},
    }
    user_cfg_path.write_text(json.dumps(user_cfg, indent=2))
    console.print(f"  [green]✓[/green] MCP server   [dim]{user_cfg_path}[/dim]  [dim](user scope)[/dim]")

    # ── 4. Add .beacon/ to .gitignore ─────────────────────────────────────────
    gitignore = workspace / ".gitignore"
    entry = ".beacon/\n"
    if gitignore.exists():
        content = gitignore.read_text()
        if ".beacon" not in content:
            gitignore.write_text(content.rstrip("\n") + "\n" + entry)
            console.print(f"  [green]✓[/green] .gitignore   [dim]added .beacon/[/dim]")
    else:
        gitignore.write_text(entry)
        console.print(f"  [green]✓[/green] .gitignore   [dim]created[/dim]")

    console.print()
    console.print(Panel(
        f"[dim]Run:[/dim]  [bold cyan]beacon index {workspace} --no-coupling[/bold cyan]\n"
        "[dim]Then restart Claude Code to activate the MCP and hook.[/dim]",
        title="[bold]Setup complete[/bold]",
        border_style="green",
        padding=(1, 2),
    ))


# ── show-config ───────────────────────────────────────────────────────────────

def cmd_show_config(args):
    workspace = str(Path(args.workspace or os.getcwd()).resolve())
    python = sys.executable
    beacon_dir = str(Path(__file__).resolve().parent.parent)
    db = str(Path(workspace) / ".beacon" / "index.db")

    config = {
        "command": python,
        "args": ["-m", "beacon.mcp", "--workspace", workspace, "--db", db],
        "env": {"PYTHONPATH": beacon_dir},
    }
    print(json.dumps({"mcpServers": {"beacon": config}}, indent=2))


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(prog="beacon")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("index", help="Index a directory")
    p.add_argument("dir", nargs="?", help="Root directory (default: cwd)")
    p.add_argument("--db", help="Path to index.db")
    p.add_argument("--no-coupling", action="store_true", help="Skip git change coupling")
    p.add_argument("--langs", help="Comma-separated languages, e.g. python,typescript")
    p.add_argument("--yes", "-y", action="store_true", help="Skip confirmation prompt")
    p.add_argument("--configure", "-c", action="store_true",
                   help="Force model selection prompt even if config exists")

    p = sub.add_parser("ask", help="Ask the codebase a natural language question")
    p.add_argument("query")
    p.add_argument("--db")
    p.add_argument("--limit", type=int, default=5, help="Seed panels to show (default: 5)")
    p.add_argument("--max-tokens", type=int, default=8000)

    p = sub.add_parser("search", help="Search the index")
    p.add_argument("query")
    p.add_argument("--db")
    p.add_argument("--limit", type=int, default=10)

    p = sub.add_parser("capsule", help="Generate a context capsule")
    p.add_argument("query")
    p.add_argument("--db")
    p.add_argument("--max-tokens", type=int, default=8000)

    p = sub.add_parser("mcp", help="Start MCP stdio server")
    p.add_argument("--workspace")
    p.add_argument("--db")

    p = sub.add_parser("show-config", help="Print MCP config for AI agents")
    p.add_argument("--workspace")

    p = sub.add_parser("setup", help="Install hook + MCP config for a project")
    p.add_argument("workspace", nargs="?", help="Project root (default: cwd)")

    args = parser.parse_args()
    {"index": cmd_index, "ask": cmd_ask, "search": cmd_search,
     "capsule": cmd_capsule, "mcp": cmd_mcp,
     "show-config": cmd_show_config, "setup": cmd_setup}[args.cmd](args)


if __name__ == "__main__":
    main()
