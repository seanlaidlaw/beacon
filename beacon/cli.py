"""
beacon CLI

Commands:
    index          [<dir>]   — scan + index a codebase
    search         <query>   — search the index
    capsule        <query>   — generate a context capsule
    run-benchmark           — measure token savings vs grep baseline
    mcp                     — start MCP stdio server
    setup                   — install hook + MCP config for a project
    show-config             — print MCP config for AI agents
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
        DownloadColumn, TransferSpeedColumn,
    )
    from rich.table import Table
    from rich.text import Text
    from rich import box
    from rich.prompt import Confirm

    _suppress_ml_noise()
    _suppress_tqdm()
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
    with _status(console, "[cyan]Scanning files…[/cyan]"):
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
                        signature, docstring, body_preview, is_exported, is_test, repo_alias)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                       ON CONFLICT(fqn) DO UPDATE SET
                         name=excluded.name, file_path=excluded.file_path,
                         kind=excluded.kind, start_line=excluded.start_line,
                         end_line=excluded.end_line, signature=excluded.signature,
                         docstring=excluded.docstring, body_preview=excluded.body_preview,
                         is_exported=excluded.is_exported, is_test=excluded.is_test
                       RETURNING id""",
                    (sym.name, sym.fqn, sym.file_path, sym.kind,
                     sym.start_line, sym.end_line, sym.signature, sym.docstring,
                     sym.body_preview, int(sym.is_exported), int(sym.is_test), "primary"),
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
        with _status(console, f"[cyan]Resolving {len(all_edges):,} call edges…[/cyan]"):
            inserted = _resolve_edges_silent(conn, all_edges)
        edge_total = conn.execute("SELECT COUNT(*) FROM edges").fetchone()[0]
        console.print(
            f"  [green]✓[/green] Edges  [bold]{inserted:,}[/bold] new  "
            f"[dim]({edge_total:,} total)[/dim]"
        )
        console.print()

    # ── Phase 5: TF-IDF + dense embeddings ───────────────────────────────────
    if changed_node_ids:
        with _status(console, "[cyan]Building TF-IDF embeddings…[/cyan]"):
            embedder.build_incremental(conn, changed_node_ids)
        console.print("  [green]✓[/green] TF-IDF embeddings")

        model_short = current_model.split("/")[-1]

        # ── Step 1: ensure model is on disk (may need to download) ───────────
        enc = embedder.get_encoder()
        if not embedder.is_model_cached(current_model):
            console.print(
                f"  [yellow]↓[/yellow]  Downloading [cyan]{model_short}[/cyan]  "
                f"[dim](first run)[/dim]"
            )
            with Progress(
                TextColumn("  [dim]{task.description}[/dim]"),
                BarColumn(bar_width=28, style="cyan", complete_style="green"),
                DownloadColumn(),
                TransferSpeedColumn(),
                TimeRemainingColumn(),
                console=console,
                transient=True,
            ) as dl_progress:
                dl_task = dl_progress.add_task(model_short, total=None)
                _patch_tqdm_with(_make_rich_tqdm_class(dl_progress, dl_task))
                try:
                    load_ok = enc._load()
                finally:
                    _suppress_tqdm()  # re-silence for the rest of indexing

            if not load_ok:
                console.print(
                    f"  [red]✗[/red]  Model download failed"
                    + (f"  [dim]{enc.error}[/dim]" if enc.error else "")
                )
                console.print()
            else:
                console.print(f"  [green]✓[/green] Model downloaded")
        else:
            # Already cached — load from disk (fast, but surface any error)
            with _status(console, f"[cyan]Loading {model_short}…[/cyan]"):
                load_ok = enc._load()
            if not load_ok:
                console.print(
                    f"  [red]✗[/red]  Model failed to load"
                    + (f"  [dim]{enc.error}[/dim]" if enc.error else "")
                )
                console.print()

        # ── Step 2: build embeddings (skip if load failed) ───────────────────
        if load_ok:
            with _status(console, f"[cyan]Building dense embeddings ({model_short})…[/cyan]"):
                n_stored, embed_err = embedder.build_dense_incremental(conn, changed_node_ids)

            if embed_err:
                console.print(
                    f"  [red]✗[/red]  Dense embeddings failed  [dim]{embed_err}[/dim]"
                )
            else:
                dense_count = conn.execute(
                    "SELECT COUNT(*) FROM node_embeddings_dense"
                ).fetchone()[0]
                console.print(
                    f"  [green]✓[/green] Dense embeddings  "
                    f"[dim]model={model_short}  n={dense_count:,}[/dim]"
                )
        console.print()

    # ── Phase 6: Git change coupling ─────────────────────────────────────────
    if not args.no_coupling:
        with _status(console, "[cyan]Computing git change coupling…[/cyan]"):
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
    with _status(console, "[cyan]Rebuilding FTS5 index…[/cyan]"):
        conn.execute("INSERT INTO nodes_fts(nodes_fts) VALUES('rebuild')")
        conn.execute("INSERT INTO observations_fts(observations_fts) VALUES('rebuild')")
        conn.commit()
    fts_count = conn.execute("SELECT COUNT(*) FROM nodes_fts").fetchone()[0]
    console.print(f"  [green]✓[/green] FTS5 index  [dim]{fts_count:,} entries[/dim]")
    console.print()

    _finish(conn, root, db_path, console)


def _resolve_edges_silent(conn, all_edges) -> int:
    """Resolve call edges without any print output. Returns inserted count."""
    from beacon.indexer.indexer import _resolve_call_edges
    _resolve_call_edges(conn, all_edges)
    conn.commit()
    return conn.execute("SELECT COUNT(*) FROM edges").fetchone()[0]


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

# (display_label, rich_color)  — full words, soft palette
_KIND_STYLE: dict[str, tuple[str, str]] = {
    "function":  ("function",  "steel_blue1"),
    "method":    ("method",    "dark_cyan"),
    "class":     ("class",     "medium_orchid"),
    "struct":    ("struct",    "medium_orchid"),
    "interface": ("interface", "medium_spring_green"),
    "type":      ("type",      "medium_spring_green"),
    "variable":  ("variable",  "light_goldenrod2"),
    "constant":  ("constant",  "light_goldenrod2"),
    "module":    ("module",    "grey70"),
    "heading":   ("heading",   "grey70"),
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


def _status(console, msg: str, spinner: str = "dots"):
    """Drop-in for ``console.status()`` that works with terminal recorders.

    Rich's Status widget clears its spinner line using cursor-up + erase-to-EOL
    (``\\x1b[1A\\r\\x1b[K``).  Tools like termframe don't handle those sequences
    correctly, leaving ghost spinner text in the captured SVG.

    Set ``BEACON_NO_SPINNERS=1`` to replace every spinner with a plain static
    print so recordings render cleanly — normal terminal use is unaffected.
    """
    import contextlib, os as _os, re as _re
    if _os.environ.get("BEACON_NO_SPINNERS"):
        plain = _re.sub(r'\[[^\]]*\]', '', msg).strip()
        console.print(f"  [dim]·[/dim] {plain}")
        return contextlib.nullcontext()
    return console.status(msg, spinner=spinner)


def _suppress_tqdm() -> None:
    """Replace tqdm progress bars with a no-op so they don't corrupt rich output."""
    try:
        import tqdm
        class _SilentTqdm:
            def __init__(self, *a, **kw): pass
            def __iter__(self): return iter([])
            def update(self, *a, **kw): pass
            def close(self): pass
            def set_description(self, *a, **kw): pass
            def __enter__(self): return self
            def __exit__(self, *a): pass
        tqdm.tqdm = _SilentTqdm
        import tqdm.auto
        tqdm.auto.tqdm = _SilentTqdm
    except Exception:
        pass


def _make_rich_tqdm_class(progress_bar, task_id):
    """Return a tqdm-compatible class that forwards download progress to a Rich task.

    Each tqdm instantiation resets the task (HF downloads one file per tqdm),
    forwarding the filename via ``desc`` and byte count via ``total`` / ``update``.
    """
    class _RichTqdm:
        def __init__(self, iterable=None, *, desc=None, total=None, **kwargs):
            self._iterable = iterable
            # Strip path separators — desc is usually the full cache path
            short = Path(desc).name if desc and ('/' in desc or '\\' in desc) else (desc or "")
            progress_bar.update(task_id, description=short, completed=0, total=total)

        def update(self, n=1):
            progress_bar.update(task_id, advance=n)

        def __enter__(self): return self
        def __exit__(self, *args): self.close()
        def __iter__(self):
            for item in (self._iterable or []):
                yield item
                self.update(1)
        def close(self): pass
        def set_postfix(self, *args, **kwargs): pass
        def set_postfix_str(self, *args, **kwargs): pass
        def set_description(self, desc=None, **kwargs):
            if desc:
                progress_bar.update(task_id, description=str(desc))
        @classmethod
        def write(cls, *args, **kwargs): pass

    return _RichTqdm


def _patch_tqdm_with(cls) -> None:
    """Install ``cls`` as the active tqdm implementation everywhere HF Hub looks."""
    try:
        import tqdm
        tqdm.tqdm = cls
        import tqdm.auto
        tqdm.auto.tqdm = cls
    except Exception:
        pass


def _suppress_ml_noise() -> None:
    """Silence noisy transformers/sentence-transformers log and warning output."""
    import logging
    import warnings
    import os
    # Set env vars before any lazy imports pick them up
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ["TRANSFORMERS_VERBOSITY"] = "error"
    os.environ["SENTENCE_TRANSFORMERS_LOGLEVEL"] = "ERROR"
    os.environ["BEACON_QUIET"] = "1"
    os.environ.setdefault("TRANSFORMERS_ATTN_IMPLEMENTATION", "eager")
    # Suppress Python warnings from these libraries
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=UserWarning, module="transformers")
    # Set Python logger levels for all relevant namespaces
    for name in ("transformers", "sentence_transformers", "huggingface_hub",
                 "transformers.modeling_utils", "transformers.configuration_utils",
                 "transformers.tokenization_utils_base"):
        logging.getLogger(name).setLevel(logging.ERROR)
    # Use transformers' own verbosity API if already imported
    try:
        import transformers
        transformers.logging.set_verbosity_error()
    except Exception:
        pass


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


# ── Interactive ask TUI ────────────────────────────────────────────────────────

def _getch() -> str:
    """
    Read one keypress from stdin without echoing.

    Returns a single character for printable keys, or one of the named
    constants 'UP', 'DOWN', 'LEFT', 'RIGHT' for arrow keys.
    Returns 'q' for Ctrl+C, Ctrl+D, and bare Escape.

    Uses os.read(fd, 1) — raw syscall with no Python buffering — so that
    select() on the same fd reliably detects whether the rest of an escape
    sequence has arrived.  sys.stdin.buffer.read() has a read-ahead buffer
    that drains the fd before select() can see the bytes.
    """
    import sys, os, select
    try:
        import tty, termios
        fd = sys.stdin.fileno()
        old = termios.tcgetattr(fd)
        try:
            tty.setraw(fd)
            b = os.read(fd, 1)
            if b in (b'\x03', b'\x04'):          # Ctrl+C / Ctrl+D
                return 'q'
            if b == b'\x1b':
                # Wait up to 100 ms for the rest of a CSI escape sequence.
                # A bare Escape key produces no further bytes → return 'q'.
                if select.select([fd], [], [], 0.1)[0]:
                    b2 = os.read(fd, 1)
                    if b2 == b'[':
                        b3 = os.read(fd, 1)
                        return {b'A': 'UP', b'B': 'DOWN',
                                b'C': 'RIGHT', b'D': 'LEFT'}.get(b3, 'q')
                return 'q'
            return b.decode('utf-8', errors='replace')
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)
    except Exception:
        return 'q'


def _source_content(node, conn, root: Path):
    """
    Return a Syntax renderable with the actual source lines for *node*.
    Falls back to the indexed signature if the file cannot be read.
    Used inside the source panel in the interactive view.
    """
    from rich.syntax import Syntax
    from rich.text import Text

    # Fetch end_line from DB (not stored in CapsuleNode)
    row = conn.execute(
        "SELECT end_line FROM nodes WHERE fqn=?", (node.fqn,)
    ).fetchone()
    end_line = row[0] if row and row[0] else None

    file_path = root / node.file_path
    start = max(0, (node.start_line or 1) - 1)

    try:
        lines = file_path.read_text(errors="replace").splitlines()
        snippet_lines = lines[start:end_line] if end_line else lines[start:start + 60]
        snippet = "\n".join(snippet_lines)
        return Syntax(
            snippet,
            _lexer(node.file_path),
            theme="nord",
            line_numbers=True,
            start_line=node.start_line or 1,
            background_color="default",
            word_wrap=False,
        )
    except Exception:
        sig = node.signature.strip() if node.signature else "(no source)"
        return Syntax(sig, _lexer(node.file_path), theme="nord",
                      background_color="default")


def _source_panel(node, conn, root: Path):
    """Standalone Panel wrapping source content — used in non-interactive ask output."""
    from rich.panel import Panel
    _, color = _KIND_STYLE.get(node.kind, ("?", "white"))
    return Panel(_source_content(node, conn, root), border_style=color, padding=(0, 1))


def _results_table(seeds, selected: int):
    """
    Build a rich Table listing all result rows.
    The selected row is highlighted; all others are dim.
    Returns a Table renderable.
    """
    from rich.table import Table
    from rich.text import Text
    from rich import box

    tbl = Table(
        box=None,
        show_header=False,
        padding=(0, 1),
        expand=False,
        show_edge=False,
    )
    tbl.add_column("num",   width=4,  no_wrap=True, justify="right")
    tbl.add_column("kind",  width=11, no_wrap=True)
    tbl.add_column("name",  width=44, no_wrap=True)
    tbl.add_column("score", width=5,  no_wrap=True, justify="right")
    tbl.add_column("loc",   no_wrap=True)
    tbl.add_column("mark",  width=1,  no_wrap=True)

    for i, node in enumerate(seeds):
        label, color = _KIND_STYLE.get(node.kind, (node.kind, "white"))
        short = node.fqn.split("::")[-1] if "::" in node.fqn else node.fqn
        if len(short) > 43:
            short = short[:41] + "…"
        fp = f"{Path(node.file_path).name}:{node.start_line}"
        sel = (i == selected)

        if sel:
            tbl.add_row(
                Text(f"[{i+1}]",    style="bold cyan"),
                Text(label,          style=f"bold {color}"),
                Text(short,          style="bold white"),
                Text(f"{node.score:.3f}", style="bold cyan"),
                Text(fp,             style="cyan"),
                Text("▶",            style="bold cyan"),
            )
        else:
            tbl.add_row(
                Text(f"[{i+1}]",    style="dim"),
                Text(label,          style=f"dim {color}"),
                Text(short,          style="dim white"),
                Text(f"{node.score:.3f}", style="dim"),
                Text(fp,             style="dim"),
                Text(""),
            )

    return tbl


def _render_interactive_view(query: str, cap, seeds, selected: int,
                              conn, root: Path):
    from rich.console import Group
    from rich.rule import Rule
    from rich.text import Text
    from rich.panel import Panel

    parts = []

    # ── Query rule ────────────────────────────────────────────────────────────
    parts.append(Rule(
        Text.assemble(("◆ ", "cyan"), (query, "bold white")),
        style="cyan",
    ))
    parts.append(Text(""))

    # ── Result table (all rows, selected highlighted) ─────────────────────────
    parts.append(_results_table(seeds, selected))
    parts.append(Text(""))

    # ── Source panel for the selected result ──────────────────────────────────
    node = seeds[selected]
    label, color = _KIND_STYLE.get(node.kind, (node.kind, "white"))
    short = node.fqn.split("::")[-1] if "::" in node.fqn else node.fqn
    panel_title = Text.assemble(
        (f" {label} ", f"bold reverse {color}"),
        ("  ", ""),
        (short, "bold white"),
        ("  ", ""),
        (f"{node.file_path}:{node.start_line}", "dim"),
    )
    parts.append(Panel(
        _source_content(node, conn, root),
        title=panel_title,
        border_style=color,
        padding=(0, 0),
    ))

    # ── Footer stats ──────────────────────────────────────────────────────────
    parts.append(Text(""))
    callers_n = sum(1 for n in cap.nodes if n.role == "caller")
    callees_n = sum(1 for n in cap.nodes if n.role == "callee")
    co_n      = sum(1 for n in cap.nodes if n.role == "co_change")
    stats = Text()
    if callers_n:
        stats.append(f"  ◄ {callers_n} callers", style="dim blue")
    if callees_n:
        stats.append(f"  ► {callees_n} callees", style="dim cyan")
    if co_n:
        stats.append(f"  ↔ {co_n} co-change", style="dim yellow")
    stats.append(f"  ·  ~{cap.token_estimate:,} tokens", style="dim")
    parts.append(stats)

    # ── Hint bar ──────────────────────────────────────────────────────────────
    parts.append(Text(""))
    hints = Text()
    hints.append("  [↑↓]", style="bold cyan")
    hints.append(" navigate  ", style="dim")
    hints.append("[1-9]", style="bold cyan")
    hints.append(" jump  ", style="dim")
    hints.append("[/]", style="bold cyan")
    hints.append(" new search  ", style="dim")
    hints.append("[q]", style="bold cyan")
    hints.append(" quit", style="dim")
    parts.append(hints)

    return Group(*parts)


_HISTORY_FILE = Path.home() / ".cache" / "beacon" / "search_history"


def _setup_readline() -> None:
    """
    Configure Python readline with persistent per-session search history.

    Uses Python's built-in readline module (backed by libedit on macOS or
    GNU readline on Linux). History is saved to ~/.cache/beacon/search_history
    so up-arrow recall works across sessions.

    Must be called before the first input() prompt.
    """
    try:
        import readline
        _HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
        readline.set_history_length(500)
        try:
            readline.read_history_file(str(_HISTORY_FILE))
        except (FileNotFoundError, OSError):
            pass
        # Ensure history is flushed on normal exit
        import atexit
        atexit.register(readline.write_history_file, str(_HISTORY_FILE))
    except ImportError:
        pass  # Windows without pyreadline — silently skip


def _readline_input(console) -> str:
    """
    Show a styled search prompt and read a line using Python's input().

    Using input() (not Prompt.ask / console.input) ensures readline key
    bindings — including up/down arrow history — work correctly.
    The styled prompt is printed via rich first; input() is then called
    with an empty string so readline attaches to the right place.
    """
    # Print the visible prompt via rich (no newline)
    console.print("  [bold cyan]❯[/bold cyan] [dim]search your codebase[/dim]",
                  end="")
    try:
        # input("") keeps the cursor on the same line and activates readline
        line = input(" ").strip()
    except (EOFError, KeyboardInterrupt):
        line = "q"
    # Save to persistent history immediately
    try:
        import readline
        if line and line.lower() not in ("q", "quit", "exit"):
            readline.add_history(line)
            readline.write_history_file(str(_HISTORY_FILE))
    except Exception:
        pass
    return line


def cmd_ask_interactive(conn, root: Path, console) -> None:
    """Full interactive TUI: search bar → live results → key-driven expand."""
    from rich.live import Live
    from beacon.search.capsule import get_capsule
    from beacon.search.query import expand_query
    from beacon.indexer import embedder as _emb

    _setup_readline()

    while True:
        console.clear()
        _header(console)
        console.print()

        query = _readline_input(console)
        console.print()   # newline after input

        if not query or query.lower() in ("q", "quit", "exit"):
            return

        # ── Run search ────────────────────────────────────────────────────────
        with console.status(
            f"[cyan]Searching[/cyan] [bold]{query!r}[/bold]…", spinner="dots"
        ):
            _, anchor_fqns = expand_query(conn, query)
            cap = get_capsule(conn, query, max_tokens=8000, anchor_fqns=anchor_fqns)

        seeds = [n for n in cap.nodes if n.role == "seed"]
        if not seeds:
            console.print("[yellow]  No results found.[/yellow]")
            console.print()
            continue  # back to prompt

        # Cap at 9 so single-digit keys always map cleanly
        seeds = seeds[:9]
        selected = 0   # first result is shown expanded by default

        def _render(sel):
            return _render_interactive_view(query, cap, seeds, sel, conn, root)

        # ── Live display + keypress loop ──────────────────────────────────────
        with Live(_render(selected), console=console, auto_refresh=False) as live:
            while True:
                ch = _getch()

                if ch in ('q',):
                    return          # quit entirely

                if ch in ('/', 's', '\r', '\n'):
                    break           # back to search prompt

                if ch == 'UP' or ch == 'k':
                    selected = (selected - 1) % len(seeds)
                    live.update(_render(selected), refresh=True)

                elif ch == 'DOWN' or ch == 'j':
                    selected = (selected + 1) % len(seeds)
                    live.update(_render(selected), refresh=True)

                elif ch.isdigit():
                    idx = int(ch) - 1
                    if 0 <= idx < len(seeds):
                        selected = idx
                        live.update(_render(selected), refresh=True)
        # Loop → new search


def cmd_ask(args):
    import time
    _suppress_ml_noise()
    _suppress_tqdm()
    from rich.console import Console
    from rich.rule import Rule
    from rich.text import Text
    from beacon.schema import open_db
    from beacon.search.capsule import get_capsule
    from beacon.search.query import expand_query

    console = Console(highlight=False)

    db = args.db or str(Path(os.getcwd()) / ".beacon" / "index.db")
    db_path = Path(db).resolve()
    # Root is two levels up from .beacon/index.db
    root = db_path.parent.parent if db_path.parent.name == ".beacon" else db_path.parent
    conn = open_db(db)

    # No query → interactive TUI
    if not getattr(args, "query", None):
        cmd_ask_interactive(conn, root, console)
        return

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


# ── run-benchmark ─────────────────────────────────────────────────────────────

def cmd_benchmark(args):
    """Run the token-savings benchmark and print a rich summary table."""
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text
    from rich import box
    from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

    from beacon.benchmark import run_benchmark, summary_stats, QUERIES

    console = Console(highlight=False)
    _header(console)

    root = str(Path(args.root or os.getcwd()).resolve())
    output_path = Path(args.output)

    console.print(f"  [dim]root  [/dim] [bold]{root}[/bold]")
    console.print(f"  [dim]output[/dim] [dim]{output_path}[/dim]")
    console.print(f"  [dim]queries[/dim] [bold]{len(QUERIES)}[/bold]")
    console.print()

    results = []

    with Progress(
        SpinnerColumn(style="cyan"),
        TextColumn("[progress.description]{task.description}"),
        TimeElapsedColumn(),
        console=console,
        transient=False,
    ) as progress:
        task = progress.add_task("Running queries…", total=len(QUERIES))

        def on_result(r):
            results.append(r)
            status = "[green]✓[/green]" if r["beacon_recall"] else "[red]✗[/red]"
            label = r["query"][:52] + "…" if len(r["query"]) > 52 else r["query"]
            progress.update(
                task,
                description=f"[dim]{label}[/dim] → {status} [cyan]{r['pct_saved']:.0f}%[/cyan] saved",
                advance=1,
            )

        run_benchmark(root, output_path=output_path, on_result=on_result)

    stats = summary_stats(results)

    # ── Results table ─────────────────────────────────────────────────────────
    console.print()
    table = Table(
        box=box.SIMPLE_HEAD,
        show_header=True,
        header_style="bold cyan",
        padding=(0, 1),
        expand=False,
    )
    table.add_column("#", justify="right", width=3, style="dim")
    table.add_column("Query type", width=30)
    table.add_column("Beacon", justify="right", width=8)
    table.add_column("Baseline", justify="right", width=10)
    table.add_column("Savings", justify="right", width=8)
    table.add_column("% Saved", justify="right", width=8)
    table.add_column("Recall", width=14)

    for r in results:
        beacon_ok = r["beacon_recall"]
        recall_text = Text()
        recall_text.append("✓ " if beacon_ok else "✗ ", style="green" if beacon_ok else "red")
        recall_text.append("beacon", style="green bold" if beacon_ok else "red")
        recall_text.append("  ")
        base_ok = r["baseline_recall"]
        recall_text.append("✓" if base_ok else "✗", style="green" if base_ok else "dim red")
        recall_text.append(" base", style="dim")

        pct_style = "green bold" if r["pct_saved"] >= 80 else ("yellow" if r["pct_saved"] >= 50 else "red")
        table.add_row(
            str(r["id"]),
            r["type"],
            f"{r['beacon_tokens']:,}",
            f"{r['baseline_tokens']:,}",
            f"{r['savings_ratio']:.1f}×",
            Text(f"{r['pct_saved']:.0f}%", style=pct_style),
            recall_text,
        )

    console.print(table)

    # ── Summary panel ─────────────────────────────────────────────────────────
    summary_text = Text()
    summary_text.append(f"  {stats['overall_pct_saved']:.0f}%", style="bold green")
    summary_text.append(" fewer tokens overall  ", style="")
    summary_text.append(f"({stats['total_beacon_tokens']:,}", style="dim")
    summary_text.append(" beacon vs ", style="dim")
    summary_text.append(f"{stats['total_baseline_tokens']:,}", style="dim")
    summary_text.append(" baseline)  ", style="dim")
    summary_text.append(f"{stats['avg_savings_ratio']:.1f}×", style="bold cyan")
    summary_text.append(" avg  |  ", style="dim")
    summary_text.append(f"{stats['beacon_recall_count']}/{stats['total_queries']}", style="bold")
    summary_text.append(" recall  |  ", style="dim")
    summary_text.append(f"{stats['beacon_wins']}/{stats['total_queries']}", style="bold")
    summary_text.append(" token wins", style="dim")

    console.print(Panel(
        summary_text,
        title="[bold green]Benchmark complete[/bold green]",
        border_style="green",
        padding=(0, 1),
    ))
    console.print(f"\n  [dim]Detailed results →[/dim] [bold]{output_path}[/bold]")

    failures = [r for r in results if not r["beacon_recall"]]
    if failures:
        console.print(f"\n  [red]Recall failures ({len(failures)}):[/red]")
        for r in failures:
            console.print(f"    [{r['id']}] {r['query'][:60]}  [dim](type: {r['type']})[/dim]")


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

    # ── 2. Hooks + MCP server → project .claude/settings.json ───────────────
    #
    # MCP config goes in the PROJECT settings (not ~/.claude.json) so that
    # each project gets its own beacon instance pointing at its own index.db.
    # Writing to the global user config caused the last-setup project's index
    # to be used for ALL projects.
    settings_path = workspace / ".claude" / "settings.json"
    settings = json.loads(settings_path.read_text()) if settings_path.exists() else {}

    # Hooks
    hooks = settings.setdefault("hooks", {})
    existing = hooks.get("PreToolUse", [])
    existing = [e for e in existing
                if not any("beacon-guard" in str(h.get("command", ""))
                           for h in e.get("hooks", []))]
    existing.extend(HOOK_CONFIG["PreToolUse"])
    hooks["PreToolUse"] = existing

    # MCP server (project-scoped)
    settings.setdefault("mcpServers", {})["beacon"] = {
        "command": python,
        "args": ["-m", "beacon.mcp",
                 "--workspace", str(workspace),
                 "--db", str(db_path)],
        "env": {"PYTHONPATH": beacon_dir},
    }

    settings_path.write_text(json.dumps(settings, indent=2))
    console.print(f"  [green]✓[/green] Hooks        [dim]{settings_path}[/dim]")
    console.print(f"  [green]✓[/green] MCP server   [dim]{settings_path}[/dim]  [dim](project scope)[/dim]")

    # ── 3. Remove any stale global entry from ~/.claude.json ─────────────────
    user_cfg_path = Path.home() / ".claude.json"
    if user_cfg_path.exists():
        user_cfg = json.loads(user_cfg_path.read_text())
        if user_cfg.get("mcpServers", {}).pop("beacon", None) is not None:
            user_cfg_path.write_text(json.dumps(user_cfg, indent=2))
            console.print(f"  [green]✓[/green] Removed stale global beacon entry from [dim]{user_cfg_path}[/dim]")

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
        "[dim]Then restart Claude Code to activate the MCP and hook.[/dim]\n\n"
        "[dim]The MCP server is scoped to this project — other projects are unaffected.[/dim]",
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
    output = {"mcpServers": {"beacon": config}}
    print(json.dumps(output, indent=2))
    print(
        "\n# Add the above to YOUR PROJECT'S .claude/settings.json (not ~/.claude.json)\n"
        "# so each project uses its own beacon index.\n"
        "# Or just run:  beacon setup",
        file=__import__("sys").stderr,
    )


# ── logs ──────────────────────────────────────────────────────────────────────

def cmd_logs(args):
    """Pretty-print MCP request logs from .beacon/logs/."""
    from rich.console import Console
    from rich.table import Table
    from rich.text import Text

    console = Console(highlight=False)
    root = Path(args.dir or os.getcwd()).resolve()
    log_dir = root / ".beacon" / "logs"

    if not log_dir.exists():
        console.print(f"[yellow]No logs found at {log_dir}[/yellow]")
        return

    log_files = sorted(log_dir.glob("mcp_*.jsonl"), reverse=True)
    if not log_files:
        console.print(f"[yellow]No MCP log files in {log_dir}[/yellow]")
        return

    # Default: most recent session; --all shows every session
    files_to_show = log_files if args.all else log_files[:1]

    for log_file in files_to_show:
        console.rule(f"[bold]{log_file.name}[/bold]")
        entries = []
        with open(log_file) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        entries.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass

        # Print session start info
        for e in entries:
            if e.get("event") == "session_start":
                console.print(f"  [dim]workspace:[/dim] {e.get('workspace')}")
                console.print(f"  [dim]started:  [/dim] {e.get('ts')}")
                break

        # Table of tool calls
        calls = [e for e in entries if "tool" in e]
        if not calls:
            console.print("  [dim](no tool calls)[/dim]")
            continue

        table = Table(show_header=True, header_style="bold", box=None, padding=(0, 1))
        table.add_column("#", style="dim", width=3)
        table.add_column("Tool", style="cyan", width=22)
        table.add_column("Query / Args", width=52)
        table.add_column("ms", justify="right", style="dim", width=6)
        table.add_column("~tok", justify="right", width=5)
        table.add_column("Status", width=7)

        for i, e in enumerate(calls, 1):
            tool = e.get("tool", "?")
            a = e.get("args", {})

            # Extract the most descriptive argument
            query_str = (
                a.get("task") or a.get("query") or a.get("symbol_fqn")
                or a.get("start") or a.get("names") or str(a)
            )
            if isinstance(query_str, list):
                query_str = ", ".join(str(x) for x in query_str)
            query_str = str(query_str)[:50]

            ms = str(e.get("elapsed_ms", "?"))
            tok = str(e.get("result_tokens_approx", "?"))
            status = "[red]ERR[/red]" if e.get("error") else "[green]OK[/green]"

            table.add_row(str(i), tool, query_str, ms, tok, status)

        console.print(table)

        if args.verbose:
            console.print()
            for i, e in enumerate(calls, 1):
                console.rule(f"Call {i}: {e.get('tool')}", style="dim")
                console.print(f"  [dim]args:[/dim] {json.dumps(e.get('args', {}), indent=2)}")
                if e.get("error"):
                    console.print(f"  [red]error:[/red] {e['error']}")
                elif e.get("result_preview"):
                    console.print(f"  [dim]result (first 300 chars):[/dim]")
                    console.print(f"  {e['result_preview'][:300]}")
                console.print()


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
    p.add_argument("query", nargs="?", default=None,
                   help="Query (omit for interactive TUI mode)")
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

    p = sub.add_parser("run-benchmark", help="Measure token savings vs grep baseline")
    p.add_argument("--root", default=None, help="Path to indexed codebase (default: cwd)")
    p.add_argument("--output", default="benchmark_results.json",
                   help="Path to write JSON results (default: benchmark_results.json)")

    p = sub.add_parser("logs", help="Show MCP request logs for a project")
    p.add_argument("dir", nargs="?", help="Project root (default: cwd)")
    p.add_argument("--all", action="store_true", help="Show all sessions, not just the latest")
    p.add_argument("--verbose", "-v", action="store_true",
                   help="Print full args and result preview for each call")

    args = parser.parse_args()
    {"index": cmd_index, "ask": cmd_ask, "search": cmd_search,
     "capsule": cmd_capsule, "run-benchmark": cmd_benchmark, "mcp": cmd_mcp,
     "show-config": cmd_show_config, "setup": cmd_setup, "logs": cmd_logs}[args.cmd](args)


if __name__ == "__main__":
    main()
