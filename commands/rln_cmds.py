"""
Rln-specific commands: ssc (pip wrapper), doedit (script editor),
python (run Python code inline), copy (curl-like downloader)
"""

import os
import re
import sys
import subprocess
from rich.console import Console

from commands.state import AppState
from commands.parse_helpers import parse_command_line


# ──────────────────────────────────────────────
#  ssc install / ssc remove / ssc list
# ──────────────────────────────────────────────

def cmd_ssc(rest: str, state: AppState, console: Console):
    """
    ssc install package1 [package2 ...]    — Install Python packages (pip install)
    ssc remove package1 [package2 ...]     — Uninstall packages (pip uninstall)
    ssc list                               — List installed packages
    ssc search keyword                     — Search PyPI for packages
    ssc update package                     — Update a package

    This wraps pip to provide econometric ssc install syntax.
    """
    parts = rest.strip().split(None, 1)
    if not parts:
        console.print("[red]Syntax: ssc install|remove|list|search|update package[/red]")
        return

    subcmd = parts[0].lower()
    args = parts[1] if len(parts) > 1 else ""

    if subcmd == "install":
        if not args:
            console.print("[red]Syntax: ssc install package_name[/red]")
            return
        packages = args.split()
        console.print(f"[dim]Installing: {', '.join(packages)}...[/dim]")
        try:
            result = subprocess.run(
                [sys.executable, "-m", "pip", "install"] + packages,
                capture_output=True, text=True, timeout=120
            )
            if result.returncode == 0:
                console.print(f"[green]Successfully installed: {', '.join(packages)}[/green]")
                # Show key info from output
                for line in result.stdout.split("\n"):
                    if "Successfully" in line or "already satisfied" in line:
                        console.print(f"  [dim]{line.strip()}[/dim]")
            else:
                console.print(f"[red]Installation failed:[/red]")
                for line in result.stderr.split("\n")[-5:]:
                    if line.strip():
                        console.print(f"  [red]{line.strip()}[/red]")
        except subprocess.TimeoutExpired:
            console.print("[red]Installation timed out (120s). Try from terminal.[/red]")
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")

    elif subcmd in ("remove", "uninstall"):
        if not args:
            console.print("[red]Syntax: ssc remove package_name[/red]")
            return
        packages = args.split()
        console.print(f"[dim]Removing: {', '.join(packages)}...[/dim]")
        try:
            result = subprocess.run(
                [sys.executable, "-m", "pip", "uninstall", "-y"] + packages,
                capture_output=True, text=True, timeout=60
            )
            if result.returncode == 0:
                console.print(f"[green]Removed: {', '.join(packages)}[/green]")
            else:
                console.print(f"[red]Removal failed: {result.stderr.strip()}[/red]")
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")

    elif subcmd == "list":
        try:
            result = subprocess.run(
                [sys.executable, "-m", "pip", "list", "--format=columns"],
                capture_output=True, text=True, timeout=30
            )
            console.print(result.stdout)
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")

    elif subcmd == "search":
        if not args:
            console.print("[red]Syntax: ssc search keyword[/red]")
            return
        console.print(f"[dim]Search PyPI: https://pypi.org/search/?q={args}[/dim]")
        console.print(f"[yellow]Tip: pip search is disabled. Visit the URL above or try:[/yellow]")
        console.print(f"  ssc install {args}")

    elif subcmd == "update":
        if not args:
            console.print("[red]Syntax: ssc update package_name[/red]")
            return
        packages = args.split()
        console.print(f"[dim]Updating: {', '.join(packages)}...[/dim]")
        try:
            result = subprocess.run(
                [sys.executable, "-m", "pip", "install", "--upgrade"] + packages,
                capture_output=True, text=True, timeout=120
            )
            if result.returncode == 0:
                console.print(f"[green]Updated: {', '.join(packages)}[/green]")
            else:
                console.print(f"[red]Update failed: {result.stderr.strip()}[/red]")
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")

    else:
        console.print(f"[red]Unknown ssc subcommand: {subcmd}[/red]")
        console.print("[dim]Available: install, remove, list, search, update[/dim]")


# ──────────────────────────────────────────────
#  doedit — Script editor with syntax highlighting
# ──────────────────────────────────────────────

def cmd_doedit(rest: str, state: AppState, console: Console):
    """
    doedit "filename.rln"
    doedit                   — Open a new untitled script file

    Opens a terminal-based script editor with syntax highlighting.
    """
    filepath = rest.strip().strip("\"'") if rest.strip() else None

    # Try TUI editor (Textual)
    try:
        from tui.doeditor import launch_editor
        launch_editor(filepath, state, console)
        return
    except ImportError:
        pass

    # Fallback: simple line editor
    _simple_editor(filepath, state, console)


def _simple_editor(filepath, state, console):
    """Basic line-based script editor when Textual is not available."""
    lines = []

    if filepath and os.path.exists(filepath):
        with open(filepath, "r", encoding="utf-8") as f:
            lines = f.readlines()
        lines = [l.rstrip("\n\r") for l in lines]
        console.print(f"[green]Loaded: {filepath} ({len(lines)} lines)[/green]")
    else:
        if filepath:
            console.print(f"[dim]New file: {filepath}[/dim]")
        else:
            filepath = "untitled.rln"
            console.print(f"[dim]New file: {filepath}[/dim]")

    console.print("[dim]Commands: :w = save, :wq = save & quit, :q = quit, :r = run[/dim]")
    console.print("[dim]         :list = show all lines, :del N = delete line N[/dim]")
    console.print("[dim]Type script file commands line by line. Empty line to finish editing.[/dim]")
    console.print()

    # Show existing content
    if lines:
        for i, line in enumerate(lines, 1):
            console.print(f"  [dim]{i:4d}[/dim]  {_highlight_line(line)}")
        console.print()

    while True:
        try:
            line_num = len(lines) + 1
            user_input = input(f"  {line_num:4d}  ").rstrip()

            # Editor commands
            if user_input == ":w":
                _save_dofile(filepath, lines, console)
                continue
            elif user_input == ":wq":
                _save_dofile(filepath, lines, console)
                break
            elif user_input == ":q":
                break
            elif user_input == ":q!":
                break
            elif user_input == ":r":
                _save_dofile(filepath, lines, console)
                # Run the script file
                from commands.parser import CommandParser
                parser = CommandParser(state, console)
                from main import run_do_file
                run_do_file(filepath, parser, console)
                continue
            elif user_input == ":list":
                for i, line in enumerate(lines, 1):
                    console.print(f"  [dim]{i:4d}[/dim]  {_highlight_line(line)}")
                continue
            elif user_input.startswith(":del "):
                try:
                    idx = int(user_input[5:]) - 1
                    if 0 <= idx < len(lines):
                        removed = lines.pop(idx)
                        console.print(f"  [dim]Deleted line {idx+1}: {removed}[/dim]")
                    else:
                        console.print(f"  [red]Line {idx+1} not found[/red]")
                except ValueError:
                    console.print("  [red]Syntax: :del N[/red]")
                continue
            elif user_input == "":
                # Double empty line to finish
                if lines and lines[-1] == "":
                    lines.pop()  # remove the empty line
                    break
                lines.append("")
                continue

            lines.append(user_input)

        except (EOFError, KeyboardInterrupt):
            console.print()
            break

    console.print(f"[dim]Editor closed. {len(lines)} lines.[/dim]")


def _save_dofile(filepath, lines, console):
    """Save lines to script file."""
    with open(filepath, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(line + "\n")
    console.print(f"[green]Saved: {filepath} ({len(lines)} lines)[/green]")


def _highlight_line(line: str) -> str:
    """Simple syntax highlighting for script file lines using rich markup."""
    stripped = line.lstrip()

    # Comments
    if stripped.startswith("*") or stripped.startswith("//"):
        return f"[green italic]{line}[/green italic]"

    # Find // comment at end of line
    comment_pos = -1
    in_quote = False
    for i in range(len(line) - 1):
        if line[i] == '"':
            in_quote = not in_quote
        elif not in_quote and line[i:i+2] == "//":
            comment_pos = i
            break

    if comment_pos >= 0:
        code_part = line[:comment_pos]
        comment_part = line[comment_pos:]
        return f"{_highlight_code(code_part)}[green italic]{comment_part}[/green italic]"

    return _highlight_code(line)


def _highlight_code(line: str) -> str:
    """Highlight other statistical tools keywords in a code line."""
    keywords = {
        "use", "import", "save", "export", "log", "describe", "desc",
        "codebook", "list", "tabulate", "tab", "summarize", "sum",
        "count", "generate", "gen", "replace", "rename", "drop", "keep",
        "order", "recode", "label", "destring", "tostring", "encode",
        "sort", "gsort", "duplicates", "append", "merge", "fuzzmerge",
        "reshape", "collapse", "fillin", "cross", "sample",
        "assert", "capture", "preserve", "restore", "egen", "notes",
        "display", "di", "isid", "levelsof", "distinct", "compress",
        "browse", "clear", "help", "do", "set", "clonevar", "split",
        "if", "in", "using", "by", "ssc",
    }

    # Tokenize and highlight
    result = []
    tokens = re.split(r'(\s+|"[^"]*")', line)
    for token in tokens:
        if token.lower() in keywords:
            result.append(f"[bold blue]{token}[/bold blue]")
        elif token.startswith('"') and token.endswith('"'):
            result.append(f"[magenta]{token}[/magenta]")
        elif re.match(r'^\d+\.?\d*$', token):
            result.append(f"[cyan]{token}[/cyan]")
        else:
            result.append(token)

    return "".join(result)


# ──────────────────────────────────────────────
#  python — Run inline Python code
# ──────────────────────────────────────────────

def cmd_python(rest: str, state: AppState, console: Console):
    """
    python: expr           — Evaluate a single Python expression
    python {               — Start a Python block (end with })

    The current DataFrame is available as 'df' and numpy as 'np'.
    """
    rest = rest.strip()

    if rest.startswith(":"):
        # Single expression
        expr = rest[1:].strip()
        _run_python_expr(expr, state, console)
        return

    if rest == "{" or rest == "":
        # When running inside a script file, the runner pre-collects the block
        # body and deposits it on state._pending_block. Use that instead of
        # reading from stdin (which may not be attached in batch mode).
        pending = getattr(state, "_pending_block", None)
        if pending:
            code = "\n".join(pending)
            state._pending_block = None
            _run_python_code(code, state, console)
            return

        # Multi-line block (interactive REPL only)
        console.print("[dim]Python mode. Type '}' on its own line to execute.[/dim]")
        lines = []
        while True:
            try:
                line = input(">>> ")
                if line.strip() == "}":
                    break
                lines.append(line)
            except (EOFError, KeyboardInterrupt):
                console.print()
                return

        code = "\n".join(lines)
        _run_python_code(code, state, console)
        return

    # Single line
    _run_python_expr(rest, state, console)


def _run_python_expr(expr, state, console):
    """Run a single Python expression with data context."""
    import numpy as np
    import pandas as pd

    ns = {
        "np": np,
        "pd": pd,
        "df": state.data if state.has_data() else pd.DataFrame(),
        "state": state,
        "print": lambda *a, **kw: console.print(*a, **kw),
    }

    try:
        result = eval(expr, ns)
        if result is not None:
            console.print(result)
    except SyntaxError:
        # Try exec for statements
        try:
            exec(expr, ns)
            if "df" in ns and ns["df"] is not state.data:
                state.data = ns["df"]
                state.mark_changed()
        except Exception as e:
            console.print(f"[red]Python error: {e}[/red]")
    except Exception as e:
        console.print(f"[red]Python error: {e}[/red]")


def _run_python_code(code, state, console):
    """Run a block of Python code with data context.

    The namespace provides:
        np, pd   — the numpy and pandas modules
        df       — a copy of the current dataset (empty DataFrame if none)
        state    — the full AppState
        print    — routed through console.print

    The block is passed through textwrap.dedent() so that code written
    with uniform leading whitespace (as when a `python { ... }` block
    inside a script file is indented for readability) still parses. This is
    what unblocks Gemini Bug 13 — preserving indentation is only half
    the fix; we also need to strip the common leading whitespace so the
    Python parser sees a well-formed module.

    After execution, if the user's code assigned to `df`, the new value is
    synced back into state.data. This works whether or not there was data
    loaded to begin with (Gemini Bug 9 fix), so a script file can start with:

        python {
            df = pd.DataFrame({'x': range(10)})
        }

    and have subsequent commands see the new dataset.
    """
    import numpy as np
    import pandas as pd
    import textwrap

    # Gemini Bug 13: strip uniform leading whitespace so `exec` doesn't reject
    # code that was indented for readability inside a script file.
    code = textwrap.dedent(code)

    had_data = state.has_data()
    starting_df = state.data.copy() if had_data else pd.DataFrame()

    ns = {
        "np": np,
        "pd": pd,
        "df": starting_df,
        "state": state,
        "print": lambda *a, **kw: console.print(*a, **kw),
    }

    try:
        exec(code, ns)
        # Bug 9 (Gemini): sync back whenever df looks like a real DataFrame,
        # even if we started with no data. Without this, a script file can't
        # bootstrap a dataset inside a python { ... } block.
        result_df = ns.get("df")
        if isinstance(result_df, pd.DataFrame):
            changed = (
                (not had_data and len(result_df.columns) + len(result_df) > 0)
                or (had_data and not result_df.equals(state.data))
            )
            if changed:
                state.data = result_df
                state.mark_changed()
                console.print("[green]DataFrame updated.[/green]")
    except Exception as e:
        console.print(f"[red]Python error: {e}[/red]")


# ──────────────────────────────────────────────
#  copy — download a URL or copy a local file
# ──────────────────────────────────────────────

def cmd_copy(rest: str, state: AppState, console: Console):
    """
    copy <from> <to> [, replace public text binary]

    Download a URL or copy a local file. Mirrors the `copy` command.

    Examples:
        copy "https://example.com/data.dta" "mydata.dta", replace
        copy "https://example.com/report.pdf" "report.pdf"
        copy "existing.csv" "backup.csv", replace
    """
    rest = rest.strip()
    if not rest:
        console.print("[red]Syntax: copy \"<from>\" \"<to>\" [, replace text binary][/red]")
        return

    # Split options off first — everything before the first ',' not in quotes
    body, _, opts_str = _split_on_comma_outside_quotes(rest)
    opts = {tok.strip().lower() for tok in opts_str.split() if tok.strip()} if opts_str else set()
    replace = "replace" in opts
    # text/binary currently only affects dest creation mode; harmless either way

    tokens = _split_quoted_tokens(body)
    if len(tokens) < 2:
        console.print("[red]copy: need both source and destination paths[/red]")
        return
    src, dst = tokens[0], tokens[1]

    dst = os.path.expanduser(dst)
    if os.path.exists(dst) and not replace:
        console.print(f"[red]copy: {dst} already exists (use 'replace' to overwrite)[/red]")
        return

    try:
        if src.startswith(("http://", "https://", "ftp://")):
            _download_url(src, dst, console)
        else:
            src = os.path.expanduser(src)
            if not os.path.exists(src):
                console.print(f"[red]copy: source not found: {src}[/red]")
                return
            import shutil
            shutil.copy2(src, dst)
            size = os.path.getsize(dst)
            console.print(f"[green]Copied {src} -> {dst} ({_fmt_bytes(size)})[/green]")
    except Exception as e:
        console.print(f"[red]copy failed: {e}[/red]")


def _download_url(url: str, dst: str, console: Console, chunk_size: int = 1 << 16):
    import urllib.request, urllib.error, time
    req = urllib.request.Request(
        url, headers={"User-Agent": "Rln/1.1 (+https://github.com/akirawisnu)"})
    console.print(f"[dim]Downloading {url}...[/dim]")
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            total = resp.getheader("Content-Length")
            total = int(total) if total and total.isdigit() else None
            written = 0
            with open(dst, "wb") as f:
                while True:
                    chunk = resp.read(chunk_size)
                    if not chunk:
                        break
                    f.write(chunk)
                    written += len(chunk)
                    if total:
                        pct = 100 * written / total
                        console.print(f"\r[dim]  {_fmt_bytes(written)}/{_fmt_bytes(total)} ({pct:.0f}%)[/dim]",
                                      end="")
        elapsed = time.time() - t0
        console.print(f"\n[green]Copied {url} -> {dst} "
                      f"({_fmt_bytes(written)} in {elapsed:.1f}s)[/green]")
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"HTTP {e.code}: {e.reason}")
    except urllib.error.URLError as e:
        raise RuntimeError(f"Network error: {e.reason}")


def _split_quoted_tokens(s: str):
    """Split a string on whitespace but respecting double/single quotes."""
    out, cur, in_q = [], [], None
    for ch in s:
        if in_q:
            if ch == in_q:
                in_q = None
            else:
                cur.append(ch)
        elif ch in ('"', "'"):
            in_q = ch
        elif ch.isspace():
            if cur:
                out.append("".join(cur)); cur = []
        else:
            cur.append(ch)
    if cur:
        out.append("".join(cur))
    return out


def _split_on_comma_outside_quotes(s: str):
    """Return (body, comma_or_empty, tail) where the comma is the first
    one that appears outside any string literal."""
    depth = 0
    in_q = None
    for i, ch in enumerate(s):
        if in_q:
            if ch == in_q:
                in_q = None
            continue
        if ch in ('"', "'"):
            in_q = ch
        elif ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        elif ch == "," and depth == 0:
            return s[:i].strip(), ",", s[i + 1:].strip()
    return s.strip(), "", ""


def _fmt_bytes(n: int) -> str:
    if n is None:
        return "?"
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"
