#!/usr/bin/env python3
"""
Rln - Free Data Exploration & Management CLI
Statistics and Data Exploratory Tools

Usage:
  python main.py                       # Interactive REPL
  python main.py do script.rln          # Run a script file (batch mode)
  python main.py -e "command"          # Execute a single command
  python main.py gui                   # Launch the GUI
  python main.py --gui                 # Launch the GUI
  python main.py --version             # Show version
"""

import sys
import os
import signal
import argparse

# Reconfigure stdout/stderr to UTF-8 so rich's box-drawing characters don't
# crash on Windows cp1252 consoles (Minimax Bug 6, 7). On Unix this is a
# no-op because stdout is already UTF-8.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, Exception):
    pass  # Python < 3.7 or non-reconfigurable stream

from rich.console import Console

from commands.parser import CommandParser
from commands.state import AppState

__version__ = "1.2.8"

LOGO = """\
[bold cyan] ____  _       [/bold cyan]
[bold cyan]|  _ \\| |_ __  [/bold cyan]
[bold cyan]| |_) | | '_ \\ [/bold cyan]   [bold white]Rln (ARLEN) v{ver}[/bold white]
[bold cyan]|  _ <| | | | |[/bold cyan]   [dim]Statistics and Data Exploratory Tools[/dim]
[bold cyan]|_| \\_\\_|_| |_|[/bold cyan]
[dim]Open Source for Researchers[/dim]         [dim]@2026 by Akirawisnu[/dim]
[dim]https://akirawisnu.github.io/[/dim]
[dim]MIT License - Type [bold]help[/bold] for commands, [bold]exit[/bold] to quit[/dim]
""".format(ver=__version__)


def get_prompt_text(state: AppState) -> str:
    """Build prompt showing loaded dataset info."""
    if state.has_data():
        nobs, nvars = state.data.shape
        name = state.dataset_name or "data"
        return f"({name}: {nobs:,} obs x {nvars} vars) . "
    return ". "


def run_do_file(filepath: str, parser: CommandParser, console: Console):
    """Execute a script file."""
    # Resolve bare names against the workspace so `do "sample.do"` finds the
    # shipped example without a full path (and tracks the latest project).
    from commands.workspace import resolve_path, remember_dir
    filepath = resolve_path(filepath)
    if not os.path.exists(filepath):
        console.print(f"[red]Script not found: {filepath}[/red]")
        return False
    remember_dir(filepath)

    console.print(f"[dim]Running: {filepath}[/dim]")

    with open(filepath, "r", encoding="utf-8") as f:
        raw_lines = f.readlines()

    from commands.dofile import preprocess_dofile
    commands = preprocess_dofile(raw_lines)

    i = 0
    while i < len(commands):
        cmd = commands[i].strip()
        i += 1

        if not cmd:
            continue

        if cmd.lower().startswith("do "):
            # Expand macros in nested paths so `do "$path/child.rln"` and
            # `do "`local'"` both work. Critical: expand macros BEFORE
            # stripping outer quotes — otherwise the strip("'") removes
            # the closing apostrophe of the backtick-apostrophe macro
            # form `name', breaking expansion. (Minimax v126 B2.)
            from commands.scripting import expand_macros
            raw_path = cmd[3:].strip()
            expanded = expand_macros(raw_path, parser.state)
            nested_path = expanded.strip("\"'")
            ok = run_do_file(nested_path, parser, console)
            if not ok:
                on_err = getattr(parser.state, "on_error", "stop")
                if on_err == "stop":
                    console.print(f"[red]Parent script file halted after nested "
                                  f"failure in {nested_path}[/red]")
                    return False
            continue

        console.print(f"[dim]. {cmd}[/dim]")

        if cmd.lower() in ("exit", "quit", "q"):
            break

        # Check if this command opens a { } block (foreach, forvalues, python, etc.)
        # Bug 13 fix (Gemini v1.2.3): for python blocks we must preserve
        # indentation, otherwise multi-line `if:` / `try:` / `for:` bodies
        # raise IndentationError at exec time.
        #
        # Bug v1.2.7 fix (Minimax B1): when the OUTER block is foreach /
        # forvalues / quietly and contains a NESTED python block, we must
        # preserve the python block's lines verbatim (no .strip()). Without
        # this, `foreach v in a b c { python { ... } }` would unindent the
        # nested Python and break it.
        if cmd.rstrip().endswith("{"):
            is_python_outer = cmd.split(None, 1)[0].lower() in ("python", "py")
            block_lines = []
            depth = 1
            inside_nested_python = False
            nested_depth = 0
            while i < len(commands) and depth > 0:
                raw_bline = commands[i]
                stripped = raw_bline.strip()
                i += 1

                if inside_nested_python:
                    # Inside a nested python { ... } — preserve raw and
                    # only check for the matching close.
                    if stripped.endswith("{"):
                        nested_depth += 1
                    if stripped == "}":
                        nested_depth -= 1
                        if nested_depth == 0:
                            inside_nested_python = False
                            block_lines.append(raw_bline)  # keep the closing }
                            depth -= 1
                            continue
                    block_lines.append(raw_bline)
                    continue

                # Not inside a nested python block.
                # Detect entry into a nested python block.
                first_word = stripped.split(None, 1)[0].lower() if stripped else ""
                opens_python = (first_word in ("python", "py")
                                and stripped.endswith("{"))

                if stripped == "}":
                    depth -= 1
                    if depth == 0:
                        break
                    block_lines.append(stripped if not is_python_outer else raw_bline)
                elif opens_python and not is_python_outer:
                    # Enter raw-preservation mode for the nested block
                    inside_nested_python = True
                    nested_depth = 1
                    depth += 1
                    block_lines.append(raw_bline)  # keep `python {` as-is
                else:
                    if stripped.endswith("{"):
                        depth += 1
                    block_lines.append(raw_bline if is_python_outer else stripped)

            parser.state._pending_block = block_lines

        try:
            parser.execute(cmd, reraise=True)
        except Exception as e:
            rc = getattr(parser.state, "_rc", 1) or 1
            console.print(f"[red]Error on line {i} (rc={rc}): {e}[/red]")
            # Honor compact `set on_error stop/continue`. Default is STOP,
            # so scripts fail loudly instead of silently running to the end.
            on_err = getattr(parser.state, "on_error", "stop")
            if on_err == "stop":
                console.print(f"[red]Script halted at line {i}. "
                              f"Set 'on_error continue' to ignore errors.[/red]")
                return False
            # `continue` — print the error and keep going

    console.print(f"[dim]End of script file: {filepath}[/dim]")
    return True


def run_interactive(console: Console):
    """Run interactive REPL."""
    try:
        from prompt_toolkit import PromptSession
        from prompt_toolkit.history import FileHistory
        from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
        has_prompt_toolkit = True
    except ImportError:
        has_prompt_toolkit = False

    console.print(LOGO)

    state = AppState()
    parser = CommandParser(state, console)

    if has_prompt_toolkit:
        history_dir = os.path.expanduser("~")
        history_file = os.path.join(history_dir, ".rln_history")
        session = PromptSession(
            history=FileHistory(history_file),
            auto_suggest=AutoSuggestFromHistory(),
        )
        get_input = lambda prompt: session.prompt(prompt)
    else:
        console.print("[dim](Install prompt_toolkit for command history & auto-suggest)[/dim]")
        get_input = lambda prompt: input(prompt)

    def handle_sigint(sig, frame):
        console.print("\n[dim]Type [bold]exit[/bold] to quit.[/dim]")

    signal.signal(signal.SIGINT, handle_sigint)

    while True:
        try:
            prompt_text = get_prompt_text(state)
            user_input = get_input(prompt_text).strip()

            if not user_input:
                continue

            if user_input.lower() in ("exit", "quit", "q"):
                if state.has_unsaved_changes:
                    console.print("[yellow]Warning: unsaved changes.[/yellow]")
                    confirm = get_input("Really exit? (y/n) ").strip().lower()
                    if confirm not in ("y", "yes", "exit"):
                        continue
                console.print("[dim]Goodbye.[/dim]")
                break

            # Handle script file command from REPL
            if user_input.lower().startswith("do "):
                do_path = user_input[3:].strip().strip("\"'")
                run_do_file(do_path, parser, console)
                continue

            parser.execute(user_input)

        except KeyboardInterrupt:
            console.print()
            continue
        except EOFError:
            console.print("\n[dim]Goodbye.[/dim]")
            break
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")


def main():
    ap = argparse.ArgumentParser(
        prog="rln",
        description="Rln - Statistics and Data Exploratory Tools",
    )
    ap.add_argument("--version", action="version", version=f"Rln v{__version__}")
    ap.add_argument("-e", "--execute", metavar="CMD",
                    help="Execute a single command and exit")
    ap.add_argument("--gui", action="store_true",
                    help="Launch the Rln graphical interface")
    ap.add_argument("command", nargs="?", help="'do' to run a script file, or 'gui' to launch GUI")
    ap.add_argument("dofile", nargs="?", help="Path to script file")

    args = ap.parse_args()
    con = Console()

    if args.gui or (args.command and args.command.lower() == "gui"):
        try:
            from gui import launch_gui
        except Exception as e:
            con.print(f"[red]Could not launch GUI: {e}[/red]")
            sys.exit(1)
        launch_gui(version=__version__)

    elif args.execute:
        state = AppState()
        parser = CommandParser(state, con)
        parser.execute(args.execute)
        # B-002: propagate the inner return code to the OS so CI and shell
        # scripts know whether the command actually succeeded.
        sys.exit(int(getattr(state, "_rc", 0) or 0))

    elif args.command and args.command.lower() == "do" and args.dofile:
        state = AppState()
        parser = CommandParser(state, con)
        success = run_do_file(args.dofile, parser, con)
        sys.exit(0 if success else 1)

    else:
        run_interactive(con)


if __name__ == "__main__":
    main()
