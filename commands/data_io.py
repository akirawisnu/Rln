"""
Data I/O commands: use, import, save, export, log
"""

import os
import re
from rich.console import Console
from commands.state import AppState
from commands.parse_helpers import parse_command_line
from rln_io.fileio import load_data, save_data


def cmd_use(rest: str, state: AppState, console: Console):
    """
    use "filepath" [, clear]
    Load a dataset. Auto-detects format from extension.
    """
    parsed = parse_command_line(rest)

    # Get filepath from varlist or raw
    filepath = None
    if parsed["varlist"]:
        filepath = parsed["varlist"][0]
    elif parsed["raw"]:
        # Handle quoted path
        m = re.match(r'"(.*?)"|\'(.*?)\'|(\S+)', parsed["raw"].strip())
        if m:
            filepath = m.group(1) or m.group(2) or m.group(3)

    if not filepath:
        console.print("[red]Syntax: use \"filename\"[/red]")
        return

    # Check for unsaved changes
    if state.has_data() and state.has_unsaved_changes:
        if "clear" not in parsed["options"]:
            console.print("[yellow]Warning: unsaved data in memory. Use 'use file, clear' to override.[/yellow]")
            return

    # Check for polars engine — redirect to LRTM
    engine = parsed["options"].get("engine", "").lower()
    if engine == "polars":
        from commands.lrtm import cmd_lrtm
        cmd_lrtm(f"use {filepath}", state, console)
        return

    # Resolve bare names against the workspace (latest project / examples), so
    # `use "demographics.csv"` finds the shipped example without a full path.
    from commands.workspace import resolve_path, remember_dir
    filepath = resolve_path(filepath)

    # Load
    console.print(f"[dim]Loading {filepath}...[/dim]")
    df, meta = load_data(filepath)

    name = os.path.splitext(os.path.basename(filepath))[0]
    state.set_data(df, name=name, source=filepath)
    remember_dir(filepath)  # this folder becomes the new default starting point

    # Apply metadata
    if "variable_labels" in meta:
        state.variable_labels = meta["variable_labels"]
    if "value_labels" in meta:
        state.value_labels = meta["value_labels"]

    nobs, nvars = df.shape
    console.print(f"[green]({nobs:,} observations, {nvars} variables)[/green]")
    state.write_log(f"Loaded: {filepath} ({nobs} obs, {nvars} vars)")


def cmd_import(rest: str, state: AppState, console: Console):
    """
    import delimited "file" [, options]
    import excel "file" [, sheet("name")]
    import html "url_or_file" [, table(N)]
    """
    parts = rest.strip().split(None, 1)
    if not parts:
        console.print("[red]Syntax: import delimited|excel|html \"filename\"[/red]")
        return

    subcommand = parts[0].lower()
    sub_rest = parts[1] if len(parts) > 1 else ""
    parsed = parse_command_line(sub_rest)

    filepath = None
    if parsed["varlist"]:
        filepath = parsed["varlist"][0]
    elif parsed["raw"]:
        m = re.match(r'"(.*?)"|\'(.*?)\'|(\S+)', parsed["raw"].strip().split(",")[0].strip())
        if m:
            filepath = m.group(1) or m.group(2) or m.group(3)

    if not filepath:
        console.print(f"[red]Syntax: import {subcommand} \"filename\"[/red]")
        return

    kwargs = {}

    if subcommand in ("delimited", "csv"):
        if "delimiter" in parsed["options"]:
            delim = parsed["options"]["delimiter"]
            delim_map = {"tab": "\t", "comma": ",", "semicolon": ";", "space": " "}
            kwargs["sep"] = delim_map.get(delim, delim)

    elif subcommand == "excel":
        if "sheet" in parsed["options"]:
            kwargs["sheet"] = parsed["options"]["sheet"]

    elif subcommand == "html":
        if "table" in parsed["options"]:
            kwargs["table_index"] = int(parsed["options"]["table"])

    else:
        console.print(f"[red]Unknown import type: {subcommand}[/red]")
        console.print("[dim]Available: delimited, excel, html[/dim]")
        return

    # Resolve bare names against the workspace (latest project / examples).
    from commands.workspace import resolve_path, remember_dir
    if subcommand != "html":  # html may be a URL — leave it untouched
        filepath = resolve_path(filepath)

    console.print(f"[dim]Importing {filepath}...[/dim]")
    df, meta = load_data(filepath, **kwargs)
    if subcommand != "html":
        remember_dir(filepath)

    name = os.path.splitext(os.path.basename(filepath))[0] if not filepath.startswith("http") else "html_data"
    state.set_data(df, name=name, source=filepath)

    nobs, nvars = df.shape
    console.print(f"[green]({nobs:,} observations, {nvars} variables)[/green]")


def cmd_save(rest: str, state: AppState, console: Console):
    """
    save "filepath" [, replace]
    Save dataset. Default format: .dta
    """
    state.require_data()
    parsed = parse_command_line(rest)

    filepath = None
    if parsed["varlist"]:
        filepath = parsed["varlist"][0]
    elif parsed["raw"]:
        m = re.match(r'"(.*?)"|\'(.*?)\'|(\S+)', parsed["raw"].strip().split(",")[0].strip())
        if m:
            filepath = m.group(1) or m.group(2) or m.group(3)

    if not filepath:
        # Default to source file
        if state.source_file:
            filepath = state.source_file
        else:
            console.print("[red]Syntax: save \"filename\"[/red]")
            return

    # Add .dta if no extension
    if not os.path.splitext(filepath)[1]:
        filepath += ".dta"

    # Check overwrite
    if os.path.exists(filepath) and "replace" not in parsed["options"]:
        console.print(f"[yellow]File exists. Use: save \"{filepath}\", replace[/yellow]")
        return

    metadata = {
        "variable_labels": state.variable_labels,
        "value_labels": state.value_labels,
        "value_label_assignments": state.value_label_assignments,
    }

    save_data(state.data, filepath, metadata)
    state.has_unsaved_changes = False


def cmd_export(rest: str, state: AppState, console: Console):
    """
    export delimited "file" [, replace]
    export excel "file" [, replace]
    """
    state.require_data()
    parts = rest.strip().split(None, 1)
    if not parts:
        console.print("[red]Syntax: export delimited|excel \"filename\"[/red]")
        return

    subcommand = parts[0].lower()
    sub_rest = parts[1] if len(parts) > 1 else ""
    parsed = parse_command_line(sub_rest)

    filepath = None
    if parsed["varlist"]:
        filepath = parsed["varlist"][0]
    elif parsed["raw"]:
        m = re.match(r'"(.*?)"|\'(.*?)\'|(\S+)', parsed["raw"].strip().split(",")[0].strip())
        if m:
            filepath = m.group(1) or m.group(2) or m.group(3)

    if not filepath:
        console.print(f"[red]Syntax: export {subcommand} \"filename\"[/red]")
        return

    # Add extension if missing
    if not os.path.splitext(filepath)[1]:
        ext_map = {"delimited": ".csv", "csv": ".csv", "excel": ".xlsx"}
        filepath += ext_map.get(subcommand, ".csv")

    if os.path.exists(filepath) and "replace" not in parsed["options"]:
        console.print(f"[yellow]File exists. Use: export {subcommand} \"{filepath}\", replace[/yellow]")
        return

    save_data(state.data, filepath)


def cmd_log(rest: str, state: AppState, console: Console):
    """
    log using "file" [, replace append text]
    log close
    """
    parts = rest.strip().split(None, 1)
    if not parts:
        if state.log_file:
            console.print(f"[dim]Log active: {state.log_file}[/dim]")
        else:
            console.print("[dim]No log active.[/dim]")
        return

    subcmd = parts[0].lower()

    if subcmd == "close":
        if state.log_file:
            lf = state.log_file
            state.stop_log()
            console.print(f"[green]Log closed: {lf}[/green]")
        else:
            console.print("[dim]No log to close.[/dim]")
        return

    if subcmd == "using":
        sub_rest = parts[1] if len(parts) > 1 else ""
        parsed = parse_command_line(sub_rest)

        filepath = None
        if parsed["varlist"]:
            filepath = parsed["varlist"][0]
        if not filepath:
            console.print("[red]Syntax: log using \"filename\"[/red]")
            return

        if not os.path.splitext(filepath)[1]:
            filepath += ".log"

        append = "append" in parsed["options"]
        state.start_log(filepath, append=append)
        console.print(f"[green]Log started: {filepath}[/green]")
    else:
        console.print("[red]Syntax: log using \"file\" | log close[/red]")
