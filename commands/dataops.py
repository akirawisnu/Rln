"""
Data operations: sort, gsort, duplicates, append, merge, fuzzmerge
"""

import os
import re
import pandas as pd
import numpy as np
from rich.console import Console
from rich.table import Table

from commands.state import AppState
from commands.parse_helpers import parse_command_line
from rln_io.fileio import load_data


def cmd_sort(rest: str, state: AppState, console: Console):
    """
    sort varlist
    Sort data in ascending order.
    """
    state.require_data()
    parsed = parse_command_line(rest)

    if not parsed["varlist"]:
        console.print("[red]Syntax: sort varlist[/red]")
        return

    for v in parsed["varlist"]:
        if v not in state.data.columns:
            console.print(f"[red]Variable '{v}' not found[/red]")
            return

    state.data = state.data.sort_values(parsed["varlist"]).reset_index(drop=True)
    console.print(f"[green]Sorted by: {', '.join(parsed['varlist'])}[/green]")
    state.mark_changed()


def cmd_gsort(rest: str, state: AppState, console: Console):
    """
    gsort [+-]var1 [+-]var2 ...
    Sort with direction control. Prefix - for descending.
    """
    state.require_data()
    tokens = rest.strip().split()

    if not tokens:
        console.print("[red]Syntax: gsort [+-]varlist[/red]")
        return

    sort_cols = []
    sort_asc = []

    for token in tokens:
        if token.startswith("-"):
            varname = token[1:]
            ascending = False
        elif token.startswith("+"):
            varname = token[1:]
            ascending = True
        else:
            varname = token
            ascending = True

        if varname not in state.data.columns:
            console.print(f"[red]Variable '{varname}' not found[/red]")
            return

        sort_cols.append(varname)
        sort_asc.append(ascending)

    state.data = state.data.sort_values(sort_cols, ascending=sort_asc).reset_index(drop=True)
    desc = ", ".join(f"{'+' if a else '-'}{c}" for c, a in zip(sort_cols, sort_asc))
    console.print(f"[green]Sorted by: {desc}[/green]")
    state.mark_changed()


def cmd_duplicates(rest: str, state: AppState, console: Console):
    """
    duplicates report [varlist]
    duplicates drop [varlist] [, force]
    duplicates tag [varlist], generate(newvar)
    duplicates list [varlist]
    """
    parts = rest.strip().split(None, 1)
    if not parts:
        console.print("[red]Syntax: duplicates report|drop|tag|list [varlist][/red]")
        return

    state.require_data()
    subcmd = parts[0].lower()
    sub_rest = parts[1] if len(parts) > 1 else ""
    parsed = parse_command_line(sub_rest)

    varlist = parsed["varlist"] if parsed["varlist"] else list(state.data.columns)

    # Validate varlist
    for v in varlist:
        if v not in state.data.columns:
            console.print(f"[red]Variable '{v}' not found[/red]")
            return

    if subcmd == "report":
        dupes = state.data.duplicated(subset=varlist, keep=False)
        n_dupes = dupes.sum()
        n_unique = len(state.data) - state.data.duplicated(subset=varlist, keep="first").sum()

        console.print(f"\n  Duplicates in terms of: {', '.join(varlist)}")
        console.print(f"  Observations:          {len(state.data):,}")
        console.print(f"  Unique values:         {n_unique:,}")
        console.print(f"  Duplicate observations:{n_dupes:,}")

        if n_dupes > 0:
            # Group counts
            group_sizes = state.data.groupby(varlist).size()
            surplus = group_sizes[group_sizes > 1]
            console.print(f"  Surplus observations:  {surplus.sum() - len(surplus):,}")

            # Distribution of copies
            copies_dist = group_sizes.value_counts().sort_index()
            console.print(f"\n  [dim]Copies | Observations | Groups[/dim]")
            for copies, n_groups in copies_dist.items():
                console.print(f"  {copies:>6} | {copies * n_groups:>12,} | {n_groups:>6,}")

    elif subcmd == "drop":
        n_before = len(state.data)
        force = "force" in parsed["options"]

        state.data = state.data.drop_duplicates(subset=varlist, keep="first").reset_index(drop=True)
        n_dropped = n_before - len(state.data)
        console.print(f"[green]({n_dropped:,} duplicate observations deleted)[/green]")
        state.mark_changed()

    elif subcmd == "tag":
        newvar = parsed["options"].get("generate") or parsed["options"].get("gen")
        if not newvar:
            console.print("[red]Syntax: duplicates tag varlist, generate(newvar)[/red]")
            return

        state.data[newvar] = state.data.groupby(varlist)[varlist[0]].transform("size") - 1
        n_tagged = (state.data[newvar] > 0).sum()
        console.print(f"[green]Tagged {n_tagged:,} duplicate observations in '{newvar}'[/green]")
        state.mark_changed()

    elif subcmd == "list":
        dupes = state.data.duplicated(subset=varlist, keep=False)
        subset = state.data.loc[dupes]
        if len(subset) == 0:
            console.print("[dim]No duplicates found.[/dim]")
        else:
            console.print(f"[yellow]{len(subset):,} duplicate observations:[/yellow]")
            from commands.explore import _display_rich_table
            _display_rich_table(subset.head(100), state, console)

    else:
        console.print(f"[red]Unknown subcommand: duplicates {subcmd}[/red]")


def cmd_append(rest: str, state: AppState, console: Console):
    """
    append using "filename" [, force generate(newvar)]
    Append another dataset to the current one.
    """
    state.require_data()
    parsed = parse_command_line(rest)

    filepath = parsed["using"]
    if not filepath:
        console.print('[red]Syntax: append using "filename"[/red]')
        return

    console.print(f"[dim]Loading {filepath}...[/dim]")
    df_append, _ = load_data(filepath)

    n_before = len(state.data)

    # Track source if requested
    gen_var = parsed["options"].get("generate") or parsed["options"].get("gen")
    if gen_var:
        state.data[gen_var] = 0
        df_append[gen_var] = 1

    force = "force" in parsed["options"]

    if force:
        # Force: align columns, fill missing with NaN
        state.data = pd.concat([state.data, df_append], ignore_index=True, sort=False)
    else:
        # Check column compatibility
        my_cols = set(state.data.columns)
        their_cols = set(df_append.columns)

        only_mine = my_cols - their_cols
        only_theirs = their_cols - my_cols

        if only_mine or only_theirs:
            if only_mine:
                console.print(f"[yellow]Variables only in master: {', '.join(sorted(only_mine))}[/yellow]")
            if only_theirs:
                console.print(f"[yellow]Variables only in using: {', '.join(sorted(only_theirs))}[/yellow]")
            console.print("[dim]These will be filled with missing values. Use 'append using file, force' to suppress.[/dim]")

        state.data = pd.concat([state.data, df_append], ignore_index=True, sort=False)

    n_added = len(state.data) - n_before
    console.print(f"[green]({n_added:,} observations appended, total now {len(state.data):,})[/green]")
    state.mark_changed()


def cmd_merge(rest: str, state: AppState, console: Console):
    """
    merge 1:1 varlist using "filename" [, keep(match) generate(_merge) nogenerate]
    merge m:1 varlist using "filename"
    merge 1:m varlist using "filename"
    """
    state.require_data()

    # Parse merge type
    m = re.match(r'(1:1|m:1|1:m|m:m)\s+(.*)', rest.strip())
    if not m:
        console.print('[red]Syntax: merge 1:1|m:1|1:m varlist using "filename"[/red]')
        return

    merge_type = m.group(1)
    sub_rest = m.group(2)
    parsed = parse_command_line(sub_rest)

    if not parsed["varlist"] or not parsed["using"]:
        console.print('[red]Syntax: merge 1:1 varlist using "filename"[/red]')
        return

    key_vars = parsed["varlist"]
    filepath = parsed["using"]

    # Validate key vars
    for v in key_vars:
        if v not in state.data.columns:
            console.print(f"[red]Variable '{v}' not found in master data[/red]")
            return

    console.print(f"[dim]Loading {filepath}...[/dim]")
    df_using, _ = load_data(filepath)

    # Validate key vars in using data
    for v in key_vars:
        if v not in df_using.columns:
            console.print(f"[red]Variable '{v}' not found in using data[/red]")
            return

    # Map merge types to pandas
    how_map = {"1:1": "outer", "m:1": "left", "1:m": "left", "m:m": "outer"}

    # Handle keep option
    keep_opt = parsed["options"].get("keep", None)
    if keep_opt:
        keep_map = {"master": "left", "using": "right", "match": "inner",
                     "1": "left", "2": "right", "3": "inner",
                     "match master": "left", "match using": "right"}
        how = keep_map.get(keep_opt.lower(), "outer")
    else:
        how = how_map[merge_type]

    # Handle overlapping non-key columns
    master_only = [c for c in state.data.columns if c not in key_vars and c in df_using.columns]
    if master_only:
        console.print(f"[yellow]Overlapping variables: {', '.join(master_only)}[/yellow]")
        console.print("[dim]Master values will be kept where they exist.[/dim]")

    # Perform merge
    nogen = "nogenerate" in parsed["options"] or "nogen" in parsed["options"]
    merge_indicator = not nogen
    gen_var = parsed["options"].get("generate", "_merge")

    result = pd.merge(
        state.data, df_using,
        on=key_vars,
        how=how,
        indicator=merge_indicator,
        suffixes=("", "_using")
    )

    # Report merge results
    if merge_indicator:
        merge_col = "_merge"
        merge_counts = result[merge_col].value_counts()

        table = Table(title="Merge Results")
        table.add_column("Result", min_width=25)
        table.add_column("Freq.", justify="right")
        table.add_column("Pct.", justify="right")

        total = len(result)
        labels = {
            "left_only": "Master only (_merge==1)",
            "right_only": "Using only (_merge==2)",
            "both": "Matched (_merge==3)",
        }
        for key, label in labels.items():
            cnt = merge_counts.get(key, 0)
            pct = cnt / total * 100 if total > 0 else 0
            table.add_row(label, f"{cnt:,}", f"{pct:.1f}%")
        table.add_row("─" * 20, "─" * 6, "─" * 6, style="dim")
        table.add_row("Total", f"{total:,}", "100.0%", style="bold")
        console.print(table)

        # Recode _merge to numeric as documented here
        merge_recode = {"left_only": 1, "right_only": 2, "both": 3}
        result[gen_var] = result[merge_col].map(merge_recode)
        if merge_col != gen_var:
            result = result.drop(columns=[merge_col])
        else:
            result[gen_var] = result[gen_var]  # already in place

    state.data = result
    state.mark_changed()


def cmd_fuzzmerge(rest: str, state: AppState, console: Console):
    """
    fuzzmerge varname using "filename" [, threshold(0.8) method(tfidf) generate(newvar)]
    Fuzzy merge using PolyFuzz for approximate string matching.
    """
    state.require_data()
    parsed = parse_command_line(rest)

    if not parsed["varlist"] or not parsed["using"]:
        console.print('[red]Syntax: fuzzmerge varname using "filename" [, threshold(0.8)][/red]')
        return

    if len(parsed["varlist"]) != 1:
        console.print("[red]fuzzmerge requires exactly one matching variable[/red]")
        return

    match_var = parsed["varlist"][0]
    filepath = parsed["using"]

    if match_var not in state.data.columns:
        console.print(f"[red]Variable '{match_var}' not found in master data[/red]")
        return

    # Options
    threshold = float(parsed["options"].get("threshold", 0.8))
    method = parsed["options"].get("method", "tfidf").lower()

    console.print(f"[dim]Loading {filepath}...[/dim]")
    df_using, _ = load_data(filepath)

    if match_var not in df_using.columns:
        console.print(f"[red]Variable '{match_var}' not found in using data[/red]")
        return

    # Get unique string values
    master_strings = state.data[match_var].dropna().astype(str).unique().tolist()
    using_strings = df_using[match_var].dropna().astype(str).unique().tolist()

    if not master_strings or not using_strings:
        console.print("[red]No valid strings to match[/red]")
        return

    console.print(f"[dim]Fuzzy matching {len(master_strings)} × {len(using_strings)} values "
                  f"(method={method}, threshold={threshold})...[/dim]")

    try:
        # Backend-agnostic matcher: PolyFuzz → rapidfuzz → difflib, so fuzzmerge
        # works on every build (lite has no polyfuzz; Android has neither
        # polyfuzz nor rapidfuzz — difflib is the pure-Python floor).
        from commands.fuzzy import fuzzy_match
        matches, backend = fuzzy_match(master_strings, using_strings,
                                       threshold=threshold, method=method)
        if backend != "polyfuzz":
            console.print(f"[dim]Using {backend} backend "
                          f"(polyfuzz unavailable on this build).[/dim]")

        console.print(f"[dim]{len(matches)} matches found above threshold {threshold}[/dim]")

        if len(matches) == 0:
            console.print("[yellow]No matches above threshold. Try lowering the threshold.[/yellow]")
            return

        # Create match lookup
        match_lookup = {frm: to for (frm, to, _sim) in matches}
        similarity_lookup = {frm: sim for (frm, _to, sim) in matches}

        # Add match columns to master
        match_col = f"{match_var}_matched"
        sim_col = f"{match_var}_similarity"

        state.data[match_col] = state.data[match_var].astype(str).map(match_lookup)
        state.data[sim_col] = state.data[match_var].astype(str).map(similarity_lookup)

        # Merge using matched values
        result = pd.merge(
            state.data,
            df_using,
            left_on=match_col,
            right_on=match_var,
            how="left",
            suffixes=("", "_using")
        )

        # Report
        n_matched = result[match_col].notna().sum()
        n_unmatched = result[match_col].isna().sum()

        console.print(f"\n[green]Fuzzy merge complete:[/green]")
        console.print(f"  Matched:   {n_matched:,}")
        console.print(f"  Unmatched: {n_unmatched:,}")
        console.print(f"  Mean similarity: {result[sim_col].mean():.3f}")

        state.data = result
        state.mark_changed()

    except ImportError:
        console.print("[red]PolyFuzz not installed. Run: pip install polyfuzz[/red]")
    except Exception as e:
        console.print(f"[red]Fuzzy merge failed: {e}[/red]")
