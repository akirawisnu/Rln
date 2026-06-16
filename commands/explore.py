"""
Exploration commands: browse, describe, codebook, list, tabulate, summarize, count
"""

import numpy as np
import pandas as pd
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from commands.state import AppState
from commands.parse_helpers import parse_command_line
from commands.expression import eval_condition, parse_in_range


def _resolve_varlist(varlist: list, df: pd.DataFrame) -> list:
    """Resolve variable names, supporting wildcards like var*."""
    if not varlist:
        return list(df.columns)

    resolved = []
    for v in varlist:
        if "*" in v:
            import fnmatch
            matches = fnmatch.filter(df.columns, v)
            resolved.extend(matches)
        elif v in df.columns:
            resolved.append(v)
        else:
            raise ValueError(f"Variable '{v}' not found")
    return resolved


def _apply_if_in(df: pd.DataFrame, if_cond: str = None, in_range: str = None) -> pd.DataFrame:
    """Apply if-condition and in-range to get a subset."""
    subset = df
    if if_cond:
        mask = eval_condition(if_cond, df)
        subset = subset.loc[mask]
    if in_range:
        sl = parse_in_range(in_range)
        if sl:
            subset = subset.iloc[sl]
    return subset


_BROWSE_FILE_EXTS = (".parquet", ".pq", ".csv", ".tsv", ".dta", ".xlsx",
                     ".xls", ".json", ".feather", ".dbf")


def cmd_browse(rest: str, state: AppState, console: Console):
    """
    browse [varlist] [if condition] [in range]
    browse "file.parquet"                       — explore a file directly

    Launch the interactive data explorer. Given a data file (parquet, csv,
    dta, …) it loads a preview and explores it WITHOUT touching the dataset in
    memory — handy for peeking at a parquet before `use`-ing it.
    """
    import os
    import re

    parsed = parse_command_line(rest)

    # Detect a file argument: `browse "file.ext"` (quoted) or a bare token that
    # has a recognised data extension. Column names never have these extensions,
    # so this won't collide with `browse var1 var2`.
    file_arg = parsed.get("using")
    if not file_arg:
        for cand in list(parsed["varlist"]):
            if os.path.splitext(cand)[1].lower() in _BROWSE_FILE_EXTS:
                file_arg = cand
                break
    if not file_arg and parsed.get("raw"):
        m = re.match(r'"(.*?)"|\'(.*?)\'', parsed["raw"].strip())
        if m:
            cand = m.group(1) or m.group(2)
            if os.path.splitext(cand)[1].lower() in _BROWSE_FILE_EXTS:
                file_arg = cand

    if file_arg:
        _browse_file(file_arg, state, console)
        return

    # If nothing is materialized but a parquet/CSV is lazy-loaded via `lrtm use`,
    # explore a streamed preview — the full dataset is available after
    # `lrtm collect`. Mirrors the GUI/Android data browser behaviour.
    if not state.has_data():
        try:
            from commands.lrtm import lrtm_preview
            prev = lrtm_preview(state, 5000)
        except Exception:
            prev = None
        if prev:
            view, total = prev
            note = (f"first {len(view):,} of {total:,} rows"
                    if isinstance(total, int) else f"first {len(view):,} rows")
            console.print(f"[dim]LRTM preview ({note}); run 'lrtm collect' to "
                          f"explore the full dataset.[/dim]")
            try:
                from tui.browser import launch_browser
                launch_browser(view, state)
            except ImportError:
                _display_rich_table(view, state, console, max_rows=50)
            return

    state.require_data()
    cols = _resolve_varlist(parsed["varlist"], state.data)
    subset = _apply_if_in(state.data[cols], parsed["if_cond"], parsed["in_range"])

    # Launch TUI browser
    try:
        from tui.browser import launch_browser
        launch_browser(subset, state)
    except ImportError:
        # Fallback: rich table display
        console.print("[yellow]TUI not available. Install textual: ssc install textual[/yellow]")
        _display_rich_table(subset, state, console, max_rows=50)


def _browse_file(path: str, state: AppState, console: Console):
    """Load a preview of a data file and open it in the explorer (read-only).

    Does not modify the in-memory dataset. Parquet is read with polars' native
    reader (no pyarrow needed), capped to a preview so large files stay snappy.
    """
    import os
    from commands.workspace import resolve_path

    rp = resolve_path(path)
    if not os.path.exists(rp):
        console.print(f"[red]File not found: {path}[/red]")
        return

    ext = os.path.splitext(rp)[1].lower()
    cap = 50000  # preview row cap for responsiveness
    try:
        if ext in (".parquet", ".pq"):
            import polars as pl
            from commands.lrtm import _pl_to_pandas
            lf = pl.scan_parquet(rp)
            total = lf.select(pl.len()).collect().item()
            df = _pl_to_pandas(lf.head(cap).collect())
        else:
            from rln_io.fileio import load_data
            df, _meta = load_data(rp)
            total = len(df)
            if len(df) > cap:
                df = df.head(cap)
    except Exception as e:
        console.print(f"[red]Could not open {os.path.basename(rp)}: {e}[/red]")
        return

    shown = len(df)
    note = f"{shown:,} rows" if shown >= total else f"first {shown:,} of {total:,} rows"
    console.print(f"[dim]Exploring {os.path.basename(rp)} ({note}). "
                  f"In-memory dataset is unchanged.[/dim]")
    try:
        from tui.browser import launch_browser
        launch_browser(df, state)
    except ImportError:
        console.print("[yellow]Interactive browser unavailable; showing a preview.[/yellow]")
        _display_rich_table(df, state, console, max_rows=50)


def cmd_describe(rest: str, state: AppState, console: Console):
    """
    describe [varlist]
    Show variable names, types, labels, and dataset summary.
    """
    state.require_data()
    parsed = parse_command_line(rest)
    df = state.data

    cols = _resolve_varlist(parsed["varlist"], df)

    # Dataset header
    nobs, nvars = df.shape
    console.print(f"\n[bold]Contains data[/bold]")
    if state.source_file:
        console.print(f"  [dim]Source: {state.source_file}[/dim]")
    console.print(f"  Observations: [cyan]{nobs:,}[/cyan]")
    console.print(f"  Variables:    [cyan]{len(cols)}[/cyan]")

    # Memory usage
    mem = df[cols].memory_usage(deep=True).sum()
    if mem > 1e9:
        mem_str = f"{mem/1e9:.1f} GB"
    elif mem > 1e6:
        mem_str = f"{mem/1e6:.1f} MB"
    else:
        mem_str = f"{mem/1e3:.1f} KB"
    console.print(f"  Size:         [dim]{mem_str}[/dim]\n")

    # Variable table
    table = Table(title="Variable Description", show_lines=False)
    table.add_column("#", style="dim", width=4, justify="right")
    table.add_column("Variable", style="bold", min_width=15)
    table.add_column("Type", min_width=12)
    table.add_column("Format", min_width=10)
    table.add_column("Value label", style="cyan", min_width=10)
    table.add_column("Variable label", style="dim")

    for i, col in enumerate(cols, 1):
        dtype = df[col].dtype
        type_str = _sb_type_str(dtype)
        fmt_str = _sb_format_str(dtype)
        var_label = state.get_variable_label(col)
        val_label = ""
        if hasattr(state, "value_label_assignments"):
            val_label = state.value_label_assignments.get(col, "") or ""
        table.add_row(str(i), col, type_str, fmt_str, val_label, var_label)

    console.print(table)
    state.write_log(f"describe: {len(cols)} variables shown")


def cmd_codebook(rest: str, state: AppState, console: Console):
    """
    codebook [varlist]
    Detailed variable summary with unique values, missing, etc.
    """
    state.require_data()
    parsed = parse_command_line(rest)
    df = state.data

    cols = _resolve_varlist(parsed["varlist"], df)

    for col in cols:
        series = df[col]
        n_total = len(series)
        n_missing = series.isna().sum()
        n_valid = n_total - n_missing
        n_unique = series.nunique()

        console.print(f"\n{'─' * 60}")
        label = state.get_variable_label(col)
        header = f"[bold]{col}[/bold]"
        if label:
            header += f"  [dim]({label})[/dim]"
        console.print(header)
        console.print(f"{'─' * 60}")

        console.print(f"  Type:      {_sb_type_str(series.dtype)}")
        console.print(f"  Obs:       {n_valid:,} valid, {n_missing:,} missing ({n_total:,} total)")
        console.print(f"  Unique:    {n_unique:,}")

        if pd.api.types.is_numeric_dtype(series):
            desc = series.describe()
            console.print(f"  Range:     [{desc.get('min', 'N/A')}, {desc.get('max', 'N/A')}]")
            console.print(f"  Mean:      {desc.get('mean', 'N/A'):.4f}")
            console.print(f"  Std. Dev:  {desc.get('std', 'N/A'):.4f}")
            console.print(f"  Pctiles:   25th={desc.get('25%', 'N/A')}, "
                          f"50th={desc.get('50%', 'N/A')}, "
                          f"75th={desc.get('75%', 'N/A')}")
        else:
            # String variable: show top values
            if n_unique <= 20 and n_unique > 0:
                console.print(f"\n  [dim]Tabulation:[/dim]")
                val_counts = series.value_counts().head(20)
                for val, cnt in val_counts.items():
                    pct = cnt / n_valid * 100
                    console.print(f"    {str(val):30s}  {cnt:>6,}  ({pct:.1f}%)")
            elif n_unique > 0:
                console.print(f"\n  [dim]Examples (first 5):[/dim]")
                for val in series.dropna().unique()[:5]:
                    console.print(f"    {val}")

    console.print(f"\n{'─' * 60}")


def cmd_list(rest: str, state: AppState, console: Console):
    """
    list [varlist] [if condition] [in range] [, noobs separator]
    Print observations in a formatted table.
    """
    state.require_data()
    parsed = parse_command_line(rest)
    df = state.data

    cols = _resolve_varlist(parsed["varlist"], df)
    subset = _apply_if_in(df[cols], parsed["if_cond"], parsed["in_range"])

    max_rows = state.settings.get("max_display_rows", 200)
    if len(subset) > max_rows:
        console.print(f"[yellow]Showing first {max_rows} of {len(subset):,} observations[/yellow]")
        subset = subset.head(max_rows)

    show_obs = "noobs" not in parsed["options"]
    _display_rich_table(subset, state, console, show_obs=show_obs)
    state.write_log(f"list: {len(subset)} observations displayed")


def cmd_tabulate(rest: str, state: AppState, console: Console):
    """
    tabulate var1 [var2] [if cond] [in range] [weight] [, missing sort nolabel]

    One-way frequency table (one variable) or cross-tabulation (two variables).
    Under weights, the cell values are sums of weights, not row counts.
    """
    from commands.weights import get_weight_series, weight_description
    state.require_data()
    parsed = parse_command_line(rest)
    df = state.data

    cols = parsed["varlist"]
    if not cols:
        console.print("[red]Syntax: tabulate var1 [var2] [if] [in] [weight] [, missing sort nolabel][/red]")
        return

    subset = _apply_if_in(df, parsed["if_cond"], parsed["in_range"])
    weights = get_weight_series(parsed, subset, console)
    if weights is False:
        return

    include_missing = "missing" in parsed["options"] or "m" in parsed["options"]
    sort_freq       = "sort" in parsed["options"]

    wdesc = weight_description(parsed)
    if len(cols) == 1:
        _tab_oneway(subset, cols[0], state, console,
                    include_missing=include_missing, sort_freq=sort_freq,
                    weights=weights, wdesc=wdesc)
    elif len(cols) == 2:
        _tab_twoway(subset, cols[0], cols[1], state, console,
                    include_missing=include_missing,
                    weights=weights, wdesc=wdesc)
    else:
        console.print("[red]tabulate: give one or two variables[/red]")


def _tab_oneway(df, var, state, console, include_missing=False, sort_freq=False,
                weights=None, wdesc=""):
    """One-way frequency table."""
    series = df[var]

    if weights is not None:
        # Weighted tabulation: cell values are sums of weights per category
        tmp = pd.DataFrame({"v": series, "w": weights})
        if not include_missing:
            tmp = tmp.dropna(subset=["v"])
        counts = tmp.groupby("v", dropna=not include_missing)["w"].sum()
    else:
        counts = series.value_counts(dropna=not include_missing)

    if not sort_freq:
        counts = counts.sort_index()

    total = counts.sum()
    label = state.get_variable_label(var)
    title = var
    if label:
        title += f" ({label})"
    if wdesc:
        title += f"  {wdesc}"

    table = Table(title=title)
    table.add_column(var, min_width=20)
    table.add_column("Freq." if weights is None else "Weight", justify="right", min_width=10)
    table.add_column("Percent", justify="right", min_width=8)
    table.add_column("Cum.", justify="right", min_width=8)

    cum = 0
    for val, cnt in counts.items():
        pct = cnt / total * 100 if total > 0 else 0
        cum += pct
        val_label = state.get_value_label_text(var, val)
        display_val = f"{val_label}" if val_label else str(val)
        cnt_fmt = f"{cnt:,.2f}" if weights is not None else f"{cnt:,}"
        table.add_row(display_val, cnt_fmt, f"{pct:.1f}", f"{cum:.1f}")

    table.add_row("─" * 15, "─" * 6, "─" * 6, "─" * 6, style="dim")
    total_fmt = f"{total:,.2f}" if weights is not None else f"{total:,}"
    table.add_row("Total", total_fmt, "100.0", "", style="bold")

    console.print(table)


def _tab_twoway(df, var1, var2, state, console, include_missing=False,
                weights=None, wdesc=""):
    """Two-way cross-tabulation."""
    if weights is not None:
        # crosstab with values= and aggfunc="sum" gives sum-of-weights per cell
        ct = pd.crosstab(
            df[var1], df[var2],
            values=weights, aggfunc="sum",
            margins=True, margins_name="Total",
            dropna=not include_missing,
        ).fillna(0)
        cell_fmt = "{:,.2f}"
    else:
        ct = pd.crosstab(
            df[var1], df[var2],
            margins=True, margins_name="Total",
            dropna=not include_missing
        )
        cell_fmt = "{:,}"

    title = f"{var1} × {var2}"
    if wdesc:
        title += f"  {wdesc}"
    table = Table(title=title)
    table.add_column(var1, style="bold")
    for col in ct.columns:
        table.add_column(str(col), justify="right")

    for idx, row in ct.iterrows():
        table.add_row(str(idx), *[cell_fmt.format(v) for v in row.values])

    console.print(table)


def cmd_summarize(rest: str, state: AppState, console: Console):
    """
    summarize [varlist] [if condition] [in range] [weight] [, detail]

    Weight forms supported:
        [fweight=var]  frequency weights
        [aweight=var]  analytic (inverse-variance) weights
        [pweight=var]  sampling weights
        [iweight=var]  generic importance weights

    With weights, Obs is the sum of weights; Mean/SD/quantiles are computed
    with the appropriate weighted formulas.
    """
    from commands.weights import get_weight_series, weight_description
    state.require_data()
    parsed = parse_command_line(rest)
    df = state.data

    cols = _resolve_varlist(parsed["varlist"], df)
    subset = _apply_if_in(df, parsed["if_cond"], parsed["in_range"])
    weights = get_weight_series(parsed, subset, console)
    if weights is False:
        return

    num_cols = [c for c in cols if pd.api.types.is_numeric_dtype(subset[c])]
    if not num_cols:
        console.print("[yellow]No numeric variables to summarize.[/yellow]")
        return

    detail = "detail" in parsed["options"] or "d" in parsed["options"]
    wtype = parsed["weight"]["type"] if parsed["weight"] else None

    if detail:
        for col in num_cols:
            _summarize_detail(subset, col, state, console, weights=weights, wtype=wtype)
    else:
        _summarize_brief(subset, num_cols, state, console,
                         weights=weights, wtype=wtype, wdesc=weight_description(parsed))


def _summarize_brief(df, cols, state, console, *, weights=None, wtype=None, wdesc=""):
    """Brief summary statistics table, weighted if `weights` is given."""
    from commands.weights import (weighted_mean, weighted_std, weighted_count)

    title = "Summary Statistics"
    if wdesc:
        title += f"  {wdesc}"
    table = Table(title=title)
    table.add_column("Variable", style="bold", min_width=15)
    table.add_column("Obs" if weights is None else "Sum(W)", justify="right")
    table.add_column("Mean", justify="right")
    table.add_column("Std. Dev.", justify="right")
    table.add_column("Min", justify="right")
    table.add_column("Max", justify="right")

    for col in cols:
        if weights is None:
            s = df[col].dropna()
            table.add_row(
                col,
                f"{len(s):,}",
                f"{s.mean():.4f}" if len(s) > 0 else ".",
                f"{s.std():.4f}" if len(s) > 1 else ".",
                f"{s.min():.4f}" if len(s) > 0 else ".",
                f"{s.max():.4f}" if len(s) > 0 else ".",
            )
        else:
            x = df[col].to_numpy()
            w = weights.to_numpy()
            n_weighted = weighted_count(x, w)
            mu = weighted_mean(x, w)
            sd = weighted_std(x, w, wtype or "aweight")
            mask = (~pd.isna(df[col])) & (weights > 0)
            if mask.any():
                vmin, vmax = df.loc[mask, col].min(), df.loc[mask, col].max()
            else:
                vmin = vmax = None
            table.add_row(
                col,
                f"{n_weighted:,.2f}",
                f"{mu:.4f}" if mu == mu else ".",
                f"{sd:.4f}" if sd == sd else ".",
                f"{vmin:.4f}" if vmin is not None else ".",
                f"{vmax:.4f}" if vmax is not None else ".",
            )

    console.print(table)


def _summarize_detail(df, col, state, console, *, weights=None, wtype=None):
    """Detailed summary for a single variable, weighted if `weights` is given."""
    from commands.weights import (weighted_mean, weighted_std,
                                   weighted_quantile, weighted_count)
    label = state.get_variable_label(col)

    header = f"\n[bold]{col}[/bold]"
    if label:
        header += f"  [dim]({label})[/dim]"
    console.print(header)
    console.print("─" * 50)

    if weights is None:
        s = df[col].dropna()
        if len(s) == 0:
            console.print("  No valid observations")
            return
        pctiles = s.quantile([0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99])
        n_label = f"Obs = {len(s):,}"
        mean = s.mean()
        sd = s.std()
        vmin, vmax = s.min(), s.max()
    else:
        x = df[col].to_numpy()
        w = weights.to_numpy()
        n_weighted = weighted_count(x, w)
        if n_weighted == 0:
            console.print("  No valid observations")
            return
        pvals = [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]
        pctiles = pd.Series(
            {p: weighted_quantile(x, w, p) for p in pvals}
        )
        n_label = f"Sum(W) = {n_weighted:,.2f}"
        mean = weighted_mean(x, w)
        sd = weighted_std(x, w, wtype or "aweight")
        mask = (~pd.isna(df[col])) & (weights > 0)
        vmin = df.loc[mask, col].min() if mask.any() else float("nan")
        vmax = df.loc[mask, col].max() if mask.any() else float("nan")

    console.print(f"  {n_label}")
    console.print(f"  Mean       = {mean:>12.4f}")
    console.print(f"  Std. Dev.  = {sd:>12.4f}")
    console.print(f"  Min        = {vmin:>12.4f}")
    console.print(f"  Max        = {vmax:>12.4f}")
    console.print()
    console.print("  Percentiles:")
    for p, v in pctiles.items():
        console.print(f"    {int(p*100):3d}%   {v:>12.4f}")


def cmd_count(rest: str, state: AppState, console: Console):
    """
    count [if condition]
    Count observations.
    """
    state.require_data()
    parsed = parse_command_line(rest)

    subset = _apply_if_in(state.data, parsed["if_cond"], parsed["in_range"])
    n = len(subset)
    console.print(f"  [bold]{n:,}[/bold]")
    state.write_log(f"count = {n}")


# ──────────────────────────────────────────────
#  Helpers
# ──────────────────────────────────────────────

def _display_rich_table(df, state, console, max_rows=None, show_obs=True):
    """Display a DataFrame as a rich table, substituting value labels where
    a variable has one attached (Gemini #4 fix).
    """
    if max_rows and len(df) > max_rows:
        df = df.head(max_rows)

    table = Table(show_lines=False, row_styles=["", "dim"])

    if show_obs:
        table.add_column("#", style="dim", justify="right", width=6)

    for col in df.columns:
        justify = "right" if pd.api.types.is_numeric_dtype(df[col]) else "left"
        table.add_column(col, justify=justify, min_width=8, max_width=30)

    # Pre-compute which columns have value labels assigned
    labeled_cols = {
        col for col in df.columns
        if hasattr(state, "value_label_assignments")
        and col in state.value_label_assignments
    }

    for i, (idx, row) in enumerate(df.iterrows()):
        row_vals = []
        if show_obs:
            row_vals.append(str(i + 1))
        for col in df.columns:
            val = row[col]
            if pd.isna(val):
                row_vals.append("[dim].[/dim]")
            elif col in labeled_cols:
                # Show "code label" when possible, fall back to raw value
                label_text = state.get_value_label_text(col, val)
                if label_text is not None:
                    row_vals.append(f"{val} {label_text}")
                else:
                    row_vals.append(str(val))
            else:
                row_vals.append(str(val))
        table.add_row(*row_vals)

    console.print(table)


def _sb_type_str(dtype) -> str:
    """Convert pandas dtype to compact type string."""
    if pd.api.types.is_integer_dtype(dtype):
        return "long"
    elif pd.api.types.is_float_dtype(dtype):
        return "double"
    elif pd.api.types.is_bool_dtype(dtype):
        return "byte"
    elif pd.api.types.is_object_dtype(dtype) or pd.api.types.is_string_dtype(dtype):
        return "str"
    elif pd.api.types.is_datetime64_any_dtype(dtype):
        return "double (date)"
    elif pd.api.types.is_categorical_dtype(dtype):
        return "long (labeled)"
    return str(dtype)


def _sb_format_str(dtype) -> str:
    """Convert pandas dtype to compact tabular format string."""
    if pd.api.types.is_integer_dtype(dtype):
        return "%12.0g"
    elif pd.api.types.is_float_dtype(dtype):
        return "%10.4f"
    elif pd.api.types.is_datetime64_any_dtype(dtype):
        return "%td"
    return "%10s"


# ──────────────────────────────────────────────
#  tabstat — compact summary stats table
# ──────────────────────────────────────────────

def cmd_tabstat(rest: str, state: AppState, console: Console):
    """
    tabstat varlist [if] [in] [, by(group) stats(mean sd min max ...) cols(stat|var) format(%fmt)]

    Compact compact summary statistics table.

    Default stats: n mean sd min max
    Available: n, count, mean, sd, var, min, max, sum, median, range, iqr,
               pN (any percentile: p1, p5, p25, p50, p75, p90, p95, p99)
    """
    state.require_data()
    parsed = parse_command_line(rest)

    varlist = parsed["varlist"]
    if not varlist:
        console.print("[red]Syntax: tabstat varlist [, by(g) stats(mean sd ...)][/red]")
        return

    df = state.data
    if parsed["if_cond"]:
        df = df.loc[eval_condition(parsed["if_cond"], df)]
    if parsed["in_range"]:
        sl = parse_in_range(parsed["in_range"])
        if sl:
            df = df.iloc[sl]

    # Pull weight vector if one was given
    from commands.weights import (get_weight_series, weighted_mean,
                                   weighted_std, weighted_quantile,
                                   weighted_sum, weighted_count)
    weights = get_weight_series(parsed, df, console)
    if weights is False:
        return
    wtype = parsed["weight"]["type"] if parsed["weight"] else None

    # Only numeric columns make sense for tabstat; warn on string cols
    numeric_vars = [v for v in varlist if v in df.columns and pd.api.types.is_numeric_dtype(df[v])]
    skipped = [v for v in varlist if v not in numeric_vars]
    if skipped:
        console.print(f"[yellow]tabstat: skipping non-numeric columns: {skipped}[/yellow]")
    if not numeric_vars:
        console.print("[red]tabstat: no numeric variables to summarize[/red]")
        return

    stats_str = parsed["options"].get("stats") or parsed["options"].get("statistics") or "n mean sd min max"
    stats = stats_str.split()
    by_var = parsed["options"].get("by")
    fmt = parsed["options"].get("format", "%.4f")
    try:
        fmt_py = fmt.replace("%", "{:") + "}" if fmt.startswith("%") else fmt
        "{:.4f}".format(1.0)
    except Exception:
        fmt_py = "{:.4f}"

    def compute_stat(series, stat, sub_weights=None):
        s = pd.to_numeric(series, errors="coerce")
        if sub_weights is None:
            s = s.dropna()
            if len(s) == 0:
                return float("nan")
            if stat in ("n", "count"): return len(s)
            if stat == "mean":   return s.mean()
            if stat == "sd":     return s.std()
            if stat == "var":    return s.var()
            if stat == "min":    return s.min()
            if stat == "max":    return s.max()
            if stat == "sum":    return s.sum()
            if stat == "median": return s.median()
            if stat == "range":  return s.max() - s.min()
            if stat == "iqr":    return s.quantile(0.75) - s.quantile(0.25)
            if stat.startswith("p") and stat[1:].isdigit():
                return s.quantile(int(stat[1:]) / 100.0)
            raise ValueError(f"Unknown stat '{stat}'")
        # Weighted branch
        x = s.to_numpy()
        w = sub_weights.to_numpy()
        if stat in ("n", "count"): return weighted_count(x, w)
        if stat == "mean":   return weighted_mean(x, w)
        if stat == "sd":     return weighted_std(x, w, wtype or "aweight")
        if stat == "var":
            v = weighted_std(x, w, wtype or "aweight")
            return v * v if v == v else float("nan")
        if stat == "sum":    return weighted_sum(x, w)
        if stat == "median": return weighted_quantile(x, w, 0.5)
        if stat == "range":
            mask = (~pd.isna(s)) & (sub_weights > 0)
            if not mask.any(): return float("nan")
            return series[mask].max() - series[mask].min()
        if stat == "iqr":
            return (weighted_quantile(x, w, 0.75) -
                    weighted_quantile(x, w, 0.25))
        if stat == "min":
            mask = (~pd.isna(s)) & (sub_weights > 0)
            return series[mask].min() if mask.any() else float("nan")
        if stat == "max":
            mask = (~pd.isna(s)) & (sub_weights > 0)
            return series[mask].max() if mask.any() else float("nan")
        if stat.startswith("p") and stat[1:].isdigit():
            return weighted_quantile(x, w, int(stat[1:]) / 100.0)
        raise ValueError(f"Unknown stat '{stat}'")

    def fmt_val(v):
        if v is None or (isinstance(v, float) and (pd.isna(v))):
            return "."
        if isinstance(v, (int, np.integer)):
            return str(int(v))
        try:
            return fmt_py.format(float(v))
        except Exception:
            return str(v)

    # Build table: rows = stats, cols = variables (if no by)
    #              rows = group × stat, cols = variables (if by)
    table = Table(show_header=True, header_style="bold cyan",
                  title=("tabstat: " + ", ".join(numeric_vars)) +
                        (f"  by {by_var}" if by_var else ""))
    if by_var:
        table.add_column(by_var, style="bold")
    table.add_column("stats", style="dim")
    for v in numeric_vars:
        table.add_column(v, justify="right")

    if by_var and by_var in df.columns:
        for grp, sub in df.groupby(by_var):
            sub_w = None if weights is None else weights.loc[sub.index]
            for stat in stats:
                row = [str(grp), stat]
                for v in numeric_vars:
                    row.append(fmt_val(compute_stat(sub[v], stat, sub_w)))
                table.add_row(*row)
    else:
        for stat in stats:
            row = [stat] + [fmt_val(compute_stat(df[v], stat, weights)) for v in numeric_vars]
            table.add_row(*row)

    console.print(table)

    # Store a flat dict in r() for programmatic access
    state.r_results = {}
    for v in numeric_vars:
        for stat in stats:
            try:
                state.r_results[f"{v}_{stat}"] = float(compute_stat(df[v], stat, weights))
            except Exception:
                pass


# ──────────────────────────────────────────────
#  contract — distinct-row tabulation
# ──────────────────────────────────────────────

def cmd_contract(rest: str, state: AppState, console: Console):
    """
    contract varlist [if] [, freq(name) cfreq(name) percent(name) nomiss]

    Reduce the dataset to one row per distinct combination of varlist,
    with a frequency column. Equivalent to SQL's `SELECT varlist, COUNT(*)
    GROUP BY varlist`. Modifies the in-memory DataFrame in place.

    Options:
        freq(name)     name of the frequency column (default: _freq)
        cfreq(name)    add a cumulative-frequency column
        percent(name)  add a percent-of-total column (0-100)
        nomiss         drop rows where any contract variable is missing
    """
    state.require_data()
    parsed = parse_command_line(rest)
    varlist = parsed["varlist"]
    if not varlist:
        console.print("[red]Syntax: contract varlist [, freq(_freq) cfreq() percent() nomiss][/red]")
        return

    missing = [v for v in varlist if v not in state.data.columns]
    if missing:
        console.print(f"[red]contract: variables not found: {missing}[/red]")
        return

    freq_name    = parsed["options"].get("freq")    or "_freq"
    cfreq_name   = parsed["options"].get("cfreq")
    percent_name = parsed["options"].get("percent")
    drop_missing = "nomiss" in parsed["options"]

    df = state.data
    if parsed["if_cond"]:
        df = df.loc[eval_condition(parsed["if_cond"], df)]
    if drop_missing:
        df = df.dropna(subset=varlist)

    result = (df.groupby(varlist, dropna=not drop_missing)
                .size()
                .reset_index(name=freq_name)
                .sort_values(freq_name, ascending=False))

    if cfreq_name:
        result[cfreq_name] = result[freq_name].cumsum()
    if percent_name:
        total = result[freq_name].sum()
        result[percent_name] = (result[freq_name] / total * 100).round(4)

    result = result.reset_index(drop=True)
    state.data = result
    state.mark_changed()

    console.print(f"[green]contract: {len(result):,} distinct combinations of "
                  f"{', '.join(varlist)}[/green]")
    console.print(f"[dim]Dataset replaced. Columns: {', '.join(result.columns)}[/dim]")
