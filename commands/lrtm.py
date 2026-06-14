"""
Polars-powered large-data commands for Rln.
LRTM = Larger-than-RAM Mode.

All lrtm_ commands use Polars lazy evaluation for memory efficiency.
Data stays on disk until results are collected.

Commands:
  lrtm use "file" [, sample(N)]           — Lazy-load large file
  lrtm describe                            — Schema without loading data
  lrtm summarize [varlist]                 — Fast summary statistics
  lrtm count [if condition]                — Count rows efficiently
  lrtm tabulate var1 [var2]                — Fast frequency tables
  lrtm generate newvar = expr              — Add computed column
  lrtm drop if condition                   — Filter rows
  lrtm keep if condition                   — Filter rows (keep matching)
  lrtm sort varlist                        — Sort large dataset
  lrtm merge using "file", on(keys)        — Fast exact merge
  lrtm fuzzmerge var using "file"          — RapidFuzz fuzzy merge (parallel)
  lrtm collapse (stat) var, by(group)      — Grouped aggregation
  lrtm save "file"                         — Save to parquet/csv
  lrtm head [N]                            — Show first N rows
  lrtm collect                             — Materialize to pandas (for Rln commands)
  lrtm status                              — Show current LRTM state
"""

import os
import time
import numpy as np
import pandas as pd
from rich.console import Console
from rich.table import Table

from commands.state import AppState
from commands.parse_helpers import parse_command_line


def _check_polars():
    try:
        import polars as pl
        return pl
    except ImportError:
        raise ImportError(
            "Polars required for LRTM commands.\n"
            "Install with: ssc install polars"
        )


def _check_rapidfuzz():
    try:
        import rapidfuzz
        return rapidfuzz
    except ImportError:
        raise ImportError(
            "RapidFuzz required for lrtm fuzzmerge.\n"
            "Install with: ssc install rapidfuzz"
        )


# ──────────────────────────────────────────────
#  Main dispatcher
# ──────────────────────────────────────────────

def cmd_lrtm(rest: str, state: AppState, console: Console):
    """
    lrtm <command> [args]
    Larger-than-RAM mode. Uses Polars for memory-efficient processing.
    """
    parts = rest.strip().split(None, 1)
    if not parts:
        console.print("[red]Syntax: lrtm use|describe|summarize|count|tabulate|...[/red]")
        _show_lrtm_help(console)
        return

    subcmd = parts[0].lower()
    sub_rest = parts[1] if len(parts) > 1 else ""

    dispatch = {
        "use": _lrtm_use,
        "load": _lrtm_use,
        "describe": _lrtm_describe,
        "desc": _lrtm_describe,
        "summarize": _lrtm_summarize,
        "sum": _lrtm_summarize,
        "count": _lrtm_count,
        "tabulate": _lrtm_tabulate,
        "tab": _lrtm_tabulate,
        "tabstat": _lrtm_tabstat,
        "generate": _lrtm_generate,
        "gen": _lrtm_generate,
        "drop": _lrtm_filter,
        "keep": _lrtm_filter,
        "sort": _lrtm_sort,
        "merge": _lrtm_merge,
        "fuzzmerge": _lrtm_fuzzmerge,
        "append": _lrtm_append,
        "collapse": _lrtm_collapse,
        "contract": _lrtm_contract,
        "list": _lrtm_list,
        "l": _lrtm_list,
        "save": _lrtm_save,
        "export": _lrtm_save,
        "head": _lrtm_head,
        "collect": _lrtm_collect,
        "status": _lrtm_status,
        "clear": _lrtm_clear,
        "convert": _lrtm_convert,
    }

    handler = dispatch.get(subcmd)
    if handler:
        if subcmd in ("drop", "keep"):
            handler(sub_rest, state, console, mode=subcmd)
        else:
            handler(sub_rest, state, console)
    else:
        console.print(f"[red]Unknown lrtm command: {subcmd}[/red]")
        _show_lrtm_help(console)


def _show_lrtm_help(console):
    console.print("[dim]LRTM: Larger-than-RAM Mode (Polars-powered)[/dim]")
    console.print('  [cyan]lrtm use[/cyan]        "big_file.parquet" [, sample(1000)]')
    console.print('  [cyan]lrtm convert[/cyan]    "data.csv" [, output("data.parquet")]')
    console.print("  [cyan]lrtm describe[/cyan]   Show schema without loading")
    console.print("  [cyan]lrtm summarize[/cyan]  [varlist] [if cond]")
    console.print("  [cyan]lrtm tabstat[/cyan]    varlist [if cond] [, by(g) stats(mean sd ...)]")
    console.print("  [cyan]lrtm list[/cyan]       [varlist] [if cond] [, n(10) noobs]")
    console.print("  [cyan]lrtm count[/cyan]      [if condition]")
    console.print("  [cyan]lrtm tabulate[/cyan]   var1 [var2] [if cond]")
    console.print("  [cyan]lrtm contract[/cyan]   varlist [if cond] [, freq() cfreq() percent() nomiss]")
    console.print("  [cyan]lrtm generate[/cyan]   newvar = expression [if cond]")
    console.print("  [cyan]lrtm drop[/cyan]       if condition")
    console.print("  [cyan]lrtm keep[/cyan]       if condition")
    console.print("  [cyan]lrtm sort[/cyan]       varlist")
    console.print('  [cyan]lrtm merge[/cyan]      using "file", on(key1 key2) [how(left|inner|outer)] [nogen]')
    console.print('  [cyan]lrtm fuzzmerge[/cyan]  var using "file", threshold(80) [workers(4)] [raw]')
    console.print('  [cyan]lrtm append[/cyan]     using "file" [, force gen(src)]')
    console.print("  [cyan]lrtm collapse[/cyan]   (stat) var [if cond], by(group)")
    console.print('  [cyan]lrtm save[/cyan]       "output.parquet" [if cond]')
    console.print("  [cyan]lrtm head[/cyan]       [N] [if cond]")
    console.print("  [cyan]lrtm collect[/cyan]    Materialize to pandas")
    console.print("  [cyan]lrtm clear[/cyan]      Drop lazy frame, free RAM")
    console.print("  [cyan]lrtm status[/cyan]     Show LRTM state")


def _require_lrtm(state, console):
    """Ensure LRTM data is loaded."""
    if not hasattr(state, '_lrtm_lf') or state._lrtm_lf is None:
        console.print("[red]No LRTM data loaded. Use: lrtm use \"file\"[/red]")
        return False
    return True


def _lrtm_snapshot(state):
    """Save the current lazy frame + schema so we can roll back on error.

    Polars LazyFrame objects are immutable in practice (each with_columns()
    returns a new plan), so saving the reference is enough — we just need to
    avoid overwriting state._lrtm_lf until we know the mutation is valid.
    """
    return {
        "lf": state._lrtm_lf,
        "schema": getattr(state, "_lrtm_schema", None),
        "row_count": getattr(state, "_lrtm_row_count", None),
    }


def _lrtm_restore(state, snap):
    state._lrtm_lf = snap["lf"]
    state._lrtm_schema = snap["schema"]
    state._lrtm_row_count = snap["row_count"]


def _lrtm_commit(state, new_lf, console, op_label=""):
    """Validate a candidate lazy frame by collecting its schema.
    If schema collection succeeds, commit; otherwise leave state untouched
    and raise. The schema collection is cheap for any sensible query plan.
    """
    # Probe the new plan by collecting its schema — this forces polars to
    # validate the plan without materializing any rows.
    new_schema = new_lf.collect_schema()
    state._lrtm_lf = new_lf
    state._lrtm_schema = new_schema
    state._lrtm_row_count = None  # invalidate cached count


def _apply_lrtm_if(lf, if_cond, pl, console):
    """Apply a if-condition to a lazy frame. Returns filtered lf."""
    if not if_cond:
        return lf
    cond = _sb_to_polars_expr(if_cond, pl)
    if cond is not None:
        return lf.filter(cond)
    else:
        console.print(f"[yellow]Could not parse condition '{if_cond}' for Polars. Ignoring filter.[/yellow]")
        return lf


# ──────────────────────────────────────────────
#  lrtm clear — Drop lazy frame and free RAM
# ──────────────────────────────────────────────

def _lrtm_clear(rest, state, console):
    """Drop the LRTM lazy frame and free associated memory."""
    if hasattr(state, '_lrtm_lf') and state._lrtm_lf is not None:
        source = getattr(state, '_lrtm_source', 'unknown')
        del state._lrtm_lf
        state._lrtm_lf = None
        state._lrtm_schema = None
        state._lrtm_source = None
        state._lrtm_row_count = None

        # Force garbage collection
        import gc
        gc.collect()

        console.print(f"[dim]LRTM cleared: {source}[/dim]")
        console.print("[dim]Lazy frame dropped. RAM freed.[/dim]")
    else:
        console.print("[dim]No LRTM data to clear.[/dim]")


# ──────────────────────────────────────────────
#  lrtm convert — Convert any format to Parquet
# ──────────────────────────────────────────────

def _lrtm_convert(rest, state, console):
    """
    Convert CSV/Excel/DBF/DTA/RData to Parquet for fast LRTM access.
    
    lrtm convert "data.csv" [, output("data.parquet") chunk(500000)]
    lrtm convert "survey.dta"
    lrtm convert "records.xlsx"
    lrtm convert "legacy.dbf"
    
    Reads the source file in chunks (for CSV) or eagerly (other formats),
    then writes to Parquet with optimal compression.
    After conversion, use: lrtm use "data.parquet" for zero-RAM lazy loading.
    """
    pl = _check_polars()
    parsed = parse_command_line(rest)

    filepath = None
    if parsed["varlist"]:
        filepath = parsed["varlist"][0]
    if not filepath:
        console.print('[red]Syntax: lrtm convert "file.csv" [, output("file.parquet")][/red]')
        return

    filepath = os.path.expanduser(filepath)
    if not os.path.exists(filepath):
        console.print(f"[red]File not found: {filepath}[/red]")
        return

    ext = os.path.splitext(filepath)[1].lower()
    base = os.path.splitext(filepath)[0]
    output_path = parsed["options"].get("output", f"{base}.parquet")
    chunk_size = int(parsed["options"].get("chunk", parsed["options"].get("chunksize", 500000)))

    t0 = time.time()
    console.print(f"[dim]Converting {filepath} -> {output_path}...[/dim]")

    try:
        if ext == ".csv":
            # CSV: Polars can read directly and write to Parquet
            console.print("[dim]  Reading CSV with Polars...[/dim]")
            df = pl.read_csv(filepath, try_parse_dates=True, infer_schema_length=10000)
            console.print(f"[dim]  Read {len(df):,} rows x {len(df.columns)} cols[/dim]")

        elif ext == ".tsv":
            df = pl.read_csv(filepath, separator="\t", try_parse_dates=True)

        elif ext in (".xlsx", ".xls"):
            console.print("[dim]  Reading Excel with pandas (then converting)...[/dim]")
            import pandas as pd_lib
            sheet = parsed["options"].get("sheet", 0)
            try:
                sheet = int(sheet)
            except ValueError:
                pass
            pdf = pd_lib.read_excel(filepath, sheet_name=sheet)
            df = pl.from_pandas(pdf)
            console.print(f"[dim]  Read {len(df):,} rows x {len(df.columns)} cols[/dim]")

        elif ext == ".dta":
            console.print("[dim]  Reading other statistical tools .dta with pandas (then converting)...[/dim]")
            import pandas as pd_lib
            pdf = pd_lib.read_stata(filepath)
            df = pl.from_pandas(pdf)
            console.print(f"[dim]  Read {len(df):,} rows x {len(df.columns)} cols[/dim]")

        elif ext == ".dbf":
            console.print("[dim]  Reading DBF with dbfread (then converting)...[/dim]")
            from dbfread import DBF
            import pandas as pd_lib
            records = list(DBF(filepath, encoding="latin-1"))
            pdf = pd_lib.DataFrame(records)
            df = pl.from_pandas(pdf)
            console.print(f"[dim]  Read {len(df):,} rows x {len(df.columns)} cols[/dim]")

        elif ext in (".rdata", ".rds", ".rda"):
            console.print("[dim]  Reading R data with pyreadr (then converting)...[/dim]")
            import pyreadr
            result = pyreadr.read_r(filepath)
            first_key = list(result.keys())[0]
            pdf = result[first_key]
            df = pl.from_pandas(pdf)
            console.print(f"[dim]  Read {len(df):,} rows x {len(df.columns)} cols[/dim]")

        elif ext == ".json":
            df = pl.read_json(filepath)

        elif ext in (".parquet", ".pq"):
            console.print("[yellow]File is already Parquet. No conversion needed.[/yellow]")
            return

        else:
            console.print(f"[red]Unsupported format: {ext}[/red]")
            console.print("[dim]Supported: .csv, .tsv, .xlsx, .xls, .dta, .dbf, .rdata, .rds, .json[/dim]")
            return

        # Write to Parquet with compression
        df.write_parquet(output_path, compression="zstd")

        elapsed = time.time() - t0
        src_size = os.path.getsize(filepath) / 1e6
        dst_size = os.path.getsize(output_path) / 1e6
        ratio = src_size / dst_size if dst_size > 0 else 0

        console.print(f"\n[green]Conversion complete:[/green]")
        console.print(f"  Source:  {filepath} ({src_size:.1f} MB)")
        console.print(f"  Output:  {output_path} ({dst_size:.1f} MB)")
        console.print(f"  Rows:    {len(df):,} x {len(df.columns)} columns")
        console.print(f"  Compression: {ratio:.1f}x smaller")
        console.print(f"  Time:    {elapsed:.1f}s")
        console.print(f"\n[dim]Now use: lrtm use \"{output_path}\"[/dim]")

    except Exception as e:
        console.print(f"[red]Conversion failed: {e}[/red]")


# ──────────────────────────────────────────────
#  lrtm use — Lazy-load large file
# ──────────────────────────────────────────────

def _lrtm_use(rest, state, console):
    """Lazy-load a file with Polars (no RAM usage until collect)."""
    pl = _check_polars()
    parsed = parse_command_line(rest)

    filepath = None
    if parsed["varlist"]:
        filepath = parsed["varlist"][0]
    if not filepath:
        console.print('[red]Syntax: lrtm use "filename" [, sample(N)][/red]')
        return

    filepath = os.path.expanduser(filepath)
    if not os.path.exists(filepath):
        console.print(f"[red]File not found: {filepath}[/red]")
        return

    # Auto-clear previous lazy frame to prevent RAM stacking
    if hasattr(state, '_lrtm_lf') and state._lrtm_lf is not None:
        state._lrtm_lf = None
        state._lrtm_schema = None
        state._lrtm_source = None
        import gc
        gc.collect()

    ext = os.path.splitext(filepath)[1].lower()
    sample_n = int(parsed["options"].get("sample", 0))

    t0 = time.time()
    console.print(f"[dim]LRTM: Lazy-loading {filepath}...[/dim]")

    try:
        if ext in (".parquet", ".pq"):
            lf = pl.scan_parquet(filepath)
        elif ext == ".csv":
            lf = pl.scan_csv(filepath, try_parse_dates=True)
        elif ext == ".json":
            # JSON needs eager load then convert to lazy
            df = pl.read_json(filepath)
            lf = df.lazy()
        elif ext == ".ipc":
            lf = pl.scan_ipc(filepath)
        else:
            console.print(f"[red]LRTM supports: .parquet, .csv, .json, .ipc[/red]")
            return

        # Store lazy frame
        state._lrtm_lf = lf
        state._lrtm_source = filepath
        state._lrtm_schema = lf.collect_schema()
        state._lrtm_row_count = None  # Invalidate cached count after mutation

        # Get row count — fast for Parquet (metadata), skip for CSV (full scan)
        if ext in (".parquet", ".pq"):
            try:
                n_rows = lf.select(pl.len()).collect().item()
                state._lrtm_row_count = n_rows
            except Exception:
                n_rows = "unknown"
                state._lrtm_row_count = None
        else:
            n_rows = "(lazy)"
            state._lrtm_row_count = None

        elapsed = time.time() - t0
        n_cols = len(state._lrtm_schema)
        if isinstance(n_rows, int):
            console.print(f"[green]LRTM loaded: {n_rows:,} rows x {n_cols} columns ({elapsed:.2f}s)[/green]")
        else:
            console.print(f"[green]LRTM loaded: {n_rows} rows x {n_cols} columns ({elapsed:.2f}s)[/green]")
        console.print(f"[dim]Data stays on disk until you run lrtm collect[/dim]")

        # Show sample if requested
        if sample_n > 0:
            sample = lf.head(sample_n).collect()
            state.set_data(sample.to_pandas(), name=os.path.basename(filepath), source=filepath)
            console.print(f"[dim]Sample of {sample_n} rows loaded into main memory[/dim]")

    except Exception as e:
        console.print(f"[red]LRTM load failed: {e}[/red]")


# ──────────────────────────────────────────────
#  lrtm describe
# ──────────────────────────────────────────────

def _lrtm_describe(rest, state, console):
    if not _require_lrtm(state, console):
        return
    pl = _check_polars()

    schema = state._lrtm_schema
    console.print(f"\n[bold]LRTM Dataset: {state._lrtm_source}[/bold]")

    # Only count rows if we have a cached count or it's a direct Parquet scan
    # (no filters/merges applied). Otherwise skip to avoid full materialization.
    if hasattr(state, '_lrtm_row_count') and state._lrtm_row_count is not None:
        console.print(f"  Rows: {state._lrtm_row_count:,}")
    else:
        console.print(f"  Rows: (lazy \u2014 use 'lrtm count' to compute)")

    console.print(f"  Columns: {len(schema)}\n")

    table = Table(show_lines=False)
    table.add_column("#", justify="right", min_width=4)
    table.add_column("Variable", style="bold", min_width=20)
    table.add_column("Type", min_width=12)

    for i, (name, dtype) in enumerate(schema.items(), 1):
        table.add_row(str(i), name, str(dtype))

    console.print(table)


# ──────────────────────────────────────────────
#  lrtm summarize
# ──────────────────────────────────────────────

def _lrtm_summarize(rest, state, console):
    if not _require_lrtm(state, console):
        return
    pl = _check_polars()

    parsed = parse_command_line(rest)
    lf = state._lrtm_lf

    # Apply if condition
    lf = _apply_lrtm_if(lf, parsed["if_cond"], pl, console)

    # Select numeric columns
    numeric_types = [pl.Int8, pl.Int16, pl.Int32, pl.Int64, pl.UInt8, pl.UInt16,
                     pl.UInt32, pl.UInt64, pl.Float32, pl.Float64]

    if parsed["varlist"]:
        cols = parsed["varlist"]
    else:
        cols = [name for name, dtype in state._lrtm_schema.items()
                if dtype in numeric_types]

    if not cols:
        console.print("[yellow]No numeric columns found[/yellow]")
        return

    t0 = time.time()
    console.print(f"[dim]Computing statistics for {len(cols)} variables...[/dim]")

    # Build aggregation expressions
    aggs = []
    for c in cols:
        aggs.extend([
            pl.col(c).count().alias(f"{c}_n"),
            pl.col(c).mean().alias(f"{c}_mean"),
            pl.col(c).std().alias(f"{c}_sd"),
            pl.col(c).min().alias(f"{c}_min"),
            pl.col(c).max().alias(f"{c}_max"),
        ])

    result = lf.select(aggs).collect()
    elapsed = time.time() - t0

    table = Table(title=f"Summary Statistics ({elapsed:.2f}s)", show_lines=False)
    table.add_column("Variable", style="bold", min_width=18)
    table.add_column("Obs", justify="right")
    table.add_column("Mean", justify="right")
    table.add_column("Std. Dev.", justify="right")
    table.add_column("Min", justify="right")
    table.add_column("Max", justify="right")

    for c in cols:
        n = result[f"{c}_n"][0]
        mean = result[f"{c}_mean"][0]
        sd = result[f"{c}_sd"][0]
        mn = result[f"{c}_min"][0]
        mx = result[f"{c}_max"][0]
        table.add_row(c,
                      f"{n:,}" if n is not None else ".",
                      f"{mean:.4f}" if mean is not None else ".",
                      f"{sd:.4f}" if sd is not None else ".",
                      f"{mn}" if mn is not None else ".",
                      f"{mx}" if mx is not None else ".")

    console.print(table)


# ──────────────────────────────────────────────
#  lrtm count
# ──────────────────────────────────────────────

def _lrtm_count(rest, state, console):
    if not _require_lrtm(state, console):
        return
    pl = _check_polars()

    t0 = time.time()
    parsed = parse_command_line(rest)

    lf = state._lrtm_lf
    if parsed["if_cond"]:
        # Convert condition to Polars expression
        cond = _sb_to_polars_expr(parsed["if_cond"], pl)
        if cond is not None:
            n = lf.filter(cond).select(pl.len()).collect().item()
        else:
            console.print("[yellow]Could not parse condition for Polars. Collecting to pandas...[/yellow]")
            from commands.expression import eval_condition
            df = lf.collect().to_pandas()
            mask = eval_condition(parsed["if_cond"], df)
            n = mask.sum()
    else:
        n = lf.select(pl.len()).collect().item()

    elapsed = time.time() - t0
    console.print(f"  {n:,}  [dim]({elapsed:.2f}s)[/dim]")


# ──────────────────────────────────────────────
#  lrtm tabulate
# ──────────────────────────────────────────────

def _lrtm_tabulate(rest, state, console):
    if not _require_lrtm(state, console):
        return
    pl = _check_polars()

    parsed = parse_command_line(rest)
    if not parsed["varlist"]:
        console.print("[red]Syntax: lrtm tabulate varname [var2][/red]")
        return

    t0 = time.time()
    lf = state._lrtm_lf

    # Apply if condition
    lf = _apply_lrtm_if(lf, parsed["if_cond"], pl, console)

    if len(parsed["varlist"]) >= 2:
        # Two-way crosstab
        var1, var2 = parsed["varlist"][0], parsed["varlist"][1]
        _lrtm_crosstab(lf, var1, var2, pl, t0, console)
        return

    # One-way frequency table
    var = parsed["varlist"][0]
    freq = (lf.group_by(var)
            .agg(pl.len().alias("count"))
            .sort("count", descending=True)
            .collect())

    elapsed = time.time() - t0
    total = freq["count"].sum()

    table = Table(title=f"{var} ({elapsed:.2f}s)", show_lines=False)
    table.add_column(var, style="bold", min_width=20)
    table.add_column("Freq.", justify="right")
    table.add_column("Percent", justify="right")
    table.add_column("Cum.", justify="right")

    cum = 0
    for row in freq.iter_rows(named=True):
        val = row[var]
        cnt = row["count"]
        pct = cnt / total * 100
        cum += pct
        table.add_row(str(val), f"{cnt:,}", f"{pct:.1f}", f"{cum:.1f}")

    table.add_row("Total", f"{total:,}", "100.0", "")
    console.print(table)


def _lrtm_crosstab(lf, var1, var2, pl, t0, console):
    """Two-way cross-tabulation using Polars group_by. Only reads 2 columns."""
    # Compute crosstab via group_by on both variables
    cross = (lf.group_by([var1, var2])
             .agg(pl.len().alias("count"))
             .sort([var1, var2])
             .collect())

    elapsed = time.time() - t0

    # Pivot to wide format
    pivot = cross.pivot(on=var2, index=var1, values="count").fill_null(0)

    # Get column order (var2 values)
    col2_vals = sorted(cross[var2].unique().to_list())
    col2_names = [str(v) for v in col2_vals]

    table = Table(title=f"{var1} x {var2} ({elapsed:.2f}s)", show_lines=False)
    table.add_column(var1, style="bold", min_width=14)
    for c in col2_names:
        table.add_column(str(c), justify="right", min_width=7)
    table.add_column("Total", justify="right", min_width=8, style="bold")

    row_totals = []
    for row in pivot.iter_rows(named=True):
        row_vals = []
        row_label = str(row[var1])
        total = 0
        for c in col2_names:
            val = row.get(c, row.get(str(c), 0)) or 0
            row_vals.append(f"{val:,}")
            total += val
        row_vals.append(f"{total:,}")
        row_totals.append(total)
        table.add_row(row_label, *row_vals)

    # Column totals
    col_totals = ["Total"]
    grand = 0
    for c in col2_names:
        ct = sum(cross.filter(pl.col(var2) == (int(c) if c.isdigit() else c))["count"].to_list())
        col_totals.append(f"{ct:,}")
        grand += ct
    col_totals.append(f"{grand:,}")
    table.add_row(*col_totals, style="bold")

    console.print(table)


# ──────────────────────────────────────────────
#  lrtm merge — Fast exact merge
# ──────────────────────────────────────────────

def _lrtm_merge(rest, state, console):
    """
    lrtm merge using "file", on(key1 key2) [how(inner|left|right|outer)] [nogen]

    Produces a `_merge` indicator column matching standard conventions:
        1 = master only (left-only)
        2 = using only  (right-only)
        3 = matched (both)

    Pass nogen (or nogenerate) to suppress the _merge column.
    """
    if not _require_lrtm(state, console):
        return
    pl = _check_polars()
    parsed = parse_command_line(rest)

    using_file = parsed["using"]
    on_keys = parsed["options"].get("on", "").split()
    how = parsed["options"].get("how", "left").lower()
    if how not in ("inner", "left", "right", "outer", "full"):
        console.print(f"[red]Unknown how({how}). Valid: inner, left, right, outer[/red]")
        return
    if how == "full":
        how = "outer"
    nogen = any(k in parsed["options"] for k in ("nogen", "nogenerate"))

    if not using_file or not on_keys:
        console.print('[red]Syntax: lrtm merge using "file", on(key1 key2) [how(left|inner|outer)] [nogen][/red]')
        return

    if not os.path.exists(using_file):
        console.print(f"[red]File not found: {using_file}[/red]")
        return

    snap = _lrtm_snapshot(state)
    t0 = time.time()
    try:
        ext = os.path.splitext(using_file)[1].lower()
        if ext in (".parquet", ".pq"):
            right = pl.scan_parquet(using_file)
        elif ext == ".csv":
            right = pl.scan_csv(using_file)
        else:
            right = pl.LazyFrame(pl.read_csv(using_file))

        if nogen:
            # Fast path — plain join without the _merge column
            new_lf = state._lrtm_lf.join(right, on=on_keys, how=how)
        else:
            # Build _merge indicator by tagging each side and using outer-join
            # semantics, then folding back to the requested `how`.
            left_tagged  = state._lrtm_lf.with_columns(pl.lit(1).alias("_merge_left"))
            right_tagged = right.with_columns(pl.lit(1).alias("_merge_right"))

            joined = left_tagged.join(right_tagged, on=on_keys, how="full", coalesce=True)
            joined = joined.with_columns(
                pl.when(pl.col("_merge_left").is_null())
                  .then(pl.lit(2))                         # using only
                  .when(pl.col("_merge_right").is_null())
                  .then(pl.lit(1))                         # master only
                  .otherwise(pl.lit(3))                    # matched
                  .alias("_merge")
            ).drop(["_merge_left", "_merge_right"])

            # If user asked for something narrower than outer, filter accordingly
            if how == "left":
                joined = joined.filter(pl.col("_merge") != 2)
            elif how == "right":
                joined = joined.filter(pl.col("_merge") != 1)
            elif how == "inner":
                joined = joined.filter(pl.col("_merge") == 3)

            new_lf = joined

        _lrtm_commit(state, new_lf, console, "merge")
        elapsed = time.time() - t0

        if not nogen:
            # Quick counts for user feedback (streams through the plan once)
            mcounts = new_lf.group_by("_merge").len().collect()
            stats = {int(r["_merge"]): int(r["len"]) for r in mcounts.iter_rows(named=True)}
            console.print(f"[green]Merge complete ({elapsed:.2f}s). _merge breakdown:[/green]")
            console.print(f"  [cyan]1 master only[/cyan]: {stats.get(1, 0):,}")
            console.print(f"  [cyan]2 using only [/cyan]: {stats.get(2, 0):,}")
            console.print(f"  [cyan]3 matched    [/cyan]: {stats.get(3, 0):,}")
        else:
            console.print(f"[green]Merge complete ({elapsed:.2f}s, no _merge indicator).[/green]")
    except Exception as e:
        _lrtm_restore(state, snap)
        console.print(f"[red]Error: {e}[/red]")
        console.print("[dim]Lazy frame left unchanged.[/dim]")


# ──────────────────────────────────────────────
#  lrtm fuzzmerge — RapidFuzz parallel fuzzy merge
# ──────────────────────────────────────────────

def _lrtm_fuzzmerge(rest, state, console):
    """
    lrtm fuzzmerge var using "file", threshold(80) [workers(4)] [scale10]

    Adds three new columns to the master:
        <var>_match  — the matched string from the using file (null if unmatched)
        <var>_score  — match score in [0, 10] with 5-decimal precision
                       (default scale). Pass `raw` to keep the raw 0-100 score.
        _merge       — 1 = master only (unmatched)
                       3 = matched above threshold
                       (2 = using-only isn't produced by fuzzmerge — only the master is enumerated)

    Options:
        threshold(N)  — min score on the 0-100 scale to keep a match (default 80)
        workers(N)    — thread pool size (default: os.cpu_count())
        raw           — keep scores on 0-100 scale instead of rescaling to 0-10
    """
    if not _require_lrtm(state, console):
        return
    pl = _check_polars()
    rf = _check_rapidfuzz()
    from rapidfuzz import fuzz, process

    parsed = parse_command_line(rest)
    if not parsed["varlist"] or not parsed["using"]:
        console.print('[red]Syntax: lrtm fuzzmerge var using "file", threshold(80) [workers(4)] [raw][/red]')
        return

    var = parsed["varlist"][0]
    using_file = parsed["using"]
    threshold = float(parsed["options"].get("threshold", 80))
    n_workers = int(parsed["options"].get("workers", os.cpu_count() or 4))
    use_raw_scale = "raw" in parsed["options"]

    if not os.path.exists(using_file):
        console.print(f"[red]File not found: {using_file}[/red]")
        return

    snap = _lrtm_snapshot(state)
    t0 = time.time()
    try:
        console.print(f"[dim]LRTM fuzzy merge on '{var}' with {n_workers} workers...[/dim]")

        left_vals = state._lrtm_lf.select(var).collect()[var].to_list()

        ext = os.path.splitext(using_file)[1].lower()
        if ext in (".parquet", ".pq"):
            right_df = pl.read_parquet(using_file)
        else:
            right_df = pl.read_csv(using_file)

        right_vals = right_df[var].to_list()
        right_strs = [str(r) for r in right_vals]

        console.print(f"[dim]  Matching {len(left_vals):,} x {len(right_vals):,} pairs...[/dim]")

        from concurrent.futures import ThreadPoolExecutor

        chunk_size = max(len(left_vals) // max(n_workers, 1), 1)
        chunks = [left_vals[i:i + chunk_size] for i in range(0, len(left_vals), chunk_size)]

        def match_chunk(chunk):
            results = []
            for val in chunk:
                if val is None:
                    results.append((None, 0.0))
                    continue
                best = process.extractOne(str(val), right_strs,
                                           scorer=fuzz.WRatio, score_cutoff=threshold)
                if best:
                    # best = (match_string, score_0_to_100, index)
                    results.append((best[0], float(best[1])))
                else:
                    results.append((None, 0.0))
            return results

        all_results = []
        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            futures = [executor.submit(match_chunk, chunk) for chunk in chunks]
            for f in futures:
                all_results.extend(f.result())

        match_col = [r[0] for r in all_results]
        # compact: rescale to 0-10 with 5-decimal precision unless `raw` given
        if use_raw_scale:
            score_col = [round(r[1], 5) for r in all_results]
        else:
            score_col = [round(r[1] / 10.0, 5) for r in all_results]
        merge_col = [3 if m is not None else 1 for m in match_col]
        matched = sum(1 for m in match_col if m is not None)

        match_df = pl.DataFrame({
            f"{var}_match": match_col,
            f"{var}_score": score_col,
            "_merge":       merge_col,
        })

        left_df = state._lrtm_lf.collect()
        combined = pl.concat([left_df, match_df], how="horizontal")
        new_lf = combined.lazy()
        _lrtm_commit(state, new_lf, console, "fuzzmerge")

        elapsed = time.time() - t0
        scale_note = "0-100 raw" if use_raw_scale else "0-10 scaled"
        console.print(f"[green]Fuzzy merge: {matched:,}/{len(left_vals):,} matched ({elapsed:.2f}s)[/green]")
        console.print(f"[dim]  Generated: {var}_match, {var}_score ({scale_note}), _merge[/dim]")
    except Exception as e:
        _lrtm_restore(state, snap)
        console.print(f"[red]Error: {e}[/red]")
        console.print("[dim]Lazy frame left unchanged.[/dim]")


# ──────────────────────────────────────────────
#  lrtm collapse, generate, filter, sort, save
# ──────────────────────────────────────────────

def _lrtm_collapse(rest, state, console):
    if not _require_lrtm(state, console):
        return
    pl = _check_polars()
    parsed = parse_command_line(rest)

    by_var = parsed["options"].get("by")
    if not by_var:
        console.print("[red]Syntax: lrtm collapse (mean) var [if cond], by(groupvar)[/red]")
        return

    import re

    # Get the part before comma (options), and strip any "if ..." clause
    raw = parsed["raw"].split(",")[0].strip()
    # Remove "if ..." from raw so we get just "(stat) var"
    raw_no_if = re.sub(r'\s+if\s+.+$', '', raw).strip()

    m = re.match(r'\((\w+)\)\s+(\w+)', raw_no_if)
    if not m:
        console.print("[red]Syntax: lrtm collapse (mean) var [if cond], by(groupvar)[/red]")
        return

    stat = m.group(1).lower()
    var = m.group(2).strip()
    by_vars = by_var.split()

    t0 = time.time()

    # Apply if condition before collapsing
    lf = state._lrtm_lf
    if parsed["if_cond"]:
        lf = _apply_lrtm_if(lf, parsed["if_cond"], pl, console)

    # Build aggregation expression (avoid creating all eagerly)
    col = pl.col(var)
    if stat == "mean":
        agg_expr = col.mean()
    elif stat == "sum":
        agg_expr = col.sum()
    elif stat == "count":
        agg_expr = col.count()
    elif stat == "min":
        agg_expr = col.min()
    elif stat == "max":
        agg_expr = col.max()
    elif stat == "median":
        agg_expr = col.median()
    elif stat == "sd":
        agg_expr = col.std()
    elif stat == "first":
        agg_expr = col.first()
    elif stat == "last":
        agg_expr = col.last()
    else:
        console.print(f"[red]Unknown stat: {stat}[/red]")
        return

    result = lf.group_by(by_vars).agg(agg_expr).sort(by_vars).collect()
    state._lrtm_lf = result.lazy()
    state._lrtm_schema = state._lrtm_lf.collect_schema()
    state._lrtm_row_count = None  # Invalidate cached count after mutation

    elapsed = time.time() - t0
    console.print(f"[green]Collapsed to {len(result):,} groups ({elapsed:.2f}s)[/green]")


# ──────────────────────────────────────────────
#  lrtm append — Stack another file below the current lazy frame
# ──────────────────────────────────────────────

def _lrtm_append(rest, state, console):
    """
    lrtm append using "file" [, force] [gen(src)]

    Stacks the rows of `file` below the current lazy frame. Columns are
    aligned by name; extra columns on either side are padded with nulls.

    Options:
        force   — allow columns whose dtypes differ (coerce via cast)
        gen(v)  — create column `v` with 0 for master rows, 1 for appended
    """
    if not _require_lrtm(state, console):
        return
    pl = _check_polars()
    parsed = parse_command_line(rest)

    using_file = parsed["using"]
    if not using_file:
        console.print('[red]Syntax: lrtm append using "file" [, force gen(src)][/red]')
        return
    if not os.path.exists(using_file):
        console.print(f"[red]File not found: {using_file}[/red]")
        return

    force_cast = "force" in parsed["options"]
    src_var = parsed["options"].get("gen") or parsed["options"].get("generate")

    snap = _lrtm_snapshot(state)
    t0 = time.time()
    try:
        ext = os.path.splitext(using_file)[1].lower()
        if ext in (".parquet", ".pq"):
            other = pl.scan_parquet(using_file)
        elif ext == ".csv":
            other = pl.scan_csv(using_file)
        else:
            other = pl.LazyFrame(pl.read_csv(using_file))

        how = "diagonal_relaxed" if force_cast else "diagonal"

        left_lf = state._lrtm_lf
        right_lf = other
        if src_var:
            left_lf = left_lf.with_columns(pl.lit(0).alias(src_var))
            right_lf = right_lf.with_columns(pl.lit(1).alias(src_var))

        new_lf = pl.concat([left_lf, right_lf], how=how)
        _lrtm_commit(state, new_lf, console, "append")

        elapsed = time.time() - t0
        gen_note = f" [{src_var}=0/1]" if src_var else ""
        console.print(f"[green]Appended rows from {os.path.basename(using_file)}{gen_note} "
                      f"({elapsed:.2f}s).[/green]")
        console.print("[dim]Run 'lrtm count' to see the new total row count.[/dim]")
    except Exception as e:
        _lrtm_restore(state, snap)
        console.print(f"[red]Error: {e}[/red]")
        if not force_cast:
            console.print("[dim]Hint: retry with `force` if column dtypes differ.[/dim]")


# ──────────────────────────────────────────────
#  lrtm contract — count distinct combinations (other statistical tools `contract`)
# ──────────────────────────────────────────────

def _lrtm_contract(rest, state, console):
    """
    lrtm contract varlist [if cond] [, freq(name) cfreq(name) percent(name) nomiss]

    Replaces the lazy frame with one row per distinct combination of `varlist`,
    plus a frequency column (default: _freq). Semantically equivalent to
    `collapse (count) , by(varlist)` but honors the standard convention's `contract` syntax.

    Options:
        freq(name)    name of the frequency column (default: _freq)
        cfreq(name)   add a cumulative-frequency column with this name
        percent(name) add a percent column with this name
        nomiss        drop rows where any contract variable is null
    """
    if not _require_lrtm(state, console):
        return
    pl = _check_polars()
    parsed = parse_command_line(rest)

    varlist = parsed["varlist"]
    if not varlist:
        console.print("[red]Syntax: lrtm contract varlist [if cond] [, freq(_freq) cfreq() percent() nomiss][/red]")
        return

    freq_name    = parsed["options"].get("freq")    or "_freq"
    cfreq_name   = parsed["options"].get("cfreq")
    percent_name = parsed["options"].get("percent")
    drop_missing = "nomiss" in parsed["options"]

    snap = _lrtm_snapshot(state)
    t0 = time.time()
    try:
        lf = state._lrtm_lf
        if parsed["if_cond"]:
            lf = _apply_lrtm_if(lf, parsed["if_cond"], pl, console)
        if drop_missing:
            lf = lf.drop_nulls(subset=varlist)

        result = lf.group_by(varlist).len(name=freq_name).sort(freq_name, descending=True)

        # Compute auxiliary columns lazily
        if cfreq_name or percent_name:
            # Need total for percent; also cumulative freq needs sorted sum
            total = result.select(pl.col(freq_name).sum()).collect().item()
            added = []
            if cfreq_name:
                added.append(pl.col(freq_name).cum_sum().alias(cfreq_name))
            if percent_name:
                added.append((pl.col(freq_name) / total * 100).round(4).alias(percent_name))
            result = result.with_columns(added)

        _lrtm_commit(state, result, console, "contract")
        elapsed = time.time() - t0
        n_groups = result.select(pl.len()).collect().item()
        console.print(f"[green]Contract: {n_groups:,} distinct combinations of "
                      f"{', '.join(varlist)} ({elapsed:.2f}s)[/green]")
        console.print("[dim]The lazy frame now contains the contracted rows. "
                      "Run 'lrtm head' or 'lrtm collect' to inspect.[/dim]")
    except Exception as e:
        _lrtm_restore(state, snap)
        console.print(f"[red]Error: {e}[/red]")


# ──────────────────────────────────────────────
#  lrtm list — print the first N rows (non-destructive preview)
# ──────────────────────────────────────────────

def _lrtm_list(rest, state, console):
    """
    lrtm list [varlist] [if cond] [in #/#] [, n(N) noobs]

    Preview rows of the lazy frame without materializing the whole dataset.
    Default row count: 10. Use `n(N)` for a different count.

    Options:
        n(N)     number of rows to show (default 10)
        noobs    suppress the observation-number column
    """
    if not _require_lrtm(state, console):
        return
    pl = _check_polars()
    parsed = parse_command_line(rest)

    n_rows = int(parsed["options"].get("n", 10))
    show_obs = "noobs" not in parsed["options"]

    t0 = time.time()
    try:
        lf = state._lrtm_lf
        if parsed["varlist"]:
            lf = lf.select(parsed["varlist"])
        if parsed["if_cond"]:
            lf = _apply_lrtm_if(lf, parsed["if_cond"], pl, console)

        # Handle `in M/N` slicing (1-indexed, inclusive as documented here)
        in_slice = parsed.get("in_range")
        if in_slice:
            m, n = in_slice
            if m is not None and n is not None:
                lf = lf.slice(m - 1, n - m + 1)
            elif n is not None:
                lf = lf.head(n)

        df = lf.head(n_rows).collect()
        elapsed = time.time() - t0

        # Print with rich Table
        from rich.table import Table
        table = Table(show_header=True, header_style="bold cyan")
        if show_obs:
            table.add_column("#", justify="right", style="dim")
        for col in df.columns:
            table.add_column(col)

        for i, row in enumerate(df.iter_rows(), start=1):
            formatted = [str(v) if v is not None else "." for v in row]
            if show_obs:
                table.add_row(str(i), *formatted)
            else:
                table.add_row(*formatted)

        console.print(table)
        console.print(f"[dim]({len(df):,} of {n_rows} requested rows, "
                      f"scanned lazily in {elapsed:.2f}s)[/dim]")
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")


# ──────────────────────────────────────────────
#  lrtm tabstat — compact summary stats table
# ──────────────────────────────────────────────

def _lrtm_tabstat(rest, state, console):
    """
    lrtm tabstat varlist [if cond] [, by(groupvar) stats(mean sd min max ...) ]

    Compact summary statistics. Default stats: n, mean, sd, min, max.

    Available stats: n, count, mean, sd, var, min, max, sum, median,
                     p1, p5, p10, p25, p50, p75, p90, p95, p99, range, iqr
    """
    if not _require_lrtm(state, console):
        return
    pl = _check_polars()
    parsed = parse_command_line(rest)

    varlist = parsed["varlist"]
    if not varlist:
        console.print("[red]Syntax: lrtm tabstat varlist [, by(g) stats(mean sd ...)][/red]")
        return

    stats_str = parsed["options"].get("stats") or parsed["options"].get("statistics") or "n mean sd min max"
    stats = stats_str.split()
    by_var = parsed["options"].get("by")

    def _stat_expr(stat, col):
        c = pl.col(col)
        if stat in ("n", "count"):   return c.count()
        if stat == "mean":           return c.mean()
        if stat == "sd":             return c.std()
        if stat == "var":            return c.var()
        if stat == "min":            return c.min()
        if stat == "max":            return c.max()
        if stat == "sum":            return c.sum()
        if stat == "median":         return c.median()
        if stat == "range":          return c.max() - c.min()
        if stat == "iqr":            return c.quantile(0.75) - c.quantile(0.25)
        if stat.startswith("p") and stat[1:].isdigit():
            return c.quantile(int(stat[1:]) / 100.0)
        raise ValueError(f"Unknown stat: {stat}")

    t0 = time.time()
    try:
        lf = state._lrtm_lf
        if parsed["if_cond"]:
            lf = _apply_lrtm_if(lf, parsed["if_cond"], pl, console)

        exprs = []
        for v in varlist:
            for s in stats:
                exprs.append(_stat_expr(s, v).alias(f"{v}__{s}"))

        if by_var:
            result = lf.group_by(by_var).agg(exprs).sort(by_var).collect()
        else:
            result = lf.select(exprs).collect()

        # Reshape into a compact stats x variables table (is a compact stats × variables layout)
        from rich.table import Table
        table = Table(show_header=True, header_style="bold cyan",
                      title=f"tabstat {', '.join(varlist)}" +
                            (f" (by {by_var})" if by_var else ""))
        if by_var:
            table.add_column(by_var, style="bold")
        table.add_column("stats", style="dim")
        for v in varlist:
            table.add_column(v, justify="right")

        def _fmt(val):
            if val is None:
                return "."
            if isinstance(val, float):
                return f"{val:.4f}"
            return str(val)

        if by_var:
            for row in result.iter_rows(named=True):
                gval = row[by_var]
                for s in stats:
                    cells = [str(gval), s] + [_fmt(row[f"{v}__{s}"]) for v in varlist]
                    table.add_row(*cells)
        else:
            row = result.row(0, named=True)
            for s in stats:
                cells = [s] + [_fmt(row[f"{v}__{s}"]) for v in varlist]
                table.add_row(*cells)

        console.print(table)
        elapsed = time.time() - t0
        console.print(f"[dim]({elapsed:.2f}s)[/dim]")
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")


def _lrtm_generate(rest, state, console):
    if not _require_lrtm(state, console):
        return
    pl = _check_polars()
    parsed = parse_command_line(rest)

    if not parsed["expression"]:
        console.print("[red]Syntax: lrtm generate newvar = expression [if cond][/red]")
        return

    newvar = parsed["varlist"][0]
    expr_str = parsed["expression"]

    snap = _lrtm_snapshot(state)
    t0 = time.time()
    try:
        expr = _sb_expr_to_polars(expr_str, pl)

        if parsed["if_cond"]:
            cond = _sb_to_polars_expr(parsed["if_cond"], pl)
            if cond is not None:
                expr = pl.when(cond).then(expr).otherwise(pl.lit(None))

        new_lf = state._lrtm_lf.with_columns(expr.alias(newvar))
        _lrtm_commit(state, new_lf, console, "generate")

        elapsed = time.time() - t0
        tag = " (conditional)" if parsed["if_cond"] else ""
        console.print(f"[green]Generated: {newvar}{tag} ({elapsed:.2f}s)[/green]")
    except Exception as e:
        _lrtm_restore(state, snap)
        console.print(f"[red]Error: {e}[/red]")
        console.print("[dim]Lazy frame left unchanged — no clear needed.[/dim]")


def _lrtm_filter(rest, state, console, mode="keep"):
    if not _require_lrtm(state, console):
        return
    pl = _check_polars()
    parsed = parse_command_line(rest)

    if not parsed["if_cond"]:
        console.print(f"[red]Syntax: lrtm {mode} if condition[/red]")
        return

    snap = _lrtm_snapshot(state)
    t0 = time.time()
    try:
        cond = _sb_to_polars_expr(parsed["if_cond"], pl)
        if cond is None:
            console.print("[red]Could not parse condition for Polars[/red]")
            return

        new_lf = state._lrtm_lf.filter(~cond if mode == "drop" else cond)
        _lrtm_commit(state, new_lf, console, "filter")
        elapsed = time.time() - t0
        console.print(f"[green]Filter applied ({elapsed:.2f}s). "
                      f"Use 'lrtm count' to check row count.[/green]")
    except Exception as e:
        _lrtm_restore(state, snap)
        console.print(f"[red]Error: {e}[/red]")
        console.print("[dim]Lazy frame left unchanged.[/dim]")


def _lrtm_sort(rest, state, console):
    if not _require_lrtm(state, console):
        return
    pl = _check_polars()
    parsed = parse_command_line(rest)

    if not parsed["varlist"]:
        console.print("[red]Syntax: lrtm sort varlist[/red]")
        return

    t0 = time.time()
    state._lrtm_lf = state._lrtm_lf.sort(parsed["varlist"])
    elapsed = time.time() - t0
    console.print(f"[green]Sorted by {', '.join(parsed['varlist'])} ({elapsed:.2f}s)[/green]")


def _lrtm_save(rest, state, console):
    if not _require_lrtm(state, console):
        return
    pl = _check_polars()
    parsed = parse_command_line(rest)

    filepath = parsed["varlist"][0] if parsed["varlist"] else None
    if not filepath:
        console.print('[red]Syntax: lrtm save "output.parquet" [if cond][/red]')
        return

    t0 = time.time()
    lf = state._lrtm_lf

    # Apply if condition
    lf = _apply_lrtm_if(lf, parsed["if_cond"], pl, console)

    ext = os.path.splitext(filepath)[1].lower()
    df = lf.collect()

    if ext in (".parquet", ".pq"):
        df.write_parquet(filepath, compression="zstd")
    elif ext == ".csv":
        df.write_csv(filepath)
    elif ext == ".json":
        df.write_json(filepath)
    elif ext == ".ipc":
        df.write_ipc(filepath)
    else:
        console.print("[red]Supported: .parquet, .csv, .json, .ipc[/red]")
        return

    elapsed = time.time() - t0
    console.print(f"[green]Saved: {filepath} ({len(df):,} rows, {elapsed:.2f}s)[/green]")


def _lrtm_head(rest, state, console):
    if not _require_lrtm(state, console):
        return
    pl = _check_polars()
    parsed = parse_command_line(rest)

    n = int(parsed["varlist"][0]) if parsed["varlist"] else 10
    lf = state._lrtm_lf

    # Apply if condition
    lf = _apply_lrtm_if(lf, parsed["if_cond"], pl, console)

    df = lf.head(n).collect()

    table = Table(show_lines=False)
    for col in df.columns:
        table.add_column(col, min_width=10)

    for row in df.iter_rows():
        table.add_row(*[str(v)[:40] for v in row])

    console.print(table)


def _lrtm_collect(rest, state, console):
    """Materialize LRTM data to pandas for use with standard Rln commands."""
    if not _require_lrtm(state, console):
        return

    t0 = time.time()
    console.print("[dim]Collecting LRTM data to memory...[/dim]")

    source = state._lrtm_source
    df = state._lrtm_lf.collect().to_pandas()
    name = os.path.basename(source) if source else "lrtm_data"
    state.set_data(df, name=name.split(".")[0], source=source)

    # Auto-clear lazy frame after collect (data is now in pandas)
    state._lrtm_lf = None
    state._lrtm_schema = None
    state._lrtm_source = None
    import gc
    gc.collect()

    elapsed = time.time() - t0
    mem = df.memory_usage(deep=True).sum()
    if mem > 1e9:
        mem_str = f"{mem/1e9:.1f} GB"
    elif mem > 1e6:
        mem_str = f"{mem/1e6:.1f} MB"
    else:
        mem_str = f"{mem/1e3:.1f} KB"

    console.print(f"[green]Collected: {len(df):,} rows x {len(df.columns)} cols ({mem_str}, {elapsed:.2f}s)[/green]")
    console.print("[dim]Standard Rln commands now available. LRTM auto-cleared.[/dim]")


def _lrtm_status(rest, state, console):
    if not _require_lrtm(state, console):
        return

    console.print(f"\n[bold]LRTM Status[/bold]")
    console.print(f"  Source: {state._lrtm_source}")
    console.print(f"  Columns: {len(state._lrtm_schema)}")
    if hasattr(state, '_lrtm_row_count') and state._lrtm_row_count is not None:
        console.print(f"  Rows: {state._lrtm_row_count:,}")
    else:
        console.print(f"  Rows: (lazy \u2014 use 'lrtm count' to compute)")
    console.print(f"  Mode: Lazy (Polars)")
    console.print()


# ──────────────────────────────────────────────
#  Expression translation helpers
# ──────────────────────────────────────────────

def _sb_to_polars_expr(cond_str, pl):
    """Convert other statistical tools condition to Polars expression. Handles & | and simple comparisons."""
    import re
    cond = cond_str.strip()

    # Handle AND: expr1 & expr2
    if " & " in cond:
        parts = cond.split(" & ")
        exprs = [_sb_to_polars_expr(p.strip(), pl) for p in parts]
        if all(e is not None for e in exprs):
            result = exprs[0]
            for e in exprs[1:]:
                result = result & e
            return result
        return None

    # Handle OR: expr1 | expr2
    if " | " in cond:
        parts = cond.split(" | ")
        exprs = [_sb_to_polars_expr(p.strip(), pl) for p in parts]
        if all(e is not None for e in exprs):
            result = exprs[0]
            for e in exprs[1:]:
                result = result | e
            return result
        return None

    # Handle NOT: !expr or ~expr
    if cond.startswith("!") or cond.startswith("~"):
        inner = _sb_to_polars_expr(cond[1:].strip(), pl)
        return ~inner if inner is not None else None

    # Handle missing(var)
    m = re.match(r'missing\((\w+)\)', cond)
    if m:
        return pl.col(m.group(1)).is_null()

    # Handle !missing(var)
    m = re.match(r'!missing\((\w+)\)', cond)
    if m:
        return pl.col(m.group(1)).is_not_null()

    # Simple comparisons: var op value
    m = re.match(r'(\w+)\s*(==|!=|>=|<=|>|<)\s*(.+)', cond)
    if m:
        var, op, val = m.group(1), m.group(2), m.group(3).strip().strip('"\'')
        try:
            val = float(val)
        except ValueError:
            pass  # Keep as string

        if op == "==":
            return pl.col(var) == val
        elif op == "!=":
            return pl.col(var) != val
        elif op == ">":
            return pl.col(var) > val
        elif op == "<":
            return pl.col(var) < val
        elif op == ">=":
            return pl.col(var) >= val
        elif op == "<=":
            return pl.col(var) <= val

    return None


def _sb_expr_to_polars(expr_str, pl):
    """Convert a compact expression to a Polars expression.

    Handles:
      * Numeric literals:           1, 3.14, -5, 1e6
      * String literals:            "hello", 'world'
      * Missing:                    . or "."
      * Bare column references:     x, some_var
      * Binary ops:                 a + b, x * 2, y / z, (a+b)*c
      * Function calls:             ln, log, log10, exp, sqrt, abs,
                                    floor, ceil, round, upper, lower,
                                    length, strlen, trim, missing
      * String concatenation:       name + " " + surname
      * Parentheses for grouping

    Raises ValueError on unparseable input — caller is responsible for
    rolling back any tentative mutations to the lazy frame.
    """
    import re

    expr_str = expr_str.strip()
    if not expr_str:
        raise ValueError("Empty expression")

    # ── Token types ───────────────────────────────────────────────
    # 0. Bare numeric literal (incl. negatives, decimals, scientific)
    if re.fullmatch(r'-?\d+(\.\d+)?([eE][+-]?\d+)?', expr_str):
        return pl.lit(float(expr_str) if ('.' in expr_str or 'e' in expr_str.lower())
                      else int(expr_str))

    # 1. String literal: "..." or '...'
    if len(expr_str) >= 2 and expr_str[0] == expr_str[-1] and expr_str[0] in ('"', "'"):
        # Make sure the closing quote is the last char and not escaped mid-string
        inner = expr_str[1:-1]
        if inner.count(expr_str[0]) == 0 or '\\' + expr_str[0] not in inner:
            return pl.lit(inner)

    # 2. missing value (.)
    if expr_str == ".":
        return pl.lit(None)

    # 3. Parenthesized: strip one layer of outer parens if balanced
    if expr_str.startswith("(") and expr_str.endswith(")"):
        depth = 0
        is_wrapped = True
        for i, ch in enumerate(expr_str):
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
            if depth == 0 and i < len(expr_str) - 1:
                is_wrapped = False
                break
        if is_wrapped:
            return _sb_expr_to_polars(expr_str[1:-1], pl)

    # 4. Binary operators — split at LOWEST precedence, right-to-left,
    #    respecting parentheses and string quotes.
    for ops in [("+", "-"), ("*", "/"), ("^", "**")]:
        split = _split_top_level_operator(expr_str, ops)
        if split is not None:
            left, op, right = split
            l_expr = _sb_expr_to_polars(left, pl)
            r_expr = _sb_expr_to_polars(right, pl)
            if op == "+":
                return l_expr + r_expr
            if op == "-":
                return l_expr - r_expr
            if op == "*":
                return l_expr * r_expr
            if op == "/":
                return l_expr / r_expr
            if op in ("^", "**"):
                return l_expr.pow(r_expr)

    # 5. Function call: fname(args)
    m = re.match(r'^(\w+)\s*\((.*)\)\s*$', expr_str)
    if m:
        fname = m.group(1).lower()
        inner = m.group(2).strip()
        arg_expr = _sb_expr_to_polars(inner, pl)
        if fname in ("ln", "log"):
            return arg_expr.log()
        if fname == "log10":
            return arg_expr.log10()
        if fname == "exp":
            return arg_expr.exp()
        if fname == "sqrt":
            return arg_expr.sqrt()
        if fname == "abs":
            return arg_expr.abs()
        if fname == "floor":
            return arg_expr.floor()
        if fname == "ceil":
            return arg_expr.ceil()
        if fname == "round":
            return arg_expr.round(0)
        if fname == "upper":
            return arg_expr.str.to_uppercase()
        if fname == "lower":
            return arg_expr.str.to_lowercase()
        if fname == "trim":
            return arg_expr.str.strip_chars()
        if fname in ("length", "strlen"):
            return arg_expr.str.len_chars()
        if fname == "missing":
            return arg_expr.is_null()
        raise ValueError(f"Unknown function: {fname}()")

    # 6. Bare identifier → column reference
    if re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]*', expr_str):
        return pl.col(expr_str)

    raise ValueError(f"Cannot parse expression: {expr_str!r}")


def _split_top_level_operator(expr_str, ops):
    """Find the RIGHTMOST top-level occurrence of any op in `ops`,
    respecting paren depth and string literals. Returns (left, op, right)
    or None if no top-level op is found.

    Rightmost-first gives correct left-associativity for non-commutative
    ops (a - b - c parses as ((a-b)-c)).
    """
    depth = 0
    in_str = None
    i = len(expr_str) - 1
    while i >= 0:
        ch = expr_str[i]
        # Track string literals (scan right-to-left means swap open/close)
        if in_str:
            if ch == in_str and (i == 0 or expr_str[i - 1] != "\\"):
                in_str = None
            i -= 1
            continue
        if ch in ('"', "'"):
            in_str = ch
            i -= 1
            continue
        if ch == ")":
            depth += 1
        elif ch == "(":
            depth -= 1
        elif depth == 0:
            # Check for 2-char ops first (**)
            if ch == "*" and i > 0 and expr_str[i - 1] == "*" and "**" in ops:
                # Don't split on unary minus (e.g. leading '-3')
                if i - 1 > 0:
                    return expr_str[:i - 1].strip(), "**", expr_str[i + 1:].strip()
            elif ch in ops:
                # Skip unary +/- at start or right after another operator
                if ch in ("+", "-") and (i == 0 or expr_str[i - 1] in "+-*/^(eE"):
                    i -= 1
                    continue
                return expr_str[:i].strip(), ch, expr_str[i + 1:].strip()
        i -= 1
    return None
