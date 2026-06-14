"""
Advanced commands: assert, capture, preserve, restore, collapse, egen, notes
"""

import re
import numpy as np
import pandas as pd
from rich.console import Console

from commands.state import AppState
from commands.parse_helpers import parse_command_line
from commands.expression import eval_condition, eval_expression


# ──────────────────────────────────────────────
#  assert
# ──────────────────────────────────────────────

def cmd_assert(rest: str, state: AppState, console: Console):
    """
    assert condition [if cond] [in range]

    Verify that a condition holds for every observation (or, with an
    `if`/`in` clause, for every observation in the specified subset).
    Raises an error (rc=9) if any row contradicts the assertion.

    The expression engine also exposes `_rc`, so after `capture`:
        assert _rc == 0
    will check the captured return code.

    Bug 11 (Gemini): previously an entire `assert X if Y` string was passed
    straight to the expression engine, which treated `if` as a Python
    keyword and failed. Now we parse the command structure first.
    """
    state.require_data()

    rest = rest.strip()
    if not rest:
        console.print("[red]Syntax: assert condition [if cond] [in range][/red]")
        state.return_code = 198
        return

    # Use parse_command_line so `if`/`in` are extracted and the actual
    # condition is just the assertion body. We build a pseudo-command line
    # where the assertion body becomes the "expression" part: prefix with
    # a dummy `= ` so the parser treats it as a post-`=` expression and
    # doesn't try to split it into varlist tokens.
    parsed = parse_command_line(rest)

    # parse_command_line may push the assertion condition into varlist
    # tokens or expression depending on shape. Safer: recover the raw
    # condition by stripping off the if/in tails manually.
    cond_text = rest
    # Strip [in range]
    import re as _re
    m_in = _re.search(r'\s+in\s+([^,]+?)(?=\s*(?:,|$))', cond_text)
    if m_in:
        cond_text = cond_text[:m_in.start()]
    # Strip [if cond]  — first top-level `if` whose preceding char is a space
    m_if = _re.search(r'(?<=\s)if\s+', cond_text)
    if m_if:
        cond_text = cond_text[:m_if.start()].rstrip()
    cond_text = cond_text.strip()

    try:
        # Apply the if/in subset first, then evaluate the assertion on it.
        df = state.data
        subset = df
        if parsed.get("if_cond"):
            subset = subset.loc[eval_condition(parsed["if_cond"], subset,
                                                state=state)]
        if parsed.get("in_range"):
            sl = parse_in_range(parsed["in_range"])
            if sl:
                subset = subset.iloc[sl]

        if len(subset) == 0:
            console.print("[yellow]Assertion skipped: no observations in range[/yellow]")
            state.return_code = 0
            return

        mask = eval_condition(cond_text, subset, state=state)
        n_fail = int((~mask).sum())

        if n_fail > 0:
            state.return_code = 9
            raise AssertionError(
                f"Assertion failed: {n_fail:,} contradiction(s) in "
                f"{len(subset):,} observations"
            )
        state.return_code = 0
        scope = f"{len(subset):,}"
        if len(subset) != len(df):
            scope += f" of {len(df):,}"
        console.print(f"[green]Assertion verified: {scope} observations[/green]")

    except AssertionError:
        raise
    except Exception as e:
        state.return_code = 198
        raise ValueError(f"Invalid assertion expression: {e}")


# ──────────────────────────────────────────────
#  capture
# ──────────────────────────────────────────────

def cmd_capture(rest: str, state: AppState, console: Console):
    """
    capture command
    capture noisily command
    Execute a command, suppressing errors. Sets _rc.
    With 'noisily', shows output but still captures errors.
    """
    rest = rest.strip()
    if not rest:
        console.print("[red]Syntax: capture [noisily] command[/red]")
        return

    noisily = False
    if rest.lower().startswith("noisily "):
        noisily = True
        rest = rest[8:].strip()

    # We need to get the parser to execute the inner command
    # Import here to avoid circular imports
    from commands.parser import CommandParser

    # Create a temporary console that captures output
    if noisily:
        inner_console = console
    else:
        from io import StringIO
        captured_output = StringIO()
        inner_console = Console(file=captured_output, force_terminal=False)

    # Build a temporary parser
    temp_parser = CommandParser(state, inner_console)

    try:
        temp_parser.execute(rest, reraise=True)
        inner_rc = 0
    except Exception as e:
        # Prefer the rc the inner parser already set on state (it records
        # distinct codes for "unknown command" vs generic failures).
        inner_rc = getattr(state, "_rc", 1) or 1
        if noisily:
            console.print(f"[yellow]Captured error (rc={inner_rc}): {e}[/yellow]")

    # Persist the captured rc through state._captured_rc. The outer parser
    # will overwrite state._rc to 0 when `capture` itself returns cleanly
    # (which is the whole point of `capture`), so users reference _rc via
    # `display _rc` which reads this captured value.
    state._rc = inner_rc
    state.return_code = inner_rc
    state._captured_rc = inner_rc

    console.print(f"[dim]_rc = {inner_rc}[/dim]")


# ──────────────────────────────────────────────
#  preserve / restore
# ──────────────────────────────────────────────

def cmd_preserve(rest: str, state: AppState, console: Console):
    """
    preserve
    Save a snapshot of the current dataset. Use 'restore' to revert.
    """
    state.require_data()
    state.preserve()
    nobs, nvars = state.data.shape
    console.print(f"[green]Data preserved ({nobs:,} obs, {nvars} vars)[/green]")


def cmd_restore(rest: str, state: AppState, console: Console):
    """
    restore
    Restore the most recently preserved dataset.
    """
    if not state.has_snapshot():
        console.print("[red]No preserved data to restore. Use 'preserve' first.[/red]")
        return

    state.restore()
    nobs, nvars = state.data.shape
    console.print(f"[green]Data restored ({nobs:,} obs, {nvars} vars)[/green]")


# ──────────────────────────────────────────────
#  collapse
# ──────────────────────────────────────────────

def cmd_collapse(rest: str, state: AppState, console: Console):
    """
    collapse (stat) varlist [, by(groupvars)]
    Collapse dataset to summary statistics.

    Statistics: mean, median, sum, count, min, max, sd, first, last,
               p25, p50, p75, p1, p5, p10, p90, p95, p99

    Examples:
      collapse (mean) income age
      collapse (mean) avg_inc=income (sum) total_pop=population, by(country year)
      collapse (median) income (sd) sd_income=income, by(region)
    """
    state.require_data()

    # Parse: collapse (stat) var [newname=var] ... [, by(groupvars)]
    parsed = parse_command_line(rest)
    by_vars = parsed["options"].get("by", "")
    by_vars = [v.strip() for v in by_vars.split() if v.strip()] if by_vars else []
    # Honor `bysort group: collapse ...` prefix context
    if getattr(state, "_by_vars", None):
        by_vars = list(state._by_vars) + [v for v in by_vars if v not in state._by_vars]

    # Validate by_vars
    for v in by_vars:
        if v not in state.data.columns:
            console.print(f"[red]Variable '{v}' not found[/red]")
            return

    # Parse stat-var specifications from raw text
    raw = rest.strip()
    # Strip weight clause [fweight=var] etc. so it doesn't pollute spec parser
    import re as _re
    raw = _re.sub(
        r'\[\s*(fweight|aweight|pweight|iweight|weight)\s*=\s*[A-Za-z_][A-Za-z0-9_]*\s*\]',
        '', raw, flags=_re.IGNORECASE).strip()
    # Remove options after comma
    comma_pos = _find_top_comma(raw)
    if comma_pos is not None:
        raw = raw[:comma_pos].strip()

    specs = _parse_collapse_specs(raw)
    if not specs:
        console.print("[red]Syntax: collapse (stat) varlist [, by(groupvars)][/red]")
        console.print("[dim]Example: collapse (mean) income (sum) population, by(country)[/dim]")
        return

    # Validate source variables
    for new_name, src_var, stat in specs:
        if src_var not in state.data.columns:
            console.print(f"[red]Variable '{src_var}' not found[/red]")
            return

    # Build aggregation dict
    agg_dict = {}
    rename_map = {}

    for new_name, src_var, stat in specs:
        pandas_func = _stat_to_pandas(stat)
        if pandas_func is None:
            console.print(f"[red]Unknown statistic: {stat}[/red]")
            return

        # Handle multiple stats on same variable
        col_key = src_var
        if col_key in agg_dict:
            # Need named agg
            col_key = f"{src_var}__{stat}"
            state.data[col_key] = state.data[src_var]

        agg_dict[col_key] = pandas_func
        if new_name != src_var or col_key != src_var:
            rename_map[col_key] = new_name

    nobs_before = len(state.data)

    # Weight support — when a weight is given, use a custom weighted aggregator
    # for mean/sd/sum/median/percentiles; count becomes sum(w).
    from commands.weights import (get_weight_series, weighted_mean,
                                   weighted_std, weighted_quantile,
                                   weighted_sum, weighted_count)
    weights = get_weight_series(parsed, state.data, console)
    if weights is False:
        return
    wtype = parsed["weight"]["type"] if parsed["weight"] else None

    if weights is not None:
        # Build the weighted result ourselves — groupby + apply
        def weighted_agg(sub):
            out = {}
            w = weights.loc[sub.index].values
            for new_name, src_var, stat in specs:
                xv = sub[src_var].values
                if stat in ("mean",):
                    out[new_name] = weighted_mean(xv, w)
                elif stat in ("sum", "total"):
                    out[new_name] = weighted_sum(xv, w)
                elif stat in ("count", "n"):
                    out[new_name] = weighted_count(xv, w)
                elif stat == "sd":
                    out[new_name] = weighted_std(xv, w, wtype or "aweight")
                elif stat == "median":
                    out[new_name] = weighted_quantile(xv, w, 0.5)
                elif stat.startswith("p") and stat[1:].isdigit():
                    out[new_name] = weighted_quantile(xv, w, int(stat[1:]) / 100.0)
                elif stat == "min":
                    mask = (~pd.isna(sub[src_var])) & (weights.loc[sub.index] > 0)
                    out[new_name] = sub.loc[mask, src_var].min() if mask.any() else float("nan")
                elif stat == "max":
                    mask = (~pd.isna(sub[src_var])) & (weights.loc[sub.index] > 0)
                    out[new_name] = sub.loc[mask, src_var].max() if mask.any() else float("nan")
                elif stat == "first":
                    out[new_name] = sub[src_var].iloc[0] if len(sub) else float("nan")
                elif stat == "last":
                    out[new_name] = sub[src_var].iloc[-1] if len(sub) else float("nan")
                else:
                    out[new_name] = float("nan")
            return pd.Series(out)

        if by_vars:
            result = state.data.groupby(by_vars).apply(weighted_agg).reset_index()
        else:
            result = pd.DataFrame([weighted_agg(state.data)])
        # rename_map is only needed for the vanilla path
        rename_map = {}
    elif by_vars:
        result = state.data.groupby(by_vars, as_index=False).agg(agg_dict)
    else:
        result_data = {}
        for col, func in agg_dict.items():
            if callable(func):
                result_data[col] = [func(state.data[col])]
            else:
                result_data[col] = [state.data[col].agg(func)]
        result = pd.DataFrame(result_data)

    # Rename columns
    if rename_map:
        result = result.rename(columns=rename_map)

    # Clean up temp columns
    temp_cols = [c for c in result.columns if "__" in c]
    result = result.drop(columns=temp_cols, errors="ignore")

    nobs_after = len(result)
    state.data = result
    state.mark_changed()

    # Clear labels that no longer apply
    keep_cols = set(result.columns)
    state.variable_labels = {k: v for k, v in state.variable_labels.items() if k in keep_cols}

    console.print(f"[green]Collapsed: {nobs_before:,} obs → {nobs_after:,} obs "
                  f"({len(result.columns)} vars)[/green]")


def _parse_collapse_specs(raw: str) -> list:
    """
    Parse collapse specifications like:
      (mean) income age (sum) total=population (sd) sd_inc=income
    Returns list of (new_name, source_var, stat)
    """
    specs = []
    current_stat = None

    # Tokenize: find (stat) blocks and variable assignments
    i = 0
    while i < len(raw):
        # Skip whitespace
        while i < len(raw) and raw[i] == " ":
            i += 1
        if i >= len(raw):
            break

        # Check for (stat)
        if raw[i] == "(":
            end = raw.find(")", i)
            if end < 0:
                break
            current_stat = raw[i+1:end].strip().lower()
            i = end + 1
            continue

        if current_stat is None:
            i += 1
            continue

        # Read next token (could be newname=srcvar or just varname)
        token_start = i
        while i < len(raw) and raw[i] not in (" ", "("):
            i += 1
        token = raw[token_start:i].strip()

        if not token:
            continue

        if "=" in token:
            new_name, src_var = token.split("=", 1)
            specs.append((new_name.strip(), src_var.strip(), current_stat))
        else:
            specs.append((token, token, current_stat))

    return specs


def _stat_to_pandas(stat: str):
    """Convert stat name to pandas aggregation function."""
    mapping = {
        "mean": "mean",
        "median": "median",
        "sum": "sum",
        "count": "count",
        "n": "count",
        "min": "min",
        "max": "max",
        "sd": "std",
        "first": "first",
        "last": "last",
        "p1": lambda x: x.quantile(0.01),
        "p5": lambda x: x.quantile(0.05),
        "p10": lambda x: x.quantile(0.10),
        "p25": lambda x: x.quantile(0.25),
        "p50": lambda x: x.quantile(0.50),
        "p75": lambda x: x.quantile(0.75),
        "p90": lambda x: x.quantile(0.90),
        "p95": lambda x: x.quantile(0.95),
        "p99": lambda x: x.quantile(0.99),
        "iqr": lambda x: x.quantile(0.75) - x.quantile(0.25),
    }
    return mapping.get(stat)


def _find_top_comma(text: str):
    """Find top-level comma not inside parens."""
    depth = 0
    for i, ch in enumerate(text):
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        elif ch == "," and depth == 0:
            return i
    return None


# ──────────────────────────────────────────────
#  egen
# ──────────────────────────────────────────────

def cmd_egen(rest: str, state: AppState, console: Console):
    """
    egen newvar = function(args) [if condition] [, by(groupvars)]
    Extended generate: group-level and cross-observation functions.

    Functions:
      mean(var)        — mean (optionally by group)
      median(var)      — median
      sum(var)         — sum
      count(var)       — count non-missing
      min(var)         — minimum
      max(var)         — maximum
      sd(var)          — standard deviation
      total(var)       — alias for sum
      rowtotal(varlist) — row-wise sum across variables
      rowmean(varlist)  — row-wise mean across variables
      rowmin(varlist)   — row-wise minimum
      rowmax(varlist)   — row-wise maximum
      rowmiss(varlist)  — count of missing values per row
      group(varlist)    — unique group ID
      rank(var)         — rank
      tag(varlist)      — tag first occurrence in each group
      seq()             — sequential counter (optionally by group)
      concat(varlist)   — concatenate string variables
      std(var)          — standardize (mean=0, sd=1)
    """
    state.require_data()

    # Parse: egen newvar = func(args) [if cond] [, by(groupvars)]
    parsed = parse_command_line(rest)

    if not parsed["varlist"] or not parsed["expression"]:
        console.print("[red]Syntax: egen newvar = function(args) [if cond] [, by(groupvars)][/red]")
        return

    newvar = parsed["varlist"][0]
    expr = parsed["expression"].strip()

    if newvar in state.data.columns:
        console.print(f"[red]Variable '{newvar}' already exists[/red]")
        return

    # Parse by() option.
    # BUGFIX (Gemini #1): ALSO honor state._by_vars if the command is running
    # under a `by varlist:` or `bysort varlist:` prefix. Previously `bysort
    # city: egen m = mean(income)` silently returned the grand mean because
    # egen only looked at its own by() option.
    by_vars = parsed["options"].get("by", "")
    by_vars = [v.strip() for v in by_vars.split() if v.strip()] if by_vars else []
    if getattr(state, "_by_vars", None):
        # Prepend prefix-set by_vars; keep explicit by() afterwards to let a
        # user nest like  `bysort country: egen ... , by(year)`
        by_vars = list(state._by_vars) + [v for v in by_vars if v not in state._by_vars]

    for v in by_vars:
        if v not in state.data.columns:
            console.print(f"[red]Variable '{v}' not found[/red]")
            return

    # Parse function call: func(args)
    m = re.match(r'(\w+)\s*\(\s*(.*?)\s*\)', expr)
    if not m:
        console.print(f"[red]Cannot parse egen function: {expr}[/red]")
        return

    func_name = m.group(1).lower()
    func_args = m.group(2).strip()

    # Dispatch to egen function
    result = _egen_dispatch(func_name, func_args, state.data, by_vars, state, console)
    if result is None:
        return

    # Apply if-condition
    if parsed["if_cond"]:
        mask = eval_condition(parsed["if_cond"], state.data)
        full_result = pd.Series(np.nan, index=state.data.index)
        full_result.loc[mask] = result.loc[mask]
        state.data[newvar] = full_result
    else:
        state.data[newvar] = result

    n_valid = state.data[newvar].notna().sum()
    console.print(f"[green]Generated: {newvar} ({n_valid:,} non-missing values)[/green]")
    state.mark_changed()


def _egen_dispatch(func, args, df, by_vars, state, console):
    """Dispatch egen function calls."""

    # --- Aggregate functions (work with by()) ---
    agg_funcs = {
        "mean": "mean", "median": "median", "sum": "sum",
        "total": "sum", "count": "count", "min": "min",
        "max": "max", "sd": "std",
    }

    if func in agg_funcs:
        if args not in df.columns:
            console.print(f"[red]Variable '{args}' not found[/red]")
            return None
        pandas_func = agg_funcs[func]
        if by_vars:
            return df.groupby(by_vars)[args].transform(pandas_func)
        else:
            val = df[args].agg(pandas_func)
            return pd.Series(val, index=df.index)

    # --- Row-wise functions ---
    if func in ("rowtotal", "rowsum"):
        vars_list = [v.strip() for v in args.split()]
        for v in vars_list:
            if v not in df.columns:
                console.print(f"[red]Variable '{v}' not found[/red]")
                return None
        return df[vars_list].sum(axis=1)

    if func == "rowmean":
        vars_list = [v.strip() for v in args.split()]
        for v in vars_list:
            if v not in df.columns:
                console.print(f"[red]Variable '{v}' not found[/red]")
                return None
        return df[vars_list].mean(axis=1)

    if func == "rowmin":
        vars_list = [v.strip() for v in args.split()]
        return df[vars_list].min(axis=1)

    if func == "rowmax":
        vars_list = [v.strip() for v in args.split()]
        return df[vars_list].max(axis=1)

    if func == "rowmiss":
        vars_list = [v.strip() for v in args.split()]
        return df[vars_list].isna().sum(axis=1)

    if func == "rownonmiss":
        vars_list = [v.strip() for v in args.split()]
        return df[vars_list].notna().sum(axis=1)

    # --- group() ---
    if func == "group":
        vars_list = [v.strip() for v in args.split()]
        for v in vars_list:
            if v not in df.columns:
                console.print(f"[red]Variable '{v}' not found[/red]")
                return None
        # Create group IDs
        groups = df.groupby(vars_list).ngroup() + 1
        return groups

    # --- rank() ---
    if func == "rank":
        if args not in df.columns:
            console.print(f"[red]Variable '{args}' not found[/red]")
            return None
        if by_vars:
            return df.groupby(by_vars)[args].rank(method="average")
        else:
            return df[args].rank(method="average")

    # --- tag() ---
    if func == "tag":
        vars_list = [v.strip() for v in args.split()]
        for v in vars_list:
            if v not in df.columns:
                console.print(f"[red]Variable '{v}' not found[/red]")
                return None
        return (~df.duplicated(subset=vars_list, keep="first")).astype(int)

    # --- seq() ---
    if func == "seq":
        if by_vars:
            return df.groupby(by_vars).cumcount() + 1
        else:
            return pd.Series(range(1, len(df) + 1), index=df.index)

    # --- concat() ---
    if func == "concat":
        vars_list = [v.strip() for v in args.split()]
        for v in vars_list:
            if v not in df.columns:
                console.print(f"[red]Variable '{v}' not found[/red]")
                return None
        return df[vars_list].astype(str).agg(" ".join, axis=1)

    # --- std() / standardize ---
    if func in ("std", "standardize"):
        if args not in df.columns:
            console.print(f"[red]Variable '{args}' not found[/red]")
            return None
        if by_vars:
            grouped = df.groupby(by_vars)[args]
            return (df[args] - grouped.transform("mean")) / grouped.transform("std")
        else:
            return (df[args] - df[args].mean()) / df[args].std()

    console.print(f"[red]Unknown egen function: {func}[/red]")
    console.print("[dim]Available: mean, median, sum, count, min, max, sd, total, "
                  "rowtotal, rowmean, rowmin, rowmax, rowmiss, group, rank, tag, "
                  "seq, concat, std[/dim]")
    return None


# ──────────────────────────────────────────────
#  notes
# ──────────────────────────────────────────────

def cmd_notes(rest: str, state: AppState, console: Console):
    """
    notes                  — Display all notes
    notes : "text"         — Add a note
    notes drop N           — Drop note number N
    notes drop _all        — Drop all notes
    """
    rest = rest.strip()

    if not rest:
        # Display notes
        if not state.notes:
            console.print("[dim]No notes.[/dim]")
            return
        for i, note in enumerate(state.notes, 1):
            console.print(f"  [cyan]{i}.[/cyan] {note}")
        return

    # Add note: notes : "text"
    if rest.startswith(":"):
        text = rest[1:].strip().strip("\"'")
        if text:
            state.notes.append(text)
            console.print(f"[green]Note {len(state.notes)} added[/green]")
        else:
            console.print("[red]Syntax: notes : \"text\"[/red]")
        return

    # Drop notes
    if rest.lower().startswith("drop"):
        target = rest[4:].strip()
        if target == "_all":
            n = len(state.notes)
            state.notes.clear()
            console.print(f"[green]{n} note(s) dropped[/green]")
        else:
            try:
                idx = int(target) - 1
                if 0 <= idx < len(state.notes):
                    removed = state.notes.pop(idx)
                    console.print(f"[green]Note {idx+1} dropped[/green]")
                else:
                    console.print(f"[red]Note {idx+1} not found[/red]")
            except ValueError:
                console.print("[red]Syntax: notes drop N | notes drop _all[/red]")
        return

    console.print("[red]Syntax: notes | notes : \"text\" | notes drop N[/red]")
