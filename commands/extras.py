"""
Extra commands: fillin, cross, sample, isid, levelsof, distinct,
display, clonevar, split
"""

import re
import numpy as np
import pandas as pd
from rich.console import Console

from commands.state import AppState
from commands.parse_helpers import parse_command_line
from commands.expression import eval_expression, eval_condition


# ──────────────────────────────────────────────
#  fillin
# ──────────────────────────────────────────────

def cmd_fillin(rest: str, state: AppState, console: Console):
    """
    fillin varlist
    Fill in all combinations of varlist, adding observations as needed.
    Creates _fillin variable (0=original, 1=added).
    """
    state.require_data()
    parsed = parse_command_line(rest)

    if not parsed["varlist"]:
        console.print("[red]Syntax: fillin varlist[/red]")
        return

    varlist = parsed["varlist"]
    for v in varlist:
        if v not in state.data.columns:
            console.print(f"[red]Variable '{v}' not found[/red]")
            return

    # Get all unique combinations
    unique_vals = [state.data[v].dropna().unique() for v in varlist]
    from itertools import product
    all_combos = pd.DataFrame(list(product(*unique_vals)), columns=varlist)

    # Mark originals
    state.data["_fillin"] = 0

    # Merge to find missing combos
    merged = all_combos.merge(state.data, on=varlist, how="left", indicator=True)
    new_rows = merged[merged["_merge"] == "left_only"].drop(columns=["_merge"])
    new_rows["_fillin"] = 1

    n_added = len(new_rows)
    state.data = pd.concat([state.data, new_rows], ignore_index=True)

    console.print(f"[green]({n_added:,} observations created)[/green]")
    state.mark_changed()


# ──────────────────────────────────────────────
#  cross
# ──────────────────────────────────────────────

def cmd_cross(rest: str, state: AppState, console: Console):
    """
    cross using "filename"
    Form all pairwise combinations of current and using dataset.
    """
    state.require_data()
    parsed = parse_command_line(rest)

    filepath = parsed["using"]
    if not filepath:
        console.print('[red]Syntax: cross using "filename"[/red]')
        return

    from rln_io.fileio import load_data
    console.print(f"[dim]Loading {filepath}...[/dim]")
    df_using, _ = load_data(filepath)

    n1 = len(state.data)
    n2 = len(df_using)

    # Cross join
    state.data["__cross_key__"] = 1
    df_using["__cross_key__"] = 1
    result = state.data.merge(df_using, on="__cross_key__", suffixes=("", "_using"))
    result = result.drop(columns=["__cross_key__"])

    state.data = result
    console.print(f"[green]Cross product: {n1:,} × {n2:,} = {len(result):,} observations[/green]")
    state.mark_changed()


# ──────────────────────────────────────────────
#  sample
# ──────────────────────────────────────────────

def cmd_sample(rest: str, state: AppState, console: Console):
    """
    sample N [, count]
    Keep a random sample.
    Without 'count': N is a percentage (0-100).
    With 'count': N is exact number of observations.
    """
    state.require_data()
    parsed = parse_command_line(rest)

    if not parsed["varlist"]:
        console.print("[red]Syntax: sample N [, count][/red]")
        return

    try:
        n_val = float(parsed["varlist"][0])
    except ValueError:
        console.print("[red]N must be a number[/red]")
        return

    n_before = len(state.data)

    if "count" in parsed["options"]:
        # Exact count
        n_keep = min(int(n_val), n_before)
        state.data = state.data.sample(n=n_keep).reset_index(drop=True)
    else:
        # Percentage
        if n_val < 0 or n_val > 100:
            console.print("[red]Percentage must be between 0 and 100[/red]")
            return
        frac = n_val / 100.0
        state.data = state.data.sample(frac=frac).reset_index(drop=True)

    n_dropped = n_before - len(state.data)
    console.print(f"[green]({n_dropped:,} observations dropped, {len(state.data):,} remaining)[/green]")
    state.mark_changed()


# ──────────────────────────────────────────────
#  isid
# ──────────────────────────────────────────────

def cmd_isid(rest: str, state: AppState, console: Console):
    """
    isid varlist [, sort]
    Check that varlist uniquely identifies observations.
    """
    state.require_data()
    parsed = parse_command_line(rest)

    if not parsed["varlist"]:
        console.print("[red]Syntax: isid varlist[/red]")
        return

    varlist = parsed["varlist"]
    for v in varlist:
        if v not in state.data.columns:
            console.print(f"[red]Variable '{v}' not found[/red]")
            return

    # Check for duplicates
    n_dupes = state.data.duplicated(subset=varlist).sum()

    if n_dupes == 0:
        console.print(f"[green]Variables uniquely identify observations[/green]")
        state.return_code = 0
        if "sort" in parsed["options"]:
            state.data = state.data.sort_values(varlist).reset_index(drop=True)
    else:
        state.return_code = 459
        raise ValueError(
            f"Variables do not uniquely identify observations. "
            f"{n_dupes:,} duplicate(s) found."
        )


# ──────────────────────────────────────────────
#  levelsof
# ──────────────────────────────────────────────

def cmd_levelsof(rest: str, state: AppState, console: Console):
    """
    levelsof varname [if condition] [, local(macroname) clean separate(sep)]
    List unique values of a variable.
    """
    state.require_data()
    parsed = parse_command_line(rest)

    if not parsed["varlist"]:
        console.print("[red]Syntax: levelsof varname[/red]")
        return

    varname = parsed["varlist"][0]
    if varname not in state.data.columns:
        console.print(f"[red]Variable '{varname}' not found[/red]")
        return

    # Apply if condition
    data = state.data
    if parsed["if_cond"]:
        mask = eval_condition(parsed["if_cond"], data)
        data = data.loc[mask]

    values = sorted(data[varname].dropna().unique())

    sep = parsed["options"].get("separate", " ")
    clean = "clean" in parsed["options"]

    if pd.api.types.is_numeric_dtype(state.data[varname]):
        if clean:
            formatted = [str(v) for v in values]
        else:
            formatted = [str(v) for v in values]
    else:
        if clean:
            formatted = [str(v) for v in values]
        else:
            formatted = [f'"{v}"' for v in values]

    result = sep.join(formatted)
    console.print(result)

    n = len(values)
    state.write_log(f"levelsof {varname}: {n} levels")


# ──────────────────────────────────────────────
#  distinct
# ──────────────────────────────────────────────

def cmd_distinct(rest: str, state: AppState, console: Console):
    """
    distinct [varlist] [if condition]
    Report number of distinct observations and values.
    """
    state.require_data()
    parsed = parse_command_line(rest)

    data = state.data
    if parsed["if_cond"]:
        mask = eval_condition(parsed["if_cond"], data)
        data = data.loc[mask]

    if parsed["varlist"]:
        varlist = parsed["varlist"]
        for v in varlist:
            if v not in data.columns:
                console.print(f"[red]Variable '{v}' not found[/red]")
                return
    else:
        varlist = list(data.columns)

    console.print(f"\n  Observations: {len(data):,}")

    if len(varlist) == 1:
        v = varlist[0]
        n_distinct = data[v].nunique()
        n_missing = data[v].isna().sum()
        console.print(f"  Distinct values of {v}: {n_distinct:,}")
        if n_missing > 0:
            console.print(f"  Missing: {n_missing:,}")
    else:
        # Distinct combinations
        n_distinct = data[varlist].drop_duplicates().shape[0]
        console.print(f"  Distinct combinations of ({', '.join(varlist)}): {n_distinct:,}")

    console.print()


# ──────────────────────────────────────────────
#  display
# ──────────────────────────────────────────────

def cmd_display(rest: str, state: AppState, console: Console):
    """
    display expression [expression ...]
    display "string" [expression] ["string" ...]

    Evaluate and display one or more tokens separated by whitespace.
    Each token can be:
      - a quoted string literal:          "Hello, world"
      - a math expression:                 2 + 3 * sqrt(16)
      - an r() or e() result reference:    r(mean), e(r2)
      - the _rc return-code macro:          _rc
      - a variable expression (first-obs):  income

    Tokens are concatenated with no added separator. Empty string between
    a number and text is fine — use explicit spaces inside quotes:
        display "Mean = " r(mean)
        display "rc=" _rc
    """
    rest = rest.strip()
    if not rest:
        console.print("[red]Syntax: display expression [expression ...][/red]")
        return

    # Split into tokens: whitespace-separated but respect quoted strings
    # and don't split inside function calls like r(mean) or inlist(a,1,2).
    tokens = _split_display_tokens(rest)

    parts = []
    for tok in tokens:
        # String literal
        if len(tok) >= 2 and tok[0] == tok[-1] and tok[0] in ('"', "'"):
            parts.append(tok[1:-1])
            continue
        # _rc — the last return code
        if tok == "_rc":
            rc = getattr(state, "_captured_rc", None)
            if rc is None:
                rc = getattr(state, "_rc", getattr(state, "return_code", 0))
            parts.append(str(rc))
            continue
        # r(key) or e(key)
        m_re = _RE_R_ERESULT.match(tok)
        if m_re:
            kind, key = m_re.group(1), m_re.group(2)
            bag = state.r_results if kind == "r" else state.e_results
            val = bag.get(key, ".") if bag else "."
            parts.append(_format_display_value(val))
            continue
        # Numeric/math expression first
        try:
            import math
            safe_ns = {
                "abs": abs, "ceil": math.ceil, "floor": math.floor,
                "round": round, "sqrt": math.sqrt, "log": math.log,
                "log10": math.log10, "exp": math.exp, "sin": math.sin,
                "cos": math.cos, "tan": math.tan, "pi": math.pi, "_pi": math.pi,
                "ln": math.log, "mod": lambda a, b: a % b,
                "int": int, "float": float, "max": max, "min": min,
            }
            val = eval(tok.replace("^", "**"), {"__builtins__": {}}, safe_ns)
            parts.append(_format_display_value(val))
            continue
        except Exception:
            pass
        # DataFrame expression → first element
        if state.has_data():
            try:
                val = eval_expression(tok, state.data)
                if hasattr(val, "__len__") and len(val):
                    parts.append(_format_display_value(val.iloc[0]))
                    continue
            except Exception:
                pass
        # Fallback: just echo the token (the display does this for
        # bare identifiers it can't resolve)
        parts.append(tok)

    console.print("  " + "".join(parts))


import re as _re
_RE_R_ERESULT = _re.compile(r"^([re])\(\s*([A-Za-z_][A-Za-z0-9_]*)\s*\)$")


def _split_display_tokens(s: str) -> list:
    """Split into alternating (string-literal) and (expression) chunks.

    the `display` treats the whole line as one big expression UNTIL a
    quoted string literal breaks it up. So:
        display 2 + 3                       →  one chunk: "2 + 3"
        display "Mean = " r(mean)           →  two chunks: '"Mean = "', 'r(mean)'
        display "Total: " total " items"    →  three chunks
    """
    out, cur, in_q = [], [], None
    i = 0
    while i < len(s):
        ch = s[i]
        if in_q:
            cur.append(ch)
            if ch == in_q:
                out.append("".join(cur))
                cur = []
                in_q = None
            i += 1
            continue
        if ch in ('"', "'"):
            # Flush any accumulated expression chunk
            buf = "".join(cur).strip()
            if buf:
                out.append(buf)
            cur = [ch]
            in_q = ch
            i += 1
            continue
        cur.append(ch)
        i += 1
    buf = "".join(cur).strip()
    if buf:
        out.append(buf)
    return out


def _format_display_value(val) -> str:
    """Pretty-print a scalar for display. Mirrors other statistical tools formatting."""
    if val is None:
        return "."
    if isinstance(val, bool):
        return "1" if val else "0"
    if isinstance(val, float):
        try:
            if val != val:  # NaN
                return "."
            if val == int(val) and abs(val) < 1e15:
                return str(int(val))
            return f"{val:.10g}"
        except Exception:
            return str(val)
    return str(val)


# ──────────────────────────────────────────────
#  clonevar
# ──────────────────────────────────────────────

def cmd_clonevar(rest: str, state: AppState, console: Console):
    """
    clonevar newvar = existingvar
    Create an exact copy of a variable including labels.
    """
    state.require_data()
    parsed = parse_command_line(rest)

    if not parsed["varlist"] or not parsed["expression"]:
        console.print("[red]Syntax: clonevar newvar = existingvar[/red]")
        return

    newvar = parsed["varlist"][0]
    srcvar = parsed["expression"].strip()

    if srcvar not in state.data.columns:
        console.print(f"[red]Variable '{srcvar}' not found[/red]")
        return

    if newvar in state.data.columns:
        console.print(f"[red]Variable '{newvar}' already exists[/red]")
        return

    state.data[newvar] = state.data[srcvar].copy()

    # Copy labels
    if srcvar in state.variable_labels:
        state.variable_labels[newvar] = state.variable_labels[srcvar]
    if srcvar in state.value_label_assignments:
        state.value_label_assignments[newvar] = state.value_label_assignments[srcvar]

    console.print(f"[green]Cloned: {srcvar} → {newvar}[/green]")
    state.mark_changed()


# ──────────────────────────────────────────────
#  split
# ──────────────────────────────────────────────

def cmd_split(rest: str, state: AppState, console: Console):
    """
    split varname [, parse(sep) generate(stub) limit(N)]
    Split a string variable into multiple variables.
    Default separator: space. Default stub: varname1, varname2, ...
    """
    state.require_data()
    parsed = parse_command_line(rest)

    if not parsed["varlist"]:
        console.print('[red]Syntax: split varname [, parse("sep") generate(stub)][/red]')
        return

    varname = parsed["varlist"][0]
    if varname not in state.data.columns:
        console.print(f"[red]Variable '{varname}' not found[/red]")
        return

    sep = parsed["options"].get("parse", parsed["options"].get("p", " "))
    stub = parsed["options"].get("generate", parsed["options"].get("gen", varname))
    limit = parsed["options"].get("limit")
    if limit:
        limit = int(limit)

    # Split
    split_df = state.data[varname].astype(str).str.split(sep, expand=True, n=limit)

    # Rename columns
    new_names = {}
    for i, col in enumerate(split_df.columns, 1):
        new_name = f"{stub}{i}"
        if new_name in state.data.columns:
            console.print(f"[yellow]Variable '{new_name}' already exists, skipping[/yellow]")
            continue
        new_names[col] = new_name

    split_df = split_df.rename(columns=new_names)

    # Replace "None" with actual NaN
    split_df = split_df.replace({"None": np.nan, "nan": np.nan})

    # Add to dataset
    for col in split_df.columns:
        state.data[col] = split_df[col]

    console.print(f"[green]Split {varname} into {len(new_names)} variables: "
                  f"{', '.join(split_df.columns)}[/green]")
    state.mark_changed()


# ──────────────────────────────────────────────
#  compress
# ──────────────────────────────────────────────

def cmd_compress(rest: str, state: AppState, console: Console):
    """
    compress [varlist]
    Reduce storage by downcasting numeric types.
    """
    state.require_data()
    parsed = parse_command_line(rest)

    if parsed["varlist"]:
        from commands.explore import _resolve_varlist
        cols = _resolve_varlist(parsed["varlist"], state.data)
    else:
        cols = list(state.data.columns)

    mem_before = state.data[cols].memory_usage(deep=True).sum()
    n_changed = 0

    for col in cols:
        if pd.api.types.is_integer_dtype(state.data[col]):
            original_dtype = state.data[col].dtype
            state.data[col] = pd.to_numeric(state.data[col], downcast="integer")
            if state.data[col].dtype != original_dtype:
                n_changed += 1
        elif pd.api.types.is_float_dtype(state.data[col]):
            original_dtype = state.data[col].dtype
            state.data[col] = pd.to_numeric(state.data[col], downcast="float")
            if state.data[col].dtype != original_dtype:
                n_changed += 1
        elif pd.api.types.is_object_dtype(state.data[col]):
            # Check if can be category
            n_unique = state.data[col].nunique()
            n_total = len(state.data[col].dropna())
            if n_total > 0 and n_unique / n_total < 0.5:
                state.data[col] = state.data[col].astype("category")
                n_changed += 1

    mem_after = state.data[cols].memory_usage(deep=True).sum()
    saved = mem_before - mem_after

    if saved > 1e6:
        saved_str = f"{saved/1e6:.1f} MB"
    elif saved > 1e3:
        saved_str = f"{saved/1e3:.1f} KB"
    else:
        saved_str = f"{saved} bytes"

    console.print(f"[green]Compressed: {n_changed} variable(s) changed, {saved_str} saved[/green]")
    if n_changed > 0:
        state.mark_changed()
