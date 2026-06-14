"""
Quantile-family commands:
    pctile     create a variable holding percentile points of another
    xtile      assign each row to a quantile bin (quartiles, deciles, ...)
    centile    report one or more percentiles with confidence intervals
    winsor2    trim outliers at given percentiles (replace or generate)

All four accept an if-clause, in-range, and a weight ([fweight/aweight/
pweight/iweight=var]) option. Under weights they use the same weighted
quantile routine as tabstat/summarize.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from rich.console import Console
from rich.table import Table

from commands.state import AppState
from commands.parse_helpers import parse_command_line
from commands.expression import eval_condition, parse_in_range
from commands.weights import get_weight_series, weighted_quantile


def _apply_if_in(df, parsed):
    if parsed["if_cond"]:
        df = df.loc[eval_condition(parsed["if_cond"], df)]
    if parsed["in_range"]:
        sl = parse_in_range(parsed["in_range"])
        if sl:
            df = df.iloc[sl]
    return df


# ──────────────────────────────────────────────────────────────
# pctile — create a new variable with percentile CUT POINTS
# ──────────────────────────────────────────────────────────────

def cmd_pctile(rest: str, state: AppState, console: Console):
    """
    pctile newvar = expr [if] [in] [weight] [, nquantiles(N) percentiles(p1 p2 ...) genp(name)]

    Creates a new variable whose first len(cuts) rows hold the percentile
    cut points of `expr`, and the rest are missing.

    Options:
        nquantiles(N)    cuts at N-1 equally-spaced percentiles (e.g. 4 = quartiles)
        percentiles(p*)  explicit percentile list (values 0..100)
        genp(name)       also create a companion column holding the p values

    Examples:
        pctile cut = income, nquantiles(4)
        pctile cut = wage [fweight=pop], percentiles(10 25 50 75 90) genp(p)
    """
    state.require_data()
    parsed = parse_command_line(rest)
    if not parsed["varlist"] or not parsed["expression"]:
        console.print("[red]Syntax: pctile newvar = expr [if] [in] [weight] "
                      "[, nquantiles(N) percentiles(p1 p2 ...) genp(name)][/red]")
        return

    newvar = parsed["varlist"][0]
    expr = parsed["expression"]

    df = state.data
    sub = _apply_if_in(df, parsed)
    weights = get_weight_series(parsed, sub, console)
    if weights is False:
        return

    # Evaluate the expression on `sub`
    from commands.expression import eval_expression
    try:
        vals = eval_expression(expr, sub)
    except Exception as e:
        console.print(f"[red]pctile: cannot evaluate '{expr}': {e}[/red]")
        return

    opts = parsed["options"]
    if "percentiles" in opts:
        try:
            pcts = [float(p) for p in opts["percentiles"].split()]
        except ValueError:
            console.print("[red]pctile: percentiles() needs space-separated numbers[/red]")
            return
    else:
        nq = int(opts.get("nquantiles", 2))
        if nq < 2:
            console.print("[red]pctile: nquantiles must be >= 2[/red]")
            return
        pcts = [100 * i / nq for i in range(1, nq)]

    x = np.asarray(vals, dtype=float)
    w = np.ones_like(x) if weights is None else weights.to_numpy()
    cuts = [weighted_quantile(x, w, p / 100.0) for p in pcts]

    # Place cuts into the first len(cuts) rows of newvar; rest are NaN
    col = pd.Series([np.nan] * len(df), index=df.index, dtype=float)
    for i, c in enumerate(cuts):
        if i < len(col):
            col.iloc[i] = c
    df[newvar] = col

    if "genp" in opts:
        genp_name = opts["genp"]
        pcol = pd.Series([np.nan] * len(df), index=df.index, dtype=float)
        for i, p in enumerate(pcts):
            if i < len(pcol):
                pcol.iloc[i] = p
        df[genp_name] = pcol

    state.mark_changed()
    console.print(f"[green]pctile: created '{newvar}' with {len(cuts)} cut points.[/green]")
    console.print(f"  [dim]percentiles: {', '.join(f'{p:g}' for p in pcts)}[/dim]")
    console.print(f"  [dim]cuts:        {', '.join(f'{c:.4f}' for c in cuts)}[/dim]")


# ──────────────────────────────────────────────────────────────
# xtile — assign each row to a quantile bin
# ──────────────────────────────────────────────────────────────

def cmd_xtile(rest: str, state: AppState, console: Console):
    """
    xtile newvar = expr [if] [in] [weight] [, nquantiles(N) cutpoints(varname) altdef]

    Assigns each row to one of N bins (1..N) based on `expr`.
    Default is quartiles (nquantiles(4)).

    Options:
        nquantiles(N)     number of bins; rows are split at N-1 percentiles
        cutpoints(var)    use the first k non-missing values of `var` as cut points
        altdef            tie-breaking rule: strictly greater than cut (default)
                          without altdef: >= cut (includes ties on lower side)

    Examples:
        xtile inc_quartile = income, nquantiles(4)
        xtile inc_decile   = income [pweight=w], nquantiles(10)
    """
    state.require_data()
    parsed = parse_command_line(rest)
    if not parsed["varlist"] or not parsed["expression"]:
        console.print("[red]Syntax: xtile newvar = expr [if] [weight] "
                      "[, nquantiles(N) cutpoints(var) altdef][/red]")
        return

    newvar = parsed["varlist"][0]
    expr = parsed["expression"]

    df = state.data
    sub = _apply_if_in(df, parsed)
    weights = get_weight_series(parsed, sub, console)
    if weights is False:
        return

    from commands.expression import eval_expression
    try:
        vals_sub = eval_expression(expr, sub)
        vals_full = eval_expression(expr, df)
    except Exception as e:
        console.print(f"[red]xtile: cannot evaluate '{expr}': {e}[/red]")
        return

    opts = parsed["options"]
    if "cutpoints" in opts:
        cp_name = opts["cutpoints"]
        if cp_name not in df.columns:
            console.print(f"[red]cutpoints variable '{cp_name}' not found[/red]")
            return
        cuts = df[cp_name].dropna().tolist()
        if not cuts:
            console.print(f"[red]cutpoints variable '{cp_name}' is all missing[/red]")
            return
    else:
        nq = int(opts.get("nquantiles", 4))
        if nq < 2:
            console.print("[red]xtile: nquantiles must be >= 2[/red]")
            return
        pcts = [100 * i / nq for i in range(1, nq)]
        x = np.asarray(vals_sub, dtype=float)
        w = np.ones_like(x) if weights is None else weights.to_numpy()
        cuts = [weighted_quantile(x, w, p / 100.0) for p in pcts]

    cuts = sorted(cuts)
    altdef = "altdef" in opts

    vals_arr = np.asarray(vals_full, dtype=float)
    bins = np.full(len(vals_arr), np.nan)
    nan_mask = np.isnan(vals_arr)
    for i, v in enumerate(vals_arr):
        if nan_mask[i]:
            continue
        bin_idx = 1
        for c in cuts:
            if (altdef and v >  c) or ((not altdef) and v >= c):
                bin_idx += 1
        bins[i] = bin_idx

    df[newvar] = bins
    state.mark_changed()
    console.print(f"[green]xtile: assigned '{newvar}' with {len(cuts)+1} bins.[/green]")
    counts = pd.Series(bins).value_counts().sort_index()
    console.print("  [dim]bin counts: " +
                  "  ".join(f"{int(k)}:{int(v)}" for k, v in counts.items()) +
                  "[/dim]")


# ──────────────────────────────────────────────────────────────
# centile — report one or more percentiles with confidence intervals
# ──────────────────────────────────────────────────────────────

def cmd_centile(rest: str, state: AppState, console: Console):
    """
    centile [varlist] [if] [in] [weight] [, centile(p1 p2 ...) level(95)]

    Report specified percentiles with binomial-based confidence intervals.
    Default percentiles: 50 (median).

    The CI uses the method of Conover (1971): find the order statistics
    whose ranks bracket the coverage probability under the binomial CDF.
    Reports NaN for CI under weights (CIs on weighted quantiles require
    bootstrap; this may come in a later release).

    Examples:
        centile wage
        centile wage, centile(10 50 90)
        centile wage educ, centile(25 50 75) level(99)
    """
    state.require_data()
    parsed = parse_command_line(rest)

    df = state.data
    sub = _apply_if_in(df, parsed)
    weights = get_weight_series(parsed, sub, console)
    if weights is False:
        return

    varlist = parsed["varlist"]
    if not varlist:
        varlist = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    numeric_vars = [v for v in varlist
                    if v in df.columns and pd.api.types.is_numeric_dtype(df[v])]
    if not numeric_vars:
        console.print("[red]centile: no numeric variables to report[/red]")
        return

    opts = parsed["options"]
    if "centile" in opts:
        try:
            pcts = [float(p) for p in opts["centile"].split()]
        except ValueError:
            console.print("[red]centile: centile() needs space-separated numbers[/red]")
            return
    else:
        pcts = [50.0]
    level = float(opts.get("level", 95))
    alpha = (100 - level) / 100.0

    table = Table(title="Percentiles" + (f" (weighted)" if weights is not None else ""))
    table.add_column("Variable", style="bold")
    table.add_column("Obs", justify="right")
    table.add_column("Pct", justify="right")
    table.add_column("Value", justify="right")
    table.add_column(f"{level:g}% CI", justify="center")

    from scipy.stats import binom

    for v in numeric_vars:
        x = sub[v].dropna().to_numpy()
        n = len(x)
        if n == 0:
            continue
        xs = np.sort(x)
        if weights is not None:
            w = weights.loc[sub[v].dropna().index].to_numpy()
            for p in pcts:
                val = weighted_quantile(x, w, p / 100.0)
                table.add_row(v, f"{n:,}", f"{p:g}", f"{val:.4f}", "(weighted: no CI)")
        else:
            for p in pcts:
                q = p / 100.0
                val = weighted_quantile(x, np.ones(n), q)
                # Binomial-based CI: find ranks l, u with
                # P(l <= X_(k) <= u) >= 1 - alpha where k ~ Bin(n, q).
                lo_rank = int(binom.ppf(alpha / 2, n, q))
                hi_rank = int(binom.ppf(1 - alpha / 2, n, q)) + 1
                lo_rank = max(0, lo_rank)
                hi_rank = min(n - 1, hi_rank)
                lo = xs[lo_rank]
                hi = xs[hi_rank]
                table.add_row(v, f"{n:,}", f"{p:g}", f"{val:.4f}",
                              f"[{lo:.4f}, {hi:.4f}]")

    console.print(table)


# ──────────────────────────────────────────────────────────────
# winsor2 — trim outliers at given percentiles
# ──────────────────────────────────────────────────────────────

def cmd_winsor2(rest: str, state: AppState, console: Console):
    """
    winsor2 varlist [if] [in] [weight] [, cuts(lo hi) suffix(_w) replace trim by(g)]

    Replace extreme values of each variable in varlist with the values at
    the given percentiles. By default winsorizes at (1, 99) and produces
    new variables with suffix `_w`.

    Options:
        cuts(lo hi)   lower and upper percentiles (default: 1 99)
        suffix(str)   suffix for the new variables (default: _w)
        replace       overwrite the original columns (no new variables)
        trim          DROP outliers instead of capping them at the cut values
        by(group)     compute cuts per group

    Examples:
        winsor2 wage                                   (creates wage_w at 1/99)
        winsor2 wage, cuts(5 95) replace
        winsor2 wage income, cuts(1 99) by(industry)
        winsor2 wage [aweight=w], cuts(2 98)
    """
    state.require_data()
    parsed = parse_command_line(rest)
    varlist = parsed["varlist"]
    if not varlist:
        console.print("[red]Syntax: winsor2 varlist [if] [weight] "
                      "[, cuts(lo hi) suffix(_w) replace trim by(g)][/red]")
        return

    opts = parsed["options"]
    cuts_str = opts.get("cuts", "1 99").split()
    try:
        p_lo, p_hi = float(cuts_str[0]), float(cuts_str[1])
    except (ValueError, IndexError):
        console.print("[red]winsor2: cuts() must be two numbers (e.g. cuts(1 99))[/red]")
        return
    if not (0 <= p_lo < p_hi <= 100):
        console.print("[red]winsor2: cuts must satisfy 0 <= lo < hi <= 100[/red]")
        return

    suffix = opts.get("suffix", "_w")
    replace = "replace" in opts
    trim = "trim" in opts
    by_var = opts.get("by")

    df = state.data
    sub_mask = pd.Series(True, index=df.index)
    if parsed["if_cond"]:
        sub_mask &= eval_condition(parsed["if_cond"], df)
    if parsed["in_range"]:
        sl = parse_in_range(parsed["in_range"])
        if sl:
            in_mask = pd.Series(False, index=df.index)
            in_mask.iloc[sl] = True
            sub_mask &= in_mask

    weights = get_weight_series(parsed, df, console)
    if weights is False:
        return

    def compute_cuts(series, w):
        x = series.to_numpy(dtype=float)
        wt = np.ones_like(x) if w is None else w.to_numpy()
        return (weighted_quantile(x, wt, p_lo / 100.0),
                weighted_quantile(x, wt, p_hi / 100.0))

    for var in varlist:
        if var not in df.columns:
            console.print(f"[yellow]winsor2: skipping '{var}' (not found)[/yellow]")
            continue
        if not pd.api.types.is_numeric_dtype(df[var]):
            console.print(f"[yellow]winsor2: skipping '{var}' (not numeric)[/yellow]")
            continue

        target = var if replace else f"{var}{suffix}"
        new_col = df[var].copy() if not trim else df[var].copy()

        if by_var:
            if by_var not in df.columns:
                console.print(f"[red]by variable '{by_var}' not found[/red]")
                return
            for grp_val, sub in df.loc[sub_mask].groupby(by_var):
                w = None if weights is None else weights.loc[sub.index]
                lo, hi = compute_cuts(sub[var], w)
                idx = sub.index
                if trim:
                    bad = (df.loc[idx, var] < lo) | (df.loc[idx, var] > hi)
                    new_col.loc[idx[bad]] = np.nan
                else:
                    new_col.loc[idx] = new_col.loc[idx].clip(lower=lo, upper=hi)
        else:
            sub_rows = df.loc[sub_mask]
            w = None if weights is None else weights.loc[sub_rows.index]
            lo, hi = compute_cuts(sub_rows[var], w)
            if trim:
                bad = sub_mask & ((df[var] < lo) | (df[var] > hi))
                new_col.loc[bad] = np.nan
            else:
                new_col.loc[sub_mask] = new_col.loc[sub_mask].clip(lower=lo, upper=hi)

        # Compute the modification count BEFORE writing back, since with
        # `replace` (target == var) the assignment otherwise makes
        # df[var] == new_col by definition. (Minimax v126 B3.)
        if trim:
            old_na = int(df[var].isna().sum())
            new_na = int(new_col.isna().sum())
            n_modified = max(0, new_na - old_na)
        else:
            # Both columns aligned by index; compare element-wise but
            # treat NaN==NaN as equal so we don't over-count.
            both_nan = df[var].isna() & new_col.isna()
            differs = (df[var] != new_col) & ~both_nan
            n_modified = int(differs.sum())

        df[target] = new_col
        action = "trimmed" if trim else "winsorized"
        console.print(f"[green]{var} -> {target}: {action} "
                      f"(cuts p{p_lo:g}={lo:.4f}, p{p_hi:g}={hi:.4f}; {n_modified} values changed)[/green]")

    state.mark_changed()
