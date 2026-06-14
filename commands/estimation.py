"""
Estimation commands: regress, predict, test, correlate, pwcorr, ttest
Wraps statsmodels for OLS with robust/clustered SE and factor variables.
"""

import re
import numpy as np
import pandas as pd
from rich.console import Console
from rich.table import Table

from commands.state import AppState
from commands.parse_helpers import parse_command_line
from commands.expression import eval_condition


def _check_statsmodels():
    try:
        import statsmodels.api as sm
        return sm
    except ImportError as e:
        raise ImportError(
            "statsmodels is required for estimation commands.\n"
            f"(underlying import error: {e})\n"
            "Install with: ssc install statsmodels"
        )


# ──────────────────────────────────────────────
#  regress
# ──────────────────────────────────────────────

def cmd_regress(rest: str, state: AppState, console: Console):
    """
    regress depvar indepvars [if condition] [, robust cluster(var) noconstant]
    
    OLS regression. Supports:
      - robust (HC1) standard errors
      - cluster(varname) clustered standard errors  
      - i.var factor variables (creates dummies)
      - noconstant
    
    Results stored in e().
    """
    state.require_data()
    sm = _check_statsmodels()

    parsed = parse_command_line(rest)
    if len(parsed["varlist"]) < 2:
        console.print("[red]Syntax: regress depvar indepvar1 [indepvar2 ...] [, robust cluster(var)][/red]")
        return

    depvar = parsed["varlist"][0]
    indepvars_raw = parsed["varlist"][1:]

    # Apply if condition
    df = state.data.copy()
    if parsed["if_cond"]:
        mask = eval_condition(parsed["if_cond"], df)
        df = df.loc[mask]

    # Check depvar
    if depvar not in df.columns:
        console.print(f"[red]Variable '{depvar}' not found[/red]")
        return

    # Process independent variables (handle i.var factor variables)
    indepvars = []
    for v in indepvars_raw:
        if v.startswith("i."):
            # Factor variable: create dummies
            base_var = v[2:]
            if base_var not in df.columns:
                console.print(f"[red]Variable '{base_var}' not found[/red]")
                return
            dummies = pd.get_dummies(df[base_var], prefix=base_var, drop_first=True, dtype=float)
            for col in dummies.columns:
                df[col] = dummies[col]
                indepvars.append(col)
        elif v.startswith("c."):
            # Continuous (explicit, just strip prefix)
            indepvars.append(v[2:])
        else:
            if v not in df.columns:
                console.print(f"[red]Variable '{v}' not found[/red]")
                return
            indepvars.append(v)

    # Drop missing. Include the weight column too so weight-NaN rows are
    # dropped, and so get_weight_series() can evaluate bare-variable weight
    # clauses against df_clean afterward.
    all_vars = [depvar] + indepvars
    drop_cols = list(all_vars)
    wclause = parsed.get("weight")
    if wclause:
        if wclause.get("is_expr"):
            # Expression form may reference any column — be safe, keep all rows
            # needed for the RHS variables and compute weight after dropna.
            pass
        elif wclause.get("var"):
            if wclause["var"] in df.columns and wclause["var"] not in drop_cols:
                drop_cols.append(wclause["var"])
    df_clean = df[drop_cols].dropna()
    if wclause and not wclause.get("is_expr") and wclause.get("var"):
        # Drop zero-weight rows (convention: weight=0 means "excluded")
        df_clean = df_clean[df_clean[wclause["var"]] > 0]
    n_dropped = len(df) - len(df_clean)

    if len(df_clean) < len(indepvars) + 1:
        console.print("[red]Not enough observations for regression[/red]")
        return

    y = df_clean[depvar].astype(float)
    X = df_clean[indepvars].astype(float)

    # Add constant unless noconstant
    noconstant = "noconstant" in parsed["options"] or "nocons" in parsed["options"]
    if not noconstant:
        X = sm.add_constant(X)

    # Weight handling — use WLS when an analytic/frequency weight is given.
    # pweight (sampling weights) requires robust SE by construction — we
    # enforce that automatically below.
    from commands.weights import get_weight_series
    weights = get_weight_series(parsed, df_clean, console)
    if weights is False:
        return
    wtype = parsed["weight"]["type"] if parsed["weight"] else None
    if weights is not None:
        # Ensure pweight triggers robust SE
        if wtype == "pweight" and not (robust_flag := (
                "robust" in parsed["options"] or "r" in parsed["options"]
                or parsed["options"].get("cluster") or parsed["options"].get("cl"))):
            parsed["options"]["robust"] = True
            console.print("[dim]pweight implies robust; enabling HC1 standard errors.[/dim]")

    # Fit model
    robust = "robust" in parsed["options"] or "r" in parsed["options"]
    cluster_var = parsed["options"].get("cluster") or parsed["options"].get("cl")

    if weights is not None:
        model = sm.WLS(y, X, weights=weights.values)
    else:
        model = sm.OLS(y, X)

    if cluster_var:
        if cluster_var not in df.columns:
            console.print(f"[red]Cluster variable '{cluster_var}' not found[/red]")
            return
        cluster_series = df.loc[df_clean.index, cluster_var]
        results = model.fit(cov_type="cluster", cov_kwds={"groups": cluster_series})
        se_type = f"(Std. err. adjusted for {cluster_series.nunique()} clusters in {cluster_var})"
    elif robust:
        results = model.fit(cov_type="HC1")
        se_type = "(Robust standard errors)"
    else:
        results = model.fit()
        se_type = "(Standard errors)"

    # Display — prepend weight banner if applicable
    if wtype:
        console.print(f"[dim]Weight: {wtype}={parsed['weight']['var']}[/dim]")
    _display_regression(results, depvar, indepvars, len(df_clean), n_dropped,
                        se_type, noconstant, state, console)

    # Store results in e()
    state.e_results = {
        "cmd": "regress",
        "depvar": depvar,
        "N": len(df_clean),
        "r2": results.rsquared,
        "r2_a": results.rsquared_adj,
        "F": results.fvalue if hasattr(results, 'fvalue') else None,
        "rmse": np.sqrt(results.mse_resid),
        "b": results.params,
        "se": results.bse,
        "t": results.tvalues,
        "pval": results.pvalues,
        "ci": results.conf_int(),
        "V": results.cov_params(),
        "predict_model": results,
        "predict_X_cols": list(X.columns),
        "predict_index": df_clean.index,
    }

    # Also store in r()
    state.r_results = {
        "N": len(df_clean),
        "r2": results.rsquared,
        "r2_a": results.rsquared_adj,
        "F": float(results.fvalue) if results.fvalue is not None else None,
        "rmse": float(np.sqrt(results.mse_resid)),
    }


def _display_regression(results, depvar, indepvars, n_obs, n_dropped,
                        se_type, noconstant, state, console):
    """Display regression results in compact tabular format."""
    console.print(f"\n[bold]Source[/bold]          SS           df       MS")
    console.print(f"{'─' * 55}")

    ss_model = results.ess
    ss_resid = results.ssr
    ss_total = ss_model + ss_resid
    df_model = results.df_model
    df_resid = results.df_resid

    console.print(f"Model     {ss_model:>14.4f}  {int(df_model):>5}  {ss_model/max(df_model,1):>14.4f}")
    console.print(f"Residual  {ss_resid:>14.4f}  {int(df_resid):>5}  {ss_resid/max(df_resid,1):>14.4f}")
    console.print(f"{'─' * 55}")
    console.print(f"Total     {ss_total:>14.4f}  {int(df_model+df_resid):>5}  {ss_total/max(df_model+df_resid,1):>14.4f}")

    console.print()
    console.print(f"  Number of obs  = {n_obs:>10,}")
    if n_dropped > 0:
        console.print(f"  [dim]({n_dropped} obs dropped due to missing values)[/dim]")
    f_val = results.fvalue
    f_prob = results.f_pvalue
    console.print(f"  F({int(df_model)}, {int(df_resid)})       = {f_val:>10.2f}")
    console.print(f"  Prob > F       = {f_prob:>10.4f}")
    console.print(f"  R-squared      = {results.rsquared:>10.4f}")
    console.print(f"  Adj R-squared  = {results.rsquared_adj:>10.4f}")
    console.print(f"  Root MSE       = {np.sqrt(results.mse_resid):>10.4f}")
    console.print()
    console.print(f"  {se_type}")
    console.print()

    # Coefficient table
    table = Table(show_lines=False)
    table.add_column(depvar, style="bold", min_width=15)
    table.add_column("Coefficient", justify="right", min_width=12)
    table.add_column("Std. err.", justify="right", min_width=10)
    table.add_column("t", justify="right", min_width=8)
    table.add_column("P>|t|", justify="right", min_width=8)
    table.add_column("[95% conf.", justify="right", min_width=10)
    table.add_column("interval]", justify="right", min_width=10)

    ci = results.conf_int()

    for var_name in results.params.index:
        coef = results.params[var_name]
        se = results.bse[var_name]
        t = results.tvalues[var_name]
        pval = results.pvalues[var_name]
        ci_lo = ci.loc[var_name, 0]
        ci_hi = ci.loc[var_name, 1]

        # Highlight significant coefficients
        p_style = ""
        if pval < 0.01:
            p_style = "bold"
        elif pval < 0.05:
            p_style = ""

        display_name = var_name if var_name != "const" else "_cons"
        table.add_row(
            display_name,
            f"{coef:.6f}",
            f"{se:.6f}",
            f"{t:.2f}",
            f"[{p_style}]{pval:.3f}[/{p_style}]" if p_style else f"{pval:.3f}",
            f"{ci_lo:.6f}",
            f"{ci_hi:.6f}",
        )

    console.print(table)
    console.print()


# ──────────────────────────────────────────────
#  predict
# ──────────────────────────────────────────────

def cmd_predict(rest: str, state: AppState, console: Console):
    """
    predict newvar [, xb residuals]
    
    Generate predicted values or residuals from the last estimation.
    """
    state.require_data()
    parsed = parse_command_line(rest)

    if not parsed["varlist"]:
        console.print("[red]Syntax: predict newvar [, xb residuals][/red]")
        return

    newvar = parsed["varlist"][0]
    if newvar in state.data.columns:
        console.print(f"[red]Variable '{newvar}' already exists[/red]")
        return

    if not state.e_results or "predict_model" not in state.e_results:
        console.print("[red]No estimation results. Run regress first.[/red]")
        return

    model = state.e_results["predict_model"]
    idx = state.e_results["predict_index"]

    if "residuals" in parsed["options"] or "residual" in parsed["options"] or "resid" in parsed["options"] or "r" in parsed["options"]:
        # Residuals
        state.data[newvar] = np.nan
        state.data.loc[idx, newvar] = model.resid.values
        console.print(f"[green]Generated: {newvar} (residuals, {len(idx)} values)[/green]")
    else:
        # Predicted values (xb is default)
        state.data[newvar] = np.nan
        state.data.loc[idx, newvar] = model.fittedvalues.values
        console.print(f"[green]Generated: {newvar} (fitted values, {len(idx)} values)[/green]")

    state.mark_changed()


# ──────────────────────────────────────────────
#  test
# ──────────────────────────────────────────────

def cmd_test(rest: str, state: AppState, console: Console):
    """
    test var1 [var2 ...]
    test var1 = 0
    test var1 = var2
    
    Wald test of linear hypotheses after estimation.
    """
    if not state.e_results or "predict_model" not in state.e_results:
        console.print("[red]No estimation results. Run regress first.[/red]")
        return

    model = state.e_results["predict_model"]
    rest = rest.strip()

    if "=" in rest:
        # test var = value
        parts = rest.split("=")
        var_name = parts[0].strip()
        val = float(parts[1].strip())

        if var_name not in model.params.index:
            # Try _cons
            if var_name == "_cons" and "const" in model.params.index:
                var_name = "const"
            else:
                console.print(f"[red]Variable '{var_name}' not in model[/red]")
                return

        hypothesis = f"{var_name} = {val}"
        try:
            t_test = model.t_test(f"{var_name} = {val}")
            console.print(f"\n  [bold]( 1)  {hypothesis}[/bold]")
            console.print(f"\n       F(  1, {int(model.df_resid)}) = {float(t_test.statistic**2):>8.2f}")
            console.print(f"            Prob > F = {float(t_test.pvalue):>8.4f}")
            console.print()
        except Exception as e:
            console.print(f"[red]Test failed: {e}[/red]")
    else:
        # test var1 var2 ... (joint test that all = 0)
        var_names = rest.split()
        resolved = []
        for v in var_names:
            if v in model.params.index:
                resolved.append(v)
            elif v == "_cons" and "const" in model.params.index:
                resolved.append("const")
            else:
                console.print(f"[red]Variable '{v}' not in model[/red]")
                return

        try:
            hypothesis = ", ".join(f"{v} = 0" for v in resolved)
            f_test = model.f_test(" = ".join([f"{resolved[0]} = 0"]) if len(resolved) == 1
                                  else ", ".join(f"{v} = 0" for v in resolved))
            console.print(f"\n  [bold]Joint test:[/bold]")
            for i, v in enumerate(resolved, 1):
                console.print(f"  ( {i})  {v} = 0")
            console.print(f"\n       F({len(resolved)}, {int(model.df_resid)}) = {float(f_test.statistic[0][0]):>8.2f}")
            console.print(f"            Prob > F = {float(f_test.pvalue):>8.4f}")
            console.print()
        except Exception as e:
            console.print(f"[red]Test failed: {e}[/red]")


# ──────────────────────────────────────────────
#  correlate / pwcorr
# ──────────────────────────────────────────────

def cmd_correlate(rest: str, state: AppState, console: Console):
    """
    correlate varlist [if condition]
    Display correlation matrix.
    """
    state.require_data()
    parsed = parse_command_line(rest)

    if not parsed["varlist"]:
        # All numeric
        num_cols = [c for c in state.data.columns if pd.api.types.is_numeric_dtype(state.data[c])]
        varlist = num_cols[:10]  # Limit
    else:
        varlist = parsed["varlist"]

    df = state.data
    if parsed["if_cond"]:
        mask = eval_condition(parsed["if_cond"], df)
        df = df.loc[mask]

    corr = df[varlist].corr()

    table = Table(title="Correlation Matrix")
    table.add_column("", style="bold", min_width=12)
    for v in varlist:
        table.add_column(v[:10], justify="right", min_width=8)

    for v1 in varlist:
        row = [v1[:12]]
        for v2 in varlist:
            val = corr.loc[v1, v2]
            row.append(f"{val:.4f}")
        table.add_row(*row)

    console.print(table)


def cmd_pwcorr(rest: str, state: AppState, console: Console):
    """
    pwcorr varlist [if condition] [, sig star(0.05)]
    Pairwise correlations with significance levels.
    """
    state.require_data()
    parsed = parse_command_line(rest)

    if not parsed["varlist"]:
        console.print("[red]Syntax: pwcorr varlist[/red]")
        return

    varlist = parsed["varlist"]
    df = state.data
    if parsed["if_cond"]:
        mask = eval_condition(parsed["if_cond"], df)
        df = df.loc[mask]

    show_sig = "sig" in parsed["options"]
    star_level = float(parsed["options"].get("star", 0.05))

    from scipy import stats as scipy_stats

    table = Table(title="Pairwise Correlations")
    table.add_column("", style="bold", min_width=12)
    for v in varlist:
        table.add_column(v[:10], justify="right", min_width=10)

    for v1 in varlist:
        row = [v1[:12]]
        for v2 in varlist:
            pair = df[[v1, v2]].dropna()
            if len(pair) < 3:
                row.append(".")
                continue
            r, p = scipy_stats.pearsonr(pair[v1], pair[v2])
            star = "*" if p < star_level else ""
            cell = f"{r:.4f}{star}"
            if show_sig:
                cell += f"\n({p:.4f})"
            row.append(cell)
        table.add_row(*row)

    console.print(table)


# ──────────────────────────────────────────────
#  ttest
# ──────────────────────────────────────────────

def cmd_ttest(rest: str, state: AppState, console: Console):
    """
    ttest var == value
    ttest var1 == var2
    ttest var, by(groupvar)
    
    One-sample, paired, or two-sample t-test.
    """
    state.require_data()
    parsed = parse_command_line(rest)

    by_var = parsed["options"].get("by")

    if by_var:
        # Two-sample t-test by group
        var = parsed["varlist"][0] if parsed["varlist"] else None
        if not var:
            console.print("[red]Syntax: ttest var, by(groupvar)[/red]")
            return
        _ttest_by_group(var, by_var, state, console)
        return

    # Parse == for one-sample or paired
    raw = parsed["raw"]
    if "==" in raw:
        parts = raw.split("==")
        var1 = parts[0].strip()
        val_str = parts[1].strip().split(",")[0].strip()  # remove options

        if var1 not in state.data.columns:
            console.print(f"[red]Variable '{var1}' not found[/red]")
            return

        try:
            val = float(val_str)
            _ttest_one_sample(var1, val, state, console)
        except ValueError:
            # Paired test
            if val_str in state.data.columns:
                _ttest_paired(var1, val_str, state, console)
            else:
                console.print(f"[red]'{val_str}' is not a number or variable[/red]")
    else:
        console.print("[red]Syntax: ttest var == value | ttest var, by(group)[/red]")


def _ttest_one_sample(var, mu, state, console):
    from scipy import stats as scipy_stats
    s = state.data[var].dropna()
    t, p = scipy_stats.ttest_1samp(s, mu)
    console.print(f"\n[bold]One-sample t-test: {var}[/bold]")
    console.print(f"  H0: mean = {mu}")
    console.print(f"  Obs: {len(s):,}, Mean: {s.mean():.6f}, Std. dev.: {s.std():.6f}")
    console.print(f"  t = {t:.4f}, P>|t| = {p:.4f}")
    console.print(f"  95% CI: [{s.mean() - 1.96*s.std()/np.sqrt(len(s)):.4f}, {s.mean() + 1.96*s.std()/np.sqrt(len(s)):.4f}]")
    console.print()


def _ttest_paired(var1, var2, state, console):
    from scipy import stats as scipy_stats
    df = state.data[[var1, var2]].dropna()
    t, p = scipy_stats.ttest_rel(df[var1], df[var2])
    diff = df[var1] - df[var2]
    console.print(f"\n[bold]Paired t-test: {var1} vs {var2}[/bold]")
    console.print(f"  Obs: {len(df):,}")
    console.print(f"  Mean diff: {diff.mean():.6f}, Std. dev.: {diff.std():.6f}")
    console.print(f"  t = {t:.4f}, P>|t| = {p:.4f}")
    console.print()


def _ttest_by_group(var, by_var, state, console):
    from scipy import stats as scipy_stats
    if var not in state.data.columns:
        console.print(f"[red]Variable '{var}' not found[/red]")
        return
    if by_var not in state.data.columns:
        console.print(f"[red]Group variable '{by_var}' not found[/red]")
        return

    groups = state.data.groupby(by_var)[var].apply(lambda x: x.dropna())
    group_names = list(state.data[by_var].dropna().unique())

    if len(group_names) != 2:
        console.print(f"[yellow]by() variable has {len(group_names)} groups (expected 2)[/yellow]")
        if len(group_names) < 2:
            return
        console.print(f"[dim]Using first two: {group_names[0]} and {group_names[1]}[/dim]")

    g1 = state.data[state.data[by_var] == group_names[0]][var].dropna()
    g2 = state.data[state.data[by_var] == group_names[1]][var].dropna()

    t, p = scipy_stats.ttest_ind(g1, g2)

    console.print(f"\n[bold]Two-sample t-test: {var} by {by_var}[/bold]")
    console.print(f"  Group '{group_names[0]}': n={len(g1):,}, mean={g1.mean():.6f}, sd={g1.std():.6f}")
    console.print(f"  Group '{group_names[1]}': n={len(g2):,}, mean={g2.mean():.6f}, sd={g2.std():.6f}")
    console.print(f"  Diff: {g1.mean() - g2.mean():.6f}")
    console.print(f"  t = {t:.4f}, P>|t| = {p:.4f}")

    # Also Welch's
    t_w, p_w = scipy_stats.ttest_ind(g1, g2, equal_var=False)
    console.print(f"  Welch's t = {t_w:.4f}, P>|t| = {p_w:.4f}")
    console.print()
