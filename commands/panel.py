"""
Panel and causal estimation commands:
  xtset, xtreg (FE/RE), didregress (diff-in-diff via diff-diff),
  margins, lincom
"""

import re
import numpy as np
import pandas as pd
from rich.console import Console
from rich.table import Table

from commands.state import AppState
from commands.parse_helpers import parse_command_line
from commands.expression import eval_condition


# ──────────────────────────────────────────────
#  xtset — declare panel structure
# ──────────────────────────────────────────────

def cmd_xtset(rest: str, state: AppState, console: Console):
    """
    xtset panelvar [timevar]
    Declare panel data structure.
    """
    state.require_data()
    parsed = parse_command_line(rest)

    if not parsed["varlist"]:
        # Display current setting
        if hasattr(state, '_panel_var') and state._panel_var:
            console.print(f"  Panel variable: {state._panel_var}")
            if state._time_var:
                console.print(f"  Time variable:  {state._time_var}")
        else:
            console.print("[dim]Panel data not set. Use: xtset panelvar [timevar][/dim]")
        return

    panel_var = parsed["varlist"][0]
    time_var = parsed["varlist"][1] if len(parsed["varlist"]) > 1 else None

    if panel_var not in state.data.columns:
        console.print(f"[red]Variable '{panel_var}' not found[/red]")
        return
    if time_var and time_var not in state.data.columns:
        console.print(f"[red]Variable '{time_var}' not found[/red]")
        return

    state._panel_var = panel_var
    state._time_var = time_var

    n_panels = state.data[panel_var].nunique()
    console.print(f"  Panel variable: [bold]{panel_var}[/bold] ({n_panels:,} panels)")
    if time_var:
        n_times = state.data[time_var].nunique()
        t_min = state.data[time_var].min()
        t_max = state.data[time_var].max()
        console.print(f"  Time variable:  [bold]{time_var}[/bold] ({t_min} to {t_max}, {n_times} periods)")

    # Check balance
    panel_sizes = state.data.groupby(panel_var).size()
    if panel_sizes.nunique() == 1:
        console.print(f"  [dim]Balanced panel: {panel_sizes.iloc[0]} obs per panel[/dim]")
    else:
        console.print(f"  [dim]Unbalanced panel: {panel_sizes.min()}-{panel_sizes.max()} obs per panel[/dim]")


# ──────────────────────────────────────────────
#  xtreg — panel regression (FE / RE)
# ──────────────────────────────────────────────

def cmd_xtreg(rest: str, state: AppState, console: Console):
    """
    xtreg depvar indepvars [if] [, fe re robust cluster(var)]
    
    Panel data regression.
    fe = fixed effects (within estimator)
    re = random effects (GLS)
    Default: re
    
    Requires: pip install linearmodels  OR  uses statsmodels entity dummies
    """
    state.require_data()
    parsed = parse_command_line(rest)

    if len(parsed["varlist"]) < 2:
        console.print("[red]Syntax: xtreg depvar indepvars [, fe re robust][/red]")
        return

    if not hasattr(state, '_panel_var') or not state._panel_var:
        console.print("[red]Panel data not set. Use xtset first.[/red]")
        return

    depvar = parsed["varlist"][0]
    indepvars = parsed["varlist"][1:]
    use_fe = "fe" in parsed["options"]
    use_re = "re" in parsed["options"]
    robust = "robust" in parsed["options"] or "r" in parsed["options"]
    cluster_var = parsed["options"].get("cluster") or parsed["options"].get("cl")

    if not use_fe and not use_re:
        use_re = True  # Default

    # Apply if condition
    df = state.data.copy()
    if parsed["if_cond"]:
        mask = eval_condition(parsed["if_cond"], df)
        df = df.loc[mask]

    # Validate
    all_vars = [depvar] + indepvars + [state._panel_var]
    if state._time_var:
        all_vars.append(state._time_var)
    for v in all_vars:
        if v not in df.columns:
            console.print(f"[red]Variable '{v}' not found[/red]")
            return

    df_clean = df[all_vars].dropna()
    n_dropped = len(df) - len(df_clean)

    # Try linearmodels first
    try:
        _xtreg_linearmodels(df_clean, depvar, indepvars, state, use_fe, robust, cluster_var, n_dropped, console)
        return
    except ImportError:
        pass

    # Fallback: FE via entity dummies with statsmodels
    _xtreg_statsmodels_fallback(df_clean, depvar, indepvars, state, use_fe, robust, n_dropped, console)


def _xtreg_linearmodels(df, depvar, indepvars, state, use_fe, robust, cluster_var, n_dropped, console):
    """Panel regression using linearmodels."""
    from linearmodels.panel import PanelOLS, RandomEffects

    panel_var = state._panel_var
    time_var = state._time_var

    # Set multi-index
    if time_var:
        df = df.set_index([panel_var, time_var])
    else:
        df = df.set_index(panel_var)

    y = df[depvar]
    X = df[indepvars]

    if use_fe:
        model = PanelOLS(y, X, entity_effects=True, check_rank=False)
        model_type = "Fixed-effects (within) regression"
    else:
        model = RandomEffects(y, X, check_rank=False)
        model_type = "Random-effects GLS regression"

    cov_type = "robust" if robust else ("clustered" if cluster_var else "unadjusted")
    if cluster_var and cluster_var in state.data.columns:
        results = model.fit(cov_type="clustered", cluster_entity=True)
    elif robust:
        results = model.fit(cov_type="robust")
    else:
        results = model.fit()

    # Display
    n_panels = df.index.get_level_values(0).nunique() if time_var else df.index.nunique()
    console.print(f"\n[bold]{model_type}[/bold]")
    console.print(f"  Group variable: {panel_var}")
    console.print(f"  Number of obs   = {len(df):>10,}")
    console.print(f"  Number of groups= {n_panels:>10,}")
    console.print(f"  R-sq within     = {results.rsquared_within:.4f}" if hasattr(results, 'rsquared_within') else "")
    console.print(f"  R-sq between    = {results.rsquared_between:.4f}" if hasattr(results, 'rsquared_between') else "")
    console.print(f"  R-sq overall    = {results.rsquared_overall:.4f}" if hasattr(results, 'rsquared_overall') else "")
    if n_dropped > 0:
        console.print(f"  [dim]({n_dropped} obs dropped due to missing)[/dim]")
    console.print()

    # Coefficient table
    _display_panel_results(results, depvar, console)

    # Store results
    state.e_results = {
        "cmd": "xtreg",
        "depvar": depvar,
        "N": len(df),
        "N_g": n_panels,
        "model": "fe" if use_fe else "re",
    }


def _xtreg_statsmodels_fallback(df, depvar, indepvars, state, use_fe, robust, n_dropped, console):
    """Panel FE via entity dummies with statsmodels (no linearmodels).

    Uses _check_statsmodels(), so on platforms without statsmodels (Android)
    this transparently runs on Rln's NumPy/SciPy OLS fallback instead of
    crashing — making `xtreg, fe` available on mobile.
    """
    from commands.estimation import _check_statsmodels
    sm = _check_statsmodels()

    panel_var = state._panel_var

    if use_fe:
        # Create entity dummies
        dummies = pd.get_dummies(df[panel_var], prefix=panel_var, drop_first=True, dtype=float)
        X = pd.concat([df[indepvars].astype(float), dummies], axis=1)
        X = sm.add_constant(X)
        model = sm.OLS(df[depvar].astype(float), X)
        cov = "HC1" if robust else None
        results = model.fit(cov_type=cov) if cov else model.fit()

        n_panels = df[panel_var].nunique()
        console.print(f"\n[bold]Fixed-effects regression (LSDV, statsmodels fallback)[/bold]")
        console.print(f"  [dim]Install linearmodels for proper panel estimators: ssc install linearmodels[/dim]")
        console.print(f"  Number of obs    = {len(df):>10,}")
        console.print(f"  Number of groups = {n_panels:>10,}")
        console.print(f"  R-squared        = {results.rsquared:.4f}")
        console.print()

        # Show only non-dummy coefficients
        table = Table(show_lines=False)
        table.add_column(depvar, style="bold", min_width=15)
        table.add_column("Coef.", justify="right")
        table.add_column("Std. err.", justify="right")
        table.add_column("t", justify="right")
        table.add_column("P>|t|", justify="right")

        for var_name in ["const"] + indepvars:
            if var_name in results.params.index:
                display = "_cons" if var_name == "const" else var_name
                table.add_row(display,
                              f"{results.params[var_name]:.6f}",
                              f"{results.bse[var_name]:.6f}",
                              f"{results.tvalues[var_name]:.2f}",
                              f"{results.pvalues[var_name]:.3f}")

        console.print(table)
        console.print(f"  [dim]({n_panels - 1} entity dummies not shown)[/dim]")
    else:
        console.print("[red]Random effects requires linearmodels: ssc install linearmodels[/red]")


def _display_panel_results(results, depvar, console):
    """Display linearmodels panel results."""
    table = Table(show_lines=False)
    table.add_column(depvar, style="bold", min_width=15)
    table.add_column("Coef.", justify="right", min_width=12)
    table.add_column("Std. err.", justify="right", min_width=10)
    table.add_column("t", justify="right", min_width=8)
    table.add_column("P>|t|", justify="right", min_width=8)
    table.add_column("[95% CI", justify="right")
    table.add_column("95% CI]", justify="right")

    params = results.params
    se = results.std_errors
    tvals = results.tstats
    pvals = results.pvalues
    ci = results.conf_int()

    for var_name in params.index:
        table.add_row(
            var_name,
            f"{params[var_name]:.6f}",
            f"{se[var_name]:.6f}",
            f"{tvals[var_name]:.2f}",
            f"{pvals[var_name]:.3f}",
            f"{ci.loc[var_name, 'lower']:.6f}",
            f"{ci.loc[var_name, 'upper']:.6f}",
        )

    console.print(table)
    console.print()


# ──────────────────────────────────────────────
#  didregress — Difference-in-Differences
# ──────────────────────────────────────────────

def cmd_didregress(rest: str, state: AppState, console: Console):
    """
    didregress (depvar) (treatment) [, group(var) time(var) method(...) ...]

    Difference-in-differences estimation.

    Methods (all require diff-diff except twfe):
      twfe        — Two-way fixed effects (default, pure statsmodels, no dependency)
      did         — Basic 2x2 DiD                             (diff_diff.DifferenceInDifferences)
      cs          — Callaway & Sant'Anna (2021), staggered    (diff_diff.CallawaySantAnna)
      sa          — Sun & Abraham (2021), staggered           (diff_diff.SunAbraham)
      bjs         — Borusyak-Jaravel-Spiess (2024) imputation (diff_diff.ImputationDiD)
      gardner     — Gardner (2022) two-stage DiD              (diff_diff.TwoStageDiD)
      stacked     — Stacked DiD, Wing et al. (2024)           (diff_diff.StackedDiD)
      sdid        — Synthetic DiD                             (diff_diff.SyntheticDiD)
      eventstudy  — Full event study, pre + post effects       (diff_diff.MultiPeriodDiD)
      bacon       — Goodman-Bacon decomposition               (diff_diff.BaconDecomposition)
      honest      — Rambachan-Roth (2023) sensitivity on event-study
                    (diff_diff.HonestDiD — requires method(eventstudy) results)

    Extra options:
      first_treat(var)  — for staggered estimators (cs/sa/bjs/gardner/stacked/bacon):
                          column giving each unit's first-treatment period
                          (0 for never-treated). If omitted, Rln derives it
                          from the treatment indicator.
      post_periods(list) — for eventstudy / sdid: post-treatment period values
      reference_period(v) — for eventstudy: the omitted pre-period (default: last pre)
      aggregate(simple|group|event_study|dynamic) — for cs/bjs/gardner/stacked
      covariates(v1 v2 ...)  — for estimators that support covariate adjustment
      M(value)                — for honest: magnitude M of allowed trend violation

    See: https://pypi.org/project/diff-diff/ for methodology references.
    """
    state.require_data()
    parsed = parse_command_line(rest)

    # Parse (depvar) (treatment) syntax or simple varlist
    raw = parsed["raw"].split(",")[0].strip()

    # Try to extract parenthesized groups
    paren_groups = re.findall(r'\(([^)]+)\)', raw)
    if len(paren_groups) >= 2:
        depvar = paren_groups[0].strip()
        treatment = paren_groups[1].strip()
    elif len(parsed["varlist"]) >= 2:
        depvar = parsed["varlist"][0]
        treatment = parsed["varlist"][1]
    else:
        console.print("[red]Syntax: didregress (depvar) (treatment) [, group(var) time(var)][/red]")
        return

    group_var = parsed["options"].get("group") or parsed["options"].get("g")
    time_var = parsed["options"].get("time") or parsed["options"].get("t")
    method = parsed["options"].get("method", "twfe").lower()

    # Use panel vars as defaults
    if not group_var and hasattr(state, '_panel_var') and state._panel_var:
        group_var = state._panel_var
    if not time_var and hasattr(state, '_time_var') and state._time_var:
        time_var = state._time_var

    # Validate
    for v in [depvar, treatment]:
        if v not in state.data.columns:
            console.print(f"[red]Variable '{v}' not found[/red]")
            return
    if group_var and group_var not in state.data.columns:
        console.print(f"[red]Group variable '{group_var}' not found[/red]")
        return
    if time_var and time_var not in state.data.columns:
        console.print(f"[red]Time variable '{time_var}' not found[/red]")
        return

    if method != "twfe":
        try:
            _did_diffdiff(state.data, depvar, treatment, group_var, time_var, method, parsed, state, console)
            return
        except ImportError as e:
            console.print(f"[yellow]diff-diff not installed: {e}[/yellow]")
            console.print("[yellow]Falling back to TWFE.[/yellow]")
            console.print("[dim]Install with: ssc install diff-diff[/dim]")
        except Exception as e:
            console.print(f"[red]diff-diff method '{method}' failed: {e}[/red]")
            console.print("[yellow]Falling back to TWFE.[/yellow]")

    # TWFE fallback
    _did_twfe(state.data, depvar, treatment, group_var, time_var, parsed, state, console)


def _did_twfe(df, depvar, treatment, group_var, time_var, parsed, state, console):
    """Two-way fixed effects DiD via statsmodels.

    Routed through _check_statsmodels() so method(twfe) — the one DiD estimator
    that needs no diff-diff package — also works on Android via the NumPy/SciPy
    OLS fallback.
    """
    from commands.estimation import _check_statsmodels
    sm = _check_statsmodels()

    # Build model: Y = treatment + group_FE + time_FE
    model_df = df[[depvar, treatment]].copy().dropna()
    X_parts = [model_df[treatment].astype(float)]

    if group_var:
        group_dummies = pd.get_dummies(df.loc[model_df.index, group_var], prefix="g", drop_first=True, dtype=float)
        X_parts.append(group_dummies)
    if time_var:
        time_dummies = pd.get_dummies(df.loc[model_df.index, time_var], prefix="t", drop_first=True, dtype=float)
        X_parts.append(time_dummies)

    X = pd.concat(X_parts, axis=1)
    X = sm.add_constant(X)
    y = model_df[depvar].astype(float)

    robust = "robust" in parsed["options"]
    model = sm.OLS(y, X)
    results = model.fit(cov_type="HC1") if robust else model.fit()

    att = results.params[treatment]
    se = results.bse[treatment]
    t = results.tvalues[treatment]
    p = results.pvalues[treatment]
    ci = results.conf_int().loc[treatment]

    console.print(f"\n[bold]Difference-in-Differences (TWFE)[/bold]")
    console.print(f"  Outcome:   {depvar}")
    console.print(f"  Treatment: {treatment}")
    if group_var:
        console.print(f"  Group FE:  {group_var} ({df[group_var].nunique()} groups)")
    if time_var:
        console.print(f"  Time FE:   {time_var} ({df[time_var].nunique()} periods)")
    console.print(f"  N = {len(model_df):,}")
    console.print()

    table = Table(title="DiD Estimate")
    table.add_column("", style="bold")
    table.add_column("ATT", justify="right")
    table.add_column("Std. err.", justify="right")
    table.add_column("t", justify="right")
    table.add_column("P>|t|", justify="right")
    table.add_column("[95% CI]", justify="right")

    table.add_row(treatment, f"{att:.6f}", f"{se:.6f}", f"{t:.2f}", f"{p:.4f}",
                  f"[{ci.iloc[0]:.4f}, {ci.iloc[1]:.4f}]")
    console.print(table)

    state.e_results = {
        "cmd": "didregress",
        "method": "twfe",
        "att": att, "se": se, "t": t, "p": p,
        "N": len(model_df),
        "depvar": depvar, "treatment": treatment,
    }
    state.r_results = {"att": att, "se": se, "N": len(model_df)}


def _did_diffdiff(df, depvar, treatment, group_var, time_var, method, parsed, state, console):
    """
    DiD using the diff-diff library (>=3.0).

    Supports every major estimator in diff-diff's public API; dispatches on `method`.
    On diff-diff <3.0 the class names and kwargs are different — we raise ImportError
    with a clear message so the caller can fall back to TWFE.
    """
    try:
        import diff_diff as dd
    except ImportError as e:
        raise ImportError("diff-diff not installed") from e

    method = (method or "did").lower()
    opts = parsed["options"]
    covariates_str = opts.get("covariates") or opts.get("covars") or opts.get("controls")
    covariates = [c.strip() for c in re.split(r"[,\s]+", covariates_str) if c.strip()] \
        if covariates_str else None
    aggregate = opts.get("aggregate") or opts.get("agg")
    first_treat = opts.get("first_treat") or opts.get("firsttreat") or opts.get("ft")
    post_periods_str = opts.get("post_periods") or opts.get("postperiods")
    reference_period = opts.get("reference_period") or opts.get("ref")
    M = opts.get("m")

    # Parse post_periods "3 4 5" or "3,4,5" into list of ints (fallback: floats, fallback: strings)
    post_periods = None
    if post_periods_str:
        raw_vals = [v.strip() for v in re.split(r"[,\s]+", post_periods_str) if v.strip()]
        parsed_vals = []
        for v in raw_vals:
            try:
                parsed_vals.append(int(v))
            except ValueError:
                try:
                    parsed_vals.append(float(v))
                except ValueError:
                    parsed_vals.append(v)
        post_periods = parsed_vals
    if reference_period is not None:
        try:
            reference_period = int(reference_period)
        except (TypeError, ValueError):
            try:
                reference_period = float(reference_period)
            except (TypeError, ValueError):
                pass

    # Derive a first_treat column on the fly if the user didn't supply one.
    # This lets staggered estimators work with just a binary treatment column.
    def _ensure_first_treat_col():
        nonlocal first_treat
        if first_treat and first_treat in df.columns:
            return first_treat
        if group_var is None or time_var is None:
            raise ValueError(
                "Staggered estimators need first_treat() (or group() + time() + a "
                "binary treatment to derive it from).")
        derived = "_ft_" + treatment
        if derived not in df.columns:
            treat_only = df[df[treatment] == 1]
            first_treat_map = treat_only.groupby(group_var)[time_var].min()
            df[derived] = df[group_var].map(first_treat_map).fillna(0)
            # Cast to match time_var dtype where feasible
            try:
                df[derived] = df[derived].astype(df[time_var].dtype)
            except Exception:
                pass
        first_treat = derived
        return derived

    # Derive a binary post-treatment indicator from the time variable.
    # Needed by method(did) — DifferenceInDifferences requires a 0/1 post
    # column, not year values. If `time_var` is already binary, we use it
    # directly; otherwise we build one from reference_period or post_periods,
    # and if neither is given, from the median year as the split point.
    def _ensure_post_binary():
        if time_var and time_var in df.columns:
            uniq = df[time_var].dropna().unique()
            if len(uniq) == 2 and set(uniq).issubset({0, 1, 0.0, 1.0}):
                return time_var  # already binary
        if time_var is None:
            raise ValueError("method(did) needs either a binary time() column "
                             "or time() plus post_periods()/reference_period() to derive one.")
        derived = "_post_" + time_var
        if derived not in df.columns:
            if post_periods:
                df[derived] = df[time_var].isin(post_periods).astype(int)
            elif reference_period is not None:
                df[derived] = (df[time_var] > reference_period).astype(int)
            else:
                # Last-resort fallback: split at the median period so basic 2x2
                # returns SOMETHING rather than crashing. Warn the user.
                split = df[time_var].median()
                df[derived] = (df[time_var] > split).astype(int)
                console.print(
                    f"[yellow]Note: method(did) derived a binary post indicator by splitting "
                    f"{time_var} at median ({split}). For precise timing, pass "
                    f"post_periods(v1 v2 ...) or reference_period(v).[/yellow]")
        return derived

    # Derive an ever-treated (absorbing) indicator: 1 for every row of a unit
    # that is treated in ANY period, 0 otherwise. Needed by:
    #   - MultiPeriodDiD's event study (expects time-invariant treatment indicator)
    #   - SyntheticDiD (expects block / absorbing treatment)
    def _ensure_ever_treated():
        if group_var is None or group_var not in df.columns:
            raise ValueError(
                "This method needs group() to derive the ever-treated indicator.")
        ever_col = "_ever_" + treatment
        if ever_col not in df.columns:
            # max() over (unit) of the binary treat indicator
            df[ever_col] = df.groupby(group_var)[treatment].transform("max").astype(int)
        return ever_col

    console.print(f"[dim]Using diff-diff {getattr(dd, '__version__', '?')} — method={method}[/dim]")

    if method in ("did", "diff", "basic"):
        # diff-diff's DifferenceInDifferences requires a binary post indicator.
        post_col = _ensure_post_binary()
        est = dd.DifferenceInDifferences()
        res = est.fit(df, outcome=depvar, treatment=treatment, time=post_col,
                      covariates=covariates)
        _print_did_result(res, "Basic 2x2 DiD", depvar, treatment, console)
        _store_did_eresults(state, res, method)

    elif method in ("cs", "callaway", "santanna", "callaway-santanna"):
        ft = _ensure_first_treat_col()
        est = dd.CallawaySantAnna()
        res = est.fit(df, outcome=depvar, unit=group_var, time=time_var,
                      first_treat=ft, covariates=covariates,
                      aggregate=aggregate or "simple")
        _print_staggered_result(res, "Callaway-Sant'Anna", depvar, treatment, console)
        _store_did_eresults(state, res, method)

    elif method in ("sa", "sun", "abraham", "sun-abraham"):
        ft = _ensure_first_treat_col()
        est = dd.SunAbraham()
        res = est.fit(df, outcome=depvar, unit=group_var, time=time_var,
                      first_treat=ft, covariates=covariates)
        _print_staggered_result(res, "Sun-Abraham interaction-weighted", depvar, treatment, console)
        _store_did_eresults(state, res, method)

    elif method in ("bjs", "imputation", "borusyak", "did_imputation",
                     "did-imputation"):
        ft = _ensure_first_treat_col()
        est = dd.ImputationDiD()
        res = est.fit(df, outcome=depvar, unit=group_var, time=time_var,
                      first_treat=ft, covariates=covariates,
                      aggregate=aggregate or "simple")
        _print_staggered_result(res, "Borusyak-Jaravel-Spiess imputation DiD",
                                depvar, treatment, console)
        _store_did_eresults(state, res, method)

    elif method in ("gardner", "two_stage", "twostage", "two-stage"):
        ft = _ensure_first_treat_col()
        est = dd.TwoStageDiD()
        res = est.fit(df, outcome=depvar, unit=group_var, time=time_var,
                      first_treat=ft, covariates=covariates,
                      aggregate=aggregate or "simple")
        _print_staggered_result(res, "Gardner (2022) Two-Stage DiD",
                                depvar, treatment, console)
        _store_did_eresults(state, res, method)

    elif method in ("stacked", "wing", "stack"):
        ft = _ensure_first_treat_col()
        est = dd.StackedDiD()
        res = est.fit(df, outcome=depvar, unit=group_var, time=time_var,
                      first_treat=ft, aggregate=aggregate or "event_study")
        _print_staggered_result(res, "Stacked DiD (Wing et al. 2024)",
                                depvar, treatment, console)
        _store_did_eresults(state, res, method)

    elif method in ("sdid", "synthetic"):
        if not post_periods:
            raise ValueError("sdid requires post_periods(v1 v2 ...) in options")
        # SyntheticDiD requires BLOCK (absorbing) treatment: treated=1 for every
        # row of an ever-treated unit. Build one if the provided treatment is
        # per-row instead.
        sdid_treat = treatment
        if group_var and group_var in df.columns:
            per_row_varies = (df.groupby(group_var)[treatment].nunique() > 1).any()
            if per_row_varies:
                sdid_treat = _ensure_ever_treated()
                console.print(f"[dim]SDID: using absorbing '{sdid_treat}' indicator "
                              f"derived from '{treatment}'.[/dim]")
        est = dd.SyntheticDiD()
        res = est.fit(df, outcome=depvar, treatment=sdid_treat, unit=group_var,
                      time=time_var, post_periods=post_periods, covariates=covariates)
        _print_did_result(res, "Synthetic DiD", depvar, treatment, console)
        # SDID sometimes returns SE=0 when placebo variance can't be computed
        # (too few control units). Surface this visibly because a SE of 0
        # silently produces zero-width confidence intervals downstream.
        sdid_se = getattr(res, "se", None)
        if sdid_se is not None and float(sdid_se) == 0.0:
            console.print(
                "[yellow]Note: SDID returned SE=0 (placebo variance unavailable). "
                "For valid inference re-fit with bootstrap variance — see the "
                "diff-diff documentation for variance_method='bootstrap'.[/yellow]"
            )
        _store_did_eresults(state, res, method)

    elif method in ("eventstudy", "event_study", "event-study", "multiperiod", "dynamic"):
        if not post_periods:
            raise ValueError(
                "eventstudy requires post_periods(v1 v2 ...). "
                "Optionally pass reference_period(v) too.")
        # MultiPeriodDiD expects a time-INVARIANT ever-treated indicator
        # (1 for every row of a unit that is ever treated). Using per-row
        # `treated` here causes pre-period coefficients to be unidentified
        # and the whole fit to collapse to NaN.
        es_treat = treatment
        if group_var and group_var in df.columns:
            per_row_varies = (df.groupby(group_var)[treatment].nunique() > 1).any()
            if per_row_varies:
                es_treat = _ensure_ever_treated()
                console.print(f"[dim]Event study: using time-invariant '{es_treat}' indicator "
                              f"derived from '{treatment}'.[/dim]")
        est = dd.MultiPeriodDiD()
        res = est.fit(df, outcome=depvar, treatment=es_treat, time=time_var,
                      post_periods=post_periods, covariates=covariates,
                      reference_period=reference_period, unit=group_var)
        _print_eventstudy_result(res, depvar, treatment, console)
        # Stash the event-study object on state so a subsequent `honest` call can use it.
        state._last_eventstudy = res
        _store_did_eresults(state, res, method)

    elif method in ("bacon", "goodman-bacon", "goodmanbacon"):
        ft = _ensure_first_treat_col()
        est = dd.BaconDecomposition()
        res = est.fit(df, outcome=depvar, unit=group_var, time=time_var, first_treat=ft)
        _print_bacon_result(res, depvar, treatment, console)
        _store_did_eresults(state, res, method)

    elif method in ("honest", "rambachan", "rambachan-roth"):
        ev = getattr(state, "_last_eventstudy", None)
        if ev is None:
            raise ValueError(
                "method(honest) needs a prior event-study fit. Run "
                "'didregress ..., method(eventstudy) post_periods(...)' first.")
        M_val = float(M) if M is not None else 1.0
        est = dd.HonestDiD(method="relative_magnitude", M=M_val)
        res = est.fit(ev)
        _print_honest_result(res, M_val, console)
        _store_did_eresults(state, res, method)

    else:
        raise ValueError(
            f"Unknown method: {method!r}. "
            "Choose from: twfe, did, cs, sa, bjs, gardner, stacked, sdid, "
            "eventstudy, bacon, honest.")


# ── Printers & result storage ─────────────────────────────────────────

def _print_did_result(res, title, depvar, treatment, console):
    """Pretty-print a scalar-ATT result (DiD, SDID, ...)."""
    att = getattr(res, "att", None)
    se = getattr(res, "se", None)
    p = getattr(res, "p_value", None)
    ci = getattr(res, "conf_int", None)
    n = getattr(res, "n_obs", None)

    console.print(f"\n[bold]{title}[/bold]")
    console.print(f"  Outcome:   {depvar}")
    console.print(f"  Treatment: {treatment}")
    if n is not None:
        console.print(f"  N = {n:,}")
    console.print()

    table = Table(title="DiD Estimate")
    table.add_column("", style="bold")
    table.add_column("ATT", justify="right")
    table.add_column("Std. err.", justify="right")
    table.add_column("P>|t|", justify="right")
    table.add_column("[95% CI]", justify="right")
    ci_lo, ci_hi = (None, None)
    if ci is not None:
        try:
            ci_lo, ci_hi = ci[0], ci[1]
        except Exception:
            pass
    ci_txt = f"[{ci_lo:.4f}, {ci_hi:.4f}]" if ci_lo is not None else "[-, -]"
    table.add_row(treatment,
                  f"{att:.6f}" if att is not None else "-",
                  f"{se:.6f}" if se is not None else "-",
                  f"{p:.4f}"  if p  is not None else "-",
                  ci_txt)
    console.print(table)


def _print_staggered_result(res, title, depvar, treatment, console):
    """Pretty-print staggered DiD results (CS, SA, BJS, Gardner, Stacked)."""
    # Attribute names vary slightly: prefer overall_att then att.
    att = getattr(res, "overall_att", None)
    if att is None:
        att = getattr(res, "att", None)
    se = getattr(res, "overall_se", None) or getattr(res, "se", None)
    p = getattr(res, "overall_p_value", None) or getattr(res, "p_value", None)
    n = getattr(res, "n_obs", None)

    console.print(f"\n[bold]{title}[/bold]")
    console.print(f"  Outcome:   {depvar}")
    console.print(f"  Treatment: {treatment}")
    if n is not None:
        console.print(f"  N = {n:,}")
    console.print()

    table = Table(title="Overall ATT")
    table.add_column("", style="bold")
    table.add_column("ATT", justify="right")
    table.add_column("Std. err.", justify="right")
    table.add_column("P>|t|", justify="right")
    table.add_row(treatment,
                  f"{att:.6f}" if att is not None else "-",
                  f"{se:.6f}" if se is not None else "-",
                  f"{p:.4f}"  if p  is not None else "-")
    console.print(table)

    # Event-study breakdown if available
    es = getattr(res, "event_study_effects", None)
    if es:
        console.print("\n[dim]Event-study (relative time):[/dim]")
        es_table = Table()
        es_table.add_column("Rel. period", justify="right")
        es_table.add_column("Effect", justify="right")
        es_table.add_column("SE", justify="right")
        es_table.add_column("P>|t|", justify="right")
        for rt, eff in sorted(es.items(), key=lambda x: (x[0] is None, x[0])):
            # eff may be a dict or a PeriodEffect-like object
            eff_val = _get(eff, "effect")
            se_val = _get(eff, "se")
            p_val = _get(eff, "p_value")
            es_table.add_row(
                str(rt),
                f"{eff_val:.4f}" if isinstance(eff_val, (int, float)) else "-",
                f"{se_val:.4f}"  if isinstance(se_val,  (int, float)) else "-",
                f"{p_val:.4f}"   if isinstance(p_val,   (int, float)) else "-",
            )
        console.print(es_table)


def _print_eventstudy_result(res, depvar, treatment, console):
    """Pretty-print MultiPeriodDiD: average ATT + period-by-period effects."""
    avg_att = getattr(res, "avg_att", None)
    avg_se = getattr(res, "avg_se", None)
    avg_p = getattr(res, "avg_p_value", None)
    n = getattr(res, "n_obs", None)

    console.print(f"\n[bold]Multi-Period Event Study[/bold]")
    console.print(f"  Outcome:   {depvar}")
    console.print(f"  Treatment: {treatment}")
    if n is not None:
        console.print(f"  N = {n:,}")
    console.print()

    table = Table(title="Average ATT (post-treatment)")
    table.add_column("", style="bold")
    table.add_column("ATT", justify="right")
    table.add_column("Std. err.", justify="right")
    table.add_column("P>|t|", justify="right")
    table.add_row(treatment,
                  f"{avg_att:.6f}" if avg_att is not None else "-",
                  f"{avg_se:.6f}"  if avg_se  is not None else "-",
                  f"{avg_p:.4f}"   if avg_p   is not None else "-")
    console.print(table)

    period_effects = getattr(res, "period_effects", None)
    pre_periods = getattr(res, "pre_periods", None) or []
    post_periods = getattr(res, "post_periods", None) or []
    ref_period = getattr(res, "reference_period", None)

    if period_effects:
        ev = Table(title="Period-by-period effects")
        ev.add_column("Period", justify="right")
        ev.add_column("Phase")
        ev.add_column("Effect", justify="right")
        ev.add_column("SE", justify="right")
        ev.add_column("P>|t|", justify="right")
        for period in sorted(period_effects.keys(), key=lambda x: (x is None, x)):
            pe = period_effects[period]
            phase = ("pre" if period in pre_periods else
                     "post" if period in post_periods else "—")
            ev.add_row(
                str(period),
                phase,
                f"{pe.effect:.4f}",
                f"{pe.se:.4f}",
                f"{pe.p_value:.4f}",
            )
        console.print(ev)
        if ref_period is not None:
            console.print(f"[dim]Reference period: {ref_period} (coefficient = 0 by construction)[/dim]")


def _print_bacon_result(res, depvar, treatment, console):
    """Pretty-print Goodman-Bacon decomposition of TWFE into 2x2 comparisons."""
    console.print(f"\n[bold]Goodman-Bacon decomposition[/bold]")
    console.print(f"  Outcome:   {depvar}")
    console.print(f"  Treatment: {treatment}")
    twfe = getattr(res, "twfe_estimate", None)
    if twfe is not None:
        console.print(f"  TWFE (pooled) estimate = {twfe:.6f}")
    console.print()

    table = Table(title="Comparison weights")
    table.add_column("Comparison type")
    table.add_column("Weight", justify="right")
    table.add_column("Weighted ATT", justify="right")
    for label, w_attr, att_attr in [
        ("Treated vs never-treated", "total_weight_treated_vs_never",
                                      "weighted_avg_treated_vs_never"),
        ("Earlier vs later-treated", "total_weight_earlier_vs_later",
                                      "weighted_avg_earlier_vs_later"),
        ("Later vs earlier-treated", "total_weight_later_vs_earlier",
                                      "weighted_avg_later_vs_earlier"),
    ]:
        w = getattr(res, w_attr, None)
        a = getattr(res, att_attr, None)
        table.add_row(label,
                      f"{w:.4f}" if w is not None else "-",
                      f"{a:.4f}" if a is not None else "-")
    console.print(table)
    console.print("[dim]High 'later vs earlier' weight is a red flag for TWFE bias.[/dim]")


def _print_honest_result(res, M, console):
    """Pretty-print Rambachan-Roth honest DiD sensitivity bounds."""
    console.print(f"\n[bold]Honest DiD sensitivity[/bold] [dim](Rambachan & Roth 2023)[/dim]")
    console.print(f"  M = {M}  (bound on post-trend violations, in units of pre-trend max)")
    orig = getattr(res, "original_estimate", None)
    if orig is not None:
        console.print(f"  Original event-study estimate: {orig:.4f}")
    lb = getattr(res, "ci_lb", None)
    ub = getattr(res, "ci_ub", None)
    sig = getattr(res, "is_significant", None)
    if lb is not None and ub is not None:
        console.print(f"  Robust 95% CI: [{lb:.4f}, {ub:.4f}]")
    if sig is not None:
        verdict = "[green]holds[/green]" if sig else "[yellow]fails[/yellow]"
        console.print(f"  Significance robust to violations: {verdict}")


def _get(obj, attr, default=None):
    """Get attr from obj whether obj is an object or a dict."""
    if isinstance(obj, dict):
        return obj.get(attr, default)
    return getattr(obj, attr, default)


def _store_did_eresults(state, res, method):
    """Persist the most common scalar fields into state.e_results for later use.

    Also stashes the per-period coefficient vector when the method produces
    one (event study, CS with aggregate=event_study, Gardner). This gives
    `coefplot` the structure it needs to render a forest plot.
    """
    e = {"cmd": "didregress", "method": method}
    for k in ("att", "overall_att", "se", "overall_se", "p_value", "overall_p_value",
              "n_obs", "avg_att", "avg_se", "avg_p_value", "twfe_estimate",
              "ci_lb", "ci_ub"):
        v = getattr(res, k, None)
        if v is not None:
            e[k] = v

    # Event-study / dynamic effects: collect into a tidy list of dicts so
    # `coefplot` can iterate without knowing diff_diff internals.
    period_effects = getattr(res, "period_effects", None)
    if period_effects:
        pre_periods  = getattr(res, "pre_periods",  None) or []
        post_periods = getattr(res, "post_periods", None) or []
        ref_period   = getattr(res, "reference_period", None)
        coefs = []
        for period in sorted(period_effects.keys(), key=lambda x: (x is None, x)):
            pe = period_effects[period]
            effect = getattr(pe, "effect", None)
            se     = getattr(pe, "se",     None)
            p      = getattr(pe, "p_value", None)
            if effect is None:
                continue
            # 95% CI — prefer stored bounds if diff_diff provides them,
            # otherwise build from se.
            ci_lb = getattr(pe, "ci_lb", None)
            ci_ub = getattr(pe, "ci_ub", None)
            if (ci_lb is None or ci_ub is None) and se is not None:
                ci_lb = effect - 1.96 * se
                ci_ub = effect + 1.96 * se
            coefs.append({
                "period":  period,
                "effect":  float(effect),
                "se":      float(se) if se is not None else None,
                "p":       float(p)  if p  is not None else None,
                "ci_lb":   float(ci_lb) if ci_lb is not None else None,
                "ci_ub":   float(ci_ub) if ci_ub is not None else None,
                "phase":   ("pre"  if period in pre_periods  else
                            "post" if period in post_periods else "ref"),
            })
        e["coefficients"] = coefs
        e["reference_period"] = ref_period

    # CS / Gardner / Stacked sometimes expose dynamic effects via an
    # `event_study_effects` attribute (a dict of period -> (effect, se)).
    es = getattr(res, "event_study_effects", None)
    if es and "coefficients" not in e:
        coefs = []
        for period, val in sorted(es.items(), key=lambda kv: (kv[0] is None, kv[0])):
            if isinstance(val, dict):
                effect = val.get("effect"); se = val.get("se"); p = val.get("p_value")
            elif hasattr(val, "effect"):
                effect = val.effect; se = val.se; p = getattr(val, "p_value", None)
            else:
                continue
            if effect is None:
                continue
            ci_lb = effect - 1.96 * se if se is not None else None
            ci_ub = effect + 1.96 * se if se is not None else None
            coefs.append({
                "period":  period,
                "effect":  float(effect),
                "se":      float(se) if se is not None else None,
                "p":       float(p)  if p  is not None else None,
                "ci_lb":   float(ci_lb) if ci_lb is not None else None,
                "ci_ub":   float(ci_ub) if ci_ub is not None else None,
                "phase":   "pre" if (isinstance(period, (int, float)) and period < 0)
                                 else ("ref" if period == 0 else "post"),
            })
        if coefs:
            e["coefficients"] = coefs

    state.e_results = e
    # r() convenience
    att = e.get("att") or e.get("overall_att") or e.get("avg_att")
    se = e.get("se")  or e.get("overall_se")  or e.get("avg_se")
    N = e.get("n_obs")
    state.r_results = {"att": att, "se": se, "N": N}



# ──────────────────────────────────────────────
#  lincom — linear combination of coefficients
# ──────────────────────────────────────────────

def cmd_lincom(rest: str, state: AppState, console: Console):
    """
    lincom expression
    
    Compute a linear combination of estimated coefficients.
    Example: lincom education + 2*age
             lincom education - age
    """
    if not state.e_results or "predict_model" not in state.e_results:
        console.print("[red]No estimation results. Run regress first.[/red]")
        return

    model = state.e_results["predict_model"]
    expr = rest.strip()

    if not expr:
        console.print("[red]Syntax: lincom expression[/red]")
        return

    try:
        # Parse linear combination
        t_test = model.t_test(expr)
        coef = float(t_test.effect[0])
        se = float(t_test.sd[0])
        t = float(t_test.tvalue[0])
        p = float(t_test.pvalue[0])
        ci = t_test.conf_int(alpha=0.05)[0]

        console.print(f"\n  [bold]( 1)  {expr}[/bold]")
        console.print()

        table = Table(show_lines=False)
        table.add_column("", style="bold", min_width=15)
        table.add_column("Coef.", justify="right")
        table.add_column("Std. err.", justify="right")
        table.add_column("t", justify="right")
        table.add_column("P>|t|", justify="right")
        table.add_column("[95% CI]", justify="right")

        table.add_row("(1)", f"{coef:.6f}", f"{se:.6f}", f"{t:.2f}", f"{p:.4f}",
                      f"[{ci[0]:.4f}, {ci[1]:.4f}]")
        console.print(table)
        console.print()

        state.r_results = {"estimate": coef, "se": se, "t": t, "p": p}

    except Exception as e:
        console.print(f"[red]lincom failed: {e}[/red]")


# ──────────────────────────────────────────────
#  margins — marginal effects / predictive margins
# ──────────────────────────────────────────────

def cmd_margins(rest: str, state: AppState, console: Console):
    """
    margins [, dydx(*) dydx(varlist) at(var=val) atmeans]
    
    Compute marginal effects or predictive margins after estimation.
    
    Examples:
      margins                        (predictive margins at means)
      margins, dydx(*)               (average marginal effects, all vars)
      margins, dydx(education age)   (AME for specific vars)
      margins, at(education=12)      (predicted value at education=12)
    """
    if not state.e_results or "predict_model" not in state.e_results:
        console.print("[red]No estimation results. Run regress first.[/red]")
        return

    state.require_data()
    parsed = parse_command_line(rest)
    model = state.e_results["predict_model"]

    dydx = parsed["options"].get("dydx")
    at_spec = parsed["options"].get("at")
    atmeans = "atmeans" in parsed["options"]

    if dydx:
        # Marginal effects (for linear model = coefficients)
        if dydx == "*":
            vars_to_show = [v for v in model.params.index if v != "const"]
        else:
            vars_to_show = dydx.split()

        console.print(f"\n[bold]Average marginal effects[/bold]")
        console.print(f"  Model: {state.e_results.get('cmd', 'regress')}")
        console.print(f"  N = {state.e_results.get('N', '?')}")
        console.print()

        table = Table(show_lines=False)
        table.add_column("Variable", style="bold", min_width=15)
        table.add_column("dy/dx", justify="right")
        table.add_column("Std. err.", justify="right")
        table.add_column("z", justify="right")
        table.add_column("P>|z|", justify="right")
        table.add_column("[95% CI]", justify="right")

        ci = model.conf_int()
        for v in vars_to_show:
            if v in model.params.index:
                table.add_row(v,
                              f"{model.params[v]:.6f}",
                              f"{model.bse[v]:.6f}",
                              f"{model.tvalues[v]:.2f}",
                              f"{model.pvalues[v]:.3f}",
                              f"[{ci.loc[v, 0]:.4f}, {ci.loc[v, 1]:.4f}]")
            else:
                console.print(f"  [yellow]{v} not in model[/yellow]")

        console.print(table)

    elif at_spec:
        # Predictive margins at specific values
        at_pairs = {}
        for pair in at_spec.split():
            if "=" in pair:
                var, val = pair.split("=", 1)
                at_pairs[var.strip()] = float(val.strip())

        # Create prediction point
        X_cols = state.e_results.get("predict_X_cols", [])
        X_mean = state.data[X_cols].mean() if X_cols else pd.Series()

        for var, val in at_pairs.items():
            if var in X_mean.index:
                X_mean[var] = val

        if "const" in model.params.index and "const" not in X_mean.index:
            X_mean["const"] = 1.0

        pred = model.predict(X_mean.values.reshape(1, -1))[0]
        console.print(f"\n[bold]Predictive margin[/bold]")
        for var, val in at_pairs.items():
            console.print(f"  at({var}={val})")
        console.print(f"\n  Margin = {pred:.6f}")
        console.print()

    else:
        # Default: predictive margin at means
        X_cols = state.e_results.get("predict_X_cols", [])
        if not X_cols:
            console.print("[red]No prediction model available.[/red]")
            return

        X_mean = state.data[X_cols].mean()
        if "const" in model.params.index and "const" not in X_mean.index:
            X_mean["const"] = 1.0

        pred = model.predict(X_mean.values.reshape(1, -1))[0]

        console.print(f"\n[bold]Predictive margins (at means)[/bold]")
        console.print(f"  Model: {state.e_results.get('cmd', 'regress')}")
        console.print(f"  N = {state.e_results.get('N', '?')}")
        console.print(f"\n  Margin = {pred:.6f}")
        console.print()

        state.r_results = {"margin": pred}
