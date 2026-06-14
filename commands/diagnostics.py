"""
Post-estimation diagnostic tests:

  vif                    — Variance Inflation Factor (multicollinearity)
  estat hettest          — Breusch-Pagan test for heteroskedasticity
  estat bgodfrey         — Breusch-Godfrey test for serial correlation
  estat imtest           — White / Cameron-Trivedi information-matrix test
  estat ovtest           — Ramsey RESET test for omitted variables
  dwstat                 — Durbin-Watson autocorrelation statistic
  xtserial               — Wooldridge test for serial correlation in panels

Most of these read the last estimation result from state.e_results["predict_model"]
(the fitted statsmodels Results object) and re-run the test against it.
"""

import numpy as np
import pandas as pd
from rich.console import Console

from commands.state import AppState
from commands.parse_helpers import parse_command_line


def _require_last_estimation(state, console, need_residuals=True):
    """Return the last fitted statsmodels Results object, or None."""
    if not state.e_results or "predict_model" not in state.e_results:
        console.print("[red]No recent estimation. Run regress/logit/probit/... first.[/red]")
        return None
    return state.e_results["predict_model"]


# ───────────────────────────────────────────────────────────────
# vif
# ───────────────────────────────────────────────────────────────

def cmd_vif(rest: str, state: AppState, console: Console):
    """
    vif

    Variance Inflation Factor for each regressor of the last regression.
    Rule of thumb: VIF > 10 signals problematic multicollinearity.
    """
    results = _require_last_estimation(state, console)
    if results is None:
        return
    from statsmodels.stats.outliers_influence import variance_inflation_factor

    try:
        X = results.model.exog
        names = state.e_results.get("predict_X_cols", list(range(X.shape[1])))
    except Exception as e:
        console.print(f"[red]vif: cannot access design matrix: {e}[/red]")
        return

    console.print("\n[bold]Variance Inflation Factors[/bold]")
    console.print(f"{'Variable':<20} {'VIF':>10}")
    console.print("─" * 32)
    max_vif = 0.0
    for i, name in enumerate(names):
        if name == "const":
            continue
        try:
            v = variance_inflation_factor(X, i)
        except Exception:
            v = float("nan")
        max_vif = max(max_vif, v if np.isfinite(v) else 0.0)
        flag = " ⚠" if np.isfinite(v) and v > 10 else ""
        console.print(f"{name:<20} {v:>10.2f}{flag}")
    console.print("─" * 32)
    console.print(f"{'Max VIF':<20} {max_vif:>10.2f}")
    if max_vif > 10:
        console.print("[yellow]VIF > 10 suggests problematic multicollinearity.[/yellow]")

    state.r_results["vif_max"] = float(max_vif)


# ───────────────────────────────────────────────────────────────
# estat (dispatcher)
# ───────────────────────────────────────────────────────────────

def cmd_estat(rest: str, state: AppState, console: Console):
    """
    estat hettest | bgodfrey | imtest | ovtest | dwatson | summarize

    Dispatch to post-estimation subcommands.
    """
    rest = rest.strip()
    if not rest:
        console.print("[red]Syntax: estat hettest | bgodfrey | imtest | ovtest | dwatson[/red]")
        return
    parts = rest.split(None, 1)
    sub = parts[0].lower()
    sub_rest = parts[1] if len(parts) > 1 else ""

    dispatch = {
        "hettest":  _estat_hettest,
        "bgodfrey": _estat_bgodfrey,
        "bg":       _estat_bgodfrey,
        "imtest":   _estat_imtest,
        "ovtest":   _estat_ovtest,
        "dwatson":  _estat_dwatson,
        "dw":       _estat_dwatson,
        "summarize": _estat_summarize,
        "sum":       _estat_summarize,
    }
    h = dispatch.get(sub)
    if h:
        h(sub_rest, state, console)
    else:
        console.print(f"[red]Unknown estat subcommand: {sub}[/red]")


def _estat_hettest(rest, state, console):
    """Breusch-Pagan heteroskedasticity test."""
    results = _require_last_estimation(state, console)
    if results is None:
        return
    from statsmodels.stats.diagnostic import het_breuschpagan
    try:
        lm, lm_p, f, f_p = het_breuschpagan(results.resid, results.model.exog)
    except Exception as e:
        console.print(f"[red]hettest failed: {e}[/red]")
        return
    console.print("\n[bold]Breusch-Pagan test for heteroskedasticity[/bold]")
    console.print(f"H0: Constant variance")
    console.print(f"  chi2(1)    = {lm:.4f}")
    console.print(f"  Prob > chi2 = {lm_p:.4f}")
    if lm_p < 0.05:
        console.print("[yellow]  Reject H0: evidence of heteroskedasticity.[/yellow]")
    state.r_results.update({"bp_chi2": float(lm), "bp_p": float(lm_p)})


def _estat_bgodfrey(rest, state, console):
    """Breusch-Godfrey test for serial correlation. Accepts lags(N), default 1."""
    results = _require_last_estimation(state, console)
    if results is None:
        return
    from statsmodels.stats.diagnostic import acorr_breusch_godfrey
    parsed = parse_command_line(rest)
    nlags = int(parsed["options"].get("lags", 1))
    try:
        lm, lm_p, f, f_p = acorr_breusch_godfrey(results, nlags=nlags)
    except Exception as e:
        console.print(f"[red]bgodfrey failed: {e}[/red]")
        return
    console.print(f"\n[bold]Breusch-Godfrey LM test for autocorrelation (lags={nlags})[/bold]")
    console.print(f"H0: No serial correlation")
    console.print(f"  chi2({nlags})   = {lm:.4f}")
    console.print(f"  Prob > chi2 = {lm_p:.4f}")
    if lm_p < 0.05:
        console.print("[yellow]  Reject H0: residuals are serially correlated.[/yellow]")
    state.r_results.update({"bg_chi2": float(lm), "bg_p": float(lm_p)})


def _estat_imtest(rest, state, console):
    """White's IM test (het + skew + kurt). Uses statsmodels.het_white."""
    results = _require_last_estimation(state, console)
    if results is None:
        return
    from statsmodels.stats.diagnostic import het_white
    try:
        lm, lm_p, f, f_p = het_white(results.resid, results.model.exog)
    except Exception as e:
        console.print(f"[red]imtest failed: {e}[/red]")
        return
    console.print("\n[bold]White's test for heteroskedasticity[/bold]")
    console.print(f"H0: Homoskedasticity; no cross-product effects")
    console.print(f"  chi2      = {lm:.4f}")
    console.print(f"  Prob > chi2 = {lm_p:.4f}")
    if lm_p < 0.05:
        console.print("[yellow]  Reject H0.[/yellow]")
    state.r_results.update({"white_chi2": float(lm), "white_p": float(lm_p)})


def _estat_ovtest(rest, state, console):
    """Ramsey RESET test for omitted variables (uses statsmodels.linear_reset)."""
    results = _require_last_estimation(state, console)
    if results is None:
        return
    try:
        from statsmodels.stats.diagnostic import linear_reset
    except ImportError:
        console.print("[red]ovtest: requires newer statsmodels (>=0.14)[/red]")
        return
    try:
        reset = linear_reset(results, power=[2, 3], use_f=True)
    except Exception as e:
        console.print(f"[red]ovtest failed: {e}[/red]")
        return
    console.print("\n[bold]Ramsey RESET test for omitted variables[/bold]")
    console.print(f"H0: Model has no omitted variables")
    try:
        console.print(f"  F({reset.df_num:.0f}, {reset.df_denom:.0f}) = {reset.fvalue:.4f}")
        console.print(f"  Prob > F = {reset.pvalue:.4f}")
        if reset.pvalue < 0.05:
            console.print("[yellow]  Reject H0: likely omitted nonlinearities.[/yellow]")
        state.r_results.update({"reset_F": float(reset.fvalue), "reset_p": float(reset.pvalue)})
    except AttributeError:
        # Older API
        console.print(f"  F = {reset[0]:.4f}, p = {reset[1]:.4f}")


def _estat_dwatson(rest, state, console):
    """Durbin-Watson statistic."""
    results = _require_last_estimation(state, console)
    if results is None:
        return
    from statsmodels.stats.stattools import durbin_watson
    dw = durbin_watson(results.resid)
    console.print(f"\n[bold]Durbin-Watson statistic[/bold] = {dw:.4f}")
    if dw < 1.5:
        console.print("[yellow]  DW < 1.5 suggests positive autocorrelation.[/yellow]")
    elif dw > 2.5:
        console.print("[yellow]  DW > 2.5 suggests negative autocorrelation.[/yellow]")
    state.r_results["dw"] = float(dw)


def _estat_summarize(rest, state, console):
    """Summarize the estimation sample for regressors used in the last fit."""
    results = _require_last_estimation(state, console)
    if results is None:
        return
    names = state.e_results.get("predict_X_cols", [])
    idx = state.e_results.get("predict_index")
    if idx is None or not state.has_data():
        console.print("[red]estat summarize: no stored estimation sample[/red]")
        return
    df = state.data.loc[idx, [n for n in names if n in state.data.columns and n != "const"]]
    if df.empty:
        console.print("[yellow]No regressors to summarize (intercept-only model?)[/yellow]")
        return
    summary = df.agg(["count", "mean", "std", "min", "max"]).T
    console.print("\n[bold]Estimation-sample summary[/bold]")
    console.print(summary.to_string())


# ───────────────────────────────────────────────────────────────
# Standalone convenience: dwstat (alias for estat dwatson)
# ───────────────────────────────────────────────────────────────

def cmd_dwstat(rest: str, state: AppState, console: Console):
    """dwstat — report the Durbin-Watson statistic from the last regression."""
    _estat_dwatson(rest, state, console)


# ───────────────────────────────────────────────────────────────
# xtserial — Wooldridge test for serial correlation in panels
# ───────────────────────────────────────────────────────────────

def cmd_xtserial(rest: str, state: AppState, console: Console):
    """
    xtserial depvar indepvars [if]

    Wooldridge (2002) test for first-order autocorrelation in the
    idiosyncratic errors of a panel data model. Requires xtset to have
    been called first. Implementation follows Drukker (2003, other statistical tools
    Journal): run OLS in first differences and test whether the
    residuals' AR(1) coefficient equals -0.5.
    """
    state.require_data()
    from commands.expression import eval_condition

    pvar = getattr(state, "panel_var", None) or state.e_results.get("panel_var")
    tvar = getattr(state, "time_var", None)  or state.e_results.get("time_var")
    if pvar is None or tvar is None:
        console.print("[red]xtserial: run xtset panelvar timevar first[/red]")
        return

    parsed = parse_command_line(rest)
    if len(parsed["varlist"]) < 2:
        console.print("[red]Syntax: xtserial depvar indepvars[/red]")
        return
    depvar = parsed["varlist"][0]
    indep = parsed["varlist"][1:]

    df = state.data.copy()
    if parsed["if_cond"]:
        df = df.loc[eval_condition(parsed["if_cond"], df)]
    df = df.dropna(subset=[depvar, pvar, tvar] + indep).sort_values([pvar, tvar])

    # First-difference every variable within panel
    def fd(col):
        return df.groupby(pvar)[col].diff()

    y = fd(depvar)
    X = pd.DataFrame({v: fd(v) for v in indep})
    mask = y.notna() & X.notna().all(axis=1)
    y = y[mask].values
    X = X[mask].values

    if len(y) < 10:
        console.print("[red]xtserial: too few observations after first-differencing[/red]")
        return

    # OLS in first differences
    import statsmodels.api as sm
    Xc = sm.add_constant(X, has_constant="add")
    results = sm.OLS(y, Xc).fit()
    resid = results.resid

    # Regress resid_t on resid_{t-1} within panel
    panel_ids = df.loc[mask.index[mask]].set_index([pvar, tvar]).index.get_level_values(0)
    df_resid = pd.DataFrame({"r": resid, "pid": panel_ids})
    df_resid["r_lag"] = df_resid.groupby("pid")["r"].shift(1)
    df_resid = df_resid.dropna()

    # Under H0 of no AR(1) in idiosyncratic errors, the coefficient on r_lag
    # in a first-differenced residual regression is -0.5.
    from scipy.stats import t as t_dist
    Xr = sm.add_constant(df_resid["r_lag"].values, has_constant="add")
    rho_fit = sm.OLS(df_resid["r"].values, Xr).fit(cov_type="cluster",
                     cov_kwds={"groups": df_resid["pid"].values})
    rho_hat = rho_fit.params[1]
    rho_se  = rho_fit.bse[1]
    t_stat  = (rho_hat - (-0.5)) / rho_se
    p_val   = 2 * (1 - t_dist.cdf(abs(t_stat), df=rho_fit.df_resid))

    console.print("\n[bold]Wooldridge test for autocorrelation in panel data[/bold]")
    console.print(f"H0: no first-order autocorrelation")
    console.print(f"  rho_hat = {rho_hat:.4f}  (H0: -0.5)")
    console.print(f"  t stat  = {t_stat:.4f}")
    console.print(f"  Prob > |t| = {p_val:.4f}")
    if p_val < 0.05:
        console.print("[yellow]  Reject H0: panel has serial correlation.[/yellow]")
    state.r_results.update({"xtserial_rho": float(rho_hat),
                            "xtserial_t": float(t_stat),
                            "xtserial_p": float(p_val)})
