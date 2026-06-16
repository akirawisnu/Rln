"""
Nonlinear and IV estimation commands:
    logit, probit, poisson, nbreg, tobit, ivregress

All of these follow the same pattern as regress: parse a econometric
varlist, build a design matrix (respecting `i.var` factor syntax), run
the fit through statsmodels, and print results in other statistical tools layout. They
share one display helper for compactness.
"""

import numpy as np
import pandas as pd
from rich.console import Console

from commands.state import AppState
from commands.parse_helpers import parse_command_line
from commands.expression import eval_condition


def _check_sm():
    """Return statsmodels, or Rln's validated NumPy/SciPy fallback when it is
    unavailable (e.g. on Android). The fallback covers logit/probit/poisson and
    GLM (Binomial/Poisson, freq_weights, offset); unsupported models such as
    nbreg/tobit raise a clear message via the fallback's module guard."""
    try:
        import statsmodels.api as sm
        return sm
    except Exception:
        # See _check_statsmodels in estimation.py: fall back on ANY load failure
        # so mobile gets working econometrics instead of a crash.
        from commands import stats_fallback as sm
        sm.notify_once()
        return sm


# ───────────────────────────────────────────────────────────────
# Shared prep: parse varlist, apply if, expand i. dummies,
# drop missing, return (y, X, df_clean, n_dropped).
# ───────────────────────────────────────────────────────────────

def _prep_design(rest: str, state, console, *, require_noconst_option=True):
    """Return (depvar, indepvar_names_in_X, y, X, df_clean, n_dropped, parsed)
    or None on failure (error already printed).
    """
    state.require_data()
    parsed = parse_command_line(rest)
    if len(parsed["varlist"]) < 2:
        console.print("[red]Need at least: depvar indepvar1 [indepvar2 ...][/red]")
        return None

    depvar = parsed["varlist"][0]
    indepvars_raw = parsed["varlist"][1:]

    df = state.data.copy()
    if parsed["if_cond"]:
        mask = eval_condition(parsed["if_cond"], df)
        df = df.loc[mask]

    if depvar not in df.columns:
        console.print(f"[red]Variable '{depvar}' not found[/red]")
        return None

    indepvars = []
    for v in indepvars_raw:
        if v.startswith("i."):
            base = v[2:]
            if base not in df.columns:
                console.print(f"[red]Variable '{base}' not found[/red]")
                return None
            dummies = pd.get_dummies(df[base], prefix=base, drop_first=True, dtype=float)
            for col in dummies.columns:
                df[col] = dummies[col]
                indepvars.append(col)
        elif v.startswith("c."):
            indepvars.append(v[2:])
        else:
            if v not in df.columns:
                console.print(f"[red]Variable '{v}' not found[/red]")
                return None
            indepvars.append(v)

    # Include weight column in the dropna set so weight NaNs drop the row
    weight_var = None
    if parsed.get("weight"):
        weight_var = parsed["weight"]["var"]
        if weight_var not in df.columns:
            console.print(f"[red]Weight variable '{weight_var}' not found[/red]")
            return None

    all_vars = [depvar] + indepvars
    drop_set = list(all_vars)
    if weight_var and weight_var not in drop_set:
        drop_set.append(weight_var)
    df_clean = df[drop_set].dropna()
    if weight_var:
        # Drop zero/negative-weight rows as well
        df_clean = df_clean[df_clean[weight_var] > 0]
    n_dropped = len(df) - len(df_clean)

    if len(df_clean) < len(indepvars) + 1:
        console.print("[red]Not enough observations[/red]")
        return None

    y = df_clean[depvar].astype(float)
    X = df_clean[indepvars].astype(float)

    noconstant = ("noconstant" in parsed["options"] or "nocons" in parsed["options"]
                  if require_noconst_option else False)
    if not noconstant:
        sm = _check_sm()
        X = sm.add_constant(X, has_constant="add")

    return depvar, indepvars, y, X, df_clean, n_dropped, parsed


# ───────────────────────────────────────────────────────────────
# Shared display for single-equation MLE models
# ───────────────────────────────────────────────────────────────

def _display_mle(results, depvar, indepvar_names, n_obs, n_dropped,
                 model_name, state, console,
                 se_type="(Std. err.)", extra_stats=None):
    """Common compact display for logit/probit/poisson/nbreg/tobit."""
    has_const = "const" in results.params.index

    console.print(f"\n[bold]{model_name}[/bold]")
    console.print(f"Number of obs     = {n_obs:>10,}")
    if n_dropped:
        console.print(f"Missing dropped   = {n_dropped:>10,}")

    if hasattr(results, "llf"):
        console.print(f"Log likelihood    = {results.llf:>12.4f}")
    if hasattr(results, "llr_pvalue"):
        try:
            console.print(f"LR chi2           = {results.llr:>12.4f}")
            console.print(f"Prob > chi2       = {results.llr_pvalue:>12.4f}")
        except Exception:
            pass
    if hasattr(results, "prsquared"):
        try:
            console.print(f"Pseudo R2         = {results.prsquared:>12.4f}")
        except Exception:
            pass

    if extra_stats:
        for k, v in extra_stats.items():
            console.print(f"{k:<18}= {v}")

    console.print()
    console.print(se_type)
    console.print(f"{'Variable':<20} {'Coef.':>12} {'Std. err.':>12} {'z':>8} {'P>|z|':>8} "
                  f"{'[95% CI]':>18}")
    console.print("─" * 82)

    for name in results.params.index:
        b  = results.params[name]
        se = results.bse[name]
        z  = results.tvalues[name] if hasattr(results, "tvalues") else b / se
        p  = results.pvalues[name]
        ci = results.conf_int().loc[name]
        label = "_cons" if name == "const" else name
        console.print(f"{label:<20} {b:>12.4f} {se:>12.4f} {z:>8.2f} {p:>8.4f} "
                      f"[{ci[0]:>7.4f}, {ci[1]:>7.4f}]")

    # Publish to e() for predict/test/lincom
    state.e_results = {
        "cmd": model_name.lower().split()[0],
        "depvar": depvar,
        "N": n_obs,
        "b": results.params,
        "se": results.bse,
        "pval": results.pvalues,
        "V": results.cov_params() if hasattr(results, "cov_params") else None,
        "predict_model": results,
        "predict_X_cols": list(results.params.index),
        "ll": getattr(results, "llf", None),
    }
    if hasattr(results, "prsquared"):
        try:
            state.e_results["r2_p"] = results.prsquared
        except Exception:
            pass


# ───────────────────────────────────────────────────────────────
# logit / probit
# ───────────────────────────────────────────────────────────────

def _fit_weight_kwargs(parsed, df_clean):
    """Return (fit_kwargs dict, display_note).

    statsmodels discrete models accept freq_weights (for fweight-like
    repetition counts) but do not accept analytic/probability weights on
    the MLE path. For aweight/pweight/iweight we normalize to sum-to-n
    and pass as freq_weights; this gives the right point estimates. For
    pweight we also force robust SE.
    """
    w = parsed.get("weight")
    if not w:
        return {}, None
    wtype = w["type"]
    wvec = df_clean[w["var"]].astype(float).values
    note = f"Weight: {wtype}={w['var']}"
    if wtype == "fweight":
        return {"freq_weights": wvec}, note
    # Normalize non-frequency weights to sum to n so the likelihood
    # stays on the correct scale for MLE
    import numpy as _np
    n = len(wvec)
    scaled = wvec * (n / wvec.sum()) if wvec.sum() > 0 else wvec
    return {"freq_weights": scaled}, note


def cmd_logit(rest: str, state: AppState, console: Console):
    """
    logit depvar indepvars [if] [weight] [, robust cluster(var) or noconstant]

    Binary logistic regression. Supports fweight, aweight, pweight, iweight.
    'or' prints odds ratios. pweight implies robust standard errors.
    """
    sm = _check_sm()
    prep = _prep_design(rest, state, console)
    if prep is None:
        return
    depvar, indepvars, y, X, df_clean, n_dropped, parsed = prep

    if not set(pd.Series(y).unique()).issubset({0, 1, 0.0, 1.0}):
        console.print(f"[red]logit: outcome '{depvar}' must be 0/1[/red]")
        return

    # pweight -> force robust SE
    if parsed.get("weight") and parsed["weight"]["type"] == "pweight":
        if not ("robust" in parsed["options"] or "cluster" in parsed["options"]):
            parsed["options"]["robust"] = True
            console.print("[dim]pweight implies robust; enabling HC1 standard errors.[/dim]")

    cov_type, cov_kwds, se_type = _cov_from_options(parsed, df_clean, console)
    if cov_type is False:
        return

    fit_kwargs, wnote = _fit_weight_kwargs(parsed, df_clean)
    if wnote:
        console.print(f"[dim]{wnote}[/dim]")

    try:
        if "freq_weights" in fit_kwargs:
            # statsmodels.Logit silently ignores freq_weights. Route weighted
            # logits through GLM(family=Binomial), which accepts them and
            # produces results that match row-duplicated OLS to floating
            # point on fweights. (Gemini v126 G16.)
            glm_kwargs = {"freq_weights": fit_kwargs["freq_weights"]}
            results = sm.GLM(y, X, family=sm.families.Binomial(),
                              **glm_kwargs).fit(
                cov_type=cov_type, cov_kwds=cov_kwds)
        else:
            results = sm.Logit(y, X).fit(
                disp=0, cov_type=cov_type, cov_kwds=cov_kwds)
    except Exception as e:
        console.print(f"[red]logit failed: {e}[/red]")
        return

    _display_mle(results, depvar, indepvars, len(df_clean), n_dropped,
                 "Logistic regression", state, console, se_type=se_type)

    if "or" in parsed["options"]:
        console.print("\n[bold]Odds ratios[/bold]")
        console.print(f"{'Variable':<20} {'OR':>12} {'[95% CI]':>20}")
        for name in results.params.index:
            if name == "const":
                continue
            or_val = float(np.exp(results.params[name]))
            ci = results.conf_int().loc[name]
            console.print(f"{name:<20} {or_val:>12.4f} [{float(np.exp(ci[0])):.4f}, {float(np.exp(ci[1])):.4f}]")


def cmd_probit(rest: str, state: AppState, console: Console):
    """
    probit depvar indepvars [if] [weight] [, robust cluster(var) noconstant]

    Binary probit regression. Supports fweight, aweight, pweight, iweight.
    pweight implies robust standard errors.
    """
    sm = _check_sm()
    prep = _prep_design(rest, state, console)
    if prep is None:
        return
    depvar, indepvars, y, X, df_clean, n_dropped, parsed = prep

    if not set(pd.Series(y).unique()).issubset({0, 1, 0.0, 1.0}):
        console.print(f"[red]probit: outcome '{depvar}' must be 0/1[/red]")
        return

    if parsed.get("weight") and parsed["weight"]["type"] == "pweight":
        if not ("robust" in parsed["options"] or "cluster" in parsed["options"]):
            parsed["options"]["robust"] = True
            console.print("[dim]pweight implies robust; enabling HC1 standard errors.[/dim]")

    cov_type, cov_kwds, se_type = _cov_from_options(parsed, df_clean, console)
    if cov_type is False:
        return

    fit_kwargs, wnote = _fit_weight_kwargs(parsed, df_clean)
    if wnote:
        console.print(f"[dim]{wnote}[/dim]")

    try:
        if "freq_weights" in fit_kwargs:
            # See cmd_logit: route weighted probit through GLM with the
            # probit link so freq_weights are actually honored.
            glm_kwargs = {"freq_weights": fit_kwargs["freq_weights"]}
            results = sm.GLM(y, X,
                              family=sm.families.Binomial(
                                  link=sm.families.links.Probit()),
                              **glm_kwargs).fit(
                cov_type=cov_type, cov_kwds=cov_kwds)
        else:
            results = sm.Probit(y, X).fit(
                disp=0, cov_type=cov_type, cov_kwds=cov_kwds)
    except Exception as e:
        console.print(f"[red]probit failed: {e}[/red]")
        return

    _display_mle(results, depvar, indepvars, len(df_clean), n_dropped,
                 "Probit regression", state, console, se_type=se_type)


# ───────────────────────────────────────────────────────────────
# poisson / nbreg
# ───────────────────────────────────────────────────────────────

def cmd_poisson(rest: str, state: AppState, console: Console):
    """
    poisson depvar indepvars [if] [, robust cluster(var) exposure(var) offset(var) irr]

    Poisson regression (count data). `irr` prints incidence-rate ratios.
    `exposure(var)` adds log(var) as an offset (standard exposure handling).
    """
    sm = _check_sm()
    prep = _prep_design(rest, state, console)
    if prep is None:
        return
    depvar, indepvars, y, X, df_clean, n_dropped, parsed = prep

    cov_type, cov_kwds, se_type = _cov_from_options(parsed, df_clean, console)
    if cov_type is False:
        return

    offset = None
    expo = parsed["options"].get("exposure")
    off  = parsed["options"].get("offset")
    if expo:
        if expo not in df_clean.columns:
            console.print(f"[red]exposure variable '{expo}' not found[/red]")
            return
        offset = np.log(df_clean[expo].astype(float))
    elif off:
        if off not in df_clean.columns:
            console.print(f"[red]offset variable '{off}' not found[/red]")
            return
        offset = df_clean[off].astype(float)

    fit_kwargs, wnote = _fit_weight_kwargs(parsed, df_clean)
    if wnote:
        console.print(f"[dim]{wnote}[/dim]")
    if offset is not None:
        fit_kwargs["offset"] = offset.values if hasattr(offset, "values") else offset

    if parsed.get("weight") and parsed["weight"]["type"] == "pweight":
        if not ("robust" in parsed["options"] or "cluster" in parsed["options"]):
            parsed["options"]["robust"] = True

    try:
        if "freq_weights" in fit_kwargs:
            # statsmodels.Poisson silently ignores freq_weights — route
            # weighted poisson through GLM(family=Poisson) instead so
            # weights are actually honored. (Gemini v126.)
            glm_kwargs = {"freq_weights": fit_kwargs["freq_weights"]}
            if "offset" in fit_kwargs:
                glm_kwargs["offset"] = fit_kwargs["offset"]
            results = sm.GLM(y, X, family=sm.families.Poisson(),
                              **glm_kwargs).fit(
                cov_type=cov_type, cov_kwds=cov_kwds)
        else:
            results = sm.Poisson(y, X, **fit_kwargs) \
                        .fit(disp=0, cov_type=cov_type, cov_kwds=cov_kwds)
    except Exception as e:
        console.print(f"[red]poisson failed: {e}[/red]")
        return

    _display_mle(results, depvar, indepvars, len(df_clean), n_dropped,
                 "Poisson regression", state, console, se_type=se_type)

    if "irr" in parsed["options"]:
        console.print("\n[bold]Incidence-rate ratios[/bold]")
        for name in results.params.index:
            if name == "const":
                continue
            irr = float(np.exp(results.params[name]))
            ci = results.conf_int().loc[name]
            console.print(f"  {name:<20} {irr:>12.4f}  [{float(np.exp(ci[0])):.4f}, {float(np.exp(ci[1])):.4f}]")


def cmd_nbreg(rest: str, state: AppState, console: Console):
    """
    nbreg depvar indepvars [if] [, robust cluster(var) exposure(var) offset(var) irr]

    Negative binomial regression (overdispersed counts) via statsmodels.NegativeBinomial.
    """
    sm = _check_sm()
    prep = _prep_design(rest, state, console)
    if prep is None:
        return
    depvar, indepvars, y, X, df_clean, n_dropped, parsed = prep

    cov_type, cov_kwds, se_type = _cov_from_options(parsed, df_clean, console)
    if cov_type is False:
        return

    offset = None
    expo = parsed["options"].get("exposure")
    off  = parsed["options"].get("offset")
    if expo:
        if expo not in df_clean.columns:
            console.print(f"[red]exposure variable '{expo}' not found[/red]")
            return
        offset = np.log(df_clean[expo].astype(float)).values
    elif off:
        if off not in df_clean.columns:
            console.print(f"[red]offset variable '{off}' not found[/red]")
            return
        offset = df_clean[off].astype(float).values

    fit_kwargs, wnote = _fit_weight_kwargs(parsed, df_clean)
    if wnote:
        console.print(f"[dim]{wnote}[/dim]")
    if offset is not None:
        fit_kwargs["offset"] = offset

    if parsed.get("weight") and parsed["weight"]["type"] == "pweight":
        if not ("robust" in parsed["options"] or "cluster" in parsed["options"]):
            parsed["options"]["robust"] = True

    try:
        if "freq_weights" in fit_kwargs:
            # statsmodels.NegativeBinomial silently ignores freq_weights.
            # GLM(family=NegativeBinomial) requires alpha up front, so we
            # first estimate alpha via the unweighted MLE, then refit
            # with GLM at fixed alpha to honor the weights.
            console.print("[dim]nbreg weights: estimating alpha unweighted, "
                          "then fitting weighted GLM at fixed alpha.[/dim]")
            unw_kwargs = {k: v for k, v in fit_kwargs.items() if k != "freq_weights"}
            unw_results = sm.NegativeBinomial(y, X, **unw_kwargs).fit(disp=0)
            alpha = float(unw_results.params.get("alpha", 1.0))
            glm_kwargs = {"freq_weights": fit_kwargs["freq_weights"]}
            if "offset" in fit_kwargs:
                glm_kwargs["offset"] = fit_kwargs["offset"]
            results = sm.GLM(y, X,
                              family=sm.families.NegativeBinomial(alpha=alpha),
                              **glm_kwargs).fit(
                cov_type=cov_type, cov_kwds=cov_kwds)
        else:
            results = sm.NegativeBinomial(y, X, **fit_kwargs).fit(
                disp=0, cov_type=cov_type, cov_kwds=cov_kwds)
    except Exception as e:
        console.print(f"[red]nbreg failed: {e}[/red]")
        return

    alpha_val = None
    try:
        if "alpha" in results.params.index:
            alpha_val = f"{results.params['alpha']:.4f}"
    except Exception:
        pass
    extra = {"alpha": alpha_val} if alpha_val else None

    _display_mle(results, depvar, indepvars, len(df_clean), n_dropped,
                 "Negative binomial regression", state, console,
                 se_type=se_type, extra_stats=extra)


# ───────────────────────────────────────────────────────────────
# tobit (left-, right-, or two-limit censoring)
# ───────────────────────────────────────────────────────────────

def cmd_tobit(rest: str, state: AppState, console: Console):
    """
    tobit depvar indepvars [if] [, ll(value) ul(value) robust cluster(var)]

    Censored (tobit) regression. Either ll() or ul() (or both) must be given.
    """
    sm = _check_sm()
    prep = _prep_design(rest, state, console)
    if prep is None:
        return
    depvar, indepvars, y, X, df_clean, n_dropped, parsed = prep

    ll = parsed["options"].get("ll")
    ul = parsed["options"].get("ul")
    if ll is None and ul is None:
        console.print("[red]tobit: at least one of ll() or ul() must be specified[/red]")
        return
    ll = float(ll) if ll is not None else None
    ul = float(ul) if ul is not None else None

    # Tobit isn't in statsmodels core; implement via custom log-likelihood.
    from scipy.stats import norm
    from scipy.optimize import minimize

    X_np = X.values
    y_np = y.values
    k = X_np.shape[1]

    # Init from OLS
    beta0 = np.linalg.lstsq(X_np, y_np, rcond=None)[0]
    resid = y_np - X_np @ beta0
    sigma0 = max(np.std(resid), 1e-3)

    def nll(params):
        b = params[:k]
        log_sigma = params[k]
        sigma = np.exp(log_sigma)
        xb = X_np @ b
        z = (y_np - xb) / sigma

        # Uncensored contributions: normal pdf / sigma
        uncens = np.ones_like(y_np, dtype=bool)
        if ll is not None:
            at_ll = (y_np <= ll)
            uncens &= ~at_ll
        if ul is not None:
            at_ul = (y_np >= ul)
            uncens &= ~at_ul

        logL = 0.0
        logL += np.sum(norm.logpdf(z[uncens]) - np.log(sigma))
        if ll is not None:
            zlow = (ll - xb) / sigma
            mask = (y_np <= ll)
            logL += np.sum(norm.logcdf(zlow[mask]))
        if ul is not None:
            zhi = (ul - xb) / sigma
            mask = (y_np >= ul)
            logL += np.sum(norm.logsf(zhi[mask]))
        return -logL

    params0 = np.concatenate([beta0, [np.log(sigma0)]])
    res = minimize(nll, params0, method="BFGS")
    if not res.success:
        console.print(f"[yellow]tobit: optimizer warning: {res.message}[/yellow]")

    beta = res.x[:k]
    sigma = np.exp(res.x[k])
    ll_val = -res.fun

    # Covariance: inverse Hessian from BFGS
    cov = res.hess_inv
    se = np.sqrt(np.diag(cov)[:k])
    z = beta / se
    from scipy.stats import norm as _n
    p = 2 * (1 - _n.cdf(np.abs(z)))
    ci_low = beta - 1.96 * se
    ci_high = beta + 1.96 * se

    # Print
    cens_desc = []
    if ll is not None: cens_desc.append(f"left at {ll}")
    if ul is not None: cens_desc.append(f"right at {ul}")
    console.print(f"\n[bold]Tobit regression[/bold]  ({', '.join(cens_desc)} censoring)")
    console.print(f"Number of obs     = {len(df_clean):>10,}")
    console.print(f"Log likelihood    = {ll_val:>12.4f}")
    console.print(f"sigma             = {sigma:>12.4f}")
    console.print()
    console.print(f"{'Variable':<20} {'Coef.':>12} {'Std. err.':>12} {'z':>8} {'P>|z|':>8} "
                  f"{'[95% CI]':>18}")
    console.print("─" * 82)
    for i, name in enumerate(X.columns):
        label = "_cons" if name == "const" else name
        console.print(f"{label:<20} {beta[i]:>12.4f} {se[i]:>12.4f} {z[i]:>8.2f} {p[i]:>8.4f} "
                      f"[{ci_low[i]:>7.4f}, {ci_high[i]:>7.4f}]")

    state.e_results = {
        "cmd": "tobit",
        "depvar": depvar,
        "N": len(df_clean),
        "b": pd.Series(beta, index=X.columns),
        "se": pd.Series(se, index=X.columns),
        "sigma": sigma,
        "ll": ll_val,
        "predict_X_cols": list(X.columns),
    }


# ───────────────────────────────────────────────────────────────
# ivregress 2SLS
# ───────────────────────────────────────────────────────────────

def cmd_ivregress(rest: str, state: AppState, console: Console):
    """
    ivregress 2sls depvar exogvars (endogvar = instruments) [if] [, robust cluster(var)]

    Two-stage least squares via linearmodels.IV2SLS.
    """
    try:
        from linearmodels.iv import IV2SLS
    except ImportError:
        raise RuntimeError("linearmodels is required for ivregress (ssc install linearmodels)")

    state.require_data()

    # Pull method token
    body = rest.strip()
    method_match = None
    for meth in ("2sls", "liml", "gmm"):
        if body.lower().startswith(meth + " ") or body.lower().startswith(meth + "\t"):
            method_match = meth
            body = body[len(meth):].strip()
            break
    if method_match is None:
        console.print("[red]Syntax: ivregress 2sls depvar exogvars (endog = iv1 iv2) [if] [, robust][/red]")
        return
    if method_match != "2sls":
        console.print(f"[yellow]ivregress {method_match} not yet implemented; using 2sls[/yellow]")

    # Parse the (endog = instruments) clause
    import re as _re
    paren = _re.search(r'\(([^)]+)=([^)]+)\)', body)
    if not paren:
        console.print("[red]ivregress: need (endogvar = iv1 iv2 ...) clause[/red]")
        return
    endog_names = paren.group(1).split()
    iv_names    = paren.group(2).split()
    body_no_paren = body[:paren.start()] + body[paren.end():]

    parsed = parse_command_line(body_no_paren)
    if len(parsed["varlist"]) < 1:
        console.print("[red]ivregress: missing depvar[/red]")
        return
    depvar = parsed["varlist"][0]
    exog_names = parsed["varlist"][1:]

    df = state.data.copy()
    if parsed["if_cond"]:
        mask = eval_condition(parsed["if_cond"], df)
        df = df.loc[mask]

    needed = [depvar] + exog_names + endog_names + iv_names
    missing = [n for n in needed if n not in df.columns]
    if missing:
        console.print(f"[red]Variables not found: {missing}[/red]")
        return
    df_clean = df[needed].dropna()
    n_dropped = len(df) - len(df_clean)

    from statsmodels.api import add_constant
    exog_df = df_clean[exog_names].astype(float)
    exog_df = add_constant(exog_df, has_constant="add")
    endog_df = df_clean[endog_names].astype(float)
    instr_df = df_clean[iv_names].astype(float)

    robust = "robust" in parsed["options"]
    cluster = parsed["options"].get("cluster")

    try:
        model = IV2SLS(df_clean[depvar].astype(float), exog_df, endog_df, instr_df)
        if cluster:
            results = model.fit(cov_type="clustered", clusters=df_clean[cluster])
        elif robust:
            results = model.fit(cov_type="robust")
        else:
            results = model.fit()
    except Exception as e:
        console.print(f"[red]ivregress failed: {e}[/red]")
        return

    console.print(f"\n[bold]IV 2SLS regression[/bold]  (instrumenting: {', '.join(endog_names)})")
    console.print(f"Number of obs     = {len(df_clean):>10,}")
    console.print(f"R2                = {results.rsquared:>12.4f}")
    console.print(f"F statistic       = {results.f_statistic.stat:>12.4f}")
    console.print()
    console.print(f"{'Variable':<20} {'Coef.':>12} {'Std. err.':>12} {'z':>8} {'P>|z|':>8}")
    console.print("─" * 72)
    for name in results.params.index:
        b = results.params[name]
        se = results.std_errors[name]
        t = results.tstats[name]
        p = results.pvalues[name]
        label = "_cons" if name == "const" else name
        console.print(f"{label:<20} {b:>12.4f} {se:>12.4f} {t:>8.2f} {p:>8.4f}")

    state.e_results = {
        "cmd": "ivregress",
        "depvar": depvar,
        "N": len(df_clean),
        "b": results.params,
        "se": results.std_errors,
        "predict_model": results,
    }


# ───────────────────────────────────────────────────────────────
# Shared: parse robust / cluster() into statsmodels cov_type
# ───────────────────────────────────────────────────────────────

def _cov_from_options(parsed, df_clean, console):
    """Return (cov_type, cov_kwds, se_type_label) or (False, None, None) on error."""
    robust = "robust" in parsed["options"] or "r" in parsed["options"]
    cluster_var = parsed["options"].get("cluster") or parsed["options"].get("cl")

    if cluster_var:
        if cluster_var not in df_clean.columns:
            console.print(f"[red]Cluster variable '{cluster_var}' not found in the estimation sample[/red]")
            return False, None, None
        return ("cluster",
                {"groups": df_clean[cluster_var]},
                f"(Std. err. adjusted for {df_clean[cluster_var].nunique()} clusters in {cluster_var})")
    if robust:
        return "HC1", None, "(Robust standard errors)"
    return "nonrobust", None, "(Standard errors)"
