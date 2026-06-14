"""
Centralized weight-handling utilities.

Every Rln command that accepts weights pulls its weight vector through
`get_weight_series()`. This keeps weight semantics consistent across
summarize, tabulate, tabstat, collapse, regress, logit, probit, poisson,
nbreg, ivregress, xtile, pctile, centile, and winsor2.

Four weight types are supported:
    fweight  Frequency weights (integer): each row represents N obs
    aweight  Analytic weights: inverse-variance; each row is mean of N
    pweight  Sampling / probability weights: inverse selection probability
    iweight  Importance weights: generic weighting, no statistical semantics

Call sites:
    from commands.weights import get_weight_series

    w = get_weight_series(parsed, df, console)
    if w is False: return   # parse error, already printed

    # w is either None (no weights) or a pd.Series aligned to df.index
"""

from __future__ import annotations
import numpy as np
import pandas as pd


def get_weight_series(parsed: dict, df: pd.DataFrame, console=None):
    """Extract and validate the weight vector from a parsed command.

    Accepts either a bare variable name (`[fweight=pop]`) or an inline
    expression (`[fweight=round(pop_frac)]`, `[aweight=w1 + w2]`). For
    the expression form the full expression engine is used, so every
    function available to `gen`/`replace` (round, sqrt, log, inlist, ...)
    is available inside the weight clause too.

    Returns:
        pd.Series of non-negative floats aligned to df.index,
        or None if no weights were given,
        or False on any error (message already printed to `console`).
    """
    w = parsed.get("weight")
    if not w:
        return None

    # Expression form
    if w.get("is_expr"):
        from commands.expression import eval_expression
        try:
            weights = eval_expression(w["expr"], df)
        except Exception as e:
            if console is not None:
                console.print(f"[red]Weight expression '{w['expr']}' failed: {e}[/red]")
            return False
        weights = pd.to_numeric(weights, errors="coerce")
    else:
        var = w["var"]
        if var not in df.columns:
            if console is not None:
                console.print(f"[red]Weight variable '{var}' not found[/red]")
            return False
        weights = pd.to_numeric(df[var], errors="coerce")

    if (weights < 0).any():
        if console is not None:
            console.print(f"[red]Weight clause contains negative values[/red]")
        return False

    weights = weights.fillna(0.0)

    if weights.sum() == 0 and console is not None:
        console.print(f"[yellow]Warning: all weights are zero or missing[/yellow]")

    return weights


def weight_description(parsed: dict) -> str:
    """One-liner describing the weight clause, for result headers."""
    w = parsed.get("weight")
    if not w:
        return ""
    key = w.get("expr") if w.get("is_expr") else w.get("var")
    return f"[{w['type']}={key}]"


# ─────────────────────────────────────────────────────────────
# Weighted statistics (used by summarize, tabstat, collapse)
# ─────────────────────────────────────────────────────────────

def weighted_mean(x, w):
    """Weighted mean ignoring NaN in x. Works for all weight types."""
    x = np.asarray(x, dtype=float)
    w = np.asarray(w, dtype=float)
    mask = ~np.isnan(x) & (w > 0)
    if not mask.any():
        return np.nan
    return np.sum(x[mask] * w[mask]) / np.sum(w[mask])


def weighted_var(x, w, wtype="aweight"):
    """Weighted variance.

    Formulas (these match conventional weighted-statistics definitions
    where applicable):

      fweight: divisor is sum(w) - 1, treating weights as replication
        counts. Equivalent to running the unweighted formula on a dataset
        where each row is duplicated w_i times.

      aweight: weights are normalized to sum to n, then the standard
        sample-variance formula sumsq / (n - 1) is applied. The result
        is invariant to uniform rescaling of the weights — multiplying
        every weight by 100 does not change the answer. This is what
        users almost always want when computing the variance of a
        weighted mean of a sample.

      pweight: same scale-invariant formula as aweight. Inference for
        pweighted estimators normally requires robust/cluster SE
        rather than this plain weighted variance, but for tabstat /
        summarize the scale-invariant version is the right default.

      iweight: no canonical convention. We use sumsq / sum(w), which
        treats weights as raw scaling factors with no inferential
        adjustment.
    """
    x = np.asarray(x, dtype=float)
    w = np.asarray(w, dtype=float)
    mask = ~np.isnan(x) & (w > 0)
    if mask.sum() < 2:
        return np.nan
    x, w = x[mask], w[mask]
    sw = np.sum(w)
    n = len(x)

    if wtype == "fweight":
        if sw <= 1:
            return np.nan
        mu = np.sum(x * w) / sw
        sumsq = np.sum(w * (x - mu) ** 2)
        return sumsq / (sw - 1)

    if wtype in ("aweight", "pweight"):
        if n <= 1:
            return np.nan
        # Normalize weights to sum to n, then apply standard (n-1) divisor.
        # This makes the result invariant to uniform rescaling of weights.
        w_norm = w * n / sw
        mu = np.sum(x * w_norm) / n
        sumsq = np.sum(w_norm * (x - mu) ** 2)
        return sumsq / (n - 1)

    # iweight or fallback
    if sw <= 0:
        return np.nan
    mu = np.sum(x * w) / sw
    sumsq = np.sum(w * (x - mu) ** 2)
    return sumsq / sw


def weighted_std(x, w, wtype="aweight"):
    v = weighted_var(x, w, wtype)
    return np.sqrt(v) if v == v and v >= 0 else np.nan  # v==v ⇔ not NaN


def weighted_quantile(x, w, p: float):
    """Weighted quantile at probability p (0 <= p <= 1).

    Uses the type-7-equivalent weighted formula:
      1. Sort by x.
      2. Build normalized partial cumulative weights
             P_i = (cw_i - 0.5 * w_i) / sum(w)
         (the midpoints of each weight interval).
      3. Linear-interpolate the value at probability p on that grid.
         Outside [P_1, P_n] we extrapolate to the nearest endpoint.

    With all w_i = 1 this reduces to the common R type-7 quantile,
    so unweighted calls behave as most users expect.
    """
    x = np.asarray(x, dtype=float)
    w = np.asarray(w, dtype=float)
    mask = ~np.isnan(x) & (w > 0)
    if not mask.any():
        return np.nan
    x, w = x[mask], w[mask]

    order = np.argsort(x)
    xs = x[order]
    ws = w[order]
    cw = np.cumsum(ws)
    total = cw[-1]
    # Midpoint probabilities
    probs = (cw - 0.5 * ws) / total

    if p <= probs[0]:
        return xs[0]
    if p >= probs[-1]:
        return xs[-1]
    # Interpolate
    idx = np.searchsorted(probs, p, side="right") - 1
    idx = max(0, min(len(xs) - 2, idx))
    p0, p1 = probs[idx], probs[idx + 1]
    x0, x1 = xs[idx], xs[idx + 1]
    if p1 == p0:
        return x0
    frac = (p - p0) / (p1 - p0)
    return x0 + frac * (x1 - x0)


def weighted_median(x, w):
    return weighted_quantile(x, w, 0.5)


def weighted_sum(x, w):
    x = np.asarray(x, dtype=float)
    w = np.asarray(w, dtype=float)
    mask = ~np.isnan(x) & (w > 0)
    return np.sum(x[mask] * w[mask]) if mask.any() else np.nan


def weighted_count(x, w):
    """Weighted count: sum of weights for non-missing x."""
    x = np.asarray(x, dtype=float)
    w = np.asarray(w, dtype=float)
    mask = ~np.isnan(x) & (w > 0)
    return float(np.sum(w[mask]))


def effective_n(w):
    """Effective sample size under weights (Kish).

    Used as the denominator for degrees-of-freedom calculations when
    reporting weighted results. Returns the raw sum for fweight (the
    natural interpretation) and the Kish-adjusted n_eff = sum(w)^2 /
    sum(w^2) for aweight/pweight/iweight.
    """
    w = np.asarray(w, dtype=float)
    w = w[w > 0]
    if len(w) == 0:
        return 0.0
    s = w.sum()
    ss = (w * w).sum()
    return float(s * s / ss) if ss > 0 else 0.0
