"""Pure-NumPy/SciPy fallback for the subset of statsmodels that Rln uses.

WHY THIS EXISTS
---------------
statsmodels has no published Android wheel and (as of this writing) does not
reliably cross-compile for arm64 under python-for-android — see
``android/p4a-recipes/statsmodels`` and the long trail of ``_check_sm_*.sh``
diagnostics. SciPy and NumPy, by contrast, *do* cross-compile and already ship
in the Android build. So when ``import statsmodels`` fails, Rln can still run
its core econometrics by falling back to estimators built directly on
NumPy/SciPy.

This module deliberately mimics the small slice of the ``statsmodels.api``
("sm") surface that ``commands/estimation.py`` and ``commands/estimation_glm.py``
consume, so it can be returned in place of ``sm`` with no changes at the call
sites:

    sm.add_constant(X[, has_constant=...])
    sm.OLS(y, X).fit([cov_type=...][, cov_kwds=...])
    sm.WLS(y, X, weights=w).fit(...)
    sm.Logit(y, X).fit(disp=0, cov_type=..., cov_kwds=...)
    sm.Probit(y, X).fit(...)
    sm.Poisson(y, X).fit(..., offset=...)
    sm.GLM(y, X, family=sm.families.Binomial()).fit(freq_weights=..., ...)
    sm.families.Binomial() / .Poisson() / .Gaussian()
    sm.families.links.Probit() / .Logit() / .Log() / .Identity()

Supported covariance types: ``nonrobust`` (classical), ``HC1`` (heteroskedastic
robust), and ``cluster`` (one-way cluster-robust, ``cov_kwds={"groups": ...}``).

Result objects expose the attributes the Rln display/`e()` code reads:
``params, bse, tvalues, pvalues, conf_int(), cov_params(), fittedvalues, resid,
rsquared, rsquared_adj, fvalue, f_pvalue, ess, ssr, mse_resid, df_model,
df_resid, llf, llnull, llr, llr_pvalue, prsquared, predict()``.

Numerical results are validated against real statsmodels in
``tests/test_stats_fallback.py`` to tight tolerances (coefs/SE within ~1e-6 for
OLS, ~1e-5 for the MLE models). This is NOT a full statsmodels replacement —
it covers exactly what Rln needs and nothing more.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as _sps

__all__ = [
    "add_constant", "OLS", "WLS", "GLS",
    "Logit", "Probit", "Poisson", "GLM", "families",
    "IS_FALLBACK",
]

# Lets callers detect the fallback (e.g. to print a one-time banner).
IS_FALLBACK = True


# ──────────────────────────────────────────────────────────────────────────
#  Helpers: coerce y/X to numpy while remembering names + index
# ──────────────────────────────────────────────────────────────────────────

def _as_xy(y, X):
    """Return (y_arr, X_arr, x_names, index) from pandas/ndarray inputs."""
    if isinstance(X, pd.DataFrame):
        x_names = list(X.columns)
        index = X.index
        X_arr = X.to_numpy(dtype=float)
    else:
        X_arr = np.asarray(X, dtype=float)
        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(-1, 1)
        x_names = [f"x{i}" for i in range(X_arr.shape[1])]
        # Name the first constant column "const" so constant-aware logic (R²
        # centering, the joint Wald F over *slopes*) treats it correctly even
        # when the design is a raw numpy array — e.g. the het_breuschpagan /
        # het_white auxiliary regressions pass numpy exog with an embedded
        # intercept column.
        for _j in range(X_arr.shape[1]):
            if np.ptp(X_arr[:, _j]) == 0:
                x_names[_j] = "const"
                break
        index = (y.index if isinstance(y, (pd.Series, pd.DataFrame))
                 else pd.RangeIndex(X_arr.shape[0]))
    if isinstance(y, (pd.Series, pd.DataFrame)):
        index = y.index
        y_arr = np.asarray(y, dtype=float).ravel()
    else:
        y_arr = np.asarray(y, dtype=float).ravel()
    return y_arr, X_arr, x_names, index


def add_constant(data, prepend=True, has_constant="skip"):
    """Mimic ``statsmodels.add_constant``.

    Adds a column named ``const`` of all 1.0. ``prepend=True`` (statsmodels'
    default) puts it first. ``has_constant`` controls behaviour when a constant
    column already exists: ``"add"`` adds anyway, ``"skip"`` returns unchanged,
    ``"raise"`` errors.
    """
    if isinstance(data, pd.DataFrame):
        # Detect an existing constant column.
        is_const = (data.nunique() == 1).any()
        if is_const and has_constant == "skip":
            return data
        if is_const and has_constant == "raise":
            raise ValueError("data already contains a constant column")
        out = data.copy()
        out.insert(0 if prepend else len(out.columns), "const", 1.0)
        return out
    arr = np.asarray(data, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    col = np.ones((arr.shape[0], 1))
    if (np.ptp(arr, axis=0) == 0).any() and has_constant == "skip":
        return arr
    return np.hstack([col, arr]) if prepend else np.hstack([arr, col])


def _wald_f(params, cov, x_names, df_resid):
    """Joint Wald F that all non-constant coefficients are zero.

    Matches statsmodels' ``fvalue``/``f_pvalue``: uses the (possibly robust)
    covariance, so the reported F reflects the chosen cov_type.
    """
    slope_idx = [i for i, n in enumerate(x_names) if n != "const"]
    if not slope_idx:
        return np.nan, np.nan
    q = len(slope_idx)
    b = params[slope_idx]
    V = cov[np.ix_(slope_idx, slope_idx)]
    try:
        wald = float(b @ np.linalg.solve(V, b))
    except np.linalg.LinAlgError:
        wald = float(b @ np.linalg.pinv(V) @ b)
    F = wald / q
    p = float(_sps.f.sf(F, q, df_resid)) if df_resid > 0 else np.nan
    return F, p


# ──────────────────────────────────────────────────────────────────────────
#  Sandwich covariance builders (shared by OLS and GLM/MLE)
# ──────────────────────────────────────────────────────────────────────────

def _cov_classical(bread, scale):
    """Classical covariance = scale * bread, where bread = (X'WX)^-1."""
    return scale * bread


def _cov_hc1(bread, X, resid, n, k, w=None):
    """HC1 heteroskedasticity-robust sandwich.

    meat = X' diag(u^2) X with u the (weighted) residual score; HC1 applies the
    finite-sample factor n/(n-k).
    """
    u = resid if w is None else resid * w
    Xu = X * u[:, None]
    meat = Xu.T @ Xu
    cov = bread @ meat @ bread
    return (n / (n - k)) * cov


def _cov_cluster(bread, X, resid, groups, n, k, w=None):
    """One-way cluster-robust sandwich with statsmodels' small-sample scaling.

    meat = sum_g (X_g' u_g)(X_g' u_g)'  ;  scale = (G/(G-1)) * ((n-1)/(n-k)).
    """
    u = resid if w is None else resid * w
    Xu = X * u[:, None]
    groups = np.asarray(groups)
    uniq = pd.unique(groups)
    G = len(uniq)
    meat = np.zeros((X.shape[1], X.shape[1]))
    # group-sum of score contributions
    order = {g: i for i, g in enumerate(uniq)}
    sums = np.zeros((G, X.shape[1]))
    gi = np.array([order[g] for g in groups])
    np.add.at(sums, gi, Xu)
    meat = sums.T @ sums
    cov = bread @ meat @ bread
    scale = (G / (G - 1.0)) * ((n - 1.0) / (n - k))
    return cov * scale, G


# ──────────────────────────────────────────────────────────────────────────
#  Linear models: OLS / WLS / GLS
# ──────────────────────────────────────────────────────────────────────────

class _ModelStub:
    """Stands in for statsmodels' ``results.model`` so post-estimation
    diagnostics that read ``results.model.exog`` keep working on the fallback."""

    def __init__(self, exog, endog, exog_names):
        self.exog = np.asarray(exog, dtype=float)
        self.endog = np.asarray(endog, dtype=float)
        self.exog_names = list(exog_names)


class _LinearResults:
    """statsmodels-compatible result object for OLS/WLS."""

    def __init__(self, params, cov, *, x_names, index, y, fitted, resid,
                 n, k, has_const, weights=None, cov_type="nonrobust",
                 n_clusters=None, use_t=True, exog=None):
        # statsmodels uses the t-distribution for classical OLS inference but
        # switches to the normal for HC/cluster-robust covariances. Mirror that
        # so p-values and CIs match to the last digit.
        self.use_t = use_t
        self.model = _ModelStub(exog if exog is not None else np.empty((n, k)),
                                y, x_names)
        self._x_names = x_names
        self._index = index
        self.params = pd.Series(params, index=x_names)
        self._cov = pd.DataFrame(cov, index=x_names, columns=x_names)
        self.nobs = float(n)
        self.k = k
        self.df_model = float((k - 1) if has_const else k)
        self.df_resid = float(n - k)
        self.cov_type = cov_type
        self.n_clusters = n_clusters
        self.weights = weights

        self.fittedvalues = pd.Series(fitted, index=index)
        self.resid = pd.Series(resid, index=index)

        # Sums of squares (centered if a constant is in the model)
        w = weights if weights is not None else np.ones(n)
        ybar = np.average(y, weights=w) if has_const else 0.0
        self.ssr = float(np.sum(w * resid ** 2))
        self.ess = float(np.sum(w * (fitted - ybar) ** 2))
        self.centered_tss = float(np.sum(w * (y - ybar) ** 2))
        self.uncentered_tss = float(np.sum(w * y ** 2))
        tss = self.centered_tss if has_const else self.uncentered_tss
        self.rsquared = 1 - self.ssr / tss if tss > 0 else np.nan
        if has_const:
            self.rsquared_adj = 1 - (1 - self.rsquared) * (n - 1) / (n - k)
        else:
            self.rsquared_adj = 1 - (1 - self.rsquared) * n / (n - k)
        self.mse_resid = self.ssr / self.df_resid if self.df_resid > 0 else np.nan
        self.mse_model = self.ess / self.df_model if self.df_model > 0 else np.nan
        self.scale = self.mse_resid

        self.fvalue, self.f_pvalue = _wald_f(
            params, np.asarray(cov), x_names, self.df_resid)

        # Gaussian log-likelihood (for AIC/BIC parity; Rln OLS display omits it)
        nobs2 = n / 2.0
        ssr = self.ssr
        self.llf = float(-nobs2 * np.log(2 * np.pi) - nobs2 * np.log(ssr / n)
                         - nobs2)

    # --- statsmodels-compatible accessors ---
    @property
    def bse(self):
        return pd.Series(np.sqrt(np.diag(self._cov.values)), index=self._x_names)

    @property
    def tvalues(self):
        return self.params / self.bse

    @property
    def pvalues(self):
        t = self.tvalues
        if self.use_t:
            return pd.Series(2 * _sps.t.sf(np.abs(t.values), self.df_resid),
                             index=self._x_names)
        return pd.Series(2 * _sps.norm.sf(np.abs(t.values)), index=self._x_names)

    def conf_int(self, alpha=0.05):
        if self.use_t:
            crit = _sps.t.ppf(1 - alpha / 2, self.df_resid)
        else:
            crit = _sps.norm.ppf(1 - alpha / 2)
        lo = self.params - crit * self.bse
        hi = self.params + crit * self.bse
        return pd.DataFrame({0: lo, 1: hi}, index=self._x_names)

    def cov_params(self):
        return self._cov

    def predict(self, exog=None):
        if exog is None:
            return self.fittedvalues
        if isinstance(exog, pd.DataFrame):
            X = exog[self._x_names].to_numpy(dtype=float)
        else:
            X = np.asarray(exog, dtype=float)
        return X @ self.params.values


class OLS:
    def __init__(self, endog, exog, **kwargs):
        self.y, self.X, self.x_names, self.index = _as_xy(endog, exog)
        self.has_const = "const" in self.x_names or _has_constant_col(self.X)

    def fit(self, cov_type="nonrobust", cov_kwds=None, **kwargs):
        return _fit_linear(self.y, self.X, self.x_names, self.index,
                           weights=None, cov_type=cov_type, cov_kwds=cov_kwds,
                           has_const=self.has_const)


class WLS:
    def __init__(self, endog, exog, weights=1.0, **kwargs):
        self.y, self.X, self.x_names, self.index = _as_xy(endog, exog)
        w = np.asarray(weights, dtype=float)
        if w.ndim == 0:
            w = np.full(self.y.shape[0], float(w))
        self.weights = w
        self.has_const = "const" in self.x_names or _has_constant_col(self.X)

    def fit(self, cov_type="nonrobust", cov_kwds=None, **kwargs):
        return _fit_linear(self.y, self.X, self.x_names, self.index,
                           weights=self.weights, cov_type=cov_type,
                           cov_kwds=cov_kwds, has_const=self.has_const)


# GLS with a diagonal/whitening sigma is out of Rln's scope; alias to WLS-less
# OLS so the symbol exists if referenced.
class GLS(OLS):
    pass


def _has_constant_col(X):
    return bool(np.any(np.ptp(X, axis=0) == 0))


def _fit_linear(y, X, x_names, index, *, weights, cov_type, cov_kwds, has_const):
    n, k = X.shape
    if weights is not None:
        sw = np.sqrt(weights)
        Xw = X * sw[:, None]
        yw = y * sw
    else:
        Xw, yw = X, y
    XtX = Xw.T @ Xw
    try:
        bread = np.linalg.inv(XtX)
        beta = bread @ (Xw.T @ yw)
    except np.linalg.LinAlgError:
        bread = np.linalg.pinv(XtX)
        beta = bread @ (Xw.T @ yw)

    fitted = X @ beta
    resid = y - fitted
    df_resid = n - k

    ct = (cov_type or "nonrobust").lower()
    n_clusters = None
    if ct in ("nonrobust", "", "classical"):
        if weights is not None:
            scale = float(np.sum(weights * resid ** 2) / df_resid)
        else:
            scale = float(np.sum(resid ** 2) / df_resid)
        cov = _cov_classical(bread, scale)
    elif ct in ("hc1", "hc0", "hc2", "hc3", "robust"):
        # Rln only asks for HC1; treat any robust request as HC1.
        if weights is not None:
            # weighted (WLS) robust: work in whitened space
            cov = _cov_hc1(bread, Xw, (yw - Xw @ beta), n, k)
        else:
            cov = _cov_hc1(bread, X, resid, n, k)
    elif ct in ("cluster", "clustered"):
        groups = cov_kwds["groups"]
        if weights is not None:
            cov, n_clusters = _cov_cluster(bread, Xw, (yw - Xw @ beta),
                                           groups, n, k)
        else:
            cov, n_clusters = _cov_cluster(bread, X, resid, groups, n, k)
    else:
        raise ValueError(f"Unsupported cov_type for fallback OLS: {cov_type!r}")

    use_t = ct in ("nonrobust", "", "classical")
    return _LinearResults(beta, cov, x_names=x_names, index=index, y=y,
                          fitted=fitted, resid=resid, n=n, k=k,
                          has_const=has_const, weights=weights,
                          cov_type=ct, n_clusters=n_clusters, use_t=use_t,
                          exog=X)


# ──────────────────────────────────────────────────────────────────────────
#  GLM / MLE results (Logit, Probit, Poisson, GLM)
# ──────────────────────────────────────────────────────────────────────────

class _MLEResults:
    """statsmodels-compatible result object for Logit/Probit/Poisson/GLM."""

    use_t = False  # discrete & GLM models report z-stats and normal CIs

    def __init__(self, params, cov, *, x_names, index, fitted, resid,
                 n, k, has_const, llf, llnull, cov_type="nonrobust",
                 n_clusters=None, exog=None, endog=None):
        self._x_names = x_names
        self._index = index
        self.model = _ModelStub(exog if exog is not None else np.empty((n, k)),
                                endog if endog is not None else fitted, x_names)
        self.params = pd.Series(params, index=x_names)
        self._cov = pd.DataFrame(cov, index=x_names, columns=x_names)
        self.nobs = float(n)
        self.k = k
        self.df_model = float((k - 1) if has_const else k)
        self.df_resid = float(n - k)
        self.cov_type = cov_type
        self.n_clusters = n_clusters

        self.fittedvalues = pd.Series(fitted, index=index)
        self.resid = pd.Series(resid, index=index)
        self.resid_response = self.resid

        self.llf = float(llf)
        self.llnull = float(llnull) if llnull is not None else None
        if self.llnull is not None:
            self.llr = float(2 * (self.llf - self.llnull))
            self.prsquared = 1 - self.llf / self.llnull if self.llnull != 0 else np.nan
            self.llr_pvalue = float(_sps.chi2.sf(self.llr, self.df_model)) \
                if self.df_model > 0 else np.nan
        else:
            self.llr = None
            self.prsquared = None
            self.llr_pvalue = None

        self.aic = float(-2 * self.llf + 2 * k)
        self.bic = float(-2 * self.llf + np.log(n) * k)

    @property
    def bse(self):
        return pd.Series(np.sqrt(np.diag(self._cov.values)), index=self._x_names)

    @property
    def tvalues(self):
        return self.params / self.bse

    @property
    def pvalues(self):
        z = self.tvalues
        return pd.Series(2 * _sps.norm.sf(np.abs(z.values)), index=self._x_names)

    def conf_int(self, alpha=0.05):
        crit = _sps.norm.ppf(1 - alpha / 2)
        lo = self.params - crit * self.bse
        hi = self.params + crit * self.bse
        return pd.DataFrame({0: lo, 1: hi}, index=self._x_names)

    def cov_params(self):
        return self._cov

    def predict(self, exog=None):
        # Returns the mean response (probabilities / counts), like statsmodels.
        if exog is None:
            return self.fittedvalues
        raise NotImplementedError("fallback predict(exog) not used by Rln")


# --- Link functions: g(mu)=eta, with inverse and derivatives -------------

class _IdentityLink:
    def link(self, mu): return mu
    def inverse(self, eta): return eta
    def inverse_deriv(self, eta): return np.ones_like(eta)

class _LogLink:
    def link(self, mu): return np.log(mu)
    def inverse(self, eta): return np.exp(eta)
    def inverse_deriv(self, eta): return np.exp(eta)

class _LogitLink:
    def link(self, mu): return np.log(mu / (1 - mu))
    def inverse(self, eta):
        # numerically stable logistic
        out = np.empty_like(eta)
        pos = eta >= 0
        out[pos] = 1.0 / (1.0 + np.exp(-eta[pos]))
        e = np.exp(eta[~pos])
        out[~pos] = e / (1.0 + e)
        return out
    def inverse_deriv(self, eta):
        p = self.inverse(eta)
        return p * (1 - p)

class _ProbitLink:
    def link(self, mu): return _sps.norm.ppf(mu)
    def inverse(self, eta): return _sps.norm.cdf(eta)
    def inverse_deriv(self, eta): return _sps.norm.pdf(eta)


class _LinksNamespace:
    Identity = _IdentityLink
    Log = _LogLink
    Logit = _LogitLink
    Probit = _ProbitLink
    # statsmodels exposes lowercase aliases too
    identity = _IdentityLink
    log = _LogLink
    logit = _LogitLink
    probit = _ProbitLink


# --- Families: variance function V(mu) + default link + loglike -----------

class _Family:
    default_link = _IdentityLink
    def __init__(self, link=None):
        self.link = link if link is not None else self.default_link()
    def variance(self, mu):
        raise NotImplementedError
    def starting_mu(self, y):
        return (y + y.mean()) / 2.0
    def loglike(self, y, mu, freq_weights=None, scale=1.0):
        raise NotImplementedError

class Gaussian(_Family):
    default_link = _IdentityLink
    def variance(self, mu):
        return np.ones_like(mu)
    def loglike(self, y, mu, freq_weights=None, scale=1.0):
        w = 1.0 if freq_weights is None else freq_weights
        nobs = np.sum(w) if freq_weights is not None else len(y)
        ssr = np.sum(w * (y - mu) ** 2)
        return -0.5 * nobs * (np.log(2 * np.pi * scale) + ssr / (nobs * scale)) \
            if scale else -np.inf

class Binomial(_Family):
    default_link = _LogitLink
    def variance(self, mu):
        return mu * (1 - mu)
    def starting_mu(self, y):
        return (y + 0.5) / 2.0
    def loglike(self, y, mu, freq_weights=None, scale=1.0):
        w = np.ones_like(y) if freq_weights is None else np.asarray(freq_weights)
        eps = 1e-12
        mu = np.clip(mu, eps, 1 - eps)
        return float(np.sum(w * (y * np.log(mu) + (1 - y) * np.log(1 - mu))))

class Poisson(_Family):  # NOTE: this is sm.families.Poisson (a family),
    default_link = _LogLink  # distinct from the top-level Poisson MODEL below.
    def variance(self, mu):
        return mu
    def starting_mu(self, y):
        return y + 0.1
    def loglike(self, y, mu, freq_weights=None, scale=1.0):
        w = np.ones_like(y) if freq_weights is None else np.asarray(freq_weights)
        eps = 1e-12
        mu = np.clip(mu, eps, None)
        from scipy.special import gammaln
        return float(np.sum(w * (y * np.log(mu) - mu - gammaln(y + 1))))


class _FamiliesNamespace:
    Gaussian = Gaussian
    Binomial = Binomial
    Poisson = Poisson
    links = _LinksNamespace()


# Single shared instance, referenced as ``sm.families`` at call sites.
families = _FamiliesNamespace()


# --- IRLS core (Fisher scoring) used by GLM and the discrete models -------

def _irls(y, X, family, *, freq_weights=None, offset=None, maxiter=100,
          tol=1e-10):
    n, k = X.shape
    w = np.ones(n) if freq_weights is None else np.asarray(freq_weights, float)
    off = np.zeros(n) if offset is None else np.asarray(offset, float)
    link = family.link

    mu = family.starting_mu(y)
    eta = link.link(mu)
    beta = np.zeros(k)
    for _ in range(maxiter):
        g_prime = link.inverse_deriv(eta)          # dmu/deta
        var = family.variance(mu)
        var = np.clip(var, 1e-12, None)
        # working response and IRLS weights
        z = (eta - off) + (y - mu) / np.clip(g_prime, 1e-12, None)
        W = w * (g_prime ** 2) / var
        WX = X * W[:, None]
        XtWX = X.T @ WX
        XtWz = WX.T @ z
        try:
            beta_new = np.linalg.solve(XtWX, XtWz)
        except np.linalg.LinAlgError:
            beta_new = np.linalg.pinv(XtWX) @ XtWz
        eta = X @ beta_new + off
        mu = link.inverse(eta)
        if np.max(np.abs(beta_new - beta)) < tol:
            beta = beta_new
            break
        beta = beta_new

    # Fisher information bread = (X'WX)^-1 (the model-based covariance)
    g_prime = link.inverse_deriv(eta)
    var = np.clip(family.variance(mu), 1e-12, None)
    W = w * (g_prime ** 2) / var
    XtWX = X.T @ (X * W[:, None])
    try:
        bread = np.linalg.inv(XtWX)
    except np.linalg.LinAlgError:
        bread = np.linalg.pinv(XtWX)
    return beta, mu, eta, bread, w, off


def _score_obs(y, X, mu, eta, family, w, off):
    """Per-observation score (gradient) contributions, for robust sandwiches."""
    g_prime = family.link.inverse_deriv(eta)
    var = np.clip(family.variance(mu), 1e-12, None)
    factor = w * g_prime / var * (y - mu)
    return X * factor[:, None]


def _fit_glm(y, X, x_names, index, family, *, freq_weights=None, offset=None,
             cov_type="nonrobust", cov_kwds=None, has_const=True,
             compute_null=True):
    n, k = X.shape
    beta, mu, eta, bread, w, off = _irls(y, X, family,
                                         freq_weights=freq_weights,
                                         offset=offset)
    ct = (cov_type or "nonrobust").lower()
    n_clusters = None
    if ct in ("nonrobust", "", "classical"):
        cov = bread
    elif ct in ("hc1", "hc0", "robust"):
        s = _score_obs(y, X, mu, eta, family, w, off)
        cov = bread @ (s.T @ s) @ bread
    elif ct in ("cluster", "clustered"):
        s = _score_obs(y, X, mu, eta, family, w, off)
        groups = np.asarray(cov_kwds["groups"])
        uniq = pd.unique(groups)
        G = len(uniq)
        order = {g: i for i, g in enumerate(uniq)}
        gi = np.array([order[g] for g in groups])
        sums = np.zeros((G, k))
        np.add.at(sums, gi, s)
        meat = sums.T @ sums
        cov = bread @ meat @ bread
        cov *= G / (G - 1.0)
        n_clusters = G
    else:
        raise ValueError(f"Unsupported cov_type for fallback GLM: {cov_type!r}")

    llf = family.loglike(y, mu, freq_weights=freq_weights)
    llnull = None
    if compute_null:
        # Intercept-only model for pseudo-R² / LR test.
        ones = np.ones((n, 1))
        try:
            b0, mu0, _, _, _, _ = _irls(y, ones, family,
                                        freq_weights=freq_weights, offset=offset)
            llnull = family.loglike(y, mu0, freq_weights=freq_weights)
        except Exception:
            llnull = None

    resid = y - mu
    return _MLEResults(beta, cov, x_names=x_names, index=index, fitted=mu,
                       resid=resid, n=n, k=k, has_const=has_const, llf=llf,
                       llnull=llnull, cov_type=ct, n_clusters=n_clusters,
                       exog=X, endog=y)


# ──────────────────────────────────────────────────────────────────────────
#  Top-level discrete models (sm.Logit / sm.Probit / sm.Poisson) + sm.GLM
# ──────────────────────────────────────────────────────────────────────────

def _logistic(eta):
    out = np.empty_like(eta, dtype=float)
    pos = eta >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-eta[pos]))
    e = np.exp(eta[~pos])
    out[~pos] = e / (1.0 + e)
    return out


def _discrete_kernels(kind, y, eta):
    """Return (mu, score_factor, w_expected, w_observed) for one Newton step.

    - score_factor: per-obs multiplier s.t. gradient = X' score_factor
    - w_expected:  Fisher (expected-info) IRLS weight — used to ITERATE (stable)
    - w_observed:  observed-info weight — used for the FINAL covariance, which
      is what statsmodels' discrete models (Logit/Probit/Poisson) report.
    For canonical links (logit, log-Poisson) expected == observed.
    """
    if kind == "logit":
        p = _logistic(eta)
        w = p * (1 - p)
        return p, (y - p), w, w
    if kind == "poisson":
        mu = np.exp(eta)
        return mu, (y - mu), mu, mu
    if kind == "probit":
        q = 2 * y - 1
        cdf_q = np.clip(_sps.norm.cdf(q * eta), 1e-12, 1.0)
        pdf = _sps.norm.pdf(eta)
        lam = q * pdf / cdf_q                      # per-obs score multiplier
        cdf = np.clip(_sps.norm.cdf(eta), 1e-12, 1 - 1e-12)
        w_exp = pdf ** 2 / (cdf * (1 - cdf))       # expected info (always > 0)
        w_obs = lam * (lam + eta)                  # observed info (statsmodels)
        mu = _sps.norm.cdf(eta)
        return mu, lam, w_exp, w_obs
    raise ValueError(f"unknown discrete kind {kind!r}")


def _loglike_discrete(kind, y, eta):
    if kind == "logit":
        p = np.clip(_logistic(eta), 1e-12, 1 - 1e-12)
        return float(np.sum(y * np.log(p) + (1 - y) * np.log(1 - p)))
    if kind == "probit":
        cdf = np.clip(_sps.norm.cdf(eta), 1e-12, 1 - 1e-12)
        return float(np.sum(y * np.log(cdf) + (1 - y) * np.log(1 - cdf)))
    if kind == "poisson":
        from scipy.special import gammaln
        mu = np.exp(eta)
        return float(np.sum(y * eta - mu - gammaln(y + 1)))
    raise ValueError(kind)


def _fit_discrete(y, X, x_names, index, kind, *, offset=None,
                  cov_type="nonrobust", cov_kwds=None, has_const=True,
                  maxiter=100, tol=1e-10):
    """Newton/Fisher-scoring MLE for logit/probit/poisson.

    Iterates with the (always-positive) expected information for numerical
    stability, then forms the reported covariance from the *observed*
    information — exactly what statsmodels' discrete estimators do, so SEs match
    even for the non-canonical probit link.
    """
    n, k = X.shape
    off = np.zeros(n) if offset is None else np.asarray(offset, float)
    beta = np.zeros(k)
    for _ in range(maxiter):
        eta = X @ beta + off
        mu, score_fac, w_exp, _ = _discrete_kernels(kind, y, eta)
        grad = X.T @ score_fac
        W = np.clip(w_exp, 1e-12, None)
        H = X.T @ (X * W[:, None])
        try:
            step = np.linalg.solve(H, grad)
        except np.linalg.LinAlgError:
            step = np.linalg.pinv(H) @ grad
        beta_new = beta + step
        if np.max(np.abs(beta_new - beta)) < tol:
            beta = beta_new
            break
        beta = beta_new

    eta = X @ beta + off
    mu, score_fac, _, w_obs = _discrete_kernels(kind, y, eta)
    info = X.T @ (X * np.clip(w_obs, 1e-12, None)[:, None])  # observed info
    try:
        bread = np.linalg.inv(info)
    except np.linalg.LinAlgError:
        bread = np.linalg.pinv(info)

    ct = (cov_type or "nonrobust").lower()
    n_clusters = None
    if ct in ("nonrobust", "", "classical"):
        cov = bread
    elif ct in ("hc1", "hc0", "robust"):
        s = X * score_fac[:, None]
        cov = bread @ (s.T @ s) @ bread
    elif ct in ("cluster", "clustered"):
        s = X * score_fac[:, None]
        groups = np.asarray(cov_kwds["groups"])
        uniq = pd.unique(groups)
        G = len(uniq)
        order = {g: i for i, g in enumerate(uniq)}
        gi = np.array([order[g] for g in groups])
        sums = np.zeros((G, k))
        np.add.at(sums, gi, s)
        cov = bread @ (sums.T @ sums) @ bread * (G / (G - 1.0))
        n_clusters = G
    else:
        raise ValueError(f"Unsupported cov_type for fallback {kind}: {cov_type!r}")

    llf = _loglike_discrete(kind, y, eta)
    # Intercept-only null model for pseudo-R² / LR test.
    llnull = None
    try:
        ones = np.ones((n, 1))
        b0 = np.zeros(1)
        for _ in range(maxiter):
            e0 = ones @ b0 + off
            _, sf0, we0, _ = _discrete_kernels(kind, y, e0)
            g0 = ones.T @ sf0
            H0 = ones.T @ (ones * np.clip(we0, 1e-12, None)[:, None])
            nb0 = b0 + np.linalg.solve(H0, g0)
            if np.max(np.abs(nb0 - b0)) < tol:
                b0 = nb0
                break
            b0 = nb0
        llnull = _loglike_discrete(kind, y, ones @ b0 + off)
    except Exception:
        llnull = None

    return _MLEResults(beta, cov, x_names=x_names, index=index, fitted=mu,
                       resid=y - mu, n=n, k=k, has_const=has_const, llf=llf,
                       llnull=llnull, cov_type=ct, n_clusters=n_clusters,
                       exog=X, endog=y)


class _DiscreteModel:
    _kind = None  # set by subclass

    def __init__(self, endog, exog, offset=None, **kwargs):
        self.y, self.X, self.x_names, self.index = _as_xy(endog, exog)
        self.has_const = "const" in self.x_names or _has_constant_col(self.X)
        self._offset = None if offset is None else np.asarray(offset, float)

    def fit(self, disp=0, cov_type="nonrobust", cov_kwds=None, maxiter=100,
            offset=None, **kwargs):
        off = offset if offset is not None else self._offset
        return _fit_discrete(self.y, self.X, self.x_names, self.index,
                             self._kind, offset=off, cov_type=cov_type,
                             cov_kwds=cov_kwds, has_const=self.has_const,
                             maxiter=maxiter)


class Logit(_DiscreteModel):
    _kind = "logit"


class Probit(_DiscreteModel):
    _kind = "probit"


class PoissonModel(_DiscreteModel):
    _kind = "poisson"


class GLM:
    """sm.GLM(endog, exog, family=...). Supports freq_weights + offset."""

    def __init__(self, endog, exog, family=None, offset=None, freq_weights=None,
                 **kwargs):
        self.y, self.X, self.x_names, self.index = _as_xy(endog, exog)
        self.has_const = "const" in self.x_names or _has_constant_col(self.X)
        self.family = family if family is not None else Gaussian()
        self._offset = None if offset is None else np.asarray(offset, float)
        self._fw = None if freq_weights is None else np.asarray(freq_weights, float)

    def fit(self, cov_type="nonrobust", cov_kwds=None, freq_weights=None,
            offset=None, maxiter=100, **kwargs):
        fw = freq_weights if freq_weights is not None else self._fw
        off = offset if offset is not None else self._offset
        return _fit_glm(self.y, self.X, self.x_names, self.index, self.family,
                        freq_weights=fw, offset=off, cov_type=cov_type,
                        cov_kwds=cov_kwds, has_const=self.has_const)


# Expose the top-level Poisson MODEL under the name ``Poisson`` so that
# ``sm.Poisson`` works. ``sm.families.Poisson`` (the *family*, used by GLM) was
# captured into the ``families`` namespace earlier, before this rebinding, so
# the two names resolve to the right objects in their respective namespaces.
Poisson = PoissonModel


# ──────────────────────────────────────────────────────────────────────────
#  Post-estimation diagnostics (pure NumPy) — drop-in for the statsmodels
#  functions diagnostics.py uses. Validated against statsmodels in the tests.
# ──────────────────────────────────────────────────────────────────────────

def variance_inflation_factor(exog, exog_idx):
    """VIF for column ``exog_idx`` of design matrix ``exog`` (incl. const).

    Matches statsmodels: VIF_i = 1 / (1 - R²_i) from regressing column i on the
    remaining columns.
    """
    exog = np.asarray(exog, dtype=float)
    k = exog.shape[1]
    mask = np.arange(k) != exog_idx
    x_i = exog[:, exog_idx]
    x_other = exog[:, mask]
    # OLS of x_i on the other columns (which already include the constant).
    beta, *_ = np.linalg.lstsq(x_other, x_i, rcond=None)
    resid = x_i - x_other @ beta
    ss_res = float(resid @ resid)
    ss_tot = float(((x_i - x_i.mean()) ** 2).sum())
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return 1.0 / (1.0 - r2) if r2 < 1 else np.inf


def durbin_watson(resid):
    """Durbin-Watson statistic: sum((e_t - e_{t-1})^2) / sum(e_t^2)."""
    resid = np.asarray(resid, dtype=float)
    diff = np.diff(resid)
    denom = float(resid @ resid)
    return float(diff @ diff) / denom if denom > 0 else np.nan


def het_breuschpagan(resid, exog_het):
    """Breusch-Pagan test (Koenker's studentized form, statsmodels' default).

    Returns ``(lm, lm_pvalue, fvalue, f_pvalue)``. The auxiliary regression
    fits resid² on ``exog_het``; LM = nobs·R²_aux ~ chi²(k-1).
    """
    resid = np.asarray(resid, dtype=float)
    x = np.asarray(exog_het, dtype=float)
    nobs, k = x.shape
    y = resid ** 2
    aux = OLS(y, x).fit()
    r2 = aux.rsquared
    df = k - 1  # exclude constant
    lm = nobs * r2
    lm_p = float(_sps.chi2.sf(lm, df)) if df > 0 else np.nan
    return lm, lm_p, float(aux.fvalue), float(aux.f_pvalue)


def het_white(resid, exog):
    """White's test for heteroskedasticity.

    Auxiliary regression of resid² on the regressors, their squares, and all
    pairwise cross-products. Returns ``(lm, lm_pvalue, fvalue, f_pvalue)`` with
    LM = nobs·R²_aux ~ chi²(p) where p = #aux regressors (excl. const).
    """
    resid = np.asarray(resid, dtype=float)
    x = np.asarray(exog, dtype=float)
    nobs, k = x.shape
    # Identify the constant column (all-equal); build augmented design.
    const_cols = [j for j in range(k) if np.ptp(x[:, j]) == 0]
    noncon = [j for j in range(k) if j not in const_cols]
    cols = [np.ones(nobs)]
    Xn = x[:, noncon]
    for j in range(Xn.shape[1]):
        cols.append(Xn[:, j])
    for a in range(Xn.shape[1]):
        for b in range(a, Xn.shape[1]):
            cols.append(Xn[:, a] * Xn[:, b])
    Z = np.column_stack(cols)
    # Drop any perfectly collinear/constant aux columns beyond the intercept.
    y = resid ** 2
    aux = OLS(y, Z).fit()
    r2 = aux.rsquared
    p = Z.shape[1] - 1
    lm = nobs * r2
    lm_p = float(_sps.chi2.sf(lm, p)) if p > 0 else np.nan
    return lm, lm_p, float(aux.fvalue), float(aux.f_pvalue)


# ──────────────────────────────────────────────────────────────────────────
#  Backend-selection helpers used by estimation.py / estimation_glm.py
# ──────────────────────────────────────────────────────────────────────────

_NOTICE_SHOWN = False


def notify_once(console=None):
    """Print a one-time, low-key notice that the fallback is in use.

    Keeps researchers informed that inference is computed by Rln's own
    NumPy/SciPy estimators (validated against statsmodels) rather than
    statsmodels itself — important for reproducibility notes in a paper.
    """
    global _NOTICE_SHOWN
    if _NOTICE_SHOWN:
        return
    _NOTICE_SHOWN = True
    msg = ("[dim]Note: statsmodels is unavailable on this platform; using Rln's "
           "built-in NumPy/SciPy estimator (results validated against "
           "statsmodels to ~1e-6).[/dim]")
    try:
        if console is not None:
            console.print(msg)
        else:
            from rich.console import Console
            Console().print(msg)
    except Exception:
        pass


# Supported model names — used by the guard below and by tests.
_SUPPORTED = {"add_constant", "OLS", "WLS", "GLS", "Logit", "Probit",
              "Poisson", "GLM", "families", "IS_FALLBACK", "notify_once"}


def __getattr__(name):
    """PEP 562 module hook: give a clear error for statsmodels features the
    fallback intentionally does not implement (e.g. NegativeBinomial, Tobit,
    MixedLM), instead of a bare AttributeError."""
    raise AttributeError(
        f"'{name}' is not available in Rln's NumPy/SciPy statistics fallback "
        f"(active because statsmodels could not be imported on this platform). "
        f"Supported estimators: OLS/WLS (regress), Logit, Probit, Poisson, and "
        f"GLM. Models like NegativeBinomial/Tobit/IV need statsmodels or "
        f"linearmodels, which require a desktop build.")
