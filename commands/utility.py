"""
Utility commands: help, clear, pwd, cd, dir, set, memory
"""

import os
import glob
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from commands.state import AppState
from commands.parse_helpers import parse_command_line


HELP_TOPICS = {
    # ─────────────────────────────── Data I/O ──────────────────────────────
    "use": (
        'use "filename" [, clear]',
        "Load a dataset into memory. The format is detected from the file\n"
        "extension. Supports: .dta, .csv, .tsv, .xlsx, .xls, .parquet,\n"
        ".feather, .dbf, .rdata, .rds, .sav, .json, .html.\n"
        "  use \"data.csv\", clear\n"
        "  use \"panel.parquet\", clear\n"
        "  use \"https://example.com/survey.dta\", clear   (works over HTTP)"),
    "import": (
        'import delimited|excel|html "filename" [, options]',
        "Load a file with explicit format options.\n"
        "  import delimited \"raw.tsv\", delim(tab) clear\n"
        "  import excel \"report.xlsx\", sheet(\"Q3\") firstrow clear\n"
        "  import html \"https://example.com/tables.html\", table(2) clear"),
    "save": (
        'save "filename" [, replace]',
        "Write the current dataset to disk. Format follows the extension:\n"
        ".dta, .csv, .parquet, .feather, .xlsx, .rdata, .json, .txt.\n"
        "  save \"clean.dta\", replace"),
    "export": (
        'export delimited|excel "filename" [, options]',
        "Write the dataset with explicit format options.\n"
        "  export delimited \"out.csv\", replace noquote\n"
        "  export excel \"out.xlsx\", sheet(\"Results\") replace"),
    "copy": (
        'copy "<from>" "<to>" [, replace]',
        "Download a URL or copy a local file. Uses only the Python standard\n"
        "library, no extra dependencies.\n"
        "  copy \"https://example.com/dataset.dta\" \"dataset.dta\", replace\n"
        "  copy \"existing.csv\" \"backup.csv\", replace"),
    "log": (
        'log using "file" [, replace append]\nlog close',
        "Mirror REPL output to a file. Closes with 'log close'.\n"
        "  log using \"session.log\", replace"),

    # ───────────────────────────── Exploration ──────────────────────────
    "browse": (
        "browse [varlist] [if cond] [in range]",
        "Open a full-screen interactive data browser. Arrow keys to navigate,\n"
        "q to exit. Shows types, labels, and supports column sorting."),
    "describe": (
        "describe [varlist]",
        "List variables with storage type, display format, value label, and\n"
        "variable label. Also reports observation count and dataset size."),
    "codebook": (
        "codebook [varlist]",
        "Per-variable summary: type, unique value count, missing count,\n"
        "mean/sd/min/max for numeric columns, top-5 frequencies for strings."),
    "list": (
        "list [varlist] [if cond] [in range] [, noobs separator(N)]",
        "Print observations as a formatted table. Value labels are shown\n"
        "automatically for any variable that has one attached.\n"
        "  list name age income in 1/20\n"
        "  list if income > 50000, noobs"),
    "summarize": (
        "summarize [varlist] [if cond] [in range] [weight] [, detail]",
        "Descriptive statistics (N, mean, sd, min, max) for numeric variables.\n"
        "Add 'detail' for percentiles, skewness, kurtosis, and the four\n"
        "smallest and largest values. Weight forms: [fweight=v], [aweight=v],\n"
        "[pweight=v], [iweight=v]. Under weights, Obs becomes Sum(W).\n"
        "  summarize wage educ [fweight=pop], detail"),
    "tabulate": (
        "tabulate var1 [var2] [if cond] [in range] [weight] [, missing sort nolabel]",
        "Frequency table (one variable) or cross-tabulation (two variables).\n"
        "Under weights, cell values are sums of weights. Weight forms:\n"
        "[fweight=v], [aweight=v], [pweight=v], [iweight=v].\n"
        "  tabulate region\n"
        "  tabulate region gender [fweight=pop], sort"),
    "tabstat": (
        "tabstat varlist [if cond] [in range] [weight] [, by(g) stats(mean sd ...) format(%fmt)]",
        "Compact summary statistics. Default stats: n mean sd min max.\n"
        "Available: n, count, mean, sd, var, min, max, sum, median,\n"
        "range, iqr, pN (any percentile: p1 p5 p25 p50 p75 p90 p95 p99).\n"
        "Weight forms: [fweight=v], [aweight=v], [pweight=v], [iweight=v].\n"
        "  tabstat income educ [aweight=wt], by(region) stats(n mean p50 iqr)"),
    "count": (
        "count [if cond] [in range]",
        "Count observations. Stores result in r(N).\n"
        "  count if missing(income)\n"
        "  count if inrange(age, 25, 65)"),
    "contract": (
        "contract varlist [if cond] [, freq(name) cfreq(name) percent(name) nomiss]",
        "Reduce the dataset to one row per distinct combination of varlist,\n"
        "with a frequency column. Modifies the in-memory dataset.\n"
        "  contract region gender, freq(n) percent(pct)"),

    # ───────────────────── Quantile family ─────────────────────
    "pctile": (
        "pctile newvar = expr [if] [in] [weight] [, nquantiles(N) percentiles(p1 p2 ...) genp(name)]",
        "Create a new variable whose first k rows hold the percentile cut\n"
        "points of `expr`. All other rows are missing. With a weight clause,\n"
        "cuts are weighted percentiles.\n"
        "  pctile cut = income, nquantiles(4)\n"
        "  pctile cut = wage [fweight=pop], percentiles(10 25 50 75 90) genp(p)"),
    "xtile": (
        "xtile newvar = expr [if] [in] [weight] [, nquantiles(N) cutpoints(var) altdef]",
        "Assign each row to one of N quantile bins (1..N) based on `expr`.\n"
        "Default is quartiles. `altdef` uses strict-greater-than tie-breaking\n"
        "instead of the default greater-or-equal.\n"
        "  xtile quartile = income, nquantiles(4)\n"
        "  xtile decile   = wage [pweight=w], nquantiles(10)"),
    "centile": (
        "centile [varlist] [if] [in] [weight] [, centile(p1 p2 ...) level(95)]",
        "Report specified percentiles with binomial-based confidence intervals.\n"
        "Default percentile: 50 (median). Under weights, CIs are not reported\n"
        "(bootstrap-based intervals may come in a later release).\n"
        "  centile wage\n"
        "  centile wage educ, centile(10 25 50 75 90) level(99)"),
    "winsor2": (
        "winsor2 varlist [if] [in] [weight] [, cuts(lo hi) suffix(_w) replace trim by(g)]",
        "Replace extreme values of each variable with the values at the given\n"
        "percentiles. Default cuts(1 99), default suffix _w. Use `replace` to\n"
        "overwrite in place. Use `trim` to drop outliers instead of capping.\n"
        "  winsor2 wage                               (creates wage_w at p1/p99)\n"
        "  winsor2 wage income, cuts(2 98) by(industry)\n"
        "  winsor2 wage [aweight=w], cuts(5 95) replace"),
    "winsorize": (
        "winsorize varlist [if] [in] [weight] [, cuts(lo hi) suffix(_w) replace trim by(g)]",
        "Alias for winsor2. See 'help winsor2'."),

    # ────────────────────────── Variable management ──────────────────────
    "generate": (
        "generate newvar = expression [if cond]",
        "Create a new variable from an expression. When run under a\n"
        "'bysort group:' prefix, _n and _N refer to per-group positions.\n"
        "  gen log_income = ln(income + 1)\n"
        "  gen bigcity = inlist(city, \"NYC\", \"LA\", \"Chicago\")\n"
        "  bysort region: gen seq = _n"),
    "replace": (
        "replace var = expression [if cond]",
        "Modify an existing variable. Unlike generate, 'replace' does not\n"
        "create new columns — it only changes values in an existing one.\n"
        "  replace income = income / 1000\n"
        "  replace gender = \"F\" if gender == \"Female\""),
    "rename": (
        "rename old_name new_name",
        "Rename a variable. Both names must be valid identifiers."),
    "drop": (
        "drop varlist          (drop variables)\n"
        "drop if condition     (drop observations matching condition)",
        "Remove variables or rows. Pick the form by whether 'if' is present."),
    "keep": (
        "keep varlist          (keep only these variables)\n"
        "keep if condition     (keep only matching observations)",
        "Inverse of drop."),
    "label": (
        "label variable varname \"text\"\n"
        "label define lblname 1 \"Male\" 2 \"Female\"\n"
        "label values varname lblname\n"
        "label list [lblname]",
        "Attach descriptive labels to variables, and create reusable\n"
        "value-code mappings that get shown by list and describe."),
    "destring": (
        "destring varlist, replace [force]",
        "Convert string variables to numeric. Unparseable values become NaN.\n"
        "Use 'force' to coerce mixed columns."),
    "tostring": (
        "tostring varlist, replace [format(fmt)]",
        "Convert numeric variables to string. Optional printf-style format.\n"
        "  tostring rate, replace format(%.4f)"),
    "encode": (
        "encode stringvar, generate(newvar) [label(lblname)]",
        "Encode a string variable as an integer code with attached labels.\n"
        "Useful before putting a categorical variable into a regression."),
    "order": (
        "order varlist [, first last after(var) before(var) alphabetical]",
        "Reorder columns. By default moves varlist to the beginning."),
    "recode": (
        "recode var (old1 = new1) (old2 = new2) ... [, generate(newvar)]",
        "Map old values to new values. Ranges (lo/hi) and the keywords\n"
        "'else', 'missing', 'nonmissing' are supported.\n"
        "  recode age (0/17 = 1) (18/64 = 2) (65/max = 3), generate(age_cat)"),
    "reshape": (
        "reshape long stubnames, i(id_var) j(time_var)\n"
        "reshape wide stubnames, i(id_var) j(time_var)",
        "Pivot a dataset between wide and long form."),
    "clonevar": (
        "clonevar newvar = existingvar",
        "Exact copy of a variable, including labels and value-label assignments."),
    "split": (
        "split varname [, parse(sep) generate(stub) limit(N)]",
        "Split a string variable on a separator into multiple new columns.\n"
        "  split fullname, parse(\" \") generate(name) limit(2)"),

    # ──────────────────────── Data operations ──────────────────────────
    "sort": (
        "sort varlist",
        "Sort data in ascending order on the given variables."),
    "gsort": (
        "gsort [+-]var1 [+-]var2 ...",
        "Sort with direction. Prefix - for descending, + for ascending."),
    "duplicates": (
        "duplicates report [varlist]\n"
        "duplicates drop [varlist] [, force]\n"
        "duplicates tag [varlist], generate(newvar)\n"
        "duplicates list [varlist]",
        "Find, drop, tag, or list duplicate observations. With a varlist,\n"
        "the check is restricted to those variables."),
    "append": (
        'append using "filename" [, force generate(var)]',
        "Stack another dataset below the current one. Columns are aligned\n"
        "by name; extras get NaN. 'force' allows differing dtypes; 'generate'\n"
        "creates a 0/1 source indicator."),
    "merge": (
        'merge 1:1|m:1|1:m varlist using "file" [, keep(...) generate(_merge)]',
        "Key-based merge. The cardinality token (1:1, m:1, 1:m) describes\n"
        "the master:using relationship. Creates a _merge indicator:\n"
        "  1 = master only, 2 = using only, 3 = matched."),
    "fuzzmerge": (
        'fuzzmerge varname using "filename" [, threshold(0.8) method(tfidf) generate(var)]',
        "Approximate-string merge. Useful when join keys differ by spelling\n"
        "(\"IBM\" vs \"I.B.M.\"). Score is stored alongside the match."),
    "collapse": (
        "collapse (stat) varlist [if] [in] [weight] [, by(groupvars)]\n"
        "collapse (stat1) newvar1=var1 (stat2) newvar2=var2 [weight] [, by(groupvars)]",
        "Aggregate the dataset. Supported stats: mean, median, sum, count,\n"
        "min, max, sd, first, last, p1, p5, p10, p25, p50, p75, p90, p95, p99.\n"
        "Weight forms: [fweight=v], [aweight=v], [pweight=v], [iweight=v].\n"
        "  collapse (mean) income age [fweight=pop], by(country year)"),
    "fillin": (
        "fillin varlist",
        "Fill in missing combinations of varlist. Rows that didn't exist\n"
        "are added with NaN elsewhere and _fillin = 1."),
    "cross": (
        'cross using "filename"',
        "Cartesian product of the current dataset and a second file."),
    "sample": (
        "sample N [, count]",
        "Keep a random sample. N is a percentage by default; 'count' makes\n"
        "it an absolute row count."),

    # ─────────────────────────── Estimation ────────────────────────────
    "regress": (
        "regress depvar indepvars [if] [in] [weight] [, robust cluster(var) noconstant]",
        "Ordinary least-squares regression. Prefix a variable with 'i.' to\n"
        "create dummy-variable sets automatically. Weight forms:\n"
        "[fweight=v], [aweight=v], [pweight=v], [iweight=v]. pweight forces\n"
        "robust standard errors.\n"
        "  regress log_wage educ exper i.region, cluster(firm_id)\n"
        "  regress wage educ exper [pweight=surveyweight]"),
    "logit": (
        "logit depvar indepvars [if] [in] [weight] [, robust cluster(var) or noconstant]",
        "Binary logistic regression. The 'or' option prints odds ratios.\n"
        "Weight forms: [fweight=v], [aweight=v], [pweight=v], [iweight=v].\n"
        "pweight forces robust SE.\n"
        "  logit employed age educ i.region [pweight=sw], robust"),
    "probit": (
        "probit depvar indepvars [if] [in] [weight] [, robust cluster(var) noconstant]",
        "Binary probit regression. Weight forms: [fweight=v], [aweight=v],\n"
        "[pweight=v], [iweight=v]. pweight forces robust SE.\n"
        "  probit passed study_hours income"),
    "poisson": (
        "poisson depvar indepvars [if] [in] [weight] [, robust cluster(var)\n"
        "                                              exposure(var) offset(var) irr]",
        "Poisson regression for count data. The 'irr' option prints\n"
        "incidence-rate ratios. 'exposure(v)' adds ln(v) as an offset.\n"
        "Weight forms: [fweight=v], [aweight=v], [pweight=v], [iweight=v].\n"
        "  poisson visits age income [aweight=w], exposure(years) irr"),
    "nbreg": (
        "nbreg depvar indepvars [if] [in] [weight] [, robust cluster(var)\n"
        "                                            exposure(var) offset(var) irr]",
        "Negative binomial regression for overdispersed counts.\n"
        "Weight forms: [fweight=v], [aweight=v], [pweight=v], [iweight=v].\n"
        "  nbreg crashes speed_limit i.weather, cluster(state)"),
    "tobit": (
        "tobit depvar indepvars [if] [, ll(value) ul(value) robust cluster(var)]",
        "Censored regression. At least one of ll() or ul() is required.\n"
        "  tobit wage age educ, ll(0)"),
    "ivregress": (
        "ivregress 2sls depvar exogvars (endog = instruments) [if] [, robust]",
        "Two-stage least-squares IV regression.\n"
        "  ivregress 2sls wage educ (exper = mom_educ dad_educ), robust"),
    "predict": (
        "predict newvar [, xb residuals]",
        "After an estimation command, write fitted values (default) or\n"
        "residuals to a new variable."),
    "test": (
        "test var1 [var2 ...]\ntest var1 = 0\ntest var1 = var2",
        "Wald test of linear hypotheses. Reports F or chi-squared with p-value."),
    "correlate": (
        "correlate varlist [if cond]",
        "Pearson correlation matrix (listwise-complete observations)."),
    "pwcorr": (
        "pwcorr varlist [if cond] [, sig star(alpha)]",
        "Pairwise correlations — each pair uses its own complete-case rows.\n"
        "  pwcorr income educ age, sig star(0.01)"),
    "ttest": (
        "ttest var == value\nttest var1 == var2\nttest var, by(groupvar)",
        "One-sample, paired, or two-sample t-test."),

    # ───────────── Panel / causal inference ────────────────────────────
    "xtset": (
        "xtset panelvar [timevar]",
        "Declare the panel structure. Required before xtreg and used as\n"
        "defaults by didregress.\n"
        "  xtset firm_id year"),
    "xtreg": (
        "xtreg depvar indepvars [if] [, fe re robust cluster(var)]",
        "Panel regression with entity fixed effects (fe) or random effects (re).\n"
        "  xtreg log_wage educ exper, fe cluster(firm_id)"),
    "didregress": (
        "didregress (depvar) (treatment) [, group(var) time(var) method(...)\n"
        "                                 first_treat(var) post_periods(v1 v2 ...)\n"
        "                                 reference_period(v) aggregate(...) M(value)]",
        "Difference-in-differences with eleven estimators via method():\n"
        "  twfe         Two-way fixed effects (baseline, no extra packages)\n"
        "  did          Basic 2x2 DiD\n"
        "  cs           Callaway-Sant'Anna (2021), staggered designs\n"
        "  sa           Sun-Abraham (2021), interaction-weighted\n"
        "  bjs          Borusyak-Jaravel-Spiess (2024), imputation\n"
        "                 (aliases: imputation, did_imputation, borusyak)\n"
        "  gardner      Gardner (2022), two-stage DiD\n"
        "  stacked      Stacked DiD (Wing et al. 2024)\n"
        "  sdid         Synthetic DiD\n"
        "  eventstudy   Full event study with pre- and post- dynamics\n"
        "  bacon        Goodman-Bacon decomposition (diagnostic)\n"
        "  honest       Rambachan-Roth (2023) sensitivity analysis\n"
        "All estimators except twfe require: ssc install diff-diff"),
    "lincom": (
        "lincom expression",
        "Linear combination of coefficients from the last estimation.\n"
        "  lincom educ + 2*exper"),
    "margins": (
        "margins [, dydx(*) dydx(varlist) at(var=val) atmeans]",
        "Marginal effects or predictive margins after an estimation.\n"
        "  margins, dydx(*)\n"
        "  margins, at(educ=16) atmeans"),
    "xtserial": (
        "xtserial depvar indepvars [if]",
        "Wooldridge (2002) test for first-order autocorrelation in panel-data\n"
        "idiosyncratic errors. Requires xtset to have been run first."),

    # ───────────────────────── Diagnostics ────────────────────────────
    "vif": (
        "vif",
        "Variance Inflation Factor for each regressor of the last regression.\n"
        "A VIF above 10 is a common warning threshold for multicollinearity."),
    "estat": (
        "estat hettest | bgodfrey | imtest | ovtest | dwatson | summarize",
        "Post-estimation diagnostic tests:\n"
        "  hettest    Breusch-Pagan test for heteroskedasticity\n"
        "  bgodfrey   Breusch-Godfrey test for serial correlation\n"
        "  imtest     White's test for heteroskedasticity\n"
        "  ovtest     Ramsey RESET test for omitted variables\n"
        "  dwatson    Durbin-Watson autocorrelation statistic\n"
        "  summarize  Summary of the estimation sample"),
    "dwstat": (
        "dwstat",
        "Report the Durbin-Watson statistic from the last regression.\n"
        "DW near 2 indicates no first-order serial correlation."),

    # ─────────────────────────── Charts ────────────────────────────────
    "histogram": (
        'histogram var [if] [, bins(N) normal frequency title("t") color(c) name(file)]',
        "Histogram with optional normal-density overlay.\n"
        "  histogram income, bins(30) normal"),
    "kdensity": (
        'kdensity var [if] [, bwidth(N) normal title("t") name(file)]',
        "Kernel density plot. Default bandwidth uses Silverman's rule."),
    "scatter": (
        'scatter yvar xvar [if] [, by(var) lfit mcolor(c) mlabel(v) title("t")]',
        "Scatter plot, optionally faceted by a group variable, with an OLS\n"
        "fit line when 'lfit' is given."),
    "twoway": (
        'twoway (plotspec1) (plotspec2) ... [, title("t") legend name(f)]',
        "Multi-layer plot. Each (plotspec) is one layer — scatter, line,\n"
        "connected, lfit, lfitci, qfit, or area."),
    "marginsplot": (
        'marginsplot [, title("t") name(f)]',
        "Plot the last 'margins' result with 95% confidence bars."),
    "coefplot": (
        'coefplot [, level(95) title("...") xlabel("...") ylabel("...")\n'
        '              ylim(lo hi) name("file.png") zero_line(off) ref_line(off)\n'
        '              connect drop_cons(off)]',
        "Plot the most recent estimation's coefficients with confidence intervals.\n"
        "Behavior is automatic from e():\n"
        "  • Event study / dynamic DiD  -> forest plot of per-period effects\n"
        "    with reference-period and zero lines (didregress eventstudy, or\n"
        "    cs/gardner/stacked with aggregate(event_study)).\n"
        "  • Scalar-ATT DiD              -> single point with CI whisker.\n"
        "  • regress/logit/probit/etc.   -> horizontal forest plot of all\n"
        "    coefficients (excludes _cons by default).\n"
        "Options:\n"
        "  level(N)        confidence level (default 95)\n"
        "  ylim(lo hi)     force y-axis range\n"
        "  zero_line(off)  suppress the y=0 reference line\n"
        "  ref_line(off)   suppress the vertical reference-period line\n"
        "  connect         draw a connecting line between adjacent points\n"
        "  drop_cons(off)  include the constant in generic plots\n"
        "  name(f.png)     save to file instead of opening\n"
        "Examples:\n"
        "  didregress (gdp) (treat), method(eventstudy) post_periods(3 4 5)\n"
        "  coefplot, title(\"Event study, 95% CI\")\n"
        "  regress wage educ exper i.region\n"
        "  coefplot, level(99) name(\"wage_coefs.png\")"),
    "graph": (
        'graph bar (stat) var [, over(var)]\n'
        'graph box var [, over(var)]\n'
        'graph line yvar xvar [, by(var) sort]\n'
        'graph export "file" [, replace width(N)]',
        "Umbrella command for bar, box, and line charts, plus export utilities.\n"
        "  graph bar (mean) income, over(city)\n"
        "  graph export \"fig1.pdf\", replace"),

    # ────────────────── Scripting & control flow ──────────────────────
    "local": (
        "local macname = expression\n"
        "local macname \"text\"\n"
        "local macname val1 val2 val3",
        "Define a local macro. Reference later with `macname' (backtick + apostrophe).\n"
        "  local yvars income wage\n"
        "  foreach y of local yvars { summarize `y' }"),
    "global": (
        'global macname = expression\nglobal macname "text"',
        "Define a global macro. Reference with $macname. Persists across do-files."),
    "foreach": (
        "foreach macname in list { commands }\n"
        "foreach macname of varlist varlist { commands }\n"
        "foreach macname of numlist numlist { commands }\n"
        "foreach macname of local mname { commands }",
        "Loop over a list of values, variable names, or numbers.\n"
        "  foreach x of varlist x* { replace `x' = 0 if missing(`x') }"),
    "forvalues": (
        "forvalues i = start/end { commands }\n"
        "forvalues i = start(step)end { commands }",
        "Loop over a numeric range.\n"
        "  forvalues y = 2010/2020 { summarize income if year == `y' }"),
    "by": (
        "by varlist: command\nbysort varlist: command",
        "Run a command once per group. 'bysort' sorts first. When applied\n"
        "to egen, generate, or replace, _n becomes a per-group counter,\n"
        "_N is the per-group row count, and egen aggregations are per-group.\n"
        "  bysort city: egen mean_income = mean(income)\n"
        "  bysort firm_id: gen seq = _n"),
    "bysort": (
        "bysort varlist: command",
        "Sort by varlist, then run the command once per group. See 'help by'."),
    "quietly": (
        "quietly command",
        "Run a command with all output suppressed. Useful in loops."),
    "capture": (
        "capture command\ncapture noisily command",
        "Run a command and swallow any error. Sets _rc to 0 on success or\n"
        "non-zero on failure. With 'noisily', shows output but still captures.\n"
        "  capture confirm variable income\n"
        "  if _rc != 0 { display \"income not found\" }"),
    "return": (
        "return list",
        "Show the r() results stored by the most recent command.\n"
        "Typical keys: r(N), r(mean), r(sd), r(min), r(max)."),
    "ereturn": (
        "ereturn list",
        "Show the e() results stored by the most recent estimation.\n"
        "Typical keys: e(N), e(r2), e(rmse), e(b), e(V), e(method)."),

    # ────────────────── Programming & diagnostics ─────────────────────
    "assert": (
        "assert condition",
        "Halt with an error if the condition is false for any row.\n"
        "  assert age >= 0 & age <= 120\n"
        "  assert !missing(id)"),
    "preserve": (
        "preserve\n... (any commands) ...\nrestore",
        "Snapshot the current dataset. Call 'restore' to revert to the snapshot.\n"
        "Useful when you want to experiment without losing the original state."),
    "restore": (
        "restore",
        "Revert to the dataset captured by the most recent 'preserve'."),
    "egen": (
        "egen newvar = function(args) [if cond] [, by(groupvars)]",
        "Group-level and cross-row computations that plain 'generate' can't do.\n"
        "Respects an outer 'bysort group:' prefix as well.\n"
        "Functions: mean, median, sum, count, min, max, sd, total, rowtotal,\n"
        "rowmean, rowmin, rowmax, rowmiss, group, rank, tag, seq, concat, std.\n"
        "  egen avg_income = mean(income), by(country year)\n"
        "  egen z_income = std(income)"),
    "notes": (
        'notes                     (show all notes)\n'
        'notes : "text"            (add a note)\n'
        'notes drop N              (drop note number N)\n'
        'notes drop _all           (drop all notes)',
        "Attach free-text notes to the dataset. Notes travel with the dataset\n"
        "when saved."),
    "display": (
        'display expression [expression ...]\ndisplay "text" [expression] ["text" ...]',
        "Evaluate and display one or more tokens. Tokens may be quoted strings,\n"
        "math expressions, r()/e() results, or the _rc macro.\n"
        "  display 2 + 3 * sqrt(16)\n"
        "  display \"Mean = \" r(mean)\n"
        "  display \"rc=\" _rc"),
    "isid": (
        "isid varlist [, sort]",
        "Verify that varlist uniquely identifies observations. Errors on duplicates.\n"
        "  isid country year"),
    "levelsof": (
        "levelsof varname [if] [, local(macname) clean separate(sep)]",
        "List the unique values of a variable. With local(), stores them in\n"
        "a macro for later use.\n"
        "  levelsof country, local(ctys)"),
    "distinct": (
        "distinct [varlist] [if]",
        "Count distinct combinations of varlist (or the full dataset)."),
    "compress": (
        "compress [varlist]",
        "Downcast numeric types where values fit, reclaiming memory. Often\n"
        "frees 30-60% of memory on datasets loaded from CSV."),

    # ───────────────────────── Packages ─────────────────────────────
    "ssc": (
        "ssc install pkg1 [pkg2 ...]\n"
        "ssc remove pkg\n"
        "ssc list\n"
        "ssc search keyword\n"
        "ssc update pkg",
        "Package manager wrapping pip. Install optional backends that some\n"
        "commands require (diff-diff, argostranslate, sumy, polyfuzz,\n"
        "linearmodels, transformers, torch).\n"
        "  ssc install diff-diff\n"
        "  ssc install argostranslate sumy"),

    # ───────────────────────────── NLP ─────────────────────────────
    "nlp": (
        "nlp translate|summarize|classify|sentiment|ner|embed|models|download|cache",
        "Natural language processing commands. Offline-first where possible.\n"
        "  OFFLINE (no neural model required):\n"
        "    nlp translate var, from(src) to(tgt) ...    (via argos-translate)\n"
        "    nlp summarize var ...                       (via sumy)\n"
        "  NEURAL (uses the HuggingFace backend):\n"
        "    nlp classify | sentiment | ner | embed | models | cache\n"
        "  Downloads:\n"
        "    nlp download translate <src> <tgt>     (argos language pair)\n"
        "    nlp download <hf_model_name>           (HuggingFace weights)"),
    "hf": (
        "hf <task> ...   (alias for 'nlp')",
        "Older spelling of the NLP dispatcher. Every 'nlp' subcommand works\n"
        "under 'hf' too. See 'help nlp'."),
    "nlp translate": (
        "nlp translate var, from(src) to(tgt) [generate(newvar) pivot(en) if cond]",
        "Offline translation via argos-translate. Language pairs live in\n"
        "rln/argos_models/ and move with the folder between machines.\n"
        "If the direct pair isn't installed, Rln auto-downloads or pivots\n"
        "through English when both src->en and en->tgt are available.\n"
        "  nlp download translate id en\n"
        "  nlp translate text, from(id) to(en) generate(text_en)\n"
        "  nlp translate text, from(id) to(de) generate(text_de)   (pivots)"),
    "nlp summarize": (
        "nlp summarize var [, generate(newvar) sentences(3) method(lsa)\n"
        "                    language(english) if cond]",
        "Offline extractive summarization via sumy — picks the N most\n"
        "informative sentences from the source. No model download required.\n"
        "Methods: lsa (default), lexrank, textrank, luhn, kl, sumbasic, edmundson.\n"
        "  nlp summarize article, generate(summary) sentences(3) method(textrank)"),
    "hf classify": (
        "hf classify var, labels(\"lab1 lab2 lab3\") [generate(newvar) model(name) multi]",
        "Zero-shot text classification (multilingual). Default model:\n"
        "facebook/bart-large-mnli.\n"
        "  hf classify review, labels(\"positive negative neutral\")"),
    "hf sentiment": (
        "hf sentiment var [, generate(newvar) model(name)]",
        "Sentiment analysis. Default: multilingual BERT with 1-5 star output."),
    "hf summarize": (
        "hf summarize var [, generate(newvar) maxlen(150) minlen(30) model(name)]",
        "Neural abstractive summarization (generates new text). Contrast with\n"
        "'nlp summarize' which is offline and extractive."),
    "hf translate": (
        "hf translate var, from(src) to(tgt) [generate(newvar) model(name)]",
        "Neural machine translation (Helsinki-NLP / NLLB). Contrast with\n"
        "'nlp translate' which runs fully offline."),
    "hf ner": (
        "hf ner var [, generate(newvar) model(name)]",
        "Named entity recognition. Returns TYPE:text pairs."),
    "hf embed": (
        "hf embed var [, generate(stub) model(name) dims(N)]",
        "Sentence embeddings, expanded into stub_1 ... stub_N columns."),
    "hf models": (
        "hf models [task]",
        "List recommended neural models for each NLP task."),

    # ────────────────────────── LRTM (big data) ─────────────────────────
    "lrtm": (
        "lrtm <subcommand> [args]",
        "Larger-than-RAM mode for datasets that won't fit in memory.\n"
        "Uses the polars lazy engine — the query plan builds up without\n"
        "reading rows until you run 'lrtm collect' or 'lrtm save'.\n"
        "Subcommands: use, describe, summarize, tabstat, tabulate, count,\n"
        "generate, filter (keep/drop), sort, merge, fuzzmerge, append,\n"
        "collapse, contract, list, head, save, collect, status, clear, convert.\n"
        "See 'help lrtm <subcommand>' for details."),

    # ─────────────────────── System & utility ───────────────────────────
    "help": (
        "help [topic]",
        "Show help for a command or expression function. Without a topic,\n"
        "lists every command grouped by category."),
    "clear": (
        "clear",
        "Drop all data from memory. Local and global macros are preserved."),
    "pwd": (
        "pwd",
        "Print the current working directory."),
    "cd": (
        'cd "directory"',
        "Change the working directory. Affects relative paths in use, save, etc."),
    "dir": (
        "dir [pattern]",
        "List files in the current (or pattern-matched) directory.\n"
        "  dir *.csv"),
    "set": (
        "set [setting [value]]",
        "Show or change settings.\n"
        "  set on_error stop     halt do-files on error (default)\n"
        "  set on_error continue  keep running past errors\n"
        "  set seed 42            reproducible random sampling"),
    "memory": (
        "memory",
        "Report memory usage of the dataset and the Python process."),
    "do": (
        'do "filename.do"',
        "Execute a do-file. Supports * and // comments, /* block comments */,\n"
        "/// line continuation, nested do-files, and every scripting\n"
        "construct (local, global, foreach, forvalues, quietly, capture).\n"
        "On error the do-file stops unless 'set on_error continue' was set."),
    "doedit": (
        'doedit ["filename.do"]',
        "Open a terminal-based do-file editor with syntax highlighting.\n"
        "Ctrl-S saves, Ctrl-R runs."),
    "python": (
        "python: expr\npython { ... multi-line code ... }",
        "Run Python inline. The current dataset is available as 'df', pandas\n"
        "as 'pd', numpy as 'np'. Assigning to df reflects back into Rln.\n"
        "Works both interactively and inside do-files."),

    # ─────────── Expression functions (visible as 'help inlist' etc.) ───────
    "inlist": (
        "inlist(var, v1, v2, ...)",
        "TRUE where var matches any of the listed values. Works on strings\n"
        "and numbers.\n"
        "  gen bigcity = inlist(city, \"NYC\", \"LA\", \"Chicago\")\n"
        "  keep if inlist(age, 18, 21, 30, 45, 65)"),
    "inrange": (
        "inrange(expr, lo, hi)",
        "TRUE where lo <= expr <= hi. The first argument can be any expression,\n"
        "not just a bare variable.\n"
        "  gen workingage = inrange(age, 18, 65)\n"
        "  keep if inrange(year, 2010, 2020)"),
    "regexm": (
        'regexm(string, "pattern")',
        "TRUE where the regex pattern matches anywhere in the string. Standard\n"
        "PCRE syntax; escape backslashes as \\\\.\n"
        "  gen has_digit = regexm(name, \"[0-9]\")\n"
        "  keep if regexm(email, \"@gmail\\\\.com$\")"),
    "regexr": (
        'regexr(string, "pattern", "replacement")',
        "Replace every regex match with the replacement text.\n"
        "  gen clean_phone = regexr(phone, \"[^0-9]\", \"\")"),
    "regexs": (
        'regexs(string, "pattern_with_groups", n)',
        "Return capture group n (1-based) of the match. Non-matching rows\n"
        "become missing.\n"
        "  gen area = regexs(phone, \"\\\\(([0-9]+)\\\\)\", 1)\n"
        "  gen domain = regexs(email, \"@(.+)$\", 1)"),
    "real": (
        "real(string_var)",
        "Convert a string column to numeric. Unparseable values become NaN."),
    "string": (
        "string(numeric_var)",
        "Convert a numeric column to string."),
    "length": (
        "length(string_var)   |   strlen(string_var)",
        "Character count of a string variable.\n"
        "  gen name_len = length(name)"),
    "missing": (
        "missing(var)   |   !missing(var)",
        "TRUE if var is NaN/null.\n"
        "  drop if missing(income)\n"
        "  keep if !missing(email)"),
    "cond": (
        "cond(condition, true_value, false_value)",
        "Vectorized if/else — the expression-language equivalent of\n"
        "numpy.where.\n"
        "  gen bonus = cond(performance > 0.8, salary * 0.1, 0)"),

    # ───────────── Numeric / rounding functions ─────────────
    "round": (
        "round(x)   |   round(x, N)",
        "Round to the nearest integer (one-argument form) or to N decimal\n"
        "places (two-argument form). Uses banker's rounding at exact halves.\n"
        "  gen wage_int = round(wage)\n"
        "  gen rate_4dp = round(rate, 4)\n"
        "  summarize wage [fweight=round(pop_frac)]   (round() inside a weight clause)"),
    "floor": (
        "floor(x)",
        "Largest integer not greater than x. floor(3.7) = 3, floor(-3.7) = -4."),
    "ceil": (
        "ceil(x)",
        "Smallest integer not less than x. ceil(3.2) = 4, ceil(-3.7) = -3."),
    "trunc": (
        "trunc(x)   |   int(x)",
        "Truncate toward zero. trunc(3.7) = 3, trunc(-3.7) = -3. int() is\n"
        "an alias for trunc()."),
    "int": (
        "int(x)",
        "Alias for trunc(x). See 'help trunc'."),
    "abs": (
        "abs(x)",
        "Absolute value. abs(-3.5) = 3.5."),
    "sqrt": (
        "sqrt(x)",
        "Square root. sqrt(16) = 4."),
    "log": (
        "log(x)   |   ln(x)",
        "Natural logarithm (base e). ln() is an alias for log()."),
    "log10": (
        "log10(x)",
        "Base-10 logarithm."),
    "exp": (
        "exp(x)",
        "e raised to the x-th power. exp(0) = 1, exp(1) = 2.718..."),
    "mod": (
        "mod(x, y)",
        "Remainder of x divided by y. mod(7, 3) = 1."),
    "min": (
        "min(x, y) | min(x, y, z, ...)",
        "Elementwise minimum across the arguments.\n"
        "  gen worst_score = min(score1, score2, score3)"),
    "max": (
        "max(x, y) | max(x, y, z, ...)",
        "Elementwise maximum across the arguments."),

    # ───────────── Weight clauses (meta-help) ─────────────
    "weight": (
        "[fweight=expr]   [aweight=expr]   [pweight=expr]   [iweight=expr]",
        "Weight clause syntax for summarize, tabulate, tabstat, collapse,\n"
        "regress, logit, probit, poisson, nbreg, pctile, xtile, centile,\n"
        "winsor2, and friends.\n"
        "\n"
        "  fweight   Frequency weights (integer): row represents N obs\n"
        "  aweight   Analytic (inverse-variance) weights\n"
        "  pweight   Sampling (probability) weights. Forces robust SE in\n"
        "             regression commands.\n"
        "  iweight   Generic importance weights, no statistical semantics\n"
        "\n"
        "The right-hand side can be any expression that gen/replace would\n"
        "accept — a bare variable, arithmetic, or a function call.\n"
        "  summarize wage [fweight=pop]\n"
        "  summarize wage [fweight=round(pop_frac)]\n"
        "  regress y x [aweight=w1 + w2]\n"
        "  tabulate region gender [pweight=sqrt(survey_wt)]"),
    "fweight": ("see 'help weight'",
                "Frequency weights — integer counts. See 'help weight'."),
    "aweight": ("see 'help weight'",
                "Analytic weights — inverse-variance. See 'help weight'."),
    "pweight": ("see 'help weight'",
                "Sampling (probability) weights. See 'help weight'."),
    "iweight": ("see 'help weight'",
                "Importance weights. See 'help weight'."),
}


def cmd_help(rest: str, state: AppState, console: Console):
    """Show help for a command or list all commands."""
    topic = rest.strip().lower()

    if topic and topic in HELP_TOPICS:
        syntax, desc = HELP_TOPICS[topic]
        console.print(f"\n[bold cyan]{topic}[/bold cyan]")
        console.print(f"  [bold]Syntax:[/bold] {syntax}")
        console.print(f"  {desc}\n")
        return

    if topic:
        console.print(f"[yellow]No help for '{topic}'. Showing all commands.[/yellow]\n")

    # Show all commands grouped
    groups = {
        "Data I/O": ["use", "import", "save", "export", "log"],
        "Exploration": ["browse", "describe", "codebook", "list", "tabulate", "summarize", "count"],
        "Variables": ["generate", "replace", "rename", "drop", "keep", "label",
                      "destring", "tostring", "encode", "order", "recode", "reshape",
                      "clonevar", "split"],
        "Data Operations": ["sort", "gsort", "duplicates", "append", "merge", "fuzzmerge",
                            "collapse", "fillin", "cross", "sample"],
        "Estimation": ["regress", "predict", "test", "lincom", "margins",
                      "correlate", "pwcorr", "ttest"],
        "Panel & Causal": ["xtset", "xtreg", "didregress"],
        "Charts": ["histogram", "kdensity", "scatter", "twoway", "marginsplot",
                   "graph"],
        "Scripting": ["foreach", "forvalues", "local", "global",
                     "by", "bysort", "quietly", "capture"],
        "Programming": ["assert", "preserve", "restore", "egen", "notes",
                        "display", "python", "do", "doedit"],
        "Inspection": ["isid", "levelsof", "distinct", "compress", "memory",
                      "return", "ereturn"],
        "Packages": ["ssc"],
        "NLP (offline)": ["nlp translate", "nlp summarize"],
        "NLP (HuggingFace)": ["hf classify", "hf sentiment", "hf summarize",
                               "hf translate", "hf ner", "hf embed", "hf models"],
        "LRTM (Large Data)": ["lrtm"],
        "Expression functions": ["inlist", "inrange", "regexm", "regexr", "regexs",
                                  "missing", "cond", "real", "string", "length"],
        "Utility": ["help", "clear", "pwd", "cd", "dir", "set"],
    }

    console.print("\n[bold cyan]Rln Commands[/bold cyan]\n")

    for group_name, cmds in groups.items():
        console.print(f"  [bold]{group_name}[/bold]")
        for cmd in cmds:
            if cmd in HELP_TOPICS:
                syntax, _ = HELP_TOPICS[cmd]
                console.print(f"    [cyan]{cmd:14s}[/cyan] {syntax}")
        console.print()

    console.print("[dim]Type 'help <command>' for detailed help on any command.[/dim]\n")


def cmd_clear(rest: str, state: AppState, console: Console):
    """Clear all data from memory (pandas DataFrame + LRTM lazy frame)."""
    had_data = state.has_data()
    had_lrtm = hasattr(state, '_lrtm_lf') and state._lrtm_lf is not None
    name = state.dataset_name or "data"

    state.clear()

    if had_data or had_lrtm:
        parts = []
        if had_data:
            parts.append(f"DataFrame '{name}'")
        if had_lrtm:
            parts.append("LRTM lazy frame")
        console.print(f"[dim]Cleared: {', '.join(parts)}. RAM freed.[/dim]")
    else:
        console.print("[dim]No data in memory.[/dim]")


def cmd_pwd(rest: str, state: AppState, console: Console):
    """Print working directory."""
    console.print(os.getcwd())


def cmd_cd(rest: str, state: AppState, console: Console):
    """Change directory."""
    path = rest.strip().strip("\"'")
    if not path:
        path = os.path.expanduser("~")
    try:
        path = os.path.expanduser(path)
        os.chdir(path)
        console.print(os.getcwd())
    except OSError as e:
        console.print(f"[red]{e}[/red]")


def cmd_dir(rest: str, state: AppState, console: Console):
    """List files in current directory."""
    pattern = rest.strip().strip("\"'") if rest.strip() else "*"

    files = sorted(glob.glob(pattern))
    if not files:
        console.print("[dim]No files matching pattern.[/dim]")
        return

    # Supported data formats for highlighting
    data_exts = {".dta", ".csv", ".tsv", ".xlsx", ".xls", ".dbf",
                 ".rdata", ".rds", ".rda", ".html", ".htm",
                 ".parquet", ".pq", ".json"}

    for f in files:
        if os.path.isdir(f):
            console.print(f"  [bold blue]{f}/[/bold blue]")
        else:
            ext = os.path.splitext(f)[1].lower()
            size = os.path.getsize(f)
            size_str = _format_size(size)

            if ext in data_exts:
                console.print(f"  [green]{f:<40s}[/green] {size_str:>10s}")
            else:
                console.print(f"  {f:<40s} {size_str:>10s}")


def cmd_set(rest: str, state: AppState, console: Console):
    """Show or change settings."""
    parts = rest.strip().split(None, 1)

    if not parts or not parts[0]:
        # Show all settings
        console.print("\n[bold]Current Settings[/bold]")
        for key, val in state.settings.items():
            console.print(f"  {key:20s} = {val}")
        console.print()
        return

    key = parts[0].lower()
    if len(parts) < 2:
        if key in state.settings:
            console.print(f"  {key} = {state.settings[key]}")
        else:
            console.print(f"[red]Unknown setting: {key}[/red]")
        return

    value = parts[1].strip()

    # compact `set on_error stop/continue` — governs whether do-files
    # halt on an error (the default) or plough through.
    if key == "on_error":
        choice = value.lower()
        if choice not in ("stop", "continue"):
            console.print("[red]set on_error: must be 'stop' or 'continue'[/red]")
            return
        state.on_error = choice
        console.print(f"[green]on_error = {choice}[/green]")
        return

    # Try to convert to int/float
    try:
        value = int(value)
    except ValueError:
        try:
            value = float(value)
        except ValueError:
            pass

    if key in state.settings:
        state.settings[key] = value
        console.print(f"[green]Set {key} = {value}[/green]")
    else:
        console.print(f"[red]Unknown setting: {key}[/red]")
        console.print(f"[dim]Available: {', '.join(list(state.settings.keys()) + ['on_error'])}[/dim]")


def cmd_memory(rest: str, state: AppState, console: Console):
    """Show memory usage."""
    if not state.has_data():
        console.print("[dim]No dataset in memory.[/dim]")
        return

    mem = state.data.memory_usage(deep=True)
    total = mem.sum()

    console.print(f"\n[bold]Memory Usage[/bold]")
    console.print(f"  Dataset: {state.dataset_name or 'unnamed'}")
    console.print(f"  Total:   {_format_size(total)}")
    console.print(f"\n  [dim]Top variables by memory:[/dim]")

    # Show top 10 columns by memory
    col_mem = mem.drop("Index", errors="ignore").sort_values(ascending=False)
    for col, size in col_mem.head(10).items():
        pct = size / total * 100
        console.print(f"    {col:30s}  {_format_size(size):>10s}  ({pct:.1f}%)")

    console.print()


def _format_size(nbytes: int) -> str:
    """Format byte size to human readable."""
    if nbytes >= 1e9:
        return f"{nbytes/1e9:.1f} GB"
    elif nbytes >= 1e6:
        return f"{nbytes/1e6:.1f} MB"
    elif nbytes >= 1e3:
        return f"{nbytes/1e3:.1f} KB"
    return f"{nbytes} B"


def cmd_do(rest: str, state: AppState, console: Console):
    """
    do "filename.do"
    Execute a do-file from within the REPL.
    """
    filepath = rest.strip().strip("\"'")
    if not filepath:
        console.print('[red]Syntax: do "filename.do"[/red]')
        return

    from commands.parser import CommandParser
    from main import run_do_file

    parser = CommandParser(state, console)
    run_do_file(filepath, parser, console)
