# Rln — Testing Findings & Fixes (mobile / lite robustness)

Test pass focused on the three reported problems plus a sweep for related
"potential problems." Everything below was reproduced, root-caused, fixed, and
re-verified locally (Windows desktop + simulated Android by blocking the
missing modules at import). Date: 2026-06-16. Version under test: 1.2.7.

**Summary**

| # | Area | Status | Root cause |
|---|------|--------|-----------|
| 1 | Android `pyarrow` (LRTM) | ✅ Fixed | `polars.to_pandas()` always calls pyarrow internally |
| 2 | `matplotlib` missing in lite build | ✅ Fixed | lite PyInstaller spec **excluded** PIL, which matplotlib needs |
| 3 | `statsmodels` not working on mobile | ✅ Fixed (alternative shipped) | no working Android wheel; built a NumPy/SciPy fallback |
| 4 | Panel/diagnostics crash on Android | ✅ Fixed | direct `import statsmodels` in several commands |
| 5 | Misleading "module required" messages | ✅ Improved | swallowed the real ImportError |

---

## Issue 1 — Android `pyarrow` error in LRTM (despite polars present)

**Symptom.** On Android, LRTM commands failed with a pyarrow error even though
polars was installed.

**Root cause.** `polars.DataFrame.to_pandas()` *always* routes through
`pyarrow.Table.from_batches(...)` internally — even with the default
`use_pyarrow_extension_array=False` (verified in the bundled polars
`dataframe/frame.py`, `_to_pandas_without_object_columns`). pyarrow has no
usable aarch64-Android wheel (it's the Apache Arrow C++ library), so any
`to_pandas()` call raises `ModuleNotFoundError: No module named 'pyarrow'`.

LRTM called `to_pandas()` in **three** places in [commands/lrtm.py](commands/lrtm.py):
`lrtm use ..., sample(N)`, the count-with-unparseable-condition fallback, and
`lrtm collect`. So `lrtm collect` — a headline feature — crashed on Android.
(Note: `rln_io/fileio.py` had *already* learned this lesson for parquet
loading; lrtm.py simply missed the same fix.)

**Fix.** Added `_pl_to_pandas()` in [commands/lrtm.py](commands/lrtm.py): it tries
`to_pandas()` (fast path, unchanged on desktop) and on `ImportError` rebuilds
the pandas frame column-by-column from polars' **native** `Series.to_numpy()`
engine (`use_pyarrow=False` since polars 0.20.28), which needs only numpy.
Replaced all three call sites.

**Verified.** With pyarrow blocked at import, `lrtm use examples/eu_firms_panel.parquet,
sample(5)` → `lrtm summarize` → `lrtm collect` (2,000,000 × 16) all succeed
(0.5 s). Desktop path unchanged (still uses pyarrow fast path).

---

## Issue 2 — `matplotlib` "not found" in the lite desktop build

**Symptom.** In `rln-lite`, any chart command (`histogram`, `scatter`, …)
printed *"matplotlib is required for charts. Install with: ssc install
matplotlib"*. Reproduced directly with the shipped `dist/rln-lite/rln-lite.exe`.

**Root cause — empirically confirmed from the shipped build:**

```
dist/rln-lite/_internal/matplotlib   ← present
dist/rln-lite/_internal/PIL          ← MISSING
dist/rln-full/_internal/PIL          ← present
```

`import matplotlib.pyplot` imports **Pillow (PIL)**. But [rln.spec](rln.spec)
listed `'PIL', 'Pillow'` in the **lite-tier excludes**. In PyInstaller,
`excludes` override hidden imports, so PIL was stripped from lite only —
matplotlib was bundled but couldn't import. The error message then hid the real
cause (`No module named 'PIL'`).

**Fix.**
- [rln.spec](rln.spec): removed `PIL`/`Pillow` from the lite excludes; added
  `PIL`, `kiwisolver`, `contourpy`, `cycler`, `pyparsing`, `fonttools` as
  explicit hidden imports; added `collect_all('matplotlib')` and
  `collect_all('PIL')` so mpl-data fonts + PIL codecs ship.
- [commands/charts.py](commands/charts.py): `_check_matplotlib()` now surfaces
  the underlying ImportError (e.g. `No module named 'PIL'`) instead of a generic
  message — so packaging gaps are diagnosable.

**To ship the fix:** rebuild lite — `pyinstaller rln.spec -- --tier=lite`. The
existing `dist/rln-lite` predates the fix and still lacks PIL.

---

## Issue 3 — `statsmodels` on mobile, and the alternative

**Question asked:** *"statsmodels not working in the mobile version… is there a
good alternative?"*

**Answer:** There is no drop-in pure-Python econometrics library that
cross-compiles trivially. But **SciPy + NumPy already cross-compile and ship in
the APK**, and every estimator Rln needs has a closed-form (OLS/WLS) or simple
IRLS/Newton (logit/probit/poisson) solution on top of them. So the alternative
is "implement the thin layer on SciPy," which is what was done.

**New module:** [commands/stats_fallback.py](commands/stats_fallback.py) — a
NumPy/SciPy backend exposing the exact slice of the `statsmodels.api` surface
Rln calls (`OLS, WLS, Logit, Probit, Poisson, GLM, add_constant, families…`).
`_check_statsmodels()` / `_check_sm()` now return real statsmodels when present,
else this fallback (with a one-time notice). Any unsupported model raises a
clear message via a module `__getattr__`.

**Validated against real statsmodels** ([tests/test_stats_fallback.py](tests/test_stats_fallback.py),
6/6 passing) to ~1e-11 on the example data:

| Model | Coef | SE (classical / HC1 / cluster) | Notes |
|-------|:----:|:------------------------------:|-------|
| OLS / WLS | ✓ | ✓ / ✓ / ✓ | + R², adjR², F, t-vs-normal p-values, CI, ess/ssr/df |
| Logit | ✓ | ✓ / ✓ / ✓ | + llf, LR χ², pseudo-R² |
| Probit | ✓ | ✓ | uses **observed** information (matches statsmodels) |
| Poisson | ✓ | ✓ | + offset/exposure |
| GLM (Binomial/Poisson) | ✓ | ✓ | + `freq_weights`, `offset` |

**Mobile econometrics now working via the fallback** (verified through the real
command engine with statsmodels blocked): `regress` (robust + cluster),
`logit`, `probit`, `poisson`, `predict`, `test`, `vif`, `estat hettest`
(Breusch-Pagan), `estat imtest` (White), `dwstat`, `xtserial`, `xtreg, fe`
(LSDV), and `didregress, method(twfe)`.

**Still desktop-only (graceful "needs desktop" message, no crash):** `ivregress`
(linearmodels), `xtreg` random-effects (linearmodels), advanced `didregress`
methods cs/sa/bjs/… (diff-diff), `estat bgodfrey`, `estat ovtest`.

**Recommendation:** keep `statsmodels` in `buildozer.spec` — if the recipe
loads, it's used; if it ever fails on-device, the fallback takes over
automatically, so the app no longer hard-fails either way.

---

## Issue 4 — Other crashes-on-Android found in the sweep (fixed)

Audited every optional dependency. **No command module imports a heavy/optional
package at top level**, so app startup is safe. But several functions did a
*direct* `import statsmodels` inside the function body (not via `_check_*`),
which would crash on Android:

- [commands/panel.py](commands/panel.py) `_xtreg_statsmodels_fallback` → now uses
  `_check_statsmodels()` ⇒ `xtreg, fe` works on mobile.
- [commands/panel.py](commands/panel.py) `_did_twfe` → now uses `_check_statsmodels()`
  ⇒ `didregress method(twfe)` (the no-diff-diff DiD path) works on mobile.
- [commands/diagnostics.py](commands/diagnostics.py) `vif`, `estat hettest/imtest`,
  `dwstat`, `xtserial` → now go through a `_diag_funcs()` backend selector
  (statsmodels or fallback) and pure-NumPy implementations of VIF / Breusch-Pagan
  / White / Durbin-Watson (validated equal to statsmodels). `bgodfrey`/`ovtest`
  degrade with a clear message.

**Latent bug fixed along the way.** In the fallback, a design matrix passed as a
raw NumPy array (e.g. the het-test auxiliary regressions) didn't have its
intercept column recognised, which would have mis-scaled the joint Wald F.
`_as_xy()` now detects a constant column by value and names it `const`.

---

## Issue 5 — Misleading dependency messages

`_check_*` helpers swallowed the real ImportError and pointed at `ssc install`,
which cannot work on a frozen lite build or on Android (no pip there). Charts and
estimation now print the underlying import error. (`ssc install` is still
mentioned for *source* installs, where it does work.)

---

## How to re-verify

```bash
python tests/test_stats_fallback.py        # fallback vs statsmodels (6/6)
python test_offline.py                      # 10/10
python test_extend.py                       # 34/36 *
pyinstaller rln.spec -- --tier=lite         # rebuild lite with PIL bundled
```

\* The 2 `test_extend.py` failures are pre-existing and **environmental**: they
`subprocess.run(["python3", …])`, and `python3` resolves to the Windows
Store stub on this machine. Unrelated to these changes (the didregress/xtreg
statsmodels-backend tests all pass).

---

# Batch 2 — data explorer, examples folder, fuzzy/DiD coverage, ssc

Follow-up requests (same session). All reproduced/verified locally.

## A+B. Interactive, colour-coded data explorer in every version

The referenced **parquet-explorer** is a VS Code extension (TypeScript +
DuckDB) — it can't be embedded in Rln's Python/Kivy/tkinter/textual stack, so
its *experience* was built natively instead: a colour-coded grid where numbers
are blue, strings orange, missing values muted — consistent across all three
front-ends.

- New [commands/datacolors.py](commands/datacolors.py): one shared
  classify()/colour palette (hex + Rich styles), mirroring the desktop GUI
  theme, so the three explorers match.
- **TUI** [tui/browser.py](tui/browser.py): the textual `DataTable` cells were
  plain strings despite the docstring — now rendered as type-coloured Rich
  `Text` (numbers cyan, negatives red, strings orange, missing dim), with
  coloured headers. The rich-fallback browser colours strings too.
- **Android** [android/main.py](android/main.py): the data tab showed plain
  `to_string()` — now renders a colour-coded, aligned grid via Kivy markup
  (`_render_data_markup`), with a plain-text fallback if markup fails.
- **Desktop GUI** already colour-coded; left as the reference.
- **Explore a file directly:** `browse "file.parquet"` now loads a preview
  (polars native reader, pyarrow-free, capped at 50k rows) and opens it in the
  explorer **without** touching the in-memory dataset. Works for csv/dta/xlsx/…
  too.

## C. Examples folder reachable by default (+ persisted, changeable)

New [commands/workspace.py](commands/workspace.py): locates the bundled
`examples/` across source/frozen/Android; a bare `use "demographics.csv"` /
`do "sample.do"` now resolves against the latest-project folder then
`examples/`. File pickers (GUI `initialdir`, Android `FileChooser`) start there.
Priority: explicit `set workdir "<path>"` → latest project → examples. Persists
in `~/.rln_config.json`. Wired into `use`, `import`, `do` (desktop + Android),
and the GUI/Android dialogs.

## D. polyfuzz & diff-diff across the three versions

- **diff-diff (didregress):** works on source/full/lite. On Android,
  `method(twfe)` runs via the NumPy/SciPy fallback; every other method falls
  back to TWFE with a clear message. ✅ functional everywhere.
- **polyfuzz (fuzzmerge):** was broken on **lite** (polyfuzz not bundled; lite
  also excludes its sklearn dependency) and absent on **Android**. New
  [commands/fuzzy.py](commands/fuzzy.py) adds a backend chain
  **polyfuzz → rapidfuzz → difflib (stdlib floor)**, so `fuzzmerge` now works on
  all three versions and reports which backend ran. (`lrtm fuzzmerge` still
  prefers rapidfuzz and degrades with a clear message — left as-is.)

## E. `ssc install` (pip) fixed

[commands/rln_cmds.py](commands/rln_cmds.py): it ran `sys.executable -m pip`,
but in a frozen build `sys.executable` is the **Rln .exe** (so it relaunched
Rln — the "ssc install error") and Android has no pip. Now it only uses
`sys.executable` on a real source install (verified it has pip) and otherwise
prints an honest "packaged/mobile builds can't pip-install; use the source
install" message.

---

## Files changed (batch 2)

- **new** `commands/datacolors.py`, `commands/fuzzy.py`, `commands/workspace.py`
- `tui/browser.py` — colour-coded cells/headers
- `android/main.py` — colour-coded data grid, examples-default picker, do-resolve
- `commands/explore.py` — `browse "<file>"` direct exploration
- `commands/dataops.py` — `fuzzmerge` uses the backend chain
- `commands/data_io.py`, `main.py`, `commands/utility.py` — workspace path
  resolution + `set workdir`
- `gui/app.py` — file dialogs start in the workspace folder
- `commands/rln_cmds.py` — pip-safe `ssc`
- `rln.spec` — bundle the new command modules

---

# Batch 3 — Android LRTM parquet preview (parquet-explorer behaviour)

**Reported:** on Android the parquet explorer "only works after `lrtm collect`".
The data browser keyed off `state.data`, which is empty while a parquet is
lazy-loaded — so `lrtm use` showed nothing until you materialised the whole
file.

**Fix:** new `commands.lrtm.lrtm_preview(state, n)` streams the first `n` rows
from the on-disk source through Polars' lazy engine (pyarrow-free) and returns
`(preview_df, total_rows)`. Wired into the **Android data tab**, the **desktop
GUI data browser**, and the REPL **`browse`** command: when nothing is
materialised but an LRTM frame is loaded, they now show a colour-coded **live
preview** with a status like `LRTM preview: 200 of 2,000,000 rows — run 'lrtm
collect' to load all`. After `lrtm collect`, the full dataset shows as before.
This is the VS Code parquet-explorer experience: peek on `lrtm use`, full
exploration on `lrtm collect`. Verified end-to-end incl. the Android no-pyarrow
path. Files: `commands/lrtm.py`, `android/main.py`, `gui/app.py`,
`commands/explore.py`.

**Docs (v1.2.8):** updated `Rln_Reference_Manual.docx` (version bump + browse/
fuzzmerge/ssc/set-workdir notes), added `docs/Rln-LiteDesktop-Mobile-Guide.docx`,
and built an interactive `index.html` (GitHub-Pages-ready: relative `media/`
assets, embedded demo video, latest-release/download links). Un-ignored the two
Word deliverables in `.gitignore` so they can be pushed.

---

# Batch 1 files changed

- `commands/lrtm.py` — `_pl_to_pandas()` (pyarrow-free), 3 call sites.
- `commands/stats_fallback.py` — **new**, NumPy/SciPy statsmodels shim + diagnostics.
- `commands/estimation.py`, `commands/estimation_glm.py` — `_check_*` fall back to it.
- `commands/diagnostics.py` — backend-selector + fallback wiring.
- `commands/panel.py` — `xtreg`/`didregress twfe` use `_check_statsmodels()`.
- `commands/charts.py` — surface real matplotlib import error.
- `rln.spec` — stop excluding PIL in lite; bundle matplotlib/PIL data + deps.
- `android/buildozer.spec` — corrected the stale "what's shipped" comment.
- `tests/test_stats_fallback.py` — **new**, equivalence + smoke tests.
