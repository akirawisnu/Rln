# Rln for Android

A phone-first build of Rln packaged as an Android APK. It reuses the **exact
same command engine** as the desktop CLI and GUI (`commands/` + `rln_io/`), so
every command parses and behaves identically. Only the front-end differs: a
single vertical (portrait) Kivy screen suited to phones.

## What's included

- Data IO: `use`, `import`, `save`, `export` (CSV, Parquet, Stata `.dta`, …)
- Exploration: `describe`, `summarize`, `tabulate`, `codebook`, `list`, `count`
- Variables / dataops: `gen`, `replace`, `recode`, `reshape`, `collapse`, `merge`, …
- Econometrics: `regress`, `logit`, `probit`, `poisson`, `ivregress`, quantile reg
- Panel & DiD: `xtset`, `didregress` (Callaway–Sant'Anna, Sun–Abraham, etc.)
- Diagnostics: `vif`, `estat hettest`, …
- Charts: rendered to an image inline (matplotlib Agg backend)
- Scripting: `foreach`, `forvalues`, `python:` / `python { }`
- **LRTM** (Larger-than-RAM Mode): `lrtm use/describe/collapse/...` via polars

## What's excluded (on purpose)

NLP / transformers / translation / summarization (`nlp …`, `hf …`). These pull
torch + transformers + argos + sumy, which are not part of the Android
dependency set. Calling them on Android just reports the optional packages are
unavailable.

## Building the APK (from Windows, via WSL Ubuntu)

The Android toolchain (python-for-android / buildozer) only runs on Linux, so we
build from WSL. The build compiles the entire scientific stack for `arm64-v8a`
and is **slow on the first run** (downloads the Android SDK/NDK and builds
numpy/scipy/pandas/matplotlib/statsmodels from source).

### 1. One-time system dependencies (needs your password — run this yourself)

Open WSL and paste:

```bash
sudo apt update && sudo apt install -y git zip unzip openjdk-17-jdk \
  python3-pip python3-venv autoconf libtool pkg-config zlib1g-dev \
  libncurses-dev libncursesw5-dev libtinfo6 cmake libffi-dev libssl-dev \
  build-essential ccache
```

### 2. Run the build

```bash
wsl -d Ubuntu -- bash /mnt/e/StatB/Rln/android/build_apk.sh
```

This stages a clean copy of the app + engine on the native WSL filesystem
(`~/rln-android`), installs `buildozer` (user-level pip, no sudo), and runs
`buildozer android debug`. The finished APK lands in:

```
~/rln-android/bin/rln-1.2.7-arm64-v8a-debug.apk
```

Copy it back to Windows with:

```bash
cp ~/rln-android/bin/*.apk /mnt/e/StatB/Rln/android/
```

Then transfer to your phone and install (enable "install from unknown
sources").

## Known cross-compilation risks

The first build will tell us empirically whether two packages cross-compile for
Android. Both are listed in `buildozer.spec` under `requirements`:

- **polars** (Rust) — powers LRTM. No upstream Android wheel; needs the Rust
  toolchain to cross-compile for `aarch64-linux-android`. This is the single
  biggest risk. If it blocks the build, remove `polars` from the `requirements`
  line to get a working APK without LRTM, then revisit it with a custom p4a
  recipe.
- **statsmodels** (Cython) — powers `regress`/`didregress`. Usually pip-buildable
  under p4a, but it compiles C/Cython extensions, so watch the log.

To produce a guaranteed-working APK first and add the risky pieces afterward,
trim the `requirements` line in `buildozer.spec` to:

```
requirements = python3,kivy==2.3.0,pillow,rich,numpy,pandas,scipy,matplotlib,patsy,python-dateutil,pytz,packaging,setuptools
```

…build, confirm the app launches on the phone, then add `statsmodels`,
`linearmodels`, `diff-diff`, and finally `polars` back one at a time.

## Files

- `main.py` — the Kivy phone front-end (buildozer entry point)
- `buildozer.spec` — Android build configuration
- `build_apk.sh` — WSL build driver (stages engine + runs buildozer)
