"""python-for-android recipe for statsmodels (compiled from source).

statsmodels powers Rln's econometrics commands (regress / logit / probit /
poisson / ivregress / didregress) and several diagnostics. It has no published
Android wheel, but termux/termux-packages#19960 proved it compiles cleanly for
android arm64 / CPython 3.11 with three environment tweaks:

    CFLAGS+=" -U__ANDROID_API__ -D__ANDROID_API__=33"
    MATHLIB=m
    LDFLAGS="-lpython3.11"
    python -m pip install . --no-build-isolation

This recipe reproduces that inside p4a: a PEP-517 build (`python -m build
--wheel --no-isolation`) against the cross-built numpy/scipy/pandas already in
the dist, with the same flags. Building from source (rather than dropping in a
prebuilt wheel) means it links against our exact numpy 2.3, avoiding any C-ABI
mismatch.
"""

from os.path import join

from pythonforandroid.recipe import PyProjectRecipe


class StatsmodelsRecipe(PyProjectRecipe):
    version = "0.14.4"
    url = ("https://github.com/statsmodels/statsmodels/archive/"
           "refs/tags/v{version}.tar.gz")

    # Runtime deps that must be cross-built and present in the APK first.
    depends = ["numpy", "scipy", "pandas"]
    # Build-time deps installed into the host build venv (used because we build
    # with --no-isolation, exactly like the Termux command above). scipy is
    # required at *build* time too: statsmodels cythonizes against scipy's
    # cython_blas/cython_lapack .pxd (e.g. zaxpy). Cython pinned to the 3.0.x
    # line that statsmodels 0.14.4 was written for.
    hostpython_prerequisites = ["numpy>=2.0", "scipy", "Cython>=3.0.10,<3.1",
                                "setuptools_scm[toml]>=8,<9"]
    extra_build_args = ["--no-isolation"]

    def get_recipe_env(self, arch, **kwargs):
        env = super().get_recipe_env(arch, **kwargs)
        # The GitHub tarball has no .git, so setuptools_scm can't infer the
        # version; pin it explicitly (this is why the Termux build was "0.0.0").
        env["SETUPTOOLS_SCM_PRETEND_VERSION"] = self.version
        env["SETUPTOOLS_SCM_PRETEND_VERSION_FOR_STATSMODELS"] = self.version
        # Proven flags from termux/termux-packages#19960, plus two fixes for
        # NDK r28c's clang-19 (the Termux user had an older, lenient clang):
        #   -include complex.h   : declare cpow/cpowf (Bionic has them, API24+)
        #                          so the complex-power paths compile correctly
        #   -Wno-error=implicit-function-declaration : net for any other
        #                          implicit decls clang-19 would now hard-error
        env["CFLAGS"] = env.get("CFLAGS", "") + \
            " -U__ANDROID_API__ -D__ANDROID_API__=33" \
            " -include complex.h" \
            " -Wno-error=implicit-function-declaration"
        env["MATHLIB"] = "m"
        # statsmodels links -lnpymath; numpy.get_include() resolves it to the
        # HOST (x86_64) libnpymath.a. Put the cross-built (arm64) numpy lib dir
        # first so -lnpymath finds the aarch64 archive instead.
        target_np_lib = join(self.ctx.get_python_install_dir(arch.arch),
                             "numpy", "_core", "lib")
        # Explicitly link libm: the extensions reference cpow/cos/cexp/etc. as
        # undefined symbols, but MATHLIB=m alone did NOT add libm.so to their
        # DT_NEEDED, so dlopen failed on-device ("import statsmodels" -> the
        # generic 'statsmodels is required'). -lm makes libm a real dependency.
        env["LDFLAGS"] = env.get("LDFLAGS", "") + \
            " -L{} -L{} -lpython{} -lm".format(
                target_np_lib,
                self.ctx.python_recipe.link_root(arch.arch),
                self.ctx.python_recipe.link_version,
            )
        return env


recipe = StatsmodelsRecipe()
