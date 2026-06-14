"""
Rln local override of the p4a `develop` numpy recipe.

Why this override exists
------------------------
The stock develop numpy recipe (a MesonRecipe) breaks when `libopenblas` is in
the build order — which happens as soon as `scipy` is also requested, because
scipy depends on libopenblas. In that case the stock `get_recipe_env` does:

    self.extra_build_args = ["-Csetup-args=-Dblas=auto", ...]

That is a *reassignment*, and it clobbers the `--cross-file` argument that
`MesonRecipe.build_arch` had already appended to `extra_build_args` via
`ensure_args`. With the cross-file gone, Meson configures a "native build",
compiles the C sanity-check program for arm64, then tries to *run* it on the
x86_64 host and dies with:

    ERROR: Could not invoke sanity test executable: [Errno 8] Exec format error

The only change here vs. upstream is in `get_recipe_env`: instead of replacing
`extra_build_args` wholesale, we preserve every arg that isn't a blas/lapack
selector (notably the two `--cross-file` entries) and only swap the
blas/lapack selection over to libopenblas. Everything else is identical to the
develop recipe so behaviour is otherwise unchanged.
"""

from pythonforandroid.recipe import Recipe, MesonRecipe
from os.path import join
import shutil

NUMPY_NDK_MESSAGE = (
    "In order to build numpy, you must set minimum ndk api (minapi) to `24`.\n"
)


class NumpyRecipe(MesonRecipe):
    version = "v2.3.0"
    url = "git+https://github.com/numpy/numpy"
    extra_build_args = ["-Csetup-args=-Dblas=none", "-Csetup-args=-Dlapack=none"]
    opt_depends = ["libopenblas"]
    need_stl_shared = True
    min_ndk_api_support = 24

    def get_include(self, arch):
        return join(
            self.ctx.get_python_install_dir(arch.arch), "numpy/_core/include",
        )

    def get_recipe_meson_options(self, arch):
        options = super().get_recipe_meson_options(arch)
        options["properties"]["longdouble_format"] = (
            "IEEE_DOUBLE_LE" if arch.arch in ["armeabi-v7a", "x86"] else "IEEE_QUAD_LE"
        )
        return options

    def get_recipe_env(self, arch, **kwargs):
        env = super().get_recipe_env(arch, **kwargs)

        # _PYTHON_HOST_PLATFORM declares that we're cross-compiling
        # and avoids issues when building on macOS for Android targets.
        env["_PYTHON_HOST_PLATFORM"] = arch.command_prefix

        # NPY_DISABLE_SVML=1 allows numpy to build for non-AVX512 CPUs
        # See: https://github.com/numpy/numpy/issues/21196
        env["NPY_DISABLE_SVML"] = "1"
        env["TARGET_PYTHON_EXE"] = join(
            Recipe.get_recipe("python3", self.ctx).get_build_dir(arch.arch),
            "android-build",
            "python",
        )
        blas_dir = join(Recipe.get_recipe("libopenblas", self.ctx
        ).get_build_dir(arch.arch), "build")
        blas_incdir = blas_dir
        blas_libdir = join(blas_dir, "lib")
        env["CXXFLAGS"] += f" -I{blas_incdir} -L{blas_libdir}"

        if 'libopenblas' in self.ctx.recipe_build_order:
            # Rln fix: do NOT clobber extra_build_args — that would drop the
            # --cross-file added by MesonRecipe.build_arch and force a broken
            # "native build". Keep every non-blas/lapack arg (the cross-file
            # entries) and only switch the blas/lapack selection to auto.
            preserved = [
                a for a in self.extra_build_args
                if "blas=" not in a and "lapack=" not in a
            ]
            self.extra_build_args = preserved + [
                "-Csetup-args=-Dblas=auto",
                "-Csetup-args=-Dlapack=auto",
                "-Csetup-args=-Dallow-noblas=False",
            ]

        return env

    def get_hostrecipe_env(self, arch=None):
        env = super().get_hostrecipe_env(arch=arch)
        env["RANLIB"] = shutil.which("ranlib")
        return env


recipe = NumpyRecipe()
