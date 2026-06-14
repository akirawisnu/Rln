"""Prebuilt polars recipe for Rln on Android.

Polars must be built with Rust *nightly* (it uses unstable `#![feature]`),
which python-for-android's toolchain can't readily provide. Instead of
cross-compiling the large Rust workspace, we ship the prebuilt arm64 wheel from
the Termux User Repository (polars 1.9.0). Its binary is `polars/polars.abi3.so`
— a CPython *stable-ABI* (abi3) extension, so it loads on Python 3.10+ including
our bundled 3.11.

The only adjustment needed: Termux links the versioned interpreter soname
`libpython3.11.so.1.0`, whereas our APK ships `libpython3.11.so`. The staged
binary under ``prebuilt/`` has already been patched (patchelf --replace-needed)
to depend on our soname, so the dynamic linker resolves it at import time.

This recipe performs no compilation; it simply copies the prebuilt package tree
into the per-arch python install dir that p4a collects into the app's
site-packages bundle (the same place numpy/scipy land).
"""

import shutil
from os import makedirs
from os.path import join, dirname, exists

from pythonforandroid.recipe import PythonRecipe
from pythonforandroid.logger import info

RECIPE_DIR = dirname(__file__)

# Top-level entries produced by the prebuilt wheel (see prebuilt/).
_PREBUILT_ENTRIES = (
    "polars",
    "polars-libs",
    "polars-licenses",
    "polars-1.9.0.dist-info",
)


class PolarsRecipe(PythonRecipe):
    version = "1.9.0"
    url = None  # prebuilt: nothing to download or unpack
    site_packages_name = "polars"
    depends = ["python3"]
    # The arm64 abi3 binary is prebuilt; never invoke host/target build tooling.
    call_hostpython_via_targetpython = False

    def should_build(self, arch):
        # Cheap idempotent copy; always (re)install to avoid stale state.
        return True

    def build_arch(self, arch):
        install_dir = self.ctx.get_python_install_dir(arch.arch)
        if not exists(install_dir):
            makedirs(install_dir)
        prebuilt = join(RECIPE_DIR, "prebuilt")
        info("polars: installing prebuilt arm64 package into {}".format(
            install_dir))
        for entry in _PREBUILT_ENTRIES:
            src = join(prebuilt, entry)
            if not exists(src):
                continue
            dst = join(install_dir, entry)
            if exists(dst):
                shutil.rmtree(dst)
            shutil.copytree(src, dst)
        info("polars: prebuilt install complete")


recipe = PolarsRecipe()
