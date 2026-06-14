#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Build the Rln Android APK from WSL (Ubuntu).
#
# One-time prerequisites (need sudo — run the line printed in README_ANDROID.md
# before the first build):
#   sudo apt update && sudo apt install -y git zip unzip openjdk-17-jdk \
#        python3-pip python3-venv autoconf libtool pkg-config zlib1g-dev \
#        libncurses-dev libncursesw5-dev libtinfo6 cmake libffi-dev libssl-dev \
#        build-essential ccache
#
# Then just run:
#   wsl -d Ubuntu -- bash /mnt/e/StatB/Rln/android/build_apk.sh
#
# buildozer (pip, user-level) and the Android SDK/NDK are installed/downloaded
# automatically on the first run. The build happens on the native WSL
# filesystem (~/rln-android) because buildozer needs symlinks and fast IO that
# /mnt/* (the Windows drive) does not provide reliably.
# ---------------------------------------------------------------------------
set -euo pipefail

SRC="/mnt/e/StatB/Rln"
DEST="$HOME/rln-android"

echo ">> Rln Android build"
echo "   source : $SRC"
echo "   workdir: $DEST"

# 1. Stage a clean copy of the Kivy app + engine on the native filesystem.
mkdir -p "$DEST"
cp -f  "$SRC/android/main.py"        "$DEST/main.py"
cp -f  "$SRC/android/buildozer.spec" "$DEST/buildozer.spec"
cp -f  "$SRC/android/icon.png"       "$DEST/icon.png"
# Local p4a recipe overrides (python3/hostpython3 pinned to 3.11; statsmodels
# kept for Phase 2). buildozer.spec points p4a.local_recipes at ./p4a-recipes,
# so the tree must sit beside the spec.
rm -rf "$DEST/p4a-recipes"
if [ -d "$SRC/android/p4a-recipes" ]; then
    cp -rf "$SRC/android/p4a-recipes" "$DEST/p4a-recipes"
fi
rm -rf "$DEST/commands" "$DEST/rln_io"
cp -rf "$SRC/commands" "$DEST/commands"
cp -rf "$SRC/rln_io"   "$DEST/rln_io"
if [ -d "$SRC/examples" ]; then
    rm -rf "$DEST/examples"
    cp -rf "$SRC/examples" "$DEST/examples"
fi
# Drop bytecode caches so stale .pyc files never ship in the APK.
find "$DEST" -name '__pycache__' -type d -prune -exec rm -rf {} + 2>/dev/null || true

# 2. Ensure buildozer + Cython are available. Ubuntu 24.04 is PEP-668
#    "externally managed", so install into a dedicated virtualenv (no sudo).
VENV="$DEST/.venv"
if [ ! -x "$VENV/bin/buildozer" ]; then
    echo ">> Creating build virtualenv and installing buildozer..."
    python3 -m venv "$VENV"
    "$VENV/bin/pip" install --upgrade pip
    "$VENV/bin/pip" install --upgrade buildozer Cython virtualenv
fi
# Activate the venv (not just PATH): buildozer only skips the invalid `pip
# --user` flag when it sees VIRTUAL_ENV set, which `activate` provides. This
# also puts buildozer/cython console scripts on PATH for its probes.
# shellcheck disable=SC1091
source "$VENV/bin/activate"

# 2b. Nuke p4a's pure-python "pymodules" venv. p4a creates it with
#     `hostpython -m venv venv` but never passes --clear, so on a rebuild
#     ensurepip reinstalls pip 24.0 ON TOP of a previously upgraded pip,
#     splicing two resolvelib versions together and breaking pip with
#     "cannot import name 'RequirementInformation'". Removing it forces a clean
#     recreate every build.
rm -rf "$DEST/.buildozer/android/platform/build-"*"/build/venv" 2>/dev/null || true

# 3. Build. First run downloads the Android SDK/NDK and compiles the whole
#    scientific stack — expect it to take a long time.
cd "$DEST"
echo ">> Running: buildozer android debug"
buildozer android debug

echo ">> Build finished. APK(s):"
ls -lh "$DEST/bin/"*.apk 2>/dev/null || echo "   (no APK produced — check the log above)"
