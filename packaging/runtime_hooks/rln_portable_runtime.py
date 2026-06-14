"""Runtime setup for Rln portable PyInstaller builds.

Keeps model caches, Argos packages, NLTK data, and matplotlib config beside
Rln's executable instead of in user-profile locations. This makes the app
folder portable and usable without admin rights.
"""
import os
import sys


def _portable_root() -> str:
    if getattr(sys, "frozen", False):
        return os.path.dirname(os.path.abspath(sys.executable))
    # Source/dev fallback: project root is two levels above packaging/runtime_hooks/
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


ROOT = _portable_root()
os.environ.setdefault("RLN_PORTABLE_ROOT", ROOT)
os.environ.setdefault("PYTHONUTF8", "1")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

# Writable folders inside the portable app directory.
DIRS = {
    "HF_HOME": os.path.join(ROOT, "hf_models"),
    "TRANSFORMERS_CACHE": os.path.join(ROOT, "hf_models"),
    "HF_DATASETS_CACHE": os.path.join(ROOT, "hf_models", "datasets"),
    "SENTENCE_TRANSFORMERS_HOME": os.path.join(ROOT, "hf_models", "sentence_transformers"),
    "ARGOS_PACKAGES_DIR": os.path.join(ROOT, "argos_models"),
    "NLTK_DATA": os.path.join(ROOT, "nltk_data"),
    "MPLCONFIGDIR": os.path.join(ROOT, "mpl_config"),
}

for env_name, path in DIRS.items():
    try:
        os.makedirs(path, exist_ok=True)
    except Exception:
        pass
    os.environ.setdefault(env_name, path)

# Offline-first default. Users can override before launch with RLN_HF_ONLINE=1.
if os.environ.get("RLN_HF_ONLINE") != "1":
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
