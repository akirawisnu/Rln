"""Workspace / default-folder helper shared by every Rln front-end.

Goals (same behaviour on desktop source, the frozen lite/full builds, and the
Android app):

* **Examples are reachable by default.** On a fresh install the file pickers
  open in the bundled ``examples/`` folder, and a bare name like
  ``use "demographics.csv"`` or ``do "sample.do"`` resolves against
  ``examples/`` so the shipped sample scripts/data "just open".
* **Follow the latest project.** After you open a file from somewhere else,
  that folder becomes the new starting point (persisted across sessions).
* **User-overridable.** ``set workdir "<path>"`` pins an explicit default that
  wins over the latest-project tracking.

State persists in ``~/.rln_config.json`` (next to the ``~/.rln_history`` the
REPL already writes).
"""

from __future__ import annotations

import json
import os
import sys

_CONFIG_NAME = ".rln_config.json"


def _app_root() -> str:
    """Directory that contains the bundled ``examples/`` folder."""
    # PyInstaller frozen build unpacks data files under sys._MEIPASS.
    meipass = getattr(sys, "_MEIPASS", None)
    if meipass and os.path.isdir(meipass):
        return meipass
    # Source / Android: repo (or app) root is the parent of this package.
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def examples_dir() -> str:
    """Best-effort path to the shipped ``examples/`` folder, across platforms."""
    candidates = [
        os.path.join(_app_root(), "examples"),
        os.path.join(os.getcwd(), "examples"),
    ]
    # Android stages the app (incl. examples) under these locations.
    for env in ("ANDROID_APP_PATH", "ANDROID_PRIVATE", "ANDROID_ARGUMENT"):
        d = os.environ.get(env)
        if d:
            candidates.append(os.path.join(d, "examples"))
    for c in candidates:
        if c and os.path.isdir(c):
            return os.path.abspath(c)
    return os.path.abspath(candidates[0])


def _config_path() -> str:
    # ANDROID_PRIVATE is the app's writable home on Android; ~ elsewhere.
    base = os.environ.get("ANDROID_PRIVATE") or os.path.expanduser("~")
    return os.path.join(base, _CONFIG_NAME)


def _load_cfg() -> dict:
    try:
        with open(_config_path(), "r", encoding="utf-8") as f:
            data = json.load(f)
            return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _save_cfg(cfg: dict) -> None:
    try:
        with open(_config_path(), "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2)
    except Exception:
        pass  # never let a config write failure break a command


def get_open_dir() -> str:
    """Folder a file picker / path resolver should start from.

    Priority: explicit ``workdir`` override → latest project folder → examples.
    """
    cfg = _load_cfg()
    for key in ("default_dir", "last_dir"):
        d = cfg.get(key)
        if d and os.path.isdir(d):
            return d
    return examples_dir()


def set_default_dir(path: str) -> str:
    """Pin an explicit default folder (``set workdir``). Returns the abs path."""
    p = os.path.abspath(os.path.expanduser(path))
    cfg = _load_cfg()
    cfg["default_dir"] = p
    _save_cfg(cfg)
    return p


def remember_dir(path: str) -> None:
    """Record the folder of a just-opened file as the latest project."""
    try:
        d = os.path.dirname(os.path.abspath(os.path.expanduser(path)))
        if os.path.isdir(d):
            cfg = _load_cfg()
            cfg["last_dir"] = d
            _save_cfg(cfg)
    except Exception:
        pass


def resolve_path(path: str) -> str:
    """Resolve a (possibly bare) filename to something that exists.

    Tries, in order: the path as given (after ~ expansion), then under the
    current open dir (latest project / workdir), then under ``examples/``. If
    nothing matches, returns the expanded original so the caller can report a
    normal "file not found".
    """
    if not path:
        return path
    expanded = os.path.expanduser(path)
    if os.path.exists(expanded):
        return expanded
    # Only fall back for relative names (don't second-guess absolute paths).
    if not os.path.isabs(expanded):
        for base in (get_open_dir(), examples_dir()):
            cand = os.path.join(base, path)
            if os.path.exists(cand):
                return cand
    return expanded

