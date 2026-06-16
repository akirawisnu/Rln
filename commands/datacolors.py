"""Shared data-type classification + colour scheme for Rln's data explorers.

The desktop GUI, the terminal TUI browser, and the Android grid all colour
cells by type so a value looks the same everywhere — numbers blue, strings
orange, missing values muted. Centralising it here keeps the three front-ends
in sync (previously only the desktop GUI coloured its data browser).
"""

from __future__ import annotations

import math

# Canonical hex colours. The light/dark variants mirror the desktop GUI themes
# in gui/app.py (data_num / data_str / data_missing) so desktop and mobile match.
COLORS = {
    "light": {"number": "#1d4ed8", "string": "#9a3412", "missing": "#6b7280",
              "header": "#2563eb", "negative": "#b91c1c"},
    "dark":  {"number": "#93c5fd", "string": "#fb923c", "missing": "#9ca3af",
              "header": "#60a5fa", "negative": "#f87171"},
}

# Rich / Textual style names for the terminal browser (named colours are safe
# across terminals; the intent matches the hex scheme above).
RICH_STYLES = {
    "number": "cyan",
    "string": "orange3",
    "missing": "grey42",
    "header": "bold cyan",
    "negative": "red",
}


def classify(value) -> str:
    """Classify a single cell value as ``'missing'``, ``'number'`` or ``'string'``.

    Booleans count as strings (categorical), NaN/None as missing. This is the
    same rule the desktop GUI used, lifted here so every front-end agrees.
    """
    if value is None:
        return "missing"
    try:
        import pandas as pd  # local import keeps this module dependency-light
        if pd.isna(value):
            return "missing"
    except Exception:
        pass
    if isinstance(value, bool):
        return "string"
    if isinstance(value, (int, float)):
        if isinstance(value, float) and math.isnan(value):
            return "missing"
        return "number"
    # numpy scalars
    try:
        import numpy as np
        if isinstance(value, np.bool_):
            return "string"
        if isinstance(value, (np.integer, np.floating)):
            if isinstance(value, np.floating) and math.isnan(float(value)):
                return "missing"
            return "number"
    except Exception:
        pass
    return "string"


def hex_for(kind: str, value=None, dark: bool = True) -> str:
    """Return the hex colour (no leading '#') for a classified cell, picking the
    negative-number shade when ``value`` is a negative number."""
    pal = COLORS["dark" if dark else "light"]
    key = kind
    if kind == "number" and value is not None:
        try:
            if float(value) < 0:
                key = "negative"
        except Exception:
            pass
    return pal.get(key, pal["string"]).lstrip("#")


def rich_style_for(kind: str, value=None) -> str:
    """Return a Rich style name for a classified cell (negative numbers red)."""
    if kind == "number" and value is not None:
        try:
            if float(value) < 0:
                return RICH_STYLES["negative"]
        except Exception:
            pass
    return RICH_STYLES.get(kind, RICH_STYLES["string"])
