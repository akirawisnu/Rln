"""
Application state: holds dataset, labels, settings, and log.
"""

import os
import pandas as pd
from typing import Optional, Dict, Any
from io import StringIO


class AppState:
    """Central state for the application."""

    def __init__(self):
        self.data: Optional[pd.DataFrame] = None
        self.dataset_name: Optional[str] = None
        self.source_file: Optional[str] = None

        # compact variable labels
        self.variable_labels: Dict[str, str] = {}

        # compact value labels: {label_name: {value: label_text}}
        self.value_labels: Dict[str, Dict[Any, str]] = {}

        # Which value label set is attached to which variable
        self.value_label_assignments: Dict[str, str] = {}

        # Notes attached to the dataset
        self.notes: list = []

        # Logging
        self.log_file: Optional[str] = None
        self.log_handle = None

        # State tracking
        self.has_unsaved_changes: bool = False

        # Return code from last command (as documented here's _rc)
        self.return_code: int = 0
        self._rc: int = 0                         # alias used by newer code
        self.capture_mode: bool = False
        self.on_error: str = "stop"               # "stop" | "continue"

        # Preserve/restore snapshot stack
        self._snapshots: list = []

        # Local macros
        self.local_macros: Dict[str, str] = {}

        # Global macros (persist across scripts)
        self.global_macros: Dict[str, str] = {}

        # Stored results: r() and e()
        self.r_results: Dict[str, Any] = {}
        self.e_results: Dict[str, Any] = {}

        # By-group context (set during by/bysort execution)
        self._by_vars: list = None
        self._pending_block: list = None

        # Settings
        self.settings: Dict[str, Any] = {
            "linesize": 120,
            "pagesize": 50,
            "max_display_rows": 200,
            "float_format": "%.4f",
        }

    def has_data(self) -> bool:
        """True if a dataset is loaded.

        Bug 12 (Gemini): pandas .empty returns True whenever EITHER rows OR
        columns are zero. That's wrong for our purposes — a DataFrame with
        10 rows and no columns yet (e.g., just after `set obs 10`) is a
        legitimate working dataset. Check len() of each axis instead.
        """
        if self.data is None:
            return False
        return len(self.data) > 0 or len(self.data.columns) > 0

    def require_data(self):
        """Raise error if no dataset is loaded."""
        if not self.has_data():
            raise ValueError("No dataset in memory. Use 'use <file>' to load data.")

    def clear(self):
        """Clear dataset, LRTM state, and force garbage collection to free RAM."""
        import gc

        # Clear pandas DataFrame
        if self.data is not None:
            del self.data
        self.data = None
        self.dataset_name = None
        self.source_file = None
        self.variable_labels.clear()
        self.value_labels.clear()
        self.value_label_assignments.clear()
        self.notes.clear()
        self.has_unsaved_changes = False

        # Clear LRTM lazy frame if present
        if hasattr(self, '_lrtm_lf') and self._lrtm_lf is not None:
            del self._lrtm_lf
            self._lrtm_lf = None
        if hasattr(self, '_lrtm_schema'):
            self._lrtm_schema = None
        if hasattr(self, '_lrtm_source'):
            self._lrtm_source = None

        # Reset panel declaration — xtset state must not leak across
        # `clear` boundaries (Gemini v126 finding).
        if hasattr(self, '_panel_var'):
            self._panel_var = None
        if hasattr(self, '_time_var'):
            self._time_var = None

        # Reset by/bysort context — must not survive `clear`.
        if hasattr(self, '_by_vars'):
            self._by_vars = None

        # Reset capture state — _captured_rc should not persist past clear.
        if hasattr(self, '_captured_rc'):
            self._captured_rc = None

        # Clear estimation results
        self.r_results.clear()
        self.e_results.clear()

        # Clear macros
        self.local_macros.clear()

        # Force garbage collection
        gc.collect()

    def set_data(self, df: pd.DataFrame, name: str = None, source: str = None):
        """Set a new dataset, releasing the old one from RAM."""
        import gc
        if self.data is not None:
            del self.data
            gc.collect()
        self.data = df
        self.dataset_name = name
        self.source_file = source
        self.has_unsaved_changes = False

    def mark_changed(self):
        self.has_unsaved_changes = True

    # --- Logging ---

    def start_log(self, filepath: str, append: bool = False):
        """Start logging output to file."""
        self.stop_log()
        mode = "a" if append else "w"
        self.log_file = filepath
        self.log_handle = open(filepath, mode, encoding="utf-8")
        self.log_handle.write(f"--- Rln Log: {filepath} ---\n\n")

    def stop_log(self):
        """Stop logging."""
        if self.log_handle:
            self.log_handle.write("\n--- End of Log ---\n")
            self.log_handle.close()
            self.log_handle = None
            self.log_file = None

    def write_log(self, text: str):
        """Write text to log if logging is active."""
        if self.log_handle:
            self.log_handle.write(text + "\n")
            self.log_handle.flush()

    # --- Variable label helpers ---

    def get_variable_label(self, var: str) -> str:
        return self.variable_labels.get(var, "")

    def set_variable_label(self, var: str, label: str):
        self.variable_labels[var] = label

    def get_value_label_text(self, var: str, value) -> Optional[str]:
        """Get the label text for a specific value of a variable."""
        label_name = self.value_label_assignments.get(var)
        if label_name and label_name in self.value_labels:
            return self.value_labels[label_name].get(value)
        return None

    # --- Preserve / Restore ---

    def preserve(self):
        """Save a snapshot of the current dataset and metadata."""
        if self.data is None:
            raise ValueError("No data to preserve.")
        snapshot = {
            "data": self.data.copy(),
            "dataset_name": self.dataset_name,
            "source_file": self.source_file,
            "variable_labels": dict(self.variable_labels),
            "value_labels": {k: dict(v) for k, v in self.value_labels.items()},
            "value_label_assignments": dict(self.value_label_assignments),
            "notes": list(self.notes),
        }
        self._snapshots.append(snapshot)

    def restore(self):
        """Restore the most recent snapshot."""
        if not self._snapshots:
            raise ValueError("No preserved data to restore.")
        snapshot = self._snapshots.pop()
        self.data = snapshot["data"]
        self.dataset_name = snapshot["dataset_name"]
        self.source_file = snapshot["source_file"]
        self.variable_labels = snapshot["variable_labels"]
        self.value_labels = snapshot["value_labels"]
        self.value_label_assignments = snapshot["value_label_assignments"]
        self.notes = snapshot["notes"]
        self.has_unsaved_changes = True

    def has_snapshot(self) -> bool:
        return len(self._snapshots) > 0
