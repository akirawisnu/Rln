"""
TUI Data Browser - Interactive terminal data explorer.
Uses Textual for a rich, scrollable data table.

Features:
- Scrollable rows and columns with mouse/keyboard
- Column sorting (press 's' on a column)
- Row numbers
- Type-colored cells (numeric=cyan, string=white, missing=dim, negative=red)
- Status bar with dataset info
- Filter with econometric if-expressions ('/' key)
- Goto row ('g' key, then type row number)
- Column info ('i' key shows type/label for current column)
- Value label display
- Keyboard navigation (arrows, Page Up/Down, Home/End)
"""

import pandas as pd
import numpy as np
from typing import Optional

from rich.text import Text
from commands.datacolors import classify, rich_style_for, RICH_STYLES

try:
    from textual.app import App, ComposeResult
    from textual.widgets import Header, Footer, DataTable, Input, Static
    from textual.containers import Vertical
    from textual.binding import Binding

    HAS_TEXTUAL = True
except ImportError:
    HAS_TEXTUAL = False


# ──────────────────────────────────────────────
#  Fallback rich-based browser (no Textual)
# ──────────────────────────────────────────────

def _fallback_browser(df: pd.DataFrame, state, console):
    """Simple paged browser using rich when Textual is not available."""
    from rich.table import Table

    page_size = state.settings.get("pagesize", 50) if state else 50
    total_rows = len(df)
    current_page = 0
    total_pages = max(1, (total_rows + page_size - 1) // page_size)

    while True:
        start = current_page * page_size
        end = min(start + page_size, total_rows)
        page_df = df.iloc[start:end]

        table = Table(
            title=f"Rows {start+1}-{end} of {total_rows:,}  (page {current_page+1}/{total_pages})",
            show_lines=False,
            row_styles=["", "dim"],
        )

        # Row number column
        table.add_column("#", style="dim", justify="right", width=max(5, len(str(total_rows))))

        # Data columns with auto-width
        for col in page_df.columns:
            is_num = pd.api.types.is_numeric_dtype(df[col])
            justify = "right" if is_num else "left"
            # Calculate width: max of header and first few values
            sample_vals = page_df[col].head(20).astype(str)
            max_val_len = max(sample_vals.str.len().max() if len(sample_vals) > 0 else 0, len(col))
            width = min(max(max_val_len + 1, 8), 35)
            table.add_column(col, justify=justify, min_width=8, max_width=35, no_wrap=True)

        for i, (idx, row) in enumerate(page_df.iterrows()):
            cells = [str(start + i + 1)]
            for col in page_df.columns:
                val = row[col]
                if pd.isna(val):
                    cells.append("[dim].[/dim]")
                elif isinstance(val, (int, np.integer)):
                    # Check for value labels
                    if state:
                        label = state.get_value_label_text(col, val)
                        if label:
                            cells.append(f"[cyan]{val}[/cyan] [dim]{label}[/dim]")
                            continue
                    cells.append(f"[cyan]{val}[/cyan]" if val >= 0 else f"[red]{val}[/red]")
                elif isinstance(val, (float, np.floating)):
                    formatted = f"{val:.4f}"
                    cells.append(f"[cyan]{formatted}[/cyan]" if val >= 0 else f"[red]{formatted}[/red]")
                else:
                    cells.append(f"[orange3]{str(val)[:35]}[/orange3]")
            table.add_row(*cells)

        console.print(table)

        # Navigation prompt
        if total_pages <= 1:
            console.print("[dim]Press Enter to close.[/dim]")
            try:
                input()
            except (EOFError, KeyboardInterrupt):
                pass
            break

        console.print(
            "[dim]Navigation: [bold]n[/bold]=next [bold]p[/bold]=prev "
            "[bold]f[/bold]=first [bold]l[/bold]=last [bold]g N[/bold]=goto page "
            "[bold]q[/bold]=quit[/dim]"
        )
        try:
            cmd = input(f"[page {current_page+1}/{total_pages}] > ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            break

        if cmd in ("q", "quit", "exit", ""):
            break
        elif cmd in ("n", "next"):
            current_page = min(current_page + 1, total_pages - 1)
        elif cmd in ("p", "prev"):
            current_page = max(current_page - 1, 0)
        elif cmd in ("f", "first"):
            current_page = 0
        elif cmd in ("l", "last"):
            current_page = total_pages - 1
        elif cmd.startswith("g "):
            try:
                pg = int(cmd[2:]) - 1
                current_page = max(0, min(pg, total_pages - 1))
            except ValueError:
                pass


# ──────────────────────────────────────────────
#  Main launcher
# ──────────────────────────────────────────────

def launch_browser(df: pd.DataFrame, state=None):
    """Launch the TUI data browser."""
    if HAS_TEXTUAL:
        try:
            app = DataBrowserApp(df, state)
            app.run()
            return
        except Exception:
            pass  # Fall through to fallback

    # Use fallback if Textual not available or fails
    from rich.console import Console
    console = Console()
    _fallback_browser(df, state, console)


# ──────────────────────────────────────────────
#  Textual TUI browser
# ──────────────────────────────────────────────

if HAS_TEXTUAL:

    class DataBrowserApp(App):
        """Interactive data browser application."""

        CSS = """
        Screen {
            layout: vertical;
        }

        #status-bar {
            dock: bottom;
            height: 1;
            background: $primary-background;
            color: $text;
            padding: 0 1;
        }

        #info-bar {
            dock: bottom;
            height: 1;
            background: $surface;
            color: $text-muted;
            padding: 0 1;
            display: none;
        }

        #info-bar.visible {
            display: block;
        }

        #filter-bar {
            dock: top;
            height: 3;
            display: none;
        }

        #filter-bar.visible {
            display: block;
        }

        #goto-bar {
            dock: top;
            height: 3;
            display: none;
        }

        #goto-bar.visible {
            display: block;
        }

        DataTable {
            height: 1fr;
        }
        """

        BINDINGS = [
            Binding("q", "quit", "Quit", show=True),
            Binding("escape", "quit", "Quit", show=False),
            Binding("/", "toggle_filter", "Filter", show=True),
            Binding("s", "sort_column", "Sort", show=True),
            Binding("i", "column_info", "Info", show=True),
            Binding("ctrl+g", "toggle_goto", "Goto", show=True),
            Binding("home", "goto_top", "Top", show=False),
            Binding("end", "goto_bottom", "Bottom", show=False),
        ]

        def __init__(self, df: pd.DataFrame, state=None):
            super().__init__()
            self.original_df = df
            self.display_df = df
            self.state = state
            self.sort_column = None
            self.sort_ascending = True
            self.filter_active = False
            self.title = "Rln Browser"

        def compose(self) -> ComposeResult:
            yield Header()
            with Vertical(id="filter-bar"):
                yield Input(
                    placeholder="Filter: econometric expression (e.g., age > 30 & city == \"Berlin\")",
                    id="filter-input"
                )
            with Vertical(id="goto-bar"):
                yield Input(placeholder="Go to row number...", id="goto-input")
            yield DataTable(id="data-table")
            yield Static("", id="info-bar")
            yield Static(self._status_text(), id="status-bar")
            yield Footer()

        def on_mount(self) -> None:
            self._populate_table(self.original_df)

        def _format_cell(self, val, col_name: str):
            """Format a cell value as a type-coloured Rich ``Text``.

            Numbers cyan (negatives red), strings orange, missing muted — the
            same scheme as the desktop GUI, so the explorer looks consistent
            across versions.
            """
            kind = classify(val)
            if kind == "missing":
                return Text(".", style=rich_style_for("missing"))

            # Check value labels (integer-coded categoricals)
            if self.state and isinstance(val, (int, np.integer)):
                label = self.state.get_value_label_text(col_name, int(val))
                if label:
                    return Text(f"{val} ({label})", style=rich_style_for("number", val))

            if isinstance(val, (float, np.floating)):
                text = f"{val:.4f}"
            else:
                text = str(val)
            return Text(text, style=rich_style_for(kind, val))

        def _populate_table(self, df: pd.DataFrame):
            """Fill the DataTable with DataFrame contents."""
            table = self.query_one("#data-table", DataTable)
            table.clear(columns=True)

            table.add_column(Text("#", style=RICH_STYLES["missing"]), key="__rownum__")

            for col in df.columns:
                label = str(col)
                if self.state:
                    vl = self.state.get_variable_label(col)
                    if vl:
                        label = f"{col} ({vl})"
                table.add_column(Text(label, style=RICH_STYLES["header"]), key=str(col))

            # Performance limit
            max_rows = 10000
            show_df = df.head(max_rows)

            for i, (idx, row) in enumerate(show_df.iterrows()):
                cells = [str(i + 1)]
                for col in df.columns:
                    cells.append(self._format_cell(row[col], col))
                table.add_row(*cells)

            if len(df) > max_rows:
                table.add_row("...", *["..." for _ in df.columns])

            self.display_df = show_df
            self.query_one("#status-bar", Static).update(self._status_text())

        def _status_text(self) -> str:
            nobs = len(self.original_df)
            nvars = len(self.original_df.columns)
            filter_text = " [FILTERED]" if self.filter_active else ""
            sort_text = ""
            if self.sort_column:
                d = "↑" if self.sort_ascending else "↓"
                sort_text = f" | Sorted: {self.sort_column} {d}"
            return (f" {nobs:,} obs × {nvars} vars{filter_text}{sort_text}"
                    f" | q:quit /:filter s:sort i:info Ctrl+G:goto")

        # ── Actions ──

        def action_toggle_filter(self) -> None:
            fb = self.query_one("#filter-bar")
            if fb.has_class("visible"):
                fb.remove_class("visible")
                self._populate_table(self.original_df)
                self.filter_active = False
            else:
                fb.add_class("visible")
                self.query_one("#filter-input", Input).focus()
            self.query_one("#status-bar", Static).update(self._status_text())

        def action_toggle_goto(self) -> None:
            gb = self.query_one("#goto-bar")
            if gb.has_class("visible"):
                gb.remove_class("visible")
            else:
                gb.add_class("visible")
                self.query_one("#goto-input", Input).focus()

        def on_input_submitted(self, event: Input.Submitted) -> None:
            input_id = event.input.id

            if input_id == "filter-input":
                expr = event.value.strip()
                if not expr:
                    self._populate_table(self.original_df)
                    self.filter_active = False
                else:
                    try:
                        from commands.expression import eval_condition
                        mask = eval_condition(expr, self.original_df)
                        filtered = self.original_df.loc[mask]
                        self._populate_table(filtered)
                        self.filter_active = True
                        self.notify(f"Showing {len(filtered):,} of {len(self.original_df):,} rows")
                    except Exception as e:
                        self.notify(f"Filter error: {e}", severity="error")

            elif input_id == "goto-input":
                try:
                    row_num = int(event.value.strip())
                    table = self.query_one("#data-table", DataTable)
                    target = max(0, min(row_num - 1, table.row_count - 1))
                    table.move_cursor(row=target)
                    self.notify(f"Jumped to row {target + 1}")
                except ValueError:
                    self.notify("Enter a row number", severity="warning")

                self.query_one("#goto-bar").remove_class("visible")

            self.query_one("#status-bar", Static).update(self._status_text())

        def action_sort_column(self) -> None:
            table = self.query_one("#data-table", DataTable)
            if table.cursor_column is not None and table.cursor_column > 0:
                col_idx = table.cursor_column - 1  # -1 for row number column
                if col_idx < len(self.original_df.columns):
                    col_key = self.original_df.columns[col_idx]

                    if self.sort_column == col_key:
                        self.sort_ascending = not self.sort_ascending
                    else:
                        self.sort_column = col_key
                        self.sort_ascending = True

                    base = self.original_df if not self.filter_active else self.display_df
                    sorted_df = base.sort_values(
                        col_key, ascending=self.sort_ascending, na_position="last"
                    ).reset_index(drop=True)
                    self._populate_table(sorted_df)

                    direction = "ascending" if self.sort_ascending else "descending"
                    self.notify(f"Sorted by {col_key} ({direction})")

        def action_column_info(self) -> None:
            table = self.query_one("#data-table", DataTable)
            info_bar = self.query_one("#info-bar", Static)

            if table.cursor_column is not None and table.cursor_column > 0:
                col_idx = table.cursor_column - 1
                if col_idx < len(self.original_df.columns):
                    col = self.original_df.columns[col_idx]
                    dtype = self.original_df[col].dtype
                    n_missing = self.original_df[col].isna().sum()
                    n_unique = self.original_df[col].nunique()

                    info_parts = [f"{col}: {dtype}"]
                    info_parts.append(f"unique={n_unique:,}")
                    info_parts.append(f"missing={n_missing:,}")

                    if self.state:
                        vl = self.state.get_variable_label(col)
                        if vl:
                            info_parts.append(f'"{vl}"')

                    if pd.api.types.is_numeric_dtype(self.original_df[col]):
                        s = self.original_df[col].dropna()
                        if len(s) > 0:
                            info_parts.append(f"range=[{s.min():.2f}, {s.max():.2f}]")
                            info_parts.append(f"mean={s.mean():.2f}")

                    info_bar.update(" | ".join(info_parts))
                    if not info_bar.has_class("visible"):
                        info_bar.add_class("visible")
                    return

            # Toggle off
            if info_bar.has_class("visible"):
                info_bar.remove_class("visible")

        def action_goto_top(self) -> None:
            self.query_one("#data-table", DataTable).move_cursor(row=0)

        def action_goto_bottom(self) -> None:
            table = self.query_one("#data-table", DataTable)
            table.move_cursor(row=table.row_count - 1)
