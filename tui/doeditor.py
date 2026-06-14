"""
TUI Script Editor with Rln syntax highlighting.
Uses Textual's TextArea with custom syntax highlighting.
"""

import os

try:
    from textual.app import App, ComposeResult
    from textual.widgets import Header, Footer, TextArea, Static
    from textual.binding import Binding
    from textual.containers import Vertical

    HAS_TEXTUAL = True
except ImportError:
    HAS_TEXTUAL = False


def launch_editor(filepath=None, state=None, console=None):
    """Launch the TUI script editor."""
    if not HAS_TEXTUAL:
        raise ImportError("Textual required for TUI editor")

    app = DoEditorApp(filepath, state, console)
    app.run()


if HAS_TEXTUAL:

    class DoEditorApp(App):
        """Script editor with syntax highlighting."""

        CSS = """
        Screen {
            layout: vertical;
        }

        #editor {
            height: 1fr;
        }

        #status-bar {
            dock: bottom;
            height: 1;
            background: $primary-background;
            color: $text;
            padding: 0 1;
        }

        #help-bar {
            dock: bottom;
            height: 1;
            background: $surface;
            color: $text-muted;
            padding: 0 1;
        }
        """

        BINDINGS = [
            Binding("ctrl+s", "save", "Save", show=True),
            Binding("ctrl+r", "run_file", "Run", show=True),
            Binding("ctrl+q", "quit_editor", "Quit", show=True),
            Binding("ctrl+n", "new_file", "New", show=True),
        ]

        def __init__(self, filepath=None, state=None, console=None):
            super().__init__()
            self.filepath = filepath or "untitled.rln"
            self.state = state
            self.outer_console = console
            self.modified = False
            self.title = f"Rln Editor - {os.path.basename(self.filepath)}"

        def compose(self) -> ComposeResult:
            yield Header()
            yield TextArea(id="editor", language="python", show_line_numbers=True)
            yield Static("Ctrl+S: Save | Ctrl+R: Run | Ctrl+Q: Quit | Ctrl+N: New", id="help-bar")
            yield Static(self._status(), id="status-bar")
            yield Footer()

        def on_mount(self) -> None:
            editor = self.query_one("#editor", TextArea)
            if os.path.exists(self.filepath):
                with open(self.filepath, "r", encoding="utf-8") as f:
                    content = f.read()
                editor.load_text(content)
                self.notify(f"Loaded: {self.filepath}")
            else:
                # Template for new file
                template = (
                    "* ============================================\n"
                    f"* {os.path.basename(self.filepath)}\n"
                    "* Rln Script\n"
                    "* ============================================\n"
                    "\n"
                    "* Load data\n"
                    '// use "data.csv"\n'
                    "\n"
                    "* Explore\n"
                    "// describe\n"
                    "// summarize\n"
                    "\n"
                )
                editor.load_text(template)

        def on_text_area_changed(self, event) -> None:
            self.modified = True
            self.query_one("#status-bar", Static).update(self._status())

        def _status(self) -> str:
            mod = " [MODIFIED]" if self.modified else ""
            return f" {self.filepath}{mod}"

        def action_save(self) -> None:
            editor = self.query_one("#editor", TextArea)
            content = editor.text
            try:
                with open(self.filepath, "w", encoding="utf-8") as f:
                    f.write(content)
                self.modified = False
                self.query_one("#status-bar", Static).update(self._status())
                n_lines = content.count("\n") + 1
                self.notify(f"Saved: {self.filepath} ({n_lines} lines)")
            except Exception as e:
                self.notify(f"Save failed: {e}", severity="error")

        def action_run_file(self) -> None:
            # Save first
            self.action_save()
            self.notify(f"File saved. Run with: do \"{self.filepath}\"")

        def action_quit_editor(self) -> None:
            if self.modified:
                self.notify("Unsaved changes! Ctrl+S to save, then Ctrl+Q again.")
                self.modified = False  # Allow second Ctrl+Q to quit
                return
            self.exit()

        def action_new_file(self) -> None:
            editor = self.query_one("#editor", TextArea)
            editor.load_text("")
            self.filepath = "untitled.rln"
            self.modified = False
            self.title = "Rln Editor - untitled.rln"
            self.query_one("#status-bar", Static).update(self._status())
