"""
Rln desktop/tablet/mobile-friendly GUI.

This module intentionally reuses the same AppState and CommandParser used by
main.py, so the GUI stays a thin shell around the canonical Rln command
engine. It depends only on the Python standard-library tkinter package, with
optional matplotlib embedding when matplotlib is installed.
"""

from __future__ import annotations

import contextlib
import io
import math
import os
import platform
import queue
import re
import tempfile
import threading
import tkinter as tk
from tkinter import filedialog, font as tkfont, messagebox, ttk
from typing import Optional

from rich.console import Console

from commands.parser import CommandParser
from commands.state import AppState


PLAIN_LOGO_TEMPLATE = """\
 ____  _
|  _ \\| |_ __
| |_) | | '_ \\    Rln (ARLEN) v{ver}
|  _ <| | | | |   Statistics and Data Exploratory Tools
|_| \\_\\_|_| |_|
Open Source for Researchers         @2026 by Akirawisnu
https://akirawisnu.github.io/
MIT License - Type help for commands, exit to quit
"""

_ABOUT_FALLBACK = """Rln is a free, open-source, offline-capable data analysis tool for researchers.

It provides a compact command language for cleaning, transforming, describing,
estimating, and visualizing data, plus modern causal-inference estimators,
offline NLP, and larger-than-RAM workflows through Polars. The GUI is a thin
front end over the same command parser used by the terminal version."""

ANSI_RE = re.compile(r"\x1b\[([0-9;?]*)m")
ANSI_ANY_RE = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")

KEYWORDS = {
    "use", "save", "export", "import", "clear", "describe", "summarize", "sum",
    "tabulate", "tab", "regress", "reg", "logit", "probit", "didregress",
    "generate", "gen", "replace", "drop", "keep", "rename", "label", "help",
    "histogram", "scatter", "line", "graph", "twoway", "coefplot", "pctile",
    "xtile", "centile", "winsor2", "foreach", "forvalues", "if", "in", "by",
    "sort", "merge", "append", "collapse", "reshape", "python", "py", "do",
    "set", "global", "local", "quietly", "capture", "display", "di",
}


def _read_about_text() -> str:
    """Use the project README summary when available."""
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    readme = os.path.join(root, "README.md")
    if not os.path.exists(readme):
        return _ABOUT_FALLBACK

    try:
        with open(readme, "r", encoding="utf-8") as fh:
            text = fh.read()
    except OSError:
        return _ABOUT_FALLBACK

    marker = "**Rln is a free, open-source, offline-capable data analysis tool for researchers.**"
    start = text.find(marker)
    design = text.find("### Design goals")
    if start >= 0 and design > start:
        summary = text[start:design].strip()
        summary = re.sub(r"[*_`#>-]", "", summary)
        summary = re.sub(r"\n{3,}", "\n\n", summary)
        return summary
    return _ABOUT_FALLBACK


def _safe_mono_font(root: tk.Tk, size: int = 10) -> tuple[str, int]:
    """Return a monospace font likely to render Unicode box drawing cleanly."""
    families = set(tkfont.families(root))
    system = platform.system().lower()
    candidates = []
    if "windows" in system:
        candidates = ["Cascadia Mono", "Consolas", "Courier New"]
    elif "darwin" in system:
        candidates = ["Menlo", "Monaco", "Courier New"]
    else:
        candidates = ["DejaVu Sans Mono", "Liberation Mono", "Ubuntu Mono", "Noto Sans Mono", "Courier New"]
    for name in candidates:
        if name in families:
            return (name, size)
    return ("TkFixedFont", size)


class GuiStream(io.TextIOBase):
    """Thread-safe stream that sends stdout/stderr/rich output to the GUI."""

    def __init__(self, app: "RlnGuiApp"):
        super().__init__()
        self.app = app

    def writable(self) -> bool:  # pragma: no cover - trivial
        return True

    def write(self, value: str) -> int:  # type: ignore[override]
        if value:
            self.app.queue_console(value)
        return len(value)

    def flush(self) -> None:  # pragma: no cover - no buffering here
        return None




def _rounded_rectangle(canvas: tk.Canvas, x1: int, y1: int, x2: int, y2: int, radius: int, **kwargs) -> int:
    """Draw a rounded rectangle using a smooth polygon, compatible with Tk 8.6."""
    radius = max(2, min(radius, max(2, (x2 - x1) // 2), max(2, (y2 - y1) // 2)))
    points = [
        x1 + radius, y1, x2 - radius, y1, x2, y1, x2, y1 + radius,
        x2, y2 - radius, x2, y2, x2 - radius, y2, x1 + radius, y2,
        x1, y2, x1, y2 - radius, x1, y1 + radius, x1, y1,
    ]
    return canvas.create_polygon(points, smooth=True, splinesteps=24, **kwargs)


class RoundedFrame(tk.Frame):
    """A portable rounded card with an inner frame for regular Tk/ttk widgets."""

    def __init__(self, master, *, radius: int = 16, padding: int = 8, bg: str = "#ffffff", border: str = "#d1d5db", outer_bg: str | None = None, **kwargs):
        super().__init__(master, bd=0, highlightthickness=0, bg=outer_bg or bg, **kwargs)
        self.radius = radius
        self.padding = padding
        self.card_bg = bg
        self.border = border
        self.canvas = tk.Canvas(self, bd=0, highlightthickness=0, bg=outer_bg or bg)
        self.canvas.pack(fill="both", expand=True)
        self.content = tk.Frame(self.canvas, bd=0, highlightthickness=0, bg=bg)
        self._window_id = self.canvas.create_window(padding, padding, anchor="nw", window=self.content)
        self.canvas.bind("<Configure>", self._draw)

    def set_theme(self, *, bg: str, border: str, outer_bg: str) -> None:
        self.card_bg = bg
        self.border = border
        self.configure(bg=outer_bg)
        self.canvas.configure(bg=outer_bg)
        self.content.configure(bg=bg)
        self._draw()

    def _draw(self, _event=None) -> None:
        width = max(1, self.canvas.winfo_width())
        height = max(1, self.canvas.winfo_height())
        self.canvas.delete("card")
        _rounded_rectangle(self.canvas, 2, 2, max(3, width - 2), max(3, height - 2), self.radius, fill=self.card_bg, outline=self.border, width=1, tags="card")
        self.canvas.coords(self._window_id, self.padding, self.padding)
        self.canvas.itemconfigure(self._window_id, width=max(1, width - 2 * self.padding), height=max(1, height - 2 * self.padding))
        self.canvas.tag_lower("card")


class CircularButton(tk.Canvas):
    """Round, draggable-friendly floating button drawn on a Canvas."""

    def __init__(self, master, *, text: str = "≡", size: int = 58, bg: str = "#2563eb", fg: str = "#ffffff", outer_bg: str | None = None, **kwargs):
        super().__init__(master, width=size, height=size, bd=0, highlightthickness=0, cursor="hand2", bg=outer_bg or master.cget("background"), **kwargs)
        self.size = size
        self.text = text
        self.button_bg = bg
        self.button_fg = fg
        self.hover_bg = bg
        self.bind("<Enter>", lambda _event: self._draw(hover=True))
        self.bind("<Leave>", lambda _event: self._draw(hover=False))
        self._draw()

    def set_colors(self, *, bg: str, fg: str, hover: str, outer_bg: str) -> None:
        self.button_bg = bg
        self.button_fg = fg
        self.hover_bg = hover
        self.configure(bg=outer_bg)
        self._draw(False)

    def _draw(self, hover: bool = False) -> None:
        self.delete("all")
        fill = self.hover_bg if hover else self.button_bg
        margin = 4
        self.create_oval(margin, margin, self.size - margin, self.size - margin, fill=fill, outline="", tags="circle")
        self.create_text(self.size // 2, self.size // 2 - 1, text=self.text, fill=self.button_fg, font=("TkDefaultFont", 22, "bold"), tags="label")


class RlnGuiApp:
    """Responsive GUI shell for Rln."""

    def __init__(self, root: tk.Tk, version: str = "1.2.7"):
        self.root = root
        self.version = version
        self.root.title(f"Rln (ARLEN) v{version}")
        self.root.geometry("1180x760")
        self.root.minsize(460, 520)

        self.state = AppState()
        self.gui_stream = GuiStream(self)
        # force_terminal=True lets Rich colorize tables and messages. The Text
        # widget consumes ANSI SGR codes and maps them to tags below.
        self.console = Console(file=self.gui_stream, force_terminal=True, color_system="standard", width=120)
        self.parser = CommandParser(self.state, self.console)

        self._plot_canvas = None
        self._drag_offset = (0, 0)
        self._fab_dragged = False
        self._menu_popup: Optional[tk.Toplevel] = None
        self._layout_mode = tk.StringVar(value="horizontal")
        self._theme = tk.StringVar(value="light")
        self._console_queue: queue.Queue[str] = queue.Queue()
        self._command_lock = threading.Lock()
        self._worker: Optional[threading.Thread] = None
        self._pending_tempfiles: list[str] = []
        self._ansi_tags: list[str] = []
        self._highlight_after: Optional[str] = None
        self._rounded_frames: list[RoundedFrame] = []

        self.mono_font = _safe_mono_font(root, 10)
        self.editor_font = _safe_mono_font(root, 11)
        self._themes = {
            "light": {
                "bg": "#f7f7f7", "fg": "#202124", "panel": "#ffffff", "entry": "#ffffff",
                "muted": "#5f6368", "select": "#d9e8ff", "accent": "#2563eb",
                "console_bg": "#ffffff", "console_fg": "#202124", "border": "#c8c8c8",
                "syntax_kw": "#7c3aed", "syntax_str": "#166534", "syntax_comment": "#6b7280",
                "syntax_num": "#b45309", "prompt": "#2563eb", "error": "#b91c1c",
                "data_num": "#1d4ed8", "data_str": "#9a3412", "data_missing": "#6b7280",
            },
            "dark": {
                "bg": "#1e1e1e", "fg": "#e8eaed", "panel": "#252526", "entry": "#1f1f1f",
                "muted": "#a5a5a5", "select": "#374151", "accent": "#60a5fa",
                "console_bg": "#111827", "console_fg": "#e5e7eb", "border": "#3f3f46",
                "syntax_kw": "#c084fc", "syntax_str": "#86efac", "syntax_comment": "#9ca3af",
                "syntax_num": "#fbbf24", "prompt": "#60a5fa", "error": "#fca5a5",
                "data_num": "#93c5fd", "data_str": "#fb923c", "data_missing": "#9ca3af",
            },
        }

        self._build_style()
        self._build_ui()
        self._wire_shortcuts()
        self.apply_theme()
        self.append_console(PLAIN_LOGO_TEMPLATE.format(ver=version) + "\n", tag="logo")
        self.refresh_all()
        self.root.after(40, self._flush_console_queue)
        self.root.bind("<Configure>", lambda _event: self.root.after_idle(self._keep_fab_visible))

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------
    def _build_style(self) -> None:
        self.style = ttk.Style(self.root)
        with contextlib.suppress(tk.TclError):
            self.style.theme_use("clam")
        self.style.configure("Rln.TButton", padding=(12, 7), relief="flat", borderwidth=0)
        self.style.configure("Rln.TFrame", padding=0, relief="flat", borderwidth=0)
        self.style.configure("Treeview", rowheight=26, borderwidth=0)
        self.root.option_add("*TCombobox*Listbox.borderWidth", 0)

    def _build_ui(self) -> None:
        self.main = ttk.Frame(self.root, style="Rln.TFrame")
        self.main.pack(fill="both", expand=True)
        self._create_split(horizontal=True)
        self._create_floating_button()
        self.root.after_idle(self._keep_fab_visible)

    def _create_split(self, horizontal: bool) -> None:
        for child in self.main.winfo_children():
            child.destroy()
        self._rounded_frames = []

        orient = tk.HORIZONTAL if horizontal else tk.VERTICAL
        self.paned = ttk.PanedWindow(self.main, orient=orient)
        self.paned.pack(fill="both", expand=True)

        self.script_frame = ttk.Frame(self.paned, style="Rln.TFrame")
        self.workspace_frame = ttk.Frame(self.paned, style="Rln.TFrame")

        if horizontal:
            self.paned.add(self.script_frame, weight=1)
            self.paned.add(self.workspace_frame, weight=2)
        else:
            self.paned.add(self.workspace_frame, weight=2)
            self.paned.add(self.script_frame, weight=1)

        self._build_script_pane()
        self._build_workspace_pane()

    def _rounded_container(self, parent, *, padding: int = 8, radius: int = 18, fill: str = "both", expand: bool = True, padx: int = 8, pady: int = 6) -> RoundedFrame:
        theme = self._themes[self._theme.get()] if hasattr(self, "_themes") else {"entry": "#ffffff", "border": "#d1d5db", "bg": "#f7f7f7"}
        card = RoundedFrame(parent, radius=radius, padding=padding, bg=theme["entry"], border=theme["border"], outer_bg=theme["bg"])
        card.pack(fill=fill, expand=expand, padx=padx, pady=pady)
        self._rounded_frames.append(card)
        return card

    def _build_script_pane(self) -> None:
        header = ttk.Frame(self.script_frame, style="Rln.TFrame")
        header.pack(fill="x")
        ttk.Label(header, text="Do-file / script", font=("TkDefaultFont", 10, "bold")).pack(side="left", padx=6, pady=4)
        ttk.Button(header, text="Run script", style="Rln.TButton", command=self.run_script).pack(side="right", padx=4, pady=3)
        ttk.Button(header, text="Open", style="Rln.TButton", command=self.open_script).pack(side="right", padx=4, pady=3)
        ttk.Button(header, text="Save", style="Rln.TButton", command=self.save_script).pack(side="right", padx=4, pady=3)

        editor_card = self._rounded_container(self.script_frame, padding=8, radius=18, padx=8, pady=(2, 8))
        editor_box = editor_card.content
        yscroll = ttk.Scrollbar(editor_box, orient="vertical")
        xscroll = ttk.Scrollbar(editor_box, orient="horizontal")
        self.script_text = tk.Text(
            editor_box,
            wrap="none",
            undo=True,
            font=self.editor_font,
            yscrollcommand=yscroll.set,
            xscrollcommand=xscroll.set,
            borderwidth=0,
            highlightthickness=1,
            tabs=(32,),
        )
        yscroll.config(command=self.script_text.yview)
        xscroll.config(command=self.script_text.xview)
        self.script_text.grid(row=0, column=0, sticky="nsew")
        yscroll.grid(row=0, column=1, sticky="ns")
        xscroll.grid(row=1, column=0, sticky="ew")
        editor_box.columnconfigure(0, weight=1)
        editor_box.rowconfigure(0, weight=1)
        self.script_text.bind("<KeyRelease>", self._schedule_syntax_highlight)

    def _build_workspace_pane(self) -> None:
        topbar = ttk.Frame(self.workspace_frame, style="Rln.TFrame")
        topbar.pack(fill="x")
        ttk.Label(topbar, text="Workspace", font=("TkDefaultFont", 10, "bold")).pack(side="left", padx=6, pady=4)
        ttk.Button(topbar, text="Menu", style="Rln.TButton", command=self.toggle_menu_popup).pack(side="right", padx=4)
        ttk.Checkbutton(topbar, text="Dark mode", variable=self._theme, onvalue="dark", offvalue="light", command=self.apply_theme).pack(side="right", padx=8)
        ttk.Radiobutton(topbar, text="Tablet/Desktop split", variable=self._layout_mode, value="horizontal", command=self.set_horizontal).pack(side="right", padx=4)
        ttk.Radiobutton(topbar, text="Phone split", variable=self._layout_mode, value="vertical", command=self.set_vertical).pack(side="right", padx=4)

        self.notebook = ttk.Notebook(self.workspace_frame)
        self.notebook.pack(fill="both", expand=True)
        self._build_console_tab()
        self._build_variables_tab()
        self._build_data_tab()
        self._build_plots_tab()
        self._build_help_tab()

    def _build_console_tab(self) -> None:
        frame = ttk.Frame(self.notebook, style="Rln.TFrame")
        self.notebook.add(frame, text="Console")

        console_card = self._rounded_container(frame, padding=8, radius=18, padx=8, pady=(8, 4))
        console_box = console_card.content
        yscroll = ttk.Scrollbar(console_box, orient="vertical")
        self.console_text = tk.Text(
            console_box,
            wrap="none",
            state="disabled",
            font=self.mono_font,
            yscrollcommand=yscroll.set,
            borderwidth=0,
            highlightthickness=1,
            tabs=(32,),
        )
        yscroll.config(command=self.console_text.yview)
        self.console_text.grid(row=0, column=0, sticky="nsew")
        yscroll.grid(row=0, column=1, sticky="ns")
        console_box.columnconfigure(0, weight=1)
        console_box.rowconfigure(0, weight=1)

        cmd_frame = ttk.Frame(frame, style="Rln.TFrame")
        cmd_frame.pack(fill="x", padx=4, pady=4)
        self.prompt_label = ttk.Label(cmd_frame, text=".")
        self.prompt_label.pack(side="left", padx=(0, 4))
        self.command_entry = ttk.Entry(cmd_frame)
        self.command_entry.pack(side="left", fill="x", expand=True)
        self.command_entry.bind("<Return>", lambda _event: self.execute_from_entry())
        self.run_button = ttk.Button(cmd_frame, text="Run", style="Rln.TButton", command=self.execute_from_entry)
        self.run_button.pack(side="right", padx=(4, 0))

    def _build_variables_tab(self) -> None:
        frame = ttk.Frame(self.notebook, style="Rln.TFrame")
        self.notebook.add(frame, text="Variables")
        columns = ("name", "type", "label")
        self.var_tree = ttk.Treeview(frame, columns=columns, show="headings")
        for col, title, width in (("name", "Variable", 160), ("type", "Type", 110), ("label", "Label", 280)):
            self.var_tree.heading(col, text=title)
            self.var_tree.column(col, width=width, stretch=True)
        yscroll = ttk.Scrollbar(frame, orient="vertical", command=self.var_tree.yview)
        self.var_tree.configure(yscrollcommand=yscroll.set)
        self.var_tree.pack(side="left", fill="both", expand=True)
        yscroll.pack(side="right", fill="y")

    def _build_data_tab(self) -> None:
        frame = ttk.Frame(self.notebook, style="Rln.TFrame")
        self.notebook.add(frame, text="Data browser")
        info = ttk.Frame(frame, style="Rln.TFrame")
        info.pack(fill="x")
        self.data_status = ttk.Label(info, text="No data loaded")
        self.data_status.pack(side="left", padx=6, pady=4)
        ttk.Button(info, text="Refresh", style="Rln.TButton", command=self.refresh_data_browser).pack(side="right", padx=4, pady=3)

        table_card = self._rounded_container(frame, padding=8, radius=18, padx=8, pady=(2, 8))
        table_frame = table_card.content
        yscroll = ttk.Scrollbar(table_frame, orient="vertical")
        xscroll = ttk.Scrollbar(table_frame, orient="horizontal")
        self.data_text = tk.Text(
            table_frame,
            wrap="none",
            state="disabled",
            font=self.mono_font,
            yscrollcommand=yscroll.set,
            xscrollcommand=xscroll.set,
            borderwidth=0,
            highlightthickness=1,
            tabs=(32,),
        )
        yscroll.config(command=self.data_text.yview)
        xscroll.config(command=self.data_text.xview)
        self.data_text.grid(row=0, column=0, sticky="nsew")
        yscroll.grid(row=0, column=1, sticky="ns")
        xscroll.grid(row=1, column=0, sticky="ew")
        table_frame.columnconfigure(0, weight=1)
        table_frame.rowconfigure(0, weight=1)
        self._configure_data_browser_tags()

    def _build_plots_tab(self) -> None:
        self.plots_frame = ttk.Frame(self.notebook, style="Rln.TFrame")
        self.notebook.add(self.plots_frame, text="Plots")
        controls = ttk.Frame(self.plots_frame, style="Rln.TFrame")
        controls.pack(fill="x")
        ttk.Button(controls, text="Refresh current plot", style="Rln.TButton", command=self.refresh_plot).pack(side="left", padx=4, pady=3)
        ttk.Button(controls, text="Export current plot", style="Rln.TButton", command=self.export_plot).pack(side="left", padx=4, pady=3)
        plot_card = self._rounded_container(self.plots_frame, padding=8, radius=18, padx=8, pady=(2, 8))
        self.plot_area = plot_card.content
        self.plot_placeholder = ttk.Label(self.plot_area, text="Run a Rln graph command, then press Refresh current plot.")
        self.plot_placeholder.pack(expand=True)

    def _build_help_tab(self) -> None:
        frame = ttk.Frame(self.notebook, style="Rln.TFrame")
        self.notebook.add(frame, text="Help")
        toolbar = ttk.Frame(frame, style="Rln.TFrame")
        toolbar.pack(fill="x")
        ttk.Button(toolbar, text="Show command help", style="Rln.TButton", command=lambda: self.execute_command("help")).pack(side="left", padx=4, pady=3)
        ttk.Button(toolbar, text="About Rln", style="Rln.TButton", command=self.show_about).pack(side="left", padx=4, pady=3)
        yscroll = ttk.Scrollbar(frame, orient="vertical")
        self.help_text = tk.Text(frame, wrap="word", yscrollcommand=yscroll.set, font=("TkDefaultFont", 10), borderwidth=0, highlightthickness=1)
        yscroll.config(command=self.help_text.yview)
        self.help_text.insert("1.0", _read_about_text())
        self.help_text.configure(state="disabled")
        self.help_text.pack(side="left", fill="both", expand=True)
        yscroll.pack(side="right", fill="y")

    def _create_floating_button(self) -> None:
        # Keep the floating button inside the main frame, not as a competing
        # root child. Some Linux window managers allow a packed root child to
        # cover a placed root child, which made the button appear to vanish.
        old = getattr(self, "fab", None)
        if old is not None:
            with contextlib.suppress(tk.TclError):
                old.destroy()
        theme = self._themes[self._theme.get()]
        self.fab = CircularButton(
            self.main,
            text="≡",
            size=58,
            bg=theme["accent"],
            fg="#ffffff",
            outer_bg=theme["bg"],
            takefocus=True,
        )
        self.fab.bind("<ButtonPress-1>", self._start_fab_drag)
        self.fab.bind("<B1-Motion>", self._drag_fab)
        self.fab.bind("<ButtonRelease-1>", self._end_fab_drag)
        self.fab.bind("<Return>", lambda _event: self.toggle_menu_popup())
        self.root.after_idle(lambda: self._place_fab_clamped(99999, 99999))

    def _wire_shortcuts(self) -> None:
        modifier = "Command" if platform.system() == "Darwin" else "Control"
        self.root.bind(f"<{modifier}-o>", lambda _event: self.open_dataset())
        self.root.bind(f"<{modifier}-s>", lambda _event: self.save_script())
        self.root.bind(f"<{modifier}-r>", lambda _event: self.run_script())
        self.root.bind(f"<{modifier}-l>", lambda _event: self.command_entry.focus_set())

    # ------------------------------------------------------------------
    # Theme, layout, floating menu
    # ------------------------------------------------------------------
    def apply_theme(self) -> None:
        theme = self._themes[self._theme.get()]
        self.root.configure(bg=theme["bg"])
        self.style.configure("Rln.TFrame", background=theme["bg"])
        self.style.configure("TFrame", background=theme["bg"])
        self.style.configure("TLabel", background=theme["bg"], foreground=theme["fg"])
        self.style.configure("TCheckbutton", background=theme["bg"], foreground=theme["fg"])
        self.style.configure("TRadiobutton", background=theme["bg"], foreground=theme["fg"])
        self.style.configure("Rln.TButton", background=theme["panel"], foreground=theme["fg"], padding=(12, 7), relief="flat", borderwidth=0)
        self.style.map("Rln.TButton", background=[("active", theme["select"]), ("pressed", theme["select"])] )
        self.style.configure("TNotebook", background=theme["bg"], borderwidth=0, tabmargins=(8, 6, 8, 0))
        self.style.configure("TNotebook.Tab", background=theme["panel"], foreground=theme["fg"], padding=(14, 8), borderwidth=0)
        self.style.map("TNotebook.Tab", background=[("selected", theme["entry"])] )
        self.style.configure("Treeview", background=theme["entry"], foreground=theme["fg"], fieldbackground=theme["entry"], rowheight=24)
        self.style.configure("Treeview.Heading", background=theme["panel"], foreground=theme["fg"])

        if hasattr(self, "console_text"):
            self._configure_text_widget(self.console_text, console=True)
        if hasattr(self, "script_text"):
            self._configure_text_widget(self.script_text, console=False)
            self.highlight_script_syntax()
        if hasattr(self, "help_text"):
            self._configure_text_widget(self.help_text, console=False)
        if hasattr(self, "data_text"):
            self._configure_text_widget(self.data_text, console=False)
            self._configure_data_browser_tags()
        for card in getattr(self, "_rounded_frames", []):
            with contextlib.suppress(tk.TclError):
                card.set_theme(bg=theme["entry"], border=theme["border"], outer_bg=theme["bg"])
        if hasattr(self, "fab"):
            if hasattr(self.fab, "set_colors"):
                self.fab.set_colors(bg=theme["accent"], fg="#ffffff", hover=theme["prompt"], outer_bg=theme["bg"])
            else:
                self.fab.configure(bg=theme["accent"], fg="#ffffff")
            self._raise_widget(self.fab)

    def _raise_widget(self, widget: tk.Widget) -> None:
        """Raise a child widget above sibling widgets, even when it is a Canvas.

        Canvas overrides lift()/tkraise() to mean raise canvas item,
        which caused TclError for the circular floating button. Calling the
        Tcl window command directly raises the widget window itself.
        """
        with contextlib.suppress(tk.TclError):
            widget.tk.call("raise", widget._w)

    def _configure_text_widget(self, widget: tk.Text, console: bool) -> None:
        theme = self._themes[self._theme.get()]
        widget.configure(
            bg=theme["console_bg"] if console else theme["entry"],
            fg=theme["console_fg"] if console else theme["fg"],
            insertbackground=theme["fg"],
            selectbackground=theme["select"],
            highlightbackground=theme["border"],
            highlightcolor=theme["accent"],
        )
        widget.tag_configure("logo", foreground=theme["accent"])
        widget.tag_configure("prompt", foreground=theme["prompt"], font=(self.mono_font[0], self.mono_font[1], "bold"))
        widget.tag_configure("error", foreground=theme["error"])
        widget.tag_configure("keyword", foreground=theme["syntax_kw"])
        widget.tag_configure("string", foreground=theme["syntax_str"])
        widget.tag_configure("comment", foreground=theme["syntax_comment"])
        widget.tag_configure("number", foreground=theme["syntax_num"])
        widget.tag_configure("ansi_red", foreground=theme["error"])
        widget.tag_configure("ansi_green", foreground=theme["syntax_str"])
        widget.tag_configure("ansi_yellow", foreground=theme["syntax_num"])
        widget.tag_configure("ansi_blue", foreground=theme["prompt"])
        widget.tag_configure("ansi_magenta", foreground=theme["syntax_kw"])
        widget.tag_configure("ansi_cyan", foreground=theme["accent"])
        widget.tag_configure("ansi_dim", foreground=theme["muted"])
        widget.tag_configure("ansi_bold", font=(self.mono_font[0], self.mono_font[1], "bold"))

    def _switch_layout(self, horizontal: bool) -> None:
        script = ""
        console_dump = None
        if hasattr(self, "script_text"):
            script = self.script_text.get("1.0", "end-1c")
        if hasattr(self, "console_text"):
            console_dump = self.console_text.dump("1.0", "end-1c", text=True, tag=True)
        self._create_split(horizontal=horizontal)
        self.script_text.insert("1.0", script)
        if console_dump:
            self._restore_console_dump(console_dump)
        self._create_floating_button()
        self.apply_theme()
        self.refresh_all()
        self.root.after_idle(self._keep_fab_visible)

    def set_horizontal(self) -> None:
        self._layout_mode.set("horizontal")
        self._switch_layout(horizontal=True)

    def set_vertical(self) -> None:
        self._layout_mode.set("vertical")
        self._switch_layout(horizontal=False)

    def _start_fab_drag(self, event) -> None:
        self._drag_offset = (event.x, event.y)
        self._fab_dragged = False

    def _drag_fab(self, event) -> None:
        dx = event.x - self._drag_offset[0]
        dy = event.y - self._drag_offset[1]
        if abs(dx) + abs(dy) > 2:
            self._fab_dragged = True
        x = self.fab.winfo_x() + dx
        y = self.fab.winfo_y() + dy
        self._place_fab_clamped(x, y)

    def _end_fab_drag(self, _event) -> None:
        self._keep_fab_visible()
        if self._fab_dragged:
            self.root.after(80, lambda: setattr(self, "_fab_dragged", False))
            return
        self.toggle_menu_popup()

    def _place_fab_clamped(self, x: int, y: int) -> None:
        if not hasattr(self, "fab") or not self.fab.winfo_exists():
            return
        self.main.update_idletasks()
        width = max(self.fab.winfo_reqwidth(), self.fab.winfo_width(), 58)
        height = max(self.fab.winfo_reqheight(), self.fab.winfo_height(), 58)
        max_x = max(6, self.main.winfo_width() - width - 10)
        max_y = max(6, self.main.winfo_height() - height - 10)
        self.fab.place(x=min(max(int(x), 6), max_x), y=min(max(int(y), 6), max_y))
        self._raise_widget(self.fab)

    def _keep_fab_visible(self, _event=None) -> None:
        if hasattr(self, "fab") and self.fab.winfo_exists():
            x = self.fab.winfo_x()
            y = self.fab.winfo_y()
            if x <= 1 and y <= 1:
                self._place_fab_clamped(99999, 99999)
            else:
                self._place_fab_clamped(x, y)

    def toggle_menu_popup(self) -> None:
        if self._menu_popup and self._menu_popup.winfo_exists():
            self._menu_popup.destroy()
            return

        theme = self._themes[self._theme.get()]
        self._menu_popup = tk.Toplevel(self.root)
        self._menu_popup.title("Rln menu")
        self._menu_popup.transient(self.root)
        self._menu_popup.resizable(False, False)
        with contextlib.suppress(tk.TclError):
            self._menu_popup.attributes("-topmost", True)
        x = self.main.winfo_rootx() + self.fab.winfo_x() - 205
        y = self.main.winfo_rooty() + self.fab.winfo_y() - 175
        self._menu_popup.geometry(f"250x190+{max(x, 10)}+{max(y, 10)}")
        self._menu_popup.configure(bg=theme["bg"])

        popup_card = RoundedFrame(self._menu_popup, radius=20, padding=12, bg=theme["entry"], border=theme["border"], outer_bg=theme["bg"])
        popup_card.pack(fill="both", expand=True, padx=8, pady=8)
        box = popup_card.content
        buttons = [
            ("Open", self.open_dataset),
            ("Export", self.export_dataset),
            ("Command", self.focus_command),
            ("About Rln", self.show_about),
        ]
        for i, (text, command) in enumerate(buttons):
            btn = ttk.Button(box, text=text, command=lambda c=command: self._menu_action(c), style="Rln.TButton")
            btn.grid(row=i // 2, column=i % 2, sticky="nsew", padx=5, pady=5, ipadx=8, ipady=14)
        box.columnconfigure(0, weight=1)
        box.columnconfigure(1, weight=1)

    def _menu_action(self, command) -> None:
        if self._menu_popup and self._menu_popup.winfo_exists():
            self._menu_popup.destroy()
        command()

    # ------------------------------------------------------------------
    # Console and commands
    # ------------------------------------------------------------------
    def queue_console(self, text: str) -> None:
        self._console_queue.put(text)

    def _flush_console_queue(self) -> None:
        chunks: list[str] = []
        while True:
            try:
                chunks.append(self._console_queue.get_nowait())
            except queue.Empty:
                break
        if chunks:
            self._insert_console_ansi("".join(chunks))
        self.root.after(40, self._flush_console_queue)

    def append_console(self, text: str, tag: Optional[str] = None) -> None:
        if tag:
            self.console_text.configure(state="normal")
            self.console_text.insert("end", ANSI_ANY_RE.sub("", text), tag)
            self.console_text.see("end")
            self.console_text.configure(state="disabled")
        else:
            self._insert_console_ansi(text)

    def _insert_console_ansi(self, text: str) -> None:
        if not hasattr(self, "console_text"):
            return
        self.console_text.configure(state="normal")
        pos = 0
        for match in ANSI_RE.finditer(text):
            if match.start() > pos:
                clean = ANSI_ANY_RE.sub("", text[pos:match.start()])
                if clean:
                    self.console_text.insert("end", clean, tuple(self._ansi_tags))
            self._update_ansi_tags(match.group(1))
            pos = match.end()
        if pos < len(text):
            clean = ANSI_ANY_RE.sub("", text[pos:])
            if clean:
                self.console_text.insert("end", clean, tuple(self._ansi_tags))
        self.console_text.see("end")
        self.console_text.configure(state="disabled")

    def _update_ansi_tags(self, params: str) -> None:
        # Parse common SGR ANSI sequences produced by Rich. Unsupported extended
        # colors are ignored safely instead of accidentally clearing all styles.
        codes = [0] if not params else [int(p) for p in params.split(";") if p.isdigit()]
        color_map = {
            30: "ansi_dim", 31: "ansi_red", 32: "ansi_green", 33: "ansi_yellow", 34: "ansi_blue",
            35: "ansi_magenta", 36: "ansi_cyan", 37: None,
            90: "ansi_dim", 91: "ansi_red", 92: "ansi_green", 93: "ansi_yellow", 94: "ansi_blue",
            95: "ansi_magenta", 96: "ansi_cyan", 97: None,
        }
        color_tags = {"ansi_red", "ansi_green", "ansi_yellow", "ansi_blue", "ansi_magenta", "ansi_cyan"}
        for code in codes:
            if code == 0:
                self._ansi_tags = []
            elif code == 1 and "ansi_bold" not in self._ansi_tags:
                self._ansi_tags.append("ansi_bold")
            elif code == 2 and "ansi_dim" not in self._ansi_tags:
                self._ansi_tags.append("ansi_dim")
            elif code == 22:
                self._ansi_tags = [t for t in self._ansi_tags if t not in ("ansi_bold", "ansi_dim")]
            elif code == 39:
                self._ansi_tags = [t for t in self._ansi_tags if t not in color_tags]
            elif code in color_map:
                self._ansi_tags = [t for t in self._ansi_tags if t not in color_tags]
                tag = color_map[code]
                if tag and tag not in self._ansi_tags:
                    self._ansi_tags.append(tag)

    def _restore_console_dump(self, dump_items) -> None:
        """Restore console text and tag ranges after rebuilding panes."""
        if not hasattr(self, "console_text"):
            return
        self.console_text.configure(state="normal")
        self.console_text.delete("1.0", "end")
        open_tags: set[str] = set()
        for kind, value, _index in dump_items:
            if kind == "tagon":
                open_tags.add(value)
            elif kind == "tagoff":
                open_tags.discard(value)
            elif kind == "text" and value:
                self.console_text.insert("end", value, tuple(open_tags))
        self.console_text.configure(state="disabled")
        self.console_text.see("end")

    def focus_command(self) -> None:
        self.notebook.select(0)
        self.command_entry.focus_set()

    def execute_from_entry(self) -> None:
        cmd = self.command_entry.get().strip()
        if not cmd:
            return
        self.command_entry.delete(0, "end")
        self.execute_command(cmd)

    def execute_command(self, cmd: str, on_done=None) -> None:
        self.notebook.select(0)
        self.append_console(f". {cmd}\n", tag="prompt")
        if self._worker and self._worker.is_alive():
            self.append_console("Another Rln command is still running. Please wait for it to finish.\n", tag="error")
            return

        self._set_running(True)
        self._worker = threading.Thread(target=self._execute_command_worker, args=(cmd, on_done), daemon=True)
        self._worker.start()

    def _execute_command_worker(self, cmd: str, on_done=None) -> None:
        try:
            with self._command_lock:
                with contextlib.redirect_stdout(self.gui_stream), contextlib.redirect_stderr(self.gui_stream):
                    if cmd.lower().startswith("do "):
                        import main as rln_main  # lazy import to avoid startup cycle
                        path = cmd[3:].strip().strip("\"'")
                        rln_main.run_do_file(path, self.parser, self.console)
                    elif cmd.lower() in ("exit", "quit", "q"):
                        self.root.after(0, self.root.quit)
                        return
                    else:
                        self.parser.execute(cmd)
        except Exception as exc:
            self.queue_console(f"Error: {exc}\n")
        finally:
            self.root.after(0, self._command_finished, on_done)

    def _command_finished(self, on_done=None) -> None:
        self._set_running(False)
        self.refresh_all()
        if callable(on_done):
            on_done()

    def _set_running(self, running: bool) -> None:
        state = "disabled" if running else "normal"
        with contextlib.suppress(tk.TclError):
            self.run_button.configure(state=state)
            self.command_entry.configure(state=state)

    def run_script(self) -> None:
        script = self.script_text.get("1.0", "end-1c")
        if not script.strip():
            return
        self.notebook.select(0)
        tmp = tempfile.NamedTemporaryFile("w", suffix=".do", encoding="utf-8", delete=False)
        try:
            tmp.write(script)
            tmp_path = tmp.name
        finally:
            tmp.close()
        self._pending_tempfiles.append(tmp_path)
        self.execute_command(f'do "{tmp_path}"', on_done=lambda path=tmp_path: self._delete_tempfile(path))

    def _delete_tempfile(self, path: str) -> None:
        with contextlib.suppress(OSError):
            os.unlink(path)
        with contextlib.suppress(ValueError):
            self._pending_tempfiles.remove(path)

    # ------------------------------------------------------------------
    # Syntax highlighting
    # ------------------------------------------------------------------
    def _schedule_syntax_highlight(self, _event=None) -> None:
        if self._highlight_after:
            with contextlib.suppress(tk.TclError):
                self.root.after_cancel(self._highlight_after)
        self._highlight_after = self.root.after(120, self.highlight_script_syntax)

    def highlight_script_syntax(self) -> None:
        if not hasattr(self, "script_text"):
            return
        text = self.script_text.get("1.0", "end-1c")
        for tag in ("keyword", "string", "comment", "number"):
            self.script_text.tag_remove(tag, "1.0", "end")
        for lineno, line in enumerate(text.splitlines(), start=1):
            comment_at = line.find("//")
            if comment_at >= 0:
                self.script_text.tag_add("comment", f"{lineno}.{comment_at}", f"{lineno}.end")
                code_part = line[:comment_at]
            else:
                code_part = line
            for match in re.finditer(r'("[^"\n]*"|\'[^\'\n]*\')', code_part):
                self.script_text.tag_add("string", f"{lineno}.{match.start()}", f"{lineno}.{match.end()}")
            for match in re.finditer(r"\b\d+(?:\.\d+)?\b", code_part):
                self.script_text.tag_add("number", f"{lineno}.{match.start()}", f"{lineno}.{match.end()}")
            for match in re.finditer(r"\b[A-Za-z_][A-Za-z0-9_]*\b", code_part):
                if match.group(0).lower() in KEYWORDS:
                    self.script_text.tag_add("keyword", f"{lineno}.{match.start()}", f"{lineno}.{match.end()}")

    # ------------------------------------------------------------------
    # File actions
    # ------------------------------------------------------------------
    def open_dataset(self) -> None:
        path = filedialog.askopenfilename(
            title="Open dataset",
            filetypes=[
                ("Data files", "*.csv *.dta *.xlsx *.xls *.parquet *.json *.sav *.feather"),
                ("All files", "*.*"),
            ],
        )
        if path:
            self.execute_command(f'use "{path}"')

    def export_dataset(self) -> None:
        if not self.state.has_data():
            messagebox.showinfo("Export", "No dataset is currently loaded.")
            return
        path = filedialog.asksaveasfilename(
            title="Export dataset",
            defaultextension=".csv",
            filetypes=[("CSV", "*.csv"), ("Stata", "*.dta"), ("Excel", "*.xlsx"), ("All files", "*.*")],
        )
        if path:
            self.execute_command(f'save "{path}", replace')

    def open_script(self) -> None:
        path = filedialog.askopenfilename(title="Open do-file", filetypes=[("Do files", "*.do"), ("All files", "*.*")])
        if not path:
            return
        try:
            with open(path, "r", encoding="utf-8") as fh:
                self.script_text.delete("1.0", "end")
                self.script_text.insert("1.0", fh.read())
            self.highlight_script_syntax()
        except OSError as exc:
            messagebox.showerror("Open script", str(exc))

    def save_script(self) -> None:
        path = filedialog.asksaveasfilename(title="Save do-file", defaultextension=".do", filetypes=[("Do files", "*.do"), ("All files", "*.*")])
        if not path:
            return
        try:
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(self.script_text.get("1.0", "end-1c"))
        except OSError as exc:
            messagebox.showerror("Save script", str(exc))

    # ------------------------------------------------------------------
    # Data, variables, plots, help
    # ------------------------------------------------------------------
    def refresh_all(self) -> None:
        self.refresh_prompt()
        self.refresh_variables()
        self.refresh_data_browser()

    def refresh_prompt(self) -> None:
        if self.state.has_data():
            rows, cols = self.state.data.shape
            name = self.state.dataset_name or "data"
            self.prompt_label.configure(text=f"({name}: {rows:,} x {cols}) .")
        else:
            self.prompt_label.configure(text=".")

    def refresh_variables(self) -> None:
        self.var_tree.delete(*self.var_tree.get_children())
        if not self.state.has_data():
            return
        for col in self.state.data.columns:
            label = self.state.get_variable_label(col) or ""
            dtype = str(self.state.data[col].dtype)
            self.var_tree.insert("", "end", values=(col, dtype, label))

    def _configure_data_browser_tags(self) -> None:
        if not hasattr(self, "data_text"):
            return
        theme = self._themes[self._theme.get()]
        self.data_text.tag_configure("data_header", foreground=theme["accent"], font=(self.mono_font[0], self.mono_font[1], "bold"))
        self.data_text.tag_configure("data_number", foreground=theme["data_num"])
        self.data_text.tag_configure("data_string", foreground=theme["data_str"])
        self.data_text.tag_configure("data_missing", foreground=theme["data_missing"])
        self.data_text.tag_configure("data_index", foreground=theme["muted"])

    @staticmethod
    def _classify_data_value(value) -> str:
        if value is None:
            return "missing"
        try:
            import pandas as pd  # type: ignore
            if pd.isna(value):
                return "missing"
        except Exception:
            pass
        if isinstance(value, bool):
            return "string"
        try:
            numeric = isinstance(value, (int, float)) and not isinstance(value, bool)
            if numeric and not (isinstance(value, float) and math.isnan(value)):
                return "number"
        except Exception:
            pass
        return "string"

    def refresh_data_browser(self) -> None:
        if not hasattr(self, "data_text"):
            return
        self.data_text.configure(state="normal")
        self.data_text.delete("1.0", "end")
        if not self.state.has_data():
            self.data_status.configure(text="No data loaded")
            self.data_text.configure(state="disabled")
            return

        df = self.state.data
        max_rows = int(self.state.settings.get("max_display_rows", 200) or 200)
        view = df.head(max_rows)
        columns = [str(c) for c in view.columns]
        widths = {col: max(8, min(28, len(col))) for col in columns}
        rendered_rows = []
        for idx, row in view.iterrows():
            rendered = []
            for col, value in zip(columns, row.tolist()):
                cls = self._classify_data_value(value)
                text = "" if cls == "missing" else str(value)
                if len(text) > 30:
                    text = text[:27] + "..."
                widths[col] = max(widths[col], min(30, len(text)))
                rendered.append((col, text, cls))
            rendered_rows.append((str(idx), rendered))

        index_width = max(5, min(12, max([len(str(i)) for i in view.index] or [0])))
        self.data_text.insert("end", " " * (index_width + 2), "data_header")
        for col in columns:
            self.data_text.insert("end", col.ljust(widths[col] + 2), "data_header")
        self.data_text.insert("end", "\n")
        self.data_text.insert("end", "-" * (index_width + 2 + sum(widths[c] + 2 for c in columns)) + "\n", "data_index")

        for idx_text, rendered in rendered_rows:
            self.data_text.insert("end", idx_text[:index_width].ljust(index_width) + "  ", "data_index")
            for col, text, cls in rendered:
                tag = {"number": "data_number", "string": "data_string", "missing": "data_missing"}[cls]
                cell = text.rjust(widths[col]) if cls == "number" else text.ljust(widths[col])
                self.data_text.insert("end", cell + "  ", tag)
            self.data_text.insert("end", "\n")

        self.data_text.configure(state="disabled")
        more = "" if len(df) <= max_rows else f"; showing first {max_rows:,}"
        self.data_status.configure(text=f"{len(df):,} observations, {len(df.columns):,} variables{more}. Numbers are blue, strings are orange, missing values are muted.")

    def refresh_plot(self) -> None:
        for child in self.plot_area.winfo_children():
            child.destroy()
        try:
            import matplotlib.pyplot as plt
            from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
        except Exception as exc:
            ttk.Label(self.plot_area, text=f"Matplotlib plot embedding is unavailable: {exc}").pack(expand=True)
            return

        fig_nums = plt.get_fignums()
        if not fig_nums:
            ttk.Label(self.plot_area, text="No active matplotlib figure found.").pack(expand=True)
            return
        fig = plt.figure(fig_nums[-1])
        self._plot_canvas = FigureCanvasTkAgg(fig, master=self.plot_area)
        self._plot_canvas.draw()
        toolbar = NavigationToolbar2Tk(self._plot_canvas, self.plot_area)
        toolbar.update()
        self._plot_canvas.get_tk_widget().pack(fill="both", expand=True)

    def export_plot(self) -> None:
        try:
            import matplotlib.pyplot as plt
        except Exception:
            messagebox.showinfo("Export plot", "Matplotlib is unavailable.")
            return
        fig_nums = plt.get_fignums()
        if not fig_nums:
            messagebox.showinfo("Export plot", "No active plot is available.")
            return
        path = filedialog.asksaveasfilename(title="Export current plot", defaultextension=".png", filetypes=[("PNG", "*.png"), ("PDF", "*.pdf"), ("SVG", "*.svg")])
        if path:
            plt.figure(fig_nums[-1]).savefig(path, bbox_inches="tight")
            self.append_console(f"Saved plot: {path}\n")

    def show_about(self) -> None:
        messagebox.showinfo("About Rln", _read_about_text())


def launch_gui(version: str = "1.2.7") -> None:
    """Launch the Rln GUI."""
    root = tk.Tk()
    RlnGuiApp(root, version=version)
    root.mainloop()
