"""
Rln for Android - phone-first Kivy front-end over the Rln command engine.

This is the mobile entry point packaged by buildozer. It reuses the exact same
CommandParser / AppState engine as the desktop CLI and Tk GUI, so every command
parses and behaves identically. Only the presentation layer differs.

The layout deliberately mirrors the desktop GUI (gui/app.py) so the experience
feels familiar: a "Workspace" header with a Dark-mode toggle and a Menu button,
a tabbed notebook (Console / Variables / Data / Plots / Do-file / Help), a
floating circular menu button, and a do-file/script editor. Tkinter cannot run
on Android, so the same design is rebuilt here with native Kivy widgets.

Supported on Android in this build: data IO, exploration, variable transforms,
dataops, diagnostics, charts (rendered to an image), and scripting. The
econometrics (statsmodels/linearmodels) and LRTM (polars) commands import lazily
and report that the optional packages are unavailable until a later build adds
them.
"""

import os
import io
import sys
import threading

# matplotlib must use a non-interactive backend on Android (no Tk/Qt display).
# Set this BEFORE the engine (which imports matplotlib lazily) ever runs a chart.
os.environ.setdefault("MPLBACKEND", "Agg")

# Make the bundled engine packages importable regardless of CWD.
APP_DIR = os.path.dirname(os.path.abspath(__file__))
if APP_DIR not in sys.path:
    sys.path.insert(0, APP_DIR)

from kivy.app import App
from kivy.clock import Clock
from kivy.core.window import Window
from kivy.metrics import dp, sp
from kivy.graphics import Color, RoundedRectangle, Ellipse
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.gridlayout import GridLayout
from kivy.uix.button import Button
from kivy.uix.togglebutton import ToggleButton
from kivy.uix.image import Image
from kivy.uix.label import Label
from kivy.uix.modalview import ModalView
from kivy.uix.scrollview import ScrollView
from kivy.uix.splitter import Splitter
from kivy.uix.textinput import TextInput
from kivy.uix.filechooser import FileChooserListView

# CodeInput gives the script editor Pygments-based syntax highlighting. It (and
# pygments) could be absent in a stripped build, so degrade to a plain
# TextInput rather than crash at import time.
try:
    from kivy.uix.codeinput import CodeInput
    from pygments.lexer import RegexLexer, words
    from pygments.token import Comment, Keyword, Name, Number, String, Text
    HAVE_CODEINPUT = True
except Exception:
    HAVE_CODEINPUT = False

from rich.console import Console

from commands.state import AppState
from commands.parser import CommandParser

__version__ = "1.2.7"

LOGO = r""" ____  _
|  _ \| |_ __
| |_) | | '_ \    Rln (ARLEN) v{ver}
|  _ <| | | | |   Statistics and Data Exploratory Tools
|_| \_\_|_| |_|
Open Source for Researchers     @2026 by Akirawisnu
https://akirawisnu.github.io/
MIT License - Type help for commands, exit to quit
""".format(ver=__version__)

ABOUT_TEXT = (
    "Rln is a free, open-source, offline-capable data analysis tool for "
    "researchers.\n\n"
    "It provides a compact command language for cleaning, transforming, "
    "describing, estimating, and visualizing data. The mobile app is a thin "
    "front end over the same command parser used by the desktop and terminal "
    "versions, so your do-files behave identically across platforms.\n\n"
    "Type  help  in the Console for the full command list. Tap a tab to browse "
    "your variables, the data, or the last plot. Write multi-line scripts in "
    "the Do-file tab and press Run script.\n\n"
    "@2026 by Akirawisnu - MIT License\n"
    "https://akirawisnu.github.io/"
)

# Palettes lifted from the desktop GUI so the two look like one product.
THEMES = {
    "light": {
        "bg": "#f7f7f7", "fg": "#202124", "panel": "#ffffff", "entry": "#ffffff",
        "muted": "#5f6368", "accent": "#2563eb", "console_bg": "#ffffff",
        "console_fg": "#202124", "prompt": "#2563eb", "error": "#b91c1c",
    },
    "dark": {
        "bg": "#1e1e1e", "fg": "#e8eaed", "panel": "#252526", "entry": "#1f1f1f",
        "muted": "#a5a5a5", "accent": "#60a5fa", "console_bg": "#111827",
        "console_fg": "#e5e7eb", "prompt": "#60a5fa", "error": "#fca5a5",
    },
}

TABS = ["Console", "Variables", "Data", "Plots", "Help"]

# Font sizes tuned for a phone screen (smaller than the desktop so monospace
# tables stay crisp and aligned rather than fuzzy/oversized).
FS_MONO = "9sp"      # console + data browser (monospace)
FS_TABLE = "11sp"    # variables table cells
FS_EDITOR = "11sp"   # do-file editor
FS_UI = "13sp"       # general UI text


if HAVE_CODEINPUT:
    RLN_KEYWORDS = (
        "use", "save", "export", "import", "clear", "describe", "summarize",
        "sum", "tabulate", "tab", "tabstat", "correlate", "corr", "regress",
        "reg", "logit", "probit", "poisson", "ivregress", "didregress",
        "predict", "margins", "test", "ttest", "anova", "generate", "gen",
        "egen", "replace", "drop", "keep", "rename", "label", "encode",
        "decode", "destring", "tostring", "help", "histogram", "scatter",
        "line", "graph", "twoway", "coefplot", "pctile", "xtile", "centile",
        "winsor2", "foreach", "forvalues", "while", "if", "in", "by", "bysort",
        "sort", "gsort", "merge", "append", "collapse", "reshape", "python",
        "py", "do", "run", "set", "global", "local", "scalar", "matrix",
        "quietly", "qui", "capture", "cap", "noisily", "display", "di", "lrtm",
    )

    class RlnLexer(RegexLexer):
        """Minimal lexer for the Rln command language (Stata-like)."""
        name = "Rln"
        aliases = ["rln"]
        tokens = {
            "root": [
                (r"//[^\n]*", Comment.Single),
                (r"^\s*\*[^\n]*", Comment.Single),
                (r'"[^"\n]*"', String.Double),
                (r"'[^'\n]*'", String.Single),
                (r"\$\w+|`\w+'", Name.Variable),
                (r"\b\d+\.?\d*\b", Number),
                (words(RLN_KEYWORDS, prefix=r"\b", suffix=r"\b"), Keyword),
                (r"[A-Za-z_]\w*", Name),
                (r"\s+", Text),
                (r".", Text),
            ],
        }


def hx(h):
    """'#rrggbb' -> (r, g, b, 1) floats for Kivy."""
    h = h.lstrip("#")
    return (int(h[0:2], 16) / 255.0, int(h[2:4], 16) / 255.0,
            int(h[4:6], 16) / 255.0, 1)


class Card(BoxLayout):
    """A panel with a flat rounded background whose color can be themed."""

    def __init__(self, radius=12, **kwargs):
        super().__init__(**kwargs)
        with self.canvas.before:
            self._col = Color(rgba=(0, 0, 0, 1))
            self._rect = RoundedRectangle(radius=[radius])
        self.bind(pos=self._sync, size=self._sync)

    def _sync(self, *_):
        self._rect.pos = self.pos
        self._rect.size = self.size

    def set_bg(self, rgba):
        self._col.rgba = rgba


class RoundFab(Button):
    """Round, draggable floating button that opens the Rln menu on tap."""

    def __init__(self, on_tap=None, **kwargs):
        super().__init__(**kwargs)
        self.on_tap = on_tap
        self.background_normal = ""
        self.background_down = ""
        self.background_color = (0, 0, 0, 0)
        self.text = ""
        with self.canvas.before:
            self._col = Color(rgba=hx("#2563eb"))
            self._circle = Ellipse()
        with self.canvas.after:
            # draw the "hamburger" as three bars so it never depends on a font
            # glyph (Roboto has no U+2261/U+2630).
            Color(rgba=(1, 1, 1, 1))
            self._bars = [RoundedRectangle(radius=[2]) for _ in range(3)]
        self.bind(pos=self._sync, size=self._sync)
        self._down_pos = None
        self._moved = False
        self._sync()  # size/pos were set before the bind above; sync now

    def _sync(self, *_):
        self._circle.pos = self.pos
        self._circle.size = self.size
        w = self.width * 0.40
        h = max(dp(2), self.height * 0.06)
        x = self.center_x - w / 2
        gap = self.height * 0.13
        for bar, off in zip(self._bars, (gap, 0, -gap)):
            bar.pos = (x, self.center_y + off - h / 2)
            bar.size = (w, h)

    def set_bg(self, rgba):
        self._col.rgba = rgba

    def on_touch_down(self, touch):
        if self.collide_point(*touch.pos):
            touch.grab(self)
            self._down_pos = touch.pos
            self._moved = False
            return True
        return super().on_touch_down(touch)

    def on_touch_move(self, touch):
        if touch.grab_current is self:
            self.center_x += touch.dx
            self.center_y += touch.dy
            self._clamp()
            if (abs(touch.x - self._down_pos[0]) +
                    abs(touch.y - self._down_pos[1])) > dp(8):
                self._moved = True
            return True
        return super().on_touch_move(touch)

    def on_touch_up(self, touch):
        if touch.grab_current is self:
            touch.ungrab(self)
            if not self._moved and self.on_tap:
                self.on_tap()
            return True
        return super().on_touch_up(touch)

    def _clamp(self):
        if not self.parent:
            return
        self.x = max(dp(4), min(self.x, self.parent.width - self.width - dp(4)))
        self.y = max(dp(4), min(self.y, self.parent.height - self.height - dp(4)))


class RlnEngine:
    """Thin wrapper that runs Rln commands and captures their text output."""

    def __init__(self):
        self.state = AppState()
        self._buffer = io.StringIO()
        self.console = Console(
            file=self._buffer, force_terminal=False, color_system=None,
            width=110, soft_wrap=False,
        )
        self.parser = CommandParser(self.state, self.console)
        self._read_pos = 0

    def run(self, command):
        try:
            self.parser.execute(command)
        except Exception as exc:  # mirror the REPL: never crash the UI
            self.console.print(f"[error] {exc}")
        self._buffer.seek(self._read_pos)
        new_text = self._buffer.read()
        self._read_pos = self._buffer.tell()
        return new_text

    def plot_from_output(self, text):
        """Return the PNG a chart command just wrote.

        The engine's chart commands save the figure to disk and immediately
        plt.close() it (see commands/charts.py), so pyplot's figure registry is
        empty afterwards. The reliable signal is the "Saved to:" / "Graph
        saved:" path the command prints; we fall back to the retained
        state._last_figure object if the message format ever changes.
        """
        for line in text.splitlines():
            for marker in ("Graph saved:", "Saved to:", "Graph saved to:"):
                idx = line.find(marker)
                if idx >= 0:
                    cand = line[idx + len(marker):].strip().strip('"')
                    if cand and os.path.exists(cand):
                        return cand
        fig = getattr(self.state, "_last_figure", None)
        if fig is not None:
            try:
                out = os.path.join(writable_dir(), "rln_last_plot.png")
                fig.savefig(out, dpi=130, bbox_inches="tight")
                return out
            except Exception:
                return None
        return None


def writable_dir():
    for key in ("ANDROID_PRIVATE", "EXTERNAL_STORAGE"):
        d = os.environ.get(key)
        if d and os.path.isdir(d):
            return d
    return APP_DIR


def browse_root():
    for cand in ("/storage/emulated/0", "/sdcard",
                 os.environ.get("EXTERNAL_STORAGE"), APP_DIR):
        if cand and os.path.isdir(cand):
            return cand
    return APP_DIR


def default_open_dir():
    """A useful starting folder for the file picker (where users put files)."""
    root = browse_root()
    for sub in ("Download", "Documents", "Downloads"):
        d = os.path.join(root, sub)
        if os.path.isdir(d):
            return d
    return root


def ensure_storage_access():
    """Make sure we can read arbitrary files from shared storage.

    Android 11+ (API 30+) enforces scoped storage: READ_EXTERNAL_STORAGE only
    exposes media, not .csv/.dta/.do files. The fix is "All files access"
    (MANAGE_EXTERNAL_STORAGE), which has no in-app dialog — we send the user to
    the system settings screen to toggle it on. On older Android we request the
    legacy runtime permission. Returns True if access is already granted;
    False if a grant flow was just launched (caller should ask the user to
    retry). On non-Android (desktop) the filesystem is open, so returns True.
    """
    try:
        from jnius import autoclass
    except Exception:
        return True  # desktop / no Android runtime
    try:
        sdk = autoclass("android.os.Build$VERSION").SDK_INT
        if sdk >= 30:
            Environment = autoclass("android.os.Environment")
            if Environment.isExternalStorageManager():
                return True
            PythonActivity = autoclass("org.kivy.android.PythonActivity")
            Intent = autoclass("android.content.Intent")
            Settings = autoclass("android.provider.Settings")
            Uri = autoclass("android.net.Uri")
            activity = PythonActivity.mActivity
            intent = Intent(
                Settings.ACTION_MANAGE_APP_ALL_FILES_ACCESS_PERMISSION)
            intent.setData(Uri.parse("package:" + activity.getPackageName()))
            activity.startActivity(intent)
            return False
        else:
            from android.permissions import (request_permissions,
                                             check_permission, Permission)
            if check_permission(Permission.READ_EXTERNAL_STORAGE):
                return True
            request_permissions([Permission.READ_EXTERNAL_STORAGE,
                                  Permission.WRITE_EXTERNAL_STORAGE])
            return False
    except Exception:
        return True  # best effort — let the chooser try anyway


def run_do_file(filepath, parser, console):
    """Execute a do-file. Ported verbatim from the desktop CLI (main.py) so the
    engine's `do` command and the Run-script button behave identically on
    Android. The engine does `from main import run_do_file`, and on Android the
    `main` module is THIS file — so it must live here."""
    filepath = os.path.expanduser(filepath)
    if not os.path.exists(filepath):
        console.print(f"[red]Do-file not found: {filepath}[/red]")
        return False

    console.print(f"[dim]Running: {filepath}[/dim]")

    with open(filepath, "r", encoding="utf-8") as f:
        raw_lines = f.readlines()

    from commands.dofile import preprocess_dofile
    commands = preprocess_dofile(raw_lines)

    i = 0
    while i < len(commands):
        cmd = commands[i].strip()
        i += 1

        if not cmd:
            continue

        if cmd.lower().startswith("do "):
            from commands.scripting import expand_macros
            raw_path = cmd[3:].strip()
            expanded = expand_macros(raw_path, parser.state)
            nested_path = expanded.strip("\"'")
            ok = run_do_file(nested_path, parser, console)
            if not ok:
                on_err = getattr(parser.state, "on_error", "stop")
                if on_err == "stop":
                    console.print(f"[red]Parent do-file halted after nested "
                                  f"failure in {nested_path}[/red]")
                    return False
            continue

        console.print(f"[dim]. {cmd}[/dim]")

        if cmd.lower() in ("exit", "quit", "q"):
            break

        if cmd.rstrip().endswith("{"):
            is_python_outer = cmd.split(None, 1)[0].lower() in ("python", "py")
            block_lines = []
            depth = 1
            inside_nested_python = False
            nested_depth = 0
            while i < len(commands) and depth > 0:
                raw_bline = commands[i]
                stripped = raw_bline.strip()
                i += 1

                if inside_nested_python:
                    if stripped.endswith("{"):
                        nested_depth += 1
                    if stripped == "}":
                        nested_depth -= 1
                        if nested_depth == 0:
                            inside_nested_python = False
                            block_lines.append(raw_bline)
                            depth -= 1
                            continue
                    block_lines.append(raw_bline)
                    continue

                first_word = (stripped.split(None, 1)[0].lower()
                              if stripped else "")
                opens_python = (first_word in ("python", "py")
                                and stripped.endswith("{"))

                if stripped == "}":
                    depth -= 1
                    if depth == 0:
                        break
                    block_lines.append(
                        stripped if not is_python_outer else raw_bline)
                elif opens_python and not is_python_outer:
                    inside_nested_python = True
                    nested_depth = 1
                    depth += 1
                    block_lines.append(raw_bline)
                else:
                    if stripped.endswith("{"):
                        depth += 1
                    block_lines.append(
                        raw_bline if is_python_outer else stripped)

            parser.state._pending_block = block_lines

        try:
            parser.execute(cmd, reraise=True)
        except Exception as e:
            rc = getattr(parser.state, "_rc", 1) or 1
            console.print(f"[red]Error on line {i} (rc={rc}): {e}[/red]")
            on_err = getattr(parser.state, "on_error", "stop")
            if on_err == "stop":
                console.print(f"[red]Do-file halted at line {i}. "
                              f"Set 'on_error continue' to ignore errors.[/red]")
                return False

    console.print(f"[dim]End of do-file: {filepath}[/dim]")
    return True


class RlnApp(App):
    title = "Rln"

    def build(self):
        self.engine = RlnEngine()
        self.theme_name = "dark"          # screenshots are dark; nicer on phones
        Window.softinput_mode = "below_target"

        # widget registries for theming
        self._cards, self._console_cards = [], []
        self._fg_labels, self._muted_labels, self._console_labels = [], [], []
        self._buttons, self._inputs = [], []

        self.tab_widgets, self.tab_buttons = {}, {}
        self.current_tab = None
        self._last_plot = None
        self.console_fs = 9.0   # zoomable console font size (sp)
        self.script_fs = 11.0   # zoomable script-editor font size (sp)

        self.root = FloatLayout()
        main = BoxLayout(orientation="vertical", padding=dp(6), spacing=dp(6),
                         size_hint=(1, 1))
        self.root.add_widget(main)

        main.add_widget(self._build_topbar())
        main.add_widget(self._build_tabbar())

        # Below the tabs: the active tab on top, plus an always-visible,
        # resizable Do-file/script editor at the bottom (mirrors the desktop
        # phone-split layout). Drag the splitter handle to resize the editor.
        mid = BoxLayout(orientation="vertical")
        self.content = BoxLayout(orientation="vertical")
        mid.add_widget(self.content)
        self.script_splitter = Splitter(
            sizable_from="top", size_hint=(1, None), height=dp(232),
            min_size=dp(96), max_size=dp(620), strip_size=dp(16))
        self.script_splitter.add_widget(self._build_script_pane())
        mid.add_widget(self.script_splitter)
        main.add_widget(mid)

        # build every tab once; keep references so state survives tab switches
        self.tab_widgets["Console"] = self._build_console_tab()
        self.tab_widgets["Variables"] = self._build_variables_tab()
        self.tab_widgets["Data"] = self._build_data_tab()
        self.tab_widgets["Plots"] = self._build_plots_tab()
        self.tab_widgets["Help"] = self._build_help_tab()

        # floating action button (the blue circle). Place it explicitly once
        # the root has a real size; pos_hint alone only resolves on a layout
        # pass and left it parked at the origin.
        self.fab = RoundFab(on_tap=self.open_menu, size_hint=(None, None),
                            size=(dp(54), dp(54)))
        self.root.add_widget(self.fab)
        Clock.schedule_once(lambda _dt: self._place_fab(), 0)
        Window.bind(on_resize=lambda *_: self._place_fab())

        self.select_tab("Console")
        self.apply_theme()
        self._append_console(LOGO + "\n")
        return self.root

    # ------------------------------------------------------------------ topbar
    def _build_topbar(self):
        bar = Card(orientation="horizontal", size_hint_y=None, height=dp(46),
                   padding=(dp(10), dp(4)), spacing=dp(6))
        self._cards.append(bar)
        title = Label(text="[b]Rln (ARLEN)[/b]  v" + __version__, markup=True,
                      halign="left", valign="middle", size_hint_x=1)
        title.bind(size=lambda *_: setattr(title, "text_size", title.size))
        self._fg_labels.append(title)
        bar.add_widget(title)

        self.dark_btn = self._make_button(
            "Dark", self._toggle_theme, size_hint_x=None, width=dp(86))
        bar.add_widget(self.dark_btn)
        bar.add_widget(self._make_button(
            "Menu", lambda *_: self.open_menu(), size_hint_x=None, width=dp(76)))
        return bar

    def _build_tabbar(self):
        scroll = ScrollView(size_hint_y=None, height=dp(44),
                            do_scroll_x=True, do_scroll_y=False,
                            bar_width=0)
        strip = BoxLayout(size_hint_x=None, spacing=dp(4))
        strip.bind(minimum_width=strip.setter("width"))
        for name in TABS:
            btn = ToggleButton(text=name, group="rln_tabs", size_hint_x=None,
                               width=dp(84), font_size=FS_UI,
                               allow_no_selection=False,
                               background_normal="", background_down="")
            btn.bind(on_release=lambda b, n=name: self.select_tab(n))
            self.tab_buttons[name] = btn
            strip.add_widget(btn)
        scroll.add_widget(strip)
        return scroll

    # ------------------------------------------------------------------ tabs
    def _build_console_tab(self):
        box = BoxLayout(orientation="vertical", spacing=dp(6))
        card = Card(orientation="vertical", padding=dp(6))
        self._console_cards.append(card)
        # No wrap + scroll both ways, like the desktop console (wrap="none"):
        # Rich tables render at full width and the user scrolls horizontally
        # instead of the rows wrapping into an unreadable blur.
        scroll = ScrollView(do_scroll_x=True, do_scroll_y=True)
        self.console_label = Label(
            text="", markup=False, halign="left", valign="top",
            font_name="RobotoMono-Regular", font_size=sp(self.console_fs),
            size_hint=(None, None), padding=(dp(4), dp(4)))
        self.console_label.bind(texture_size=lambda *_: setattr(
            self.console_label, "size", self.console_label.texture_size))
        self._console_labels.append(self.console_label)
        scroll.add_widget(self.console_label)
        self.console_scroll = scroll
        card.add_widget(scroll)
        box.add_widget(card)

        row = BoxLayout(size_hint_y=None, height=dp(48), spacing=dp(6))
        self.prompt_label = Label(text=".", size_hint_x=None, width=dp(18),
                                  halign="left", valign="middle")
        self._muted_labels.append(self.prompt_label)
        row.add_widget(self.prompt_label)
        self.cmd = self._make_input(hint="Enter Rln command...",
                                    multiline=False)
        self.cmd.bind(on_text_validate=lambda *_: self._on_run())
        row.add_widget(self.cmd)
        row.add_widget(self._make_button(
            "A-", lambda *_: self._zoom_console(-1),
            size_hint_x=None, width=dp(34)))
        row.add_widget(self._make_button(
            "A+", lambda *_: self._zoom_console(1),
            size_hint_x=None, width=dp(34)))
        self.run_btn = self._make_button("Run", lambda *_: self._on_run(),
                                         size_hint_x=None, width=dp(64))
        row.add_widget(self.run_btn)
        box.add_widget(row)
        return box

    def _build_variables_tab(self):
        box = BoxLayout(orientation="vertical", spacing=dp(4))
        head = BoxLayout(size_hint_y=None, height=dp(30), padding=(dp(6), 0))
        for txt, w in (("Variable", 0.4), ("Type", 0.3), ("Label", 0.3)):
            lab = Label(text="[b]%s[/b]" % txt, markup=True, halign="left",
                        valign="middle", size_hint_x=w)
            lab.bind(size=lambda l, *_: setattr(l, "text_size", l.size))
            self._fg_labels.append(lab)
            head.add_widget(lab)
        box.add_widget(head)
        scroll = ScrollView()
        self.var_grid = GridLayout(cols=3, size_hint_y=None, padding=(dp(6), 0),
                                   spacing=(0, dp(2)))
        self.var_grid.bind(minimum_height=self.var_grid.setter("height"))
        scroll.add_widget(self.var_grid)
        box.add_widget(scroll)
        return box

    def _build_data_tab(self):
        box = BoxLayout(orientation="vertical", spacing=dp(4))
        row = BoxLayout(size_hint_y=None, height=dp(34), spacing=dp(6),
                        padding=(dp(6), 0))
        self.data_status = Label(text="No data loaded", halign="left",
                                 valign="middle")
        self.data_status.bind(size=lambda l, *_: setattr(
            l, "text_size", l.size))
        self._muted_labels.append(self.data_status)
        row.add_widget(self.data_status)
        row.add_widget(self._make_button("Refresh", lambda *_: self.refresh_data(),
                                         size_hint_x=None, width=dp(90)))
        box.add_widget(row)
        card = Card(orientation="vertical", padding=dp(6))
        self._cards.append(card)
        scroll = ScrollView(do_scroll_x=True, do_scroll_y=True)
        self.data_label = Label(text="", font_name="RobotoMono-Regular",
                                font_size=FS_MONO, halign="left", valign="top",
                                size_hint=(None, None), markup=False)
        self.data_label.bind(texture_size=lambda *_: setattr(
            self.data_label, "size", self.data_label.texture_size))
        self._console_labels.append(self.data_label)
        scroll.add_widget(self.data_label)
        card.add_widget(scroll)
        box.add_widget(card)
        return box

    def _build_plots_tab(self):
        box = BoxLayout(orientation="vertical", spacing=dp(4))
        row = BoxLayout(size_hint_y=None, height=dp(40), spacing=dp(6))
        row.add_widget(self._make_button("Refresh plot",
                                         lambda *_: self.refresh_plot()))
        row.add_widget(self._make_button("Export plot",
                                         lambda *_: self.export_plot()))
        box.add_widget(row)
        card = Card(orientation="vertical", padding=dp(6))
        self._cards.append(card)
        self.plot_image = Image(allow_stretch=True, keep_ratio=True)
        self.plot_placeholder = Label(
            text="Run a graph command (e.g. histogram income),\n"
                 "then open this tab or press Refresh plot.",
            halign="center", valign="middle")
        self.plot_placeholder.bind(size=lambda l, *_: setattr(
            l, "text_size", l.size))
        self._muted_labels.append(self.plot_placeholder)
        self.plot_card = card
        card.add_widget(self.plot_placeholder)
        box.add_widget(card)
        return box

    def _build_script_pane(self):
        # Always-visible editor pinned to the bottom (inside the splitter),
        # matching the desktop "Do-file / script" pane.
        box = BoxLayout(orientation="vertical", spacing=dp(4),
                        padding=(dp(2), dp(2), dp(2), 0))
        row = BoxLayout(size_hint_y=None, height=dp(38), spacing=dp(5))
        lab = Label(text="[b]Do-file[/b]", markup=True, halign="left",
                    valign="middle", size_hint_x=1, font_size=FS_UI)
        lab.bind(size=lambda l, *_: setattr(l, "text_size", l.size))
        self._fg_labels.append(lab)
        row.add_widget(lab)
        row.add_widget(self._make_button(
            "A-", lambda *_: self._zoom_script(-1),
            size_hint_x=None, width=dp(32)))
        row.add_widget(self._make_button(
            "A+", lambda *_: self._zoom_script(1),
            size_hint_x=None, width=dp(32)))
        row.add_widget(self._make_button("Save", lambda *_: self.save_script(),
                                         size_hint_x=None, width=dp(52)))
        row.add_widget(self._make_button("Open", lambda *_: self.open_script(),
                                         size_hint_x=None, width=dp(52)))
        row.add_widget(self._make_button("Run script",
                                         lambda *_: self.run_script(),
                                         size_hint_x=None, width=dp(86)))
        box.add_widget(row)
        hint = "Write a multi-line do-file here, then Run script..."
        if HAVE_CODEINPUT:
            # Syntax-highlighted editor (keywords, strings, // and * comments).
            self.script_input = CodeInput(
                lexer=RlnLexer(), hint_text=hint,
                font_name="RobotoMono-Regular", font_size=sp(self.script_fs))
        else:
            self.script_input = TextInput(
                hint_text=hint, multiline=True,
                font_name="RobotoMono-Regular", font_size=sp(self.script_fs))
        box.add_widget(self.script_input)
        return box

    def _build_help_tab(self):
        box = BoxLayout(orientation="vertical", spacing=dp(4))
        row = BoxLayout(size_hint_y=None, height=dp(40), spacing=dp(6))
        row.add_widget(self._make_button(
            "Command help", lambda *_: self._run_now("help")))
        row.add_widget(self._make_button("About Rln",
                                         lambda *_: self.show_about()))
        box.add_widget(row)
        card = Card(orientation="vertical", padding=dp(8))
        self._cards.append(card)
        scroll = ScrollView()
        lab = Label(text=ABOUT_TEXT, halign="left", valign="top",
                    size_hint_y=None, padding=(dp(6), dp(6)))
        lab.bind(width=lambda *_: setattr(lab, "text_size", (lab.width, None)))
        lab.bind(texture_size=lambda *_: setattr(
            lab, "height", lab.texture_size[1]))
        self._fg_labels.append(lab)
        scroll.add_widget(lab)
        card.add_widget(scroll)
        box.add_widget(card)
        return box

    # ------------------------------------------------------------------ helpers
    def _make_button(self, text, on_release, **kwargs):
        btn = Button(text=text, background_normal="", background_down="",
                     **kwargs)
        btn.bind(on_release=on_release)
        self._buttons.append(btn)
        return btn

    def _make_input(self, hint="", multiline=False, **kwargs):
        ti = TextInput(hint_text=hint, multiline=multiline, font_size="13sp",
                       **kwargs)
        self._inputs.append(ti)
        return ti

    def select_tab(self, name):
        if self.current_tab != name:
            self.content.clear_widgets()
            self.content.add_widget(self.tab_widgets[name])
            self.current_tab = name
            if name == "Variables":
                self.refresh_variables()
            elif name == "Data":
                self.refresh_data()
            elif name == "Plots":
                self.refresh_plot()
        for n, b in self.tab_buttons.items():
            b.state = "down" if n == name else "normal"
        self._restyle_tabs()

    def _restyle_tabs(self):
        t = self.theme()
        accent, panel, fg = hx(t["accent"]), hx(t["panel"]), hx(t["fg"])
        for n, b in self.tab_buttons.items():
            sel = (n == self.current_tab)
            b.background_color = accent if sel else panel
            b.color = (1, 1, 1, 1) if sel else fg

    def _place_fab(self):
        self.fab.x = self.root.width - self.fab.width - dp(12)
        self.fab.y = self.root.height - self.fab.height - dp(96)
        self.fab._sync()

    # ------------------------------------------------------------------ zoom
    def _zoom_console(self, delta):
        self.console_fs = max(6.0, min(30.0, self.console_fs + delta))
        self.console_label.font_size = sp(self.console_fs)

    def _zoom_script(self, delta):
        self.script_fs = max(6.0, min(30.0, self.script_fs + delta))
        self.script_input.font_size = sp(self.script_fs)

    # ------------------------------------------------------------------ theme
    def theme(self):
        return THEMES[self.theme_name]

    def _toggle_theme(self, *_):
        self.theme_name = "light" if self.theme_name == "dark" else "dark"
        self.dark_btn.text = "Dark" if self.theme_name == "dark" else "Light"
        self.apply_theme()

    def apply_theme(self):
        t = self.theme()
        bg, fg, panel, entry = hx(t["bg"]), hx(t["fg"]), hx(t["panel"]), hx(t["entry"])
        muted, accent = hx(t["muted"]), hx(t["accent"])
        cbg, cfg = hx(t["console_bg"]), hx(t["console_fg"])
        Window.clearcolor = bg
        for c in self._cards:
            c.set_bg(panel)
        for c in self._console_cards:
            c.set_bg(cbg)
        for l in self._fg_labels:
            l.color = fg
        for l in self._muted_labels:
            l.color = muted
        for l in self._console_labels:
            l.color = cfg
        for b in self._buttons:
            b.background_color = panel
            b.color = fg
        for ti in self._inputs:
            ti.background_color = entry
            ti.foreground_color = fg
            ti.cursor_color = accent
            ti.hint_text_color = muted
        if hasattr(self, "script_input"):
            si = self.script_input
            si.background_color = entry
            si.foreground_color = fg
            si.cursor_color = accent
            si.hint_text_color = muted
            if HAVE_CODEINPUT and hasattr(si, "style_name"):
                # Pygments style whose token colors suit the current theme.
                si.style_name = "native" if self.theme_name == "dark" \
                    else "default"
        self._restyle_tabs()
        if hasattr(self, "fab"):
            self.fab.set_bg(accent)

    # ------------------------------------------------------------------ run
    def _on_run(self):
        cmd = self.cmd.text.strip()
        if not cmd:
            return
        self.cmd.text = ""
        self._run_now(cmd)

    def _run_now(self, cmd):
        self.select_tab("Console")
        self._append_console("\n. " + cmd + "\n")
        self.run_btn.disabled = True
        threading.Thread(target=self._worker, args=(cmd,), daemon=True).start()

    def _worker(self, cmd):
        out = self.engine.run(cmd)
        plot = self.engine.plot_from_output(out)
        Clock.schedule_once(lambda _dt: self._finish(out, plot), 0)

    def _finish(self, out, plot):
        if out:
            self._append_console(out)
        if plot:
            self._last_plot = plot
            self._show_plot(plot)
        self.run_btn.disabled = False
        self.refresh_prompt()
        if self.current_tab == "Variables":
            self.refresh_variables()
        elif self.current_tab == "Data":
            self.refresh_data()
        self.cmd.focus = True

    def _append_console(self, text):
        self.console_label.text += text
        Clock.schedule_once(
            lambda _dt: setattr(self.console_scroll, "scroll_y", 0), 0)

    # ------------------------------------------------------------------ refresh
    def refresh_prompt(self):
        # The single-char prompt mirrors the desktop console gutter. Dataset
        # dimensions are surfaced on the Data tab's status line instead.
        self.prompt_label.text = "."

    def refresh_variables(self):
        self.var_grid.clear_widgets()
        st = self.engine.state
        t = self.theme()
        if not st.has_data():
            return
        for col in st.data.columns:
            try:
                label = st.get_variable_label(col) or ""
            except Exception:
                label = ""
            dtype = str(st.data[col].dtype)
            for txt, w, key in ((str(col), 0.4, "fg"), (dtype, 0.3, "muted"),
                                (str(label), 0.3, "fg")):
                cell = Label(text=txt, halign="left", valign="middle",
                             size_hint=(w, None), height=dp(24),
                             color=hx(t[key]), font_size=FS_TABLE)
                cell.bind(size=lambda l, *_: setattr(l, "text_size", l.size))
                self.var_grid.add_widget(cell)

    def refresh_data(self):
        st = self.engine.state
        if not st.has_data():
            self.data_status.text = "No data loaded"
            self.data_label.text = ""
            return
        df = st.data
        try:
            max_rows = int(st.settings.get("max_display_rows", 200) or 200)
        except Exception:
            max_rows = 200
        view = df.head(max_rows)
        try:
            import pandas as pd
            with pd.option_context("display.max_columns", None,
                                   "display.width", None):
                self.data_label.text = view.to_string(max_rows=max_rows)
        except Exception:
            self.data_label.text = view.to_string()
        more = "" if len(df) <= max_rows else f"; showing first {max_rows}"
        self.data_status.text = (f"{len(df)} obs, {len(df.columns)} vars{more}")

    def refresh_plot(self):
        if self._last_plot and os.path.exists(self._last_plot):
            self._show_plot(self._last_plot)

    def _show_plot(self, path):
        if not (path and os.path.exists(path)):
            return
        self.plot_card.clear_widgets()
        self.plot_image.source = ""
        self.plot_image.source = path
        self.plot_image.reload()
        if self.plot_image.parent is None:
            self.plot_card.add_widget(self.plot_image)

    def export_plot(self):
        if not (self._last_plot and os.path.exists(self._last_plot)):
            self._toast("No plot yet. Run a graph command first.")
            return
        out = os.path.join(writable_dir(), "rln_plot_export.png")
        try:
            import shutil
            shutil.copyfile(self._last_plot, out)
            self.select_tab("Console")
            self._append_console(f"\nSaved plot: {out}\n")
        except Exception as exc:
            self._toast(f"Export failed: {exc}")

    # ------------------------------------------------------------------ scripts
    def run_script(self):
        script = self.script_input.text
        if not script.strip():
            return
        import tempfile
        tmp = tempfile.NamedTemporaryFile(
            "w", suffix=".do", encoding="utf-8", delete=False,
            dir=writable_dir())
        try:
            tmp.write(script)
            path = tmp.name
        finally:
            tmp.close()
        self._run_now(f'do "{path}"')

    def open_script(self):
        if not ensure_storage_access():
            self._toast("Storage access needed. Enable 'All files access' for "
                        "Rln in the screen that opened, then tap Open again.")
            return

        def on_pick(path):
            try:
                with open(path, "r", encoding="utf-8") as fh:
                    self.script_input.text = fh.read()
            except OSError as exc:
                self._toast(str(exc))
        self._file_dialog("Open do-file", on_pick,
                          filters=["*.do", "*.txt", "*"])

    def save_script(self):
        def on_name(name):
            path = os.path.join(writable_dir(), name)
            try:
                with open(path, "w", encoding="utf-8") as fh:
                    fh.write(self.script_input.text)
                self._toast(f"Saved: {path}")
            except OSError as exc:
                self._toast(str(exc))
        self._name_dialog("Save do-file as", "script.do", on_name)

    # ------------------------------------------------------------------ menu
    def open_menu(self):
        view = ModalView(size_hint=(None, None), size=(dp(280), dp(220)),
                         auto_dismiss=True)
        t = self.theme()
        card = Card(orientation="vertical", padding=dp(12), spacing=dp(8))
        card.set_bg(hx(t["panel"]))
        title = Label(text="[b]Rln menu[/b]", markup=True, size_hint_y=None,
                      height=dp(28), color=hx(t["fg"]))
        card.add_widget(title)
        grid = GridLayout(cols=2, spacing=dp(8))
        actions = [
            ("Open", lambda *_: (view.dismiss(), self.menu_open())),
            ("Export", lambda *_: (view.dismiss(), self.menu_export())),
            ("Command", lambda *_: (view.dismiss(), self.menu_command())),
            ("About Rln", lambda *_: (view.dismiss(), self.show_about())),
        ]
        for text, cb in actions:
            b = Button(text=text, background_normal="", background_down="",
                       background_color=hx(t["accent"]), color=(1, 1, 1, 1))
            b.bind(on_release=cb)
            grid.add_widget(b)
        card.add_widget(grid)
        view.add_widget(card)
        view.open()

    def menu_open(self):
        if not ensure_storage_access():
            self._toast("Storage access needed. Enable 'All files access' for "
                        "Rln in the screen that opened, then tap Open again.")
            return

        def on_pick(path):
            self._run_now(f'use "{path}"')
        self._file_dialog("Open dataset", on_pick, filters=[
            "*.csv", "*.dta", "*.xlsx", "*.xls", "*.parquet", "*.json",
            "*.feather", "*.tsv", "*.txt", "*"])

    def menu_export(self):
        if not self.engine.state.has_data():
            self._toast("No dataset is currently loaded.")
            return

        def on_name(name):
            path = os.path.join(writable_dir(), name)
            self._run_now(f'save "{path}", replace')
        self._name_dialog("Export dataset as", "export.csv", on_name)

    def menu_command(self):
        self.select_tab("Console")
        self.cmd.focus = True

    def show_about(self):
        view = ModalView(size_hint=(0.9, 0.7))
        t = self.theme()
        card = Card(orientation="vertical", padding=dp(12), spacing=dp(8))
        card.set_bg(hx(t["panel"]))
        scroll = ScrollView()
        lab = Label(text=ABOUT_TEXT, halign="left", valign="top",
                    size_hint_y=None, color=hx(t["fg"]), padding=(dp(6), dp(6)))
        lab.bind(width=lambda *_: setattr(lab, "text_size", (lab.width, None)))
        lab.bind(texture_size=lambda *_: setattr(
            lab, "height", lab.texture_size[1]))
        scroll.add_widget(lab)
        card.add_widget(scroll)
        btn = Button(text="Close", size_hint_y=None, height=dp(44),
                     background_normal="", background_color=hx(t["accent"]),
                     color=(1, 1, 1, 1))
        btn.bind(on_release=lambda *_: view.dismiss())
        card.add_widget(btn)
        view.add_widget(card)
        view.open()

    # ------------------------------------------------------------------ dialogs
    def _file_dialog(self, title, on_pick, filters=None):
        view = ModalView(size_hint=(0.96, 0.92))
        t = self.theme()
        card = Card(orientation="vertical", padding=dp(8), spacing=dp(6))
        card.set_bg(hx(t["panel"]))
        card.add_widget(Label(text=title, size_hint_y=None, height=dp(26),
                              color=hx(t["fg"]), font_size=FS_UI))
        path_lbl = Label(text="", size_hint_y=None, height=dp(22),
                         color=hx(t["muted"]), font_size="10sp",
                         halign="left", valign="middle", shorten=True)
        path_lbl.bind(size=lambda l, *_: setattr(l, "text_size", l.size))
        card.add_widget(path_lbl)
        chooser = FileChooserListView(path=default_open_dir(),
                                      filters=filters or ["*"])
        chooser.bind(path=lambda _c, p: setattr(path_lbl, "text", p))
        path_lbl.text = chooser.path
        card.add_widget(chooser)
        row = BoxLayout(size_hint_y=None, height=dp(46), spacing=dp(8))

        def do_open(*_):
            if chooser.selection:
                view.dismiss()
                on_pick(chooser.selection[0])
        root_btn = Button(text="Storage root", background_normal="",
                          background_color=hx(t["entry"]), color=hx(t["fg"]),
                          size_hint_x=None, width=dp(116))
        root_btn.bind(on_release=lambda *_: setattr(
            chooser, "path", browse_root()))
        ok = Button(text="Open", background_normal="",
                    background_color=hx(t["accent"]), color=(1, 1, 1, 1))
        ok.bind(on_release=do_open)
        cancel = Button(text="Cancel", background_normal="",
                        background_color=hx(t["entry"]), color=hx(t["fg"]),
                        size_hint_x=None, width=dp(84))
        cancel.bind(on_release=lambda *_: view.dismiss())
        row.add_widget(root_btn)
        row.add_widget(ok)
        row.add_widget(cancel)
        card.add_widget(row)
        view.add_widget(card)
        view.open()

    def _name_dialog(self, title, default, on_name):
        view = ModalView(size_hint=(None, None), size=(dp(300), dp(180)))
        t = self.theme()
        card = Card(orientation="vertical", padding=dp(12), spacing=dp(8))
        card.set_bg(hx(t["panel"]))
        card.add_widget(Label(text=title, size_hint_y=None, height=dp(28),
                              color=hx(t["fg"])))
        ti = TextInput(text=default, multiline=False, size_hint_y=None,
                       height=dp(44), background_color=hx(t["entry"]),
                       foreground_color=hx(t["fg"]))
        card.add_widget(ti)
        row = BoxLayout(size_hint_y=None, height=dp(46), spacing=dp(8))

        def do_save(*_):
            name = ti.text.strip()
            if name:
                view.dismiss()
                on_name(name)
        ok = Button(text="Save", background_normal="",
                    background_color=hx(t["accent"]), color=(1, 1, 1, 1))
        ok.bind(on_release=do_save)
        cancel = Button(text="Cancel", background_normal="",
                        background_color=hx(t["entry"]), color=hx(t["fg"]))
        cancel.bind(on_release=lambda *_: view.dismiss())
        row.add_widget(ok)
        row.add_widget(cancel)
        card.add_widget(row)
        view.add_widget(card)
        view.open()

    def _toast(self, message):
        view = ModalView(size_hint=(0.8, None), height=dp(140))
        t = self.theme()
        card = Card(orientation="vertical", padding=dp(12), spacing=dp(8))
        card.set_bg(hx(t["panel"]))
        lab = Label(text=message, color=hx(t["fg"]), halign="center",
                    valign="middle")
        lab.bind(size=lambda l, *_: setattr(l, "text_size", l.size))
        card.add_widget(lab)
        btn = Button(text="OK", size_hint_y=None, height=dp(42),
                     background_normal="", background_color=hx(t["accent"]),
                     color=(1, 1, 1, 1))
        btn.bind(on_release=lambda *_: view.dismiss())
        card.add_widget(btn)
        view.add_widget(card)
        view.open()


if __name__ == "__main__":
    RlnApp().run()
