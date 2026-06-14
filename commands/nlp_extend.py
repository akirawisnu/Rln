"""
NLP extension module for Rln.

This module extends the functionality of nlp.py WITHOUT modifying it.
It provides a drop-in replacement dispatcher `cmd_nlp` that handles
all the subcommands of the original `hf` command, plus two offline-first
re-implementations:

  * translate  — via argos-translate (offline OpenNMT/CTranslate2 models)
  * summarize  — via sumy (extractive summarization, no neural model)

All other subcommands (classify, sentiment, ner, embed, models, cache,
download) are delegated to the existing nlp._hf_* helpers so that
previously-working dofiles continue to work identically.

The user-facing command name is `nlp` (e.g. `nlp translate text, from(id) to(en)`).
For backwards compatibility, `hf` is kept as an alias for `nlp` in parser.py,
so existing scripts using `hf ...` continue to work unchanged.

Portability:
  * Argos translation models are stored in <project_root>/argos_models/,
    controlled via the ARGOS_PACKAGES_DIR environment variable. Move the
    whole rln/ folder between machines/OSes and translation keeps working.
  * Sumy has no model weights — it's a pure-Python extractive summarizer.
    A lightweight regex tokenizer and tiny stopword list are bundled so
    the module works even without any NLTK data downloads.

Syntax (all existing `hf ...` forms still work with `nlp ...`):

  nlp classify var, labels("lab1 lab2 lab3") [generate(newvar) model(name)]
  nlp sentiment var [, generate(newvar) model(name)]
  nlp ner var [, generate(newvar) model(name)]
  nlp embed var [, generate(stub) model(name) dims(N)]
  nlp models [task]
  nlp cache

  nlp translate var, from(src) to(tgt) [generate(newvar) pivot(en)]
  nlp summarize var [, generate(newvar) sentences(3) method(lsa|lexrank|textrank|luhn|kl) language(english)]
                      # legacy maxlen()/minlen() accepted as aliases

  nlp download translate <src> <tgt>   — download an argos-translate pair into argos_models/
  nlp download <huggingface_model>      — delegates to hf download (neural summary / classify / etc.)
"""

import os
import re
import sys
import time
import zipfile
import pathlib
import urllib.request
import urllib.error

import pandas as pd
from rich.console import Console
from rich.table import Table

from commands.state import AppState
from commands.parse_helpers import parse_command_line
from commands.expression import eval_condition


# ══════════════════════════════════════════════════════════════════════
#  Paths — portable model storage
# ══════════════════════════════════════════════════════════════════════

def _project_root() -> str:
    """Resolve the portable Rln root folder.

    PyInstaller portable builds set RLN_PORTABLE_ROOT to the directory
    beside the executable, so Argos models remain outside the bundled
    internals and can be copied, updated, or moved with the app folder.
    Source mode falls back to the rln/ folder one level above commands/.
    """
    portable_root = os.environ.get("RLN_PORTABLE_ROOT")
    if portable_root:
        return portable_root
    this_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.dirname(this_dir)


def _argos_dir() -> str:
    """The folder that holds argos-translate packages. Portable across OSes."""
    d = os.path.join(_project_root(), "argos_models")
    os.makedirs(d, exist_ok=True)
    return d


def _setup_argos_env():
    """Pin argos-translate's model / cache dirs to rln/argos_models/
    BEFORE the argostranslate modules are imported the first time.

    argostranslate reads ARGOS_PACKAGES_DIR at import time to set
    settings.package_data_dir. We also disable any network auto-update
    so offline use is the default.
    """
    base = _argos_dir()
    os.environ.setdefault("ARGOS_PACKAGES_DIR", base)
    # Keep argos's config + cache inside the portable folder too so that
    # moving the rln/ folder is genuinely self-contained.
    os.environ.setdefault("ARGOS_PACKAGES_DOWNLOAD_URL",
                          "https://data.argosopentech.com/argospm/v1/")
    return base


# ══════════════════════════════════════════════════════════════════════
#  Main dispatcher — routes to nlp._hf_* for familiar subcommands,
#  and to local _nlp_* handlers for translate + summarize.
# ══════════════════════════════════════════════════════════════════════

def cmd_nlp(rest: str, state: AppState, console: Console):
    """
    nlp <task> <args>
    Offline-first NLP commands.

    Tasks:
      translate, summarize    — offline (argos-translate, sumy)
      classify, sentiment,
      ner, embed, models,
      cache, download         — delegated to nlp (HuggingFace backend)
    """
    parts = rest.strip().split(None, 1)
    if not parts:
        _show_nlp_help(console)
        return

    task = parts[0].lower()
    sub_rest = parts[1] if len(parts) > 1 else ""

    # Local handlers
    local = {
        "translate": _nlp_translate,
        "summarize": _nlp_summarize,
        "summarise": _nlp_summarize,
        "download":  _nlp_download,
    }
    if task in local:
        local[task](sub_rest, state, console)
        return

    # Help / cache: custom preambles then delegate
    if task in ("help", "?"):
        _show_nlp_help(console)
        return

    # Everything else: delegate to the original hf dispatcher so existing
    # logic (zero-shot classify, sentiment, ner, embed, models, cache) is
    # reused unchanged. We pass the same `rest` string to preserve syntax.
    from commands import nlp as _nlp
    _nlp.cmd_hf(rest, state, console)


def _show_nlp_help(console: Console):
    console.print("[dim]Rln NLP commands (offline-first):[/dim]")
    console.print("  [cyan]nlp translate[/cyan]  var, from(src) to(tgt) [generate(newvar) pivot(en)]")
    console.print("                 [dim]argos-translate — offline, portable[/dim]")
    console.print("  [cyan]nlp summarize[/cyan]  var [, generate(newvar) sentences(3) method(lsa|lexrank|textrank|luhn|kl)]")
    console.print("                 [dim]sumy — extractive, no neural model[/dim]")
    console.print("  [cyan]nlp classify[/cyan]   var, labels(\"pos neg neu\") [generate(newvar)]")
    console.print("  [cyan]nlp sentiment[/cyan]  var [, generate(newvar)]")
    console.print("  [cyan]nlp ner[/cyan]        var [, generate(newvar)]")
    console.print("  [cyan]nlp embed[/cyan]      var [, generate(stub) dims(384)]")
    console.print("  [cyan]nlp models[/cyan]     [task]")
    console.print("  [cyan]nlp cache[/cyan]      show model cache info")
    console.print("  [cyan]nlp download translate[/cyan] <src> <tgt>    argos pair -> argos_models/")
    console.print("  [cyan]nlp download[/cyan] <hf_model>                HF weights  -> hf_models/")
    console.print()
    console.print(f"[dim]Argos models dir:    {_argos_dir()}[/dim]")


# ══════════════════════════════════════════════════════════════════════
#  translate — argos-translate backend
# ══════════════════════════════════════════════════════════════════════

def _check_argos():
    """Ensure argos-translate is installed and pointed at the local model dir."""
    _setup_argos_env()
    try:
        import argostranslate.package  # noqa: F401
        import argostranslate.translate  # noqa: F401
    except ImportError:
        raise ImportError(
            "argos-translate is required for offline translation.\n"
            "Install with: ssc install argostranslate"
        )


def _argos_installed_pairs():
    """Return a set of (from_code, to_code) for installed argos packages."""
    _check_argos()
    import argostranslate.package
    pairs = set()
    for pkg in argostranslate.package.get_installed_packages():
        pairs.add((pkg.from_code, pkg.to_code))
    return pairs


def _argos_direct_or_pivot(src: str, tgt: str, pivot: str = "en"):
    """
    Return a list of (from, to) hops needed to go src -> tgt using
    installed argos packages.
      * If a direct pair is installed, returns [(src, tgt)].
      * If src->pivot and pivot->tgt are both installed, returns
        [(src, pivot), (pivot, tgt)].
      * Otherwise returns None.
    """
    installed = _argos_installed_pairs()
    if (src, tgt) in installed:
        return [(src, tgt)]
    if src == tgt:
        return []  # no-op
    if (src, pivot) in installed and (pivot, tgt) in installed and src != pivot and tgt != pivot:
        return [(src, pivot), (pivot, tgt)]
    return None


def _nlp_translate(rest: str, state: AppState, console: Console):
    """
    nlp translate var, from(src) to(tgt) [generate(newvar) pivot(en) if <cond>]

    Offline translation using argos-translate. Models are stored in
    <project_root>/argos_models/ and survive a folder move between machines.

    If a direct language pair is not installed, the translator will
    automatically pivot through English (configurable via `pivot()`)
    when both src->pivot and pivot->tgt are available.

    Download pairs first with:
      nlp download translate <src> <tgt>
    """
    state.require_data()
    _check_argos()
    parsed = parse_command_line(rest)

    if not parsed["varlist"]:
        console.print("[red]Syntax: nlp translate var, from(src) to(tgt) [generate(newvar)][/red]")
        return

    var = parsed["varlist"][0]
    if var not in state.data.columns:
        console.print(f"[red]Variable '{var}' not found[/red]")
        return

    src = parsed["options"].get("from") or parsed["options"].get("source")
    tgt = parsed["options"].get("to") or parsed["options"].get("target")
    if not src or not tgt:
        console.print("[red]Must specify from() and to() language codes[/red]")
        console.print("[dim]Example: nlp translate text, from(id) to(en) generate(text_en)[/dim]")
        return
    src, tgt = src.lower(), tgt.lower()

    pivot = (parsed["options"].get("pivot") or "en").lower()
    gen_var = parsed["options"].get("generate") or parsed["options"].get("gen")
    out_col = gen_var or f"{var}_{tgt}"

    # Resolve translation path
    hops = _argos_direct_or_pivot(src, tgt, pivot)
    if hops is None:
        # v1.1.1: try one-shot auto-download before giving up.
        # This makes 'nlp translate' usable even without a prior
        # 'nlp download translate ...' step, as long as the machine
        # is online once.
        auto_enabled = os.environ.get("RLN_NLP_NO_AUTODOWNLOAD") != "1"
        if auto_enabled:
            console.print(f"[dim]No installed path for {src}->{tgt}. "
                          f"Attempting one-shot auto-download...[/dim]")
            if _try_autodownload_argos(src, tgt, pivot, console):
                hops = _argos_direct_or_pivot(src, tgt, pivot)
        if hops is None:
            _report_missing_argos_pair(src, tgt, pivot, console)
            return

    # Apply optional if-filter
    df = state.data
    mask = None
    if parsed["if_cond"]:
        mask = eval_condition(parsed["if_cond"], df)
        working_series = df.loc[mask, var]
    else:
        working_series = df[var]

    import argostranslate.translate as atrans

    if hops:
        path_desc = " -> ".join([src] + [t for _, t in hops])
    else:
        path_desc = f"{src} -> {tgt} (no-op)"
    console.print(f"[dim]Translating {len(working_series)} texts ({path_desc})...[/dim]")

    texts = working_series.fillna("").astype(str).tolist()
    translations = []
    t0 = time.time()
    # argos-translate is sentence-by-sentence; batching is a Python loop.
    # We report progress every ~max(1, N/20) items.
    N = len(texts)
    step = max(1, N // 20)
    for i, text in enumerate(texts, 1):
        if not text.strip():
            translations.append("")
        else:
            try:
                if not hops:
                    translations.append(text)
                else:
                    cur = text
                    for h_src, h_tgt in hops:
                        cur = atrans.translate(cur, h_src, h_tgt)
                    translations.append(cur)
            except Exception as e:
                translations.append("")
                if i == 1:  # only warn once to avoid spamming
                    console.print(f"[yellow]Translation warning on row {i}: {e}[/yellow]")
        if i % step == 0 or i == N:
            elapsed = time.time() - t0
            rate = i / elapsed if elapsed > 0 else 0
            console.print(f"[dim]  {i}/{N} ({i / N * 100:.0f}%) | {rate:.1f} texts/s[/dim]")

    if mask is not None:
        if out_col not in state.data.columns:
            state.data[out_col] = ""
        state.data.loc[mask, out_col] = translations
    else:
        state.data[out_col] = translations

    console.print(f"\n[green]Translation complete ({path_desc}):[/green]")
    console.print(f"  Generated: {out_col}")
    console.print(f"  {len([t for t in translations if t])} of {N} texts translated")
    console.print(f"  Backend: argos-translate (offline)")
    state.mark_changed()


def _report_missing_argos_pair(src, tgt, pivot, console):
    console.print(f"[red]No installed argos-translate path for {src} -> {tgt}.[/red]")
    console.print("[dim]Installed pairs:[/dim]")
    pairs = sorted(_argos_installed_pairs())
    if not pairs:
        console.print("[dim]  (none installed yet)[/dim]")
    else:
        for a, b in pairs:
            console.print(f"  [cyan]{a} -> {b}[/cyan]")
    # Each print() is rendered as one rich string — markup tags must be
    # balanced within a single print call, so close every [dim] we open.
    console.print(f"[dim]To install the direct pair:   nlp download translate {src} {tgt}[/dim]")
    console.print(f"[dim]Or install {src}->{pivot} and {pivot}->{tgt} for pivot translation.[/dim]")
    console.print(f"[dim]Models will be stored in: {_argos_dir()}[/dim]")
    console.print("[dim]Disable auto-download with: set RLN_NLP_NO_AUTODOWNLOAD=1[/dim]")


def _try_autodownload_argos(src, tgt, pivot, console) -> bool:
    """Attempt to download whatever argos packages are needed to go src->tgt.

    Strategy:
      1. Refresh the package index (one network call).
      2. If a direct src->tgt package exists, install it.
      3. Otherwise, if both src->pivot and pivot->tgt exist, install
         both — Rln's _argos_direct_or_pivot() will then happily pivot.

    Returns True if at least one install succeeded and a viable path now
    exists, False otherwise. Network failures (no internet, DNS, firewall)
    are caught and reported gracefully so we never crash a translation run.
    """
    _check_argos()
    import argostranslate.package as apkg

    try:
        apkg.update_package_index()
    except Exception as e:
        console.print(f"[yellow]  Auto-download skipped: cannot reach "
                      f"argos package index ({type(e).__name__}).[/yellow]")
        return False

    try:
        available = apkg.get_available_packages()
    except Exception as e:
        console.print(f"[yellow]  Auto-download skipped: cannot list "
                      f"available packages ({type(e).__name__}).[/yellow]")
        return False

    # Look up helper
    def _find(a, b):
        for p in available:
            if p.from_code == a and p.to_code == b:
                return p
        return None

    def _install(pkg, label):
        try:
            console.print(f"[dim]  Downloading {label}...[/dim]")
            path = pkg.download()
            apkg.install_from_path(path)
            return True
        except Exception as e:
            console.print(f"[yellow]  {label} install failed: {e}[/yellow]")
            return False

    # Direct pair first
    direct = _find(src, tgt)
    if direct is not None:
        if _install(direct, f"{src} -> {tgt} package"):
            console.print(f"[green]  Auto-downloaded: {src} -> {tgt}[/green]")
            return True

    # Else try pivot via English (or user-chosen pivot)
    if src != pivot and tgt != pivot:
        p1 = _find(src, pivot)
        p2 = _find(pivot, tgt)
        if p1 is not None and p2 is not None:
            # Install whichever is missing
            installed = _argos_installed_pairs()
            ok = True
            if (src, pivot) not in installed:
                ok &= _install(p1, f"{src} -> {pivot} (pivot leg 1)")
            if (pivot, tgt) not in installed:
                ok &= _install(p2, f"{pivot} -> {tgt} (pivot leg 2)")
            if ok:
                console.print(f"[green]  Auto-downloaded pivot path: "
                              f"{src} -> {pivot} -> {tgt}[/green]")
                return True

    console.print(f"[yellow]  Argos has no direct or {pivot}-pivot path "
                  f"available for {src} -> {tgt}.[/yellow]")
    return False


# ══════════════════════════════════════════════════════════════════════
#  summarize — sumy backend (extractive, offline)
# ══════════════════════════════════════════════════════════════════════

# A minimal stopword set. Used as a fallback when NLTK's `stopwords`
# corpus is unavailable (i.e. in strict offline mode). Small but enough
# for LSA/LexRank/TextRank to behave sensibly on English text.
_TINY_STOPWORDS = {
    "english": set("""
        a an and are as at be been being but by can could did do does doing done
        down during each for from further had has have having he her here hers
        herself him himself his how i if in into is it its itself just me more
        most my myself nor not now of off on once only or other our ours ourselves
        out over own s same she should so some such t than that the their theirs
        them themselves then there these they this those through to too under until
        up very was we were what when where which while who whom why will with would
        you your yours yourself yourselves
    """.split()),
    # Minimal sets for a few common languages. Users can also install NLTK
    # data packs and we'll pick those up automatically.
    "german":    set("und oder der die das ein eine ist war sind sein im in an auf für nicht mit von zu dem den des es sie er wir ihr ihnen".split()),
    "french":    set("et ou le la les un une est était sont être dans sur pour avec de du des à au aux il elle nous vous ils elles ne pas".split()),
    "spanish":   set("y o el la los las un una es era son ser en sobre para con de del al a él ella nosotros vosotros ellos ellas no".split()),
    "indonesian": set("dan atau yang adalah ini itu di ke pada untuk dengan dari dalam tidak akan sudah telah ada saya kamu dia kami mereka".split()),
}


def _check_sumy():
    try:
        import sumy  # noqa: F401
    except ImportError:
        raise ImportError(
            "sumy is required for extractive summarization.\n"
            "Install with: ssc install sumy"
        )


class _OfflineTokenizer:
    """Sumy-compatible sentence/word tokenizer that needs no NLTK data.

    Sumy's Tokenizer(language) tries to load NLTK's punkt_tab and crashes
    if that corpus isn't available. This fallback implements the two
    methods sumy calls: `to_sentences(str)` and `to_words(str)`.

    When NLTK data *is* available, sumy's own Tokenizer is preferred
    because it handles abbreviations more gracefully.
    """
    def __init__(self, language="english"):
        self.language = language

    def to_sentences(self, paragraph):
        if not paragraph:
            return ()
        # Split on sentence-terminating punctuation followed by whitespace.
        parts = re.split(r"(?<=[.!?])\s+(?=[A-Z\u00C0-\u024F\u0400-\u04FF])", paragraph.strip())
        # Fallback: if the regex produced nothing (e.g. text without any
        # capitalized sentence starts), split on punctuation alone.
        if len(parts) <= 1:
            parts = re.split(r"(?<=[.!?])\s+", paragraph.strip())
        return tuple(p.strip() for p in parts if p.strip())

    def to_words(self, sentence):
        # Unicode letter classes for Latin-1 supplement + Latin Extended + Cyrillic
        return tuple(w.lower() for w in re.findall(
            r"[A-Za-z\u00C0-\u024F\u0400-\u04FF]+", sentence
        ))


class _OfflineStemmer:
    """Null stemmer — just lowercase. Adequate for extractive summary."""
    def __call__(self, word):
        return word.lower() if word else word


def _make_tokenizer_stemmer(language):
    """Return (tokenizer, stemmer, stop_words) for the given language.

    Tries sumy's native Tokenizer + NLTK stopwords first (better quality);
    falls back to _OfflineTokenizer + _TINY_STOPWORDS on any failure.
    """
    from sumy.nlp.tokenizers import Tokenizer as SumyTokenizer
    # Try native tokenizer (needs NLTK data)
    try:
        tok = SumyTokenizer(language)
        # touch it once to trigger the NLTK lookup error if data is missing
        tok.to_sentences("Test.")
    except Exception:
        tok = _OfflineTokenizer(language)

    # Stemmer — NLTK Snowball; fallback to null stemmer
    try:
        from sumy.nlp.stemmers import Stemmer as SumyStemmer
        stemmer = SumyStemmer(language)
        # Probe it — if NLTK data missing this may raise lazily
        _ = stemmer("running")
    except Exception:
        stemmer = _OfflineStemmer()

    # Stopwords — NLTK; fallback to our tiny set
    try:
        from sumy.utils import get_stop_words
        stop = set(get_stop_words(language))
    except Exception:
        stop = _TINY_STOPWORDS.get(language.lower(), _TINY_STOPWORDS["english"])

    return tok, stemmer, stop


_SUMY_METHODS = {
    "lsa":      "sumy.summarizers.lsa.LsaSummarizer",
    "lexrank":  "sumy.summarizers.lex_rank.LexRankSummarizer",
    "lex-rank": "sumy.summarizers.lex_rank.LexRankSummarizer",
    "lex_rank": "sumy.summarizers.lex_rank.LexRankSummarizer",
    "textrank": "sumy.summarizers.text_rank.TextRankSummarizer",
    "text-rank": "sumy.summarizers.text_rank.TextRankSummarizer",
    "text_rank": "sumy.summarizers.text_rank.TextRankSummarizer",
    "luhn":     "sumy.summarizers.luhn.LuhnSummarizer",
    "kl":       "sumy.summarizers.kl.KLSummarizer",
    "sumbasic": "sumy.summarizers.sum_basic.SumBasicSummarizer",
    "sum-basic": "sumy.summarizers.sum_basic.SumBasicSummarizer",
    "edmundson": "sumy.summarizers.edmundson.EdmundsonSummarizer",
}


def _load_summarizer_class(method: str):
    method = (method or "lsa").lower()
    dotted = _SUMY_METHODS.get(method)
    if dotted is None:
        raise ValueError(f"Unknown sumy method: {method!r}. "
                         f"Choose from: {sorted(set(_SUMY_METHODS))}")
    mod_path, cls_name = dotted.rsplit(".", 1)
    import importlib
    return getattr(importlib.import_module(mod_path), cls_name)


def _nlp_summarize(rest: str, state: AppState, console: Console):
    """
    nlp summarize var [, generate(newvar) sentences(3) method(lsa|lexrank|textrank|luhn|kl)
                        language(english) if <cond>]

    Extractive summarization with sumy — picks the N most informative
    sentences from the input text. Runs entirely offline in pure Python.

    Legacy hf-summarize options are accepted as aliases:
      maxlen(N)  -> sentences  ~  max(1, N / 40)   (N chars)
      minlen(N)  -> ignored (sumy is extractive; min length is unused)
    """
    state.require_data()
    _check_sumy()
    parsed = parse_command_line(rest)

    if not parsed["varlist"]:
        console.print("[red]Syntax: nlp summarize var [, generate(newvar) sentences(3) method(lsa)][/red]")
        return

    var = parsed["varlist"][0]
    if var not in state.data.columns:
        console.print(f"[red]Variable '{var}' not found[/red]")
        return

    opts = parsed["options"]
    gen_var = opts.get("generate") or opts.get("gen")
    method = (opts.get("method") or "lsa").lower()
    language = (opts.get("language") or opts.get("lang") or "english").lower()

    # sentences() is canonical; maxlen() is a legacy alias.
    if "sentences" in opts:
        n_sent = int(opts["sentences"])
    elif "sent" in opts:
        n_sent = int(opts["sent"])
    elif "maxlen" in opts or "maxlength" in opts:
        maxlen = int(opts.get("maxlen") or opts.get("maxlength") or 150)
        n_sent = max(1, round(maxlen / 40))
    else:
        n_sent = 3
    n_sent = max(1, n_sent)

    try:
        SummClass = _load_summarizer_class(method)
    except Exception as e:
        console.print(f"[red]{e}[/red]")
        return

    tokenizer, stemmer, stop_words = _make_tokenizer_stemmer(language)

    # Build summarizer once and reuse
    try:
        summ = SummClass(stemmer)
    except TypeError:
        # Some summarizers take no args
        summ = SummClass()
    try:
        summ.stop_words = stop_words
    except Exception:
        pass

    df = state.data
    mask = None
    if parsed["if_cond"]:
        mask = eval_condition(parsed["if_cond"], df)
        working_series = df.loc[mask, var]
    else:
        working_series = df[var]

    console.print(f"[dim]Summarizing {len(working_series)} texts "
                  f"(method={method}, sentences={n_sent}, language={language})...[/dim]")

    from sumy.parsers.plaintext import PlaintextParser

    texts = working_series.fillna("").astype(str).tolist()
    summaries = []
    t0 = time.time()
    N = len(texts)
    step = max(1, N // 20)

    for i, text in enumerate(texts, 1):
        try:
            if not text.strip():
                summaries.append("")
                continue
            parser = PlaintextParser.from_string(text, tokenizer)
            out = summ(parser.document, n_sent)
            summary = " ".join(str(s) for s in out)
            # If the extractive algorithm returned nothing (very short input),
            # fall back to the first sentence or the original text.
            if not summary.strip():
                sents = tokenizer.to_sentences(text)
                summary = sents[0] if sents else text
            summaries.append(summary)
        except Exception as e:
            summaries.append("")
            if i == 1:
                console.print(f"[yellow]Summarization warning on row {i}: {e}[/yellow]")
        if i % step == 0 or i == N:
            elapsed = time.time() - t0
            rate = i / elapsed if elapsed > 0 else 0
            console.print(f"[dim]  {i}/{N} ({i / N * 100:.0f}%) | {rate:.1f} texts/s[/dim]")

    out_col = gen_var or f"{var}_summary"
    if mask is not None:
        if out_col not in state.data.columns:
            state.data[out_col] = ""
        state.data.loc[mask, out_col] = summaries
    else:
        state.data[out_col] = summaries

    # Report
    import numpy as _np
    lens = [len(s) for s in summaries if s]
    avg_len = _np.mean(lens) if lens else 0
    console.print(f"\n[green]Summarization complete:[/green]")
    console.print(f"  Generated: {out_col}")
    console.print(f"  Average summary length: {avg_len:.0f} chars  ({n_sent} sentences, {method})")
    console.print(f"  Backend: sumy (extractive, offline)")
    state.mark_changed()


# ══════════════════════════════════════════════════════════════════════
#  download — dual-mode: translate <src> <tgt> or <hf_model>
# ══════════════════════════════════════════════════════════════════════

def _nlp_download(rest: str, state: AppState, console: Console):
    """
    nlp download translate <src> <tgt>    Install an argos-translate pair
    nlp download summary                   Verify sumy backend is working
    nlp download <hf_model_name>           Delegate to hf download
    """
    tokens = rest.strip().split()
    if not tokens:
        console.print("[red]Syntax: nlp download translate <src> <tgt>   |   "
                      "nlp download <hf_model>[/red]")
        return

    head = tokens[0].lower()

    if head == "translate":
        if len(tokens) < 3:
            console.print("[red]Syntax: nlp download translate <src> <tgt>[/red]")
            console.print("[dim]Example: nlp download translate en de[/dim]")
            return
        src, tgt = tokens[1].lower(), tokens[2].lower()
        _download_argos_pair(src, tgt, console)
        return

    if head == "summary" or head in ("summarize", "summarise"):
        _verify_sumy(console)
        return

    # Fall through: treat the whole rest as an HF model name (legacy).
    from commands import nlp as _nlp
    _nlp._hf_download(rest, state, console)


def _download_argos_pair(src: str, tgt: str, console: Console):
    """Download+install a direct argos-translate package, or warn about pivoting."""
    _check_argos()
    import argostranslate.package as apkg

    argos_dir = _argos_dir()
    console.print(f"[dim]Argos models dir: {argos_dir}[/dim]")
    console.print(f"[dim]Updating argos package index (needs internet)...[/dim]")

    try:
        apkg.update_package_index()
    except Exception as e:
        console.print(f"[red]Could not reach argos package index: {e}[/red]")
        console.print(f"[dim]Tip: if you already have a .argosmodel file, drop it in {argos_dir} "
                      "and run your translate command — Rln will use it.[/dim]")
        return

    try:
        available = apkg.get_available_packages()
    except Exception as e:
        console.print(f"[red]Could not list available packages: {e}[/red]")
        return

    candidates = [p for p in available if p.from_code == src and p.to_code == tgt]
    if not candidates:
        console.print(f"[yellow]No direct {src} -> {tgt} package published by argos.[/yellow]")
        # Try to auto-install pivot legs via English so the user doesn't
        # have to issue two extra commands manually.
        src_to_en = next((p for p in available if p.from_code == src and p.to_code == "en"), None)
        en_to_tgt = next((p for p in available if p.from_code == "en" and p.to_code == tgt), None)
        if src_to_en is not None and en_to_tgt is not None:
            console.print(f"[dim]Auto-installing pivot legs: {src} -> en -> {tgt}...[/dim]")
            installed = _argos_installed_pairs()
            ok = True
            if (src, "en") not in installed:
                try:
                    apkg.install_from_path(src_to_en.download())
                    console.print(f"[green]  Installed: {src} -> en[/green]")
                except Exception as e:
                    console.print(f"[red]  {src} -> en failed: {e}[/red]")
                    ok = False
            if ok and ("en", tgt) not in installed:
                try:
                    apkg.install_from_path(en_to_tgt.download())
                    console.print(f"[green]  Installed: en -> {tgt}[/green]")
                except Exception as e:
                    console.print(f"[red]  en -> {tgt} failed: {e}[/red]")
                    ok = False
            if ok:
                console.print(f"[green]Ready: Rln will pivot {src} -> en -> {tgt} "
                              f"at translation time.[/green]")
        return

    pkg = candidates[0]
    console.print(f"[dim]Downloading {src} -> {tgt} package ({pkg})...[/dim]")
    try:
        downloaded_path = pkg.download()
        apkg.install_from_path(downloaded_path)
    except Exception as e:
        console.print(f"[red]Download/install failed: {e}[/red]")
        return

    console.print(f"[green]Installed: {src} -> {tgt}[/green]")
    console.print(f"[dim]Location: {argos_dir}[/dim]")
    console.print(f"[dim]This model will work offline when you transfer the rln/ folder.[/dim]")


def _verify_sumy(console: Console):
    """Smoke-test the sumy backend — no download, just a self-check."""
    try:
        _check_sumy()
        tok, stem, stop = _make_tokenizer_stemmer("english")
        from sumy.parsers.plaintext import PlaintextParser
        SummClass = _load_summarizer_class("lsa")
        try:
            s = SummClass(stem)
        except TypeError:
            s = SummClass()
        try:
            s.stop_words = stop
        except Exception:
            pass
        parser = PlaintextParser.from_string(
            "This is a test. Rln bundles an offline summarizer. "
            "You do not need an internet connection to use it.",
            tok)
        _ = list(s(parser.document, 1))
        console.print("[green]sumy backend OK — extractive summarization available offline.[/green]")
        console.print(f"[dim]Tokenizer: {type(tok).__name__}   Stemmer: {type(stem).__name__}   "
                      f"Stopwords: {len(stop)} words[/dim]")
    except Exception as e:
        console.print(f"[red]sumy backend failed: {e}[/red]")
