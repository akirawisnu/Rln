"""Backend-agnostic fuzzy string matching for `fuzzmerge` / `lrtm fuzzmerge`.

Why this exists: the original `fuzzmerge` used PolyFuzz directly, but PolyFuzz
(and its TF-IDF model's scikit-learn dependency) isn't bundled in the lite
desktop build and isn't available on Android at all; `rapidfuzz` is present on
lite but also missing on Android. To make fuzzy matching work on **every**
version, this picks the best backend available and always has a pure-Python
floor:

    1. PolyFuzz  (TF-IDF / EditDistance)  — best quality, desktop full/source
    2. rapidfuzz (token_sort WRatio)      — fast C++, desktop lite
    3. difflib   (SequenceMatcher)        — pure stdlib, ALWAYS works (Android)

All backends return the same shape so callers don't care which ran.
"""

from __future__ import annotations

from typing import List, Tuple


def _match_polyfuzz(master, using, method):
    from polyfuzz import PolyFuzz  # may raise ImportError
    model = PolyFuzz("EditDistance" if method == "editdistance" else "TF-IDF")
    model.match(master, using)
    mdf = model.get_matches()
    out = []
    for frm, to, sim in zip(mdf["From"], mdf["To"], mdf["Similarity"]):
        if to is None:
            continue
        out.append((str(frm), str(to), float(sim)))
    return out


def _match_rapidfuzz(master, using):
    from rapidfuzz import process, fuzz  # may raise ImportError
    out = []
    for s in master:
        res = process.extractOne(s, using, scorer=fuzz.WRatio)
        if res is not None:
            to, score, _ = res
            out.append((str(s), str(to), float(score) / 100.0))
    return out


def _match_difflib(master, using):
    import difflib
    out = []
    for s in master:
        # get_close_matches returns the best candidates by SequenceMatcher ratio.
        best = difflib.get_close_matches(s, using, n=1, cutoff=0.0)
        if best:
            to = best[0]
            sim = difflib.SequenceMatcher(None, s, to).ratio()
            out.append((str(s), str(to), float(sim)))
    return out


def fuzzy_match(master, using, threshold: float = 0.8,
                method: str = "auto") -> Tuple[List[Tuple[str, str, float]], str]:
    """Best one-to-one match for each string in ``master`` against ``using``.

    Returns ``(matches, backend)`` where ``matches`` is a list of
    ``(from, to, similarity_0_to_1)`` filtered to ``similarity >= threshold``,
    and ``backend`` names the engine actually used (for reporting).

    ``method`` may be ``tfidf`` / ``editdistance`` (hints for PolyFuzz) or
    ``auto``. The function degrades automatically: PolyFuzz → rapidfuzz →
    difflib, so it never raises ImportError — fuzzy matching is available on
    every Rln build.
    """
    master = [str(x) for x in master]
    using = [str(x) for x in using]

    matches: List[Tuple[str, str, float]] = []
    backend = ""
    # PolyFuzz first (unless the caller explicitly wants the lightweight path).
    if method in ("auto", "tfidf", "editdistance"):
        try:
            matches = _match_polyfuzz(master, using, method)
            backend = "polyfuzz"
        except Exception:
            matches = []
    if not backend:
        try:
            matches = _match_rapidfuzz(master, using)
            backend = "rapidfuzz"
        except Exception:
            matches = []
    if not backend:
        matches = _match_difflib(master, using)
        backend = "difflib"

    filtered = [(f, t, s) for (f, t, s) in matches if s >= threshold]
    return filtered, backend
