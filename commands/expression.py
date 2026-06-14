"""
Expression evaluator: translates econometric expressions to pandas operations.

Handles:
  - Arithmetic: +, -, *, /, ^
  - Comparison: ==, !=, >, <, >=, <=
  - Logical: &, |, !
  - String functions: substr(), strlen(), length(), upper(), lower(), trim(),
                      ltrim(), rtrim(), word(), strmatch(), strpos(),
                      subinstr(), real(), string()
  - Regex functions:  regexm(s, "pat"), regexr(s, "pat", "repl"),
                      regexs(s, "pat", n) — submatch group n (1-based)
  - Set membership:   inlist(var, v1, v2, ...),  inrange(expr, lo, hi)
  - Numeric functions: abs(), ceil(), floor(), round(), ln(), log(), log10(),
                       exp(), sqrt(), min(), max(), mod(), int()
  - Conditional:      cond(condition, true_val, false_val)
  - Special:          _n (row number), _N (total rows), missing(var)
  - String literals:  "text" or `"text"'  (compact)
  - Variable references: just the variable name
"""

import re
import numpy as np
import pandas as pd
from typing import Optional


class ExpressionError(Exception):
    pass


def _sanitize_varname(name: str) -> str:
    """Make sure variable name is valid for pandas."""
    return name.strip()


def eval_expression(expr: str, df: pd.DataFrame, by_vars: Optional[list] = None,
                    state=None) -> pd.Series:
    """
    Evaluate a econometric expression against a DataFrame.
    Returns a pandas Series with the result.

    If `by_vars` is given, `_n` and `_N` are computed per group (required
    for correct `bysort group: gen seq = _n` semantics).

    If `state` is given, `_rc` resolves to the last command's return code
    (Gemini Bug 10 fix). Without `state`, referencing `_rc` will raise
    NameError as before.
    """
    translated = translate_expression(expr, df)
    try:
        namespace = _build_namespace(df, by_vars=by_vars, state=state)
        result = eval(translated, {"__builtins__": {}}, namespace)
        if isinstance(result, (int, float, str, bool, np.integer, np.floating)):
            return pd.Series([result] * len(df), index=df.index)
        return result
    except Exception as e:
        raise ExpressionError(f"Cannot evaluate expression '{expr}': {e}")


def eval_condition(cond: str, df: pd.DataFrame, by_vars: Optional[list] = None,
                   state=None) -> pd.Series:
    """
    Evaluate a if-condition. Returns boolean Series.
    """
    if not cond.strip():
        return pd.Series([True] * len(df), index=df.index, dtype=bool)

    translated = translate_expression(cond, df)
    try:
        namespace = _build_namespace(df, by_vars=by_vars, state=state)
        result = eval(translated, {"__builtins__": {}}, namespace)
        if isinstance(result, bool):
            return pd.Series([result] * len(df), index=df.index, dtype=bool)
        return result.astype(bool)
    except Exception as e:
        raise ExpressionError(f"Cannot evaluate condition '{cond}': {e}")


def translate_expression(expr: str, df: pd.DataFrame) -> str:
    """
    Translate expression syntax to Python/pandas syntax.
    """
    result = expr.strip()

    # Remove alternate string delimiters: `"text"' -> "text"
    result = re.sub(r'`"(.*?)"\'', r'"\1"', result)

    # Replace ^ with ** for exponentiation — but NOT inside string literals,
    # because regex patterns use ^ as the start-of-string anchor.
    result = _replace_outside_strings(result, "^", "**")

    # Replace the != with Python's !=  (other statistical tools also uses ~=)
    result = result.replace("~=", "!=")

    # Logical operators: need to parenthesize sub-expressions around & and |
    # because in Python, & binds tighter than comparison operators.
    # Split on & and |, wrap each part in parens, rejoin.
    result = _parenthesize_logical(result)

    # Handle ! as not (but not != )
    result = re.sub(r'(?<!=)!(?!=)', ' _NOT_ ', result)

    # --- Function translations ---

    # _n -> row number (1-based)
    result = re.sub(r'\b_n\b', '_ROW_N_', result)
    # _N -> total rows
    result = re.sub(r'\b_N\b', '_TOTAL_N_', result)

    # missing(var) -> var.isna()
    result = re.sub(
        r'\bmissing\(\s*(\w+)\s*\)',
        r'_COL_\1.isna()',
        result
    )

    # !missing(var) -> var.notna()
    result = re.sub(
        r'_NOT_\s*_COL_(\w+)\.isna\(\)',
        r'_COL_\1.notna()',
        result
    )

    # inrange(expr, lo, hi) -> ((expr >= lo) & (expr <= hi))
    # inlist(expr, v1, v2, ...) -> expr.isin([v1, v2, ...])
    # regexm(s, "pat") -> s.str.contains("pat", regex=True)
    # regexr(s, "pat", "repl") -> s.str.replace("pat", "repl", regex=True)
    # regexs(s, "pat", n) -> s.str.extract("(pat)").iloc[:,0] for group n (1-based)
    # All of these need full argument-respecting parsing because their args
    # can themselves contain commas (inside parens / quotes).
    result = _translate_ext_calls(result)

    # String functions
    result = re.sub(r'\bsubstr\(\s*(\w+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\)',
                    r'_COL_\1.str[\2-1:\2-1+\3]', result)
    result = re.sub(r'\bstrlen\(\s*(\w+)\s*\)', r'_COL_\1.str.len()', result)
    # length() is a common alias for strlen() on string columns
    result = re.sub(r'\blength\(\s*(\w+)\s*\)', r'_COL_\1.str.len()', result)
    # real("string" or string_col) -> numeric conversion
    result = re.sub(r'\breal\(\s*(\w+)\s*\)',
                    r'_PD_.to_numeric(_COL_\1, errors="coerce")', result)
    # string(numeric_col) -> string conversion
    result = re.sub(r'\bstring\(\s*(\w+)\s*\)', r'_COL_\1.astype("string")', result)
    result = re.sub(r'\bupper\(\s*(\w+)\s*\)', r'_COL_\1.str.upper()', result)
    result = re.sub(r'\blower\(\s*(\w+)\s*\)', r'_COL_\1.str.lower()', result)
    result = re.sub(r'\btrim\(\s*(\w+)\s*\)', r'_COL_\1.str.strip()', result)
    result = re.sub(r'\bltrim\(\s*(\w+)\s*\)', r'_COL_\1.str.lstrip()', result)
    result = re.sub(r'\brtrim\(\s*(\w+)\s*\)', r'_COL_\1.str.rstrip()', result)
    result = re.sub(r'\bword\(\s*(\w+)\s*,\s*(\d+)\s*\)',
                    r'_COL_\1.str.split().str[\2-1]', result)
    result = re.sub(r'\bstrmatch\(\s*(\w+)\s*,\s*"([^"]+)"\s*\)',
                    r'_COL_\1.str.match(r"\2")', result)
    result = re.sub(r'\bstrpos\(\s*(\w+)\s*,\s*"([^"]+)"\s*\)',
                    r'(_COL_\1.str.find("\2") + 1)', result)
    result = re.sub(r'\bsubinstr\(\s*(\w+)\s*,\s*"([^"]+)"\s*,\s*"([^"]*)"\s*,\s*\.\s*\)',
                    r'_COL_\1.str.replace("\2", "\3", regex=False)', result)

    # Numeric functions — use negative lookbehind to prevent re-matching _NP_.log etc.
    result = re.sub(r'(?<!\.)(?<!\w)abs\(\s*', r'_NP_.abs(', result)
    result = re.sub(r'(?<!\.)(?<!\w)ceil\(\s*', r'_NP_.ceil(', result)
    result = re.sub(r'(?<!\.)(?<!\w)floor\(\s*', r'_NP_.floor(', result)
    result = re.sub(r'(?<!\.)(?<!\w)round\(\s*', r'_NP_.round(', result)
    result = re.sub(r'(?<!\.)(?<!\w)ln\(\s*', r'_NP_.log(', result)
    result = re.sub(r'(?<!\.)(?<!\w)log\(\s*', r'_NP_.log(', result)
    result = re.sub(r'(?<!\.)(?<!\w)log10\(\s*', r'_NP_.log10(', result)
    result = re.sub(r'(?<!\.)(?<!\w)exp\(\s*', r'_NP_.exp(', result)
    result = re.sub(r'(?<!\.)(?<!\w)sqrt\(\s*', r'_NP_.sqrt(', result)
    result = re.sub(r'(?<!\.)(?<!\w)mod\(\s*([^,]+)\s*,\s*([^)]+)\s*\)', r'_NP_.mod(\1, \2)', result)
    result = re.sub(r'(?<!\.)(?<!\w)min\(\s*', r'_NP_.minimum(', result)
    result = re.sub(r'(?<!\.)(?<!\w)max\(\s*', r'_NP_.maximum(', result)
    result = re.sub(r'(?<!\.)(?<!\w)int\(\s*', r'_NP_.trunc(', result)
    result = re.sub(r'(?<!\.)(?<!\w)trunc\(\s*', r'_NP_.trunc(', result)

    # cond() function: cond(condition, true_val, false_val)
    result = re.sub(
        r'\bcond\(\s*([^,]+)\s*,\s*([^,]+)\s*,\s*([^)]+)\s*\)',
        r'_NP_.where(\1, \2, \3)',
        result
    )

    # Now replace _COL_varname with actual column references
    # First handle explicitly tagged ones
    result = re.sub(r'_COL_(\w+)', r'_DF_["\1"]', result)

    # Then handle remaining bare variable names that match DataFrame columns
    col_names = set(df.columns)
    tokens = re.split(r'(\W+)', result)
    for i, token in enumerate(tokens):
        if (token in col_names
                and not token.startswith('"')
                and token not in ('_NP_', '_DF_', '_ROW_N_', '_TOTAL_N_',
                                  'True', 'False')):
            # Check it's not already wrapped
            before = tokens[i-1] if i > 0 else ""
            if not before.endswith('"') and not before.endswith('['):
                tokens[i] = f'_DF_["{token}"]'
    result = "".join(tokens)

    # Restore logical operators
    result = result.replace("_AND_", "&")
    result = result.replace("_OR_", "|")
    result = result.replace("_NOT_", "~")

    return result


def _build_namespace(df: pd.DataFrame, by_vars: Optional[list] = None,
                     state=None) -> dict:
    """Build safe evaluation namespace.

    When `by_vars` is given, `_n` becomes a per-group 1-based row counter
    and `_N` becomes a per-group row count. This honors the standard
    convention's semantics under a `by varlist:` prefix. Outside such a
    context both are global, as before.

    When `state` is given, `_rc` resolves to the last return code (prefers
    a captured rc over the parser-level rc) so `assert _rc != 0` works
    after `capture`.
    """
    if by_vars and all(v in df.columns for v in by_vars):
        row_n = df.groupby(by_vars).cumcount() + 1
        # Group size aligned back to every row:
        total_n = df.groupby(by_vars)[by_vars[0]].transform("size")
        row_n.index = df.index
        total_n.index = df.index
    else:
        row_n   = pd.Series(range(1, len(df) + 1), index=df.index)
        total_n = len(df)

    ns = {
        "_DF_": df,
        "_NP_": np,
        "_PD_": pd,
        "_ROW_N_": row_n,
        "_TOTAL_N_": total_n,
        "True": True,
        "False": False,
    }
    if state is not None:
        ns["_rc"] = int(
            getattr(state, "_captured_rc", None)
            or getattr(state, "_rc", 0)
            or 0
        )
    return ns


def parse_in_range(in_clause: str) -> Optional[slice]:
    """
    Parse 'in' range specification.
    Examples: 'in 1/10', 'in 5/20', 'in 1/l' (l = last)
    Returns a slice object.
    """
    in_clause = in_clause.strip()
    match = re.match(r'(\d+)\s*/\s*(\d+|[lL])', in_clause)
    if match:
        start = int(match.group(1)) - 1  # input is 1-based
        end_str = match.group(2)
        if end_str.lower() == 'l':
            end = None
        else:
            end = int(end_str)
        return slice(start, end)
    # Single number
    match = re.match(r'(\d+)', in_clause)
    if match:
        idx = int(match.group(1)) - 1
        return slice(idx, idx + 1)
    return None


def _parenthesize_logical(expr: str) -> str:
    """
    Wrap sub-expressions around & and | in parentheses so that
    pandas bitwise operators get correct precedence.
    e.g. 'age >= 30 & income > 40000' -> '(age >= 30) & (income > 40000)'
    """
    # Don't process if no logical operators
    if ' & ' not in expr and ' | ' not in expr:
        return expr

    # Split on top-level & and | (not inside parens/quotes)
    parts = []
    operators = []
    depth = 0
    in_quote = False
    quote_char = None
    current = ""

    i = 0
    while i < len(expr):
        ch = expr[i]

        if in_quote:
            current += ch
            if ch == quote_char:
                in_quote = False
            i += 1
            continue

        if ch in ('"', "'"):
            in_quote = True
            quote_char = ch
            current += ch
            i += 1
            continue

        if ch == '(':
            depth += 1
            current += ch
            i += 1
            continue
        elif ch == ')':
            depth -= 1
            current += ch
            i += 1
            continue

        if depth == 0:
            # Check for ' & ' or ' | '
            if expr[i:i+3] == ' & ':
                parts.append(current.strip())
                operators.append('&')
                current = ""
                i += 3
                continue
            elif expr[i:i+3] == ' | ':
                parts.append(current.strip())
                operators.append('|')
                current = ""
                i += 3
                continue

        current += ch
        i += 1

    parts.append(current.strip())

    # Wrap each part in parens if it contains a comparison
    wrapped = []
    for p in parts:
        p = p.strip()
        if p and not p.startswith('('):
            wrapped.append(f'({p})')
        else:
            wrapped.append(p)

    # Rejoin
    result = wrapped[0]
    for op, part in zip(operators, wrapped[1:]):
        result += f' {op} {part}'

    return result


# ══════════════════════════════════════════════════════════════════════
# Extension function translator — handles inlist / inrange / regex*
# properly by splitting arguments with paren & quote awareness.
# ══════════════════════════════════════════════════════════════════════

_EXT_CALL_RE = re.compile(
    r'\b(inlist|inrange|regexm|regexr|regexs)\s*\(',
    re.IGNORECASE,
)


def _split_top_level_args(s: str) -> list:
    """Split a comma-separated argument list at the top level, respecting
    quotes and nested parentheses. Returns a list of arg strings (trimmed)."""
    parts = []
    depth = 0
    in_quote = False
    quote_char = None
    current = []
    for ch in s:
        if in_quote:
            current.append(ch)
            if ch == quote_char:
                in_quote = False
            continue
        if ch in ('"', "'"):
            in_quote = True
            quote_char = ch
            current.append(ch)
            continue
        if ch == '(':
            depth += 1
            current.append(ch)
            continue
        if ch == ')':
            depth -= 1
            current.append(ch)
            continue
        if ch == ',' and depth == 0:
            parts.append("".join(current).strip())
            current = []
            continue
        current.append(ch)
    if current:
        parts.append("".join(current).strip())
    return parts


def _find_matching_paren(s: str, open_idx: int) -> int:
    """Given the index of an open paren, return the index of its match,
    honoring quotes and nested parens. Returns -1 if unbalanced."""
    depth = 0
    in_quote = False
    quote_char = None
    i = open_idx
    while i < len(s):
        ch = s[i]
        if in_quote:
            if ch == quote_char:
                in_quote = False
        elif ch in ('"', "'"):
            in_quote = True
            quote_char = ch
        elif ch == '(':
            depth += 1
        elif ch == ')':
            depth -= 1
            if depth == 0:
                return i
        i += 1
    return -1


def _strip_str_quotes(s: str) -> str:
    s = s.strip()
    if len(s) >= 2 and s[0] == s[-1] and s[0] in ('"', "'"):
        return s[1:-1]
    return s


def _translate_ext_calls(expr: str) -> str:
    """Replace every top-level inlist/inrange/regexm/regexr/regexs call
    with its Pandas-native equivalent, parsing arguments robustly.

    Operates iteratively so that nested calls are also rewritten
    (outer-first; a second pass picks up any newly-exposed calls that
    might have been created as a result of inlining)."""
    prev = None
    out = expr
    # A few iterations are enough for realistic nesting depth.
    for _ in range(5):
        if out == prev:
            break
        prev = out
        out = _translate_ext_calls_once(out)
    return out


def _translate_ext_calls_once(expr: str) -> str:
    result = []
    i = 0
    while i < len(expr):
        m = _EXT_CALL_RE.search(expr, i)
        if not m:
            result.append(expr[i:])
            break
        # Copy the untouched prefix
        result.append(expr[i:m.start()])
        fname = m.group(1).lower()
        open_paren = m.end() - 1  # the '(' we matched
        close_paren = _find_matching_paren(expr, open_paren)
        if close_paren == -1:
            # Unbalanced; bail out and copy the rest verbatim
            result.append(expr[m.start():])
            break
        args_text = expr[open_paren + 1:close_paren]
        args = _split_top_level_args(args_text)
        rewritten = _render_ext_call(fname, args)
        result.append(rewritten)
        i = close_paren + 1
    return "".join(result)


def _render_ext_call(fname: str, args: list) -> str:
    """Render a extension call as equivalent pandas expression text.

    We re-emit the arguments verbatim — they are still input syntax at
    this point and will be processed by the outer translator's later
    passes (numeric functions, column-name substitution, etc.).
    """
    if fname == "inrange":
        if len(args) != 3:
            raise ExpressionError(
                f"inrange() expects 3 arguments (value, lo, hi), got {len(args)}")
        v, lo, hi = args
        return f"(({v} >= {lo}) & ({v} <= {hi}))"

    if fname == "inlist":
        if len(args) < 2:
            raise ExpressionError(
                f"inlist() expects at least 2 arguments (value, v1, ...), got {len(args)}")
        v = args[0]
        values = args[1:]
        # Render the value list Python-style; leave the atoms as-is so the
        # outer translator can process column refs, string literals, etc.
        # If any value is a bare quoted literal ("x" or 'x') we leave
        # it untouched — Python accepts the same syntax.
        rendered = "[" + ", ".join(values) + "]"
        return f"({v}).isin({rendered})"

    if fname == "regexm":
        if len(args) != 2:
            raise ExpressionError(
                f"regexm() expects 2 arguments (string, pattern), got {len(args)}")
        s, pat = args
        pat_lit = _strip_str_quotes(pat)
        pat_py = pat_lit.replace("\\", "\\\\").replace('"', '\\"')
        return f'({s}).astype("string").str.contains("{pat_py}", regex=True, na=False)'

    if fname == "regexr":
        if len(args) != 3:
            raise ExpressionError(
                f"regexr() expects 3 arguments (string, pattern, replacement), got {len(args)}")
        s, pat, repl = args
        pat_py = _strip_str_quotes(pat).replace("\\", "\\\\").replace('"', '\\"')
        rep_py = _strip_str_quotes(repl).replace("\\", "\\\\").replace('"', '\\"')
        return f'({s}).astype("string").str.replace("{pat_py}", "{rep_py}", regex=True)'

    if fname == "regexs":
        # Rln-specific signature: regexs(string, pattern, group_n)
        #   Group numbering is 1-based as documented here.
        # (some alternative regexs(n) is stateful and unsuitable for vectorized use.)
        if len(args) != 3:
            raise ExpressionError(
                "regexs() in Rln takes 3 arguments: regexs(string, \"pattern\", group_n). "
                "Group numbering is 1-based."
            )
        s, pat, n_str = args
        try:
            n = int(n_str.strip())
        except ValueError:
            raise ExpressionError(f"regexs group index must be a literal integer, got {n_str!r}")
        if n < 1:
            raise ExpressionError("regexs group index must be >= 1")
        pat_py = _strip_str_quotes(pat).replace("\\", "\\\\").replace('"', '\\"')
        # pandas .str.extract uses a single capture-group regex and returns
        # a DataFrame (one col per group). We index the nth group (0-based in pandas).
        return (f'({s}).astype("string").str.extract("{pat_py}", expand=True).iloc[:, {n - 1}]')

    # Should not reach here
    raise ExpressionError(f"Internal: unknown extension call {fname!r}")


def _replace_outside_strings(text: str, needle: str, replacement: str) -> str:
    """Replace `needle` with `replacement` everywhere in `text` EXCEPT inside
    quoted string literals. Used to translate the `^` (exponentiation)
    into Python's `**` without corrupting regex patterns like "^abc$"."""
    out = []
    i = 0
    in_quote = False
    quote_char = None
    nlen = len(needle)
    while i < len(text):
        ch = text[i]
        if in_quote:
            out.append(ch)
            if ch == quote_char and (i == 0 or text[i - 1] != "\\"):
                in_quote = False
            i += 1
            continue
        if ch in ('"', "'"):
            in_quote = True
            quote_char = ch
            out.append(ch)
            i += 1
            continue
        if text[i:i + nlen] == needle:
            out.append(replacement)
            i += nlen
            continue
        out.append(ch)
        i += 1
    return "".join(out)
