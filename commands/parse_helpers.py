"""
Parsing helpers for command lines.
Separated from parser.py to avoid circular imports.
"""

import re
from typing import Optional, List


def _find_keyword_outside_quotes(text: str, keyword: str) -> int:
    """Return the start index of the LAST occurrence of `keyword` (as a
    standalone word) in `text` that is NOT inside a quoted string,
    nor inside parentheses or brackets. Returns -1 if not found.

    "Last" is the right semantic for `if` and `in`: the rightmost
    keyword is the boundary between the body of the command and the
    if/in clause. Anything to the left of that boundary may be an
    expression containing `if` or `in` inside a string literal, which
    we must not split.

    A "standalone" keyword is one bordered on both sides by whitespace
    or a string boundary, and matched case-insensitively.
    """
    # Walk left-to-right tracking quote and bracket state, recording
    # every position where `keyword` appears outside of those.
    candidates = []
    i, n = 0, len(text)
    in_quote = None
    paren_depth = 0
    bracket_depth = 0
    klen = len(keyword)
    while i < n:
        ch = text[i]
        if in_quote:
            if ch == in_quote:
                in_quote = None
            elif ch == "\\" and i + 1 < n:
                i += 1   # skip the escaped char
        else:
            if ch in ("'", '"'):
                in_quote = ch
            elif ch == "(":
                paren_depth += 1
            elif ch == ")":
                paren_depth = max(0, paren_depth - 1)
            elif ch == "[":
                bracket_depth += 1
            elif ch == "]":
                bracket_depth = max(0, bracket_depth - 1)
            elif paren_depth == 0 and bracket_depth == 0:
                # Check for word-boundary match of `keyword`
                if (text[i:i+klen].lower() == keyword.lower()
                        and (i == 0 or not (text[i-1].isalnum() or text[i-1] == "_"))
                        and (i + klen >= n or not
                             (text[i+klen].isalnum() or text[i+klen] == "_"))):
                    candidates.append(i)
        i += 1
    return candidates[-1] if candidates else -1


def parse_command_line(rest: str) -> dict:
    """
    Parse a command line into components:
      {
        'varlist': [...],
        'expression': '...',
        'if_cond': '...',
        'in_range': '...',
        'using': '...',
        'options': {...},
        'raw': '...',
      }
    """
    result = {
        "varlist": [],
        "expression": None,
        "if_cond": None,
        "in_range": None,
        "using": None,
        "options": {},
        "weight": None,       # {"type": "fweight|aweight|pweight|iweight", "var": "weight_col"}
        "raw": rest,
    }

    if not rest.strip():
        return result

    working = rest.strip()

    # Extract weight clause: [fweight=expr] / [aweight=expr] / [pweight=expr] / [iweight=expr]
    # Must be stripped BEFORE the options-comma parse because a weight clause
    # commonly sits after the comma: `tab x [fweight=w], missing`.
    #
    # The expression can be a bare variable (`pop`), an arithmetic combination
    # (`w1 + w2`), or a function call (`round(pop_frac)`, `sqrt(x)`). We scan
    # for the opening `[`, then advance to the matching `]` while tracking
    # paren depth so the closing bracket of a nested `f(...)` doesn't fool us.
    weight_header = re.search(
        r'\[\s*(fweight|aweight|pweight|iweight|weight)\s*=\s*',
        working, re.IGNORECASE,
    )
    if weight_header:
        wtype = weight_header.group(1).lower()
        if wtype == "weight":
            wtype = "fweight"

        # Walk forward from the end of the header to find the balanced ']'
        depth = 0
        i = weight_header.end()
        close_pos = -1
        while i < len(working):
            ch = working[i]
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
            elif ch == "]" and depth == 0:
                close_pos = i
                break
            i += 1

        if close_pos != -1:
            expr = working[weight_header.end():close_pos].strip()
            is_bare_var = bool(re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]*', expr))
            result["weight"] = {
                "type":   wtype,
                "var":    expr if is_bare_var else None,
                "expr":   None if is_bare_var else expr,
                "is_expr": not is_bare_var,
            }
            working = (working[:weight_header.start()] + " "
                       + working[close_pos + 1:]).strip()

    # Extract options after comma (but be careful with commas in expressions)
    comma_pos = _find_options_comma(working)
    if comma_pos is not None:
        options_str = working[comma_pos + 1:].strip()
        working = working[:comma_pos].strip()
        result["options"] = _parse_options(options_str)

    # Extract 'using "file"' or 'using file'
    using_match = re.search(r'\busing\s+(".*?"|\'.*?\'|\S+)', working, re.IGNORECASE)
    if using_match:
        using_file = using_match.group(1).strip("\"'")
        result["using"] = using_file
        working = working[:using_match.start()].strip()

    # Extract 'in range' — must respect quotes/brackets so that, e.g.,
    # `gen s = "row 5 in 10"` doesn't get split at the literal "in".
    in_pos = _find_keyword_outside_quotes(working, "in")
    if in_pos != -1:
        # The portion after `in ` should look like a row range
        # (1, 1/10, 1/L, etc.). If it doesn't, ignore the match.
        tail = working[in_pos + 3:].strip()
        m_tail = re.match(r'^(\d+(?:\s*/\s*(?:\d+|[lL]))?)\s*$', tail)
        if m_tail:
            result["in_range"] = m_tail.group(1)
            working = working[:in_pos].strip()

    # Extract 'if condition' — same quote-aware logic so that
    # `gen s = "Check if True"` doesn't break.
    if_pos = _find_keyword_outside_quotes(working, "if")
    if if_pos != -1:
        result["if_cond"] = working[if_pos + 3:].strip()
        working = working[:if_pos].strip()

    # Check for assignment: var = expression
    eq_match = re.match(r'(\w+)\s*=\s*(.+)', working)
    if eq_match:
        result["varlist"] = [eq_match.group(1)]
        result["expression"] = eq_match.group(2).strip()
    else:
        # Just a varlist
        if working:
            result["varlist"] = _tokenize_varlist(working)

    return result


def _find_options_comma(text: str) -> Optional[int]:
    """Find the position of the options comma (top-level, not in quotes/parens)."""
    depth = 0
    in_quote = False
    quote_char = None

    for i, ch in enumerate(text):
        if in_quote:
            if ch == quote_char:
                in_quote = False
            continue
        if ch in ('"', "'"):
            in_quote = True
            quote_char = ch
            continue
        if ch == '(':
            depth += 1
        elif ch == ')':
            depth -= 1
        elif ch == ',' and depth == 0:
            return i
    return None


def _parse_options(opts_str: str) -> dict:
    """Parse command options: option1(value) option2 option3("text")
    Also handles: option (value) with space before paren."""
    options = {}
    if not opts_str.strip():
        return options

    # Normalize: collapse spaces before ( so "to (de)" becomes "to(de)"
    normalized = re.sub(r'(\w)\s+\(', r'\1(', opts_str)

    pattern = r'(\w+)\s*\(\s*((?:[^()]*|\([^()]*\))*)\s*\)|(\w+)'
    for m in re.finditer(pattern, normalized):
        if m.group(1):
            key = m.group(1).lower()
            val = m.group(2).strip().strip("\"'")
            options[key] = val
        elif m.group(3):
            options[m.group(3).lower()] = True

    return options


def _tokenize_varlist(text: str) -> List[str]:
    """Tokenize a variable list, handling quoted strings."""
    tokens = []
    current = ""
    in_quote = False
    quote_char = None

    for ch in text:
        if in_quote:
            if ch == quote_char:
                in_quote = False
                tokens.append(current)
                current = ""
            else:
                current += ch
        elif ch in ('"', "'"):
            in_quote = True
            quote_char = ch
            current = ""
        elif ch == " ":
            if current:
                tokens.append(current)
                current = ""
        else:
            current += ch

    if current:
        tokens.append(current)

    return tokens
