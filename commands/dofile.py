"""
Do-file preprocessor: parses do-file batch scripts.
Handles comments, line continuation, and blank lines.
"""


def preprocess_dofile(lines: list) -> list:
    """
    Pre-process do-file lines:
    - Remove // line comments
    - Remove * line comments (only at start of line)
    - Remove /* block comments */
    - Handle /// line continuation

    Preserves leading whitespace on every non-blank line so that Python code
    inside `python { ... }` blocks retains its indentation (Gemini Bug 13).
    Other callers that want unindented text can call `.strip()` themselves.
    """
    commands = []
    current = ""
    in_block_comment = False

    for line in lines:
        raw = line.rstrip("\n\r")

        if in_block_comment:
            end_pos = raw.find("*/")
            if end_pos >= 0:
                in_block_comment = False
                raw = raw[end_pos + 2:]
            else:
                continue

        while "/*" in raw:
            start = raw.find("/*")
            end = raw.find("*/", start + 2)
            if end >= 0:
                raw = raw[:start] + raw[end + 2:]
            else:
                raw = raw[:start]
                in_block_comment = True
                break

        stripped = raw.lstrip()
        # Line-level `*` comment: the asterisk must be the first non-whitespace
        # char on a fresh logical line (empty `current`). Otherwise `*` inside
        # an ongoing /// continuation could be misread.
        if stripped.startswith("*") and not current:
            continue

        comment_pos = find_line_comment(raw)
        if comment_pos >= 0:
            raw = raw[:comment_pos]

        raw = raw.rstrip()

        if raw.endswith("///"):
            # Continuation: join with a space, keep leading indent of first part
            current += raw[:-3].rstrip() + " "
            continue

        current += raw

        if current.strip():
            # Preserve leading whitespace (only rstrip was already applied).
            commands.append(current)
        current = ""

    if current.strip():
        commands.append(current)

    return commands


def find_line_comment(line: str) -> int:
    """Find // comment position, ignoring // inside quotes and /// continuation."""
    in_quote = False
    quote_char = None
    i = 0
    while i < len(line) - 1:
        ch = line[i]
        if in_quote:
            if ch == quote_char:
                in_quote = False
        elif ch in ('"', "'"):
            in_quote = True
            quote_char = ch
        elif line[i:i+2] == "//":
            # Check if this is /// (line continuation, not a comment)
            if i + 2 < len(line) and line[i+2] == "/":
                i += 3
                continue
            return i
        i += 1
    return -1
