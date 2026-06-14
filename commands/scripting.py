"""
Script control: foreach, forvalues, local macros, quietly, by/bysort prefix.
These are the features that make do-files feel like real other statistical tools.
"""

import re
import pandas as pd
from rich.console import Console
from io import StringIO

from commands.state import AppState
from commands.parse_helpers import parse_command_line
from commands.expression import eval_condition


# ──────────────────────────────────────────────
#  Local macros
# ──────────────────────────────────────────────

def cmd_local(rest: str, state: AppState, console: Console):
    """
    local macname = expression
    local macname "string"
    local macname value1 value2 value3
    
    Define a local macro. Access it later with `macname' syntax.
    """
    _set_macro(rest, state, console, macro_type="local")


def cmd_global(rest: str, state: AppState, console: Console):
    """
    global macname = expression
    global macname "string"
    
    Define a global macro. Access with $macname or ${macname} syntax.
    Persists across do-files (unlike local which is scoped).
    """
    _set_macro(rest, state, console, macro_type="global")


def _set_macro(rest: str, state: AppState, console: Console, macro_type="local"):
    """Set a local or global macro."""
    macros = state.local_macros if macro_type == "local" else state.global_macros
    rest = rest.strip()
    if not rest:
        # Show all macros
        if not macros:
            console.print(f"[dim]No {macro_type} macros defined.[/dim]")
        else:
            for k, v in macros.items():
                console.print(f"  [cyan]{k}[/cyan] = {v}")
        return

    # Parse: macname = expr  OR  macname "string"  OR  macname value
    m = re.match(r'(\w+)\s*=\s*(.+)', rest)
    if m:
        name = m.group(1)
        expr = m.group(2).strip()
        # Try to evaluate as expression
        if state.has_data():
            try:
                from commands.expression import eval_expression
                result = eval_expression(expr, state.data)
                if hasattr(result, 'iloc'):
                    val = str(result.iloc[0])
                else:
                    val = str(result)
            except Exception:
                val = expr
        else:
            # Try as pure math
            try:
                val = str(eval(expr.replace("^", "**"), {"__builtins__": {}}))
            except Exception:
                val = expr
        macros[name] = val
        return

    # macname "string" or macname value
    parts = rest.split(None, 1)
    name = parts[0]
    val = parts[1].strip().strip("\"'") if len(parts) > 1 else ""
    macros[name] = val


def expand_macros(text: str, state: AppState) -> str:
    """
    Expand macros in text:
      Local:  `macname'  -> value
      Global: $macname   -> value
      Both:   ${macname} -> value
    """
    result = text

    # Global macros FIRST (so $name in locals doesn't collide)
    # Use simple string replace — NO regex (Windows paths have backslashes)
    if state.global_macros:
        for name, val in state.global_macros.items():
            result = result.replace(f"${{{name}}}", str(val))
            result = result.replace(f"${name}", str(val))

    # Local macros: `macname'
    if state.local_macros:
        for name, val in state.local_macros.items():
            result = result.replace(f"`{name}'", str(val))
            result = result.replace(f"${{{name}}}", str(val))

    return result


# ──────────────────────────────────────────────
#  foreach / forvalues
# ──────────────────────────────────────────────

def cmd_foreach(rest: str, state: AppState, console: Console):
    """
    foreach macname in list {
        commands
    }
    foreach macname of varlist varlist {
        commands
    }
    foreach macname of numlist numlist {
        commands
    }
    
    Loop over a list of values.
    NOTE: In interactive mode, collects lines until closing }.
          In do-files, the preprocessor handles braces.
    """
    # Parse: foreach var in/of list {
    m = re.match(r'(\w+)\s+(in|of\s+\w+)\s+(.+?)(\s*\{)?$', rest.strip())
    if not m:
        console.print("[red]Syntax: foreach macname in list { ... }[/red]")
        return

    mac_name = m.group(1)
    loop_type = m.group(2).strip()
    items_str = m.group(3).strip()

    # Get the loop body
    body_lines = _collect_block(state, console)
    if body_lines is None:
        return

    # Parse items
    if loop_type == "in":
        items = _parse_list(items_str)
    elif loop_type.startswith("of varlist"):
        if state.has_data():
            from commands.explore import _resolve_varlist
            items = _resolve_varlist(items_str.split(), state.data)
        else:
            items = items_str.split()
    elif loop_type.startswith("of numlist"):
        items = _parse_numlist(items_str)
    else:
        items = items_str.split()

    # Execute loop
    from commands.parser import CommandParser
    temp_parser = CommandParser(state, console)

    for item in items:
        state.local_macros[mac_name] = str(item)
        j = 0
        while j < len(body_lines):
            line = body_lines[j]
            expanded = expand_macros(line, state)
            stripped = expanded.strip()

            # Recognize nested python { ... } blocks within the loop body
            # and pre-load them into _pending_block before dispatching the
            # `python {` opener. Without this, the inner braces would be
            # interpreted by the parser as a fresh interactive prompt.
            first_word = stripped.split(None, 1)[0].lower() if stripped else ""
            if first_word in ("python", "py") and stripped.endswith("{"):
                inner = []
                depth = 1
                j += 1
                while j < len(body_lines) and depth > 0:
                    inner_raw = body_lines[j]
                    inner_stripped = inner_raw.strip()
                    j += 1
                    if inner_stripped == "}":
                        depth -= 1
                        if depth == 0:
                            break
                    if inner_stripped.endswith("{"):
                        depth += 1
                    # Expand macros (`var') inside the python block too —
                    # this is what users expect when writing
                    # `python { v = "`v'" }` inside a foreach loop.
                    inner.append(expand_macros(inner_raw, state))
                state._pending_block = inner
                try:
                    temp_parser.execute(stripped)   # `python {`
                except Exception as e:
                    console.print(f"[red]Error in foreach: {e}[/red]")
                continue

            if stripped:
                try:
                    temp_parser.execute(expanded)
                except Exception as e:
                    console.print(f"[red]Error in foreach: {e}[/red]")
            j += 1

    # Clean up loop macro
    state.local_macros.pop(mac_name, None)


def cmd_forvalues(rest: str, state: AppState, console: Console):
    """
    forvalues macname = start/end {
        commands
    }
    forvalues macname = start(step)end {
        commands
    }
    
    Loop over a numeric range.
    """
    m = re.match(r'(\w+)\s*=\s*(.+?)(\s*\{)?$', rest.strip())
    if not m:
        console.print("[red]Syntax: forvalues macname = start/end { ... }[/red]")
        return

    mac_name = m.group(1)
    range_str = m.group(2).strip()

    # Parse range: start/end or start(step)end
    values = _parse_forvalues_range(range_str)
    if values is None:
        console.print(f"[red]Cannot parse range: {range_str}[/red]")
        console.print("[dim]Use: start/end or start(step)end[/dim]")
        return

    body_lines = _collect_block(state, console)
    if body_lines is None:
        return

    from commands.parser import CommandParser
    temp_parser = CommandParser(state, console)

    for val in values:
        state.local_macros[mac_name] = str(val)
        j = 0
        while j < len(body_lines):
            line = body_lines[j]
            expanded = expand_macros(line, state)
            stripped = expanded.strip()

            # Same nested python-block handling as foreach (Minimax v126 B1).
            first_word = stripped.split(None, 1)[0].lower() if stripped else ""
            if first_word in ("python", "py") and stripped.endswith("{"):
                inner = []
                depth = 1
                j += 1
                while j < len(body_lines) and depth > 0:
                    inner_raw = body_lines[j]
                    inner_stripped = inner_raw.strip()
                    j += 1
                    if inner_stripped == "}":
                        depth -= 1
                        if depth == 0:
                            break
                    if inner_stripped.endswith("{"):
                        depth += 1
                    inner.append(expand_macros(inner_raw, state))
                state._pending_block = inner
                try:
                    temp_parser.execute(stripped)
                except Exception as e:
                    console.print(f"[red]Error in forvalues: {e}[/red]")
                continue

            if stripped:
                try:
                    temp_parser.execute(expanded)
                except Exception as e:
                    console.print(f"[red]Error in forvalues: {e}[/red]")
            j += 1

    state.local_macros.pop(mac_name, None)


def _collect_block(state, console):
    """Collect lines until closing }. For interactive or do-file mode."""
    # Check if we're in a do-file (lines pre-collected via _pending_block)
    if hasattr(state, '_pending_block') and state._pending_block is not None:
        lines = state._pending_block
        state._pending_block = None
        return lines

    # Interactive: prompt for lines until closing }
    lines = []
    try:
        depth = 1
        while depth > 0:
            line = input("  > ").rstrip()
            stripped = line.strip()
            if stripped == "}":
                depth -= 1
                if depth == 0:
                    break
            if stripped.endswith("{"):
                depth += 1
            lines.append(line)
    except (EOFError, KeyboardInterrupt):
        console.print("[dim]Block cancelled.[/dim]")
        return None

    return lines


def _parse_list(s):
    """Parse a space-separated list, handling quoted items."""
    items = []
    current = ""
    in_quote = False
    for ch in s:
        if in_quote:
            if ch == '"':
                in_quote = False
                items.append(current)
                current = ""
            else:
                current += ch
        elif ch == '"':
            in_quote = True
        elif ch == ' ':
            if current:
                items.append(current)
                current = ""
        else:
            current += ch
    if current:
        items.append(current)
    return items


def _parse_numlist(s):
    """Parse a other statistical tools numlist: 1 2 3 or 1/5 or 1(2)10."""
    items = []
    for part in s.split():
        if "/" in part:
            lo, hi = part.split("/", 1)
            items.extend(range(int(lo), int(hi) + 1))
        elif "(" in part:
            m = re.match(r'(\d+)\((\d+)\)(\d+)', part)
            if m:
                start, step, end = int(m.group(1)), int(m.group(2)), int(m.group(3))
                items.extend(range(start, end + 1, step))
            else:
                items.append(part)
        else:
            try:
                items.append(int(part))
            except ValueError:
                items.append(part)
    return items


def _parse_forvalues_range(s):
    """Parse forvalues range: start/end or start(step)end."""
    s = s.strip()
    m = re.match(r'(-?\d+)\s*/\s*(-?\d+)', s)
    if m:
        return list(range(int(m.group(1)), int(m.group(2)) + 1))

    m = re.match(r'(-?\d+)\s*\(\s*(-?\d+)\s*\)\s*(-?\d+)', s)
    if m:
        start, step, end = int(m.group(1)), int(m.group(2)), int(m.group(3))
        return list(range(start, end + 1, step))

    return None


# ──────────────────────────────────────────────
#  quietly
# ──────────────────────────────────────────────

def cmd_quietly(rest: str, state: AppState, console: Console):
    """
    quietly command
    Execute a command suppressing all output.
    """
    if not rest.strip():
        console.print("[red]Syntax: quietly command[/red]")
        return

    from commands.parser import CommandParser

    # Redirect to null console
    null_output = StringIO()
    null_console = Console(file=null_output, force_terminal=False)

    temp_parser = CommandParser(state, null_console)
    try:
        temp_parser.execute(rest.strip())
        state.return_code = 0
    except Exception:
        state.return_code = 1


# ──────────────────────────────────────────────
#  by / bysort prefix
# ──────────────────────────────────────────────

def cmd_by(rest: str, state: AppState, console: Console):
    """
    by varlist: command
    bysort varlist: command
    
    Execute a command separately for each group defined by varlist.
    bysort also sorts the data first.
    """
    _do_by(rest, state, console, sort_first=False)


def cmd_bysort(rest: str, state: AppState, console: Console):
    """
    bysort varlist: command
    Sort by varlist, then execute command for each group.
    """
    _do_by(rest, state, console, sort_first=True)


def _do_by(rest: str, state: AppState, console: Console, sort_first: bool):
    """Execute a command by groups."""
    state.require_data()

    # Parse: varlist: command
    if ":" not in rest:
        console.print("[red]Syntax: by varlist: command[/red]")
        return

    by_part, cmd_part = rest.split(":", 1)
    by_vars = by_part.strip().split()
    cmd = cmd_part.strip()

    if not by_vars or not cmd:
        console.print("[red]Syntax: by varlist: command[/red]")
        return

    for v in by_vars:
        if v not in state.data.columns:
            console.print(f"[red]Variable '{v}' not found[/red]")
            return

    # Sort if bysort
    if sort_first:
        state.data = state.data.sort_values(by_vars).reset_index(drop=True)

    # Check if command is one that should aggregate vs. run per-group
    # For generate/replace/egen, run on whole dataset with group context
    cmd_word = cmd.split(None, 1)[0].lower() if cmd else ""

    if cmd_word in ("gen", "generate", "g", "egen", "replace"):
        # Set by-group context for the command
        state._by_vars = by_vars
        from commands.parser import CommandParser
        temp_parser = CommandParser(state, console)
        try:
            temp_parser.execute(cmd)
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")
        finally:
            state._by_vars = None
        return

    # For other commands, execute per group
    from commands.parser import CommandParser
    groups = state.data.groupby(by_vars)
    original_data = state.data

    for group_vals, group_df in groups:
        if isinstance(group_vals, tuple):
            header = ", ".join(f"{v}={g}" for v, g in zip(by_vars, group_vals))
        else:
            header = f"{by_vars[0]}={group_vals}"

        console.print(f"\n[bold cyan]-> {header}[/bold cyan]")

        # Temporarily set data to this group
        state.data = group_df.reset_index(drop=True)
        temp_parser = CommandParser(state, console)
        try:
            temp_parser.execute(cmd)
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")

    # Restore full data
    state.data = original_data


# ──────────────────────────────────────────────
#  return / ereturn (stored results)
# ──────────────────────────────────────────────

def cmd_return(rest: str, state: AppState, console: Console):
    """
    return list
    Display stored r() results from the last command.
    """
    if not state.r_results:
        console.print("[dim]No stored results.[/dim]")
        return

    console.print("\n[bold]Stored results:[/bold]")
    for key, val in state.r_results.items():
        if isinstance(val, float):
            console.print(f"  r({key}) = {val:.6g}")
        else:
            console.print(f"  r({key}) = {val}")
    console.print()


def cmd_ereturn(rest: str, state: AppState, console: Console):
    """
    ereturn list
    Display stored e() results from the last estimation command.
    """
    if not state.e_results:
        console.print("[dim]No estimation results stored.[/dim]")
        return

    console.print("\n[bold]Stored estimation results:[/bold]")
    for key, val in state.e_results.items():
        if isinstance(val, float):
            console.print(f"  e({key}) = {val:.6g}")
        elif isinstance(val, pd.DataFrame):
            console.print(f"  e({key}) = [{val.shape[0]}x{val.shape[1]} matrix]")
        else:
            console.print(f"  e({key}) = {val}")
    console.print()
