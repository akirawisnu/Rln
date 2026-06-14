"""
Variable management commands: generate, replace, rename, drop, keep,
label, destring, tostring, encode
"""

import re
import numpy as np
import pandas as pd
from rich.console import Console

from commands.state import AppState
from commands.parse_helpers import parse_command_line
from commands.expression import eval_expression, eval_condition


def cmd_generate(rest: str, state: AppState, console: Console):
    """
    generate [type] newvar = expression [if condition]
    Create a new variable.
    """
    state.require_data()

    # Handle optional type prefix: gen byte/int/long/float/double/str var = expr
    type_prefix = None
    m = re.match(r'(byte|int|long|float|double|str\d*)\s+', rest)
    if m:
        type_prefix = m.group(1)
        rest = rest[m.end():]

    parsed = parse_command_line(rest)

    if not parsed["varlist"] or not parsed["expression"]:
        console.print("[red]Syntax: generate newvar = expression [if condition][/red]")
        return

    varname = parsed["varlist"][0]

    if varname in state.data.columns:
        console.print(f"[red]Variable '{varname}' already exists. Use 'replace' instead.[/red]")
        return

    # If running under a `bysort group:` prefix, _n and _N must be per-group.
    by_vars = getattr(state, "_by_vars", None)

    # Evaluate expression. Pass `state` so `_rc` is visible in the
    # expression namespace (e.g. `gen err_flag = _rc != 0`).
    result = eval_expression(parsed["expression"], state.data,
                              by_vars=by_vars, state=state)

    # Apply if-condition
    if parsed["if_cond"]:
        mask = eval_condition(parsed["if_cond"], state.data,
                               by_vars=by_vars, state=state)
        new_col = pd.Series(np.nan, index=state.data.index)
        new_col.loc[mask] = result.loc[mask]
        state.data[varname] = new_col
    else:
        state.data[varname] = result

    # Apply type conversion if specified
    if type_prefix:
        state.data[varname] = _apply_type(state.data[varname], type_prefix)

    n_set = state.data[varname].notna().sum()
    console.print(f"[green]Generated: {varname} ({n_set:,} real changes made)[/green]")
    state.mark_changed()


def cmd_replace(rest: str, state: AppState, console: Console):
    """
    replace var = expression [if condition]
    Modify an existing variable.
    """
    state.require_data()
    parsed = parse_command_line(rest)

    if not parsed["varlist"] or not parsed["expression"]:
        console.print("[red]Syntax: replace var = expression [if condition][/red]")
        return

    varname = parsed["varlist"][0]

    if varname not in state.data.columns:
        console.print(f"[red]Variable '{varname}' not found. Use 'generate' to create it.[/red]")
        return

    by_vars = getattr(state, "_by_vars", None)
    result = eval_expression(parsed["expression"], state.data,
                              by_vars=by_vars, state=state)

    if parsed["if_cond"]:
        mask = eval_condition(parsed["if_cond"], state.data,
                               by_vars=by_vars, state=state)
        n_changed = mask.sum()
        state.data.loc[mask, varname] = result.loc[mask]
    else:
        n_changed = len(state.data)
        state.data[varname] = result

    console.print(f"[green]({n_changed:,} real changes made)[/green]")
    state.mark_changed()


def cmd_rename(rest: str, state: AppState, console: Console):
    """
    rename old_name new_name
    Rename a variable.
    """
    state.require_data()
    parts = rest.strip().split()

    if len(parts) != 2:
        console.print("[red]Syntax: rename old_name new_name[/red]")
        return

    old_name, new_name = parts[0], parts[1]

    if old_name not in state.data.columns:
        console.print(f"[red]Variable '{old_name}' not found[/red]")
        return

    if new_name in state.data.columns:
        console.print(f"[red]Variable '{new_name}' already exists[/red]")
        return

    state.data = state.data.rename(columns={old_name: new_name})

    # Update labels
    if old_name in state.variable_labels:
        state.variable_labels[new_name] = state.variable_labels.pop(old_name)
    if old_name in state.value_label_assignments:
        state.value_label_assignments[new_name] = state.value_label_assignments.pop(old_name)

    console.print(f"[green]Renamed: {old_name} → {new_name}[/green]")
    state.mark_changed()


def cmd_drop(rest: str, state: AppState, console: Console):
    """
    drop varlist          — Drop variables
    drop if condition     — Drop observations
    """
    state.require_data()

    # Check if it's "drop if ..."
    if rest.strip().startswith("if "):
        cond = rest.strip()[3:]
        mask = eval_condition(cond, state.data)
        n_drop = mask.sum()
        state.data = state.data.loc[~mask].reset_index(drop=True)
        console.print(f"[green]({n_drop:,} observations deleted)[/green]")
        state.mark_changed()
        return

    # Drop variables
    parsed = parse_command_line(rest)
    if not parsed["varlist"]:
        console.print("[red]Syntax: drop varlist | drop if condition[/red]")
        return

    from commands.explore import _resolve_varlist
    vars_to_drop = _resolve_varlist(parsed["varlist"], state.data)

    state.data = state.data.drop(columns=vars_to_drop)

    # Clean up labels
    for v in vars_to_drop:
        state.variable_labels.pop(v, None)
        state.value_label_assignments.pop(v, None)

    console.print(f"[green]Dropped {len(vars_to_drop)} variable(s): {', '.join(vars_to_drop)}[/green]")
    state.mark_changed()


def cmd_keep(rest: str, state: AppState, console: Console):
    """
    keep varlist          — Keep only specified variables
    keep if condition     — Keep only matching observations
    """
    state.require_data()

    # Check if it's "keep if ..."
    if rest.strip().startswith("if "):
        cond = rest.strip()[3:]
        mask = eval_condition(cond, state.data)
        n_before = len(state.data)
        state.data = state.data.loc[mask].reset_index(drop=True)
        n_drop = n_before - len(state.data)
        console.print(f"[green]({n_drop:,} observations deleted)[/green]")
        state.mark_changed()
        return

    # Keep variables
    parsed = parse_command_line(rest)
    if not parsed["varlist"]:
        console.print("[red]Syntax: keep varlist | keep if condition[/red]")
        return

    from commands.explore import _resolve_varlist
    vars_to_keep = _resolve_varlist(parsed["varlist"], state.data)

    dropped = [c for c in state.data.columns if c not in vars_to_keep]
    state.data = state.data[vars_to_keep]

    for v in dropped:
        state.variable_labels.pop(v, None)
        state.value_label_assignments.pop(v, None)

    console.print(f"[green]Kept {len(vars_to_keep)} variable(s), dropped {len(dropped)}[/green]")
    state.mark_changed()


def cmd_label(rest: str, state: AppState, console: Console):
    """
    label variable varname "label text"
    label define lblname val1 "label1" val2 "label2" ...
    label values varname lblname
    label list [lblname]
    """
    state.require_data()
    parts = rest.strip().split(None, 1)

    if not parts:
        console.print("[red]Syntax: label variable|define|values|list ...[/red]")
        return

    subcmd = parts[0].lower()
    sub_rest = parts[1] if len(parts) > 1 else ""

    if subcmd == "variable" or subcmd == "var":
        # label variable varname "label"
        m = re.match(r'(\w+)\s+"([^"]*)"', sub_rest)
        if not m:
            m = re.match(r"(\w+)\s+'([^']*)'", sub_rest)
        if not m:
            console.print('[red]Syntax: label variable varname "label text"[/red]')
            return
        varname, label_text = m.group(1), m.group(2)
        if varname not in state.data.columns:
            console.print(f"[red]Variable '{varname}' not found[/red]")
            return
        state.set_variable_label(varname, label_text)
        console.print(f"[green]Label set: {varname} → \"{label_text}\"[/green]")

    elif subcmd == "define":
        # label define lblname val1 "label1" val2 "label2"
        m = re.match(r'(\w+)\s+(.*)', sub_rest)
        if not m:
            console.print('[red]Syntax: label define lblname value "label" ...[/red]')
            return
        lbl_name = m.group(1)
        pairs_str = m.group(2)

        # Parse value-label pairs
        label_dict = {}
        for pm in re.finditer(r'(\S+)\s+"([^"]*)"', pairs_str):
            val = pm.group(1)
            try:
                val = int(val)
            except ValueError:
                try:
                    val = float(val)
                except ValueError:
                    pass
            label_dict[val] = pm.group(2)

        if not label_dict:
            console.print('[red]No value-label pairs found[/red]')
            return

        state.value_labels[lbl_name] = label_dict
        console.print(f"[green]Defined label '{lbl_name}' with {len(label_dict)} values[/green]")

    elif subcmd == "values":
        # label values varname lblname
        parts2 = sub_rest.strip().split()
        if len(parts2) != 2:
            console.print("[red]Syntax: label values varname lblname[/red]")
            return
        varname, lbl_name = parts2
        if varname not in state.data.columns:
            console.print(f"[red]Variable '{varname}' not found[/red]")
            return
        if lbl_name not in state.value_labels:
            console.print(f"[red]Label '{lbl_name}' not defined[/red]")
            return
        state.value_label_assignments[varname] = lbl_name
        console.print(f"[green]Attached label '{lbl_name}' to {varname}[/green]")

    elif subcmd == "list":
        lbl_name = sub_rest.strip() if sub_rest.strip() else None
        if lbl_name:
            if lbl_name in state.value_labels:
                console.print(f"\n[bold]{lbl_name}:[/bold]")
                for val, txt in state.value_labels[lbl_name].items():
                    console.print(f"  {val:>10} {txt}")
            else:
                console.print(f"[red]Label '{lbl_name}' not found[/red]")
        else:
            if state.value_labels:
                for name, mapping in state.value_labels.items():
                    console.print(f"\n[bold]{name}:[/bold]")
                    for val, txt in mapping.items():
                        console.print(f"  {val:>10} {txt}")
            else:
                console.print("[dim]No value labels defined.[/dim]")


def cmd_destring(rest: str, state: AppState, console: Console):
    """
    destring varlist, replace [force]
    Convert string variables to numeric.
    """
    state.require_data()
    parsed = parse_command_line(rest)

    if not parsed["varlist"]:
        console.print("[red]Syntax: destring varlist, replace[/red]")
        return

    force = "force" in parsed["options"]

    for var in parsed["varlist"]:
        if var not in state.data.columns:
            console.print(f"[red]Variable '{var}' not found[/red]")
            continue

        original = state.data[var].copy()

        if force:
            state.data[var] = pd.to_numeric(state.data[var], errors="coerce")
        else:
            try:
                state.data[var] = pd.to_numeric(state.data[var])
            except ValueError:
                # Try stripping common non-numeric chars
                cleaned = state.data[var].astype(str).str.replace(r'[,$%\s]', '', regex=True)
                try:
                    state.data[var] = pd.to_numeric(cleaned)
                except ValueError:
                    console.print(f"[yellow]{var}: contains non-numeric values. Use 'destring {var}, replace force'[/yellow]")
                    state.data[var] = original
                    continue

        n_missing = state.data[var].isna().sum() - original.isna().sum()
        console.print(f"[green]{var}: converted to numeric"
                      + (f" ({n_missing} values set to missing)" if n_missing > 0 else "")
                      + "[/green]")

    state.mark_changed()


def cmd_tostring(rest: str, state: AppState, console: Console):
    """
    tostring varlist, replace [format(fmt)]
    Convert numeric variables to string.
    """
    state.require_data()
    parsed = parse_command_line(rest)

    if not parsed["varlist"]:
        console.print("[red]Syntax: tostring varlist, replace[/red]")
        return

    fmt = parsed["options"].get("format", None)

    for var in parsed["varlist"]:
        if var not in state.data.columns:
            console.print(f"[red]Variable '{var}' not found[/red]")
            continue

        if fmt:
            state.data[var] = state.data[var].apply(lambda x: format(x, fmt) if pd.notna(x) else "")
        else:
            state.data[var] = state.data[var].astype(str)
            state.data.loc[state.data[var] == "nan", var] = ""

        console.print(f"[green]{var}: converted to string[/green]")

    state.mark_changed()


def cmd_encode(rest: str, state: AppState, console: Console):
    """
    encode var, generate(newvar)
    Encode string variable to numeric with value labels.
    """
    state.require_data()
    parsed = parse_command_line(rest)

    if not parsed["varlist"]:
        console.print("[red]Syntax: encode var, generate(newvar)[/red]")
        return

    var = parsed["varlist"][0]
    newvar = parsed["options"].get("generate") or parsed["options"].get("gen")

    if not newvar:
        console.print("[red]Syntax: encode var, generate(newvar)[/red]")
        return

    if var not in state.data.columns:
        console.print(f"[red]Variable '{var}' not found[/red]")
        return

    if newvar in state.data.columns:
        console.print(f"[red]Variable '{newvar}' already exists[/red]")
        return

    # Create encoding
    unique_vals = sorted(state.data[var].dropna().unique())
    encoding = {val: i + 1 for i, val in enumerate(unique_vals)}
    reverse = {i + 1: val for val, i in encoding.items()}

    state.data[newvar] = state.data[var].map(encoding)

    # Create value labels
    lbl_name = newvar
    state.value_labels[lbl_name] = reverse
    state.value_label_assignments[newvar] = lbl_name

    console.print(f"[green]Encoded {var} → {newvar} ({len(encoding)} categories)[/green]")
    state.mark_changed()


def _apply_type(series, type_str):
    """Apply econometric type to a pandas Series."""
    type_map = {
        "byte": "int8",
        "int": "int16",
        "long": "int32",
        "float": "float32",
        "double": "float64",
    }
    if type_str.startswith("str"):
        return series.astype(str)
    if type_str in type_map:
        try:
            return series.astype(type_map[type_str])
        except (ValueError, OverflowError):
            return series
    return series


def cmd_order(rest: str, state: AppState, console: Console):
    """
    order varlist [, first last after(var) before(var) alphabetical]
    Reorder variables in the dataset.
    """
    state.require_data()
    parsed = parse_command_line(rest)

    # alphabetical shortcut
    if "alphabetical" in parsed["options"] or "alpha" in parsed["options"]:
        state.data = state.data[sorted(state.data.columns)]
        console.print(f"[green]Variables reordered alphabetically[/green]")
        state.mark_changed()
        return

    if not parsed["varlist"]:
        console.print("[red]Syntax: order varlist [, first|last|after(var)|before(var)|alphabetical][/red]")
        return

    from commands.explore import _resolve_varlist
    move_vars = _resolve_varlist(parsed["varlist"], state.data)
    all_cols = list(state.data.columns)
    remaining = [c for c in all_cols if c not in move_vars]

    after_var = parsed["options"].get("after")
    before_var = parsed["options"].get("before")

    if after_var:
        if after_var not in state.data.columns:
            console.print(f"[red]Variable '{after_var}' not found[/red]")
            return
        idx = remaining.index(after_var) + 1
        new_order = remaining[:idx] + move_vars + remaining[idx:]
    elif before_var:
        if before_var not in state.data.columns:
            console.print(f"[red]Variable '{before_var}' not found[/red]")
            return
        idx = remaining.index(before_var)
        new_order = remaining[:idx] + move_vars + remaining[idx:]
    elif "last" in parsed["options"]:
        new_order = remaining + move_vars
    else:
        # Default: first
        new_order = move_vars + remaining

    state.data = state.data[new_order]
    console.print(f"[green]Variables reordered: {', '.join(move_vars)} moved[/green]")
    state.mark_changed()


def cmd_recode(rest: str, state: AppState, console: Console):
    """
    recode var (old1 = new1) (old2 = new2) ... [, generate(newvar)]
    recode var (1/5 = 1) (6/10 = 2) (else = 3) [, generate(newvar)]
    Recode values of a variable.
    """
    state.require_data()

    # Parse: recode varname (rule1) (rule2) ... [, options]
    # First extract options
    opts = {}
    options_match = re.search(r',\s*(\w.*)', rest)
    working = rest
    if options_match:
        opts_str = options_match.group(1)
        working = rest[:options_match.start()]
        from commands.parse_helpers import _parse_options
        opts = _parse_options(opts_str)

    # Get variable name
    parts = working.strip().split(None, 1)
    if len(parts) < 2:
        console.print("[red]Syntax: recode var (old=new) (old=new) ... [, generate(newvar)][/red]")
        return

    varname = parts[0]
    rules_str = parts[1]

    if varname not in state.data.columns:
        console.print(f"[red]Variable '{varname}' not found[/red]")
        return

    # Parse recode rules: (value = newvalue) or (lo/hi = newvalue) or (else = newvalue)
    rules = []
    else_value = None

    for m in re.finditer(r'\(\s*([^)]+?)\s*=\s*([^)]+?)\s*\)', rules_str):
        lhs = m.group(1).strip()
        rhs = m.group(2).strip()

        # Try to convert rhs to numeric
        rhs_val = _try_numeric(rhs)

        if lhs.lower() in ("else", "*", "nonmissing"):
            else_value = rhs_val
        elif "/" in lhs:
            # Range: lo/hi
            lo_str, hi_str = lhs.split("/", 1)
            lo = _try_numeric(lo_str.strip())
            hi = _try_numeric(hi_str.strip())
            rules.append(("range", lo, hi, rhs_val))
        elif " " in lhs:
            # Multiple values
            vals = [_try_numeric(v.strip()) for v in lhs.split()]
            rules.append(("multi", vals, rhs_val))
        else:
            # Single value
            rules.append(("single", _try_numeric(lhs), rhs_val))

    if not rules and else_value is None:
        console.print("[red]No valid recode rules found. Use: (old=new)[/red]")
        return

    # Determine target column
    gen_var = opts.get("generate") or opts.get("gen")
    target = gen_var if gen_var else varname

    if gen_var:
        state.data[target] = state.data[varname].copy()
    elif target != varname:
        state.data[target] = state.data[varname].copy()

    n_changed = 0

    for rule_type, *rule_args in rules:
        if rule_type == "single":
            old_val, new_val = rule_args
            mask = state.data[target] == old_val
            n_changed += mask.sum()
            state.data.loc[mask, target] = new_val
        elif rule_type == "range":
            lo, hi, new_val = rule_args
            mask = (state.data[target] >= lo) & (state.data[target] <= hi)
            n_changed += mask.sum()
            state.data.loc[mask, target] = new_val
        elif rule_type == "multi":
            vals, new_val = rule_args
            mask = state.data[target].isin(vals)
            n_changed += mask.sum()
            state.data.loc[mask, target] = new_val

    if else_value is not None:
        # Apply else to anything not already recoded
        # Track which rows were recoded
        already = pd.Series(False, index=state.data.index)
        for rule_type, *rule_args in rules:
            if rule_type == "single":
                already |= (state.data[varname] == rule_args[0])
            elif rule_type == "range":
                already |= ((state.data[varname] >= rule_args[0]) & (state.data[varname] <= rule_args[1]))
            elif rule_type == "multi":
                already |= state.data[varname].isin(rule_args[0])
        else_mask = ~already & state.data[target].notna()
        n_changed += else_mask.sum()
        state.data.loc[else_mask, target] = else_value

    if gen_var:
        console.print(f"[green]Recoded {varname} → {target} ({n_changed:,} changes)[/green]")
    else:
        console.print(f"[green]({n_changed:,} changes made to {target})[/green]")
    state.mark_changed()


def cmd_reshape(rest: str, state: AppState, console: Console):
    """
    reshape long stubnames, i(id_var) j(time_var)
    reshape wide stubnames, i(id_var) j(time_var)
    Transform between wide and long format.
    """
    state.require_data()

    parts = rest.strip().split(None, 1)
    if len(parts) < 2 or parts[0].lower() not in ("long", "wide"):
        console.print("[red]Syntax: reshape long|wide stubnames, i(idvar) j(timevar)[/red]")
        return

    direction = parts[0].lower()
    sub_rest = parts[1]

    parsed = parse_command_line(sub_rest)
    stubs = parsed["varlist"]

    i_var = parsed["options"].get("i")
    j_var = parsed["options"].get("j")

    if not i_var or not j_var:
        console.print("[red]Must specify i() and j() options[/red]")
        console.print("[dim]Example: reshape long score, i(id) j(year)[/dim]")
        return

    # i can be multiple vars
    i_vars = [v.strip() for v in i_var.split()]

    nobs_before = len(state.data)
    nvars_before = len(state.data.columns)
    data_before_id = id(state.data)

    if direction == "long":
        _reshape_long(stubs, i_vars, j_var, state, console)
    else:
        _reshape_wide(stubs, i_vars, j_var, state, console)

    # If the helper bailed early, state.data is unchanged — don't print
    # a misleading "Reshaped" summary.
    if id(state.data) == data_before_id:
        return

    nobs_after = len(state.data)
    nvars_after = len(state.data.columns)
    console.print(f"[green]Reshaped: {nobs_before:,} obs / {nvars_before} vars → "
                  f"{nobs_after:,} obs / {nvars_after} vars[/green]")
    state.mark_changed()


def _reshape_long(stubs, i_vars, j_var, state, console):
    """Reshape from wide to long."""
    df = state.data

    # Find stub columns: e.g. stub="income" matches income1, income2, income2020
    stub_cols = {}
    for stub in stubs:
        matching = [c for c in df.columns if c.startswith(stub) and c != stub
                    and c[len(stub):].strip("_").replace(".", "").isdigit()]
        if not matching:
            # Also try with underscore separator: income_1, income_2
            matching = [c for c in df.columns if c.startswith(stub + "_")
                        and c[len(stub)+1:].replace(".", "").isdigit()]
        if not matching:
            console.print(f"[yellow]No wide columns found for stub '{stub}'[/yellow]")
            console.print(f"[dim]Expected columns like {stub}1, {stub}2, ... or {stub}_1, {stub}_2, ...[/dim]")
            return
        stub_cols[stub] = matching

    # Determine j values from column suffixes
    first_stub = stubs[0]
    first_cols = stub_cols[first_stub]
    j_values = []
    sep = ""
    for col in sorted(first_cols):
        suffix = col[len(first_stub):]
        if suffix.startswith("_"):
            sep = "_"
            suffix = suffix[1:]
        j_values.append(suffix)

    # Build value_vars mapping for pd.melt or manual reshape
    id_cols = i_vars + [c for c in df.columns if c not in sum(stub_cols.values(), []) and c not in i_vars]

    # Use pd.wide_to_long
    try:
        result = pd.wide_to_long(
            df, stubnames=stubs, i=i_vars, j=j_var, sep=sep, suffix=r'\d+'
        ).reset_index()
        state.data = result
    except Exception as e:
        console.print(f"[red]Reshape failed: {e}[/red]")


def _reshape_wide(stubs, i_vars, j_var, state, console):
    """Reshape from long to wide."""
    df = state.data

    if j_var not in df.columns:
        console.print(f"[red]Variable '{j_var}' not found[/red]")
        return

    for v in i_vars:
        if v not in df.columns:
            console.print(f"[red]Variable '{v}' not found[/red]")
            return

    for stub in stubs:
        if stub not in df.columns:
            console.print(f"[red]Variable '{stub}' not found[/red]")
            return

    # Determine non-stub, non-id, non-j columns to keep
    other_cols = [c for c in df.columns if c not in stubs and c not in i_vars and c != j_var]

    # Detect non-unique (i_vars, j_var) combinations BEFORE pivoting.
    # Otherwise pd.pivot_table with aggfunc="first" silently drops every
    # value except the first per cell — a major data-integrity hazard
    # (Gemini v126 finding).
    dup_keys = df.duplicated(subset=i_vars + [j_var], keep=False)
    if dup_keys.any():
        n_dups = int(dup_keys.sum())
        # Show a few of the offending key tuples so the user can fix the data.
        sample_keys = (df.loc[dup_keys, i_vars + [j_var]]
                         .drop_duplicates().head(5).to_dict("records"))
        console.print(
            f"[red]reshape wide: {n_dups} observations have duplicate "
            f"({', '.join(i_vars + [j_var])}) combinations.[/red]")
        console.print(
            f"[red]These cells would silently lose data. Aborting.[/red]")
        console.print("[dim]First few duplicate keys:[/dim]")
        for k in sample_keys:
            console.print(f"[dim]  {k}[/dim]")
        console.print(
            "[dim]Fix: deduplicate with `duplicates drop` or aggregate first "
            "with `collapse`.[/dim]")
        return

    try:
        # Pivot each stub
        result = df[i_vars + other_cols].drop_duplicates(subset=i_vars)

        for stub in stubs:
            pivot = df.pivot_table(
                index=i_vars, columns=j_var, values=stub, aggfunc="first"
            )
            pivot.columns = [f"{stub}{j}" for j in pivot.columns]
            result = result.merge(pivot, on=i_vars, how="left")

        state.data = result.reset_index(drop=True)
    except Exception as e:
        console.print(f"[red]Reshape failed: {e}[/red]")


def _try_numeric(val_str: str):
    """Try to convert a string to numeric."""
    val_str = val_str.strip().strip("\"'")
    try:
        return int(val_str)
    except ValueError:
        try:
            return float(val_str)
        except ValueError:
            return val_str
