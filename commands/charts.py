"""
Charting commands for Rln.
Wraps matplotlib for compact graph syntax.

Commands:
  histogram var [if] [, bins(N) normal title("text") name(file)]
  kdensity var [if] [, title("text") bwidth(N)]
  scatter yvar xvar [if] [, title("text") mcolor(color) by(var)]
  graph bar (stat) var [, over(var) title("text") horizontal]
  graph box var [, over(var) title("text")]
  graph line yvar xvar [if] [, title("text") sort]
  twoway (scatter y x) (lfit y x) [, title("text") legend]
  graph export "filename" [, replace width(N) height(N)]
  marginsplot [, title("text")]
  graph close
"""

import re
import os
import numpy as np
import pandas as pd
from rich.console import Console

from commands.state import AppState
from commands.parse_helpers import parse_command_line
from commands.expression import eval_condition


def _check_matplotlib():
    try:
        import matplotlib
        matplotlib.use("Agg")  # Always non-interactive — graphs open in system viewer
    except ImportError:
        raise ImportError(
            "matplotlib is required for charts.\n"
            "Install with: ssc install matplotlib"
        )
    import matplotlib.pyplot as plt
    return plt


def _apply_common_options(ax, plt, parsed, default_title=""):
    """Apply common graph options: title, xlabel, ylabel, legend."""
    opts = parsed["options"]
    title = opts.get("title", opts.get("ti", default_title))
    xlabel = opts.get("xtitle", opts.get("xlabel", ""))
    ylabel = opts.get("ytitle", opts.get("ylabel", ""))

    if title:
        ax.set_title(title, fontsize=13, fontweight="bold")
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)

    ax.grid(True, alpha=0.3)
    plt.tight_layout()


def _show_or_save(plt, parsed, state, console):
    """Show the plot interactively or save to file."""
    opts = parsed["options"]
    name = opts.get("name") or opts.get("saving")

    if name:
        # Save to file
        name = name.strip("\"'")
        if not os.path.splitext(name)[1]:
            name += ".png"
        width = int(opts.get("width", 800))
        height = int(opts.get("height", 600))
        dpi = int(opts.get("dpi", 150))
        plt.gcf().set_size_inches(width / dpi, height / dpi)
        plt.savefig(name, dpi=dpi, bbox_inches="tight")
        console.print(f"[green]Graph saved: {name}[/green]")
        plt.close()
    else:
        # Save to temp file and open with system viewer (non-blocking)
        import tempfile
        import subprocess
        import sys

        state._last_figure = plt.gcf()
        tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False, prefix="rln_graph_")
        plt.savefig(tmp.name, dpi=150, bbox_inches="tight")
        plt.close()

        # Try to open with system default viewer
        opened = False
        try:
            if sys.platform == "win32":
                os.startfile(tmp.name)
                opened = True
            elif sys.platform == "darwin":
                subprocess.Popen(["open", tmp.name])
                opened = True
            else:
                for viewer in ["xdg-open", "eog", "feh", "display"]:
                    try:
                        subprocess.Popen([viewer, tmp.name], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                        opened = True
                        break
                    except FileNotFoundError:
                        continue
        except Exception:
            pass

        if opened:
            console.print(f"[dim]Graph opened in viewer. Saved to: {tmp.name}[/dim]")
        else:
            console.print(f"[dim]Graph saved to: {tmp.name}[/dim]")
        console.print(f"[dim]Use 'graph export \"file.png\"' to save to a custom path.[/dim]")


# ──────────────────────────────────────────────
#  histogram
# ──────────────────────────────────────────────

def cmd_histogram(rest: str, state: AppState, console: Console):
    """
    histogram var [if condition] [, bins(N) normal frequency density
         title("text") color(color) name(file)]
    """
    state.require_data()
    plt = _check_matplotlib()
    parsed = parse_command_line(rest)

    if not parsed["varlist"]:
        console.print("[red]Syntax: histogram varname [, bins(N) normal][/red]")
        return

    var = parsed["varlist"][0]
    if var not in state.data.columns:
        console.print(f"[red]Variable '{var}' not found[/red]")
        return

    df = state.data
    if parsed["if_cond"]:
        mask = eval_condition(parsed["if_cond"], df)
        df = df.loc[mask]

    series = df[var].dropna()
    opts = parsed["options"]

    bins = int(opts.get("bins", opts.get("bin", min(max(int(np.sqrt(len(series))), 10), 50))))
    color = opts.get("color", opts.get("fcolor", "#4A90D9"))
    show_normal = "normal" in opts
    use_density = "density" in opts or show_normal
    use_freq = "frequency" in opts or "freq" in opts

    fig, ax = plt.subplots(figsize=(8, 5))

    if use_freq and not use_density:
        ax.hist(series, bins=bins, color=color, edgecolor="white", alpha=0.85)
        ax.set_ylabel("Frequency")
    else:
        ax.hist(series, bins=bins, density=True, color=color, edgecolor="white", alpha=0.85)
        ax.set_ylabel("Density")

    if show_normal:
        x = np.linspace(series.min(), series.max(), 200)
        from scipy.stats import norm
        ax.plot(x, norm.pdf(x, series.mean(), series.std()),
                color="#E74C3C", linewidth=2, label="Normal")
        ax.legend()

    label = state.get_variable_label(var)
    ax.set_xlabel(label if label else var)

    _apply_common_options(ax, plt, parsed, default_title=f"Histogram of {var}")
    _show_or_save(plt, parsed, state, console)


# ──────────────────────────────────────────────
#  kdensity
# ──────────────────────────────────────────────

def cmd_kdensity(rest: str, state: AppState, console: Console):
    """
    kdensity var [if condition] [, bwidth(N) normal title("text") name(file)]
    Kernel density plot.
    """
    state.require_data()
    plt = _check_matplotlib()
    parsed = parse_command_line(rest)

    if not parsed["varlist"]:
        console.print("[red]Syntax: kdensity varname[/red]")
        return

    var = parsed["varlist"][0]
    if var not in state.data.columns:
        console.print(f"[red]Variable '{var}' not found[/red]")
        return

    df = state.data
    if parsed["if_cond"]:
        mask = eval_condition(parsed["if_cond"], df)
        df = df.loc[mask]

    series = df[var].dropna()
    opts = parsed["options"]
    show_normal = "normal" in opts

    fig, ax = plt.subplots(figsize=(8, 5))

    from scipy.stats import gaussian_kde
    bw = float(opts.get("bwidth", opts.get("bw", "0"))) or None
    kde = gaussian_kde(series, bw_method=bw)
    x = np.linspace(series.min() - series.std(), series.max() + series.std(), 300)
    ax.plot(x, kde(x), color="#4A90D9", linewidth=2, label="Kernel density")
    ax.fill_between(x, kde(x), alpha=0.15, color="#4A90D9")

    if show_normal:
        from scipy.stats import norm
        ax.plot(x, norm.pdf(x, series.mean(), series.std()),
                color="#E74C3C", linewidth=2, linestyle="--", label="Normal")

    ax.legend()
    ax.set_ylabel("Density")
    ax.set_xlabel(state.get_variable_label(var) or var)

    _apply_common_options(ax, plt, parsed, default_title=f"Kernel density of {var}")
    _show_or_save(plt, parsed, state, console)


# ──────────────────────────────────────────────
#  scatter
# ──────────────────────────────────────────────

def cmd_scatter(rest: str, state: AppState, console: Console):
    """
    scatter yvar xvar [if] [, by(var) mcolor(color) msize(N)
         title("text") lfit mlabel(var) name(file)]
    """
    state.require_data()
    plt = _check_matplotlib()
    parsed = parse_command_line(rest)

    if len(parsed["varlist"]) < 2:
        console.print("[red]Syntax: scatter yvar xvar [, by(var) lfit][/red]")
        return

    yvar, xvar = parsed["varlist"][0], parsed["varlist"][1]
    for v in [yvar, xvar]:
        if v not in state.data.columns:
            console.print(f"[red]Variable '{v}' not found[/red]")
            return

    df = state.data
    if parsed["if_cond"]:
        mask = eval_condition(parsed["if_cond"], df)
        df = df.loc[mask]

    opts = parsed["options"]
    by_var = opts.get("by")
    show_lfit = "lfit" in opts
    mcolor = opts.get("mcolor", opts.get("color", None))
    msize = float(opts.get("msize", 20))
    mlabel_var = opts.get("mlabel")

    fig, ax = plt.subplots(figsize=(8, 6))

    if by_var and by_var in df.columns:
        groups = df.groupby(by_var)
        colors = plt.cm.Set2(np.linspace(0, 1, len(groups)))
        for (name, group), color in zip(groups, colors):
            clean = group[[xvar, yvar]].dropna()
            ax.scatter(clean[xvar], clean[yvar], s=msize, alpha=0.7,
                       label=str(name), color=color)
        ax.legend(title=by_var)
    else:
        clean = df[[xvar, yvar]].dropna()
        ax.scatter(clean[xvar], clean[yvar], s=msize, alpha=0.7,
                   color=mcolor or "#4A90D9")

    if show_lfit:
        clean = df[[xvar, yvar]].dropna()
        z = np.polyfit(clean[xvar], clean[yvar], 1)
        p = np.poly1d(z)
        x_line = np.linspace(clean[xvar].min(), clean[xvar].max(), 100)
        ax.plot(x_line, p(x_line), color="#E74C3C", linewidth=2,
                linestyle="--", label=f"Fitted: y={z[0]:.3f}x+{z[1]:.3f}")
        ax.legend()

    if mlabel_var and mlabel_var in df.columns:
        clean = df[[xvar, yvar, mlabel_var]].dropna()
        for _, row in clean.iterrows():
            ax.annotate(str(row[mlabel_var]), (row[xvar], row[yvar]),
                        fontsize=7, alpha=0.7)

    ax.set_xlabel(state.get_variable_label(xvar) or xvar)
    ax.set_ylabel(state.get_variable_label(yvar) or yvar)

    _apply_common_options(ax, plt, parsed, default_title=f"{yvar} vs {xvar}")
    _show_or_save(plt, parsed, state, console)


# ──────────────────────────────────────────────
#  graph bar
# ──────────────────────────────────────────────

def cmd_graph(rest: str, state: AppState, console: Console):
    """
    graph bar (stat) var [, over(var) title("text") horizontal]
    graph box var [, over(var) title("text")]
    graph line yvar xvar [, title("text") sort]
    graph export "filename" [, replace width(N) height(N)]
    graph close
    """
    state.require_data()

    parts = rest.strip().split(None, 1)
    if not parts:
        console.print("[red]Syntax: graph bar|box|line|export|close ...[/red]")
        return

    subtype = parts[0].lower()
    sub_rest = parts[1] if len(parts) > 1 else ""

    if subtype == "bar":
        _graph_bar(sub_rest, state, console)
    elif subtype == "box":
        _graph_box(sub_rest, state, console)
    elif subtype == "line":
        _graph_line(sub_rest, state, console)
    elif subtype == "export":
        _graph_export(sub_rest, state, console)
    elif subtype == "close":
        plt = _check_matplotlib()
        plt.close("all")
        console.print("[dim]All graphs closed.[/dim]")
    else:
        console.print(f"[red]Unknown graph type: {subtype}[/red]")
        console.print("[dim]Available: bar, box, line, export, close[/dim]")


def _graph_bar(rest, state, console):
    """Bar chart: graph bar (stat) var [, over(var)]"""
    plt = _check_matplotlib()
    parsed = parse_command_line(rest)

    # Parse (stat) prefix
    stat = "mean"
    raw = parsed["raw"].split(",")[0].strip()
    m = re.match(r'\((\w+)\)\s*(.*)', raw)
    if m:
        stat = m.group(1).lower()
        varlist_str = m.group(2).strip()
        varlist = varlist_str.split() if varlist_str else parsed["varlist"]
    else:
        varlist = parsed["varlist"]

    if not varlist:
        console.print("[red]Syntax: graph bar (mean) varname [, over(groupvar)][/red]")
        return

    var = varlist[0]
    opts = parsed["options"]
    over_var = opts.get("over")
    horizontal = "horizontal" in opts or "horiz" in opts

    fig, ax = plt.subplots(figsize=(8, 5))

    if over_var and over_var in state.data.columns:
        grouped = state.data.groupby(over_var)[var].agg(stat)
        if horizontal:
            ax.barh(range(len(grouped)), grouped.values, color="#4A90D9", alpha=0.85)
            ax.set_yticks(range(len(grouped)))
            ax.set_yticklabels([str(x) for x in grouped.index])
            ax.set_xlabel(f"{stat}({var})")
        else:
            ax.bar(range(len(grouped)), grouped.values, color="#4A90D9", alpha=0.85)
            ax.set_xticks(range(len(grouped)))
            ax.set_xticklabels([str(x) for x in grouped.index], rotation=45, ha="right")
            ax.set_ylabel(f"{stat}({var})")
    else:
        val = state.data[var].agg(stat)
        ax.bar([var], [val], color="#4A90D9", alpha=0.85)
        ax.set_ylabel(f"{stat}")

    _apply_common_options(ax, plt, parsed, default_title=f"{stat.title()} of {var}")
    _show_or_save(plt, parsed, state, console)


def _graph_box(rest, state, console):
    """Box plot: graph box var [, over(var)]"""
    plt = _check_matplotlib()
    parsed = parse_command_line(rest)

    if not parsed["varlist"]:
        console.print("[red]Syntax: graph box varname [, over(groupvar)][/red]")
        return

    var = parsed["varlist"][0]
    opts = parsed["options"]
    over_var = opts.get("over")

    fig, ax = plt.subplots(figsize=(8, 5))

    if over_var and over_var in state.data.columns:
        groups = sorted(state.data[over_var].dropna().unique())
        data = [state.data[state.data[over_var] == g][var].dropna().values for g in groups]
        bp = ax.boxplot(data, labels=[str(g) for g in groups], patch_artist=True)
        colors = plt.cm.Set2(np.linspace(0, 1, len(groups)))
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        ax.set_xlabel(over_var)
    else:
        bp = ax.boxplot(state.data[var].dropna().values, patch_artist=True)
        bp["boxes"][0].set_facecolor("#4A90D9")
        bp["boxes"][0].set_alpha(0.7)

    ax.set_ylabel(state.get_variable_label(var) or var)

    _apply_common_options(ax, plt, parsed, default_title=f"Box plot of {var}")
    _show_or_save(plt, parsed, state, console)


def _graph_line(rest, state, console):
    """Line plot: graph line yvar xvar [, sort by(var)]"""
    plt = _check_matplotlib()
    parsed = parse_command_line(rest)

    if len(parsed["varlist"]) < 2:
        console.print("[red]Syntax: graph line yvar xvar [, sort by(var)][/red]")
        return

    yvar, xvar = parsed["varlist"][0], parsed["varlist"][1]
    opts = parsed["options"]
    by_var = opts.get("by")
    do_sort = "sort" in opts

    df = state.data.copy()
    if parsed["if_cond"]:
        mask = eval_condition(parsed["if_cond"], df)
        df = df.loc[mask]

    if do_sort:
        df = df.sort_values(xvar)

    fig, ax = plt.subplots(figsize=(8, 5))

    if by_var and by_var in df.columns:
        for name, group in df.groupby(by_var):
            clean = group[[xvar, yvar]].dropna().sort_values(xvar)
            ax.plot(clean[xvar], clean[yvar], marker="o", markersize=3,
                    linewidth=1.5, label=str(name), alpha=0.8)
        ax.legend(title=by_var)
    else:
        clean = df[[xvar, yvar]].dropna()
        ax.plot(clean[xvar], clean[yvar], color="#4A90D9",
                marker="o", markersize=3, linewidth=1.5, alpha=0.8)

    ax.set_xlabel(state.get_variable_label(xvar) or xvar)
    ax.set_ylabel(state.get_variable_label(yvar) or yvar)

    _apply_common_options(ax, plt, parsed, default_title=f"{yvar} over {xvar}")
    _show_or_save(plt, parsed, state, console)


def _graph_export(rest, state, console):
    """Export current graph: graph export "filename" [, replace width(N) height(N)]"""
    plt = _check_matplotlib()
    parsed = parse_command_line(rest)

    filepath = None
    if parsed["varlist"]:
        filepath = parsed["varlist"][0]
    if not filepath:
        console.print('[red]Syntax: graph export "filename.png" [, replace][/red]')
        return

    filepath = filepath.strip("\"'")
    if not os.path.splitext(filepath)[1]:
        filepath += ".png"

    opts = parsed["options"]
    width = int(opts.get("width", 800))
    height = int(opts.get("height", 600))
    dpi = int(opts.get("dpi", 150))

    if os.path.exists(filepath) and "replace" not in opts:
        console.print(f"[yellow]File exists. Use: graph export \"{filepath}\", replace[/yellow]")
        return

    fig = getattr(state, "_last_figure", None)
    if fig is None:
        # Try current figure
        fig = plt.gcf()
        if not fig.get_axes():
            console.print("[red]No graph to export. Create one first.[/red]")
            return

    fig.set_size_inches(width / dpi, height / dpi)
    fig.savefig(filepath, dpi=dpi, bbox_inches="tight")
    console.print(f"[green]Graph exported: {filepath}[/green]")


# ──────────────────────────────────────────────
#  twoway — multi-layered plots
# ──────────────────────────────────────────────

def cmd_twoway(rest: str, state: AppState, console: Console):
    """
    twoway (scatter y x) (lfit y x) [, title("text") legend name(file)]
    twoway (line y x, sort) (scatter y x)
    
    Layer multiple plot types. Each (plottype args) is one layer.
    Supported: scatter, line, lfit, lfitci, qfit, connected, area
    """
    state.require_data()
    plt = _check_matplotlib()

    # Parse layers: (type args [, opts]) (type args) ..., global_opts
    # Split global options
    global_opts_str = ""
    raw = rest.strip()

    # Find last ) then check for comma after it
    last_paren = raw.rfind(")")
    if last_paren >= 0 and last_paren < len(raw) - 1:
        after = raw[last_paren + 1:].strip()
        if after.startswith(","):
            global_opts_str = after[1:].strip()
            raw = raw[:last_paren + 1]

    global_parsed = parse_command_line(f"dummy, {global_opts_str}" if global_opts_str else "dummy")

    # Extract layers
    layers = re.findall(r'\(([^)]+)\)', raw)
    if not layers:
        console.print("[red]Syntax: twoway (scatter y x) (lfit y x) [, title(\"text\")][/red]")
        return

    fig, ax = plt.subplots(figsize=(9, 6))
    colors = ["#4A90D9", "#E74C3C", "#2ECC71", "#F39C12", "#9B59B6", "#1ABC9C"]

    for i, layer in enumerate(layers):
        _draw_layer(layer, state, ax, plt, colors[i % len(colors)], console)

    if ax.get_legend_handles_labels()[1]:
        ax.legend()

    _apply_common_options(ax, plt, global_parsed)
    _show_or_save(plt, global_parsed, state, console)


def _draw_layer(layer_str, state, ax, plt, color, console):
    """Draw a single twoway layer."""
    parts = layer_str.strip().split(None, 1)
    plot_type = parts[0].lower()
    layer_rest = parts[1] if len(parts) > 1 else ""
    layer_parsed = parse_command_line(layer_rest)

    if len(layer_parsed["varlist"]) < 2:
        console.print(f"[yellow]Layer '{plot_type}' needs at least 2 variables, skipping[/yellow]")
        return

    yvar = layer_parsed["varlist"][0]
    xvar = layer_parsed["varlist"][1]

    df = state.data
    if layer_parsed["if_cond"]:
        mask = eval_condition(layer_parsed["if_cond"], df)
        df = df.loc[mask]

    clean = df[[xvar, yvar]].dropna()
    if "sort" in layer_parsed["options"]:
        clean = clean.sort_values(xvar)

    if plot_type == "scatter":
        ax.scatter(clean[xvar], clean[yvar], s=20, alpha=0.6, color=color, label=yvar)

    elif plot_type in ("line", "connected"):
        clean = clean.sort_values(xvar)
        marker = "o" if plot_type == "connected" else ""
        ax.plot(clean[xvar], clean[yvar], color=color, linewidth=1.5,
                marker=marker, markersize=3, label=yvar)

    elif plot_type == "lfit":
        z = np.polyfit(clean[xvar], clean[yvar], 1)
        p = np.poly1d(z)
        x_line = np.linspace(clean[xvar].min(), clean[xvar].max(), 100)
        ax.plot(x_line, p(x_line), color=color, linewidth=2,
                linestyle="--", label=f"Linear fit")

    elif plot_type == "lfitci":
        from scipy import stats as sp_stats
        z = np.polyfit(clean[xvar], clean[yvar], 1)
        p = np.poly1d(z)
        x_line = np.linspace(clean[xvar].min(), clean[xvar].max(), 100)
        y_pred = p(x_line)
        ax.plot(x_line, y_pred, color=color, linewidth=2, label="Linear fit")
        # Approximate CI
        resid = clean[yvar] - p(clean[xvar])
        se = resid.std()
        ax.fill_between(x_line, y_pred - 1.96 * se, y_pred + 1.96 * se,
                         alpha=0.15, color=color)

    elif plot_type == "qfit":
        z = np.polyfit(clean[xvar], clean[yvar], 2)
        p = np.poly1d(z)
        x_line = np.linspace(clean[xvar].min(), clean[xvar].max(), 100)
        ax.plot(x_line, p(x_line), color=color, linewidth=2,
                linestyle="-.", label="Quadratic fit")

    elif plot_type == "area":
        clean = clean.sort_values(xvar)
        ax.fill_between(clean[xvar], clean[yvar], alpha=0.3, color=color, label=yvar)
        ax.plot(clean[xvar], clean[yvar], color=color, linewidth=1)

    else:
        console.print(f"[yellow]Unknown plot type: {plot_type}[/yellow]")

    ax.set_xlabel(state.get_variable_label(xvar) or xvar)
    ax.set_ylabel(state.get_variable_label(yvar) or yvar)


# ──────────────────────────────────────────────
#  marginsplot
# ──────────────────────────────────────────────

def cmd_marginsplot(rest: str, state: AppState, console: Console):
    """
    marginsplot [, title("text") name(file)]
    Plot margins from the last margins command.
    Requires a preceding 'margins, dydx(*)' or similar.
    """
    state.require_data()
    plt = _check_matplotlib()
    parsed = parse_command_line(rest)

    if not state.e_results or "predict_model" not in state.e_results:
        console.print("[red]No estimation results. Run regress then margins first.[/red]")
        return

    model = state.e_results["predict_model"]

    # Plot coefficient magnitudes as a margins-style plot
    params = model.params.drop("const", errors="ignore")
    ci = model.conf_int()
    if "const" in ci.index:
        ci = ci.drop("const")

    fig, ax = plt.subplots(figsize=(8, max(4, len(params) * 0.5 + 2)))

    y_pos = range(len(params))
    ax.barh(y_pos, params.values, xerr=[params.values - ci.iloc[:, 0].values,
                                          ci.iloc[:, 1].values - params.values],
            color="#4A90D9", alpha=0.7, ecolor="#333", capsize=3)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(params.index)
    ax.axvline(x=0, color="#E74C3C", linestyle="--", alpha=0.5)
    ax.set_xlabel("Coefficient")

    _apply_common_options(ax, plt, parsed, default_title="Marginal Effects")
    _show_or_save(plt, parsed, state, console)


# ──────────────────────────────────────────────────────────────────────
# coefplot — plot stored estimation coefficients with confidence intervals
# ──────────────────────────────────────────────────────────────────────

def cmd_coefplot(rest: str, state: AppState, console: Console):
    """
    coefplot [, level(95) title("...") xlabel("...") ylabel("...")
                ylim(lo hi) name("file.png") horizontal vertical
                connect zero_line(on|off) ref_line(on|off)]

    Plot the most recent estimation's coefficients with confidence intervals.

    Behavior depends on what the last estimation stored in e():

      * Event study / dynamic DiD (didregress with method(eventstudy), or
        cs/gardner/stacked with aggregate=event_study) -> forest plot of
        per-period effects vs. period number, with horizontal zero line and
        a vertical reference line at the omitted pre-period.

      * Scalar-ATT DiD (twfe, did, cs default, sa, bjs, sdid) -> a single
        point with CI whisker labeled "ATT".

      * Other linear estimations (regress, logit, probit, poisson, nbreg)
        -> horizontal forest plot of every coefficient (excluding _cons by
        default) with CI whiskers.

    Options:
        level(95)       Confidence level (default 95)
        ylim(lo hi)     Force y-axis range
        horizontal      Force horizontal layout (one row per coefficient)
        vertical        Force vertical layout (one column per coefficient)
        connect         Draw a connecting line between adjacent points
                        (default: on for event studies, off otherwise)
        zero_line(off)  Suppress the y=0 reference line
        ref_line(off)   Suppress the vertical reference-period line
        title("...")    Plot title
        xlabel/ylabel   Axis labels
        name("f.png")   Save to file instead of opening interactively
    """
    plt = _check_matplotlib()
    parsed = parse_command_line(rest)
    opts = parsed["options"]

    if not state.e_results:
        console.print("[red]coefplot: no estimation results in memory.[/red]")
        console.print("[dim]Run regress, didregress, logit, probit, etc. first.[/dim]")
        return

    e = state.e_results
    cmd = e.get("cmd", "")
    method = e.get("method", "")

    # ── Decide which plot type to produce ──────────────────────────────
    coefs = e.get("coefficients")  # list of dicts (event study / dynamic)
    is_event_study = bool(coefs) and isinstance(coefs, list) and len(coefs) > 1

    if is_event_study:
        _coefplot_event_study(coefs, e, parsed, plt, state, console)
        return

    # Scalar ATT (DiD without dynamics)
    if cmd == "didregress":
        _coefplot_scalar_att(e, parsed, plt, state, console)
        return

    # Generic estimation (regress, logit, etc.) — read the b/se vectors
    _coefplot_generic(e, parsed, plt, state, console)


def _coefplot_event_study(coefs, e, parsed, plt, state, console):
    """Forest plot of per-period effects with CI whiskers."""
    import numpy as np

    opts = parsed["options"]
    level = float(opts.get("level", 95))
    z = 1.96 if abs(level - 95) < 0.5 else _z_for_level(level)

    periods = [c["period"] for c in coefs]
    effects = [c["effect"] for c in coefs]
    lows  = [c.get("ci_lb") if c.get("ci_lb") is not None
             else (c["effect"] - z * c["se"] if c.get("se") is not None else c["effect"])
             for c in coefs]
    highs = [c.get("ci_ub") if c.get("ci_ub") is not None
             else (c["effect"] + z * c["se"] if c.get("se") is not None else c["effect"])
             for c in coefs]
    phases = [c.get("phase", "") for c in coefs]

    fig, ax = plt.subplots(figsize=(9, 5))

    # Color points by phase: pre = grey, post = blue, ref = highlighted
    colors = {"pre": "#7F8C8D", "post": "#2E86AB", "ref": "#E74C3C"}
    for i, p in enumerate(periods):
        color = colors.get(phases[i], "#34495E")
        ax.errorbar(p, effects[i],
                    yerr=[[effects[i] - lows[i]], [highs[i] - effects[i]]],
                    fmt="o", color=color, ecolor=color,
                    capsize=4, markersize=7, linewidth=1.5)

    # Optional: connect adjacent points with a line (default ON for event studies)
    connect_default = "on"
    if opts.get("connect") in ("off", "0", "no", "false"):
        connect_default = "off"
    if connect_default == "on":
        # Sort by period for the line
        order = sorted(range(len(periods)),
                       key=lambda i: (periods[i] is None, periods[i]))
        ax.plot([periods[i] for i in order],
                [effects[i] for i in order],
                color="#95A5A6", linestyle="-", alpha=0.4, zorder=0)

    # Zero line (default ON)
    if opts.get("zero_line", "on") not in ("off", "0", "no", "false"):
        ax.axhline(0, color="#2C3E50", linestyle="--", linewidth=1, alpha=0.6)

    # Vertical reference line at the omitted period (default ON)
    ref_period = e.get("reference_period")
    if (ref_period is not None
            and opts.get("ref_line", "on") not in ("off", "0", "no", "false")):
        ax.axvline(ref_period, color="#E74C3C", linestyle=":", linewidth=1, alpha=0.5)
        # Annotate
        try:
            y_text = ax.get_ylim()[1]
            ax.text(ref_period, y_text, "  reference",
                    fontsize=8, color="#E74C3C", verticalalignment="top")
        except Exception:
            pass

    if "ylim" in opts:
        try:
            lo, hi = [float(x) for x in opts["ylim"].split()[:2]]
            ax.set_ylim(lo, hi)
        except (ValueError, IndexError):
            pass

    method = e.get("method", "")
    default_title = f"Event-study coefficients ({method})" if method else "Event-study coefficients"
    parsed["options"].setdefault("xlabel", "Period")
    parsed["options"].setdefault("ylabel", f"Effect ({int(level)}% CI)")
    _apply_common_options(ax, plt, parsed, default_title=default_title)
    _show_or_save(plt, parsed, state, console)


def _coefplot_scalar_att(e, parsed, plt, state, console):
    """Single-point CI plot for scalar-ATT DiD methods."""
    import math

    opts = parsed["options"]
    level = float(opts.get("level", 95))
    z = 1.96 if abs(level - 95) < 0.5 else _z_for_level(level)

    att = e.get("att") or e.get("overall_att") or e.get("avg_att")
    se  = e.get("se")  or e.get("overall_se")  or e.get("avg_se")
    if att is None:
        console.print("[red]coefplot: no scalar ATT in e().[/red]")
        return

    # SE may be None, NaN, or 0 (some estimators return 0 when variance
    # cannot be computed — e.g. SDID without enough control units for the
    # placebo variance estimator). In all three cases we should NOT draw
    # zero-width whiskers, which silently misleads the reader.
    se_missing = (
        se is None
        or (isinstance(se, float) and (math.isnan(se) or se == 0.0))
    )

    fig, ax = plt.subplots(figsize=(7, 3.5))

    if se_missing:
        # Point only, no whisker
        ax.plot(att, 0, "o", color="#2E86AB", markersize=10)
        # Annotation explaining why no CI
        ax.annotate(
            "SE not estimated",
            xy=(att, 0), xytext=(8, -12),
            textcoords="offset points",
            fontsize=9, color="#C0392B",
        )
    else:
        ci_lb = e.get("ci_lb")
        ci_ub = e.get("ci_ub")
        if ci_lb is None or ci_ub is None:
            ci_lb = att - z * se
            ci_ub = att + z * se
        ax.errorbar(att, 0,
                    xerr=[[att - ci_lb], [ci_ub - att]],
                    fmt="o", color="#2E86AB", ecolor="#2E86AB",
                    capsize=6, markersize=10, linewidth=2)

    if opts.get("zero_line", "on") not in ("off", "0", "no", "false"):
        ax.axvline(0, color="#E74C3C", linestyle="--", linewidth=1, alpha=0.6)

    method = e.get("method", "")
    ax.set_yticks([0])
    ax.set_yticklabels([f"ATT ({method})" if method else "ATT"])

    # Give the y-axis breathing room so the annotation is visible
    ax.set_ylim(-0.5, 0.5)

    parsed["options"].setdefault("xlabel", f"Effect ({int(level)}% CI)" if not se_missing
                                            else "Effect (no CI)")
    _apply_common_options(ax, plt, parsed,
                          default_title=f"DiD coefficient: {method}" if method else "DiD coefficient")
    _show_or_save(plt, parsed, state, console)


def _coefplot_generic(e, parsed, plt, state, console):
    """Horizontal forest plot of all coefficients from a regress/logit/poisson/etc."""
    import numpy as np

    opts = parsed["options"]
    level = float(opts.get("level", 95))
    z = 1.96 if abs(level - 95) < 0.5 else _z_for_level(level)

    b = e.get("b")
    se = e.get("se")
    if b is None or se is None:
        console.print("[red]coefplot: no coefficient vector found in e().[/red]")
        console.print("[dim]coefplot supports regress, logit, probit, poisson, nbreg, didregress.[/dim]")
        return

    # b and se are pandas Series. Drop _cons by default (clutter).
    drop_cons = opts.get("drop_cons", "on") not in ("off", "0", "no", "false")
    keep_names = [n for n in b.index if not (drop_cons and n.lower() in ("const", "_cons"))]
    if not keep_names:
        keep_names = list(b.index)

    effects = [float(b[n]) for n in keep_names]
    ses     = [float(se[n]) for n in keep_names]
    lows    = [eff - z * s for eff, s in zip(effects, ses)]
    highs   = [eff + z * s for eff, s in zip(effects, ses)]

    n = len(keep_names)
    fig, ax = plt.subplots(figsize=(8, max(3, 0.5 * n + 1.5)))
    ys = list(range(n))[::-1]   # top-down
    for y, eff, lo, hi in zip(ys, effects, lows, highs):
        ax.errorbar(eff, y, xerr=[[eff - lo], [hi - eff]],
                    fmt="o", color="#2E86AB", ecolor="#2E86AB",
                    capsize=4, markersize=7, linewidth=1.5)

    ax.set_yticks(ys)
    ax.set_yticklabels(keep_names)
    if opts.get("zero_line", "on") not in ("off", "0", "no", "false"):
        ax.axvline(0, color="#E74C3C", linestyle="--", linewidth=1, alpha=0.6)

    parsed["options"].setdefault("xlabel", f"Coefficient ({int(level)}% CI)")
    _apply_common_options(ax, plt, parsed,
                          default_title=f"Coefficient plot ({e.get('cmd', 'estimation')})")
    _show_or_save(plt, parsed, state, console)


def _z_for_level(level: float) -> float:
    """Return the two-sided z critical value for a confidence level."""
    try:
        from scipy.stats import norm
        return float(norm.ppf(0.5 + level / 200.0))
    except ImportError:
        # Common levels — fall back to lookup
        table = {90: 1.6449, 95: 1.96, 99: 2.5758}
        return table.get(int(level), 1.96)
