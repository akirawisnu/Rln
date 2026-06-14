"""
Command parser: tokenizes econometric input and routes to handlers.
"""

from rich.console import Console
from commands.state import AppState
from commands.parse_helpers import parse_command_line  # re-export for convenience


class CommandParser:
    """Parse and dispatch econometric commands."""

    def __init__(self, state: AppState, console: Console):
        self.state = state
        self.console = console
        self._handlers = None

    def _load_handlers(self):
        """Lazy-load command handlers to avoid circular imports."""
        from commands import (data_io, explore, variables, dataops, utility,
                              advanced, extras, rln_cmds, scripting,
                              estimation, estimation_glm, diagnostics,
                              panel, charts, nlp, nlp_extend, lrtm,
                              quantile_cmds)

        self._handlers = {
            "use": data_io.cmd_use,
            "import": data_io.cmd_import,
            "save": data_io.cmd_save,
            "export": data_io.cmd_export,
            "log": data_io.cmd_log,
            "browse": explore.cmd_browse,
            "describe": explore.cmd_describe,
            "desc": explore.cmd_describe,
            "d": explore.cmd_describe,
            "codebook": explore.cmd_codebook,
            "list": explore.cmd_list,
            "l": explore.cmd_list,
            "tabulate": explore.cmd_tabulate,
            "tab": explore.cmd_tabulate,
            "tabstat": explore.cmd_tabstat,
            "contract": explore.cmd_contract,
            "summarize": explore.cmd_summarize,
            "sum": explore.cmd_summarize,
            "su": explore.cmd_summarize,
            "count": explore.cmd_count,
            "generate": variables.cmd_generate,
            "gen": variables.cmd_generate,
            "g": variables.cmd_generate,
            "replace": variables.cmd_replace,
            "rename": variables.cmd_rename,
            "drop": variables.cmd_drop,
            "keep": variables.cmd_keep,
            "label": variables.cmd_label,
            "destring": variables.cmd_destring,
            "tostring": variables.cmd_tostring,
            "encode": variables.cmd_encode,
            "order": variables.cmd_order,
            "recode": variables.cmd_recode,
            "reshape": variables.cmd_reshape,
            "sort": dataops.cmd_sort,
            "gsort": dataops.cmd_gsort,
            "duplicates": dataops.cmd_duplicates,
            "append": dataops.cmd_append,
            "merge": dataops.cmd_merge,
            "fuzzmerge": dataops.cmd_fuzzmerge,
            "fuzzymerge": dataops.cmd_fuzzmerge,
            "assert": advanced.cmd_assert,
            "capture": advanced.cmd_capture,
            "cap": advanced.cmd_capture,
            "preserve": advanced.cmd_preserve,
            "restore": advanced.cmd_restore,
            "collapse": advanced.cmd_collapse,
            "egen": advanced.cmd_egen,
            "notes": advanced.cmd_notes,
            "note": advanced.cmd_notes,
            "fillin": extras.cmd_fillin,
            "cross": extras.cmd_cross,
            "sample": extras.cmd_sample,
            "isid": extras.cmd_isid,
            "levelsof": extras.cmd_levelsof,
            "distinct": extras.cmd_distinct,
            "display": extras.cmd_display,
            "di": extras.cmd_display,
            "clonevar": extras.cmd_clonevar,
            "split": extras.cmd_split,
            "compress": extras.cmd_compress,
            "ssc": rln_cmds.cmd_ssc,
            "pip": rln_cmds.cmd_ssc,
            "doedit": rln_cmds.cmd_doedit,
            "edit": rln_cmds.cmd_doedit,
            "python": rln_cmds.cmd_python,
            "py": rln_cmds.cmd_python,
            "copy": rln_cmds.cmd_copy,
            "local": scripting.cmd_local,
            "global": scripting.cmd_global,
            "foreach": scripting.cmd_foreach,
            "forvalues": scripting.cmd_forvalues,
            "forval": scripting.cmd_forvalues,
            "quietly": scripting.cmd_quietly,
            "qui": scripting.cmd_quietly,
            "by": scripting.cmd_by,
            "bysort": scripting.cmd_bysort,
            "bys": scripting.cmd_bysort,
            "return": scripting.cmd_return,
            "ereturn": scripting.cmd_ereturn,
            "regress": estimation.cmd_regress,
            "reg": estimation.cmd_regress,
            "predict": estimation.cmd_predict,
            "test": estimation.cmd_test,
            "correlate": estimation.cmd_correlate,
            "corr": estimation.cmd_correlate,
            "pwcorr": estimation.cmd_pwcorr,
            "ttest": estimation.cmd_ttest,
            "logit": estimation_glm.cmd_logit,
            "probit": estimation_glm.cmd_probit,
            "poisson": estimation_glm.cmd_poisson,
            "nbreg": estimation_glm.cmd_nbreg,
            "tobit": estimation_glm.cmd_tobit,
            "ivregress": estimation_glm.cmd_ivregress,
            "vif": diagnostics.cmd_vif,
            "estat": diagnostics.cmd_estat,
            "dwstat": diagnostics.cmd_dwstat,
            "xtserial": diagnostics.cmd_xtserial,
            "pctile": quantile_cmds.cmd_pctile,
            "xtile": quantile_cmds.cmd_xtile,
            "centile": quantile_cmds.cmd_centile,
            "winsor2": quantile_cmds.cmd_winsor2,
            "winsorize": quantile_cmds.cmd_winsor2,
            "xtset": panel.cmd_xtset,
            "xtreg": panel.cmd_xtreg,
            "didregress": panel.cmd_didregress,
            "diff": panel.cmd_didregress,
            "lincom": panel.cmd_lincom,
            "margins": panel.cmd_margins,
            "histogram": charts.cmd_histogram,
            "hist": charts.cmd_histogram,
            "kdensity": charts.cmd_kdensity,
            "scatter": charts.cmd_scatter,
            "graph": charts.cmd_graph,
            "twoway": charts.cmd_twoway,
            "tw": charts.cmd_twoway,
            "marginsplot": charts.cmd_marginsplot,
            "coefplot": charts.cmd_coefplot,
            "hf": nlp_extend.cmd_nlp,
            "nlp": nlp_extend.cmd_nlp,
            "lrtm": lrtm.cmd_lrtm,
            "help": utility.cmd_help,
            "clear": utility.cmd_clear,
            "pwd": utility.cmd_pwd,
            "cd": utility.cmd_cd,
            "dir": utility.cmd_dir,
            "ls": utility.cmd_dir,
            "set": utility.cmd_set,
            "memory": utility.cmd_memory,
            "do": utility.cmd_do,
        }

    def execute(self, raw_input: str, reraise: bool = False):
        """Parse and execute a command line.

        Args:
            raw_input: the raw command to execute.
            reraise:   if True, propagate exceptions to the caller after
                       logging and setting state._rc. This is how `capture`
                       and the script file runner detect errors. By default
                       (False), exceptions are caught and only printed — the
                       interactive REPL stays alive.

        Side effects:
            Always sets `state._rc` to 0 on success, non-zero on failure,
            so that `display _rc` and `if _rc != 0 { ... }` work correctly.
        """
        if self._handlers is None:
            self._load_handlers()

        raw_input = raw_input.strip()
        if not raw_input:
            self.state._rc = 0
            return

        # Expand local macros
        from commands.scripting import expand_macros
        raw_input = expand_macros(raw_input, self.state)

        self.state.write_log(f". {raw_input}")

        parts = raw_input.split(None, 1)
        cmd_word = parts[0].lower()
        rest = parts[1] if len(parts) > 1 else ""

        if cmd_word.endswith(","):
            cmd_word = cmd_word[:-1]
            rest = ", " + rest

        handler = self._handlers.get(cmd_word)
        if handler is None:
            msg = f"Unknown command: '{cmd_word}'"
            self.state._rc = 199  # rc for unknown command
            if reraise:
                raise KeyError(msg)
            self.console.print(f"[red]{msg}[/red]")
            self.console.print("[dim]Type 'help' for available commands.[/dim]")
            return

        try:
            handler(rest, self.state, self.console)
            # `capture` runs an inner command, captures its rc, and returns
            # successfully. The outer state._rc should remain whatever
            # `capture` stored, not be reset to 0 here.
            if cmd_word != "capture":
                self.state._rc = 0
        except Exception as e:
            # Always record the return code so `display _rc` works.
            # Rln uses rc=1 for generic failures (other statistical tools uses command-specific
            # codes but rc!=0 is what callers actually branch on).
            self.state._rc = getattr(e, "rc", 1)
            self.state.write_log(f"ERROR: {e}")
            if reraise:
                raise
            self.console.print(f"[red]Error: {e}[/red]")
