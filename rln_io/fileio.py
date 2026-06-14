"""
Multi-format data loader and saver.

Supported input:  .dta, .csv, .tsv, .xlsx, .xls, .dbf, .RData, .rds, .html
Supported output: .dta, .csv, .xlsx, .txt/.log
"""

import os
import pandas as pd
from typing import Optional, Tuple, Dict, Any
from rich.console import Console

console = Console()


# ──────────────────────────────────────────────
#  LOADERS
# ──────────────────────────────────────────────

def detect_format(filepath: str) -> str:
    """Detect file format from extension."""
    ext = os.path.splitext(filepath)[1].lower()
    format_map = {
        ".dta": "dta",
        ".csv": "csv",
        ".tsv": "tsv",
        ".xlsx": "excel",
        ".xls": "excel",
        ".dbf": "dbf",
        ".rdata": "rdata",
        ".rds": "rds",
        ".rda": "rdata",
        ".html": "html",
        ".htm": "html",
        ".parquet": "parquet",
        ".pq": "parquet",
        ".json": "json",
    }
    fmt = format_map.get(ext)
    if fmt is None:
        raise ValueError(
            f"Unknown file format '{ext}'. "
            f"Supported: {', '.join(format_map.keys())}"
        )
    return fmt


def load_data(filepath: str, **kwargs) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Load data from file. Returns (DataFrame, metadata_dict).
    metadata_dict may contain: variable_labels, value_labels, source_format
    """
    # URL support: any http(s) URL is downloaded to a local tempfile and
    # then dispatched to the normal format-specific loader.
    #   use "https://example.com/data.dta", clear
    #   use "https://example.com/data.csv", clear
    #   use "https://example.com/data.parquet", clear
    # HTML tables pass through directly (pandas handles URL fetch for those).
    if filepath.startswith(("http://", "https://")):
        ext = os.path.splitext(filepath.split("?", 1)[0])[1].lower().lstrip(".")
        # HTML: pandas.read_html can fetch the URL itself
        if ext in ("html", "htm", "") and "html" in detect_format_hint(filepath):
            return _load_html(filepath, **kwargs)
        # Everything else: download to a tempfile, then load normally
        filepath = _download_url_to_tempfile(filepath)

    filepath = os.path.expanduser(filepath)

    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")

    fmt = detect_format(filepath)
    loader = {
        "dta": _load_dta,
        "csv": _load_csv,
        "tsv": _load_tsv,
        "excel": _load_excel,
        "dbf": _load_dbf,
        "rdata": _load_rdata,
        "rds": _load_rds,
        "html": _load_html,
        "parquet": _load_parquet,
        "json": _load_json,
    }[fmt]

    return loader(filepath, **kwargs)


def detect_format_hint(url: str) -> str:
    """Quick-and-dirty hint from a URL path (used before download)."""
    path = url.split("?", 1)[0].lower()
    if path.endswith((".html", ".htm")):
        return "html"
    if path.endswith(".csv"):  return "csv"
    if path.endswith(".tsv"):  return "tsv"
    if path.endswith(".dta"):  return "dta"
    if path.endswith(".parquet") or path.endswith(".pq"): return "parquet"
    if path.endswith(".json"): return "json"
    if path.endswith(".xlsx") or path.endswith(".xls"): return "excel"
    if path.endswith(".dbf"):  return "dbf"
    if path.endswith((".rdata", ".rda")): return "rdata"
    if path.endswith(".rds"):  return "rds"
    # Default: assume HTML (tables on web pages have no file extension)
    return "html"


def _download_url_to_tempfile(url: str, chunk_size: int = 1 << 16) -> str:
    """Stream a URL to a temp file. Returns the local path.

    The temp file keeps the original extension so downstream format
    detection works. We deliberately use stdlib urllib (no `requests`
    dependency) so this works in the lite build tier too.
    """
    import tempfile, urllib.request, urllib.error

    # Preserve extension so detect_format() works on the download
    ext = os.path.splitext(url.split("?", 1)[0])[1] or ".dat"
    fd, local_path = tempfile.mkstemp(suffix=ext, prefix="rln_dl_")
    os.close(fd)

    req = urllib.request.Request(
        url,
        headers={"User-Agent": "Rln/1.1 (+https://github.com/akirawisnu)"},
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp, open(local_path, "wb") as out:
            while True:
                chunk = resp.read(chunk_size)
                if not chunk:
                    break
                out.write(chunk)
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"HTTP {e.code} fetching {url}: {e.reason}")
    except urllib.error.URLError as e:
        raise RuntimeError(f"Network error fetching {url}: {e.reason}")

    return local_path


def _load_dta(filepath: str, **kwargs) -> Tuple[pd.DataFrame, dict]:
    """Load other statistical tools .dta file, preserving variable and value labels."""
    meta = {"source_format": "dta"}

    try:
        reader = pd.io.stata.StataReader(filepath)
        df = reader.read()

        # Extract variable labels
        try:
            meta["variable_labels"] = reader.variable_labels()
        except Exception:
            meta["variable_labels"] = {}

        # Extract value labels
        try:
            meta["value_labels"] = reader.value_labels()
        except Exception:
            meta["value_labels"] = {}

        try:
            reader.close()
        except Exception:
            pass

    except Exception:
        # Fallback: simple read
        df = pd.read_stata(filepath)
        meta["variable_labels"] = {}
        meta["value_labels"] = {}

    return df, meta


def _load_csv(filepath: str, **kwargs) -> Tuple[pd.DataFrame, dict]:
    """Load CSV with auto-detection of delimiter and encoding."""
    # Try common encodings
    for encoding in ["utf-8", "latin-1", "cp1252"]:
        try:
            df = pd.read_csv(filepath, encoding=encoding, **kwargs)
            return df, {"source_format": "csv"}
        except UnicodeDecodeError:
            continue
    raise ValueError(f"Cannot decode {filepath}. Try specifying encoding.")


def _load_tsv(filepath: str, **kwargs) -> Tuple[pd.DataFrame, dict]:
    """Load TSV file."""
    df = pd.read_csv(filepath, sep="\t", **kwargs)
    return df, {"source_format": "tsv"}


def _load_excel(filepath: str, **kwargs) -> Tuple[pd.DataFrame, dict]:
    """Load Excel file. If multiple sheets, asks user to pick.

    BUGFIX (Gemini #6): pd.ExcelFile opens a file handle that is NOT
    released until the object is garbage-collected. On Windows this causes
    PermissionError if the user tries to delete or move the file right
    after loading. Use `with` so the handle closes immediately.
    """
    sheet = kwargs.pop("sheet", None)

    if sheet is None:
        # Context-managed open so the underlying file handle is released
        # deterministically — otherwise Windows holds a lock on the file.
        with pd.ExcelFile(filepath) as xl:
            sheets = xl.sheet_names
        if len(sheets) == 1:
            sheet = sheets[0]
        else:
            console.print(f"[yellow]Multiple sheets found:[/yellow]")
            for i, s in enumerate(sheets, 1):
                console.print(f"  {i}. {s}")
            console.print(f"\n[dim]Loading first sheet: '{sheets[0]}'[/dim]")
            console.print(f"[dim]Use: import excel \"file\", sheet(\"name\") to pick another[/dim]")
            sheet = sheets[0]

    df = pd.read_excel(filepath, sheet_name=sheet, **kwargs)
    return df, {"source_format": "excel", "sheet": sheet}


def _load_dbf(filepath: str, **kwargs) -> Tuple[pd.DataFrame, dict]:
    """Load DBF file."""
    try:
        from dbfread import DBF
    except ImportError:
        raise ImportError("Install dbfread: pip install dbfread")

    table = DBF(filepath, load=True)
    df = pd.DataFrame(iter(table))
    return df, {"source_format": "dbf"}


def _load_rdata(filepath: str, **kwargs) -> Tuple[pd.DataFrame, dict]:
    """Load R .RData/.rda file. Takes the first data frame found."""
    try:
        import pyreadr
    except ImportError:
        raise ImportError("Install pyreadr: pip install pyreadr")

    result = pyreadr.read_r(filepath)
    if not result:
        raise ValueError(f"No data frames found in {filepath}")

    names = list(result.keys())
    if len(names) > 1:
        console.print(f"[yellow]Multiple objects found: {', '.join(names)}[/yellow]")
        console.print(f"[dim]Loading first: '{names[0]}'[/dim]")

    df = result[names[0]]
    return df, {"source_format": "rdata", "object_name": names[0]}


def _load_rds(filepath: str, **kwargs) -> Tuple[pd.DataFrame, dict]:
    """Load R .rds file."""
    try:
        import pyreadr
    except ImportError:
        raise ImportError("Install pyreadr: pip install pyreadr")

    result = pyreadr.read_r(filepath)
    names = list(result.keys())
    df = result[names[0]]
    return df, {"source_format": "rds"}


def _load_html(filepath_or_url: str, **kwargs) -> Tuple[pd.DataFrame, dict]:
    """Load HTML table(s). From URL or local file."""
    table_index = kwargs.pop("table_index", None)

    try:
        tables = pd.read_html(filepath_or_url)
    except Exception as e:
        raise ValueError(f"Cannot parse HTML tables from {filepath_or_url}: {e}")

    if not tables:
        raise ValueError(f"No tables found in {filepath_or_url}")

    if table_index is not None:
        if table_index < 0 or table_index >= len(tables):
            raise ValueError(f"Table index {table_index} out of range (found {len(tables)} tables)")
        df = tables[table_index]
    else:
        if len(tables) > 1:
            console.print(f"[yellow]Found {len(tables)} tables. Loading the largest one.[/yellow]")
            console.print(f"[dim]Use: import html \"url\", table(N) to pick a specific table (0-based)[/dim]")
            df = max(tables, key=lambda t: t.shape[0] * t.shape[1])
        else:
            df = tables[0]

    return df, {"source_format": "html"}


def _load_parquet(filepath: str, **kwargs) -> Tuple[pd.DataFrame, dict]:
    """Load Parquet file. Prefer polars — its Rust reader needs no pyarrow, so
    this works on platforms without pyarrow (e.g. Android). When pyarrow is
    absent, convert polars -> pandas column-by-column via numpy instead of
    .to_pandas() (which itself requires pyarrow)."""
    try:
        import polars as pl
    except ImportError:
        pl = None
    if pl is not None:
        pdf = pl.read_parquet(filepath)
        try:
            df = pdf.to_pandas()  # fast path when pyarrow is installed
        except (ImportError, ModuleNotFoundError):
            df = pd.DataFrame(
                {col: pdf.get_column(col).to_numpy() for col in pdf.columns}
            )
        return df, {"source_format": "parquet", "engine": "polars"}
    df = pd.read_parquet(filepath, **kwargs)
    return df, {"source_format": "parquet"}


def _load_json(filepath: str, **kwargs) -> Tuple[pd.DataFrame, dict]:
    """Load JSON file."""
    df = pd.read_json(filepath, **kwargs)
    return df, {"source_format": "json"}


# ──────────────────────────────────────────────
#  SAVERS
# ──────────────────────────────────────────────

def save_data(df: pd.DataFrame, filepath: str, metadata: dict = None, **kwargs):
    """Save DataFrame to file. Format auto-detected from extension."""
    filepath = os.path.expanduser(filepath)
    ext = os.path.splitext(filepath)[1].lower()

    if ext == ".dta":
        _save_dta(df, filepath, metadata, **kwargs)
    elif ext == ".csv":
        df.to_csv(filepath, index=False, **kwargs)
    elif ext in (".xlsx", ".xls"):
        df.to_excel(filepath, index=False, **kwargs)
    elif ext in (".parquet", ".pq"):
        df.to_parquet(filepath, index=False, **kwargs)
    elif ext == ".json":
        df.to_json(filepath, orient="records", indent=2, **kwargs)
    elif ext in (".txt", ".log"):
        _save_text(df, filepath, **kwargs)
    else:
        raise ValueError(f"Unsupported output format: {ext}. Use .dta, .csv, .xlsx, or .txt")

    console.print(f"[green]File saved: {filepath}[/green]")
    console.print(f"[dim]({df.shape[0]:,} observations, {df.shape[1]} variables)[/dim]")


def _save_dta(df: pd.DataFrame, filepath: str, metadata: dict = None, **kwargs):
    """Save as other statistical tools .dta with labels if available."""
    write_kwargs = {}

    if metadata:
        var_labels = metadata.get("variable_labels", {})
        val_labels = metadata.get("value_labels", {})
        val_assignments = metadata.get("value_label_assignments", {})

        # Filter variable_labels to only include columns that exist
        if var_labels:
            write_kwargs["variable_labels"] = {
                k: v for k, v in var_labels.items() if k in df.columns
            }

        # Resolve value labels: pandas expects {column_name: {value: label}}
        if val_labels and val_assignments:
            resolved = {}
            for col, lbl_name in val_assignments.items():
                if col in df.columns and lbl_name in val_labels:
                    resolved[col] = val_labels[lbl_name]
            if resolved:
                write_kwargs["value_labels"] = resolved

    # Handle other statistical tools column name restrictions (max 32 chars, no spaces)
    df_copy = df.copy()
    rename_map = {}
    for col in df_copy.columns:
        new_name = col
        new_name = new_name.replace(" ", "_")
        if len(new_name) > 32:
            new_name = new_name[:32]
        if new_name != col:
            rename_map[col] = new_name
    if rename_map:
        df_copy = df_copy.rename(columns=rename_map)
        console.print(f"[yellow]Renamed {len(rename_map)} columns for other statistical tools compatibility[/yellow]")

    try:
        df_copy.to_stata(filepath, write_index=False, version=118, **write_kwargs)
    except Exception as e:
        # Try without value labels
        try:
            write_kwargs.pop("value_labels", None)
            df_copy.to_stata(filepath, write_index=False, version=118, **write_kwargs)
        except Exception as e2:
            # Final fallback: convert non-Latin strings to ASCII approximation
            try:
                for col in df_copy.select_dtypes(include=["object"]).columns:
                    df_copy[col] = df_copy[col].apply(
                        lambda x: x.encode("ascii", "replace").decode("ascii") if isinstance(x, str) else x
                    )
                df_copy.to_stata(filepath, write_index=False, **write_kwargs)
            except Exception as e3:
                raise ValueError(
                    f"Cannot save to .dta: {e3}\n"
                    "Try: export delimited \"file.csv\" or save \"file.parquet\""
                )


def _save_text(df: pd.DataFrame, filepath: str, **kwargs):
    """Save as formatted text file."""
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(df.to_string(index=False))
