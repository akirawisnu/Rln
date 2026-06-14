"""
HuggingFace NLP integration for Rln.
Wraps transformers pipeline API into familiar econometric syntax.

All models are cached in rln/hf_models/ for portability.
Transfer the entire rln/ folder between devices for offline use.

Commands:
  hf classify var, labels("lab1 lab2 lab3") [generate(newvar) model(name) batch(N)]
  hf sentiment var [, generate(newvar) model(name)]  
  hf summarize var [, generate(newvar) maxlen(N) minlen(N) model(name)]
  hf translate var, from(lang) to(lang) [generate(newvar) model(name)]
  hf ner var [, generate(newvar) model(name)]
  hf embed var [, generate(stub) model(name) dims(N)]
  hf models [task]     — List recommended models
  hf download model    — Pre-download a model for offline use
  hf cache             — Show cache location and size
"""

import re
import os
import sys
import numpy as np
import pandas as pd
from rich.console import Console
from rich.table import Table

from commands.state import AppState
from commands.parse_helpers import parse_command_line
from commands.expression import eval_condition


def _get_cache_dir():
    """
    Get the local HuggingFace model cache directory.

    In a PyInstaller portable build, RLN_PORTABLE_ROOT is set by the
    runtime hook to the folder beside rln-full.exe / rln-offline.exe,
    so models stay editable and movable with the portable app folder.
    In source mode, fall back to rln/hf_models/ relative to this file.
    """
    portable_root = os.environ.get("RLN_PORTABLE_ROOT")
    if portable_root:
        cache_dir = os.path.join(portable_root, "hf_models")
    else:
        # Find project root: go up from commands/ to rln/
        this_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(this_dir)
        cache_dir = os.path.join(project_root, "hf_models")
    os.makedirs(cache_dir, exist_ok=True)
    return cache_dir


def _setup_hf_cache():
    """
    Set HuggingFace environment variables to use local cache.
    Must be called before any transformers import.
    """
    cache_dir = _get_cache_dir()
    os.environ["TRANSFORMERS_CACHE"] = cache_dir
    os.environ["HF_HOME"] = cache_dir
    os.environ["HF_DATASETS_CACHE"] = os.path.join(cache_dir, "datasets")
    os.environ["SENTENCE_TRANSFORMERS_HOME"] = os.path.join(cache_dir, "sentence_transformers")
    # Suppress Windows symlink warnings
    os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
    # Enable offline mode if models are cached (prevents network checks)
    # User can override with: set hf_offline off
    if os.environ.get("RLN_HF_ONLINE") != "1":
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
    return cache_dir


def _verify_model_ready(model_name, cache_dir):
    """
    Check if a model appears to be fully downloaded.
    Returns True if model directory exists and has config.json.
    """
    safe_name = model_name.replace("/", "--")
    model_dir = os.path.join(cache_dir, f"models--{safe_name}")

    if not os.path.isdir(model_dir):
        return False

    # Check for snapshot with config.json
    snapshots_dir = os.path.join(model_dir, "snapshots")
    if os.path.isdir(snapshots_dir):
        for snap in os.listdir(snapshots_dir):
            snap_path = os.path.join(snapshots_dir, snap)
            if os.path.isdir(snap_path):
                config = os.path.join(snap_path, "config.json")
                if os.path.exists(config):
                    return True

    # Check blobs dir has files
    blobs_dir = os.path.join(model_dir, "blobs")
    if os.path.isdir(blobs_dir) and len(os.listdir(blobs_dir)) > 0:
        return True

    return False


def _check_transformers():
    """Check if transformers is installed and set up local cache."""
    _setup_hf_cache()
    try:
        import transformers
        return transformers
    except ImportError:
        raise ImportError(
            "HuggingFace transformers required for NLP commands.\n"
            "Install with: ssc install transformers torch\n"
            "Or for CPU-only: ssc install transformers torch --extra-index-url https://download.pytorch.org/whl/cpu"
        )


def _get_pipeline(task, model_name=None, **kwargs):
    """Create a HuggingFace pipeline with local model cache and optimal device."""
    _setup_hf_cache()
    from transformers import pipeline
    cache_dir = _get_cache_dir()
    device = _detect_device()

    pipe_kwargs = {"model_kwargs": {"cache_dir": cache_dir}, **kwargs}
    if device != "cpu":
        pipe_kwargs["device"] = device

    if model_name:
        return pipeline(task, model=model_name, **pipe_kwargs)
    return pipeline(task, **pipe_kwargs)


def _detect_device():
    """Detect best available compute device."""
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_mem = torch.cuda.get_device_properties(0).total_mem / 1e9
            return 0  # GPU index for pipeline
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"  # Apple Silicon
    except ImportError:
        pass
    return "cpu"


def _get_device_info():
    """Get device info string for display."""
    try:
        import torch
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            mem = torch.cuda.get_device_properties(0).total_mem / 1e9
            return f"GPU: {name} ({mem:.1f} GB)"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "Apple Silicon (MPS)"
    except ImportError:
        pass

    # CPU info
    cpu_count = os.cpu_count() or 1
    return f"CPU ({cpu_count} cores)"


def _auto_batch_size(n_texts, model_type="default"):
    """Calculate optimal batch size based on device and dataset size."""
    try:
        import torch
        if torch.cuda.is_available():
            gpu_mem = torch.cuda.get_device_properties(0).total_mem / 1e9
            if gpu_mem >= 16:
                base = 64
            elif gpu_mem >= 8:
                base = 32
            elif gpu_mem >= 4:
                base = 16
            else:
                base = 8
        else:
            base = 8
    except ImportError:
        base = 8

    # Adjust for model type (translation/summarization need more memory)
    if model_type in ("translate", "summarize"):
        base = max(base // 2, 2)
    elif model_type == "embed":
        base = base * 2

    return min(base, n_texts)


def _process_batches(series, pipeline_fn, batch_size=32, console=None):
    """Process a text series in batches with progress reporting."""
    texts = series.fillna("").astype(str).tolist()
    results = []
    total = len(texts)

    import time
    t0 = time.time()

    for i in range(0, total, batch_size):
        batch = texts[i:i + batch_size]
        batch_results = pipeline_fn(batch)
        results.extend(batch_results)

        if console:
            done = min(i + batch_size, total)
            pct = done / total * 100
            elapsed = time.time() - t0
            rate = done / elapsed if elapsed > 0 else 0
            eta = (total - done) / rate if rate > 0 else 0
            console.print(f"[dim]  {done}/{total} ({pct:.0f}%) | {rate:.1f} texts/s | ETA: {eta:.0f}s[/dim]")

    elapsed = time.time() - t0
    if console and total > 0:
        console.print(f"[dim]  Completed in {elapsed:.1f}s ({total/elapsed:.1f} texts/s)[/dim]")

    return results


def _process_parallel(texts, process_fn, n_workers=None, batch_size=32, console=None):
    """
    Process texts using multiprocessing for CPU-bound tasks.
    Falls back to sequential if multiprocessing fails.
    """
    import time
    t0 = time.time()
    total = len(texts)

    if n_workers is None:
        n_workers = min(os.cpu_count() or 1, 4)

    # For small datasets, sequential is faster (no process spawn overhead)
    if total < 50 or n_workers <= 1:
        results = []
        for i in range(0, total, batch_size):
            batch = texts[i:i + batch_size]
            results.extend(process_fn(batch))
            if console:
                done = min(i + batch_size, total)
                console.print(f"[dim]  {done}/{total} ({done/total*100:.0f}%)[/dim]")
        return results

    # Split into chunks for parallel processing
    from concurrent.futures import ThreadPoolExecutor, as_completed

    chunk_size = max(total // n_workers, batch_size)
    chunks = [texts[i:i + chunk_size] for i in range(0, total, chunk_size)]

    if console:
        console.print(f"[dim]  Parallel: {len(chunks)} chunks across {n_workers} workers[/dim]")

    results_dict = {}
    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = {}
        for idx, chunk in enumerate(chunks):
            future = executor.submit(process_fn, chunk)
            futures[future] = idx

        for future in as_completed(futures):
            idx = futures[future]
            try:
                results_dict[idx] = future.result()
            except Exception as e:
                if console:
                    console.print(f"[yellow]  Worker {idx} error: {e}[/yellow]")
                results_dict[idx] = []

    # Reassemble in order
    results = []
    for idx in sorted(results_dict.keys()):
        results.extend(results_dict[idx])

    elapsed = time.time() - t0
    if console and total > 0:
        console.print(f"[dim]  Completed in {elapsed:.1f}s ({total/elapsed:.1f} texts/s)[/dim]")

    return results


# ──────────────────────────────────────────────
#  Main dispatcher
# ──────────────────────────────────────────────

def cmd_hf(rest: str, state: AppState, console: Console):
    """
    hf <task> <args>
    HuggingFace NLP commands. Tasks: classify, sentiment, summarize, translate, ner, embed, models, download
    """
    parts = rest.strip().split(None, 1)
    if not parts:
        console.print("[red]Syntax: hf classify|sentiment|summarize|translate|ner|embed|models|download ...[/red]")
        _show_hf_help(console)
        return

    task = parts[0].lower()
    sub_rest = parts[1] if len(parts) > 1 else ""

    # Show device info on first NLP call
    if not hasattr(state, '_hf_device_shown'):
        device_info = _get_device_info()
        console.print(f"[dim]Device: {device_info}[/dim]")
        state._hf_device_shown = True

    dispatch = {
        "classify": _hf_classify,
        "classification": _hf_classify,
        "zeroshot": _hf_classify,
        "sentiment": _hf_sentiment,
        "summarize": _hf_summarize,
        "summarise": _hf_summarize,
        "translate": _hf_translate,
        "ner": _hf_ner,
        "embed": _hf_embed,
        "embedding": _hf_embed,
        "models": _hf_models,
        "download": _hf_download,
        "cache": _hf_cache,
    }

    handler = dispatch.get(task)
    if handler:
        handler(sub_rest, state, console)
    else:
        console.print(f"[red]Unknown hf task: {task}[/red]")
        _show_hf_help(console)


def _show_hf_help(console):
    cache_dir = _get_cache_dir()
    console.print("[dim]Available tasks:[/dim]")
    console.print("  [cyan]hf classify[/cyan]   var, labels(\"pos neg neu\") generate(newvar)")
    console.print("  [cyan]hf sentiment[/cyan]  var [, generate(newvar)]")
    console.print("  [cyan]hf summarize[/cyan]  var [, generate(newvar) maxlen(150)]")
    console.print("  [cyan]hf translate[/cyan]  var, from(id) to(en) generate(newvar)")
    console.print("  [cyan]hf ner[/cyan]        var [, generate(newvar)]")
    console.print("  [cyan]hf embed[/cyan]      var [, generate(stub) dims(384)]")
    console.print("  [cyan]hf models[/cyan]     [task]")
    console.print("  [cyan]hf download[/cyan]   model_name")
    console.print("  [cyan]hf cache[/cyan]      Show cache location and size")
    console.print()
    console.print(f"[dim]Model cache: {cache_dir}[/dim]")
    console.print("[dim]All models are stored locally for portability.[/dim]")


# ──────────────────────────────────────────────
#  hf classify — Zero-shot text classification
# ──────────────────────────────────────────────

def _hf_classify(rest, state, console):
    """
    hf classify var, labels("label1 label2 label3") [generate(newvar) model(name) 
        multi batch(N) threshold(0.5) if condition]
    
    Zero-shot text classification using NLI models.
    Default model: facebook/bart-large-mnli (multilingual: joeddav/xlm-roberta-large-xnli)
    """
    state.require_data()
    _check_transformers()
    parsed = parse_command_line(rest)

    if not parsed["varlist"]:
        console.print("[red]Syntax: hf classify varname, labels(\"label1 label2 ...\")[/red]")
        return

    var = parsed["varlist"][0]
    if var not in state.data.columns:
        console.print(f"[red]Variable '{var}' not found[/red]")
        return

    labels_str = parsed["options"].get("labels") or parsed["options"].get("label")
    if not labels_str:
        console.print("[red]Must specify labels: hf classify var, labels(\"positive negative neutral\")[/red]")
        return

    labels = [l.strip() for l in labels_str.replace(",", " ").split() if l.strip()]
    model_name = parsed["options"].get("model", "facebook/bart-large-mnli")
    gen_var = parsed["options"].get("generate") or parsed["options"].get("gen")
    multi_label = "multi" in parsed["options"]
    batch_size = int(parsed["options"].get("batch", 0)) or _auto_batch_size(len(state.data), "classify")
    threshold = float(parsed["options"].get("threshold", 0.5))

    # Apply if condition
    df = state.data
    mask = None
    if parsed["if_cond"]:
        mask = eval_condition(parsed["if_cond"], df)
        working_series = df.loc[mask, var]
    else:
        working_series = df[var]

    console.print(f"[dim]Loading model: {model_name}...[/dim]")
    console.print(f"[dim]Labels: {', '.join(labels)}[/dim]")
    console.print(f"[dim]Classifying {len(working_series)} texts...[/dim]")

    from transformers import pipeline
    cache_dir = _get_cache_dir()
    classifier = pipeline("zero-shot-classification", model=model_name, model_kwargs={"cache_dir": cache_dir})

    def classify_batch(texts):
        return classifier(texts, candidate_labels=labels, multi_label=multi_label)

    results = _process_batches(working_series, classify_batch, batch_size, console)

    # Extract predicted labels and scores
    pred_labels = [r["labels"][0] for r in results]
    pred_scores = [r["scores"][0] for r in results]

    # Store results
    label_col = gen_var or f"{var}_class"
    score_col = f"{var}_score" if not gen_var else f"{gen_var}_score"

    if mask is not None:
        state.data[label_col] = ""
        state.data[score_col] = np.nan
        state.data.loc[mask, label_col] = pred_labels
        state.data.loc[mask, score_col] = pred_scores
    else:
        state.data[label_col] = pred_labels
        state.data[score_col] = pred_scores

    # Report
    console.print(f"\n[green]Classification complete:[/green]")
    console.print(f"  Generated: {label_col} (predicted label), {score_col} (confidence)")
    label_dist = state.data[label_col].value_counts()
    for lab, cnt in label_dist.items():
        if lab:
            console.print(f"    {lab:20s}  {cnt:,}")
    console.print(f"  Mean confidence: {state.data[score_col].mean():.3f}")

    state.mark_changed()


# ──────────────────────────────────────────────
#  hf sentiment — Sentiment analysis
# ──────────────────────────────────────────────

def _hf_sentiment(rest, state, console):
    """
    hf sentiment var [, generate(newvar) model(name) batch(N) if condition]
    
    Sentiment analysis. Returns label (POSITIVE/NEGATIVE) and score.
    Default model: nlptown/bert-base-multilingual-uncased-sentiment (multilingual, 1-5 stars)
    Alternative: distilbert-base-uncased-finetuned-sst-2-english (English, pos/neg)
    """
    state.require_data()
    _check_transformers()
    parsed = parse_command_line(rest)

    if not parsed["varlist"]:
        console.print("[red]Syntax: hf sentiment varname [, generate(newvar)][/red]")
        return

    var = parsed["varlist"][0]
    if var not in state.data.columns:
        console.print(f"[red]Variable '{var}' not found[/red]")
        return

    model_name = parsed["options"].get("model", "nlptown/bert-base-multilingual-uncased-sentiment")
    gen_var = parsed["options"].get("generate") or parsed["options"].get("gen")
    batch_size = int(parsed["options"].get("batch", 0)) or _auto_batch_size(len(state.data), "classify")

    df = state.data
    mask = None
    if parsed["if_cond"]:
        mask = eval_condition(parsed["if_cond"], df)
        working_series = df.loc[mask, var]
    else:
        working_series = df[var]

    console.print(f"[dim]Loading model: {model_name}...[/dim]")
    console.print(f"[dim]Analyzing sentiment for {len(working_series)} texts...[/dim]")

    from transformers import pipeline
    cache_dir = _get_cache_dir()
    sentiment = pipeline("sentiment-analysis", model=model_name, truncation=True, model_kwargs={"cache_dir": cache_dir})

    # Truncate long texts to 512 chars for safety
    def safe_batch(texts):
        truncated = [t[:512] if len(t) > 512 else t for t in texts]
        return sentiment(truncated)

    results = _process_batches(working_series, safe_batch, batch_size, console)

    labels = [r["label"] for r in results]
    scores = [r["score"] for r in results]

    label_col = gen_var or f"{var}_sentiment"
    score_col = f"{var}_sent_score" if not gen_var else f"{gen_var}_score"

    if mask is not None:
        state.data[label_col] = ""
        state.data[score_col] = np.nan
        state.data.loc[mask, label_col] = labels
        state.data.loc[mask, score_col] = scores
    else:
        state.data[label_col] = labels
        state.data[score_col] = scores

    console.print(f"\n[green]Sentiment analysis complete:[/green]")
    console.print(f"  Generated: {label_col}, {score_col}")
    for lab, cnt in state.data[label_col].value_counts().items():
        if lab:
            console.print(f"    {lab:20s}  {cnt:,}")

    state.mark_changed()


# ──────────────────────────────────────────────
#  hf summarize — Text summarization
# ──────────────────────────────────────────────

def _hf_summarize(rest, state, console):
    """
    hf summarize var [, generate(newvar) maxlen(150) minlen(30) model(name) batch(N)]
    
    Abstractive text summarization.
    Default model: facebook/bart-large-cnn
    Multilingual: csebuetnlp/mT5_multilingual_XLSum
    """
    state.require_data()
    _check_transformers()
    parsed = parse_command_line(rest)

    if not parsed["varlist"]:
        console.print("[red]Syntax: hf summarize varname [, generate(newvar) maxlen(150)][/red]")
        return

    var = parsed["varlist"][0]
    if var not in state.data.columns:
        console.print(f"[red]Variable '{var}' not found[/red]")
        return

    model_name = parsed["options"].get("model", "facebook/bart-large-cnn")
    gen_var = parsed["options"].get("generate") or parsed["options"].get("gen")
    max_length = int(parsed["options"].get("maxlen", parsed["options"].get("maxlength", 150)))
    min_length = int(parsed["options"].get("minlen", parsed["options"].get("minlength", 30)))
    batch_size = int(parsed["options"].get("batch", 0)) or _auto_batch_size(len(state.data), "summarize")

    df = state.data
    mask = None
    if parsed["if_cond"]:
        mask = eval_condition(parsed["if_cond"], df)
        working_series = df.loc[mask, var]
    else:
        working_series = df[var]

    console.print(f"[dim]Loading model: {model_name}...[/dim]")
    console.print(f"[dim]Summarizing {len(working_series)} texts (max_length={max_length})...[/dim]")

    cache_dir = _get_cache_dir()

    try:
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        import torch

        tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name, cache_dir=cache_dir)

        # Move model to GPU if available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        model.eval()
    except Exception as e:
        console.print(f"[red]Failed to load summarization model: {e}[/red]")
        console.print(f"[dim]Download first: hf download {model_name}[/dim]")
        return

    import time
    t0 = time.time()
    texts = working_series.fillna("").astype(str).tolist()
    summaries = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        # Pad very short texts
        batch = [t if len(t) > 50 else t + " . " * 20 for t in batch]

        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=1024)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=max_length,
                min_length=min_length,
                num_beams=4,
                length_penalty=2.0,
                early_stopping=True,
            )
        decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        summaries.extend(decoded)

        done = min(i + batch_size, len(texts))
        elapsed = time.time() - t0
        rate = done / elapsed if elapsed > 0 else 0
        console.print(f"[dim]  {done}/{len(texts)} ({done/len(texts)*100:.0f}%) | {rate:.1f} texts/s[/dim]")

    out_col = gen_var or f"{var}_summary"

    if mask is not None:
        state.data[out_col] = ""
        state.data.loc[mask, out_col] = summaries
    else:
        state.data[out_col] = summaries

    avg_len = np.mean([len(s) for s in summaries])
    console.print(f"\n[green]Summarization complete:[/green]")
    console.print(f"  Generated: {out_col}")
    console.print(f"  Average summary length: {avg_len:.0f} chars")

    state.mark_changed()


# ──────────────────────────────────────────────
#  hf translate — Translation
# ──────────────────────────────────────────────

def _hf_translate(rest, state, console):
    """
    hf translate var, from(source_lang) to(target_lang) [generate(newvar) model(name) batch(N)]
    
    Translate text between languages.
    Default model: Helsinki-NLP/opus-mt-{from}-{to}
    Multilingual fallback: facebook/nllb-200-distilled-600M
    
    Language codes: en, de, fr, es, id, zh, ja, ko, ar, ru, pt, etc.
    """
    state.require_data()
    _check_transformers()
    parsed = parse_command_line(rest)

    if not parsed["varlist"]:
        console.print("[red]Syntax: hf translate varname, from(en) to(de) [generate(newvar)][/red]")
        return

    var = parsed["varlist"][0]
    if var not in state.data.columns:
        console.print(f"[red]Variable '{var}' not found[/red]")
        return

    src_lang = parsed["options"].get("from") or parsed["options"].get("source")
    tgt_lang = parsed["options"].get("to") or parsed["options"].get("target")

    if not src_lang or not tgt_lang:
        console.print("[red]Must specify from() and to() languages[/red]")
        console.print("[dim]Example: hf translate text, from(id) to(en) generate(text_en)[/dim]")
        return

    model_name = parsed["options"].get("model")
    gen_var = parsed["options"].get("generate") or parsed["options"].get("gen")
    batch_size = int(parsed["options"].get("batch", 0)) or _auto_batch_size(len(state.data), "translate")

    # Auto-select model if not specified
    if not model_name:
        model_name = f"Helsinki-NLP/opus-mt-{src_lang}-{tgt_lang}"

    df = state.data
    mask = None
    if parsed["if_cond"]:
        mask = eval_condition(parsed["if_cond"], df)
        working_series = df.loc[mask, var]
    else:
        working_series = df[var]

    console.print(f"[dim]Loading model: {model_name}...[/dim]")
    console.print(f"[dim]Translating {len(working_series)} texts ({src_lang} -> {tgt_lang})...[/dim]")

    cache_dir = _get_cache_dir()
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name, cache_dir=cache_dir)
    except Exception as e1:
        console.print(f"[yellow]Model {model_name} failed: {str(e1)[:80]}[/yellow]")
        console.print(f"[dim]Trying facebook/nllb-200-distilled-600M...[/dim]")
        model_name = "facebook/nllb-200-distilled-600M"
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
            model = AutoModelForSeq2SeqLM.from_pretrained(model_name, cache_dir=cache_dir)
        except Exception as e2:
            console.print(f"[red]Translation failed: {e2}[/red]")
            console.print("[dim]Download a translation model first: hf download Helsinki-NLP/opus-mt-en-de[/dim]")
            return

    # NLLB uses special language codes
    nllb_lang_map = {
        "en": "eng_Latn", "de": "deu_Latn", "fr": "fra_Latn", "es": "spa_Latn",
        "id": "ind_Latn", "pt": "por_Latn", "nl": "nld_Latn", "it": "ita_Latn",
        "ru": "rus_Cyrl", "zh": "zho_Hans", "ja": "jpn_Jpan", "ko": "kor_Hang",
        "ar": "arb_Arab", "hi": "hin_Deva", "tr": "tur_Latn", "pl": "pol_Latn",
        "sv": "swe_Latn", "da": "dan_Latn", "no": "nob_Latn", "fi": "fin_Latn",
        "th": "tha_Thai", "vi": "vie_Latn", "ms": "zsm_Latn", "tl": "tgl_Latn",
    }

    is_nllb = "nllb" in model_name.lower()
    is_mbart = "mbart" in model_name.lower()

    # Set source lang for NLLB
    if is_nllb:
        src_code = nllb_lang_map.get(src_lang, src_lang)
        tgt_code = nllb_lang_map.get(tgt_lang, tgt_lang)
        tokenizer.src_lang = src_code
    elif is_mbart:
        mbart_map = {
            "en": "en_XX", "de": "de_DE", "fr": "fr_XX", "es": "es_XX",
            "id": "id_ID", "ru": "ru_RU", "zh": "zh_CN", "ja": "ja_XX",
            "ko": "ko_KR", "ar": "ar_AR", "pt": "pt_XX", "nl": "nl_XX",
        }
        src_code = mbart_map.get(src_lang, f"{src_lang}_XX")
        tgt_code = mbart_map.get(tgt_lang, f"{tgt_lang}_XX")
        tokenizer.src_lang = src_code

    # Translate in batches
    import torch, time
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    t0 = time.time()
    texts = working_series.fillna("").astype(str).tolist()
    translations = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=512)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            if is_nllb:
                forced_bos = tokenizer.convert_tokens_to_ids(tgt_code)
                outputs = model.generate(**inputs, forced_bos_token_id=forced_bos, max_length=512)
            elif is_mbart:
                forced_bos = tokenizer.convert_tokens_to_ids(tgt_code)
                outputs = model.generate(**inputs, forced_bos_token_id=forced_bos, max_length=512)
            else:
                outputs = model.generate(**inputs, max_length=512)

        decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        translations.extend(decoded)

        done = min(i + batch_size, len(texts))
        elapsed = time.time() - t0
        rate = done / elapsed if elapsed > 0 else 0
        console.print(f"[dim]  {done}/{len(texts)} ({done/len(texts)*100:.0f}%) | {rate:.1f} texts/s[/dim]")

    out_col = gen_var or f"{var}_{tgt_lang}"

    if mask is not None:
        state.data[out_col] = ""
        state.data.loc[mask, out_col] = translations
    else:
        state.data[out_col] = translations

    console.print(f"\n[green]Translation complete ({src_lang} -> {tgt_lang}):[/green]")
    console.print(f"  Generated: {out_col}")
    console.print(f"  {len(translations)} texts translated")
    console.print(f"  Model: {model_name}")

    state.mark_changed()


# ──────────────────────────────────────────────
#  hf ner — Named Entity Recognition
# ──────────────────────────────────────────────

def _hf_ner(rest, state, console):
    """
    hf ner var [, generate(newvar) model(name) batch(N)]
    
    Extract named entities from text.
    Default model: dslim/bert-base-NER
    Multilingual: Davlan/xlm-roberta-large-ner-hrl
    """
    state.require_data()
    _check_transformers()
    parsed = parse_command_line(rest)

    if not parsed["varlist"]:
        console.print("[red]Syntax: hf ner varname [, generate(newvar)][/red]")
        return

    var = parsed["varlist"][0]
    if var not in state.data.columns:
        console.print(f"[red]Variable '{var}' not found[/red]")
        return

    model_name = parsed["options"].get("model", "dslim/bert-base-NER")
    gen_var = parsed["options"].get("generate") or parsed["options"].get("gen")
    batch_size = int(parsed["options"].get("batch", 0)) or _auto_batch_size(len(state.data), "classify")

    df = state.data
    mask = None
    if parsed["if_cond"]:
        mask = eval_condition(parsed["if_cond"], df)
        working_series = df.loc[mask, var]
    else:
        working_series = df[var]

    console.print(f"[dim]Loading model: {model_name}...[/dim]")
    console.print(f"[dim]Extracting entities from {len(working_series)} texts...[/dim]")

    from transformers import pipeline
    cache_dir = _get_cache_dir()
    ner = pipeline("ner", model=model_name, aggregation_strategy="simple", model_kwargs={"cache_dir": cache_dir})

    def ner_batch(texts):
        return ner(texts)

    results = _process_batches(working_series, ner_batch, batch_size, console)

    # Flatten entities to comma-separated string
    entity_strings = []
    entity_counts = []
    for ents in results:
        if isinstance(ents, list):
            entity_strings.append(", ".join(
                f"{e.get('word','')}/{e.get('entity_group','')}" for e in ents
            ))
            entity_counts.append(len(ents))
        else:
            entity_strings.append("")
            entity_counts.append(0)

    out_col = gen_var or f"{var}_entities"
    count_col = f"{var}_nentities" if not gen_var else f"{gen_var}_count"

    if mask is not None:
        state.data[out_col] = ""
        state.data[count_col] = 0
        state.data.loc[mask, out_col] = entity_strings
        state.data.loc[mask, count_col] = entity_counts
    else:
        state.data[out_col] = entity_strings
        state.data[count_col] = entity_counts

    total_ents = sum(entity_counts)
    console.print(f"\n[green]NER complete:[/green]")
    console.print(f"  Generated: {out_col} (entities), {count_col} (count)")
    console.print(f"  Total entities found: {total_ents:,}")
    console.print(f"  Mean entities per text: {np.mean(entity_counts):.1f}")

    state.mark_changed()


# ──────────────────────────────────────────────
#  hf embed — Text embeddings
# ──────────────────────────────────────────────

def _hf_embed(rest, state, console):
    """
    hf embed var [, generate(stub) model(name) dims(N) batch(N)]
    
    Generate text embeddings (sentence vectors).
    Default model: sentence-transformers/all-MiniLM-L6-v2 (384 dims)
    Multilingual: sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
    
    Creates stub_1, stub_2, ..., stub_N variables.
    """
    state.require_data()
    _check_transformers()
    parsed = parse_command_line(rest)

    if not parsed["varlist"]:
        console.print("[red]Syntax: hf embed varname [, generate(stub) dims(384)][/red]")
        return

    var = parsed["varlist"][0]
    if var not in state.data.columns:
        console.print(f"[red]Variable '{var}' not found[/red]")
        return

    model_name = parsed["options"].get("model", "sentence-transformers/all-MiniLM-L6-v2")
    stub = parsed["options"].get("generate") or parsed["options"].get("gen") or f"{var}_emb"
    max_dims = int(parsed["options"].get("dims", 0))  # 0 = all dims
    batch_size = int(parsed["options"].get("batch", 0)) or _auto_batch_size(len(state.data), "embed")

    df = state.data
    mask = None
    if parsed["if_cond"]:
        mask = eval_condition(parsed["if_cond"], df)
        working_series = df.loc[mask, var]
    else:
        working_series = df[var]

    console.print(f"[dim]Loading model: {model_name}...[/dim]")
    console.print(f"[dim]Generating embeddings for {len(working_series)} texts...[/dim]")

    try:
        from sentence_transformers import SentenceTransformer
        st_cache = os.path.join(_get_cache_dir(), "sentence_transformers")
        model = SentenceTransformer(model_name, cache_folder=st_cache)
        texts = working_series.fillna("").astype(str).tolist()
        embeddings = model.encode(texts, batch_size=batch_size, show_progress_bar=True)
    except ImportError:
        # Fallback to transformers feature-extraction pipeline
        from transformers import pipeline
        extractor = pipeline("feature-extraction", model=model_name, model_kwargs={"cache_dir": _get_cache_dir()})
        texts = working_series.fillna("").astype(str).tolist()
        embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_emb = extractor(batch)
            # Average pool across tokens
            for emb in batch_emb:
                arr = np.array(emb)
                embeddings.append(arr.mean(axis=0) if arr.ndim > 1 else arr)
            done = min(i + batch_size, len(texts))
            console.print(f"[dim]  Processing: {done}/{len(texts)}[/dim]")
        embeddings = np.array(embeddings)

    n_dims = embeddings.shape[1] if embeddings.ndim > 1 else len(embeddings[0])
    if max_dims > 0 and max_dims < n_dims:
        embeddings = embeddings[:, :max_dims]
        n_dims = max_dims

    # Store as separate columns
    for d in range(n_dims):
        col_name = f"{stub}_{d+1}"
        if mask is not None:
            state.data[col_name] = np.nan
            state.data.loc[mask, col_name] = embeddings[:, d]
        else:
            state.data[col_name] = embeddings[:, d]

    console.print(f"\n[green]Embeddings complete:[/green]")
    console.print(f"  Generated: {stub}_1 to {stub}_{n_dims} ({n_dims} dimensions)")
    console.print(f"  {len(working_series)} texts embedded")

    state.mark_changed()


# ──────────────────────────────────────────────
#  hf models — List recommended models
# ──────────────────────────────────────────────

def _hf_models(rest, state, console):
    """List recommended models for each task."""
    task_filter = rest.strip().lower() if rest.strip() else None

    models = {
        "classify": [
            ("facebook/bart-large-mnli", "English zero-shot (best quality)"),
            ("joeddav/xlm-roberta-large-xnli", "Multilingual zero-shot"),
            ("MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli", "English (fast, accurate)"),
        ],
        "sentiment": [
            ("nlptown/bert-base-multilingual-uncased-sentiment", "Multilingual, 1-5 stars"),
            ("distilbert-base-uncased-finetuned-sst-2-english", "English, pos/neg (fast)"),
            ("cardiffnlp/twitter-roberta-base-sentiment-latest", "English Twitter sentiment"),
        ],
        "summarize": [
            ("facebook/bart-large-cnn", "English (best quality)"),
            ("sshleifer/distilbart-cnn-12-6", "English (faster)"),
            ("csebuetnlp/mT5_multilingual_XLSum", "Multilingual summarization"),
        ],
        "translate": [
            ("Helsinki-NLP/opus-mt-{src}-{tgt}", "Bilingual pairs (auto-selected)"),
            ("facebook/mbart-large-50-many-to-many-mmt", "50 languages, any direction"),
            ("facebook/nllb-200-distilled-600M", "200 languages"),
        ],
        "ner": [
            ("dslim/bert-base-NER", "English NER (PER, ORG, LOC, MISC)"),
            ("Davlan/xlm-roberta-large-ner-hrl", "Multilingual NER"),
        ],
        "embed": [
            ("sentence-transformers/all-MiniLM-L6-v2", "English (384d, fast)"),
            ("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", "Multilingual (384d)"),
            ("sentence-transformers/all-mpnet-base-v2", "English (768d, best quality)"),
        ],
    }

    console.print("\n[bold cyan]Recommended HuggingFace Models[/bold cyan]\n")

    for task, model_list in models.items():
        if task_filter and task_filter != task:
            continue

        console.print(f"  [bold]{task}[/bold]")
        for model_name, desc in model_list:
            console.print(f"    [cyan]{model_name}[/cyan]")
            console.print(f"      {desc}")
        console.print()


# ──────────────────────────────────────────────
#  hf download — Pre-download model
# ──────────────────────────────────────────────

def _hf_download(rest, state, console):
    """Pre-download a model to rln/hf_models/ for offline/portable use."""
    # Enable online mode for download
    os.environ.pop("HF_HUB_OFFLINE", None)
    os.environ.pop("TRANSFORMERS_OFFLINE", None)
    _check_transformers()
    model_name = rest.strip().strip("\"'")
    # Strip model() wrapper if user typed: hf download model(name)
    import re as _re
    m = _re.match(r'model\(([^)]+)\)', model_name)
    if m:
        model_name = m.group(1).strip()
    if not model_name:
        console.print("[red]Syntax: hf download model_name[/red]")
        console.print("[dim]Example: hf download facebook/bart-large-mnli[/dim]")
        return

    cache_dir = _get_cache_dir()
    console.print(f"[dim]Downloading model: {model_name}[/dim]")
    console.print(f"[dim]Cache directory: {cache_dir}[/dim]")
    console.print(f"[dim]Please wait until download completes fully...[/dim]")

    try:
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModel
        console.print(f"[dim]Downloading tokenizer...[/dim]")
        tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
        console.print(f"[dim]Downloading model weights (this may take several minutes)...[/dim]")
        try:
            model = AutoModelForSeq2SeqLM.from_pretrained(model_name, cache_dir=cache_dir)
        except Exception:
            model = AutoModel.from_pretrained(model_name, cache_dir=cache_dir)

        # Verify
        if _verify_model_ready(model_name, cache_dir):
            console.print(f"[green]Model downloaded and verified: {model_name}[/green]")
        else:
            console.print(f"[yellow]Model downloaded but could not verify: {model_name}[/yellow]")

        console.print(f"[dim]Location: {cache_dir}[/dim]")
        console.print(f"[dim]This model will work offline when you transfer the rln/ folder.[/dim]")
    except Exception as e:
        console.print(f"[red]Download failed: {e}[/red]")
        console.print("[dim]Check your internet connection and model name.[/dim]")


def _hf_cache(rest, state, console):
    """Show HuggingFace model cache info."""
    cache_dir = _get_cache_dir()
    console.print(f"\n[bold]HuggingFace Model Cache[/bold]")
    console.print(f"  Location: {cache_dir}")

    # Calculate total size
    total_size = 0
    n_files = 0
    models_found = set()

    for root, dirs, files in os.walk(cache_dir):
        for f in files:
            fp = os.path.join(root, f)
            try:
                total_size += os.path.getsize(fp)
                n_files += 1
            except OSError:
                pass

        # Try to identify model directories
        for d in dirs:
            if d.startswith("models--"):
                model_name = d.replace("models--", "").replace("--", "/")
                models_found.add(model_name)

    # Format size
    if total_size > 1e9:
        size_str = f"{total_size/1e9:.2f} GB"
    elif total_size > 1e6:
        size_str = f"{total_size/1e6:.1f} MB"
    elif total_size > 1e3:
        size_str = f"{total_size/1e3:.1f} KB"
    else:
        size_str = f"{total_size} bytes"

    console.print(f"  Total size: {size_str} ({n_files:,} files)")

    if models_found:
        console.print(f"\n  [bold]Cached models ({len(models_found)}):[/bold]")
        for m in sorted(models_found):
            console.print(f"    [cyan]{m}[/cyan]")
    else:
        console.print(f"\n  [dim]No models cached yet. Use: hf download model_name[/dim]")

    console.print()
