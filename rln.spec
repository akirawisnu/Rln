# -*- mode: python ; coding: utf-8 -*-
"""
Rln PyInstaller spec — tier-aware, cross-platform.

Usage:
    # Default is 'lite' (smallest, no NLP):
    pyinstaller rln.spec

    # Build a specific tier (pass args AFTER a -- separator):
    pyinstaller rln.spec -- --tier=lite      (~80-120 MB)
    pyinstaller rln.spec -- --tier=offline   (~200-300 MB, offline NLP)
    pyinstaller rln.spec -- --tier=full      (~2-4 GB, neural NLP)

    # Single-file executable (slower startup, easier to share):
    pyinstaller rln.spec -- --tier=lite --onefile

Tiers:
    lite     Core stats + econometrics + charts + LRTM + diff-diff.
             No NLP. This is what most users want.
    offline  lite + sumy + argostranslate. Offline summarize + translate.
             No neural models, no torch dependency.
    full     offline + transformers + torch. Every hf/nlp command works.
             Enormous build size. Only use if you actually need neural NLP.

Platform notes:
    Windows:  produces rln-<tier>.exe. Console kept (this is a REPL).
    macOS:    also produces Rln-<tier>.app. Add codesign_identity for
              notarization if distributing publicly.
    Linux:    produces ELF binary. Works on any glibc >= build machine's.
    Termux:   cannot build with PyInstaller on Termux directly — torch/scipy
              wheels aren't available for aarch64-Android. Use the source
              distribution on Termux; see packaging/termux_install.sh.
"""

import os
import sys
import platform
from PyInstaller.utils.hooks import collect_all, collect_submodules, collect_data_files

# ─────────────────────────────────────────────────────────────
# Parse custom args passed after `--` to PyInstaller
# ─────────────────────────────────────────────────────────────
TIER = "lite"
ONEFILE = False

_custom_args = sys.argv[1:] if len(sys.argv) > 1 else []
for arg in _custom_args:
    if arg.startswith("--tier="):
        TIER = arg.split("=", 1)[1].strip().lower()
    elif arg == "--onefile":
        ONEFILE = True

if TIER not in ("lite", "offline", "full"):
    raise SystemExit(f"Unknown tier {TIER!r}. Use lite, offline, or full.")

print(f"\n*** Rln build: tier={TIER}, onefile={ONEFILE}, "
      f"platform={platform.system()}/{platform.machine()} ***\n")


# ─────────────────────────────────────────────────────────────
# Data files — what ships with the binary
# ─────────────────────────────────────────────────────────────
datas = [
    ('examples', 'examples'),
    ('rln_io', 'rln_io'),
    ('README.md', '.'),
    ('LICENSE', '.'),
]

# Ship model folders only if they exist (empty dirs would error on copy)
if os.path.isdir('argos_models'):
    datas.append(('argos_models', 'argos_models'))
if os.path.isdir('hf_models') and TIER == "full":
    datas.append(('hf_models', 'hf_models'))


# ─────────────────────────────────────────────────────────────
# Hidden imports — things Rln imports lazily at runtime
# ─────────────────────────────────────────────────────────────

core_hidden = [
    # Rln internals — enumerated explicitly so any refactor bug
    # (renamed/moved module) fails the build rather than at runtime.
    'commands',
    'commands.parse_helpers',
    'commands.parser',
    'commands.dofile',
    'commands.state',
    'commands.expression',
    'commands.data_io',
    'commands.explore',
    'commands.variables',
    'commands.dataops',
    'commands.utility',
    'commands.advanced',
    'commands.extras',
    'commands.rln_cmds',
    'commands.scripting',
    'commands.estimation',
    'commands.panel',
    'commands.charts',
    'commands.nlp',
    'commands.nlp_extend',
    'commands.lrtm',
    'rln_io',
    'rln_io.fileio',
    'tui',
    'tui.browser',
    'tui.doeditor',
    'gui',
    'gui.app',
    'tkinter',
    'tkinter.ttk',
    'tkinter.filedialog',
    'tkinter.messagebox',
    # 3rd-party core
    'pandas',
    'numpy',
    'openpyxl',
    'xlrd',
    'dbfread',
    'pyreadr',
    'polyfuzz',
    'textual',
    'rich',
    'prompt_toolkit',
    'lxml',
    'lxml.etree',
    'html5lib',
    'statsmodels',
    'statsmodels.api',
    'statsmodels.formula.api',
    'statsmodels.regression.linear_model',
    'statsmodels.stats.api',
    'scipy',
    'scipy.stats',
    'scipy.special',
    'scipy.sparse',
    'scipy._lib.messagestream',
    'matplotlib',
    'matplotlib.pyplot',
    'matplotlib.backends.backend_agg',
    'matplotlib.backends.backend_tkagg',
    # diff-diff — always included; small and frequently used
    'diff_diff',
    'diff_diff._backend',
    # polars for lrtm; soft-include (PyInstaller skips missing modules)
    'polars',
    'rapidfuzz',
]

offline_hidden = core_hidden + [
    'argostranslate',
    'argostranslate.package',
    'argostranslate.translate',
    'argostranslate.settings',
    'sumy',
    'sumy.parsers',
    'sumy.parsers.plaintext',
    'sumy.nlp',
    'sumy.nlp.tokenizers',
    'sumy.nlp.stemmers',
    'sumy.summarizers',
    'sumy.summarizers.lsa',
    'sumy.summarizers.lex_rank',
    'sumy.summarizers.text_rank',
    'sumy.summarizers.luhn',
    'sumy.summarizers.kl',
    'sumy.summarizers.sum_basic',
    'sumy.summarizers.edmundson',
    'sumy.utils',
    'nltk',
    'nltk.tokenize',
    'nltk.corpus',
    'nltk.stem',
]

full_hidden = offline_hidden + [
    'transformers',
    'transformers.models',
    'torch',
    'torch.jit',
    'torch.nn',
    'torch.nn.functional',
    'torch.utils.data',
    'sentence_transformers',
    'huggingface_hub',
    'tokenizers',
    'safetensors',
]

hiddenimports = {
    "lite":    list(core_hidden),
    "offline": list(offline_hidden),
    "full":    list(full_hidden),
}[TIER]

# Make refactors safer: include all current Rln command and GUI modules.
try:
    hiddenimports += collect_submodules("commands")
    hiddenimports += collect_submodules("gui")
except Exception as exc:
    print(f"  [warn] collect_submodules(commands/gui) failed: {exc}")


# ─────────────────────────────────────────────────────────────
# Collect data/submodules for packages with dynamic imports
# ─────────────────────────────────────────────────────────────
binaries = []

for pkg in ("statsmodels", "scipy", "pandas"):
    try:
        _b, _d, _h = collect_all(pkg)
        binaries += _b
        datas    += _d
        hiddenimports += _h
    except Exception as exc:
        print(f"  [warn] collect_all({pkg}) failed: {exc}")

if TIER in ("offline", "full"):
    try:
        datas += collect_data_files("sumy")
        hiddenimports += collect_submodules("sumy")
    except Exception as exc:
        print(f"  [warn] collect_data_files(sumy) failed: {exc}")
    try:
        datas += collect_data_files("nltk")
    except Exception:
        pass  # NLTK data is optional
    # argostranslate has no bundled data itself (models are external),
    # but it does have a few compiled helpers
    try:
        hiddenimports += collect_submodules("argostranslate")
    except Exception:
        pass

if TIER == "full":
    for pkg in ("transformers", "tokenizers", "sentence_transformers",
                "huggingface_hub", "safetensors"):
        try:
            _b, _d, _h = collect_all(pkg)
            binaries += _b
            datas    += _d
            hiddenimports += _h
        except Exception as exc:
            print(f"  [warn] collect_all({pkg}) failed: {exc}")
    # torch is enormous; rely on PyInstaller's built-in torch hook + the
    # explicit hiddenimports above. collect_all('torch') would add 3+ GB
    # of test fixtures and docs we don't need.


# ─────────────────────────────────────────────────────────────
# Excludes — keep lite actually lite
# ─────────────────────────────────────────────────────────────
base_excludes = [
    'PyQt5', 'PyQt6', 'PySide2', 'PySide6',
    'IPython', 'jupyter', 'notebook', 'jupyterlab',
    'pytest', 'hypothesis', 'mock',
    'tensorflow', 'keras', 'jax', 'flax',
    'cv2', 'pygame',
]

if TIER == "lite":
    base_excludes += [
        'torch', 'transformers', 'tokenizers', 'sentence_transformers',
        'huggingface_hub', 'safetensors',
        'argostranslate', 'stanza', 'ctranslate2', 'sentencepiece',
        'sumy', 'nltk',
        'sklearn', 'PIL', 'Pillow',
    ]
elif TIER == "offline":
    base_excludes += [
        'torch', 'transformers', 'tokenizers', 'sentence_transformers',
        'huggingface_hub', 'safetensors',
        'sklearn',
    ]
# 'full' tier excludes nothing extra

excludes = base_excludes


# ─────────────────────────────────────────────────────────────
# Analysis + PYZ
# ─────────────────────────────────────────────────────────────
a = Analysis(
    ['main.py'],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=['packaging/runtime_hooks/rln_portable_runtime.py'],
    excludes=excludes,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data)


# ─────────────────────────────────────────────────────────────
# Platform-specific binary assembly
# ─────────────────────────────────────────────────────────────
exe_name = f"rln-{TIER}"
icon_path = None
if platform.system() == 'Windows' and os.path.isfile('resources/icon.ico'):
    icon_path = 'resources/icon.ico'
elif platform.system() == 'Darwin' and os.path.isfile('resources/icon.icns'):
    icon_path = 'resources/icon.icns'

if ONEFILE:
    exe = EXE(
        pyz,
        a.scripts,
        a.binaries,
        a.datas,
        [],
        name=exe_name,
        debug=False,
        bootloader_ignore_signals=False,
        strip=False,
        upx=True,
        upx_exclude=[
            # UPX corrupts these on some Python / Windows combinations
            'vcruntime140.dll', 'python3*.dll',
            # And never compress ML runtime libs (huge, already compressed internally)
            'libtorch*.so*', 'libcudart*', 'torch_cpu*', 'torch_cuda*',
            'c10*.so*', 'c10*.dll',
        ],
        runtime_tmpdir=None,
        console=True,
        disable_windowed_traceback=False,
        argv_emulation=False,
        target_arch=None,
        codesign_identity=None,
        entitlements_file=None,
        icon=icon_path,
    )
else:
    # Folder distribution — faster startup, easier to inspect/patch
    exe = EXE(
        pyz,
        a.scripts,
        [],
        exclude_binaries=True,
        name=exe_name,
        debug=False,
        bootloader_ignore_signals=False,
        strip=False,
        upx=True,
        console=True,
        disable_windowed_traceback=False,
        argv_emulation=False,
        target_arch=None,
        codesign_identity=None,
        entitlements_file=None,
        icon=icon_path,
    )
    coll = COLLECT(
        exe,
        a.binaries,
        a.datas,
        strip=False,
        upx=True,
        upx_exclude=[],
        name=f"rln-{TIER}",
    )


# ─────────────────────────────────────────────────────────────
# macOS .app bundle — only when building on macOS, folder mode
# ─────────────────────────────────────────────────────────────
if platform.system() == 'Darwin' and not ONEFILE:
    app = BUNDLE(
        coll,
        name=f'Rln-{TIER}.app',
        icon=icon_path,
        bundle_identifier=f'org.akirawisnu.rln.{TIER}',
        info_plist={
            'NSHighResolutionCapable': 'True',
            'CFBundleShortVersionString': '1.2.7',
            'CFBundleVersion': '1.2.7',
            'LSBackgroundOnly': 'False',
            'NSRequiresAquaSystemAppearance': 'False',
            'LSEnvironment': {
                # Keep model caches inside the app bundle so it stays portable
                'HF_HOME': '@executable_path/../Resources/hf_models',
                'ARGOS_PACKAGES_DIR': '@executable_path/../Resources/argos_models',
            },
        },
    )
