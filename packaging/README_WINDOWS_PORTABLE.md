# Building Rln as a no-admin Windows portable app

This builds a portable **folder** plus a `.zip`, not an installer. Users can unzip it anywhere they have write access, such as Desktop, Downloads, or a USB drive. No admin rights are required.

## Recommended build

Open PowerShell in the Rln project root and run:

```powershell
powershell -ExecutionPolicy Bypass -File packaging\windows_build_portable.ps1 -Tier full -Clean
```

The output will be:

```text
dist\rln-full\
dist\Rln-v1.2.8-windows-portable-full.zip
```

Inside the portable folder:

```text
Rln-GUI.bat       launches the GUI
Rln-Console.bat   launches the TUI/REPL
rln.bat           forwards any command-line arguments
rln-full.exe      the bundled executable
hf_models\          portable HuggingFace model cache
argos_models\       portable Argos model/package folder
examples\           bundled examples
_internal\          PyInstaller libraries, do not delete
```

## Tiers

```powershell
# Smallest core build, GUI included, no HF/Argos neural stack
powershell -ExecutionPolicy Bypass -File packaging\windows_build_portable.ps1 -Tier lite -Clean

# Offline NLP build with Argos/Sumy-related dependencies, no torch/transformers
powershell -ExecutionPolicy Bypass -File packaging\windows_build_portable.ps1 -Tier offline -Clean

# Full build with HF/transformers/torch/sentence-transformers
powershell -ExecutionPolicy Bypass -File packaging\windows_build_portable.ps1 -Tier full -Clean
```

For your goal, use **full**.

## How to include HF and Argos models

Before building, put or download models into the source folders:

```text
hf_models\
argos_models\
```

The build script copies those folders beside the executable. At runtime, Rln sets:

```text
RLN_PORTABLE_ROOT=<portable app folder>
HF_HOME=<portable app folder>\hf_models
TRANSFORMERS_CACHE=<portable app folder>\hf_models
SENTENCE_TRANSFORMERS_HOME=<portable app folder>\hf_models\sentence_transformers
ARGOS_PACKAGES_DIR=<portable app folder>\argos_models
MPLCONFIGDIR=<portable app folder>\mpl_config
NLTK_DATA=<portable app folder>\nltk_data
```

That keeps models and caches inside the portable folder instead of the user's Windows profile.

## Why folder mode is recommended

Do **not** use one-file mode for the full portable build. One-file executables unpack to a temporary directory on every launch. That is slow for large torch/transformers builds and can confuse local model paths. Folder mode is faster, more transparent, and better for very large model folders.

## Distributing

Share the generated zip:

```text
dist\Rln-v1.2.8-windows-portable-full.zip
```

Users should unzip the folder and launch:

```text
Rln-GUI.bat
```

or:

```text
Rln-Console.bat
```

No installer and no admin rights are needed.
