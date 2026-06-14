<#
Build a no-admin Windows portable Rln folder using PyInstaller.

Run from the Rln project root in PowerShell:
  powershell -ExecutionPolicy Bypass -File packaging\windows_build_portable.ps1 -Tier full

Tiers:
  lite     core Rln, charts, econometrics, GUI, examples
  offline  lite + Argos/Sumy/NLTK-style offline NLP support
  full     offline + transformers/torch/sentence-transformers/HF support
#>
param(
    [ValidateSet("lite", "offline", "full")]
    [string]$Tier = "full",
    [switch]$Clean,
    [switch]$SkipInstall
)

$ErrorActionPreference = "Stop"
$Root = (Resolve-Path ".").Path
$Venv = Join-Path $Root ".venv-build"
$Python = Join-Path $Venv "Scripts\python.exe"
$DistDir = Join-Path $Root "dist\rln-$Tier"
$ZipPath = Join-Path $Root "dist\Rln-v1.2.7-windows-portable-$Tier.zip"

Write-Host "Rln portable Windows build" -ForegroundColor Cyan
Write-Host "Root: $Root"
Write-Host "Tier: $Tier"

if ($Clean) {
    Remove-Item -Recurse -Force "build", "dist" -ErrorAction SilentlyContinue
}

if (!(Test-Path $Python)) {
    Write-Host "Creating build virtual environment..." -ForegroundColor Cyan
    py -3 -m venv $Venv
}

if (-not $SkipInstall) {
    Write-Host "Installing build dependencies..." -ForegroundColor Cyan
    & $Python -m pip install --upgrade pip wheel setuptools pyinstaller
    if ($Tier -eq "lite") {
        & $Python -m pip install -r requirements.txt
    } elseif ($Tier -eq "offline") {
        & $Python -m pip install -r requirements.txt -r requirements-offline.txt
    } else {
        & $Python -m pip install -r requirements.txt -r requirements-offline.txt -r requirements-full.txt
    }
}

Write-Host "Building with PyInstaller..." -ForegroundColor Cyan
& $Python -m PyInstaller --clean --noconfirm rln.spec -- --tier=$Tier

if (!(Test-Path $DistDir)) {
    throw "Expected output folder not found: $DistDir"
}

Write-Host "Copying portable model and documentation folders beside the executable..." -ForegroundColor Cyan
foreach ($folder in @("hf_models", "argos_models", "examples")) {
    if (Test-Path $folder) {
        Copy-Item $folder -Destination $DistDir -Recurse -Force
    }
}
foreach ($file in @("README.md", "LICENSE", "Rln_Reference_Manual.docx", "Rln_v1.2.7_release_notes.docx")) {
    if (Test-Path $file) {
        Copy-Item $file -Destination $DistDir -Force
    }
}

$ExeName = "rln-$Tier.exe"
$ExePath = Join-Path $DistDir $ExeName
if (!(Test-Path $ExePath)) {
    throw "Expected executable not found: $ExePath"
}

@"
@echo off
cd /d "%~dp0"
set RLN_PORTABLE_ROOT=%~dp0
"%~dp0$ExeName" --gui
"@ | Set-Content -Encoding ASCII (Join-Path $DistDir "Rln-GUI.bat")

@"
@echo off
cd /d "%~dp0"
set RLN_PORTABLE_ROOT=%~dp0
"%~dp0$ExeName"
"@ | Set-Content -Encoding ASCII (Join-Path $DistDir "Rln-Console.bat")

@"
@echo off
cd /d "%~dp0"
set RLN_PORTABLE_ROOT=%~dp0
"%~dp0$ExeName" %*
"@ | Set-Content -Encoding ASCII (Join-Path $DistDir "rln.bat")

Write-Host "Smoke testing executable..." -ForegroundColor Cyan
& $ExePath --version

if (Test-Path $ZipPath) { Remove-Item $ZipPath -Force }
Write-Host "Creating portable zip..." -ForegroundColor Cyan
Compress-Archive -Path (Join-Path $DistDir "*") -DestinationPath $ZipPath -Force

Write-Host "DONE" -ForegroundColor Green
Write-Host "Portable folder: $DistDir"
Write-Host "Portable zip:    $ZipPath"
Write-Host "Run GUI:         $DistDir\Rln-GUI.bat"
Write-Host "Run console:     $DistDir\Rln-Console.bat"
