@echo off
setlocal
cd /d "%~dp0\.."
powershell -ExecutionPolicy Bypass -File packaging\windows_build_portable.ps1 -Tier full %*
