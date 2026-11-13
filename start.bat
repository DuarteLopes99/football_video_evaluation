@echo off
rem Windows: double-click to open the dashboard (macOS / Linux: start.command).
rem setup_env.py does the work: picks a supported Python (3.12-3.14; installs 3.13 if you agree), builds .venv with
rem the exact package versions of constraints.txt (the same on every computer), onnxruntime-gpu when nvidia-smi
rem finds an NVIDIA GPU, then starts the dashboard. Any Python 3.8+ can bootstrap it, even an unsupported 3.15.
setlocal
cd /d "%~dp0"
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
set BOOT=
rem "py" = Python launcher / Python install manager; "python" can be the Microsoft Store alias, so it comes second
py -c "import sys" >nul 2>&1 && set BOOT=py
if not defined BOOT python -c "import sys" >nul 2>&1 && set BOOT=python
if not defined BOOT goto :nopython
%BOOT% setup_env.py --run %*
if errorlevel 1 pause
exit /b %errorlevel%

:nopython
echo Python not found. Install Python 3.13 (64-bit):
echo   https://www.python.org/downloads/windows/   or   winget install -e --id Python.Python.3.13
echo then double-click start.bat again.
pause
exit /b 1
