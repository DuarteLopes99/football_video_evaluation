@echo off
rem Windows launcher (same as start.command on macOS / Linux): dashboard of global_video_analysis + the Offside tab.
rem Double-click it in Explorer. First run creates .venv and installs both projects' requirements (a few minutes).
rem With an NVIDIA GPU (nvidia-smi works) onnxruntime-gpu replaces onnxruntime and the detector runs on CUDA
rem automatically; any other DirectX 12 GPU can use DirectML:
rem   .venv\Scripts\python -m pip uninstall -y onnxruntime  &&  .venv\Scripts\python -m pip install onnxruntime-directml
setlocal
cd /d "%~dp0"
rem UTF-8 everywhere (reports, setups and logs contain accents and symbols; Windows defaults to cp1252)
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
set REQ=-r global_video_analysis\requirements.txt -r offside_code\requirements.txt
set PY=.venv\Scripts\python.exe
if exist %PY% goto :have_venv
echo Creating the virtual environment (first run, a few minutes)...
rem "py" is the Python launcher from python.org; "python" can be the Microsoft Store alias, so it comes second
py -3 -m venv .venv >nul 2>&1 || python -m venv .venv
if not exist %PY% (
  echo Setup failed: install Python 3.10+ from https://www.python.org/downloads/ ^(tick "Add python.exe to PATH"^)
  pause & exit /b 1
)
%PY% -m pip install -q --upgrade pip
%PY% -m pip install -q %REQ% || (echo Setup failed & pause & exit /b 1)
type nul > .venv\.requirements_ok
:have_venv
rem requirements changed since the last install: install what is missing
%PY% -c "import os,sys; m=os.path.getmtime; ok='.venv/.requirements_ok'; sys.exit(0 if os.path.exists(ok) and m(ok) >= max(m('global_video_analysis/requirements.txt'), m('offside_code/requirements.txt')) else 1)"
if errorlevel 1 (
  echo Updating the Python packages...
  %PY% -m pip install -q %REQ% && type nul > .venv\.requirements_ok
)
rem NVIDIA GPU: swap the CPU onnxruntime for onnxruntime-gpu (CUDA / cuDNN come from pip, only the driver is needed).
rem Checked on every start, so a package update that brings the CPU build back is repaired here.
nvidia-smi -L >nul 2>&1
if errorlevel 1 goto :run
%PY% -c "import sys, onnxruntime as o; sys.exit('CUDAExecutionProvider' not in o.get_available_providers())" >nul 2>&1
if not errorlevel 1 goto :run
echo NVIDIA GPU found: installing onnxruntime-gpu (once, a few minutes)...
%PY% -m pip uninstall -y -q onnxruntime onnxruntime-gpu >nul 2>&1
%PY% -m pip install -q "onnxruntime-gpu[cuda,cudnn]>=1.21" || %PY% -m pip install -q "onnxruntime>=1.17"
:run
%PY% global_video_analysis\scripts\launcher.py %*
if errorlevel 1 pause
