@echo off
rem Windows launcher (same as start.command on macOS / Linux): dashboard of global_video_analysis + the Offside tab.
rem First run creates .venv and installs both projects' requirements. With an NVIDIA GPU (nvidia-smi works)
rem onnxruntime-gpu replaces onnxruntime and the detector runs on CUDA automatically; any other DirectX 12 GPU
rem can use DirectML:  .venv\Scripts\pip install onnxruntime-directml  (after uninstalling onnxruntime).
setlocal
cd /d "%~dp0"
set REQ=-r global_video_analysis\requirements.txt -r offside_code\requirements.txt
set PY=.venv\Scripts\python.exe
if not exist %PY% (
  echo Creating the virtual environment ^(first run, a few minutes^)...
  python -m venv .venv || (echo Setup failed: is Python 3.10+ installed and on PATH? & pause & exit /b 1)
  %PY% -m pip install -q %REQ% || (echo Setup failed & pause & exit /b 1)
)
nvidia-smi -L >nul 2>&1
if %errorlevel%==0 (
  %PY% -c "import sys, onnxruntime as o; sys.exit('CUDAExecutionProvider' not in o.get_available_providers())" >nul 2>&1
  if errorlevel 1 (
    echo NVIDIA GPU found: installing onnxruntime-gpu ^(once^)...
    %PY% -m pip uninstall -y -q onnxruntime onnxruntime-gpu >nul 2>&1
    %PY% -m pip install -q "onnxruntime-gpu[cuda,cudnn]>=1.21" || %PY% -m pip install -q "onnxruntime>=1.17"
  )
)
%PY% global_video_analysis\scripts\launcher.py %*
