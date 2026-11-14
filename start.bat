@echo off
rem Windows launcher (same as start.command on macOS / Linux): dashboard of global_video_analysis + the Offside tab.
rem Double-click it in Explorer. First run creates .venv and installs both projects' requirements (a few minutes).
rem With an NVIDIA GPU (nvidia-smi works) the onnxruntime build that matches the driver and the GPU is installed
rem (CUDA 13 build for driver >= 580 on RTX 20xx / GTX 16xx or newer, else the CUDA 12 build), and a CUDA session is
rem actually tried; if CUDA still fails, DirectML is installed instead (any DirectX 12 GPU, no CUDA needed).
rem Diagnosis: .venv\Scripts\python -m football_eval.accel --doctor   (run inside global_video_analysis)
rem Retry the GPU set-up: delete .venv\.gpu_setup_tried
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
  rem the CPU onnxruntime of the requirements may have overwritten the GPU build: check the GPU again
  if exist .venv\.gpu_setup_tried del .venv\.gpu_setup_tried
)
rem NVIDIA GPU: make the detector run on it. "CUDAExecutionProvider" being LISTED does not mean CUDA works (the
rem onnxruntime-gpu build must match the driver and the GPU generation), so a real CUDA session is tried
rem (football_eval/accel.py --probe-cuda, a few seconds).
nvidia-smi -L >nul 2>&1
if errorlevel 1 goto :run
set ACCEL=%PY% -c "import sys; sys.path.insert(0, 'global_video_analysis'); from football_eval.accel import _main; sys.exit(_main(sys.argv[1:]))"
%ACCEL% --probe-cuda >nul 2>&1
if errorlevel 2 goto :run
if not errorlevel 1 goto :run
%PY% -c "import sys, onnxruntime as o; sys.exit('DmlExecutionProvider' not in o.get_available_providers())" >nul 2>&1
if not errorlevel 1 goto :run
if exist .venv\.gpu_setup_tried goto :gpu_hint
set ORTPKG=
for /f "usebackq delims=" %%P in (`%ACCEL% --recommend 2^>nul`) do if not defined ORTPKG set "ORTPKG=%%P"
if not defined ORTPKG set "ORTPKG=onnxruntime-gpu[cuda,cudnn]>=1.21,<1.27"
echo NVIDIA GPU found but CUDA does not work with the installed onnxruntime.
echo Installing "%ORTPKG%" (matches this driver / GPU; once, a few minutes)...
%PY% -m pip uninstall -y -q onnxruntime onnxruntime-gpu onnxruntime-directml >nul 2>&1
%PY% -m pip install -q "%ORTPKG%"
%ACCEL% --probe-cuda
if not errorlevel 1 goto :gpu_done
echo CUDA still does not work here: installing DirectML instead (uses the GPU through DirectX 12, no CUDA needed).
%PY% -m pip uninstall -y -q onnxruntime onnxruntime-gpu >nul 2>&1
%PY% -m pip install -q onnxruntime-directml || %PY% -m pip install -q "onnxruntime>=1.17"
:gpu_done
> .venv\.gpu_setup_tried echo "%ORTPKG%"
goto :run
:gpu_hint
echo [gpu] the NVIDIA GPU is not used (CUDA fails). Diagnosis: cd global_video_analysis ^&^& ..\.venv\Scripts\python -m football_eval.accel --doctor
echo [gpu] to retry the automatic GPU set-up, delete .venv\.gpu_setup_tried
:run
%PY% global_video_analysis\scripts\launcher.py %*
if errorlevel 1 pause
