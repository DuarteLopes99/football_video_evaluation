#!/bin/bash
# Double-click in Finder to open the dashboard (macOS): global video analysis + the Offside tab.
# Also runs on Linux (./start.command); on Windows use start.bat. A GPU is used automatically when there is one.
# First run creates .venv (shared by global_video_analysis/ and offside_code/) and installs the requirements.
cd "$(dirname "$0")" || exit 1
REQ="-r global_video_analysis/requirements.txt -r offside_code/requirements.txt"
if [ ! -x .venv/bin/python ]; then
  echo "Creating the virtual environment (first run, a few minutes)…"
  python3 -m venv .venv && .venv/bin/pip install -q $REQ && touch .venv/.requirements_ok || { echo "Setup failed"; read -r; exit 1; }
fi
# requirements changed since the last install: install what is missing
if [ global_video_analysis/requirements.txt -nt .venv/.requirements_ok ] || [ offside_code/requirements.txt -nt .venv/.requirements_ok ]; then
  echo "Updating the Python packages…"
  .venv/bin/pip install -q $REQ && touch .venv/.requirements_ok && rm -f .venv/.gpu_setup_tried || echo "(package update failed — continuing; cv2.dnn is used instead of onnxruntime)"
fi
# NVIDIA GPU (Linux / WSL): replace the CPU onnxruntime by onnxruntime-gpu so the detector runs on CUDA. Checked on
# every start, so a requirements update that brings the CPU build back is repaired here.
# The CUDA / cuDNN libraries come from pip ([cuda,cudnn] extras), no CUDA toolkit needed. On a Mac nothing changes:
# the default onnxruntime already has CoreML (Apple GPU / Neural Engine) and the detector picks it when faster.
if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
  # "CUDAExecutionProvider" listed does not mean CUDA works: try a real CUDA session (football_eval/accel.py)
  ACCEL=(.venv/bin/python -c "import sys; sys.path.insert(0, 'global_video_analysis'); from football_eval.accel import _main; sys.exit(_main(sys.argv[1:]))")
  "${ACCEL[@]}" --probe-cuda >/dev/null 2>&1; rc=$?
  if [ $rc -eq 1 ] && [ ! -f .venv/.gpu_setup_tried ]; then
    PKG=$("${ACCEL[@]}" --recommend 2>/dev/null | head -n1); PKG=${PKG:-"onnxruntime-gpu[cuda,cudnn]>=1.21,<1.27"}
    echo "NVIDIA GPU found but CUDA does not work with the installed onnxruntime: installing $PKG (once)…"
    .venv/bin/pip uninstall -y -q onnxruntime onnxruntime-gpu >/dev/null 2>&1
    .venv/bin/pip install -q "$PKG" || { echo "(onnxruntime-gpu failed, back to the CPU build)"; .venv/bin/pip install -q "onnxruntime>=1.17"; }
    "${ACCEL[@]}" --probe-cuda || echo "(CUDA still fails: python -m football_eval.accel --doctor in global_video_analysis)"
    echo "$PKG" > .venv/.gpu_setup_tried
  fi
fi
exec .venv/bin/python global_video_analysis/scripts/launcher.py "$@"
