#!/bin/bash
# Double-click in Finder to open the dashboard (macOS; on Linux: ./start.command). Windows: start.bat.
# setup_env.py does the work: picks a supported Python (3.12-3.14), builds .venv with the exact package versions
# of constraints.txt (the same on every computer), the GPU runtime when there is an NVIDIA GPU, then starts the
# dashboard. Any Python 3.8+ can bootstrap it (the Mac's /usr/bin/python3 is enough).
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
done
if [ -z "$BOOT" ]; then
  echo "Python 3 not found. Install Python 3.13 from https://www.python.org/downloads/ (or: brew install python@3.13)"
  read -r -p "Press Enter to close"; exit 1
fi
"$BOOT" setup_env.py --run "$@"
rc=$?
if [ $rc -ne 0 ]; then read -r -p "(exit code $rc) Press Enter to close"; fi
exit $rc
