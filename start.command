#!/bin/bash
# Double-click in Finder to open the launcher (macOS). First run creates .venv and installs requirements.
cd "$(dirname "$0")" || exit 1
if [ ! -x .venv/bin/python ]; then
  echo "Creating the virtual environment (first run, a few minutes)…"
  python3 -m venv .venv && .venv/bin/pip install -q -r requirements.txt || { echo "Setup failed"; read -r; exit 1; }
fi
# requirements changed since the last install (e.g. onnxruntime added): install what is missing
if [ requirements.txt -nt .venv/.requirements_ok ]; then
  echo "Updating the Python packages…"
  .venv/bin/pip install -q -r requirements.txt && touch .venv/.requirements_ok || echo "(package update failed — continuing; cv2.dnn is used instead of onnxruntime)"
fi
exec .venv/bin/python scripts/launcher.py "$@"
