#!/bin/bash
# Double-click in Finder to open the dashboard (macOS; on Linux: ./start.command). Windows: start.bat.
# setup_env.py does the work: picks a supported Python (3.12-3.14), builds .venv with the exact package versions
# of constraints.txt (the same on every computer), the GPU runtime when there is an NVIDIA GPU, then starts the
# dashboard. Any Python 3.8+ can bootstrap it (the Mac's /usr/bin/python3 is enough).
cd "$(dirname "$0")" || exit 1
export PYTHONUTF8=1 PYTHONIOENCODING=utf-8
BOOT=""
for c in python3.13 python3.12 python3.14 python3 /opt/homebrew/bin/python3 /usr/local/bin/python3 /usr/bin/python3; do
  if command -v "$c" >/dev/null 2>&1 && "$c" -c "import sys; sys.exit(sys.version_info < (3, 8))" 2>/dev/null; then
    BOOT="$c"; break
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
