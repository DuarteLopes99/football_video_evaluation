#!/usr/bin/env python3
"""Offside studio on its own page — the same thing as the dashboard's *Offside* tab.

    python offside_studio.py                 # http://127.0.0.1:8775
    python offside_studio.py --port 8780 --no-browser

Normally you open it from the dashboard (../start.command -> tab "Offside"). Venues are kept in venues/<name>/.
Guide: docs/STUDIO_GUIDE.md.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from offside.studio import serve  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8775)
    ap.add_argument("--no-browser", action="store_true")
    a = ap.parse_args()
    serve(a.host, a.port, not a.no_browser)
