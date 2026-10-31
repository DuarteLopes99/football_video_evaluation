"""Simple front end for the whole project: pick a match video, run the analysis, make the manual decisions (setup /
review), watch the analysis live or as a replay with overlays, and browse the results and the flagged opportunities.

    python scripts/launcher.py                                   # opens http://127.0.0.1:8770 in the browser
    python scripts/launcher.py --video 2026-10-18_UD_Mansores_vs_Canedo_FC_J2.mp4          # that recording selected
    python scripts/launcher.py --video <file> --config configs/canedo_cfctv.yaml --start   # ... and processing starts
    python scripts/launcher.py --video <file> --open setup       # straight into the setup of that recording

``--video`` takes a file name (searched in every video library) or a path. Without ``--config`` / ``--setup`` the
match setup of that video (``configs/match_setup_<video>.json``) and the config recorded in it are used.
On macOS ``start.command`` (and the Desktop shortcuts) run this with the project's virtual environment and pass
their arguments through. User guide: docs/USER_GUIDE.md (also in the launcher, *Guide* tab).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from football_eval.ui.launcher import Launcher, serve  # noqa: E402

TABS = ("analyse", "setup", "review", "screen", "live", "results", "guide", "jobs")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", default=None, help="recording to select (file name in a video library, or a path)")
    ap.add_argument("--config", default=None, help="venue / match config, e.g. configs/canedo_cfctv.yaml")
    ap.add_argument("--setup", default=None, help="match setup json (default: the one of this video)")
    ap.add_argument("--run", default=None, help="output folder name (default: the video name)")
    ap.add_argument("--open", default=None, choices=TABS, help="screen to open first")
    ap.add_argument("--start", action="store_true", help="start processing the video at once (step 2)")
    ap.add_argument("--force", action="store_true", help="with --start: run even if the setup checklist has red items")
    ap.add_argument("--port", type=int, default=8770)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--no-browser", action="store_true")
    ap.add_argument("--python", default=None, help="interpreter for the jobs (default: this one)")
    a = ap.parse_args()
    (ROOT / "data" / "videos").mkdir(parents=True, exist_ok=True)
    (ROOT / "outputs").mkdir(exist_ok=True)
    pre = {}
    if a.video:
        from football_eval.storage import find_video, video_id
        path = find_video(a.video)
        if not path:
            sys.exit(f"video not found: {a.video} (put it in data/videos or a library of configs/storage.yaml)")
        pre["video_id"] = video_id(path)
        print(f"Selected: {path}")
    for k in ("config", "setup", "run"):
        v = getattr(a, k)
        if v:
            if k != "run" and not (ROOT / v).exists() and not Path(v).exists():
                sys.exit(f"{k} not found: {v}")
            pre[k] = v
    if a.open:
        pre["tab"] = a.open
    if a.start:
        if not a.video:
            sys.exit("--start needs --video")
        pre["start"] = True
        pre.setdefault("tab", "live")
    if a.force:
        pre["force"] = True
    serve(Launcher(a.python, pre), a.host, a.port, not a.no_browser)


if __name__ == "__main__":
    main()
