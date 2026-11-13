"""Manual, explainable steps of the pipeline in the browser.

Before processing — validate what the system cannot reliably infer (~10 minutes per match):

    python scripts/match_ui.py setup --video data/videos/match.mp4 --setup configs/match_setup_match1.json \
        [--config configs/<venue>.yaml]

    1 match & pitch size   2 periods & direction   3 overlays   4 kits (click players)   5 pitch landmarks
    6 match sheet (score, goals, cards, rosters)   7 checklist

After processing (run_analysis.py) — measure the quality and fix the key events (~10 minutes):

    python scripts/match_ui.py review --run outputs/match1 --setup configs/match_setup_match1.json \
        [--config configs/<venue>.yaml]

    1 frame checks (lines / teams on ~16 sampled frames)   2 key events (confirm / reject / flip team, shirt no.)
    3 add missing events   4 summary

Then re-run the analytics only (seconds): ``python run_analysis.py --tracks outputs/match1 --setup ... --out outputs/match1``
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from football_eval.config import resolve_path  # noqa: E402
from football_eval.ui.server import App, serve  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["setup", "review"])
    ap.add_argument("--video", default=None, help="match video (setup); review reads it from the run's meta.json")
    ap.add_argument("--setup", required=True, help="match_setup.json (created if missing)")
    ap.add_argument("--config", default=None, help="venue / match YAML (detector settings, PTZ camera)")
    ap.add_argument("--run", default=None, help="review: output folder of run_analysis.py")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--no-browser", action="store_true")
    a = ap.parse_args()
    video = a.video
    if a.mode == "review":
        if not a.run:
            sys.exit("review needs --run outputs/<match>")
        meta = Path(a.run) / "meta.json"
        if not video and meta.exists():
            video = json.loads(meta.read_text()).get("video", {}).get("path")
        if not video and Path(a.setup).exists():
            video = json.loads(Path(a.setup).read_text()).get("video", {}).get("path")
    if not video:
        sys.exit("give --video")
    video = resolve_path(video)
    if not Path(video).exists():
        sys.exit(f"video not found: {video}")
    app = App(a.mode, video, a.setup, a.config, a.run)
    serve(app, a.host, a.port, not a.no_browser)


if __name__ == "__main__":
    main()
