"""Cut one video clip per opportunity of an analysed match (also a button in the launcher).

    python scripts/cut_opportunities.py --run outputs/match1            # flagged opportunities only
    python scripts/cut_opportunities.py --run outputs/match1 --all      # every opportunity
    python scripts/cut_opportunities.py --run outputs/match1 --video data/videos/match1.mp4

Clips go to ``outputs/<match>/clips/opportunities/`` (ffmpeg stream copy when installed — about a second per clip —
otherwise OpenCV). The ▶ clip links are added to the existing report.html (no re-analysis).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from football_eval.config import resolve_path  # noqa: E402
from football_eval.opportunities import cut_clips, link_clips_in_report  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, help="output folder of run_analysis.py")
    ap.add_argument("--video", default=None, help="match video (default: the one recorded in meta.json)")
    ap.add_argument("--all", action="store_true", help="every opportunity, not only the flagged ones")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--no-report", action="store_true", help="do not refresh report.html")
    a = ap.parse_args()
    run = Path(a.run)
    f = run / "opportunities.csv"
    if not f.exists():
        sys.exit(f"{f} not found: run the analysis (or re-run the analytics) first")
    opps = pd.read_csv(f)
    opps["flag"] = opps.flag.astype(str).str.lower().isin(["true", "1"])
    video = a.video
    if not video and (run / "meta.json").exists():
        video = json.loads((run / "meta.json").read_text()).get("video", {}).get("path")
    video = resolve_path(video) if video else None
    if not video or not Path(video).exists():
        sys.exit("video not found: pass --video (or put the recording in a video library, see configs/storage.yaml)")
    sel = opps if a.all else opps[opps.flag]
    print(f"Cutting {len(sel)} clip(s) from {video}")
    done = cut_clips(opps, video, run, only_flagged=not a.all, overwrite=a.overwrite)
    print(f"{len(done)} clip(s) in {run / 'clips' / 'opportunities'}")
    if not a.no_report and (run / "report.html").exists():
        n = link_clips_in_report(run)
        print(f"report.html: {n} ▶ clip link(s) added")


if __name__ == "__main__":
    main()
