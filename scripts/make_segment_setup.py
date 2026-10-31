"""Derive a match setup for a CUT of the full recording (debug segments such as canedo_seg_4980.mp4).

Every time in the setup (periods, kit clicks, landmark frames, match-sheet events, exclusions) is shifted by the
segment start, so the manual input of the full match is reused on the short clip:

    python scripts/make_segment_setup.py --setup configs/match_setup_<match>.json \
        --video data/videos/canedo_seg_4980.mp4 --offset 4980 --out configs/match_setup_canedo_seg_4980.json

* periods are clipped to the segment;
* kit samples clicked outside the segment keep their stored descriptor (``desc``) and are still used by the team
  classifier (``frame`` is set to -1: there is no image to snap them to);
* landmark frames outside the segment are dropped (they cannot be re-read), unless the venue camera is already solved.
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from football_eval.match_setup import KIT_CLASSES, load_setup, save_setup  # noqa: E402
from football_eval.video import probe  # noqa: E402


def segment_setup(setup: dict, video: str, offset_s: float, duration_s: float, fps_full: float) -> dict:
    s = copy.deepcopy(setup)
    info = probe(video)
    end = offset_s + duration_s
    inside = lambda t: t is not None and offset_s <= float(t) <= end
    s["video"] = {"path": video, "fps": info.fps, "n_frames": info.n_frames, "width": info.width,
                  "height": info.height, "duration_s": info.duration_s,
                  "segment_of": setup.get("video", {}).get("path"), "segment_offset_s": offset_s}
    per = []
    for p in setup.get("periods") or []:
        a, b = max(float(p["start"]), offset_s), min(float(p["end"]), end)
        if b - a > 1.0:
            per.append({"start": round(a - offset_s, 3), "end": round(b - offset_s, 3)})
    s["periods"] = per
    for c in KIT_CLASSES:
        out = []
        for it in setup["kits"].get(c, []):
            it = dict(it)
            if inside(it.get("t")):
                it["t"] = round(float(it["t"]) - offset_s, 3)
                it["frame"] = int(round(it["t"] * info.fps))
            else:
                it["frame"] = -1                   # descriptor only (no frame in this clip)
                it["t"] = None
            out.append(it)
        s["kits"][c] = out
    cal = s.get("calibration", {})
    if not cal.get("camera_file"):
        lf = []
        for f in cal.get("landmark_frames") or []:
            if inside(f.get("t")):
                t = float(f["t"]) - offset_s
                lf.append(dict(f, t=round(t, 3), frame=int(round(t * info.fps))))
        cal["landmark_frames"] = lf
    ev = []
    for e in setup.get("known_events") or []:
        e = dict(e)
        if e.get("t") not in (None, ""):
            if not inside(e["t"]):
                continue
            e["t"] = round(float(e["t"]) - offset_s, 3)
        ev.append(e)
    s["known_events"] = ev
    s["final_score"] = None                         # a clip does not contain the whole match sheet
    s["exclusions"] = [dict(x, t0=max(0.0, x["t0"] - offset_s), t1=min(duration_s, x["t1"] - offset_s))
                       for x in setup.get("exclusions") or [] if x["t1"] > offset_s and x["t0"] < end]
    s.setdefault("meta", {})["segment"] = {"of": setup.get("video", {}).get("path"), "offset_s": offset_s,
                                           "duration_s": duration_s, "fps_full": fps_full}
    return s


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--setup", required=True, help="setup of the full match")
    ap.add_argument("--video", required=True, help="the cut clip")
    ap.add_argument("--offset", type=float, required=True, help="start of the clip in the full video (s)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    full = load_setup(a.setup)
    info = probe(a.video)
    s = segment_setup(full, a.video, a.offset, info.duration_s, float(full["video"].get("fps") or info.fps))
    save_setup(s, a.out)
    n_lf = len(s["calibration"].get("landmark_frames") or [])
    print(f"-> {a.out}: periods {s['periods']}, {n_lf} landmark frame(s) inside the clip, "
          f"{sum(len(v) for v in s['kits'].values())} kit samples, {len(s['known_events'])} match-sheet event(s)")


if __name__ == "__main__":
    main()
