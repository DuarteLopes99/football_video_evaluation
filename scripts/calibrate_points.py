"""Click pitch landmarks on a frame -> calibration json (static camera) or an anchor (panning camera).

    python scripts/calibrate_points.py --video data/videos/match.mp4 --t 600 --out configs/calibration.json

Keys: the console asks for each landmark in turn — click it, or press 's' to skip it (not visible),
'u' to undo, 'q' to finish (>= 4 points; 6-10 well spread points recommended, e.g. corners of the
penalty area, halfway line ends, centre spot, centre-circle top/bottom).

For a *panning* camera, run it on several frames and pass ``--anchor``: the points are added under
``anchors.<frame_index>`` of the same json (use pitch.homography: flow).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from football_eval.pitch import Homography, Pitch  # noqa: E402
from football_eval.video import probe, read_frame  # noqa: E402

ORDER = ["corner_tl", "halfway_top", "corner_tr", "pa_l_top", "pa_l_bottom", "ga_l_top", "ga_l_bottom",
         "circle_top", "centre", "circle_bottom", "pa_r_top", "pa_r_bottom", "ga_r_top", "ga_r_bottom",
         "corner_bl", "halfway_bottom", "corner_br", "spot_l", "spot_r", "pa_l_top_line", "pa_l_bottom_line",
         "pa_r_top_line", "pa_r_bottom_line"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--t", type=float, default=0.0, help="time (s) of the frame to calibrate")
    ap.add_argument("--out", default="configs/calibration.json")
    ap.add_argument("--anchor", action="store_true", help="store as an anchor for pitch.homography=flow")
    ap.add_argument("--scale", type=float, default=0.6, help="display scale")
    a = ap.parse_args()
    from football_eval.config import resolve_path as _rp
    a.video = _rp(a.video) if getattr(a, "video", None) else a.video   # also finds it in the video libraries

    img = read_frame(a.video, a.t)
    info = probe(a.video)
    pitch = Pitch()
    lm = pitch.landmarks()
    pts = {}
    disp = cv2.resize(img, None, fx=a.scale, fy=a.scale)
    state = {"click": None}

    def on_mouse(ev, x, y, *_):
        if ev == cv2.EVENT_LBUTTONDOWN:
            state["click"] = (x / a.scale, y / a.scale)

    cv2.namedWindow("calibrate", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("calibrate", on_mouse)
    i = 0
    while i < len(ORDER):
        name = ORDER[i]
        view = disp.copy()
        for n, (u, v) in pts.items():
            cv2.circle(view, (int(u * a.scale), int(v * a.scale)), 5, (255, 0, 255), -1)
            cv2.putText(view, n, (int(u * a.scale) + 6, int(v * a.scale) - 6), 0, 0.5, (255, 0, 255), 1)
        cv2.putText(view, f"click: {name}   [s]kip [u]ndo [q]uit", (10, 25), 0, 0.8, (0, 255, 255), 2)
        cv2.imshow("calibrate", view)
        k = cv2.waitKey(30) & 0xFF
        if state["click"] is not None:
            pts[name] = state["click"]
            state["click"] = None
            i += 1
        elif k == ord("s"):
            i += 1
        elif k == ord("u") and pts:
            last = list(pts)[-1]
            pts.pop(last)
            i = ORDER.index(last)
        elif k == ord("q"):
            break
    cv2.destroyAllWindows()
    if len(pts) < 4:
        sys.exit("need >= 4 points")
    H = Homography.fit(list(pts.values()), [lm[n] for n in pts])
    err = H.reprojection_error(list(pts.values()), [lm[n] for n in pts])
    print(f"{len(pts)} points, mean reprojection error {err:.1f} px")
    entry = {"points": {k: [round(v[0], 1), round(v[1], 1)] for k, v in pts.items()}, "t": a.t,
             "frame_size": [img.shape[1], img.shape[0]]}
    out = Path(a.out)
    data = json.loads(out.read_text()) if out.exists() else {}
    if a.anchor:
        data.setdefault("anchors", {})[str(int(round(a.t * info.fps)))] = entry
    else:
        data.update(entry)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, indent=2))
    # overlay for checking
    for ln in pitch.lines():
        cv2.polylines(img, [H.to_image(ln).astype(np.int32)], False, (255, 0, 255), 2)
    cv2.imwrite(str(out.with_suffix(".png")), img)
    print("->", out, "and", out.with_suffix(".png"), "(check that the magenta lines sit on the painted lines)")


if __name__ == "__main__":
    main()
