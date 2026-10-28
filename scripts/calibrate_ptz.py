"""Venue calibration for a fixed-position pan/tilt/zoom camera (one-time, per filming spot).

    # 1) click 3-6 pitch landmarks on 1-3 frames (centre circle top/bottom, halfway line ends, corners...)
    python scripts/calibrate_ptz.py --video data/videos/match.mp4 --click 60 --landmarks configs/landmarks_myvenue.json

    # 2) solve the camera position and save the camera model used by `pitch.homography: ptz`
    python scripts/calibrate_ptz.py --video data/videos/match.mp4 --landmarks configs/landmarks_myvenue.json \
        --out configs/camera_myvenue_ptz.json --track-until 900

What it does: (a) multi-start fit of the camera centre C on the first landmark frame, (b) tracks the camera
(background motion + pitch-line alignment) over the first minutes, (c) bundle-adjusts C, roll, grass margins
and the poses of well-aligned keyframes, (d) repeats (b)-(c) once and writes an overlay sheet for checking.
Re-use the output for every match filmed from the same spot (only ``anchors`` depend on the video).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import least_squares

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from football_eval.config import load_config  # noqa: E402
from football_eval.pitch import Pitch  # noqa: E402
from football_eval.ptz import (Aligner, Evidence, PTZCamera, bundle_adjust, draw_template,  # noqa: E402
                               pose_from_landmarks, save_ptz, track_video)
from football_eval.video import probe, read_frame  # noqa: E402


def click_landmarks(video, t, path):
    """Minimal clicking UI (same keys as calibrate_points.py)."""
    from calibrate_points import ORDER  # noqa
    img = read_frame(video, t)
    pts, state = {}, {"c": None}
    s = 1280 / img.shape[1]
    disp = cv2.resize(img, None, fx=s, fy=s)
    cv2.namedWindow("landmarks", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("landmarks", lambda e, x, y, *_: state.update(c=(x / s, y / s)) if e == cv2.EVENT_LBUTTONDOWN else None)
    i = 0
    while i < len(ORDER):
        v = disp.copy()
        for n_, (u, w) in pts.items():
            cv2.circle(v, (int(u * s), int(w * s)), 4, (255, 0, 255), -1)
        cv2.putText(v, f"click {ORDER[i]}  [s]kip [u]ndo [q]uit", (10, 25), 0, 0.7, (0, 255, 255), 2)
        cv2.imshow("landmarks", v)
        k = cv2.waitKey(30) & 0xFF
        if state["c"] is not None:
            pts[ORDER[i]], state["c"] = state["c"], None
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
    p = Path(path)
    d = json.loads(p.read_text()) if p.exists() else {"frames": []}
    d["frames"] = [f for f in d["frames"] if abs(f["t"] - t) > 0.5] + [{"t": t, "points": {k: [round(a, 1), round(b, 1)] for k, (a, b) in pts.items()}}]
    d["frames"].sort(key=lambda f: f["t"])
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(d, indent=2))
    print(f"{len(pts)} landmarks at t={t}s -> {p}")


def initial_fit(img, points, ignore, pitch):
    """Multi-start fit of C, roll and the pose of the first landmark frame."""
    lm = pitch.landmarks()
    ev = Evidence(img, None, ignore)
    names = list(points)
    uvs = np.array([points[n] for n in names])
    best = None
    for Y in (-3, -6, -10, -15, -22, -30):
        for Z in (3, 5, 8, 12):
            for X in (35.0, 52.5, 70.0):
                def res(z):
                    cam = PTZCamera(z[:3], z[3], pitch=pitch)
                    uv, _ = cam.project_ours(np.array([lm[n] for n in names]), z[4:])
                    return np.r_[((uv - uvs) / 2).ravel(), Aligner(cam).residuals(z[4:], ev), z[3] / 0.03]
                d = np.hypot(52.5 - X, 34 - Y)
                z0 = [X, Y, Z, 0.0, np.arctan2(52.5 - X, 34 - Y), np.arctan2(Z, d) - 0.1, 700.0]
                lo, hi = [-30, -80, 1, -0.1, -1.7, -0.3, 150], [135, -0.3, 40, 0.1, 1.7, 0.7, 8000]
                try:
                    s = least_squares(res, z0, bounds=(lo, hi), x_scale=[1, 1, 1, 0.01, 0.01, 0.01, 20],
                                      diff_step=1e-4, loss="soft_l1", max_nfev=80)
                except ValueError:
                    continue
                c = float((s.fun ** 2).sum())
                if best is None or c < best[0]:
                    best = (c, s.x)
    z = best[1]
    return PTZCamera(z[:3], z[3], pitch=pitch), z[4:]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--config", default=None, help="for video.ignore_regions (overlays)")
    ap.add_argument("--landmarks", default=None, help="landmarks json ({frames: [{t, points}]})")
    ap.add_argument("--setup", default=None, help="match_setup.json: landmark frames, overlays and the REAL pitch size "
                    "are read from it, and the solved camera file is written back into it")
    ap.add_argument("--pitch", nargs=2, type=float, default=None, metavar=("L", "W"),
                    help="known pitch size in metres: keeps it fixed instead of estimating it")
    ap.add_argument("--click", type=float, default=None, help="open the clicking UI on the frame at this time (s)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--track-until", type=float, default=900.0)
    ap.add_argument("--fps", type=float, default=5.0)
    ap.add_argument("--iterations", type=int, default=2)
    a = ap.parse_args()
    from football_eval.config import resolve_path as _rp
    a.video = _rp(a.video) if getattr(a, "video", None) else a.video   # also finds it in the video libraries
    if a.click is not None:
        click_landmarks(a.video, a.click, a.landmarks)
        if not a.out:
            return
    cfg = load_config(a.config)
    setup = None
    if a.setup:
        from football_eval.match_setup import apply_to_config, load_setup
        setup = load_setup(a.setup)
        apply_to_config(setup, cfg)
        if a.pitch is None and cfg.pitch.fixed_size:
            a.pitch = [cfg.pitch.length, cfg.pitch.width]
    if a.pitch:
        cfg.pitch.length, cfg.pitch.width = map(float, a.pitch)
        print(f"pitch size fixed to {a.pitch[0]:.1f} x {a.pitch[1]:.1f} m (not estimated)")
    ignore = cfg.video.ignore_regions
    pitch = Pitch(cfg.pitch.length, cfg.pitch.width)
    info = probe(a.video)
    if setup is not None:
        L = [f for f in setup["calibration"]["landmark_frames"] if len(f.get("points", {})) >= 2]
    else:
        if not a.landmarks:
            sys.exit("give --landmarks or --setup")
        L = json.loads(Path(a.landmarks).read_text())["frames"]
    if not L:
        sys.exit("no landmark frame with >= 2 points")
    L = sorted(L, key=lambda f: -len(f["points"]))          # the richest frame starts the fit
    t_first = L[0]["t"]
    a.track_until = max(a.track_until, max(f["t"] for f in L) + 10)   # every landmark frame needs a tracked pose
    img0 = read_frame(a.video, t_first)
    print("[1] initial fit on the first landmark frame")
    cam, p0 = initial_fit(img0, L[0]["points"], ignore, pitch)
    print(f"    C={np.round(cam.C, 2)} pose={np.round(p0, 4)}")
    for it in range(a.iterations):
        print(f"[2.{it}] tracking {t_first:.0f}s -> {a.track_until:.0f}s")
        log, snaps = track_video(a.video, cam, p0, t_first, a.track_until, a.fps, ignore)
        tl = np.array([l["t"] for l in log])
        items = []
        for t, im in snaps.items():
            l = log[int(np.argmin(np.abs(tl - t)))]
            if l["pan"] is None:
                continue
            ev = Evidence(im, None, ignore)
            if ev.n_line < 150:
                continue
            p = np.array([l["pan"], l["tilt"], l["f"]])
            err = Aligner(cam).mean_line_error(p, ev)
            if err < 2.5:                                   # only clearly locked frames
                items.append((t, ev, p, err))
        items.sort(key=lambda z: z[2][0])
        keep = [items[i] for i in np.linspace(0, len(items) - 1, min(20, len(items))).astype(int)] if items else []
        evs = [Evidence(img0, None, ignore)] + [k[1] for k in keep]
        P0 = np.vstack([p0] + [k[2] for k in keep])
        lms = [L[0]["points"]] + [{}] * len(keep)
        for extra in L[1:]:                               # further landmark frames: pose from their own clicks
            im = read_frame(a.video, extra["t"])
            ev = Evidence(im, None, ignore)
            p_lm, lm_err = pose_from_landmarks(cam, ev, extra["points"], cam.pitch)
            print(f"    landmark frame t={extra['t']:.1f}s: pose {np.round(p_lm, 3)} (max landmark error {lm_err:.1f}px)")
            evs.append(ev)
            P0 = np.vstack([P0, p_lm])
            lms.append(extra["points"])
        print(f"[3.{it}] bundle adjustment on {len(evs)} keyframes")
        cam, P, errs = bundle_adjust(cam, evs, P0, lms, fit_pitch=a.pitch is None)
        p0 = P[0]
        print(f"    median line error {np.median(errs):.2f}px")
    out = a.out or "configs/camera_ptz.json"
    save_ptz(out, cam, {int(round(t_first * info.fps)): p0},
             meta={"video": a.video, "keyframe_line_err_px_median": float(np.median(errs)),
                   "landmarks_file": a.landmarks or a.setup, "pitch_fixed": bool(a.pitch)})
    if setup is not None:
        from football_eval.match_setup import mark, save_setup
        setup["calibration"].update(mode="ptz", camera_file=out)
        if not a.pitch:
            setup["pitch"].update(length=round(cam.pitch.length, 2), width=round(cam.pitch.width, 2), source="estimated")
        mark(setup, "calibration", "manual", f"camera solved by calibrate_ptz.py, median line error {np.median(errs):.2f}px")
        save_setup(setup, a.setup)
        print("-> camera file recorded in", a.setup)
    sheet = [draw_template(read_frame(a.video, t), cam, P[i + 1]) for i, (t, *_rest) in enumerate(keep[:8])]
    if sheet:
        while len(sheet) % 4:
            sheet.append(np.zeros_like(sheet[0]))
        g = np.vstack([np.hstack(sheet[i:i + 4]) for i in range(0, len(sheet), 4)])
        cv2.imwrite(str(Path(out).with_suffix(".png")), g)
    print("->", out, "(check the overlay PNG: magenta template must sit on the pitch lines)")


if __name__ == "__main__":
    main()
