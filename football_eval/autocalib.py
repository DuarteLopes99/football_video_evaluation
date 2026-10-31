"""Camera calibration straight from the landmarks clicked in the setup UI.

Problem it fixes: a setup with ``calibration.mode = "ptz"`` but no ``camera_file`` (the venue camera was never solved
with ``scripts/calibrate_ptz.py``) used to give **no calibration at all** -> every pitch coordinate NaN -> zero
possession / events / stats in the report, although the landmarks had been clicked.

:func:`ensure_calibration` is called by ``run_analysis.py`` after the setup is applied:

* ``ptz`` without camera file and >= 1 landmark frame with >= 4 clicks -> the venue camera (position, roll) and the
  pose of every landmark frame are solved from the clicks alone (:func:`football_eval.ptz.solve_from_landmarks`, at a
  640-px working resolution, ~15 s). The camera file is written next to the setup and recorded in it, so it is solved
  once per match (and re-usable for every match filmed from the same spot).
* if that fit is poor (median landmark error above 1.5 % of the frame width), it falls back to the ``flow`` mode:
  one homography per landmark frame, propagated by optical flow.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from .pitch import Pitch


def _read(cap, frame_idx: int):
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
    ok, img = cap.read()
    return img if ok else None


def landmark_frames(video_path: str, setup: dict, work_width: int = 640):
    """The setup's landmark frames (>= 4 clicks) read from the video at the working resolution.
    Returns (VideoInfo, scale, [(image, {landmark: [u, v]})], [(frame index, setup entry)])."""
    from .video import probe
    info = probe(video_path)
    s = work_width / info.width
    wh = int(round(info.height * s))
    lf = [f for f in setup.get("calibration", {}).get("landmark_frames") or [] if len(f.get("points", {})) >= 4]
    if not lf:
        raise ValueError("no landmark frame with >= 4 clicked points")
    cap = cv2.VideoCapture(video_path)
    frames, used = [], []
    try:
        for f in lf:
            fr = int(f.get("frame") if f.get("frame") not in (None, -1) else round(float(f["t"]) * info.fps))
            img = _read(cap, fr)
            if img is None:
                continue
            frames.append((cv2.resize(img, (work_width, wh), interpolation=cv2.INTER_AREA),
                           {k: [u * s, v * s] for k, (u, v) in f["points"].items()}))
            used.append((fr, f))
    finally:
        cap.release()
    if not frames:
        raise ValueError("the landmark frames could not be read from the video")
    return info, s, frames, used


def score_camera(cam, frames) -> tuple:
    """How well a (venue) camera explains the clicks: per frame, the pose that best fits the consistent clicks
    (RANSAC inliers of :func:`camera_from_homography`) and the median click error in working pixels.
    Returns (median error over frames, [pose per frame], [dropped clicks per frame])."""
    from .ptz import _pose_given_camera, camera_from_homography, landmark_errors
    h, w = frames[0][0].shape[:2]
    poses, meds, dropped = [], [], []
    for _, pts in frames:
        r = camera_from_homography(pts, cam.pitch, w, h)
        kept = {k: v for k, v in pts.items() if r is None or k in r[3]}
        p_init = r[2] if r is not None else np.array([0.0, 0.05, 1.1 * w])
        best = None
        for fs in (0.7, 1.0, 1.4):
            p = _pose_given_camera(cam, kept, [p_init[0], p_init[1], p_init[2] * fs])
            e = landmark_errors(cam, p, kept)
            m = float(np.median(e)) if len(e) else np.inf
            if best is None or m < best[0]:
                best = (m, p)
        meds.append(best[0])
        poses.append(best[1])
        dropped.append([] if r is None else list(r[4]))
    return float(np.median(meds)), poses, dropped


def venue_cameras(root: Path, width: int, height: int, exclude=()) -> list:
    """PTZ camera files in configs/ with the aspect ratio of this video (a venue camera is re-usable for every match
    filmed from the same spot, at any resolution)."""
    import json
    out = []
    for p in sorted((Path(root) / "configs").glob("camera_*ptz*.json")):
        if p.resolve() in {Path(e).resolve() for e in exclude}:
            continue
        try:
            d = json.loads(p.read_text())
        except (OSError, ValueError):
            continue
        if d.get("type") == "ptz" and abs(d.get("width", 640) / d.get("height", 360) - width / height) < 0.01:
            out.append(p)
    return out


def solve_ptz_from_setup(video_path: str, setup: dict, cfg, out_path: str, work_width: int = 640,
                         log=print, root: Optional[Path] = None) -> dict:
    """Solve and save a PTZ camera for this match from the setup's landmark frames. Returns a report dict.

    Two candidates are compared on your clicks and the one that explains them better is kept:
    (a) the camera solved from the clicks alone; (b) every venue camera already in configs/ (solved once per ground
    from many frames, usually more accurate) with the poses of your landmark frames as anchors."""
    from .ptz import load_ptz, save_ptz, solve_from_landmarks
    info, s, frames, used = landmark_frames(video_path, setup, work_width)
    pitch = Pitch(cfg.pitch.length, cfg.pitch.width)
    ignore = [[v * s for v in r] for r in (cfg.video.ignore_regions or [])]
    fit_pitch = not bool(getattr(cfg.pitch, "fixed_size", False))
    log(f"[calib] solving the camera from {len(frames)} landmark frame(s), "
        f"{sum(len(p) for _, p in frames)} clicks (pitch {'estimated' if fit_pitch else f'{pitch.length:g} x {pitch.width:g} m fixed'})")
    cam, poses, errs, dropped = solve_from_landmarks(frames, pitch, ignore, fit_pitch=fit_pitch)
    med = [float(np.median(e)) / s if len(e) else np.nan for e in errs]
    worst = [float(np.max(e)) / s if len(e) else np.nan for e in errs]
    rep = dict(method="ptz from clicked landmarks", work_width=work_width, frames=[u[0] for u in used],
               own_fit_err_median_px=[round(m, 1) for m in med], own_fit_err_max_px=[round(m, 1) for m in worst],
               C=[round(float(v), 2) for v in cam.C], pitch=[round(cam.pitch.length, 2), round(cam.pitch.width, 2)],
               ignored_clicks={f"{float(used[i][1]['t']):.1f}s": v for i, v in dropped.items() if v})
    for i, v in dropped.items():
        if v:
            log(f"[calib] landmark frame t={float(used[i][1]['t']):.1f}s: ignored click(s) {', '.join(v)} — they do not "
                f"fit the other clicks of that frame (wrong landmark name?). Fix them in the setup UI.")
    own_score = float(np.median([np.median(e) for e in errs if len(e)]))
    rep["candidates"] = {"solved from your clicks": round(own_score / s, 1)}
    best = (own_score, cam, poses, "setup landmark clicks")
    for vp in venue_cameras(root or Path("."), info.width, info.height, exclude=[out_path]):
        try:
            vcam, _ = load_ptz(str(vp))
            if vcam.width != work_width:
                continue
            sc, vposes, _ = score_camera(vcam, frames)
        except Exception as e:  # noqa: BLE001 — a broken file is just not a candidate
            log(f"[calib] venue camera {vp.name} skipped ({e})")
            continue
        rep["candidates"][vp.name] = round(sc / s, 1)
        if sc < best[0]:
            best = (sc, vcam, vposes, f"venue camera {vp.name} + poses from your clicks")
    sc, cam, poses, src = best
    rep.update(method=src, landmark_err_median_px=round(sc / s, 1), C=[round(float(v), 2) for v in cam.C],
               pitch=[round(cam.pitch.length, 2), round(cam.pitch.width, 2)])
    log(f"[calib] candidates (median click error, video px): "
        + ", ".join(f"{k} {v}" for k, v in rep["candidates"].items()) + f" -> {src}")
    anchors = {fr: p for (fr, _), p in zip(used, poses)}
    save_ptz(out_path, cam, anchors, meta=dict(video=str(video_path), solved_from=src,
                                               landmark_err_median_px=rep["landmark_err_median_px"],
                                               video_width=info.width, video_height=info.height))
    rep["camera_file"] = str(out_path)
    log(f"[calib] camera C={rep['C']} m, pitch {rep['pitch'][0]:g} x {rep['pitch'][1]:g} m; median click error "
        f"{rep['landmark_err_median_px']} px at {info.width}x{info.height} -> {out_path}")
    return rep


def ensure_calibration(setup: Optional[dict], setup_path: Optional[str], cfg, video_path: Optional[str],
                       root: Path, max_err_frac: float = 0.015, log=print) -> list:
    """Make sure the clicked landmarks give a camera calibration. Returns a change log (also for the provenance)."""
    if not setup or not video_path:
        return []
    cal = setup.get("calibration") or {}
    lf = [f for f in cal.get("landmark_frames") or [] if len(f.get("points", {})) >= 4]
    if cal.get("mode") != "ptz" or cal.get("camera_file") or not lf:
        return []
    stem = Path(setup_path).stem.replace("match_setup_", "") if setup_path else "match"
    out = Path(root) / "configs" / f"camera_{stem}_ptz.json"
    try:
        rep = solve_ptz_from_setup(video_path, setup, cfg, str(out), log=log, root=Path(root))
    except Exception as e:  # noqa: BLE001 — never stop the run: fall back to the flow mode below
        rep = dict(error=str(e))
        log(f"[calib] PTZ solve failed ({e})")
    e0 = rep.get("landmark_err_median_px")
    errs = [e0] if e0 is not None and np.isfinite(e0) else []
    vw = float(setup.get("video", {}).get("width") or 1920)
    max_err_px = max_err_frac * vw                     # 1.5 % of the frame width (29 px at 1080p): click accuracy
    if "error" not in rep and errs and max(errs) <= max_err_px:
        rel = str(out.relative_to(root)) if out.is_relative_to(root) else str(out)
        cal.update(camera_file=rel)
        cal.setdefault("auto", {})["ptz_from_landmarks"] = rep
        cfg.pitch.homography = "ptz"
        cfg.pitch.calibration_file = rel
        if setup_path:
            from .match_setup import save_setup
            save_setup(setup, setup_path)
            log(f"[calib] camera file recorded in {setup_path} (solved once; delete 'camera_file' there to re-solve)")
        msg = (f"pitch: PTZ camera from your {sum(len(f['points']) for f in lf)} clicked landmarks — {rep['method']} "
               f"(median click error {max(errs):.1f} px) -> {rel}")
        if getattr(cfg.pitch, "fixed_size", False) and (abs(rep["pitch"][0] - cfg.pitch.length) > 1.0 or
                                                         abs(rep["pitch"][1] - cfg.pitch.width) > 1.0):
            msg += (f"; NOTE the camera's pitch is {rep['pitch'][0]:.1f} x {rep['pitch'][1]:.1f} m, your setup says "
                    f"{cfg.pitch.length:g} x {cfg.pitch.width:g} m — check the real size")
        return [msg]
    # fallback: one homography per landmark frame, propagated with optical flow between them
    fps = float(setup.get("video", {}).get("fps") or 25.0)
    cfg.pitch.homography = "flow"
    cfg.pitch.flow_anchor_points = {int(f.get("frame") if f.get("frame") not in (None, -1) else round(f["t"] * fps)):
                                    dict(f["points"]) for f in lf}
    why = rep.get("error") or f"landmark error {max(errs) if errs else float('nan'):.1f} px > {max_err_px:g}"
    log(f"[calib] PTZ fit not good enough ({why}) -> flow homography from {len(lf)} landmark frame(s)")
    return [f"pitch: flow homography from {len(lf)} clicked landmark frame(s) (PTZ solve rejected: {why})"]
