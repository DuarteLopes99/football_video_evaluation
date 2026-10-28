"""Perception pipeline: match video -> tracking table in pitch coordinates.

Output tables (saved as CSV in ``<output_dir>/``):

``tracks.csv``  one row per person detection
    frame, t, track_id, cls (player|goalkeeper|referee), conf, x0, y0, x1, y1,
    u, v (foot pixel), x, y (pitch metres), edge (1 = cut by the bottom of the frame: feet estimated), team (0|1|-1)
``ball.csv``    one row per analysed frame (see :func:`football_eval.ball.clean_ball`) + vis_near_y
``frames.csv``  frame, t, has_h (homography available), n_people, vis_near_y (pitch y where the view ends on the
                camera side; < pitch width = near touchline out of view), n_tiles (detector tiles), roi_top
                (detector rows skipped above the pitch, px)
``meta.json``   video info, config, scoreboard changes
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import pandas as pd

from .ball import clean_ball
from .config import Config
from .detection import build_detector
from .pitch import Pitch, build_provider
from .scoreboard import ScoreboardWatcher
from .teams import (CODE, CODE_NAME, KitClassifier, TeamClassifier, TrackVotes, apply_manual, person_like,
                    snap_box, vote, vote_roles)
from .tracking import build_tracker, iou_matrix
from .video import iter_frames, probe


def run_perception(video_path: str, cfg: Config, progress: bool = True, max_frames: Optional[int] = None,
                   team_every: int = 2, min_hits_for_team: bool = True, live_dir: Optional[str] = None,
                   team_names=None) -> dict:
    """``live_dir``: write a live preview of the frame being analysed (football_eval/live.py) for the launcher."""
    info = probe(video_path)
    pitch = Pitch(cfg.pitch.length, cfg.pitch.width)
    detector = build_detector(cfg)
    tracker = build_tracker(cfg, cfg.video.analysis_fps)
    provider = build_provider(cfg, pitch, info.fps)
    if getattr(provider, "cam", None) is not None:      # PTZ venue file carries the measured pitch size
        pitch = provider.cam.pitch
        cfg.pitch.length, cfg.pitch.width = float(pitch.length), float(pitch.width)
        print(f"[ptz] pitch {pitch.length:.1f} x {pitch.width:.1f} m from {cfg.pitch.calibration_file}")
    teams = TeamClassifier(cfg.teams.backend, cfg.teams.outlier_quantile, cfg.teams.outlier_factor, cfg.detection.device)
    sb = ScoreboardWatcher(cfg.scoreboard.roi, cfg.scoreboard.change_threshold, cfg.scoreboard.min_persist_s) \
        if cfg.scoreboard.enabled and cfg.scoreboard.roi else None
    step = max(1, int(round(info.fps / cfg.video.analysis_fps)))
    windows = processing_windows(cfg, info.duration_s)

    # ---- the players clicked in the setup: one classifier for the live view AND the final team assignment
    kit, seeds, kit_report = None, {}, {}
    if cfg.teams.kit_samples and cfg.teams.backend == "color":
        kit, seeds, kit_report = build_kit_bank(video_path, cfg, teams, detector, info)
    votes = TrackVotes()

    live = None
    if live_dir:
        from .live import LivePreview
        total = int(sum((b - a) for a, b in windows) * cfg.video.analysis_fps)
        total = min(total, max_frames) if max_frames else total
        live = LivePreview(live_dir, pitch, info, cfg, total_frames=total or None, team_names=team_names,
                           kit_report=kit_report)

    # PTZ camera: track the camera from the calibration anchor up to the requested start time
    anchor = getattr(provider, "anchor_frame", None)
    start_s = windows[0][0]
    if anchor is not None and start_s > anchor / info.fps + 1.0:
        print(f"[ptz] camera warm-up from {anchor / info.fps:.0f}s to {start_s:.0f}s")
        if live is not None:
            live.finish(f"camera warm-up: tracking the camera from {anchor / info.fps:.0f} s to the start "
                        f"({start_s:.0f} s)")
        provider.warmup(video_path, anchor / info.fps, start_s, cfg.video.analysis_fps)
    motion_ball = None
    if cfg.ball.detector in ("motion", "both"):
        from .ballcv import MotionBallDetector
        motion_ball = MotionBallDetector(cfg.video.ignore_regions)
    if provider is None:
        print("[warn] no homography configured -> pitch coordinates will be NaN; analytics needs them "
              "(set pitch.calibration_file, see scripts/calibrate_points.py)")
    if len(windows) > 1 or windows[0] != (cfg.video.start_s, cfg.video.end_s):
        print("[video] processing " + ", ".join(f"{a / 60:.1f}-{b / 60:.1f} min" for a, b in windows)
              + f" ({sum(b - a for a, b in windows) / 60:.0f} of {info.duration_s / 60:.0f} min)")

    rows, ball_rows, frame_rows, feats, feat_rows = [], [], [], [], []
    seed_log = []
    if live is not None:
        live.begin()
    t0 = time.time()
    n = 0
    prev_H, prev_top = None, None
    for f, t, img in iter_frames(video_path, cfg.video.analysis_fps, windows=windows,
                                 resize_width=cfg.video.resize_width):
        roi = pitch_roi(prev_H, pitch, img.shape, prev_top) if (cfg.detection.pitch_roi and prev_H is not None) else None
        det = detector(img, roi=roi)
        people = det.of("player", "goalkeeper", "referee")
        prev_top = float(people.xyxy[:, 1].min()) if len(people) else None      # ROI never cuts a seen player
        ids = tracker.update(people)
        H = provider.get(f, img, people.xyxy) if provider is not None else None
        prev_H = H
        foot, edge = foot_points(people.xyxy, img.shape)
        xy = H.to_pitch(foot) if (H is not None and len(foot)) else np.full((len(foot), 2), np.nan)
        do_feat = (n % team_every == 0)
        desc = teams.describe(img, people.xyxy) if do_feat and len(people) else [None] * len(people)
        codes = np.full(len(people), -1, int)
        if kit is not None:
            ok = [k for k in range(len(people)) if desc[k] is not None]
            if ok:
                codes[ok] = kit.predict(np.asarray([desc[k] for k in ok]))
                votes.add(ids[ok], codes[ok])
        if seeds:
            _apply_seeds(seeds, f, step, people, ids, tracker, votes, seed_log)
        for k in range(len(people)):
            r = len(rows)
            rows.append((f, t, int(ids[k]), people.cls[k], float(people.conf[k]), *people.xyxy[k], *foot[k], *xy[k],
                         int(edge[k])))
            if desc[k] is not None and (ids[k] >= 0 or not min_hits_for_team):
                feats.append(desc[k])
                feat_rows.append(r)
        balls = det.of("ball") if cfg.ball.detector in ("model", "both") else det.of("__none__")
        n_ball0 = len(ball_rows)
        if motion_ball is not None and H is not None:
            ev = getattr(provider, "last_evidence", None)
            for u, v, sc in motion_ball(img, getattr(provider, "last_image_motion", None),
                                        None if ev is None else ev.field, people.xyxy):
                bxy = H.to_pitch(np.array([[u, v]]))[0]
                ball_rows.append((f, t, float(min(0.99, max(0.05, sc / 3))), u, v, *bxy))
        if len(balls):
            bf = balls.foot
            bxy = H.to_pitch(bf) if H is not None else np.full((len(bf), 2), np.nan)
            for k in range(len(balls)):
                ball_rows.append((f, t, float(balls.conf[k]), *bf[k], *bxy[k]))
        if live is not None and live.due():
            labels = [CODE_NAME.get(votes.label(int(ids[k]), int(codes[k])) if ids[k] >= 0 else int(codes[k]))
                      for k in range(len(people))]
            manual = [int(ids[k]) in votes.manual for k in range(len(people))]
            live.update(n + 1, f, t, img, people, ids, xy, H, provider,
                        [(r[3], r[4]) for r in ball_rows[n_ball0:]], labels=labels, manual=manual,
                        tiles=getattr(detector, "last_tiles", None))
        if sb is not None:
            sb.update(t, img)
        frame_rows.append((f, t, H is not None, len(people), visible_near_y(H, pitch, img.shape),
                           len(getattr(detector, "last_tiles", []) or [1]), int(roi[1]) if roi else 0))
        n += 1
        if progress and n % 100 == 0:
            el = time.time() - t0
            print(f"  {n} frames | video t={t/60:5.1f} min | {n/el:4.1f} fps", flush=True)
        if max_frames and n >= max_frames:
            break

    cols = ["frame", "t", "track_id", "cls", "conf", "x0", "y0", "x1", "y1", "u", "v", "x", "y", "edge"]
    tracks = pd.DataFrame(rows, columns=cols)
    frames = pd.DataFrame(frame_rows, columns=["frame", "t", "has_h", "n_people", "vis_near_y", "n_tiles", "roi_top"])
    ball_raw = pd.DataFrame(ball_rows, columns=["frame", "t", "conf", "u", "v", "x", "y"])

    # ---- teams
    tracks["team"] = -1
    team_report, assignment = {}, None
    if kit is not None and len(feats) >= 20:
        # supervised: the players clicked in the setup (nearest clicked samples), same rule as the live view
        F = np.asarray(feats)
        kit.fit(F)
        codes, margin = kit.predict(F, return_margin=True)
        assignment = vote_roles(tracks.loc[feat_rows, "track_id"].to_numpy(), codes)
        assignment = apply_manual(assignment, votes.manual)
        a = assignment.set_index("track_id")
        tracks["team"] = tracks.track_id.map(a.team).fillna(-1).astype(int)
        role = tracks.track_id.map(a.role)
        tracks.loc[role.isin(["goalkeeper", "referee"]), "cls"] = role[role.isin(["goalkeeper", "referee"])]
        team_report = dict(kit.report, crop_margin_median=float(np.median(margin)), samples=kit_report,
                           tracks_seeded_by_clicks=len(votes.manual))
        print(f"[teams] clicked kits: {int((assignment.decision == 'majority').sum())}/{len(assignment)} tracks by "
              f"clear majority, {len(votes.manual)} tracks seeded by your clicks; clicked samples recognised "
              f"{kit.report.get('loo_accuracy')} in leave-one-out")
    elif len(feats) >= 20:
        # unsupervised: fit on early frames, predict all crops, vote per track
        F = np.asarray(feats)
        fr = tracks.loc[feat_rows, "frame"].to_numpy()
        early = fr <= np.sort(np.unique(fr))[min(len(np.unique(fr)) - 1, cfg.teams.fit_frames)]
        teams.fit(F[early] if early.sum() >= 20 else F)
        lab = teams.predict(F)
        tv = vote(tracks.loc[feat_rows, "track_id"].to_numpy(), lab)
        tracks["team"] = tracks.track_id.map(tv).fillna(-1).astype(int)
        assignment = vote_roles(tracks.loc[feat_rows, "track_id"].to_numpy(), lab)
        team_report = dict(method="unsupervised KMeans(2) on the first %d sampled frames" % cfg.teams.fit_frames,
                           team_order="arbitrary (cluster index)")
        if cfg.teams.kit_colors and cfg.teams.backend == "color" and teams.km is not None:
            # cluster centres hold chroma quantiles: dims 2:4 = median a, b (scaled by 3/255)
            cen = teams.km.cluster_centers_[:, 2:4] * 255 / 3
            kits = cv2.cvtColor(np.uint8([[k[:3] for k in cfg.teams.kit_colors[:2]]]), cv2.COLOR_BGR2LAB)[0].astype(float)
            d = lambda i, j: float(np.linalg.norm(cen[i] - kits[j, 1:]))           # compare chroma (a, b) only
            team_report["team_order"] = "matched to teams.kit_colors (typed BGR colours)"
            if d(0, 1) + d(1, 0) < d(0, 0) + d(1, 1):                               # clusters are swapped
                tracks["team"] = tracks.team.map({0: 1, 1: 0, -1: -1}).astype(int)
                assignment["team"] = assignment.team.map({0: 1, 1: 0, -1: -1})
                print("[teams] cluster order swapped to match teams.kit_colors")
    tracks.loc[tracks.cls == "referee", "team"] = -1
    # people standing clearly outside the pitch (bench, staff, ball boys, crowd) are not players
    off = tracks.x.notna() & ~pitch.in_pitch(tracks.x.fillna(0), tracks.y.fillna(0), cfg.pitch.offpitch_margin_m)
    tracks.loc[off, "cls"] = "outside"
    tracks.loc[off, "team"] = -1

    camera_log = pd.DataFrame(provider.log) if getattr(provider, "log", None) else None
    if camera_log is not None and len(camera_log):
        tracks, ball_raw, frames = apply_camera_quality(tracks, ball_raw, frames, camera_log,
                                                        2.0 * cfg.pitch.ptz_max_err_px)
    ball = clean_ball(ball_raw, frames.frame.to_numpy(), frames.t.to_numpy(), cfg.ball.max_speed_ms,
                      cfg.ball.max_gap_s, cfg.ball.smooth_window)
    if "cam_ok" in frames:
        ball["cam_ok"] = frames.cam_ok.to_numpy()
    ball["vis_near_y"] = frames.vis_near_y.to_numpy()
    el = time.time() - t0
    if live is not None:
        live.finish("perception done — running the analytics", n=n)
    meta = {"video": info.__dict__, "n_frames_analysed": n, "seconds": el,
            "speed": dict(fps=round(n / el, 2) if el > 0 else None, detector=getattr(getattr(detector, "runner", None), "name", None),
                          tiles_per_frame=round(float(frames.n_tiles.mean()), 2) if len(frames) else None,
                          windows=[[round(a, 1), round(b, 1)] for a, b in windows]),
            "visibility": visibility_summary(frames, pitch),
            "scoreboard_changes": sb.changes if sb else [], "config": cfg.to_dict(), "teams": team_report,
            "seeds": seed_log}
    out = {"tracks": tracks, "ball": ball, "ball_raw": ball_raw, "frames": frames, "meta": meta}
    if assignment is not None:
        out["team_assignment"] = assignment                     # per-track votes: why each player got his team
    if camera_log is not None:
        out["camera_log"] = camera_log                          # per-frame pan / tilt / zoom / line error
    return out


# ----------------------------------------------------------------------------- helpers of the frame loop
def processing_windows(cfg: Config, duration_s: float) -> list:
    """Video stretches to analyse: ``[start_s, end_s]``, or only the periods (+ margin) when the setup gave them
    and ``video.only_periods`` is on — no detector time spent on the warm-up, half-time or the walk-off."""
    a0 = float(cfg.video.start_s or 0.0)
    b0 = float(cfg.video.end_s) if cfg.video.end_s is not None else float(duration_s)
    halves = cfg.match.halves or []
    if not (cfg.video.only_periods and halves):
        return [(a0, b0)]
    m = float(cfg.video.period_margin_s)
    w = []
    for a, b in sorted((float(a), float(b)) for a, b in halves):
        a, b = max(a0, a - m), min(b0, b + m)
        if b - a < 1.0:
            continue
        if w and a <= w[-1][1] + 1.0:                           # overlapping / touching: one window
            w[-1] = (w[-1][0], max(w[-1][1], b))
        else:
            w.append((a, b))
    return w or [(a0, b0)]


def pitch_roi(H, pitch: Pitch, shape, people_top: Optional[float] = None, margin_m: float = 4.0,
              head_room: float = 0.15, max_crop: float = 0.45):
    """Rows of the frame worth running the detector on: from just above the far touchline (the pitch + margin,
    projected with the previous frame's camera pose, + head room for people standing there) to the bottom.
    Safety: never below the highest person of the previous frame (a drifting camera pose must not hide players),
    never more than ``max_crop`` of the height. Returns ``(0, top, W, H)`` or None when the projection is not usable."""
    h, w = shape[:2]
    L, Wd = pitch.length, pitch.width
    xs = np.linspace(-margin_m, L + margin_m, 25)
    far = np.c_[xs, np.full_like(xs, -margin_m)]
    try:
        uv = H.to_image(far)
    except Exception:  # noqa: BLE001
        return None
    ok = np.isfinite(uv).all(1) & (np.abs(uv[:, 0]) < 4 * w) & (np.abs(uv[:, 1]) < 4 * h)
    ok &= (uv[:, 0] >= -0.2 * w) & (uv[:, 0] <= 1.2 * w)
    if ok.sum() < 3:
        return None
    top = float(np.min(uv[ok, 1])) - head_room * h
    if people_top is not None:
        top = min(top, people_top - 0.05 * h)
    top = int(np.clip(top, 0, max_crop * h))
    return (0, top, w, h) if top > 8 else None


def foot_points(xyxy: np.ndarray, shape) -> tuple:
    """Ground contact point of each box; for people cut by the bottom of the frame (near touchline, camera side)
    the feet are below the image: estimate them from the box width (standing person ~ 0.4 x as wide as tall).
    Returns (foot uv, edge flag)."""
    xyxy = np.asarray(xyxy, float).reshape(-1, 4)
    h = shape[0]
    u = (xyxy[:, 0] + xyxy[:, 2]) / 2
    v = xyxy[:, 3].copy()
    edge = xyxy[:, 3] >= h - 2
    if edge.any():
        est_h = (xyxy[:, 2] - xyxy[:, 0]) / 0.4
        v = np.where(edge, np.maximum(v, xyxy[:, 1] + est_h), v)
    return np.c_[u, v], edge


def visible_near_y(H, pitch: Pitch, shape) -> float:
    """Pitch y (m) seen at the bottom of the frame — where the camera's view ends on the near (camera) side.
    >= pitch width: the near touchline is in view. NaN: no calibration."""
    if H is None:
        return float("nan")
    h, w = shape[:2]
    uv = np.c_[np.linspace(0.05 * w, 0.95 * w, 9), np.full(9, h - 1.0)]
    try:
        xy = H.to_pitch(uv)
    except Exception:  # noqa: BLE001
        return float("nan")
    ok = np.isfinite(xy).all(1) & (xy[:, 0] >= 0) & (xy[:, 0] <= pitch.length)
    if not ok.any():
        return float("nan")
    return float(np.round(np.median(xy[ok, 1]), 2))


def visibility_summary(frames: pd.DataFrame, pitch: Pitch) -> dict:
    v = frames.vis_near_y.dropna() if "vis_near_y" in frames else pd.Series(dtype=float)
    if not len(v):
        return {}
    hidden = v < pitch.width - 0.5
    return dict(near_touchline_out_of_view_share=round(float(hidden.mean()), 3),
                median_near_edge_y_m=round(float(v.median()), 2), pitch_width_m=float(pitch.width))


def build_kit_bank(video_path: str, cfg: Config, teams: TeamClassifier, detector, info):
    """The players clicked in the setup -> classifier + seeds for the tracker.

    Each clicked box is first snapped to the detector's box of that player on that frame (dragged boxes are often
    loose and full of grass — on J3 that alone made the clicked samples unrecognisable), then described with the same
    descriptor as the match crops. Returns (KitClassifier | None, {frame: [(box, code)]}, report)."""
    scale = (cfg.video.resize_width / info.width) if cfg.video.resize_width else 1.0
    samples, seeds = {}, {}
    rep = dict(clicked=0, snapped=0, kept_as_drawn=0, dropped=0)
    by_frame = {}
    for cls, items in (cfg.teams.kit_samples or {}).items():
        for it in items:
            fr = int(it.get("frame", round(float(it["t"]) * info.fps)))
            by_frame.setdefault(fr, []).append((cls, it))
    cap = cv2.VideoCapture(video_path)
    try:
        for fr, items in sorted(by_frame.items()):
            cap.set(cv2.CAP_PROP_POS_FRAMES, fr)
            ok, img = cap.read()
            if not ok:
                continue
            if scale != 1.0:
                img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            people = detector(img).of("player", "goalkeeper", "referee")
            used = set()
            for cls, it in items:
                rep["clicked"] += 1
                box = np.asarray(it["box"], float) * scale
                sb = snap_box(box, people.xyxy)
                if sb is not None:
                    key = tuple(np.round(sb, 1))
                    if key in used:                              # same player clicked twice
                        continue
                    used.add(key)
                    rep["snapped"] += 1
                    box = sb
                elif person_like(box):
                    rep["kept_as_drawn"] += 1
                else:
                    rep["dropped"] += 1
                    continue
                d = teams.describe(img, box[None])[0]
                if d is None:
                    rep["dropped"] += 1
                    continue
                samples.setdefault(cls, []).append(d)
                seeds.setdefault(fr, []).append((box, CODE[cls]))
    finally:
        cap.release()
    try:
        kit = KitClassifier(samples, cfg.teams.outlier_quantile, cfg.teams.outlier_factor)
    except ValueError as e:
        print(f"[teams] kit samples ignored: {e}")
        return None, {}, rep
    rep.update(loo_accuracy=kit.report.get("loo_accuracy"), loo_n=kit.report.get("loo_n"))
    print(f"[teams] {rep['clicked']} clicked players: {rep['snapped']} snapped to the detector box, "
          f"{rep['kept_as_drawn']} kept as drawn, {rep['dropped']} dropped; leave-one-out "
          f"{kit.report.get('loo_accuracy')} of {kit.report.get('loo_n')}")
    return kit, seeds, rep


def _apply_seeds(seeds: dict, f: int, step: int, people, ids, tracker, votes: TrackVotes, log: list):
    """On (or next to) a frame where the user clicked players, the matching tracks take the clicked class."""
    near = [sf for sf in seeds if abs(sf - f) <= step and sf not in {e["frame"] for e in log}]
    for sf in near:
        hits = 0
        for box, code in seeds[sf]:
            if not len(people):
                break
            iou = iou_matrix(np.asarray(box, float)[None], people.xyxy)[0]
            j = int(np.argmax(iou))
            if iou[j] < 0.4:
                continue
            tid = int(ids[j]) if ids[j] >= 0 else (tracker.id_for(people.xyxy[j]) if hasattr(tracker, "id_for") else -1)
            if tid >= 0:
                votes.manual[tid] = int(code)
                hits += 1
        log.append(dict(frame=int(sf), matched=hits, clicked=len(seeds[sf])))


def apply_camera_quality(tracks: pd.DataFrame, ball_raw: pd.DataFrame, frames: pd.DataFrame,
                         camera_log: pd.DataFrame, max_err_px: float = 6.0, window: int = 10):
    """Moving camera: blank pitch coordinates while the camera calibration is unreliable.

    Quality = rolling median (``window`` analysed frames) of the per-frame line-alignment error; frames above
    ``max_err_px`` (or with no pose) get NaN positions, so they cannot pollute positions-based statistics.
    """
    cl = camera_log.set_index("frame")
    q = cl.line_err.rolling(window, center=True, min_periods=1).median()
    ok = (q <= max_err_px) & cl.pan.notna()
    frames = frames.copy()
    frames["cam_ok"] = frames.frame.map(ok).fillna(False).astype(bool)
    bad = set(frames.frame[~frames.cam_ok])
    tracks, ball_raw = tracks.copy(), ball_raw.copy()
    tracks.loc[tracks.frame.isin(bad), ["x", "y"]] = np.nan
    ball_raw.loc[ball_raw.frame.isin(bad), ["x", "y"]] = np.nan
    return tracks, ball_raw, frames


def save_perception(res: dict, out_dir: str):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for k in ("tracks", "ball", "ball_raw", "frames", "camera_log", "team_assignment"):
        if k in res:
            res[k].to_csv(out / f"{k}.csv", index=False)
    with open(out / "meta.json", "w") as f:
        json.dump(res["meta"], f, indent=2, default=str)


def load_perception(out_dir: str) -> dict:
    out = Path(out_dir)
    res = {k: pd.read_csv(out / f"{k}.csv") for k in ("tracks", "ball", "frames", "camera_log", "team_assignment")
           if (out / f"{k}.csv").exists()}
    if (out / "meta.json").exists():
        res["meta"] = json.load(open(out / "meta.json"))
    return res


# ----------------------------------------------------------------------------- camera repair (review UI)
def repair_run(video_path: str, cfg: Config, run_dir: str, repairs: list, progress: bool = True) -> list:
    """Re-process only the windows of the camera repairs accepted in the review UI and splice them into the run.

    Each repair is a camera pose the user validated on one frame (clicked landmarks -> PTZ pose). The tracker restarts
    from it at the first analysed frame after the anchor and runs to the end of the lost stretch (+5 s, max 3 min,
    :func:`review.repair_window`). Rows of those frames are replaced in tracks / ball_raw / frames / camera_log; the
    ball is re-cleaned over the whole run. Returns the list of repairs with ``applied`` and the measured effect.
    """
    import copy
    from .review import repair_window
    d = Path(run_dir)
    tracks = pd.read_csv(d / "tracks.csv")
    frames = pd.read_csv(d / "frames.csv")
    ball_raw = pd.read_csv(d / "ball_raw.csv")
    cam_log = pd.read_csv(d / "camera_log.csv") if (d / "camera_log.csv").exists() else None
    ta = pd.read_csv(d / "team_assignment.csv") if (d / "team_assignment.csv").exists() else None
    meta = json.loads((d / "meta.json").read_text()) if (d / "meta.json").exists() else {}
    if cfg.pitch.homography != "ptz" or cam_log is None:
        raise ValueError("camera repair needs a PTZ run (pitch.homography: ptz) with camera_log.csv")
    info = probe(video_path)
    grid = np.sort(frames.frame.unique())
    out = []
    for r in repairs:
        if r.get("applied"):
            out.append(r)
            continue
        a_frame = int(r["frame"])
        later = grid[grid >= a_frame]
        if not len(later):
            out.append(dict(r, applied=False, error="anchor after the last analysed frame"))
            continue
        f_start = int(later[0])
        t0, t1 = repair_window(frames, f_start / info.fps)
        c = copy.deepcopy(cfg)
        c.video.start_s, c.video.end_s = (f_start + 0.01) / info.fps, t1
        c.pitch.extra_anchors = {a_frame: list(map(float, r["pose"]))}
        before = frames[(frames.t >= t0) & (frames.t <= t1)]
        cov0 = float(before.cam_ok.mean()) if "cam_ok" in before and len(before) else float("nan")
        if progress:
            print(f"[repair] {t0 / 60:5.1f}-{t1 / 60:5.1f} min from your pose at {a_frame / info.fps:.1f}s "
                  f"(camera usable before: {100 * cov0:.0f} %)", flush=True)
        w = run_perception(video_path, c, progress=False)
        W = set(w["frames"].frame)
        off = int(max(tracks.track_id.max(), 0)) + 1
        wt = w["tracks"].copy()
        wt["track_id"] = np.where(wt.track_id >= 0, wt.track_id + off, -1)
        tracks = pd.concat([tracks[~tracks.frame.isin(W)], wt], ignore_index=True)
        ball_raw = pd.concat([ball_raw[~ball_raw.frame.isin(W)], w["ball_raw"]], ignore_index=True)
        wf = w["frames"]
        frames = pd.concat([frames[~frames.frame.isin(W)], wf], ignore_index=True)
        if "camera_log" in w:
            wl = w["camera_log"].copy()
            wl["src"] = "repair:" + wl.src.astype(str)
            cam_log = pd.concat([cam_log[~cam_log.frame.isin(W)], wl], ignore_index=True)
        if ta is not None and w.get("team_assignment") is not None:
            wa = w["team_assignment"].copy()
            wa["track_id"] = wa.track_id + off
            ta = pd.concat([ta, wa], ignore_index=True)
        cov1 = float(wf.cam_ok.mean()) if "cam_ok" in wf and len(wf) else float("nan")
        if progress:
            print(f"         camera usable after: {100 * cov1:.0f} % of {len(wf)} frames", flush=True)
        out.append(dict(r, applied=True, window_used=[round(t0, 2), round(t1, 2)], cam_ok_before=round(cov0, 3),
                        cam_ok_after=round(cov1, 3)))
        meta.setdefault("repairs", []).append(dict(t=r["t"], window=[t0, t1], cam_ok_before=cov0, cam_ok_after=cov1))
    tracks = tracks.sort_values(["frame", "track_id"]).reset_index(drop=True)
    frames = frames.sort_values("frame").reset_index(drop=True)
    ball_raw = ball_raw.sort_values("frame").reset_index(drop=True)
    ball = clean_ball(ball_raw, frames.frame.to_numpy(), frames.t.to_numpy(), cfg.ball.max_speed_ms,
                      cfg.ball.max_gap_s, cfg.ball.smooth_window)
    if "cam_ok" in frames:
        ball["cam_ok"] = frames.cam_ok.astype(bool).to_numpy()
    res = {"tracks": tracks, "ball": ball, "ball_raw": ball_raw, "frames": frames, "meta": meta,
           "camera_log": cam_log.sort_values("frame").reset_index(drop=True)}
    if ta is not None:
        res["team_assignment"] = ta
    save_perception(res, str(d))
    return out
