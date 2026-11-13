"""Two-pass pipeline: video -> tracks + ball -> annotated video.

Pass 1 (analyse): pitch mask, detect people + ball (YOLOX), track people (IoU tracker), collect shirt colours,
                  track the ball.
Then:             decide one team per track (whole clip), interpolate / smooth the ball path.
Pass 2 (render):  re-read the video and draw silhouettes, ball trail, legend and team ball control: a team has the
                  ball from the moment one of its players is closest to it until a player of the other team is.

Outputs in ``out_dir``: ``annotated.mp4``, ``tracks.csv`` (frame, t, track_id, team, box) and ``ball.csv``.
"""
from __future__ import annotations

import os
import time
from collections import deque

import cv2
import numpy as np
import pandas as pd

from ..detection import YoloxOnnxDetector
from ..tracking import IoUTracker
from .ball import BallTracker, fill_gaps
from .draw import draw_frame
from .segment import field_mask
from .teams import TeamColors


def _frames(path, start_s, duration_s, stride):
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    first = int(round(start_s * fps))
    last = first + int(round(duration_s * fps)) if duration_s else int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, first)
    i = first
    while i < last:
        ok, img = cap.read()
        if not ok:
            break
        if (i - first) % stride == 0:
            yield i, i / fps, img
        i += 1
    cap.release()


class CameraMotion:
    """Global image shift between consecutive frames (camera pan), by phase correlation on a small grey image.

    Fast pans move every player and the ball by 100+ px per frame; shifting the predictions by the camera motion
    keeps the people tracker and the ball Kalman filter locked on."""

    def __init__(self, width: int = 480):
        self.width, self.prev, self.win = width, None, None

    def __call__(self, img):
        s = self.width / img.shape[1]
        g = cv2.cvtColor(cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
        g = g.astype(np.float32)
        if self.win is None:
            self.win = cv2.createHanningWindow((g.shape[1], g.shape[0]), cv2.CV_32F)
        prev, self.prev = self.prev, g
        if prev is None:
            return 0.0, 0.0
        (dx, dy), resp = cv2.phaseCorrelate(prev, g, self.win)
        return (dx / s, dy / s) if resp > 0.2 else (0.0, 0.0)


def _writer(path, fps, size):
    for fourcc in ("avc1", "mp4v"):                       # H.264 when available (plays everywhere)
        w = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*fourcc), fps, size)
        if w.isOpened():
            return w
    raise RuntimeError("could not open a video writer")


def run(video, out_dir, start_s=0.0, duration_s=None, stride=1, model="models/yolox.onnx",
        tile=1280, conf_player=0.35, conf_ball=0.05, roi_top=0.0, upscale=1.0, team_every=3, trail_len=20):
    """``upscale`` > 1 enlarges low-resolution video (e.g. 640x360 streams) before detection: far more players
    are found, at the cost of speed."""
    os.makedirs(out_dir, exist_ok=True)
    fps = (cv2.VideoCapture(video).get(cv2.CAP_PROP_FPS) or 25.0) / stride
    det = YoloxOnnxDetector(model, conf_player=conf_player, conf_ball=conf_ball, tile=tile, overlap=96)
    people_tracker = IoUTracker(max_age=int(fps), max_center_dist=80.0)
    ball_tracker = BallTracker(det, max_lost=max(3, int(0.5 * fps)), max_still=int(1.5 * fps))
    teams = TeamColors()
    camera = CameraMotion()

    # ---------------- pass 1: analyse ----------------
    rows, balls, t0 = [], [], time.time()
    for n, (f, t, img) in enumerate(_frames(video, start_s, duration_s, stride)):
        H, W = img.shape[:2]
        dx, dy = camera(img)                                # move predictions with the camera pan
        for tr in people_tracker.tracks:
            tr.box = tr.box + np.array([dx, dy, dx, dy])
        ball_tracker.shift(dx, dy)
        big = img if upscale == 1 else cv2.resize(img, None, fx=upscale, fy=upscale, interpolation=cv2.INTER_CUBIC)
        d = det(big, roi=(0, roi_top * H * upscale, W * upscale, H * upscale))
        d.xyxy = d.xyxy / upscale
        field = field_mask(img)
        people = d.of("player")
        feet = people.foot.astype(int)                      # keep people standing on the pitch
        people = people.select([field[min(y, H - 1), min(x, W - 1)] > 0 for x, y in feet])
        ids = people_tracker.update(people)
        if n % team_every == 0:
            teams.add(big, people.xyxy * upscale, ids)      # shirt colours from the enlarged image
        b = d.of("ball")
        res = ball_tracker.update(img, (b.xyxy, b.conf), people.xyxy, field)
        balls.append(dict(frame=f, t=t, x=res[0] if res else np.nan, y=res[1] if res else np.nan,
                          r=res[2] if res else np.nan, state=res[3] if res else None, cam_dx=dx, cam_dy=dy))
        if ball_tracker.erase:                              # a static object was followed: those frames had no ball
            for row in balls[-ball_tracker.erase:]:
                row.update(x=np.nan, y=np.nan, r=np.nan, state=None)
            ball_tracker.erase = 0
        for box, tid in zip(people.xyxy, ids):
            if tid >= 0:
                rows.append(dict(frame=f, t=t, track_id=int(tid), x0=box[0], y0=box[1], x1=box[2], y1=box[3]))
        if n % 50 == 0:
            print(f"[pass 1] frame {f}  t={t:7.1f}s  people={len(people)}  ball={res[3] if res else '-'}  "
                  f"{(time.time() - t0) / (n + 1):.2f}s/frame")

    tracks = pd.DataFrame(rows)
    teams.fit()
    tracks["team"] = [teams.team(t) for t in tracks.track_id] if len(tracks) else []
    ball = fill_gaps(pd.DataFrame(balls), max_gap=int(0.5 * fps))
    tracks.to_csv(os.path.join(out_dir, "tracks.csv"), index=False)
    ball.to_csv(os.path.join(out_dir, "ball.csv"), index=False)
    seen = ball.x.notna().mean() * 100
    print(f"[ball] visible in {seen:.0f}% of frames (detected {ball.state.eq('det').mean() * 100:.0f}%, "
          f"interpolated {ball.state.eq('interp').mean() * 100:.0f}%)")

    # ---------------- pass 2: render ----------------
    by_frame = {f: g for f, g in tracks.groupby("frame")} if len(tracks) else {}
    ball_at = ball.set_index("frame")
    trail, writer = deque(maxlen=trail_len), None
    control, last_team = {0: 0, 1: 0}, None                 # frames of ball control per team
    out_path = os.path.join(out_dir, "annotated.mp4")
    for f, t, img in _frames(video, start_s, duration_s, stride):
        g = by_frame.get(f)
        players = [] if g is None else [((r.x0, r.y0, r.x1, r.y1), r.track_id, r.team) for r in g.itertuples()]
        b = ball_at.loc[f]
        bxy = None if pd.isna(b.x) else (b.x, b.y, b.r, b.state)
        trail = deque(((x + b.cam_dx, y + b.cam_dy) for x, y in trail), maxlen=trail_len)  # trail follows the pan
        if bxy is not None:
            if trail and np.hypot(b.x - trail[-1][0], b.y - trail[-1][1]) > 0.1 * img.shape[1]:
                trail.clear()                               # a jump = new ball track, don't draw a line
            trail.append((b.x, b.y))
        elif trail:
            trail.popleft()
        carrier = _carrier(players, bxy)
        team = next((tm for _, tid, tm in players if tid == carrier), None)
        if team in control:
            last_team = team                                # a team keeps control until the other one gets the ball
        if last_team is not None:
            control[last_team] += 1
        n = control[0] + control[1]
        share = (control[0] / n, control[1] / n) if n else None
        frame = draw_frame(img, players, bxy, list(trail), teams.colors, carrier, share)
        if writer is None:
            writer = _writer(out_path, fps, (frame.shape[1], frame.shape[0]))
        writer.write(frame)
    if writer is not None:
        writer.release()
    n = max(control[0] + control[1], 1)
    print(f"[ball control] team A {control[0] / n:.0%}  team B {control[1] / n:.0%}  ({n} frames counted)")
    print(f"[done] {out_path}")
    return out_path


def _carrier(players, ball, max_dist=1.5):
    """Track id of the player whose feet are closest to the ball (within ``max_dist`` x box width)."""
    if ball is None or not players:
        return None
    best, best_d = None, np.inf
    for (x0, y0, x1, y1), tid, _ in players:
        w = x1 - x0
        d = np.hypot((x0 + x1) / 2 - ball[0], y1 - ball[1]) / max(w, 1)
        if d < best_d:
            best, best_d = tid, d
    return best if best_d <= max_dist else None
