"""Debug the simple tracker step by step on one moment of a video.

    python debug_simple.py data/videos/J3_MansoresSanguedo_1080p.mp4 --time 24:40 --roi-top 0.2

Writes one image per step to outputs/simple/debug/<video>_<time>/ and prints what each step found, so you can see
exactly where something goes wrong (pitch mask? detector? team colours? silhouette? ball?).

MAP OF THE MAIN FUNCTIONS (in the order the real run calls them)
================================================================
simple_track.py                      command line -> run()
football_eval/simple/run.py          run(video, out_dir, ...)            the whole two-pass pipeline
  PASS 1, every frame:
    run.py       CameraMotion()(img)                  camera pan (dx, dy) between frames      -> step_camera
    segment.py   field_mask(img)                      pitch area (0/255)                      -> step_field
    detection.py YoloxOnnxDetector()(img, roi)        people + ball boxes (existing module)   -> step_detect
    tracking.py  IoUTracker.update(people)            track ids (existing module)
    teams.py     TeamColors.add(img, boxes, ids)      collects shirt_feature() per player     -> step_teams
    ball.py      BallTracker.update(img, balls, people, field)
                   ._zoom_candidates()   detector on a 2x crop around the prediction       -> step_ball
                   ._blob_candidates()   white-blob fallback
                   ._static()            drops "balls" that never move (trees, posts)
  AFTER PASS 1:
    teams.py     TeamColors.fit()  -> _cluster()      2 kits + "other", one label per track
    ball.py      fill_gaps(ball_df)                   interpolate short gaps, smooth
  PASS 2, every frame:
    run.py       _carrier(players, ball)              player closest to the ball (ball control)
    draw.py      draw_frame(...)                      outlines, ball trail, legend, ball control bar
                   -> segment.py player_mask(img, box)  generous silhouette of one player     -> step_masks
"""
import argparse
import os

import cv2
import numpy as np

from football_eval.detection import YoloxOnnxDetector
from football_eval.simple.ball import BallTracker
from football_eval.simple.draw import draw_frame
from football_eval.simple.run import CameraMotion
from football_eval.simple.segment import field_mask, player_mask
from football_eval.simple.teams import TeamColors, shirt_feature
from simple_track import seconds


def read(cap, fps, t):
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(round(t * fps)))
    ok, img = cap.read()
    if not ok:
        raise SystemExit(f"no frame at {t:.1f}s")
    return img


def step_camera(cap, fps, t):
    """Camera shift between the previous frame and this one."""
    cam = CameraMotion()
    cam(read(cap, fps, t - 1 / fps))
    dx, dy = cam(read(cap, fps, t))
    print(f"[camera] shift since previous frame: dx={dx:.1f}px dy={dy:.1f}px")


def step_field(img, out):
    m = field_mask(img)
    vis = img.copy()
    vis[m == 0] = (vis[m == 0] * 0.3).astype(np.uint8)
    cv2.imwrite(f"{out}/02_field_mask.jpg", vis)
    print(f"[field] pitch = {(m > 0).mean():.0%} of the frame (dark = ignored)")
    return m


def step_detect(det, img, field, roi_top, out):
    H, W = img.shape[:2]
    d = det(img, roi=(0, roi_top * H, W, H))
    people, balls = d.of("player"), d.of("ball")
    vis = img.copy()
    for (x0, y0, x1, y1), c in zip(people.xyxy.astype(int), people.conf):
        on = field[min(y1, H - 1), min((x0 + x1) // 2, W - 1)] > 0
        cv2.rectangle(vis, (x0, y0), (x1, y1), (0, 255, 0) if on else (0, 0, 255), 2)
        cv2.putText(vis, f"{c:.2f}", (x0, y0 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    for (x0, y0, x1, y1), c in zip(balls.xyxy.astype(int), balls.conf):
        cv2.circle(vis, ((x0 + x1) // 2, (y0 + y1) // 2), 14, (0, 255, 255), 2)
        cv2.putText(vis, f"ball {c:.2f}", (x1 + 4, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    cv2.imwrite(f"{out}/03_detections.jpg", vis)
    print(f"[detect] {len(people)} people (green = feet on the pitch, red = ignored), "
          f"{len(balls)} ball candidates: {np.round(balls.conf, 2).tolist()}")
    keep = [field[min(int(y), H - 1), min(int(x), W - 1)] > 0 for x, y in people.foot]
    return people.select(keep), balls


def step_masks(img, people, out):
    vis = img.copy()
    for box in people.xyxy:
        r = player_mask(img, box)
        if r is None:
            cv2.rectangle(vis, tuple(box[:2].astype(int)), tuple(box[2:].astype(int)), (0, 0, 255), 2)
            continue
        m, x0, y0 = r
        cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(vis, [c + (x0, y0) for c in cnts], -1, (255, 0, 255), 2)
    cv2.imwrite(f"{out}/04_masks.jpg", vis)
    print("[masks] magenta = silhouette, red box = no silhouette (drawn as an ellipse in the video)")


def step_teams(cap, fps, t, det, roi_top, img, people, out, n_frames=15):
    """Fit the team colours on n_frames around t (every detection is its own 'track' here)."""
    tc = TeamColors()
    k = 0
    for tt in np.linspace(max(t - 5, 0), t + 5, n_frames):
        f = read(cap, fps, tt)
        H, W = f.shape[:2]
        p = det(f, roi=(0, roi_top * H, W, H)).of("player")
        tc.add(f, p.xyxy, np.arange(k, k + len(p)))
        k += len(p)
    tc.fit()
    vis = img.copy()
    for i, box in enumerate(people.xyxy):
        fe = shirt_feature(img, box)
        team = "-" if fe is None else {0: "A", 1: "B", -1: "other"}[_nearest(tc, fe)]
        x0, y0 = box[:2].astype(int)
        cv2.putText(vis, f"{i}:{team}", (x0, y0 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        print(f"   player {i:2d} box={box.astype(int).tolist()} feature(a, b, light)="
              f"{None if fe is None else np.round(fe, 1).tolist()} -> {team}")
    cv2.imwrite(f"{out}/05_teams.jpg", vis)
    return tc


def _nearest(tc, fe):
    """Team of one feature: majority label of the 5 closest collected samples."""
    X = np.stack([f for _, f in tc.samples])
    near = np.argsort(np.linalg.norm(X - fe, axis=1))[:5]
    labels = [tc.team(tc.samples[i][0]) for i in near]
    return max(set(labels), key=labels.count)


def step_ball(det, img, balls, people, field, out):
    """Zoomed search around the best full-frame ball candidate (what the tracker does around its prediction)."""
    if len(balls) == 0:
        print("[ball] no full-frame candidate: the tracker would rely on its prediction + zoom search")
        return None
    b = balls.xyxy[int(np.argmax(balls.conf))]
    cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
    bt = BallTracker(det)
    zc, (x0, y0, crop) = bt._zoom_candidates(img, cx, cy)
    blobs = bt._blob_candidates(x0, y0, crop, people.xyxy)
    vis = cv2.resize(crop, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    for x, y, s, r in zc:
        cv2.circle(vis, (int(2 * (x - x0)), int(2 * (y - y0))), 16, (0, 255, 255), 2)
    for x, y, s, r in blobs:
        cv2.circle(vis, (int(2 * (x - x0)), int(2 * (y - y0))), 10, (255, 0, 255), 1)
    cv2.imwrite(f"{out}/06_ball_zoom.jpg", vis)
    print(f"[ball] zoom search: {len(zc)} detector hits {[round(c[2], 2) for c in zc]}, "
          f"{len(blobs)} white blobs (yellow / magenta circles)")
    return cx, cy, max((b[2] - b[0]) / 2, 3), "det"


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video")
    ap.add_argument("--time", default="0", help="moment to inspect, e.g. 24:40")
    ap.add_argument("--roi-top", type=float, default=0.0)
    ap.add_argument("--model", default="models/yolox.onnx")
    a = ap.parse_args()
    t = seconds(a.time)
    name = os.path.splitext(os.path.basename(a.video))[0]
    out = os.path.join("outputs", "simple", "debug", f"{name}_{a.time.replace(':', '-')}")
    os.makedirs(out, exist_ok=True)

    cap = cv2.VideoCapture(a.video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    img = read(cap, fps, t)
    cv2.imwrite(f"{out}/01_frame.jpg", img)
    det = YoloxOnnxDetector(a.model, conf_player=0.35, conf_ball=0.05, tile=1280, overlap=96)

    step_camera(cap, fps, t)
    field = step_field(img, out)
    people, balls = step_detect(det, img, field, a.roi_top, out)
    step_masks(img, people, out)
    tc = step_teams(cap, fps, t, det, a.roi_top, img, people, out)
    ball = step_ball(det, img, balls, people, field, out)
    players = [(box, i, _nearest(tc, f) if (f := shirt_feature(img, box)) is not None else -1)
               for i, box in enumerate(people.xyxy)]
    cv2.imwrite(f"{out}/07_overlay.jpg", draw_frame(img, players, ball, [], tc.colors))
    print(f"[done] images in {out}")
