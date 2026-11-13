"""Ball tracking for the main pipeline (``ball.detector: tracker``, the default for HD video).

Why the plain detector misses the ball: the 1080p frame is cut into ~1024-px tiles that the network sees at 640 px,
so a 8-12 px ball becomes 5-7 px — below what a COCO "sports ball" detector fires on (Canedo full match: ball in
10 % of the frames). The tracker of the simple pipeline (:mod:`football_eval.simple.ball`) does much better on the
same footage, so it is reused here and adapted to the main pipeline:

1. **Kalman filter** on the ball in image pixels; the PTZ camera motion between two analysed frames
   (``PTZProvider.last_image_motion``) is applied to the filter, so a pan does not look like a ball jump;
2. **zoomed re-detection**: a 320-px window around the prediction is upscaled x2 (the ball becomes ~20 px);
3. **acquisition scan** while no ball is tracked: every ``scan_every`` analysed frames, a few windows at native
   resolution (no down-scaling) along the band of the grass the camera is centred on — the operator follows the
   ball, so it is almost always there;
4. white-blob fallback, static-object filter (trees, posts, spots on the ad boards), candidates only on / just above
   the grass, never on a player's upper body.

Returns per analysed frame one measured position or nothing; gaps are interpolated later by
:func:`football_eval.ball.clean_ball` (speed gating in pitch metres).
"""
from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

from .simple.ball import BallTracker


class PitchBallTracker(BallTracker):
    def __init__(self, detector, analysis_fps: float = 10.0, width: int = 1920, scan_every: int = 2,
                 scan_tile: int = 640, ignore=(), use_motion: bool = True, **kw):
        fps_scale = 25.0 / max(analysis_fps, 1e-6)          # the simple tracker was tuned per video frame (~25 fps)
        kw.setdefault("gate", max(60.0, 0.045 * width) * min(2.5, max(1.0, fps_scale / 2)))
        kw.setdefault("max_lost", max(3, int(round(0.6 * analysis_fps))))       # coast <= 0.6 s
        kw.setdefault("max_still", max(8, int(round(2.0 * analysis_fps))))      # static object after 2 s
        kw.setdefault("window", int(round(160 * width / 1920)))
        kw.setdefault("init_conf", 0.30)
        kw.setdefault("zoom_conf", 0.12)
        super().__init__(detector, **kw)
        self.scan_every, self.scan_tile = scan_every, scan_tile
        self.k = 0
        self.stats = dict(frames=0, det_full=0, det_zoom=0, det_blob=0, det_scan=0, det_motion=0, pred=0, none=0,
                          erased=0)
        self.motion = None
        if use_motion:                        # moving bright blobs at 640 px (camera motion compensated)
            from .ballcv import MotionBallDetector
            self.ms = 640.0 / width
            self.motion = MotionBallDetector([[v * self.ms for v in r] for r in ignore])

    def _motion(self, img, people, gtop, M):
        if self.motion is None:
            return []
        h, w = img.shape[:2]
        s = self.ms
        small = cv2.resize(img, (int(round(w * s)), int(round(h * s))), interpolation=cv2.INTER_AREA)
        Ms = None
        if M is not None:
            S = np.diag([s, s, 1.0])
            Ms = S @ np.asarray(M, float) @ np.linalg.inv(S)
        field = _field_from_top(gtop, img.shape)[::int(round(1 / s)) or 1, ::int(round(1 / s)) or 1] \
            if gtop is not None else None
        if field is not None and field.shape[:2] != small.shape[:2]:
            field = cv2.resize(field, (small.shape[1], small.shape[0]), interpolation=cv2.INTER_NEAREST)
        out = []
        for u, v, sc in self.motion(small, Ms, field, np.asarray(people, float).reshape(-1, 4) * s):
            r = 4.0
            out.append((u / s, v / s - r, float(np.clip(0.04 * sc, 0.02, 0.2)), r))   # weak: needs the track gate
        return out

    # ---------------------------------------------------------------- camera motion
    def warp(self, M: Optional[np.ndarray]):
        """Apply the camera's image motion (3x3, previous frame -> this frame) to the filter and banned spots."""
        if M is None:
            return
        M = np.asarray(M, float)

        def tr(x, y):
            q = M @ np.array([x, y, 1.0])
            return (q[0] / q[2], q[1] / q[2]) if abs(q[2]) > 1e-9 else (x, y)
        if self.kf is not None:
            x, y, vx, vy = self.kf.statePost[:, 0]
            nx, ny = tr(x, y)
            ex, ey = tr(x + vx, y + vy)
            self.kf.statePost[:, 0] = np.array([nx, ny, ex - nx, ey - ny], np.float32)
        self.banned = [tr(x, y) for x, y in self.banned]
        self.recent.clear()

    # ---------------------------------------------------------------- acquisition
    def _scan(self, img, gtop: Optional[np.ndarray]):
        """Native-resolution windows over the grass near the image centre (the camera follows the ball)."""
        H, W = img.shape[:2]
        T = min(self.scan_tile, H, W)
        if gtop is not None and np.isfinite(gtop).any():
            top = float(np.nanmedian(gtop[int(0.3 * W):int(0.7 * W)]))
        else:
            top = 0.35 * H
        y0 = int(np.clip(top - 0.15 * T, 0, H - T))
        out = []
        n = int(np.ceil(W / T))
        for x0 in np.linspace(0, W - T, n).astype(int):             # the whole width of the grass band
            _, _, boxes, scores = self.det._infer(img[y0:y0 + T, x0:x0 + T])
            for (a, b, c, d), s in zip(boxes, scores):
                out.append(((a + c) / 2 + x0, (b + d) / 2 + y0, float(s), max(c - a, d - b) / 2))
        return out

    # ---------------------------------------------------------------- one analysed frame
    def step(self, img, ball_boxes, ball_scores, people, gtop=None, motion=None):
        """Returns ``(u, v, radius, source)`` for a measured ball (source full | zoom | blob | scan) or None."""
        self.k += 1
        self.stats["frames"] += 1
        self.warp(motion)
        field = _field_from_top(gtop, img.shape) if gtop is not None else None
        n_full = len(ball_scores)
        tracking_before = self.kf is not None
        mc = self._motion(img, people, gtop, motion)
        if mc:
            ball_boxes = np.r_[np.asarray(ball_boxes, float).reshape(-1, 4),
                               np.array([[x - r, y - r, x + r, y + r] for x, y, _, r in mc])]
            ball_scores = np.r_[np.asarray(ball_scores, float), [c[2] for c in mc]]
        n_motion = len(mc)
        if not tracking_before and self.k % self.scan_every == 0:
            sc = self._scan(img, gtop)
            if len(sc):
                b2 = np.array([[x - r, y - r, x + r, y + r] for x, y, s, r in sc])
                ball_boxes = np.r_[np.asarray(ball_boxes, float).reshape(-1, 4), b2]
                ball_scores = np.r_[np.asarray(ball_scores, float), [s for _, _, s, _ in sc]]
        r = self.update(img, (np.asarray(ball_boxes, float).reshape(-1, 4), np.asarray(ball_scores, float)),
                        np.asarray(people, float).reshape(-1, 4), field)
        if self.erase:
            self.stats["erased"] += self.erase
        if r is None:
            self.stats["none"] += 1
            return None
        x, y, rad, state = r
        if state != "det":
            self.stats["pred"] += 1
            return None
        src = self._source(x, y, ball_boxes, n_full, tracking_before, n_motion)
        self.stats["det_" + src] += 1
        return float(x), float(y), float(rad), src

    def _source(self, x, y, boxes, n_full, tracking_before, n_motion=0) -> str:
        B = np.asarray(boxes, float).reshape(-1, 4)
        if len(B):
            c = np.c_[(B[:, 0] + B[:, 2]) / 2, (B[:, 1] + B[:, 3]) / 2]
            d = np.hypot(c[:, 0] - x, c[:, 1] - y)
            j = int(np.argmin(d))
            if d[j] < max(8.0, 2 * self.radius):
                return "full" if j < n_full else ("motion" if j < n_full + n_motion else "scan")
        return "zoom" if tracking_before else "full"


def _field_from_top(gtop: np.ndarray, shape, above_frac: float = 0.12) -> np.ndarray:
    """Pixels where a ball may be: on the grass or up to ``above_frac`` of the height above its far edge (lofted)."""
    h, w = shape[:2]
    top = np.clip(np.asarray(gtop, float) - above_frac * h, 0, h)
    return (np.arange(h)[:, None] >= top[None, :w]).astype(np.uint8)
