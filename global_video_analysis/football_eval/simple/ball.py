"""Ball tracking in image pixels: Kalman filter + zoomed re-detection around the prediction.

Why this works better than "take the detector's ball box each frame":

1. **Kalman filter** (constant velocity): predicts where the ball should be in the next frame and
   rejects candidates that would need an impossible jump (players' heads, socks, line marks).
2. **Zoom search**: COCO detectors rarely fire on a 6-12 px ball. Around the predicted position we
   crop a small window, upscale it x2 and run the detector again: the ball becomes 15-25 px and is
   detected far more often.
3. **White-blob fallback** inside the same window (small, round, bright, low saturation), for frames
   where even the zoomed detector misses.
4. **Static-object filter**: a "ball" that does not move for ``max_still`` frames with no player near it is a
   tree / post / sign. The track is dropped (also in the frames already returned: see ``erase``) and that spot is
   ignored from then on.
5. **Offline clean-up** (:func:`fill_gaps`): short gaps are interpolated, the path is lightly smoothed.
"""
from __future__ import annotations

from collections import deque

import cv2
import numpy as np


def _kalman(x: float, y: float) -> cv2.KalmanFilter:
    k = cv2.KalmanFilter(4, 2)                      # state (x, y, vx, vy), measurement (x, y)
    k.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
    k.measurementMatrix = np.eye(2, 4, dtype=np.float32)
    k.processNoiseCov = np.diag([1, 1, 4, 4]).astype(np.float32)
    k.measurementNoiseCov = np.eye(2, dtype=np.float32) * 4
    k.errorCovPost = np.eye(4, dtype=np.float32) * 50
    k.statePost = np.array([[x], [y], [0], [0]], np.float32)
    return k


class BallTracker:
    """Call :meth:`update` once per frame. Returns ``(x, y, radius, state)`` with state 'det' (measured),
    'pred' (coasting on the prediction) or None (no ball)."""

    def __init__(self, detector, window: int = 160, zoom: float = 2.0, gate: float = 60.0,
                 max_lost: int = 12, init_conf: float = 0.25, zoom_conf: float = 0.10, use_blobs: bool = True,
                 max_still: int = 36):
        self.det, self.window, self.zoom, self.gate, self.zoom_conf = detector, window, zoom, gate, zoom_conf
        self.max_lost, self.init_conf, self.use_blobs, self.max_still = max_lost, init_conf, use_blobs, max_still
        self.kf, self.lost, self.radius = None, 0, 6.0
        self.still, self.last_xy, self.banned = 0, None, []   # static-object filter
        self.erase = 0                                # frames the caller must blank (they were a static object)
        self.width = 1920
        self.recent = deque(maxlen=3)               # candidates of the last frames, used to (re)start a track

    # -- candidate sources -------------------------------------------------------------------------
    def _zoom_candidates(self, img, cx, cy):
        """Run the detector on an upscaled window centred on (cx, cy)."""
        H, W = img.shape[:2]
        w = self.window
        x0, y0 = int(np.clip(cx - w, 0, max(W - 2 * w, 0))), int(np.clip(cy - w, 0, max(H - 2 * w, 0)))
        crop = img[y0:y0 + 2 * w, x0:x0 + 2 * w]
        if crop.size == 0:
            return [], (x0, y0, crop)
        big = cv2.resize(crop, None, fx=self.zoom, fy=self.zoom, interpolation=cv2.INTER_CUBIC)
        _, _, boxes, scores = self.det._infer(big)  # YOLOX "sports ball" outputs at a low threshold
        out = []
        for (a, b, c, d), s in zip(boxes / self.zoom, scores):
            out.append(((a + c) / 2 + x0, (b + d) / 2 + y0, float(s), max(c - a, d - b) / 2))
        return out, (x0, y0, crop)

    def _blob_candidates(self, x0, y0, crop, people):
        """Small white round blobs in the search window (fallback when the detector misses)."""
        if crop.size == 0:
            return []
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        white = ((hsv[..., 2] > 150) & (hsv[..., 1] < 80)).astype(np.uint8)
        n, _, st, cen = cv2.connectedComponentsWithStats(white, connectivity=8)
        r = self.radius
        out = []
        for i in range(1, n):
            bx, by, bw, bh, area = st[i]
            if not (0.3 * r * r <= area <= 6 * r * r) or max(bw, bh) > 3.5 * r:
                continue
            if max(bw, bh) / max(1, min(bw, bh)) > 1.8 or area / float(bw * bh) < 0.45:
                continue                                 # line fragments are long / thin
            x, y = cen[i][0] + x0, cen[i][1] + y0
            if _on_upper_body(x, y, people):
                continue                                 # white shirts / socks / numbers
            out.append((x, y, 0.08, max(bw, bh) / 2))
        return out

    # -- main step -------------------------------------------------------------------------------
    def update(self, img, ball_dets, people, field=None):
        """``ball_dets``: (N, 4) ball boxes + (N,) scores from the full-frame detector;
        ``people``: (M, 4) person boxes (used to reject candidates on shirts / heads);
        ``field``: pitch mask (uint8, 0 = not pitch) — candidates outside it are ignored."""
        boxes, scores = ball_dets
        self.width = img.shape[1]
        cands = [((a + c) / 2, (b + d) / 2, float(s), max(c - a, d - b) / 2) for (a, b, c, d), s in zip(boxes, scores)]
        ok = lambda c: (not _on_upper_body(c[0], c[1], people) and _on_field(c[0], c[1], field)
                        and all(np.hypot(c[0] - bx, c[1] - by) > 12 for bx, by in self.banned))
        cands = [c for c in cands if ok(c)]

        if self.kf is None:                              # no track: (re)start from a confident detection, or from
            self.recent.append(cands)                    # a weak one seen at the same place in the last 3 frames
            best = max(cands, key=lambda c: c[2], default=None)
            if best is None or best[2] < self.init_conf:
                best = max((c for c in cands if len(self.recent) == 3 and all(
                    any(np.hypot(c[0] - p[0], c[1] - p[1]) < self.gate for p in prev) for prev in list(self.recent)[:-1])),
                    key=lambda c: c[2], default=None)
            if best is None:
                return None
            self._start(best)
            return best[0], best[1], self.radius, "det"

        px, py = self.kf.predict()[:2, 0]
        zc, (x0, y0, crop) = self._zoom_candidates(img, px, py)
        cands += [c for c in zc if c[2] >= self.zoom_conf and ok(c)]
        if self.use_blobs and not cands:
            cands += [c for c in self._blob_candidates(x0, y0, crop, people) if ok(c)]

        gate = self.gate * (1 + 0.5 * self.lost)         # the search widens while the ball is lost
        best, best_cost = None, np.inf
        for c in cands:
            d = np.hypot(c[0] - px, c[1] - py)
            cost = d - 40 * c[2]
            if d <= gate and cost < best_cost:
                best, best_cost = c, cost
        # a strong detection far away = the track drifted (pass, long ball): jump to it
        strong = [c for c in cands if c[2] >= max(0.45, self.init_conf)]
        if best is None and strong and self.lost >= 3:
            best = max(strong, key=lambda c: c[2])
            self._start(best)
            return best[0], best[1], self.radius, "det"
        if best is None:
            self.lost += 1
            if self.lost > self.max_lost:
                self.kf = None
                return None
            return float(px), float(py), self.radius, "pred"
        self.kf.correct(np.array([[best[0]], [best[1]]], np.float32))
        self.lost = 0
        self.radius = 0.8 * self.radius + 0.2 * float(np.clip(best[3], 2, 30))
        x, y = self.kf.statePost[:2, 0]
        if self._static(x, y, people):
            return None
        return float(x), float(y), self.radius, "det"

    def shift(self, dx, dy):
        """Camera moved by (dx, dy) px since the last frame: move the filter state (and the banned spots) with it."""
        if self.kf is not None:
            self.kf.statePost[0, 0] += dx
            self.kf.statePost[1, 0] += dy
        self.recent = deque(([(c[0] + dx, c[1] + dy, *c[2:]) for c in cs] for cs in self.recent), maxlen=3)
        self.banned = [(x + dx, y + dy) for x, y in self.banned]

    def _static(self, x, y, people) -> bool:
        """True (and the spot is banned) when the ball has not moved for ``max_still`` frames, far from everyone."""
        tol = max(0.008 * self.width, 2 * self.radius)      # 15 px at 1080p (camera-shift errors drift a bit)
        moved = self.last_xy is None or np.hypot(x - self.last_xy[0], y - self.last_xy[1]) > tol
        self.still = 0 if moved else self.still + 1
        if moved:
            self.last_xy = (x, y)
        if self.still < self.max_still:
            return False
        near = any(np.hypot((a + c) / 2 - x, d - y) < 3 * (c - a) + 20 for a, b, c, d in people)
        if near:                                     # e.g. ball waiting for a free kick
            return False
        self.banned.append((x, y))
        self.erase = self.still + 3                  # it was never the ball: also blank the frames already returned
        self.kf, self.still, self.last_xy = None, 0, None
        return True

    def _start(self, c):
        self.kf, self.lost, self.radius = _kalman(c[0], c[1]), 0, float(np.clip(c[3], 2, 30))
        self.still, self.last_xy = 0, (c[0], c[1])
        self.recent.clear()


def _on_upper_body(x, y, people) -> bool:
    """True when (x, y) is inside the top 70 % of a person box (the ball at the feet is allowed)."""
    for a, b, c, d in people:
        if a <= x <= c and b <= y <= b + 0.7 * (d - b):
            return True
    return False


def _on_field(x, y, field) -> bool:
    if field is None:
        return True
    h, w = field.shape[:2]
    return 0 <= int(y) < h and 0 <= int(x) < w and field[int(y), int(x)] > 0


def fill_gaps(df, max_gap: int = 12, smooth: int = 5):
    """Offline clean-up of the per-frame ball table (columns frame, x, y, r, state).

    Trailing 'pred' frames after the ball was lost are dropped, gaps of at most ``max_gap`` frames are
    interpolated, and each continuous run is smoothed with a centred rolling mean."""
    df = df.copy()
    # coasting frames that never got re-confirmed are not trusted
    det = df.state.eq("det")
    confirmed_later = det[::-1].cummax()[::-1]
    df.loc[df.state.eq("pred") & ~confirmed_later, ["x", "y"]] = np.nan
    df.loc[df.state.isna(), ["x", "y"]] = np.nan
    for c in ("x", "y", "r"):
        s = df[c]
        f = s.interpolate(limit_area="inside")
        gap_len = s.isna().groupby(s.notna().cumsum()).transform("sum")
        f[s.isna() & (gap_len > max_gap)] = np.nan
        df[c] = f
    df["state"] = np.where(df.x.isna(), None, np.where(det, "det", "interp"))
    valid = df.x.notna()
    seg = (valid != valid.shift()).cumsum()
    for c in ("x", "y"):
        df[c] = df.groupby(seg)[c].transform(lambda s: s.rolling(smooth, center=True, min_periods=1).mean())
    return df
