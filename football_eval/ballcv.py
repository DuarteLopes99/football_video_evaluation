"""Ball candidates for low-resolution, panning club footage (e.g. 640x360 streams).

At 640x360 the ball is 2-6 px wide: COCO detectors (YOLOX / YOLO 'sports ball') practically never
fire on it. This module uses the classic broadcast-ball recipe instead:

1. **camera-motion compensation** — warp the previous frame onto the current one with the
   rotation+zoom homography given by the PTZ tracker (``PTZProvider.last_image_motion``);
2. **difference image** on the grass → moving blobs;
3. keep **small, compact, bright, low-chroma** blobs outside player boxes (but allow the feet zone);
4. hand all candidates to :func:`football_eval.ball.clean_ball`, which keeps the physically
   consistent sequence (speed gating, gap interpolation, smoothing).

It returns several candidates per frame with a score; false positives (socks, line fragments,
flicker) are expected and filtered temporally. A TrackNet-style heat-map network trained on a few
hundred labelled frames of this venue is the planned replacement (docs/SOTA_REVIEW.md §5).
"""
from __future__ import annotations

from typing import List, Optional, Sequence

import cv2
import numpy as np


class MotionBallDetector:
    def __init__(self, ignore: Sequence = (), diff_thr: int = 18, min_area: int = 2, max_area: int = 45,
                 max_aspect: float = 2.6, min_contrast: float = 18.0, max_candidates: int = 6):
        self.ignore = list(ignore)
        self.diff_thr, self.min_area, self.max_area = diff_thr, min_area, max_area
        self.max_aspect, self.min_contrast, self.k = max_aspect, min_contrast, max_candidates
        self.prev: Optional[np.ndarray] = None

    def __call__(self, img: np.ndarray, H_prev_to_cur: Optional[np.ndarray], field: Optional[np.ndarray] = None,
                 boxes: Optional[np.ndarray] = None) -> List[tuple]:
        """Returns [(u, v, score)] with (u, v) the ball's ground contact point (bottom of the blob)."""
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        prev, self.prev = self.prev, g
        if prev is None or H_prev_to_cur is None:
            return []
        h, w = g.shape
        warped = cv2.warpPerspective(prev, H_prev_to_cur, (w, h), flags=cv2.INTER_LINEAR, borderValue=0)
        valid = cv2.warpPerspective(np.full_like(prev, 255), H_prev_to_cur, (w, h), flags=cv2.INTER_NEAREST) > 0
        diff = cv2.absdiff(g, warped)
        m = (diff > self.diff_thr) & valid
        if field is not None:
            m &= field > 0
        for x0, y0, x1, y1 in self.ignore:
            m[int(y0):int(y1), int(x0):int(x1)] = False
        if boxes is not None:
            for x0, y0, x1, y1 in np.asarray(boxes, int):
                yf = int(y1 - 0.2 * (y1 - y0))           # keep the feet zone: the ball is often there
                m[max(y0 - 1, 0):yf, max(x0 - 1, 0):x1 + 2] = False
        m = m.astype(np.uint8)
        n, lab, st, cen = cv2.connectedComponentsWithStats(m, connectivity=8)
        L = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        bg = cv2.medianBlur(L[..., 0], 9).astype(np.float32)
        chroma = np.hypot(L[..., 1].astype(np.float32) - 128, L[..., 2].astype(np.float32) - 128)
        out = []
        for i in range(1, n):
            x, y, bw, bh, area = st[i]
            if not (self.min_area <= area <= self.max_area):
                continue
            if max(bw, bh) / max(1, min(bw, bh)) > self.max_aspect:
                continue
            sel = lab[y:y + bh, x:x + bw] == i
            Lb = L[y:y + bh, x:x + bw, 0][sel].astype(np.float32)
            contrast = float(Lb.max() - bg[y:y + bh, x:x + bw][sel].mean())
            if contrast < self.min_contrast:
                continue
            ch = float(chroma[y:y + bh, x:x + bw][sel].mean())
            fill = area / float(bw * bh)
            score = contrast / 40.0 + fill - ch / 30.0
            out.append((float(cen[i][0]), float(y + bh), score))
        out.sort(key=lambda z: -z[2])
        return out[: self.k]

    def reset(self):
        self.prev = None
