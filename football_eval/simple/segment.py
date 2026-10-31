"""Very simple player segmentation (no SAM, no extra model).

Inside a slightly enlarged player box, the pixels along the box border are almost all pitch. Their median colour
is the *local* grass colour (it follows sun / shade). Every pixel that is clearly different from it (Lab distance)
is "player". Morphology + the biggest blob that touches the box centre give the silhouette, which is then grown a
few pixels so the whole player is always inside the outline (it does not need to be pixel-exact).
"""
from __future__ import annotations

import cv2
import numpy as np


def player_mask(img: np.ndarray, box, pad: float = 0.12, thr: float = 16.0, grow: float = 0.10):
    """Returns (mask uint8 0/255, x0, y0) for the enlarged box, or None when the box is too small.

    The silhouette is deliberately generous: holes are filled and it is grown by ``grow`` x the box width
    (about 10 px for a 100 px wide player) and rounded, so arms, feet and blurred edges stay inside the outline."""
    H, W = img.shape[:2]
    x0, y0, x1, y1 = box
    pw, ph = pad * (x1 - x0), pad * (y1 - y0)
    x0, y0 = int(max(0, x0 - pw)), int(max(0, y0 - ph))
    x1, y1 = int(min(W, x1 + pw)), int(min(H, y1 + ph * 0.5))
    if x1 - x0 < 6 or y1 - y0 < 12:
        return None
    lab = cv2.cvtColor(img[y0:y1, x0:x1], cv2.COLOR_BGR2LAB).astype(np.float32)
    ring = np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]])
    # two grass references (darker / lighter half of the border) = shade and sun inside the same box
    lo = ring[:, 0] <= np.median(ring[:, 0])
    halves = [ring[lo], ring[~lo]] if (~lo).any() else [ring]
    dist = np.full(lab.shape[:2], np.inf, np.float32)
    for g in (np.median(h, 0) for h in halves):
        d = np.hypot(lab[..., 1] - g[1], lab[..., 2] - g[2]) + 0.25 * np.abs(lab[..., 0] - g[0])
        dist = np.minimum(dist, d)                          # colour counts more than brightness
    m = (dist > thr).astype(np.uint8) * 255
    k = max(3, (x1 - x0) // 12) | 1
    ko = max(3, (x1 - x0) // 25)                           # removes thin pitch lines crossing the box
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((ko, ko), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    n, lab_cc, st, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    if n <= 1:
        return None
    # keep the biggest blob crossing the box's centre column (the player, not a line or a neighbour)
    cx = m.shape[1] // 2
    col = lab_cc[:, max(cx - 2, 0):cx + 3]
    ids = [i for i in np.unique(col) if i > 0] or list(range(1, n))
    best = max(ids, key=lambda i: st[i, cv2.CC_STAT_AREA])
    if st[best, cv2.CC_STAT_AREA] < 0.08 * m.size:
        return None
    m = (lab_cc == best).astype(np.uint8) * 255
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(m, cnts, -1, 255, -1)                  # fill holes (shirt numbers, stripes)
    g = max(3, int(grow * (x1 - x0))) | 1
    m = cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (g, g)))
    m = cv2.GaussianBlur(m, (g, g), 0)                      # rounder, smoother outline
    return (m > 100).astype(np.uint8) * 255, x0, y0


def field_mask(img: np.ndarray, margin: float = 0.05) -> np.ndarray:
    """Pitch area of the frame (uint8 0/255), used to ignore ball candidates in the trees / sky / stands and
    people behind the barriers.

    Grass colour = median of the lower half of the frame. Big grass-coloured regions touching the bottom of the
    image form the pitch; in every column, everything below its top edge is pitch (fills players and lines)."""
    H, W = img.shape[:2]
    s = 4                                                   # work at 1/4 resolution (fast)
    hsv = cv2.cvtColor(cv2.resize(img, (W // s, H // s), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2HSV)
    hsv = hsv.astype(np.int16)
    g = np.median(hsv[hsv.shape[0] // 2:].reshape(-1, 3), 0)
    dh = np.abs(hsv[..., 0] - g[0])
    green = ((np.minimum(dh, 180 - dh) < 18) & (hsv[..., 1] > 0.3 * g[1]) & (hsv[..., 2] > 0.25 * g[2]))
    green = cv2.morphologyEx(green.astype(np.uint8), cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(green, connectivity=4)
    bottom = [i for i in np.unique(lab[-1]) if i > 0 and st[i, cv2.CC_STAT_AREA] > 0.01 * lab.size]
    if not bottom:                                          # (pitch lines / shadows may split it in parts)
        return np.full((H, W), 255, np.uint8)
    pitch = np.isin(lab, bottom)
    h4, w4 = pitch.shape
    top = np.where(pitch.any(0), pitch.argmax(0), h4)       # first pitch row of each column
    top = cv2.medianBlur(top.astype(np.uint8).reshape(1, -1), 9).ravel() if h4 < 256 else top
    top = np.maximum(top.astype(int) - int(margin * h4), 0)
    mask = (np.arange(h4)[:, None] >= top[None, :]).astype(np.uint8) * 255
    return cv2.resize(mask, (W, H), interpolation=cv2.INTER_NEAREST)
