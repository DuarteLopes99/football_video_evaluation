"""Overlay drawing: coloured player silhouettes, ball + trail, ball-carrier marker and a small legend."""
from __future__ import annotations

import cv2
import numpy as np

from .segment import player_mask
from .teams import OTHER

BALL_COLOR = (0, 255, 255)        # BGR yellow
NAMES = {0: "Team A", 1: "Team B", OTHER: "GK / Ref"}


def draw_frame(img, players, ball, trail, colors, carrier=None, control=None, alpha=0.40):
    """``players``: list of (box, track id, team); ``ball``: (x, y, r, state) or None;
    ``trail``: list of recent (x, y); ``carrier``: track id of the player closest to the ball;
    ``control``: (share team 0, share team 1) of ball control so far, or None."""
    out = img.copy()
    fill = img.copy()
    for box, tid, team in players:
        col = colors[team]
        seg = player_mask(img, box)
        if seg is not None:
            m, x0, y0 = seg
            cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cnts = [c + (x0, y0) for c in cnts]
            cv2.drawContours(fill, cnts, -1, col, -1)
            cv2.drawContours(out, cnts, -1, (20, 20, 20), 4, cv2.LINE_AA)      # dark halo = contrast on grass
            cv2.drawContours(out, cnts, -1, col, 2, cv2.LINE_AA)
        else:                                                                # fallback: ellipse at the feet
            cx, w, yb = int((box[0] + box[2]) / 2), int(box[2] - box[0]), int(box[3])
            cv2.ellipse(out, (cx, yb), (max(w // 2, 6), max(w // 6, 3)), 0, -45, 235, col, 2, cv2.LINE_AA)
    # translucent fill only where silhouettes were painted
    painted = np.any(fill != img, axis=2)
    out[painted] = cv2.addWeighted(out, 1 - alpha, fill, alpha, 0)[painted]

    for box, tid, team in players:
        if tid == carrier:                                                  # ball carrier: triangle above head
            cx, top = int((box[0] + box[2]) / 2), int(box[1]) - 6
            s = max(6, int((box[2] - box[0]) * 0.25))
            tri = np.array([[cx, top], [cx - s, top - int(1.5 * s)], [cx + s, top - int(1.5 * s)]], np.int32)
            cv2.fillPoly(out, [tri], colors[team], cv2.LINE_AA)
            cv2.polylines(out, [tri], True, (20, 20, 20), 1, cv2.LINE_AA)

    if len(trail) > 1:                                                      # fading ball trail
        for i in range(1, len(trail)):
            a = i / len(trail)
            cv2.line(out, tuple(map(int, trail[i - 1])), tuple(map(int, trail[i])), BALL_COLOR,
                     max(1, int(3 * a)), cv2.LINE_AA)
    if ball is not None:
        x, y, r, state = ball
        rr = int(max(r, 4) + 4)
        cv2.circle(out, (int(x), int(y)), rr + 1, (20, 20, 20), 3, cv2.LINE_AA)
        cv2.circle(out, (int(x), int(y)), rr, BALL_COLOR, 2 if state == "det" else 1, cv2.LINE_AA)

    _legend(out, players, colors, ball, control)
    return out


def _legend(img, players, colors, ball, control=None):
    s = img.shape[0] / 1080                                                   # scale with resolution
    counts = {k: sum(1 for p in players if p[2] == k) for k in NAMES}
    x, y, h = int(20 * s), int(20 * s), int(34 * s)
    cv2.rectangle(img, (x - 10, y - 10), (x + int(260 * s), y + (6 if control else 4) * h), (25, 25, 25), -1)
    for i, k in enumerate((0, 1, OTHER)):
        yy = y + i * h
        cv2.rectangle(img, (x, yy), (x + int(24 * s), yy + int(24 * s)), colors[k], -1)
        cv2.putText(img, f"{NAMES[k]}: {counts[k]}", (x + int(36 * s), yy + int(20 * s)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7 * s, (240, 240, 240), max(1, int(2 * s)), cv2.LINE_AA)
    yy = y + 3 * h
    cv2.circle(img, (x + int(12 * s), yy + int(12 * s)), int(9 * s), BALL_COLOR, 2)
    txt = "Ball: tracked" if ball is not None else "Ball: lost"
    cv2.putText(img, txt, (x + int(36 * s), yy + int(20 * s)), cv2.FONT_HERSHEY_SIMPLEX, 0.7 * s,
                (240, 240, 240), max(1, int(2 * s)), cv2.LINE_AA)
    if control:                                                               # team ball control bar
        yy = y + 4 * h + int(4 * s)
        cv2.putText(img, "Ball control", (x, yy + int(16 * s)), cv2.FONT_HERSHEY_SIMPLEX, 0.6 * s,
                    (240, 240, 240), max(1, int(2 * s)), cv2.LINE_AA)
        bx0, bx1, by0, by1 = x, x + int(240 * s), yy + int(26 * s), yy + int(50 * s)
        split = bx0 + int((bx1 - bx0) * control[0])
        cv2.rectangle(img, (bx0, by0), (split, by1), colors[0], -1)
        cv2.rectangle(img, (split, by0), (bx1, by1), colors[1], -1)
        for share, xx, col in ((control[0], bx0 + 4, colors[0]), (control[1], bx1 - int(58 * s), colors[1])):
            ink = (20, 20, 20) if sum(col) > 380 else (245, 245, 245)        # readable on light / dark kits
            cv2.putText(img, f"{share * 100:.0f}%", (xx, by1 - int(6 * s)), cv2.FONT_HERSHEY_SIMPLEX, 0.6 * s,
                        ink, max(1, int(2 * s)), cv2.LINE_AA)
