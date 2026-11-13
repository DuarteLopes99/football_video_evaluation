"""Tests of the simple tracker (football_eval/simple) on synthetic images — no video, no model, a few seconds.

Run: `python -m pytest tests/test_simple.py -q`
"""
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from football_eval.simple.ball import BallTracker, fill_gaps  # noqa: E402
from football_eval.simple.segment import field_mask, player_mask  # noqa: E402
from football_eval.simple.teams import OTHER, TeamColors  # noqa: E402

GRASS = (40, 140, 60)                                          # BGR


def _pitch(h=360, w=640):
    img = np.zeros((h, w, 3), np.uint8)
    img[:] = GRASS
    img[: h // 3] = (230, 180, 120)                             # sky
    return img


def _player(img, x, y, shirt, h=80, w=30):
    cv2.rectangle(img, (x, y), (x + w, y + int(0.5 * h)), shirt, -1)              # shirt
    cv2.rectangle(img, (x + 4, y + int(0.5 * h)), (x + w - 4, y + h), (30, 30, 30), -1)   # shorts / legs
    return np.array([x - 5, y - 5, x + w + 5, y + h + 3], float)


def test_field_mask_keeps_grass_only():
    m = field_mask(_pitch())
    assert m[300, 320] == 255 and m[20, 320] == 0


def test_player_mask_finds_the_player():
    img = _pitch()
    box = _player(img, 300, 200, (0, 0, 220))
    mask, x0, y0 = player_mask(img, box)
    assert mask[205 - y0, 315 - x0] == 255                    # inside the shirt
    assert 0.3 < (mask > 0).mean() < 0.9


def test_team_colors_split_two_kits_and_one_outlier():
    tc = TeamColors()
    rng = np.random.default_rng(0)
    jitter = lambda c: tuple(int(np.clip(v + rng.normal(0, 12), 0, 255)) for v in c)   # sun / shade / blur
    for f in range(10):
        img = _pitch()
        boxes, tids = [], []
        for i in range(5):                                      # red team = tracks 0..4, white team = 5..9
            boxes.append(_player(img, 20 + 60 * i, 150 + int(rng.integers(0, 40)), jitter((0, 0, 200))))
            tids.append(i)
            boxes.append(_player(img, 330 + 60 * i, 200 + int(rng.integers(0, 40)), jitter((235, 235, 235))))
            tids.append(5 + i)
        boxes.append(_player(img, 600, 140, jitter((0, 220, 255))))   # referee in yellow = track 99
        tids.append(99)
        tc.add(img, np.array(boxes), np.array(tids))
    tc.fit()
    red, white = {tc.team(i) for i in range(5)}, {tc.team(i) for i in range(5, 10)}
    assert len(red) == 1 and len(white) == 1 and red != white and OTHER not in red | white
    assert tc.team(99) == OTHER


class _NoZoomDetector:
    """Stands in for YOLOX: the zoomed search finds nothing (only the full-frame detections count)."""

    def _infer(self, tile):
        z = np.zeros((0, 4))
        return z, np.zeros(0), z, np.zeros(0)


def test_ball_tracker_ignores_far_false_positive():
    bt = BallTracker(_NoZoomDetector(), use_blobs=False)
    img = _pitch()
    xs = []
    for f in range(20):
        x = 100 + 10 * f
        dets = [[x - 4, 250, x + 4, 258]]
        scores = [0.6]
        if f == 10:                                              # a strong false hit far away, same frame
            dets.append([600, 130, 608, 138])
            scores.append(0.5)
        res = bt.update(img, (np.array(dets, float), np.array(scores)), np.zeros((0, 4)))
        xs.append(res[0])
    assert abs(xs[10] - 200) < 15 and abs(xs[-1] - 290) < 15


def test_ball_tracker_drops_static_object_far_from_players():
    bt = BallTracker(_NoZoomDetector(), use_blobs=False, max_still=10)
    img = _pitch()
    still = np.array([[400, 200, 408, 208]], float)              # a "ball" that never moves, nobody near
    out, erase = [], []
    for _ in range(30):
        out.append(bt.update(img, (still, np.array([0.6])), np.zeros((0, 4))))
        erase.append(bt.erase)
    assert out[5] is not None and out[-1] is None               # tracked at first, then banned for good
    assert max(erase) >= 10                                     # the caller is told to blank the frames already sent
    near = np.array([[380, 150, 420, 215]], float)              # same, but a player stands on it (free kick)
    bt = BallTracker(_NoZoomDetector(), use_blobs=False, max_still=10)
    out = [bt.update(img, (still, np.array([0.6])), near) for _ in range(30)]
    assert out[-1] is not None


def test_fill_gaps_interpolates_short_gaps_only():
    n = 40
    x = np.arange(n, dtype=float)
    state = np.array(["det"] * n, object)
    state[5:8] = None                                           # short gap (3 frames)
    state[15:35] = None                                         # long gap (20 frames)
    x[state == None] = np.nan                                   # noqa: E711
    df = pd.DataFrame(dict(frame=np.arange(n), x=x, y=x, r=3.0, state=state))
    out = fill_gaps(df, max_gap=5, smooth=1)
    assert out.x[5:8].notna().all() and out.x[15:35].isna().all()
    assert (out.state[5:8] == "interp").all()
