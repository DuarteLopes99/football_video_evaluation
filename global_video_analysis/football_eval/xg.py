"""Expected goals (xG).

Model: logistic regression on geometric + freeze-frame features, trained on StatsBomb open data
(scripts/train_xg.py; ~8.8k shots from 351 matches across WC 2022, Euro 2020/2024, AFCON 2023,
Bundesliga 23/24, Ligue 1, La Liga 20/21, MLS). Coefficients are stored as plain JSON
(``models/xg_model.json``) so inference needs only numpy.

Why these features: they are exactly what a single camera can observe at the moment of the shot —
shot location, goalkeeper position and opponents between ball and goal (StatsBomb 360-style
"freeze frame"). Body part (header vs foot) and shot height are NOT reliably observable from a
wide single view, so they are deliberately left out of the model (rulebook §5).

Feature definitions (attack frame: goal centre at (L, W/2)):
    dist        distance shot -> goal centre (m)
    angle       open goal-mouth angle seen from the shot location (rad)
    n_def_tri   outfield opponents inside the triangle shot / left post / right post
    gk_in_tri   goalkeeper inside that triangle (0/1)
    gk_dist     distance goalkeeper -> shot location (m, capped at 40)
    n_press     opponents within 3 m of the shot location
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

GOAL_W = 7.32
BASIC = ["dist", "log_dist", "angle"]
FULL = BASIC + ["n_def_tri", "gk_in_tri", "gk_dist", "n_press"]


def shot_geometry(x, y, L=105.0, W=68.0):
    dx = L - np.asarray(x, float)
    dy = np.abs(np.asarray(y, float) - W / 2)
    dist = np.hypot(dx, dy)
    num = GOAL_W * dx
    den = dx ** 2 + dy ** 2 - (GOAL_W / 2) ** 2
    angle = np.arctan2(num, den)
    angle = np.where(angle < 0, angle + np.pi, angle)
    return dist, angle


def _in_triangle(p, a, b, c):
    def s(p1, p2, p3):
        return (p1[..., 0] - p3[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[..., 1] - p3[1])
    d1, d2, d3 = s(p, a, b), s(p, b, c), s(p, c, a)
    neg = (d1 < 0) | (d2 < 0) | (d3 < 0)
    pos = (d1 > 0) | (d2 > 0) | (d3 > 0)
    return ~(neg & pos)


def freeze_frame_features(shot_xy, opponents_xy: np.ndarray, gk_xy: Optional[Sequence[float]], L=105.0, W=68.0):
    """Opponents (outfield, attack frame) and goalkeeper position at the moment of the shot."""
    s = np.asarray(shot_xy, float)
    a, b = np.array([L, W / 2 - GOAL_W / 2]), np.array([L, W / 2 + GOAL_W / 2])
    opp = np.asarray(opponents_xy, float).reshape(-1, 2)
    n_tri = int(_in_triangle(opp, s, a, b).sum()) if len(opp) else 0
    n_press = int((np.hypot(*(opp - s).T) < 3.0).sum()) if len(opp) else 0
    if gk_xy is None or np.any(np.isnan(gk_xy)):
        gk_in, gk_dist = 1, float(np.hypot(L - s[0], W / 2 - s[1]))   # assume GK on his line
    else:
        g = np.asarray(gk_xy, float)
        gk_in = int(_in_triangle(g[None], s, a, b)[0])
        gk_dist = float(min(np.hypot(*(g - s)), 40.0))
    return dict(n_def_tri=n_tri, gk_in_tri=gk_in, gk_dist=gk_dist, n_press=n_press)


def feature_row(x, y, ff: Optional[dict] = None, L=105.0, W=68.0) -> dict:
    d, a = shot_geometry(x, y, L, W)
    r = dict(dist=float(d), log_dist=float(np.log1p(d)), angle=float(a))
    if ff:
        r.update(ff)
    return r


class XGModel:
    def __init__(self, params: dict):
        self.p = params

    @classmethod
    def load(cls, path: str | Path) -> "XGModel":
        with open(path) as f:
            return cls(json.load(f))

    def predict(self, rows: Sequence[dict], variant: str = "full", situation: Sequence[str] = None) -> np.ndarray:
        """``situation`` per shot: 'open_play' (default) | 'penalty' | 'direct_free_kick'."""
        out = np.zeros(len(rows))
        for i, r in enumerate(rows):
            sit = situation[i] if situation is not None else "open_play"
            if sit == "penalty":
                out[i] = self.p["penalty_xg"]
                continue
            v = variant if (variant == "basic" or all(k in r and r[k] is not None for k in FULL)) else "basic"
            m = self.p[v]
            z = (np.array([r[k] for k in m["features"]], float) - np.array(m["mean"])) / np.array(m["std"])
            p = 1.0 / (1.0 + np.exp(-(m["intercept"] + z @ np.array(m["coef"]))))
            if sit == "direct_free_kick":
                p *= self.p.get("dfk_factor", 1.0)
            out[i] = p
        return out
