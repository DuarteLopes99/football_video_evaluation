"""Team colours without any clicking: shirt colour -> 2-cluster Gaussian mixture -> majority vote per track.

Shirt feature (3 numbers per player crop), chosen to survive sun / shade on amateur footage:

* the pixels along the player box border give the *local* grass colour;
* torso pixels whose chroma is clearly different from that grass are the shirt;
* feature = median Lab chroma (a, b) of those pixels + their lightness relative to the local grass
  (so white-vs-black kits are separated too, but a shadow does not change the team).

Then:

1. samples far from everything (3 x the median distance to the median) are left out of the fit (goalkeepers,
   referee). If that filter removed a whole team (one kit far more visible than the other), it is skipped;
2. a Gaussian mixture with 2 components = the two outfield kits (full covariance copes with the
   sun / shade spread inside each kit better than KMeans); samples far from their kit (``outlier_factor`` x
   its median spread) are "other" and the mixture is refitted without them;
3. every track gets ONE label for the whole clip (majority vote), so colours never flicker.

Limitation: a referee / goalkeeper kit lying *between* the two team kits in colour cannot be told apart by
colour alone and may join a team.

The drawing colour of each team has the hue of its real kit, made vivid so it stands out on the grass.
"""
from __future__ import annotations

from collections import Counter, defaultdict

import cv2
import numpy as np
from sklearn.mixture import GaussianMixture

OTHER = -1
FALLBACK = {0: (60, 60, 230), 1: (230, 140, 40), OTHER: (0, 215, 255)}   # BGR: red, blue, yellow


def shirt_feature(img, box):
    """Returns the 3-number shirt feature, or None for tiny / mostly-grass crops."""
    x0, y0, x1, y1 = [int(round(v)) for v in box]
    H, W = img.shape[:2]
    x0, y0, x1, y1 = max(x0, 0), max(y0, 0), min(x1, W), min(y1, H)
    h, w = y1 - y0, x1 - x0
    if h < 24 or w < 8:
        return None
    crop = img[y0:y1, x0:x1]
    lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB).astype(np.float32)
    grass = np.median(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]]), 0)
    sl = (slice(int(0.15 * h), int(0.5 * h)), slice(int(0.2 * w), int(0.8 * w)))   # torso
    t = lab[sl].reshape(-1, 3)
    keep = np.hypot(t[:, 1] - grass[1], t[:, 2] - grass[2]) > 10
    if keep.sum() < max(8, 0.15 * len(t)):
        return None
    p = np.median(t[keep], 0)
    return np.array([p[1] - 128, p[2] - 128, 8 * p[0] / max(grass[0], 1)], np.float32)


class TeamColors:
    def __init__(self, outlier_factor: float = 6.0, min_spread: float = 2.0):
        self.outlier_factor, self.min_spread = outlier_factor, min_spread
        self.samples = []                         # (track id, feature)
        self.team_of = {}                         # track id -> 0 / 1 / OTHER
        self.colors = dict(FALLBACK)

    def add(self, img, boxes, tids):
        """Collect shirt samples of the tracked people in this frame."""
        for box, tid in zip(boxes, tids):
            if tid < 0:
                continue
            f = shirt_feature(img, box)
            if f is not None:
                self.samples.append((int(tid), f))

    def fit(self):
        tids = np.array([t for t, _ in self.samples])
        if len(self.samples) < 20:
            print("[teams] too few shirt samples, every player stays 'other'")
            return self
        X = np.stack([f for _, f in self.samples])
        d = np.linalg.norm(X - np.median(X, 0), axis=1)
        lab = self._cluster(X, core=d <= 3 * max(np.median(d), self.min_spread))   # GK / referee out of the fit
        n_tracks = lambda k: len(set(tids[lab == k]))
        if n_tracks(OTHER) > min(n_tracks(0), n_tracks(1)):     # the filter removed a whole kit: no filter
            lab = self._cluster(X, core=np.ones(len(X), bool))
        votes = defaultdict(Counter)
        for t, l in zip(tids, lab):
            votes[t][l] += 1
        self.team_of = {t: c.most_common(1)[0][0] for t, c in votes.items()}
        for k in (0, 1, OTHER):                  # drawing colour = real kit hue, made vivid
            if (lab == k).sum() >= 3:
                self.colors[k] = _vivid(np.median(X[lab == k], 0))
        if _lab_dist(self.colors[0], self.colors[1]) < 35:      # kits look alike after boosting
            self.colors[0], self.colors[1] = FALLBACK[0], FALLBACK[1]
        if min(_lab_dist(self.colors[OTHER], self.colors[k]) for k in (0, 1)) < 35:
            self.colors[OTHER] = FALLBACK[OTHER]
        n = Counter(self.team_of.values())
        print(f"[teams] tracks: A={n[0]} B={n[1]} other={n[OTHER]}  colours (BGR) {self.colors}")
        return self

    def _cluster(self, X, core):
        """2-component Gaussian mixture fitted on ``core``. Samples far from their component (``outlier_factor`` x
        the component's median spread — robust, unlike the fitted covariance which a referee can stretch) are
        "other"; the mixture is refitted without them. Returns the label of every sample (0 / 1 / OTHER)."""
        for _ in range(3):
            gmm = GaussianMixture(2, covariance_type="full", n_init=3, reg_covar=0.05, random_state=0).fit(X[core])
            lab = gmm.predict(X)
            far = np.zeros(len(X), bool)
            for k in (0, 1):
                m = core & (lab == k)
                if m.sum() < 3:
                    continue
                c = np.median(X[m], 0)
                spread = max(np.median(np.linalg.norm(X[m] - c, axis=1)), self.min_spread)
                far |= (lab == k) & (np.linalg.norm(X - c, axis=1) > self.outlier_factor * spread)
            core = core & ~far
        lab[far] = OTHER
        return lab

    def team(self, tid) -> int:
        return self.team_of.get(int(tid), OTHER)


def _vivid(feat) -> tuple:
    """Drawing colour from a kit feature (a, b, relative lightness): same hue as the kit, but saturated."""
    a, b, rel = feat
    chroma = float(np.hypot(a, b))
    if chroma < 5:                                # white / grey / black kit
        v = 240 if rel > 8 else 35
        return (v, v, v)
    lab = np.uint8([[[175, 128 + 75 * a / chroma, 128 + 75 * b / chroma]]])
    return tuple(int(c) for c in cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)[0, 0])


def _lab_dist(a, b) -> float:
    la, lb = (cv2.cvtColor(np.uint8([[c]]), cv2.COLOR_BGR2LAB)[0, 0].astype(float) for c in (a, b))
    return float(np.linalg.norm(la - lb))
