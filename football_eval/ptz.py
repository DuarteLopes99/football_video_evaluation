"""Pan-tilt-zoom (PTZ) camera model for a single operator-driven camera on a fixed tripod.

Typical club recording (e.g. "CFC TV", UD Mansores vs Canedo FC): one camera on the near side of the
pitch, close to the halfway line, that **pans, tilts and zooms** to follow the ball. Its optical centre
does not move, so every frame is described by the same camera position ``C`` and roll, plus three
per-frame parameters (pan, tilt, focal length). That gives three big advantages over a general
8-DOF homography per frame:

* frame-to-frame motion is a *rotation + zoom*: it can be measured on **any** static background
  (stands, trees, ad boards), not only on pitch lines, and it does not depend on depth;
* the absolute calibration only has 3 unknowns per frame, so aligning the pitch template to the
  white lines (chamfer matching) is well posed even when only a corner of the pitch is visible;
* ``C`` and roll are calibrated once per venue and re-used for every match filmed from that spot.

World frame (physical, right-handed): X along the near touchline (0 = left goal line as seen from the
camera, L = right), Y from the near touchline (0) to the far touchline (W), Z up. The package's pitch
frame ("ours", minimap with the far touchline at the top) is ``x = X, y = W - Y``.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Sequence

import cv2
import numpy as np
from scipy.optimize import least_squares

from .pitch import Homography, HomographyProvider, Pitch

R0 = np.array([[1.0, 0, 0], [0, 0, -1.0], [0, 1.0, 0]])   # camera looking along +Y, image right = +X, down = -Z


def _rx(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def _ry(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def _rz(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def rotation(pan: float, tilt: float, roll: float = 0.0) -> np.ndarray:
    """World -> camera rotation. pan > 0 turns right, tilt > 0 looks down (radians)."""
    return _rz(roll) @ _rx(tilt) @ _ry(-pan) @ R0


@dataclass
class PTZCamera:
    C: np.ndarray                      # camera centre in the physical world frame (m)
    roll: float = 0.0
    margin_end: float = 3.0            # grass beyond each goal line (m) — the visible edge of the grass
    margin_far: float = 2.0            # grass beyond the far touchline (m)
    width: int = 640
    height: int = 360
    pitch: Pitch = field(default_factory=Pitch)

    @property
    def cx(self):
        return self.width / 2.0

    @property
    def cy(self):
        return self.height / 2.0

    def K(self, f):
        return np.array([[f, 0, self.cx], [0, f, self.cy], [0, 0, 1.0]])

    def grass_edge(self, step: float = 1.0) -> tuple:
        """Segments of the far grass boundary (package frame): far side + both ends, as (A, B)."""
        L, W = self.pitch.length, self.pitch.width
        e, fa = self.margin_end, self.margin_far
        poly = np.array([(-e, W), (-e, -fa), (L + e, -fa), (L + e, W)], float)
        A, B = [], []
        for a, b in zip(poly[:-1], poly[1:]):
            n = max(1, int(np.ceil(np.linalg.norm(b - a) / step)))
            t = np.linspace(0, 1, n + 1)[:, None]
            pts = a + (b - a) * t
            A.append(pts[:-1]), B.append(pts[1:])
        return np.vstack(A), np.vstack(B)

    def H_world(self, pan, tilt, f) -> np.ndarray:
        """(X, Y, 1) on the ground plane -> image."""
        R = rotation(pan, tilt, self.roll)
        return self.K(f) @ np.c_[R[:, 0], R[:, 1], -R @ self.C]

    def H_ours(self, pan, tilt, f) -> np.ndarray:
        """Package pitch frame (x, y) -> image."""
        M = np.array([[1.0, 0, 0], [0, -1.0, self.pitch.width], [0, 0, 1.0]])
        return self.H_world(pan, tilt, f) @ M

    def homography(self, p) -> Homography:
        return Homography(self.H_ours(*p))

    def project_ours(self, xy: np.ndarray, p) -> tuple:
        """Project pitch points; returns (uv, in_front mask)."""
        H = self.H_ours(*p)
        q = np.c_[xy, np.ones(len(xy))] @ H.T
        front = q[:, 2] > 1e-6
        uv = q[:, :2] / np.where(np.abs(q[:, 2:3]) < 1e-9, 1e-9, q[:, 2:3])
        return uv, front

    def rays(self, uv: np.ndarray, p) -> np.ndarray:
        """Pixels -> unit viewing directions in the world frame."""
        pan, tilt, f = p
        R = rotation(pan, tilt, self.roll)
        d = np.c_[(uv[:, 0] - self.cx) / f, (uv[:, 1] - self.cy) / f, np.ones(len(uv))] @ R
        return d / np.linalg.norm(d, axis=1, keepdims=True)

    def project_dirs(self, d: np.ndarray, p) -> np.ndarray:
        pan, tilt, f = p
        R = rotation(pan, tilt, self.roll)
        c = d @ R.T
        z = np.where(c[:, 2] > 1e-6, c[:, 2], np.nan)
        return np.c_[f * c[:, 0] / z + self.cx, f * c[:, 1] / z + self.cy]

    def to_json(self) -> dict:
        return {"type": "ptz", "C": list(map(float, self.C)), "roll": float(self.roll),
                "margin_end": float(self.margin_end), "margin_far": float(self.margin_far),
                "pitch_length": float(self.pitch.length), "pitch_width": float(self.pitch.width),
                "width": self.width, "height": self.height}

    @classmethod
    def from_json(cls, d: dict, pitch: Pitch = None) -> "PTZCamera":
        if "pitch_length" in d:                       # the venue's measured pitch overrides the default
            pitch = Pitch(float(d["pitch_length"]), float(d["pitch_width"]))
        return cls(np.asarray(d["C"], float), float(d.get("roll", 0.0)), float(d.get("margin_end", 3.0)),
                   float(d.get("margin_far", 2.0)), int(d.get("width", 640)), int(d.get("height", 360)),
                   pitch or Pitch())


# ----------------------------------------------------------------------------- image evidence
def field_mask(img: np.ndarray) -> np.ndarray:
    """Pixels below the far edge of the grass (uint8 0/1). Robust to stands, trees and sky."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    grass = ((hsv[..., 0] > 32) & (hsv[..., 0] < 90) & (hsv[..., 1] > 50)).astype(np.uint8) * 255
    grass = cv2.morphologyEx(grass, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
    grass = cv2.morphologyEx(grass, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
    n, lab, st, _ = cv2.connectedComponentsWithStats(grass)
    h = img.shape[0]
    keep = [i for i in range(1, n) if st[i, 1] + st[i, 3] >= h - 5 and st[i, 4] > 0.01 * grass.size]
    f = np.isin(lab, keep)
    top = np.where(f.any(0), np.argmax(f, axis=0), h)
    return (np.arange(h)[:, None] >= top[None, :]).astype(np.uint8)


def line_mask(img: np.ndarray, fmask: Optional[np.ndarray] = None, boxes: Optional[np.ndarray] = None,
              ignore: Sequence = (), l_thr: int = 9, c_thr: int = 16) -> np.ndarray:
    """White pitch markings: locally brighter (top-hat on L) AND locally less chromatic (black-hat on
    chroma) than the surrounding grass. The chroma test rejects the sunlit strips between tree shadows,
    which are brighter but *more* saturated than their surroundings."""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    L = lab[..., 0]
    ch = np.clip(np.hypot(lab[..., 1].astype(np.float32) - 128, lab[..., 2].astype(np.float32) - 128) * 4, 0,
                 255).astype(np.uint8)
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    th = cv2.morphologyEx(L, cv2.MORPH_TOPHAT, k)
    bh = cv2.morphologyEx(ch, cv2.MORPH_BLACKHAT, k)
    m = (th > l_thr) & (bh > c_thr)
    if fmask is None:
        fmask = field_mask(img)
    m &= fmask > 0
    if boxes is not None:
        for x0, y0, x1, y1 in np.asarray(boxes, int):
            m[max(y0 - 2, 0):y1 + 3, max(x0 - 2, 0):x1 + 3] = False
    for x0, y0, x1, y1 in ignore:
        m[int(y0):int(y1), int(x0):int(x1)] = False
    return m.astype(np.uint8)


def template_points(pitch: Pitch, step: float = 0.6) -> np.ndarray:
    """Dense samples along every pitch marking (package frame)."""
    pts = []
    for ln in pitch.lines(n_arc=90):
        for a, b in zip(ln[:-1], ln[1:]):
            n = max(2, int(np.ceil(np.linalg.norm(b - a) / step)))
            t = np.linspace(0, 1, n, endpoint=False)[:, None]
            pts.append(a + (b - a) * t)
    return np.vstack(pts)


class Evidence:
    """Distance transforms of one frame used by the alignment cost."""

    def __init__(self, img: np.ndarray, boxes=None, ignore=(), tau: float = 12.0, max_line_px: int = 800):
        self.h, self.w = img.shape[:2]
        self.field = field_mask(img)
        self.lines = line_mask(img, self.field, boxes, ignore)
        self.n_line = int(self.lines.sum())
        self.dt = cv2.distanceTransform((1 - self.lines).astype(np.uint8), cv2.DIST_L2, 3)
        self.tau = tau
        # far edge of the grass (top boundary of the field mask), robust to notches made by players
        top = np.argmax(self.field > 0, axis=0).astype(np.float32)
        has = (self.field > 0).any(0) & (top > 2)
        med = cv2.medianBlur(np.where(has, top, 0).astype(np.float32).reshape(1, -1), 5).ravel()
        okc = has & (np.abs(top - med) < 2.0)
        for x0, y0, x1, y1 in ignore:
            okc[int(max(x0, 0)):int(min(x1, self.w))] &= ~((top[int(max(x0, 0)):int(min(x1, self.w))] >= y0) &
                                                           (top[int(max(x0, 0)):int(min(x1, self.w))] <= y1))
        if boxes is not None:
            for x0, y0, x1, y1 in np.asarray(boxes, int):
                okc[max(x0 - 2, 0):x1 + 3] &= ~((top[max(x0 - 2, 0):x1 + 3] >= y0 - 3) & (top[max(x0 - 2, 0):x1 + 3] <= y1 + 3))
        xs_e = np.nonzero(okc)[0][::2]
        self.edge_px = np.c_[xs_e, top[xs_e]].astype(np.float32)
        ys, xs = np.nonzero(self.lines)
        if len(xs) > max_line_px:
            k = np.random.default_rng(0).choice(len(xs), max_line_px, replace=False)
            xs, ys = xs[k], ys[k]
        self.line_px = np.c_[xs, ys].astype(np.float32)


def template_segments(pitch: Pitch, max_len: float = 1.0):
    """All pitch markings as short segments (package frame): returns (A, B) arrays (N, 2).

    The subdivision counts come from the standard 105 x 68 pitch so that N does not depend on the pitch size
    (the bundle adjustment estimates the size and needs a fixed-length residual vector)."""
    A, B = [], []
    ref = Pitch()
    for ln, ln_ref in zip(pitch.lines(n_arc=120), ref.lines(n_arc=120)):
        for (a, b), (ar, br) in zip(zip(ln[:-1], ln[1:]), zip(ln_ref[:-1], ln_ref[1:])):
            n = max(1, int(np.ceil(np.linalg.norm(br - ar) / max_len)))
            t = np.linspace(0, 1, n + 1)[:, None]
            pts = a + (b - a) * t
            A.append(pts[:-1])
            B.append(pts[1:])
    return np.vstack(A), np.vstack(B)


class Aligner:
    """Chamfer alignment of the pitch template to the detected white lines (two-sided, truncated, smooth).

    forward  : template samples visible on the grass -> bilinear distance-transform of the line mask
    backward : detected line pixels -> exact distance to the nearest projected template segment
    """

    def __init__(self, cam: PTZCamera, max_len: float = 1.0, w_edge: float = 0.7):
        self.cam = cam
        self.A, self.B = template_segments(cam.pitch, max_len)
        self.tpl = np.vstack([self.A, (self.A + self.B) / 2])
        self.w_edge = w_edge

    @staticmethod
    def _seg_dist(P, a, b, tau):
        from scipy.spatial import cKDTree
        if len(a) < 2 or not len(P):
            return np.full(len(P), tau)
        tree = cKDTree((a + b) / 2)
        _, idx = tree.query(P, k=min(4, len(a)))
        idx = idx.reshape(len(P), -1)
        A_, B_ = a[idx], b[idx]
        d = B_ - A_
        t = np.clip(((P[:, None, :] - A_) * d).sum(-1) / np.maximum((d * d).sum(-1), 1e-9), 0, 1)
        return np.minimum(np.linalg.norm(P[:, None, :] - (A_ + t[..., None] * d), axis=-1).min(1), tau)

    def edge(self, p, ev: "Evidence") -> np.ndarray:
        """Grass far edge pixels -> distance to the projected grass boundary."""
        if not len(ev.edge_px):
            return np.zeros(0)
        A, B = self.cam.grass_edge()
        a, fa = self._project(A, p)
        b, fb = self._project(B, p)
        ok = fa & fb & np.all(np.abs(a) < 5e3, 1) & np.all(np.abs(b) < 5e3, 1)
        return self._seg_dist(ev.edge_px.astype(float), a[ok], b[ok], ev.tau)

    def _project(self, P, p):
        uv, front = self.cam.project_ours(P, p)
        return uv, front

    def backward(self, p, ev: Evidence) -> np.ndarray:
        from scipy.spatial import cKDTree
        if not len(ev.line_px):
            return np.zeros(0)
        a, fa = self._project(self.A, p)
        b, fb = self._project(self.B, p)
        ok = fa & fb & np.all(np.abs(a) < 5e3, 1) & np.all(np.abs(b) < 5e3, 1)
        if ok.sum() < 2:
            return np.full(len(ev.line_px), ev.tau)
        a, b = a[ok], b[ok]
        tree = cKDTree((a + b) / 2)
        P = ev.line_px.astype(float)
        _, idx = tree.query(P, k=min(4, len(a)))
        idx = np.atleast_2d(idx) if idx.ndim == 1 else idx
        A_, B_ = a[idx], b[idx]                                   # (n, k, 2)
        d = B_ - A_
        t = np.clip(((P[:, None, :] - A_) * d).sum(-1) / np.maximum((d * d).sum(-1), 1e-9), 0, 1)
        dist = np.linalg.norm(P[:, None, :] - (A_ + t[..., None] * d), axis=-1).min(1)
        return np.minimum(dist, ev.tau)

    def forward(self, p, ev: Evidence) -> tuple:
        from scipy.ndimage import map_coordinates
        uv, front = self._project(self.tpl, p)
        ins = front & (uv[:, 0] >= 0) & (uv[:, 0] < ev.w - 1) & (uv[:, 1] >= 0) & (uv[:, 1] < ev.h - 1)
        out = np.zeros(len(self.tpl))
        if ins.sum():
            idx = np.nonzero(ins)[0]
            u = uv[idx]
            on = ev.field[u[:, 1].astype(int), u[:, 0].astype(int)] > 0
            idx, u = idx[on], u[on]
            out[idx] = np.minimum(map_coordinates(ev.dt, [u[:, 1], u[:, 0]], order=1), ev.tau)
        return out, int(ins.sum())

    def residuals(self, p, ev: Evidence, prior=None, prior_sigma=None, w_fwd=0.4, w_bwd=1.0):
        fwd, n_vis = self.forward(p, ev)
        r = [w_fwd * fwd * 6 / np.sqrt(max(n_vis, 100))]
        if len(ev.line_px):
            r.append(w_bwd * self.backward(p, ev) * 6 / np.sqrt(len(ev.line_px)))
        if self.w_edge and len(ev.edge_px):
            r.append(self.w_edge * self.edge(p, ev) * 6 / np.sqrt(len(ev.edge_px)))
        if prior is not None:
            r.append((np.asarray(p) - prior) / prior_sigma)
        return np.concatenate(r)

    def cost(self, p, ev, **kw) -> float:
        r = self.residuals(p, ev, **kw)
        return float((r ** 2).sum())

    def mean_line_error(self, p, ev) -> float:
        """Median distance (px) of detected line pixels to the projected template (quality score)."""
        b = self.backward(p, ev)
        return float(np.median(b)) if len(b) else np.nan       # median: robust to non-line white pixels

    def refine(self, p0, ev: Evidence, prior=None, prior_sigma=None, max_nfev=60):
        sol = least_squares(lambda q: self.residuals(q, ev, prior, prior_sigma), np.asarray(p0, float),
                            x_scale=[0.01, 0.01, 0.01 * p0[2]], diff_step=1e-4, loss="soft_l1", f_scale=1.0,
                            max_nfev=max_nfev)
        return sol.x, float((sol.fun ** 2).sum())

    def global_search(self, ev: Evidence, pan_range=(-1.3, 1.3), tilt_range=(-0.12, 0.30),
                      f_range=(350, 3000), n=(53, 11, 13), top=6):
        """Coarse grid over (pan, tilt, log f) on the backward cost, then local refinement of the best."""
        best = []
        for f in np.geomspace(*f_range, n[2]):
            for tilt in np.linspace(*tilt_range, n[1]):
                for pan in np.linspace(*pan_range, n[0]):
                    p = (pan, tilt, f)
                    b = self.backward(p, ev)
                    e = self.edge(p, ev) if self.w_edge else np.zeros(0)
                    c = float((b ** 2).mean()) if len(b) else 0.0
                    c += 0.5 * float((e ** 2).mean()) if len(e) else 0.0
                    best.append((c, p))
        best.sort(key=lambda z: z[0])
        out = [self.refine(p, ev) for _, p in best[:top]]
        return min(out, key=lambda z: z[1])


# ----------------------------------------------------------------------------- frame-to-frame motion
class MotionEstimator:
    """Rotation + zoom between consecutive frames from background features (players & overlays masked)."""

    def __init__(self, cam: PTZCamera, ignore=(), n_features: int = 1200):
        self.cam = cam
        self.ignore = list(ignore)
        self.orb = cv2.ORB_create(n_features, fastThreshold=10)
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        self.prev = None

    def _mask(self, shape, boxes):
        m = np.full(shape[:2], 255, np.uint8)
        for x0, y0, x1, y1 in self.ignore:
            m[int(y0):int(y1), int(x0):int(x1)] = 0
        if boxes is not None:
            for x0, y0, x1, y1 in np.asarray(boxes, int):
                m[max(y0 - 4, 0):y1 + 4, max(x0 - 4, 0):x1 + 4] = 0
        return m

    def step(self, img, boxes, p_prev):
        """Returns (p_pred, n_inliers). p_pred is None when motion could not be measured."""
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        kp, des = self.orb.detectAndCompute(g, self._mask(g.shape, boxes))
        prev, self.prev = self.prev, (kp, des)
        if prev is None or des is None or prev[1] is None or p_prev is None:
            return None, 0
        m = self.bf.match(prev[1], des)
        if len(m) < 15:
            return None, len(m)
        a = np.float32([prev[0][x.queryIdx].pt for x in m])
        b = np.float32([kp[x.trainIdx].pt for x in m])
        H, inl = cv2.findHomography(a, b, cv2.RANSAC, 2.5)
        if H is None or inl.sum() < 12:
            return None, 0 if inl is None else int(inl.sum())
        inl = inl.ravel().astype(bool)
        a, b = a[inl], b[inl]
        d = self.cam.rays(a.astype(float), p_prev)

        def res(q):
            uv = self.cam.project_dirs(d, q)
            r = (uv - b).ravel()
            return np.nan_to_num(r, nan=50.0)

        sol = least_squares(res, np.asarray(p_prev, float), x_scale=[0.01, 0.01, 20], loss="huber", f_scale=2.0,
                            max_nfev=50)
        rms = float(np.sqrt(np.mean(sol.fun ** 2)))
        if rms > 6.0 or abs(np.log(sol.x[2] / p_prev[2])) > 0.2 or abs(sol.x[0] - p_prev[0]) > 0.25:
            return None, int(inl.sum())          # implausible jump between two frames (0.2-s apart)
        return sol.x, int(inl.sum())

    def reset(self):
        self.prev = None


# ----------------------------------------------------------------------------- per-frame provider
class PTZProvider(HomographyProvider):
    """Tracks (pan, tilt, f) through the video: predict with background motion, correct with line alignment,
    re-initialise with a global search when both fail."""

    def __init__(self, cam: PTZCamera, ignore=(), refine_every: int = 1, init_params: Optional[Sequence] = None,
                 min_line_px: int = 150, lost_after: int = 8, max_err: float = 4.0,
                 search_kw: Optional[dict] = None):
        self.cam = cam
        self.al = Aligner(cam)
        self.mo = MotionEstimator(cam, ignore)
        self.ignore = list(ignore)
        self.p = None if init_params is None else np.asarray(init_params, float)
        self.refine_every = refine_every
        self.min_line_px = min_line_px
        self.lost_after = lost_after
        self.max_err = max_err
        self.taus, self.nfev = (20.0, 8.0), 15              # tracking: light refinement (big moves -> relocalize)
        self.pan_range, self.f_range = (-1.6, 1.6), (350.0, 1800.0)     # plausible camera poses (rad, px)
        self.n_lost, self.lost_global, self.last_global = 0, 10, -10**9
        self.global_attempts: List[tuple] = []
        self.n_bad = 0
        self.k = 0
        self.search_kw = search_kw or {}
        self.log: List[dict] = []
        self.last_image_motion = None
        self.last_evidence = None
        self.last_field = None                  # grass mask of the last evaluated frame, video pixels
        self.init_candidates: list = []         # poses tried on the first frame (all anchors of the camera file)
        self.anchors: dict = {}                 # frame -> validated pose: the tracker restarts from it when passing it
        self.anchor_snap_frames = 15            # ... if the analysed frame is at most this many frames after it
        self.anchor_hits: list = []
        self.prev_frame = None
        self.init_report: dict = {}

    def refine_guarded(self, p, ev, max_step=(0.035, 0.025, 0.08), taus=None, nfev=None):
        """Coarse-to-fine line alignment around ``p``; returns (q, err_before, err_after, accepted)."""
        ev.tau = 10.0
        e0 = self.al.mean_line_error(p, ev)
        q = np.asarray(p, float)
        sig = np.array([max_step[0], max_step[1], max_step[2] * p[2]])
        for tau in (taus or self.taus):
            ev.tau = tau
            q, _ = self.al.refine(q, ev, prior=np.asarray(p, float), prior_sigma=sig, max_nfev=nfev or self.nfev)
        ev.tau = 10.0
        e1 = self.al.mean_line_error(q, ev)
        d = np.abs(q - p)
        ok = (e1 < e0 - 0.05) and d[0] < 2 * max_step[0] and d[1] < 2 * max_step[1] and d[2] < 2 * max_step[2] * p[2]
        return (q if ok else np.asarray(p, float)), e0, e1, ok

    def warmup(self, video: str, t_from: float, t_to: float, fps: float = 5.0, refine_every: int = 4,
               progress: bool = True):
        """Track the camera (no detection) from an anchor time up to ``t_to`` so that processing can start
        anywhere in the video. ~20-40 ms per frame with refinement every ``refine_every`` frames."""
        from .video import iter_frames
        old, self.refine_every = self.refine_every, refine_every
        n = 0
        for f, t, im in iter_frames(video, fps, t_from, t_to):
            self.get(f, im, None)
            n += 1
            if progress and n % 500 == 0:
                print(f"  camera warm-up t={t / 60:5.1f} min", flush=True)
        self.refine_every = old
        self.log.clear()

    def relocalize(self, p, ev, d_pan=np.radians(7), d_tilt=np.radians(2.0), f_scales=(0.86, 0.93, 1.0, 1.07, 1.15)):
        """Local grid search around a drifted pose (backward + edge cost), then a wide guarded refinement."""
        best = (np.inf, p)
        for fs in f_scales:
            for dt in np.linspace(-d_tilt, d_tilt, 5):
                for dp in np.linspace(-d_pan, d_pan, 15):
                    q = np.array([p[0] + dp, p[1] + dt, p[2] * fs])
                    b = self.al.backward(q, ev)
                    e = self.al.edge(q, ev)
                    c = float((b ** 2).mean() if len(b) else 0) + 0.5 * float((e ** 2).mean() if len(e) else 0)
                    if c < best[0]:
                        best = (c, q)
        q, e0, e1, ok = self.refine_guarded(best[1], ev, max_step=(0.05, 0.03, 0.15), taus=(40.0, 20.0, 10.0, 6.0),
                                            nfev=40)
        return q, (e1 if ok else self.al.mean_line_error(best[1], ev))

    def get(self, frame_idx, image, boxes=None):
        """Pose of this frame -> pitch homography in the coordinates of ``image``.

        The venue camera is solved at a working resolution (640 x 360 by default: line alignment and background
        motion are as accurate and ~9x cheaper than at 1080p). A frame of another size with the same aspect ratio is
        downscaled for the tracking and the homography is scaled back, so a 1080p recording of the same venue uses the
        same camera file."""
        h, w = image.shape[:2]
        if w == self.cam.width:
            H = self._get(frame_idx, image, boxes)
            self.last_field = None if self.last_evidence is None else self.last_evidence.field
            return H
        s = self.cam.width / w
        if abs(h * s - self.cam.height) > 2.0:
            raise ValueError("PTZ camera was calibrated at %dx%d, frame is %dx%d (different aspect ratio)"
                             % (self.cam.width, self.cam.height, w, h))
        if getattr(self, "_scale", None) != s:          # ignore regions are given in video pixels
            self._scale = s
            self._ignore_full = getattr(self, "_ignore_full", None) or list(self.ignore)
            self.ignore = [[v * s for v in r] for r in self._ignore_full]
            self.mo.ignore = self.ignore
        small = cv2.resize(image, (self.cam.width, self.cam.height), interpolation=cv2.INTER_AREA)
        b = None if boxes is None else np.asarray(boxes, float).reshape(-1, 4) * s
        Hs = self._get(frame_idx, small, b)
        S = np.diag([1 / s, 1 / s, 1.0])                  # working pixels -> video pixels
        if self.last_image_motion is not None:
            self.last_image_motion = S @ self.last_image_motion @ np.linalg.inv(S)
        self.last_field = None if self.last_evidence is None else cv2.resize(
            self.last_evidence.field, (w, h), interpolation=cv2.INTER_NEAREST)
        return None if Hs is None else Homography(S @ Hs.H)

    def initialize(self, image, boxes=None):
        """First frame: the best of (a) every anchor pose re-localised on this frame and (b) a global search, by line
        alignment error. Without it a run starting far from the anchor (e.g. at the 2nd-half kick-off, or a clip
        starting before the landmark frame) followed a wrong pose for minutes."""
        ev = Evidence(image, boxes, self.ignore)
        if ev.n_line < self.min_line_px and len(ev.edge_px) <= 50:
            return
        cands = [np.asarray(c, float) for c in (self.init_candidates or [])]
        if self.p is not None:
            cands.insert(0, np.asarray(self.p, float))
        best = (np.inf, self.p, "anchor")
        for c in cands:
            q, e = self.relocalize(c, ev)
            if np.isfinite(e) and e < best[0]:
                best = (e, q, "anchor+relocalized")
        if not np.isfinite(best[0]) or best[0] > 2.0:
            q, _ = self.al.global_search(ev, pan_range=self.pan_range, f_range=self.f_range, n=(81, 5, 9), top=6)
            e = self.al.mean_line_error(q, ev)
            if np.isfinite(e) and e < best[0]:
                best = (e, q, "global")
        if best[1] is not None:
            self.p = np.asarray(best[1], float)
            self.init_report = dict(source=best[2], line_err=float(best[0]), n_candidates=len(cands))

    def _get(self, frame_idx, image, boxes=None):
        if self.k == 0 and (self.init_candidates or self.p is not None) and not getattr(self, "init_report", None):
            self.initialize(image, boxes)
        # passing a frame whose pose you validated (setup landmark frame / review repair): restart from it
        hit = [a for a in sorted(self.anchors) if self.prev_frame is not None and self.prev_frame < a <= frame_idx]
        if hit and frame_idx - hit[-1] <= self.anchor_snap_frames:
            self.p = np.asarray(self.anchors[hit[-1]], float)
            self.mo.reset()
            self.n_bad = self.n_lost = 0
            self.anchor_hits.append(int(hit[-1]))
        self.prev_frame = frame_idx
        pred, n_inl = self.mo.step(image, boxes, self.p)
        src = "motion" if pred is not None else "hold"
        p = pred if pred is not None else self.p
        ev, err = None, np.nan
        fast = pred is not None and self.p is not None and abs(pred[0] - self.p[0]) > 0.05   # fast pan: check every frame
        if p is None or pred is None or fast or self.k % self.refine_every == 0:
            ev = Evidence(image, boxes, self.ignore)
            if ev.n_line >= self.min_line_px or len(ev.edge_px) > 50:
                if p is None:
                    p, _ = self.al.global_search(ev, **self.search_kw)
                    src = "search"
                    err = self.al.mean_line_error(p, ev)
                else:
                    big = pred is None
                    q, e0, e1, ok = self.refine_guarded(p, ev, max_step=(0.08, 0.05, 0.2) if big else (0.035, 0.025, 0.08))
                    p, err = q, (e1 if ok else e0)
                    if ok:
                        src += "+lines"
        if np.isfinite(err):
            self.n_bad = self.n_bad + 1 if err > self.max_err else 0
        implausible = p is not None and not (self.pan_range[0] <= p[0] <= self.pan_range[1] and
                                             self.f_range[0] <= p[2] <= self.f_range[1] and -0.2 <= p[1] <= 0.5)
        if implausible or (np.isfinite(err) and err >= 0.9 * 10.0):
            self.n_lost += 1
        else:
            self.n_lost = 0
        if self.n_bad >= self.lost_after and ev is not None and p is not None and not implausible:
            q, e = self.relocalize(p, ev)
            if e < 0.8 * err:
                p, err, src = q, e, "relocalized"
            self.n_bad = 0
        if (self.n_lost >= self.lost_global and self.k - self.last_global >= self.lost_global) or (implausible and self.n_lost == 1):
            if ev is None:
                ev = Evidence(image, boxes, self.ignore)
            if ev.n_line >= self.min_line_px or len(ev.edge_px) > 100:
                q, _ = self.al.global_search(ev, pan_range=self.pan_range, f_range=self.f_range, n=(81, 5, 9), top=6)
                e = self.al.mean_line_error(q, ev)
                ee = self.al.edge(q, ev)
                e_edge = float(np.median(ee)) if len(ee) else 99.0
                self.last_global = self.k
                self.global_attempts.append((frame_idx, float(e), float(e_edge), int(ev.n_line)))
                if ev.n_line >= 150 and (e < 2.0 or (e < 3.0 and e_edge < 3.0)):   # strict: a wrong pose is worse than none
                    p, err, src = q, e, "global"
                    self.n_lost = self.n_bad = 0
                    self.mo.reset()
                    self.mo.step(image, boxes, p)
        self.last_image_motion = None
        if p is not None and self.p is not None:
            Rp, Rc = rotation(self.p[0], self.p[1], self.cam.roll), rotation(p[0], p[1], self.cam.roll)
            self.last_image_motion = self.cam.K(p[2]) @ Rc @ Rp.T @ np.linalg.inv(self.cam.K(self.p[2]))
        self.last_evidence = ev
        self.p = p
        self.k += 1
        self.log.append(dict(frame=frame_idx, src=src, inliers=n_inl, line_err=err,
                             line_px=None if ev is None else ev.n_line,
                             pan=None if p is None else float(p[0]), tilt=None if p is None else float(p[1]),
                             f=None if p is None else float(p[2])))
        return None if p is None else self.cam.homography(p)


# ----------------------------------------------------------------------------- venue calibration
def bundle_adjust(cam0: PTZCamera, evs: List["Evidence"], P0: np.ndarray, landmarks: Optional[List[dict]] = None,
                  taus=(30.0, 15.0, 8.0), max_nfev: int = 80, fit_margins: bool = True, fit_pitch: bool = True,
                  verbose: bool = True):
    """Jointly refine the venue (camera centre C, roll, grass margins, **pitch length & width**) and the poses
    of a set of keyframes.

    Lower-league pitches are rarely 105 x 68 m (IFAB allows 90-120 x 45-90 m), and a wrong pitch size makes the
    goals project too far out — so the pitch dimensions are estimated together with the camera.
    ``evs``: :class:`Evidence` per keyframe; ``P0``: starting (pan, tilt, f) — typically from the tracker.
    ``landmarks``: optional {landmark_name: [u, v]} per keyframe (clicked points).
    """
    from .pitch import Pitch as _Pitch
    n = len(evs)

    def make(z):
        pitch = _Pitch(z[6], z[7]) if fit_pitch else cam0.pitch
        cam = PTZCamera(z[:3], z[3], z[4], z[5], cam0.width, cam0.height, pitch)
        al = Aligner(cam)
        return cam, al, z[8:].reshape(-1, 3)

    def res(z):
        cam, al, P = make(z)
        lm = cam.pitch.landmarks()
        r = []
        for i, (ev, p) in enumerate(zip(evs, P)):
            r.append(al.residuals(p, ev))
            if landmarks and i < len(landmarks) and landmarks[i]:
                names = list(landmarks[i])
                uv, _ = cam.project_ours(np.array([lm[k] for k in names]), p)
                r.append(((uv - np.array([landmarks[i][k] for k in names])) / 1.5).ravel())
        r.append(np.array([z[3] / 0.03]))
        if fit_pitch:                                   # weak prior on the pitch size (keeps it physical)
            r.append(np.array([(z[6] - L0) / 8.0, (z[7] - W0) / 8.0]))
        return np.concatenate(r)

    L0, W0 = cam0.pitch.length, cam0.pitch.width
    z = np.r_[cam0.C, cam0.roll, cam0.margin_end, cam0.margin_far, L0, W0, np.asarray(P0, float).ravel()]
    fm = 0 if fit_margins else 1
    fp = 0 if fit_pitch else 1
    lo = np.r_[-30, -80, 1.0, -0.1, [0.0, 0.0] if fm == 0 else [cam0.margin_end - 1e-3, cam0.margin_far - 1e-3],
               [88.0, 44.0] if fp == 0 else [L0 - 1e-3, W0 - 1e-3], np.tile([-1.7, -0.3, 150], n)]
    hi = np.r_[135, -0.3, 40, 0.1, [10.0, 10.0] if fm == 0 else [cam0.margin_end + 1e-3, cam0.margin_far + 1e-3],
               [122.0, 92.0] if fp == 0 else [L0 + 1e-3, W0 + 1e-3], np.tile([1.7, 0.7, 8000], n)]
    z = np.clip(z, lo + 1e-6, hi - 1e-6)
    xs = np.r_[1, 1, 1, 0.01, 0.5, 0.5, 1, 1, np.tile([0.01, 0.01, 20], n)]
    for tau in taus:
        for ev in evs:
            ev.tau = tau
        sol = least_squares(res, z, bounds=(lo, hi), x_scale=xs, diff_step=1e-4, loss="soft_l1", f_scale=1.0,
                            max_nfev=max_nfev)
        z = sol.x
        if verbose:
            print(f"  tau={tau:4.0f}: C={np.round(z[:3], 2)} roll={np.degrees(z[3]):.2f}° margins={np.round(z[4:6], 2)} "
                  f"pitch={z[6]:.1f}x{z[7]:.1f} cost={float((sol.fun ** 2).sum()):.1f}", flush=True)
    for ev in evs:
        ev.tau = 10.0
    cam, al, P = make(z)
    errs = np.array([al.mean_line_error(p, ev) for ev, p in zip(evs, P)])
    return cam, P, errs


def track_video(video: str, cam: PTZCamera, p0, t0: float, t1: float, fps: float = 5.0, ignore=(),
                detector=None, every_snapshot_s: float = 20.0, refine_every: int = 1):
    """Run the PTZ tracker on [t0, t1]; returns (log DataFrame-like list, {t: frame} snapshots)."""
    from .video import iter_frames
    prov = PTZProvider(cam, ignore, init_params=p0, refine_every=refine_every)
    snaps = {}
    next_snap = t0
    for f, t, im in iter_frames(video, fps, t0, t1):
        boxes = detector(im).of("player", "goalkeeper", "referee").xyxy if detector is not None else None
        prov.get(f, im, boxes)
        prov.log[-1]["t"] = t
        if t >= next_snap:
            snaps[round(t, 2)] = im
            next_snap = t + every_snapshot_s
    return prov.log, snaps


def initial_fit(img, points, ignore, pitch, width: Optional[int] = None, height: Optional[int] = None):
    """Multi-start fit of the camera centre C, roll and the pose of ONE frame from its clicked landmarks + lines.

    ``img`` is the frame at the working resolution (``width`` x ``height``, default the image size); ``points`` are
    {landmark: [u, v]} in that resolution. Returns (PTZCamera, pose)."""
    h_img, w_img = img.shape[:2]
    width, height = int(width or w_img), int(height or h_img)
    lm = pitch.landmarks()
    ev = Evidence(img, None, ignore)
    names = [n for n in points if n in lm]
    uvs = np.array([points[n] for n in names], float)
    best = None
    for Y in (-3, -6, -10, -15, -22, -30):
        for Z in (3, 5, 8, 12):
            for X in (0.33 * pitch.length, 0.5 * pitch.length, 0.67 * pitch.length):
                def res(z):
                    cam = PTZCamera(z[:3], z[3], pitch=pitch, width=width, height=height)
                    uv, _ = cam.project_ours(np.array([lm[n] for n in names]), z[4:])
                    return np.r_[((uv - uvs) / 2).ravel(), Aligner(cam).residuals(z[4:], ev), z[3] / 0.03]
                d = np.hypot(pitch.length / 2 - X, pitch.width / 2 - Y)
                z0 = [X, Y, Z, 0.0, np.arctan2(pitch.length / 2 - X, pitch.width / 2 - Y), np.arctan2(Z, d) - 0.1,
                      700.0 * width / 640]
                lo, hi = [-30, -80, 1, -0.1, -1.7, -0.3, 150], [135, -0.3, 40, 0.1, 1.7, 0.7, 8000]
                try:
                    s = least_squares(res, z0, bounds=(lo, hi), x_scale=[1, 1, 1, 0.01, 0.01, 0.01, 20],
                                      diff_step=1e-4, loss="soft_l1", max_nfev=80)
                except ValueError:
                    continue
                c = float((s.fun ** 2).sum())
                if best is None or c < best[0]:
                    best = (c, s.x)
    z = best[1]
    return PTZCamera(z[:3], z[3], pitch=pitch, width=width, height=height), z[4:]


def landmark_errors(cam: "PTZCamera", p, points: dict) -> np.ndarray:
    """Pixel distance between clicked landmarks and their projection with pose ``p``."""
    lm = cam.pitch.landmarks()
    names = [n for n in points if n in lm]
    if not names:
        return np.zeros(0)
    uv, _ = cam.project_ours(np.array([lm[n] for n in names]), p)
    return np.linalg.norm(uv - np.array([points[n] for n in names], float), axis=1)


def camera_from_homography(points: dict, pitch: Pitch, width: int, height: int, ransac_frac: float = 0.03):
    """Closed-form camera from >= 4 clicked landmarks of one frame (milliseconds instead of a multi-start search).

    Plane-to-image homography (RANSAC: a mislabelled click is found and reported) -> focal length from the two
    rotation constraints (principal point at the image centre) -> rotation, camera centre C -> pan / tilt / roll.
    Returns ``(C, roll, (pan, tilt, f), inlier names, outlier names)`` or None."""
    lm = pitch.landmarks()
    names = [n for n in points if n in lm]
    if len(names) < 4:
        return None
    X = np.array([lm[n] for n in names], float)
    uv = np.array([points[n] for n in names], float)
    Hh, inl = cv2.findHomography(X, uv, cv2.RANSAC if len(names) > 4 else 0, ransac_frac * width)
    if Hh is None:
        return None
    inl = np.ones(len(names), bool) if inl is None else inl.ravel().astype(bool)
    M = np.array([[1.0, 0, 0], [0, -1.0, pitch.width], [0, 0, 1.0]])         # ours <-> world (an involution)
    G = np.array([[1.0, 0, -width / 2], [0, 1.0, -height / 2], [0, 0, 1.0]]) @ Hh @ M
    h1, h2 = G[:, 0], G[:, 1]
    cands = []
    if abs(h1[2] * h2[2]) > 1e-12:
        cands.append(-(h1[0] * h2[0] + h1[1] * h2[1]) / (h1[2] * h2[2]))
    if abs(h1[2] ** 2 - h2[2] ** 2) > 1e-12:
        cands.append(-(h1[0] ** 2 + h1[1] ** 2 - h2[0] ** 2 - h2[1] ** 2) / (h1[2] ** 2 - h2[2] ** 2))
    pos = [c for c in cands if c > 0]
    f = float(np.sqrt(np.median(pos))) if pos else float(width)
    A = np.diag([1 / f, 1 / f, 1.0]) @ G
    lam = 2.0 / (np.linalg.norm(A[:, 0]) + np.linalg.norm(A[:, 1]))
    best = None
    for sg in (1.0, -1.0):
        r1, r2, t = sg * lam * A[:, 0], sg * lam * A[:, 1], sg * lam * A[:, 2]
        U, _, Vt = np.linalg.svd(np.c_[r1, r2, np.cross(r1, r2)])
        R = U @ Vt
        if np.linalg.det(R) < 0:
            continue
        C = -R.T @ t
        if C[2] > 0:                                    # camera above the ground
            best = (R, C)
    if best is None:
        return None
    R, C = best
    sol = None
    for pan in np.linspace(-1.6, 1.6, 9):
        s_ = least_squares(lambda q: (rotation(q[0], q[1], q[2]) - R).ravel(), [pan, 0.1, 0.0])
        if sol is None or s_.cost < sol.cost:
            sol = s_
    pan, tilt, roll = sol.x
    roll = float((roll + np.pi) % (2 * np.pi) - np.pi)
    return (C, roll, np.array([pan, tilt, f]), [n for n, k in zip(names, inl) if k],
            [n for n, k in zip(names, inl) if not k])


def _pose_given_camera(cam: "PTZCamera", points: dict, p0) -> np.ndarray:
    lm = cam.pitch.landmarks()
    names = [n for n in points if n in lm]
    X = np.array([lm[n] for n in names])
    uvs = np.array([points[n] for n in names], float)

    def res(p):
        uv, front = cam.project_ours(X, p)
        return np.where(front[:, None], uv - uvs, 500.0).ravel()
    return least_squares(res, np.asarray(p0, float), bounds=([-1.7, -0.3, 150], [1.7, 0.7, 8000]),
                         x_scale=[0.01, 0.01, 20], loss="soft_l1", f_scale=3.0, max_nfev=200).x


def solve_from_landmarks(frames: List[tuple], pitch: Pitch, ignore=(), fit_pitch: bool = False,
                         verbose: bool = False, outlier_px: Optional[float] = None) -> tuple:
    """Venue camera from the landmark frames clicked in the setup UI only (no video tracking needed).

    ``frames``: [(image, {landmark: [u, v]}), ...] at the working resolution.

    1. closed-form camera per frame (:func:`camera_from_homography`); clicks that do not fit the pitch model of their
       frame (RANSAC) are dropped and reported — typically a landmark given the wrong name;
    2. the camera of the frame with the most consistent clicks starts the fit; every frame gets its pose from its own
       clicks with that camera (multi-start search only when no closed form exists);
    3. bundle adjustment of C / roll / poses with the clicks AND the white lines of every frame;
    4. clicks still farther than ``outlier_px`` from their projection are dropped and step 3 is repeated once.

    Returns (PTZCamera, [pose per input frame], [errors px per input frame], {frame index: [dropped clicks]})."""
    h, w = frames[0][0].shape[:2]
    outlier_px = outlier_px or 0.03 * w
    inits, kept = {}, []
    dropped = {i: [] for i in range(len(frames))}
    for i, (_, pts) in enumerate(frames):
        r = camera_from_homography(pts, pitch, w, h)
        if r is not None:
            inits[i] = r
            dropped[i] = list(r[4])
        kept.append({k: v for k, v in pts.items() if k not in dropped[i]})
    if inits:
        i0 = max(inits, key=lambda i: len(inits[i][3]))
        C, roll, p0 = inits[i0][:3]
        cam = PTZCamera(np.asarray(C, float), roll, pitch=pitch, width=w, height=h)
    else:                                               # fewer than 4 clicks everywhere: slow multi-start search
        i0 = max(range(len(frames)), key=lambda i: len(frames[i][1]))
        cam, p0 = initial_fit(frames[i0][0], frames[i0][1], ignore, pitch)
    P0 = []
    for i, (img, _) in enumerate(frames):
        if i in inits:
            p = _pose_given_camera(cam, kept[i], inits[i][2])
        else:
            p, _ = pose_from_landmarks(cam, Evidence(img, None, ignore), kept[i], cam.pitch)
        P0.append(p)
    evs = [Evidence(img, None, ignore) for img, _ in frames]
    for it in range(2):
        cam, P, _ = bundle_adjust(cam, evs, np.vstack(P0), kept, fit_pitch=fit_pitch, verbose=verbose,
                                  taus=(15.0, 8.0), max_nfev=60)
        P0 = list(P)
        again = False
        for i in range(len(frames)):
            e = landmark_errors(cam, P0[i], kept[i])
            names = [n for n in kept[i] if n in cam.pitch.landmarks()]
            bad = [n for n, ee in zip(names, e) if ee > max(outlier_px, 3 * float(np.median(e)))]
            if bad and len(names) - len(bad) >= 4:
                dropped[i] += bad
                kept[i] = {k: v for k, v in kept[i].items() if k not in bad}
                again = True
        if not again:
            break
    errs = [landmark_errors(cam, P0[i], kept[i]) for i in range(len(frames))]
    return cam, P0, errs, dropped


def pose_from_landmarks(cam, ev, points, pitch, pan_step: float = 0.08, use_lines: bool = True):
    """Pose (pan, tilt, f) of one frame from >= 2 clicked landmarks + its lines, camera fixed. Multi-start."""
    lm = pitch.landmarks()
    names = list(points)
    X = np.array([lm[n] for n in names])
    uvs = np.array([points[n] for n in names], float)
    al = Aligner(cam)
    best = None
    for pan in np.arange(-1.6, 1.61, pan_step):
        for f in (400.0, 650.0, 1000.0, 1500.0):
            def res(p):
                uv, front = cam.project_ours(X, p)
                r = np.where(front[:, None], uv - uvs, 500.0).ravel() / 2.0
                return np.r_[r, 0.3 * al.residuals(p, ev)] if (use_lines and ev is not None) else r
            try:
                s = least_squares(res, [pan, 0.05, f], bounds=([-1.7, -0.3, 150], [1.7, 0.7, 8000]),
                                  x_scale=[0.01, 0.01, 20], diff_step=1e-4, loss="soft_l1", max_nfev=60)
            except ValueError:
                continue
            uv, _ = cam.project_ours(X, s.x)
            lm_err = float(np.abs(uv - uvs).max())
            if best is None or lm_err < best[0]:
                best = (lm_err, s.x)
    return best[1], best[0]


def draw_template(img: np.ndarray, cam: PTZCamera, p, color=(255, 0, 255), thickness: int = 1) -> np.ndarray:
    """Overlay the projected pitch markings (segment by segment, so nothing behind the camera is drawn)."""
    out = img.copy()
    A, B = template_segments(cam.pitch, 1.0)
    a, fa = cam.project_ours(A, p)
    b, fb = cam.project_ours(B, p)
    ok = fa & fb & np.all(np.abs(a) < 5e3, 1) & np.all(np.abs(b) < 5e3, 1)
    for (u0, v0), (u1, v1) in zip(a[ok], b[ok]):
        cv2.line(out, (int(u0), int(v0)), (int(u1), int(v1)), color, thickness, cv2.LINE_AA)
    return out


def save_ptz(path: str, cam: PTZCamera, anchors: Optional[dict] = None, meta: Optional[dict] = None):
    d = cam.to_json()
    if anchors:
        d["anchors"] = {str(k): list(map(float, v)) for k, v in anchors.items()}
    if meta:
        d["meta"] = meta
    Path(path).write_text(json.dumps(d, indent=2))


def load_ptz(path: str, pitch: Pitch = None):
    d = json.loads(Path(path).read_text())
    cam = PTZCamera.from_json(d, pitch)
    return cam, {int(k): np.array(v) for k, v in d.get("anchors", {}).items()}
