"""Pitch geometry (IFAB Law 1) and image <-> pitch homographies for a single camera.

Pitch frame (metres): origin at the top-left corner flag as seen on the minimap,
x along the touchline in [0, L], y across the pitch in [0, W]. Goals are centred at
(0, W/2) and (L, W/2).

A single camera sees the pitch plane through a homography H (3x3). Three ways to get H:

* ``StaticHomography``   — tripod / fixed camera: calibrate once (>= 4 named landmarks).
* ``KeypointHomography`` — panning broadcast camera: a pose model predicts pitch keypoints
  every frame (SoTA: HRNet/YOLO-pose keypoints, or line-based PnLCalib / NBJW calibrators).
* ``FlowHomography``     — panning camera without a keypoint model: chain frame-to-frame
  background homographies (ORB + RANSAC, players masked) from manually calibrated anchors.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Dict, Optional, Sequence

import cv2
import numpy as np


# ----------------------------------------------------------------------------- geometry
@dataclass
class Pitch:
    length: float = 105.0
    width: float = 68.0
    goal_width: float = 7.32
    penalty_area_depth: float = 16.5
    penalty_area_width: float = 40.32
    goal_area_depth: float = 5.5
    goal_area_width: float = 18.32
    penalty_spot: float = 11.0
    circle_radius: float = 9.15

    # ---- landmarks
    def landmarks(self) -> Dict[str, tuple]:
        L, W = self.length, self.width
        c = W / 2
        pa, ga = self.penalty_area_width / 2, self.goal_area_width / 2
        d = {
            "corner_tl": (0, 0), "corner_bl": (0, W), "corner_tr": (L, 0), "corner_br": (L, W),
            "halfway_top": (L / 2, 0), "halfway_bottom": (L / 2, W), "centre": (L / 2, c),
            "circle_top": (L / 2, c - self.circle_radius), "circle_bottom": (L / 2, c + self.circle_radius),
            "circle_left": (L / 2 - self.circle_radius, c), "circle_right": (L / 2 + self.circle_radius, c),
        }
        for side, x0, sgn in (("l", 0.0, 1), ("r", L, -1)):
            d[f"pa_{side}_top_line"] = (x0, c - pa)
            d[f"pa_{side}_bottom_line"] = (x0, c + pa)
            d[f"pa_{side}_top"] = (x0 + sgn * self.penalty_area_depth, c - pa)
            d[f"pa_{side}_bottom"] = (x0 + sgn * self.penalty_area_depth, c + pa)
            d[f"ga_{side}_top_line"] = (x0, c - ga)
            d[f"ga_{side}_bottom_line"] = (x0, c + ga)
            d[f"ga_{side}_top"] = (x0 + sgn * self.goal_area_depth, c - ga)
            d[f"ga_{side}_bottom"] = (x0 + sgn * self.goal_area_depth, c + ga)
            d[f"spot_{side}"] = (x0 + sgn * self.penalty_spot, c)
            d[f"post_{side}_top"] = (x0, c - self.goal_width / 2)
            d[f"post_{side}_bottom"] = (x0, c + self.goal_width / 2)
        return {k: (float(v[0]), float(v[1])) for k, v in d.items()}

    def roboflow32(self) -> np.ndarray:
        """Keypoint order of the Roboflow `sports` SoccerPitchConfiguration (32 vertices).

        Verify against the config of the model you load — the order must match training.
        """
        L, W = self.length, self.width
        pw, pl = self.penalty_area_width, self.penalty_area_depth
        gw, gl = self.goal_area_width, self.goal_area_depth
        r, ps = self.circle_radius, self.penalty_spot
        v = [(0, 0), (0, (W - pw) / 2), (0, (W - gw) / 2), (0, (W + gw) / 2), (0, (W + pw) / 2), (0, W),
             (gl, (W - gw) / 2), (gl, (W + gw) / 2), (ps, W / 2),
             (pl, (W - pw) / 2), (pl, (W - gw) / 2), (pl, (W + gw) / 2), (pl, (W + pw) / 2),
             (L / 2, 0), (L / 2, W / 2 - r), (L / 2, W / 2 + r), (L / 2, W),
             (L - pl, (W - pw) / 2), (L - pl, (W - gw) / 2), (L - pl, (W + gw) / 2), (L - pl, (W + pw) / 2),
             (L - ps, W / 2), (L - gl, (W - gw) / 2), (L - gl, (W + gw) / 2),
             (L, 0), (L, (W - pw) / 2), (L, (W - gw) / 2), (L, (W + gw) / 2), (L, (W + pw) / 2), (L, W),
             (L / 2 - r, W / 2), (L / 2 + r, W / 2)]
        return np.array(v, float)

    # ---- polylines for drawing the minimap
    def lines(self, n_arc: int = 60):
        L, W, c = self.length, self.width, self.width / 2
        pa, ga = self.penalty_area_width / 2, self.goal_area_width / 2
        out = [np.array([(0, 0), (L, 0), (L, W), (0, W), (0, 0)]), np.array([(L / 2, 0), (L / 2, W)])]
        for x0, s in ((0, 1), (L, -1)):
            out.append(np.array([(x0, c - pa), (x0 + s * self.penalty_area_depth, c - pa),
                                 (x0 + s * self.penalty_area_depth, c + pa), (x0, c + pa)]))
            out.append(np.array([(x0, c - ga), (x0 + s * self.goal_area_depth, c - ga),
                                 (x0 + s * self.goal_area_depth, c + ga), (x0, c + ga)]))
            # penalty arc (outside the area only)
            a = np.arccos((self.penalty_area_depth - self.penalty_spot) / self.circle_radius)
            th = np.linspace(-a, a, n_arc)
            sx = x0 + s * self.penalty_spot
            out.append(np.c_[sx + s * self.circle_radius * np.cos(th), c + self.circle_radius * np.sin(th)])
        th = np.linspace(0, 2 * np.pi, 2 * n_arc)
        out.append(np.c_[L / 2 + self.circle_radius * np.cos(th), c + self.circle_radius * np.sin(th)])
        return out

    # ---- helpers used by the rules
    def goal_centre(self, right: bool) -> np.ndarray:
        return np.array([self.length if right else 0.0, self.width / 2])

    def in_pitch(self, x, y, margin: float = 0.0):
        x, y = np.asarray(x), np.asarray(y)
        return (x >= -margin) & (x <= self.length + margin) & (y >= -margin) & (y <= self.width + margin)

    def in_penalty_area(self, x, y, right: bool):
        x, y = np.asarray(x), np.asarray(y)
        inx = (x >= self.length - self.penalty_area_depth) if right else (x <= self.penalty_area_depth)
        return inx & (np.abs(y - self.width / 2) <= self.penalty_area_width / 2)


# ----------------------------------------------------------------------------- homographies
class Homography:
    """Wraps H (pitch -> image) and its inverse."""

    def __init__(self, H_pitch2img: np.ndarray):
        self.H = np.asarray(H_pitch2img, float)
        self.Hinv = np.linalg.inv(self.H)

    @staticmethod
    def _apply(H, pts):
        pts = np.asarray(pts, float).reshape(-1, 2)
        if len(pts) == 0:
            return pts
        return cv2.perspectiveTransform(pts[None], H)[0]

    def to_pitch(self, img_pts):
        return self._apply(self.Hinv, img_pts)

    def to_image(self, pitch_pts):
        return self._apply(self.H, pitch_pts)

    def reprojection_error(self, img_pts, pitch_pts) -> float:
        e = self.to_image(pitch_pts) - np.asarray(img_pts, float)
        return float(np.sqrt((e ** 2).sum(1)).mean())

    @classmethod
    def fit(cls, img_pts, pitch_pts, ransac_px: float = 8.0) -> "Homography":
        img_pts, pitch_pts = np.asarray(img_pts, np.float32), np.asarray(pitch_pts, np.float32)
        if len(img_pts) < 4:
            raise ValueError("Need >= 4 correspondences for a homography")
        method = cv2.RANSAC if len(img_pts) > 4 else 0
        H, _ = cv2.findHomography(pitch_pts, img_pts, method, ransac_px)
        if H is None:
            raise ValueError("Homography estimation failed (degenerate points?)")
        return cls(H)


def load_calibration(path: str, pitch: Pitch) -> Homography:
    """Load a calibration json. Two formats are accepted:

    1. Correspondences::

        {"points": {"centre": [960, 540], "halfway_top": [955, 210], ...}}   # landmark name -> pixel
        {"pairs": [[[u, v], [x, y]], ...]}                                   # explicit pixel/metre pairs

    2. The camera model saved by the sibling ``offiside_code`` project
       (``{"camera": {f, rvec, C, cx, cy}, "half_length", "half_width"}``, origin at the centre spot).
    """
    with open(path) as f:
        return calibration_from_dict(json.load(f), pitch)


def calibration_from_dict(d: dict, pitch: Pitch) -> Homography:
    """See :func:`load_calibration` for the accepted formats."""
    if "camera" in d:
        cam = d["camera"]
        R = cv2.Rodrigues(np.asarray(cam["rvec"], float).reshape(3, 1))[0]
        C = np.asarray(cam["C"], float)
        K = np.array([[cam["f"], 0, cam["cx"]], [0, cam["f"], cam["cy"]], [0, 0, 1.0]])
        H_centre = K @ np.c_[R[:, 0], R[:, 1], -R @ C]      # (X, Y, 1) centred frame -> image
        # centred frame (X right, Y towards far touchline) -> ours (x right, y from top touchline)
        hl, hw = d.get("half_length", pitch.length / 2), d.get("half_width", pitch.width / 2)
        T = np.array([[1, 0, -hl], [0, -1, hw], [0, 0, 1.0]])   # ours -> centred
        return Homography(H_centre @ T)
    lm = pitch.landmarks()
    img, pit = [], []
    for name, uv in (d.get("points") or {}).items():
        if name not in lm:
            raise KeyError(f"Unknown landmark '{name}'. Valid: {sorted(lm)}")
        img.append(uv)
        pit.append(lm[name])
    for uv, xy in d.get("pairs", []):
        img.append(uv)
        pit.append(xy)
    return Homography.fit(img, pit)


class HomographyProvider:
    """Interface: ``get(frame_idx, image, player_boxes) -> Homography | None``."""

    def get(self, frame_idx: int, image: np.ndarray, boxes: Optional[np.ndarray] = None) -> Optional[Homography]:
        raise NotImplementedError


class StaticHomography(HomographyProvider):
    def __init__(self, H: Homography):
        self.H = H

    def get(self, frame_idx, image, boxes=None):
        return self.H


class KeypointHomography(HomographyProvider):
    """Per-frame pitch keypoints from an ultralytics pose model (e.g. Roboflow football-field-detection)."""

    def __init__(self, model_path: str, pitch: Pitch, conf: float = 0.5, layout: str = "roboflow32",
                 device: str = "cpu", smooth: float = 0.6):
        from ultralytics import YOLO  # optional dependency
        self.model = YOLO(model_path)
        self.ref = pitch.roboflow32() if layout == "roboflow32" else None
        if self.ref is None:
            raise ValueError(f"Unknown keypoint layout {layout}")
        self.conf, self.device, self.smooth = conf, device, smooth
        self._last: Optional[np.ndarray] = None

    def get(self, frame_idx, image, boxes=None):
        r = self.model.predict(image, verbose=False, device=self.device)[0]
        if r.keypoints is None or len(r.keypoints) == 0:
            return Homography(self._last) if self._last is not None else None
        xy = r.keypoints.xy[0].cpu().numpy()
        cf = r.keypoints.conf[0].cpu().numpy() if r.keypoints.conf is not None else np.ones(len(xy))
        ok = (cf > self.conf) & (xy[:, 0] > 1) & (xy[:, 1] > 1)
        if ok.sum() < 4:
            return Homography(self._last) if self._last is not None else None
        try:
            H = Homography.fit(xy[ok], self.ref[ok]).H
        except ValueError:
            return Homography(self._last) if self._last is not None else None
        H = H / H[2, 2]
        if self._last is not None and self.smooth > 0:      # EMA on normalised H against jitter
            H = self.smooth * self._last + (1 - self.smooth) * H
        self._last = H
        return Homography(H)


class FlowHomography(HomographyProvider):
    """Panning camera: H_t = H_{t-1} composed with the background motion between frames.

    ``anchors`` maps frame index -> Homography (from manual calibration). Drift accumulates, so
    add anchors every few minutes or combine with a keypoint model (``keypoint_provider``).
    """

    def __init__(self, anchors: Dict[int, Homography], keypoint_provider: Optional[KeypointHomography] = None,
                 reanchor_every: int = 0, n_features: int = 1500, snap_frames: int = 3):
        self.anchors = dict(anchors)
        # anchors are clicked on any source frame, but only every k-th frame is analysed (k = fps / analysis_fps):
        # an anchor is used on the first analysed frame within ``snap_frames`` of it (before: exact match only, so
        # anchors on odd frames were silently ignored at 24 fps -> 10 fps)
        self.snap = int(snap_frames)
        self._used: set = set()
        self.kp = keypoint_provider
        self.reanchor_every = reanchor_every
        self.orb = cv2.ORB_create(n_features)
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        self._prev = None  # (gray, kps, des, H)

    def _features(self, gray, boxes):
        mask = np.full(gray.shape, 255, np.uint8)
        if boxes is not None:
            for x0, y0, x1, y1 in np.asarray(boxes, int):
                mask[max(y0 - 5, 0):y1 + 5, max(x0 - 5, 0):x1 + 5] = 0
        return self.orb.detectAndCompute(gray, mask)

    def get(self, frame_idx, image, boxes=None):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        kps, des = self._features(gray, boxes)
        H = None
        near = [a for a in self.anchors if abs(a - frame_idx) <= self.snap and a not in self._used]
        if near:
            a = min(near, key=lambda q: abs(q - frame_idx))
            self._used.add(a)
            H = self.anchors[a].H
        elif self.kp is not None and self.reanchor_every and frame_idx % self.reanchor_every == 0:
            h = self.kp.get(frame_idx, image, boxes)
            H = h.H if h is not None else None
        if H is None and self._prev is not None and des is not None and self._prev[2] is not None:
            m = self.bf.match(self._prev[2], des)
            if len(m) >= 12:
                p0 = np.float32([self._prev[1][x.queryIdx].pt for x in m])
                p1 = np.float32([kps[x.trainIdx].pt for x in m])
                F, inl = cv2.findHomography(p0, p1, cv2.RANSAC, 3.0)   # image_{t-1} -> image_t
                if F is not None and inl.sum() >= 10:
                    H = F @ self._prev[3]
            if H is None:
                H = self._prev[3]   # hold last estimate
        if H is None:
            return None
        self._prev = (gray, kps, des, H)
        return Homography(H)


def build_provider(cfg, pitch: Pitch, fps: float) -> Optional[HomographyProvider]:
    """Homography source for every frame. Points clicked in the setup UI (``match_setup.json``) take precedence
    over calibration files: ``pitch.calibration_points`` (static), ``pitch.flow_anchor_points`` (flow),
    ``pitch.extra_anchors`` (ptz poses validated by the user)."""
    from .config import resolve_path
    pc = cfg.pitch
    pc.calibration_file = resolve_path(pc.calibration_file)
    pc.keypoint_model = resolve_path(pc.keypoint_model)
    if pc.homography == "static":
        if pc.calibration_points:
            return StaticHomography(calibration_from_dict({"points": pc.calibration_points}, pitch))
        if not pc.calibration_file:
            return None
        return StaticHomography(load_calibration(pc.calibration_file, pitch))
    if pc.homography == "keypoints":
        return KeypointHomography(pc.keypoint_model, pitch, pc.keypoint_conf, pc.keypoint_layout, cfg.detection.device)
    if pc.homography == "ptz":
        from .ptz import PTZProvider, load_ptz
        cam, anchors = load_ptz(pc.calibration_file, pitch)
        if pc.fixed_size and (abs(cam.pitch.length - pitch.length) > 1.0 or abs(cam.pitch.width - pitch.width) > 1.0):
            print(f"[warn] venue camera was solved for a {cam.pitch.length:.1f} x {cam.pitch.width:.1f} m pitch but the "
                  f"setup says {pitch.length:.1f} x {pitch.width:.1f} m -> using the camera's size; re-run "
                  "scripts/calibrate_ptz.py --setup <match_setup.json> to solve it with the real size")
        for k, v in (pc.extra_anchors or {}).items():
            anchors[int(k)] = np.asarray(v, float)
        start_frame = int(cfg.video.start_s * fps)
        init = None
        if anchors:
            k = min(anchors, key=lambda a: (a > start_frame, abs(a - start_frame)))
            init = anchors[k]
        prov = PTZProvider(cam, cfg.video.ignore_regions, refine_every=pc.ptz_refine_every, init_params=init,
                           max_err=pc.ptz_max_err_px)
        prov.anchor_frame = k if anchors else None
        return prov
    if pc.homography == "flow":
        anchors = {}
        if pc.flow_anchor_points:
            for k, pts in pc.flow_anchor_points.items():
                anchors[int(k)] = calibration_from_dict({"points": pts}, pitch)
        elif pc.calibration_file:
            with open(pc.calibration_file) as f:
                d = json.load(f)
            # {"anchors": {"<frame_idx>": {"points": {...}}}} or a single calibration for frame 0
            for k, v in (d.get("anchors") or {"0": d}).items():
                anchors[int(k)] = calibration_from_dict(v, pitch)
        kp = KeypointHomography(pc.keypoint_model, pitch, pc.keypoint_conf) if pc.keypoint_model else None
        step = max(1, int(round(fps / max(cfg.video.analysis_fps, 1e-6))))
        return FlowHomography(anchors, kp, int(pc.reanchor_every_s * fps), snap_frames=step)
    raise ValueError(f"Unknown homography mode {pc.homography}")


def draw_pitch(ax, pitch: Pitch, color="#bbbbbb", lw=1.0, grass="#2e7d32", alpha_grass=0.12):
    """Matplotlib minimap (y axis inverted so the top touchline is at the top)."""
    import matplotlib.patches as mpatches
    ax.add_patch(mpatches.Rectangle((-3, -3), pitch.length + 6, pitch.width + 6, color=grass, alpha=alpha_grass, lw=0))
    for ln in pitch.lines():
        ax.plot(ln[:, 0], ln[:, 1], color=color, lw=lw, zorder=1)
    for x0, s in ((0, -1), (pitch.length, 1)):
        g = pitch.goal_width / 2
        ax.plot([x0, x0 + s * 1.5, x0 + s * 1.5, x0], [pitch.width / 2 - g] * 2 + [pitch.width / 2 + g] * 2,
                color=color, lw=lw)
    ax.set_xlim(-4, pitch.length + 4)
    ax.set_ylim(pitch.width + 4, -4)
    ax.set_aspect("equal")
    ax.axis("off")
    return ax
