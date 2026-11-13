"""Synthetic broadcast frames rendered from a known camera — for testing calibration and for the demo.

`render_frame` paints the pitch template (optionally with an imperfectly painted centre circle) seen
by a ground-truth camera; `hint_specs` produces the rough click-style hints a user would give.
"""
import cv2
import numpy as np

from .camera import make_camera, look_at_rvec, project, ground
from .template import polyline, all_element_names, kind_of, line_elements, circle_elements


def camera_looking_at(C, target, f, size):
    return make_camera(f, look_at_rvec(C, target), C, size[0] / 2, size[1] / 2)


def _visible(P, cam, size):
    Xc = (P - cam['C']) @ cam['R'].T
    uv = project(P, cam)
    ok = (Xc[:, 2] > 1) & (uv[:, 0] >= 0) & (uv[:, 0] < size[0]) & (uv[:, 1] >= 0) & (uv[:, 1] < size[1])
    return uv, ok


def painted(name, hl, hw, circle_error=(0.0, 0.0), n=2000):
    P = polyline(name, hl, hw, n)
    if name == 'centre_circle' and any(circle_error):
        a = np.arctan2(P[:, 1], P[:, 0])
        r = 9.15 * (1 + circle_error[0] * np.cos(2 * a + circle_error[1]))
        P = np.c_[r * np.cos(a), r * np.sin(a), 0 * a]
    return P


def render_frame(cam, size, hl=52.5, hw=34.0, circle_error=(0.0, 0.0), seed=0, line_w=0.12):
    W, H = size
    rng = np.random.default_rng(seed)
    img = np.zeros((H, W, 3), np.uint8)
    img[:] = (40, 130, 60)
    img = cv2.add(img, rng.integers(0, 25, (H, W, 3), dtype=np.uint8))
    # stands above the horizon
    G = ground(np.c_[np.full(H, W / 2), np.arange(H)].astype(float), cam)
    sky = (G[:, 1] - cam['C'][1] < 0) | (G[:, 1] > hw + 6)
    img[sky] = (110, 110, 115)
    for name in all_element_names():
        if kind_of(name) == 'point':
            continue
        P = painted(name, hl, hw, circle_error)
        uv, ok = _visible(P, cam, (W * 3, H * 3))
        depth = ((P - cam['C']) @ cam['R'].T)[:, 2]
        for a, b, d, o1, o2 in zip(uv[:-1], uv[1:], depth[:-1], ok[:-1], ok[1:]):
            if o1 and o2 and d > 1:
                th = max(2, int(round(cam['f'] * line_w / d)))
                cv2.line(img, tuple(np.round(a).astype(int)), tuple(np.round(b).astype(int)), (235, 240, 235), th, cv2.LINE_AA)
    return cv2.GaussianBlur(img, (3, 3), 0)


def visible_elements(cam, size, hl=52.5, hw=34.0, min_px=300):
    out = []
    for name in all_element_names():
        if kind_of(name) == 'point':
            continue
        uv, ok = _visible(polyline(name, hl, hw, 400), cam, size)
        if ok.sum() > 20 and np.ptp(uv[ok], axis=0).max() > min_px:
            out.append(name)
    return out


def hint_specs(cam, size, elements, hl=52.5, hw=34.0, jitter=4.0, seed=0):
    """Rough hints like a user would click: 2 points per straight line, a bbox per circle, 5 points per arc."""
    rng = np.random.default_rng(seed)
    specs = []
    for name in elements:
        uv, ok = _visible(polyline(name, hl, hw, 400), cam, size)
        uv = uv[ok]
        if kind_of(name) == 'line':
            pts = uv[[0, -1]] + rng.normal(0, jitter, (2, 2))
            specs.append(dict(element=name, polyline=pts.round().tolist(), band=20))
        elif name == 'centre_circle' and ok.all():
            x0, y0 = uv.min(0) - 25
            x1, y1 = uv.max(0) + 25
            specs.append(dict(element=name, bbox=[float(x0), float(y0), float(x1), float(y1)]))
        else:
            idx = np.linspace(0, len(uv) - 1, 5).astype(int)
            specs.append(dict(element=name, polyline=(uv[idx] + rng.normal(0, jitter, (5, 2))).round().tolist(), band=15))
    return specs
