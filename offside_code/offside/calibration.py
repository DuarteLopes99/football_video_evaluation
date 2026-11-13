"""Single- and multi-frame camera calibration against the pitch template.

Unknowns
  shared by all frames : pitch half-length hl, half-width hw, principal-point offset (dx, dy)
                         from the image centre, player shoulder height hs (weak cue)
  per camera group     : camera centre C (a broadcast camera on a fixed head / tripod pans, tilts
                         and zooms but does not translate, so all its frames share C)
  per frame            : focal length f and rotation R (pan / tilt / roll)

Residuals (pixels)
  line element  : back-project the pixel to the pitch, snap the fixed coordinate (e.g. X = 0 for the
                  halfway line), re-project, take the image distance          (PnLCalib/TVCalib-style)
  circle element: back-project, snap to the circle along its own angle, re-project
  point element : direct re-projection error (x5 weight)
  inside-pitch  : hinge — pixels the user marks as "turf inside the touchlines" must back-project
                  inside the template (encodes e.g. "the near touchline is NOT in this frame")
  people        : players as calibration objects — scale-free height ratios at many depths and, for
                  upright players, the direction of the body axis (vertical vanishing point)
  priors        : FIFA-template dimensions, principal point near the centre, shoulder height
Everything is minimised jointly with a soft-L1 robust loss.

Why extra frames help: a single frame of the centre circle constrains the perspective (where the
goal-line vanishing point is) only weakly, because painted circles are never perfect. Frames that
show the penalty area / goal area / goal line add *lines parallel to the goal line* and metric
distances along X, and because they share C, hl, hw with the decision frame they pin down the
decision-frame camera much better.
"""
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import least_squares

from .camera import make_camera, project, rays, vanishing_point, look_at_rvec
from .template import (line_elements, circle_elements, point_elements, kind_of, anchor,
                       DEFAULT_HALF_LENGTH, DEFAULT_HALF_WIDTH)


@dataclass
class Frame:
    name: str
    size: tuple                      # (W, H)
    features: dict                   # element -> (N,2) pixels
    group: str = 'main'              # frames with the same group share the camera centre
    inside_pitch: np.ndarray = None  # (M,2) pixels known to be turf inside the field of play
    people: list = field(default_factory=list)   # [(foot, shoulder)] or [{'foot','shoulder','upright'}]


@dataclass
class Priors:
    half_length: tuple = (DEFAULT_HALF_LENGTH, 2.5)
    half_width: tuple = (DEFAULT_HALF_WIDTH, 2.0)
    shoulder_h: tuple = (1.40, 0.25)       # free scale of the people cue; only ratios matter
    pp_sigma: float = 40.0
    pp_offset: tuple = (0.0, 0.0)


class Layout:
    """Index bookkeeping for the parameter vector."""
    def __init__(self, frames):
        self.frames = frames
        self.groups = list(dict.fromkeys(f.group for f in frames))
        self.n_global = 5                                   # hl, hw, hs, dx, dy
        self.gidx = {g: self.n_global + 3 * k for k, g in enumerate(self.groups)}
        base = self.n_global + 3 * len(self.groups)
        self.fidx = {f.name: base + 4 * k for k, f in enumerate(frames)}
        self.size = base + 4 * len(frames)

    def camera(self, p, frame):
        i, g = self.fidx[frame.name], self.gidx[frame.group]
        W, H = frame.size
        return make_camera(np.exp(p[i]), p[i + 1:i + 4], p[g:g + 3], W / 2 + p[3], H / 2 + p[4])


def _ground_safe(P, cam):
    v = rays(P, cam)
    t = -cam['C'][2] / v[:, 2]
    bad = t <= 0                                            # ray above the horizon
    G = cam['C'] + np.abs(t)[:, None] * v
    return G, bad


def _subsample(feats, rng, boot):
    out = {}
    for name, P in feats.items():
        n = {'line': 600, 'circle': 1500, 'point': len(P)}[kind_of(name)]
        if kind_of(name) == 'point' or len(P) == 0:
            out[name] = P
        else:
            out[name] = P[rng.choice(len(P), n if boot else min(n, len(P)), replace=boot)]
    return out


def _residuals(x, free, p_full, layout, data, priors, perturb):
    with np.errstate(all='ignore'):        # rays near the horizon during the multi-start search
        return _residuals_impl(x, free, p_full, layout, data, priors, perturb)


# Weighting (all residuals are in "standard deviations"):
#   * a painted element is NOT N independent pixels: its errors are correlated (painting, line width,
#     lens), so each element counts as at most N_EFF observations with SIGMA_PX pixel error;
#   * each player observation counts as one observation (see people_residuals);
#   * priors count as one observation each; the inside-pitch hinge is a near-hard constraint.
N_EFF, SIGMA_PX, HINGE_W = 30, 3.0, 200.0


def element_pixel_residuals(name, P, cam, hl, hw, perturb=None):
    """Raw image residuals (N,2) of the pixels P assigned to template element `name`."""
    k = kind_of(name)
    if k == 'point':
        x0, y0 = point_elements(hl, hw)[name]
        return project(np.array([[x0, y0, 0.0]]), cam) - P
    G, bad = _ground_safe(P, cam)
    if k == 'line':
        axis, v, _ = line_elements(hl, hw)[name]
        G[:, 0 if axis == 'X' else 1] = v
    else:
        (ccx, ccy), r = circle_elements(hl, hw)[name]
        a = np.arctan2(G[:, 1] - ccy, G[:, 0] - ccx)
        c = perturb.get('circle', (0, 0, 0, 0, 0)) if perturb else (0, 0, 0, 0, 0)
        rad = r * (1 + c[0] + c[1] * np.cos(2 * a + c[2]) + c[3] * np.cos(3 * a + c[4]))
        G = np.c_[ccx + rad * np.cos(a), ccy + rad * np.sin(a), 0 * a]
    res = project(G, cam) - P
    res[bad] = 500.0
    return res


def _residuals_impl(x, free, p_full, layout, data, priors, perturb):
    p = p_full.copy()
    p[free] = x
    hl, hw, hs = p[0], p[1], p[2]
    out = []
    for fr, d in zip(layout.frames, data):
        if not d['active']:
            continue
        cam = layout.camera(p, fr)
        for name, P in d['features'].items():
            if len(P) == 0:
                continue
            w = np.sqrt(min(1.0, N_EFF / len(P))) / SIGMA_PX
            out.append(w * element_pixel_residuals(name, P, cam, hl, hw, perturb).ravel())
        if fr.inside_pitch is not None and len(fr.inside_pitch):
            G, bad = _ground_safe(np.asarray(fr.inside_pitch, float), cam)
            viol = np.maximum(0, np.abs(G[:, 0]) - (hl - 0.3)) + np.maximum(0, np.abs(G[:, 1]) - (hw - 0.3))
            viol[bad] = 50.0
            out.append(viol * HINGE_W)
        if d.get('people'):
            out.append(people_residuals(d['people'], cam, hs))
    pr = priors
    out.append(np.array([(hl - pr.half_length[0]) / pr.half_length[1],
                         (hw - pr.half_width[0]) / pr.half_width[1],
                         (hs - pr.shoulder_h[0]) / pr.shoulder_h[1],
                         (p[3] - pr.pp_offset[0]) / pr.pp_sigma,
                         (p[4] - pr.pp_offset[1]) / pr.pp_sigma]))
    return np.concatenate(out)


def _people_arrays(people):
    """people: list of (foot, shoulder) tuples or dicts {foot, shoulder, upright}."""
    F, S, U = [], [], []
    for q in people:
        if isinstance(q, dict):
            F.append(q['foot']); S.append(q['shoulder']); U.append(bool(q.get('upright', False)))
        else:
            F.append(q[0]); S.append(q[1]); U.append(False)
    return np.array(F, float), np.array(S, float), np.array(U, bool)


def people_residuals(people, cam, hs, sig_any=0.10, sig_upright=0.06, sig_angle=np.deg2rad(4), w=1.0):
    """Players as calibration objects (Lv/Zhao/Nevatia-style, scale-free because hs is a free parameter):
      * height ratio  |foot->shoulder| observed / projected hs        (all players; tighter if upright)
      * verticality   foot->shoulder direction vs the projected world vertical at the foot (upright only):
                      upright bodies point at the vertical vanishing point, which fixes tilt and focal length.
    With players spread over many depths and frames this is what pins down the perspective of a stationary
    camera that only sees the centre-circle markings."""
    F, S, U = _people_arrays(people)
    G, _ = _ground_safe(F, cam)
    top = project(G + [0, 0, hs], cam)
    pred = top - F
    obs = S - F
    lp, lo = np.linalg.norm(pred, axis=1), np.linalg.norm(obs, axis=1)
    r_h = (lp / lo - 1) / np.where(U, sig_upright, sig_any) * w
    ang = np.arctan2(pred[:, 0] * obs[:, 1] - pred[:, 1] * obs[:, 0], (pred * obs).sum(1))
    r_v = np.where(U, ang / sig_angle * w, 0.0)
    return np.r_[r_h, r_v[U]]


def _solve(p, free, layout, data, priors, perturb, max_nfev):
    r = least_squares(_residuals, p[free], args=(free, p, layout, data, priors, perturb),
                      loss='soft_l1', f_scale=1.5, max_nfev=max_nfev)
    q = p.copy()
    q[free] = r.x
    return q, r.cost


def _frame_starts(fr, C=None, hl=DEFAULT_HALF_LENGTH, hw=DEFAULT_HALF_WIDTH):
    target = np.mean([anchor(n, hl, hw) for n in fr.features], axis=0)
    target[2] = 0
    W = fr.size[0]
    for f0 in (0.7, 1.2, 2.0, 3.2, 5.0):
        if C is not None:
            yield np.log(f0 * W), look_at_rvec(C, target), C
        else:
            for d0 in (45, 70, 100, 140):
                C0 = np.array([target[0], target[1] - d0, 0.17 * d0])
                yield np.log(f0 * W), look_at_rvec(C0, target), C0


def calibrate(frames, priors=None, rng=None, x0=None, perturb=None, boot=False, max_nfev=4000):
    """Joint calibration. frames[0] is the reference (decision) frame.
    Returns a dict with the parameter vector, the per-frame cameras and diagnostics."""
    priors = priors or Priors()
    rng = rng or np.random.default_rng(0)
    layout = Layout(frames)
    data = []
    for fr in frames:
        people = list(fr.people or [])
        if boot and people:                                  # bootstrap the people cue too
            people = [people[k] for k in rng.choice(len(people), len(people), replace=True)]
        data.append(dict(features=_subsample(fr.features, rng, boot), people=people, active=True))
    p = np.zeros(layout.size)
    if x0 is not None:
        p[:] = x0
        p, cost = _solve(p, np.arange(layout.size), layout, data, priors, perturb, max_nfev)
    else:
        p[:5] = [priors.half_length[0], priors.half_width[0], priors.shoulder_h[0], *priors.pp_offset]
        done_groups = {}
        for k, fr in enumerate(frames):
            for j, d in enumerate(data):
                d['active'] = j == k
            i, g = layout.fidx[fr.name], layout.gidx[fr.group]
            best = None
            C = done_groups.get(fr.group)
            for lf, rv, C0 in _frame_starts(fr, C, p[0], p[1]):
                q = p.copy()
                q[i], q[i + 1:i + 4], q[g:g + 3] = lf, rv, C0
                free = [i, i + 1, i + 2, i + 3] + ([] if C is not None else [g, g + 1, g + 2])
                if k == 0:
                    free += [0, 1, 2, 3, 4]
                try:
                    q, cost = _solve(q, np.array(free), layout, data, priors, perturb, 1500)
                except (ValueError, np.linalg.LinAlgError):
                    continue
                if np.isfinite(cost) and (best is None or cost < best[1]):
                    best = (q, cost)
            p = best[0]
            done_groups.setdefault(fr.group, p[g:g + 3].copy())
        for d in data:
            d['active'] = True
        p, cost = _solve(p, np.arange(layout.size), layout, data, priors, perturb, max_nfev)
    return _result(p, layout, data, priors, perturb, cost)


def _result(p, layout, data, priors, perturb, cost):
    hl, hw = p[0], p[1]
    cams, diag = {}, {}
    for fr, d in zip(layout.frames, data):
        cam = layout.camera(p, fr)
        cams[fr.name] = cam
        rms = {name: float(np.sqrt((element_pixel_residuals(name, P, cam, hl, hw, perturb) ** 2).sum(1).mean()))
               for name, P in d['features'].items() if len(P)}
        diag[fr.name] = dict(f_px=cam['f'], camera_xyz_m=cam['C'].round(2).tolist(),
                             hfov_deg=float(2 * np.degrees(np.arctan(fr.size[0] / 2 / cam['f']))),
                             goal_line_vanishing_point=vanishing_point(cam, (0, 1, 0)).round(1).tolist(),
                             rms_px=rms)
    return dict(params=p, layout=layout, cams=cams, half_length=float(hl), half_width=float(hw),
                shoulder_h=float(p[2]), principal_offset=p[3:5].tolist(), cost=float(cost), diagnostics=diag)
