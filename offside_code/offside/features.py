"""Pitch-marking extraction: which image pixels belong to which template element.

Two ways to label pixels:
  1. hints  — a rough polyline (2+ clicked points) or a bounding box per element; pixels of the
              white-line mask inside a band around the hint are kept and cleaned with RANSAC.
  2. candidates — automatic sequential-RANSAC lines and ellipses, drawn with ids; you then map
              ids to element names (e.g. {'halfway_line': 'L0', 'centre_circle': 'E0'}).
A learned keypoint/line detector (PnLCalib / NBJW weights) can replace this module: it only has to
return {element_name: (N,2) pixel array}.
"""
import cv2
import numpy as np

from .template import kind_of


# ---------------------------------------------------------------- white-line mask
def line_response(img):
    """Local-contrast response of thin bright structures, robust to cast shadows."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    L = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)[..., 0].astype(np.float32)
    grass = ((hsv[..., 0] > 25) & (hsv[..., 0] < 80) & (hsv[..., 1] > 25)).astype(np.uint8)
    grass = cv2.morphologyEx(grass, cv2.MORPH_CLOSE, np.ones((31, 31), np.uint8)) > 0
    bg = cv2.morphologyEx(L, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21)))
    return (L - bg) / (bg + 5), hsv, grass


def line_mask(img, ignore_regions=(), loose=False):
    ratio, hsv, grass = line_response(img)
    if loose:   # thin far-away lines (e.g. the far touchline) — no grass gating
        m = (ratio > 0.10) & (hsv[..., 1] < 120)
    else:
        m = (ratio > 0.18) & (hsv[..., 1] < 110) & grass
    for x0, y0, x1, y1 in ignore_regions:
        m[int(y0):int(y1), int(x0):int(x1)] = False
    return m


# ---------------------------------------------------------------- robust fitting helpers
def ransac_line(pts, rng, thr=2.0, iters=3000, widen=True):
    """RANSAC line, then (widen=True) re-select ALL pixels of the painted stripe so that the kept set is
    symmetric about the stripe centre-line (a thick line otherwise gives an edge-biased fit)."""
    if len(pts) < 2:
        return pts
    best, bc = None, -1
    for _ in range(iters):
        a, b = pts[rng.choice(len(pts), 2, replace=False)]
        n = np.array([a[1] - b[1], b[0] - a[0]], float)
        nn = np.linalg.norm(n)
        if nn < 1e-6:
            continue
        n /= nn
        inl = np.abs(pts @ n - n @ a) < thr
        if inl.sum() > bc:
            bc, best = inl.sum(), inl
    if not widen:
        return pts[best]
    sel = best
    for _ in range(3):                                   # re-centre on the whole painted stripe
        P = pts[sel]
        m = P.mean(0)
        n = np.linalg.svd(P - m, full_matrices=False)[2][1]
        sel = _stripe(( pts - m) @ n)
    return pts[sel]


def _stripe(d, max_half=25):
    """Pixels forming the contiguous stripe around the densest offset (1-px histogram of signed distances)."""
    near = np.abs(d) < max_half
    if near.sum() < 10:
        return near
    edges = np.arange(-max_half, max_half + 1)
    h, _ = np.histogram(d[near], edges)
    k = int(np.argmax(h))
    lo = hi = k
    thr = 0.25 * h[k]
    while lo > 0 and h[lo - 1] >= thr:
        lo -= 1
    while hi < len(h) - 1 and h[hi + 1] >= thr:
        hi += 1
    return (d >= edges[lo]) & (d < edges[hi + 1])


def _ell_dist(e, P):
    (cx, cy), (a, b), ang = e
    t = np.deg2rad(ang)
    c, s = np.cos(t), np.sin(t)
    x = (P[:, 0] - cx) * c + (P[:, 1] - cy) * s
    y = -(P[:, 0] - cx) * s + (P[:, 1] - cy) * c
    return np.abs(np.sqrt((x / (a / 2)) ** 2 + (y / (b / 2)) ** 2) - 1) * min(a, b) / 2


def ransac_ellipse(pts, rng, iters=4000, thr=3.0, min_axis=50, bbox=None):
    """RANSAC ellipse. With `bbox` (the user's box around the circle) candidates must roughly fill it:
    centre in the middle 40 % of the box, major axis 60-110 % of the box width."""
    P = pts.astype(np.float32)
    best, bc = None, -1
    for _ in range(iters):
        try:
            e = cv2.fitEllipse(P[rng.choice(len(P), 6, replace=False)])
        except cv2.error:
            continue
        if not np.all(np.isfinite(e[1])) or min(e[1]) < min_axis * 0.2 or max(e[1]) < min_axis:
            continue
        if bbox is not None:
            x0, y0, x1, y1 = bbox
            (cx, cy), ax = e[0], max(e[1])
            if not (x0 + 0.3 * (x1 - x0) < cx < x1 - 0.3 * (x1 - x0) and y0 + 0.3 * (y1 - y0) < cy < y1 - 0.3 * (y1 - y0)
                    and 0.6 * (x1 - x0) < ax < 1.1 * (x1 - x0)):
                continue
        inl = _ell_dist(e, P) < thr
        if inl.sum() > bc:
            bc, best = inl.sum(), inl
    if best is None:
        return pts[:0]
    for k in range(6):
        if best.sum() < 6:
            return pts[:0]
        e = cv2.fitEllipseDirect(P[best])
        if not np.all(np.isfinite(e[1])) or min(e[1]) <= 0:
            return pts[:0]
        # first refine on the RANSAC band, then keep the whole painted ring, centred
        best = _ell_dist(e, P) < thr * 0.85 if k < 3 else _stripe(_signed_ell(e, P))
    return pts[best]


def _signed_ell(e, P):
    (cx, cy), (a, b), ang = e
    t = np.deg2rad(ang)
    c, s = np.cos(t), np.sin(t)
    x = (P[:, 0] - cx) * c + (P[:, 1] - cy) * s
    y = -(P[:, 0] - cx) * s + (P[:, 1] - cy) * c
    return (np.sqrt((x / (a / 2)) ** 2 + (y / (b / 2)) ** 2) - 1) * min(a, b) / 2


def _band(mask, polyline, band):
    poly = np.asarray(polyline, np.int32).reshape(-1, 1, 2)
    m = np.zeros(mask.shape, np.uint8)
    cv2.polylines(m, [poly], False, 1, thickness=int(2 * band))
    ys, xs = np.nonzero(mask & (m > 0))
    return np.c_[xs, ys].astype(float)


# ---------------------------------------------------------------- 1. hint-based extraction
def extract_features(img, specs, ignore_regions=(), rng=None):
    """specs: list of dicts, one per element, e.g.
         {'element': 'halfway_line', 'polyline': [[1710,225],[1715,1570]], 'band': 30}
         {'element': 'touchline_far', 'polyline': [[400,223],[2929,216]], 'band': 12, 'loose': True}
         {'element': 'centre_circle', 'bbox': [900,380,2550,720]}
         {'element': 'penalty_arc_right', 'polyline': [[..],[..],[..]], 'band': 15}   # arcs: several clicks
         {'element': 'penalty_spot_right', 'pixel': [x, y]}
       Returns {element: (N,2) pixels}."""
    rng = rng or np.random.default_rng(0)
    masks = {False: line_mask(img, ignore_regions), True: None}
    feats = {}
    line_specs = [s for s in specs if 'polyline' in s and kind_of(s['element']) == 'line']
    for s in specs:
        name = s['element']
        kind = kind_of(name)
        if kind == 'point':
            feats[name] = np.array([s['pixel']], float)
            continue
        loose = bool(s.get('loose', False))
        if masks[loose] is None:
            masks[loose] = line_mask(img, ignore_regions, loose=True)
        m = masks[loose]
        if kind == 'line':
            P = _band(m, s['polyline'], s.get('band', 25))
            feats[name] = ransac_line(P, rng, thr=s.get('thr', 2.0)) if s.get('ransac', True) else P
        else:
            if 'bbox' in s:
                x0, y0, x1, y1 = s['bbox']
                mm = m.copy()
                outside = np.ones(m.shape, bool)
                outside[int(y0):int(y1), int(x0):int(x1)] = False
                mm[outside] = False
                for ls in line_specs:                       # drop pixels of straight lines crossing the circle
                    poly = np.asarray(ls['polyline'], np.int32).reshape(-1, 1, 2)
                    k = np.zeros(m.shape, np.uint8)
                    cv2.polylines(k, [poly], False, 1, thickness=24)
                    mm &= k == 0
                ys, xs = np.nonzero(mm)
                P = np.c_[xs, ys].astype(float)
                if len(P) > 30000:
                    P = P[rng.choice(len(P), 30000, replace=False)]
                feats[name] = ransac_ellipse(P, rng, iters=6000, min_axis=0.3 * (x1 - x0), bbox=s['bbox'])
            else:
                feats[name] = _band(m, s['polyline'], s.get('band', 15))
    return feats


# ---------------------------------------------------------------- 2. automatic candidates
def find_candidates(img, ignore_regions=(), max_lines=10, min_inliers=600, rng=None):
    """Sequential RANSAC on the line mask. Returns {'L0': pixels, ..., 'E0': pixels, ...}."""
    rng = rng or np.random.default_rng(0)
    m = line_mask(img, ignore_regions)
    # drop speckle (grass texture in shadow): keep elongated components only
    n_cc, lab, stats, _ = cv2.connectedComponentsWithStats(m.astype(np.uint8), connectivity=8)
    good = (stats[:, cv2.CC_STAT_AREA] >= 120) & (np.maximum(stats[:, cv2.CC_STAT_WIDTH], stats[:, cv2.CC_STAT_HEIGHT]) >= 40)
    good[0] = False
    m = good[lab]
    ys, xs = np.nonzero(m)
    P = np.c_[xs, ys].astype(float)
    if len(P) > 60000:
        P = P[rng.choice(len(P), 60000, replace=False)]
    out = {}
    remaining = P
    n_lines = 0
    for _ in range(3 * max_lines):
        if n_lines >= max_lines or len(remaining) < min_inliers:
            break
        inl = ransac_line(remaining, rng, thr=3.0, iters=2000, widen=True)
        # keep only the longest dense run along the line (players / noise give scattered inliers)
        c0 = inl.mean(0)
        u = np.linalg.svd(inl - c0, full_matrices=False)[2][0]
        n = np.array([-u[1], u[0]])
        t = np.sort((inl - c0) @ u)
        runs = np.split(np.arange(len(t)), np.where(np.diff(t) > 150)[0] + 1)
        run = max(runs, key=len)
        lo, hi = t[run[0]], t[run[-1]]
        if len(run) >= min_inliers and hi - lo >= 250 and len(run) / (hi - lo) > 0.5:
            keep = (((inl - c0) @ u) >= lo) & (((inl - c0) @ u) <= hi)
            out[f'L{n_lines}'] = inl[keep]
            n_lines += 1
        # remove this line's pixels from the pool either way and keep searching
        along = (remaining - c0) @ u
        remaining = remaining[~((np.abs((remaining - c0) @ n) < 8) & (along >= t[0] - 5) & (along <= t[-1] + 5))]
    # ellipse candidates: RANSAC over the union of large leftover components (a circle is cut into
    # pieces where straight lines were removed, so components are pooled, not fitted one by one)
    for k in range(2):
        if len(remaining) < min_inliers:
            break
        mm = np.zeros(m.shape, np.uint8)
        mm[remaining[:, 1].astype(int), remaining[:, 0].astype(int)] = 1
        _, lab, stats, _ = cv2.connectedComponentsWithStats(cv2.dilate(mm, np.ones((9, 9), np.uint8)))
        big = np.where(stats[:, cv2.CC_STAT_WIDTH] > 150)[0]
        big = big[big > 0]
        sel = np.isin(lab[remaining[:, 1].astype(int), remaining[:, 0].astype(int)], big)
        pool = remaining[sel]
        if len(pool) < min_inliers:
            break
        if len(pool) > 20000:
            pool = pool[rng.choice(len(pool), 20000, replace=False)]
        E = ransac_ellipse(pool, rng, iters=3000, min_axis=300)
        if len(E) < min_inliers:
            break
        out[f'E{k}'] = E
        e = cv2.fitEllipseDirect(E.astype(np.float32))
        remaining = remaining[np.abs(_signed_ell(e, remaining.astype(np.float32))) > 6]
    return out


def draw_candidates(img, cands):
    vis = img.copy()
    rng = np.random.default_rng(3)
    for cid, P in cands.items():
        col = tuple(int(c) for c in rng.integers(80, 255, 3))
        for x, y in P[:: max(1, len(P) // 4000)].astype(int):
            cv2.circle(vis, (x, y), 2, col, -1)
        x, y = P.mean(0).astype(int)
        cv2.putText(vis, cid, (x + 8, y - 8), cv2.FONT_HERSHEY_DUPLEX, 1.6, (0, 0, 0), 6, cv2.LINE_AA)
        cv2.putText(vis, cid, (x + 8, y - 8), cv2.FONT_HERSHEY_DUPLEX, 1.6, col, 2, cv2.LINE_AA)
    return vis


def features_from_candidates(cands, mapping):
    """mapping: {'halfway_line': 'L0', 'centre_circle': 'E0', 'touchline_far': ['L1', 'L4']}"""
    feats = {}
    for name, ids in mapping.items():
        kind_of(name)
        ids = [ids] if isinstance(ids, str) else ids
        feats[name] = np.concatenate([cands[i] for i in ids])
    return feats


def draw_features(img, feats):
    vis = img.copy()
    pal = [(0, 0, 255), (255, 0, 0), (0, 255, 255), (255, 0, 255), (0, 255, 0), (255, 255, 0), (0, 128, 255)]
    for k, (name, P) in enumerate(feats.items()):
        col = pal[k % len(pal)]
        for x, y in P.astype(int):
            cv2.circle(vis, (x, y), 2 if len(P) > 1 else 10, col, -1)
        x, y = P.mean(0).astype(int)
        cv2.putText(vis, name, (x + 10, y - 10), cv2.FONT_HERSHEY_DUPLEX, 1.0, col, 2, cv2.LINE_AA)
    return vis
