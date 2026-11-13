"""Ball trajectory cleaning on the pitch plane.

Single-camera caveat: the homography maps the *ground* plane. An airborne ball is projected to
where its line of sight hits the grass, i.e. too far from the camera. Speed-based gating and
the 'in flight' flag below limit the damage; true 3-D ball reconstruction from one view needs
physics fitting (parabolic arcs) — listed as future work in docs/SOTA_REVIEW.md.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter


def clean_ball(ball: pd.DataFrame, frames: np.ndarray, times: np.ndarray, max_speed: float = 40.0,
               max_gap_s: float = 1.0, smooth_window: int = 5) -> pd.DataFrame:
    """``ball``: rows (frame, t, conf, x, y) possibly several per frame (multiple candidates).

    Returns one row per analysed frame with columns
    frame, t, x, y, vx, vy, speed, visible (detected), interp (filled), valid (x/y usable).
    """
    out = pd.DataFrame({"frame": frames, "t": times})
    b = ball.dropna(subset=["x", "y"]).sort_values(["frame", "conf"], ascending=[True, False])
    # pick, per frame, the candidate closest to the previous accepted position (else highest conf)
    acc = {}
    last = None
    for f, g in b.groupby("frame", sort=True):
        t = g.t.iloc[0]
        cand = g[["x", "y"]].to_numpy()
        if last is not None:
            dt = max(t - last[0], 1e-3)
            d = np.linalg.norm(cand - last[1], axis=1)
            ok = d / dt <= max_speed
            if not ok.any():
                if t - last[0] > max_gap_s:      # long gap: accept a fresh start
                    k = 0
                else:
                    continue
            else:
                k = int(np.argmin(np.where(ok, d - 5 * g.conf.to_numpy(), np.inf)))
        else:
            k = 0
        acc[f] = cand[k]
        last = (t, cand[k])
    xy = np.array([acc.get(f, (np.nan, np.nan)) for f in frames], float)
    out["x"], out["y"] = xy[:, 0], xy[:, 1]
    out["visible"] = ~np.isnan(xy[:, 0])
    # interpolate short gaps only
    fps = 1.0 / np.median(np.diff(times)) if len(times) > 1 else 10.0
    lim = max(1, int(max_gap_s * fps))
    for c in ("x", "y"):
        s = out[c]
        filled = s.interpolate(limit=lim, limit_area="inside")
        # do not bridge gaps longer than lim (interpolate(limit) fills the first `lim` of a long gap)
        gap_id = s.notna().cumsum()
        gap_len = s.isna().groupby(gap_id).transform("sum")
        filled[(s.isna()) & (gap_len > lim)] = np.nan
        out[c] = filled
    out["interp"] = out.x.notna() & ~out.visible
    out["valid"] = out.x.notna()
    # smooth each contiguous valid run
    sx, sy = out.x.to_numpy().copy(), out.y.to_numpy().copy()
    run = (out.valid != out.valid.shift()).cumsum()
    for _, idx in out[out.valid].groupby(run).groups.items():
        idx = np.asarray(idx)
        w = min(smooth_window, len(idx) - (1 - len(idx) % 2))
        if w >= 3:
            w = w if w % 2 else w - 1
            sx[idx] = savgol_filter(sx[idx], w, 2)
            sy[idx] = savgol_filter(sy[idx], w, 2)
    out["x"], out["y"] = sx, sy
    t = out.t.to_numpy()
    vx = np.gradient(sx, t) if len(t) > 1 else np.zeros_like(sx)
    vy = np.gradient(sy, t) if len(t) > 1 else np.zeros_like(sy)
    out["vx"], out["vy"] = vx, vy
    out["speed"] = np.hypot(vx, vy)
    return out
