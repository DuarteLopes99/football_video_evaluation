"""Who is a detected person? Players / goalkeepers / referee vs **spectators, staff (coaches, bench, ball boys) and
stewards (security)**.

The person detector (COCO "person") fires on everybody in the frame: on a lower-league recording the crowd on the
terrace, the coaches in the technical area, ball boys and stewards are often a third of the boxes. Counted as players
they pollute the team assignment (a fan in a yellow shirt "is" Canedo), the minimap and every positional statistic.

Rule (per detection, then per track):

=============  ==============================================================================================
zone           where the feet are
=============  ==============================================================================================
``pitch``      inside the lines + ``offpitch_margin_m`` (throw-ins, goalkeepers behind the line)
``touchline``  up to ``staff_band_m`` beyond that margin: benches, technical area, ball boys, stewards
``stands``     farther away (terraces, behind the fences)
=============  ==============================================================================================

With a camera calibration the zone comes from the pitch coordinates of the feet. Without one (or where it is
unusable) it comes from the **grass mask** of the frame (:func:`grass_top`): feet on the grass = pitch; feet above the
far edge of the grass by less than ~half a body height = touchline, more = stands.

Person type: ``pitch`` -> kit classifier (player / goalkeeper / referee / unknown); ``touchline`` -> ``staff``, or
``steward`` when the torso is fluorescent hi-vis; ``stands`` -> ``spectator`` (``steward`` when hi-vis).
Per track, the type seen in most of its frames wins (:func:`track_types`).
"""
from __future__ import annotations

from typing import Optional

import cv2
import numpy as np
import pandas as pd

ZONES = ("pitch", "touchline", "stands")
OFF_PITCH_TYPES = ("staff", "spectator", "steward")
TYPE_LABEL = {"staff": "STAFF", "spectator": "FAN", "steward": "SEC"}
TYPE_DESCRIPTION = {
    "player": "outfield player (kit of one of the two teams)",
    "goalkeeper": "goalkeeper (kit clicked in the setup, or found in the penalty area)",
    "referee": "referee / assistant referee",
    "staff": "off the pitch, next to the touchline: coaches, substitutes, bench, ball boys",
    "spectator": "off the pitch, in the stands / behind the fences: fans",
    "steward": "off the pitch in a fluorescent hi-vis vest: security / stewards",
    "unknown": "on the pitch but not matching any clicked kit (occluded, blurred, or a person not in the setup)",
}


def grass_top(img: np.ndarray, scale: int = 4) -> np.ndarray:
    """Row of the far edge of the grass for every column of ``img`` (video pixels; ``h`` = no grass in that column).
    Computed on a 1/``scale`` image: ~3 ms per 1080p frame."""
    from .ptz import field_mask
    h, w = img.shape[:2]
    small = cv2.resize(img, (max(8, w // scale), max(8, h // scale)), interpolation=cv2.INTER_AREA)
    f = field_mask(small)
    top = np.where(f.any(0), np.argmax(f > 0, axis=0), f.shape[0]).astype(float) * (h / f.shape[0])
    top = cv2.medianBlur(top.astype(np.float32).reshape(1, -1), 5).ravel()
    return np.interp(np.arange(w), np.linspace(0, w - 1, len(top)), top)


def grass_roi(top: np.ndarray, shape, head_room: float = 0.18, max_crop: float = 0.45):
    """Detector region without calibration: from just above the highest point of the grass (+ head room for people
    standing on the far touchline) to the bottom. None when the grass is not visible."""
    h, w = shape[:2]
    ok = top < h - 2
    if ok.mean() < 0.3:
        return None
    t = int(np.clip(np.quantile(top[ok], 0.05) - head_room * h, 0, max_crop * h))
    return (0, t, w, h) if t > 8 else None


def hivis_share(img: np.ndarray, box) -> float:
    """Share of fluorescent yellow / orange / lime pixels in the torso (hi-vis vest)."""
    from .teams import torso_crop
    c = torso_crop(img, box)
    if c is None or c.size == 0:
        return 0.0
    hsv = cv2.cvtColor(c, cv2.COLOR_BGR2HSV)
    hue, s, v = hsv[..., 0].astype(int), hsv[..., 1].astype(int), hsv[..., 2].astype(int)
    fluo = (((hue >= 5) & (hue <= 42)) | ((hue > 42) & (hue <= 55) & (s > 200))) & (s > 150) & (v > 170)
    return float(fluo.mean())


def zones(xy: np.ndarray, foot_uv: np.ndarray, boxes: np.ndarray, pitch, top: Optional[np.ndarray],
          margin_m: float = 2.5, staff_band_m: float = 9.0) -> list:
    """Zone of every person: from the pitch coordinates when available, else from the grass mask."""
    out = []
    for k in range(len(foot_uv)):
        x, y = xy[k] if len(xy) else (np.nan, np.nan)
        if np.isfinite(x) and np.isfinite(y):
            dx = max(-x, x - pitch.length, 0.0)
            dy = max(-y, y - pitch.width, 0.0)
            d = float(np.hypot(dx, dy))
            out.append("pitch" if d <= margin_m else "touchline" if d <= margin_m + staff_band_m else "stands")
            continue
        if top is None:
            out.append("pitch")
            continue
        u, v = foot_uv[k]
        col = int(np.clip(u, 0, len(top) - 1))
        gap = top[col] - v                                   # > 0: feet above the far edge of the grass
        bh = max(boxes[k][3] - boxes[k][1], 1.0)
        out.append("pitch" if gap <= 0.08 * bh else "touchline" if gap <= 0.6 * bh else "stands")
    return out


def frame_types(img: np.ndarray, boxes: np.ndarray, zone: list, hivis_min: float = 0.30) -> list:
    """Per detection: None (on the pitch: decided by the kits) or staff / spectator / steward."""
    out = []
    for k, z in enumerate(zone):
        if z == "pitch":
            out.append(None)
            continue
        hv = hivis_share(img, boxes[k]) >= hivis_min
        out.append("steward" if hv else ("staff" if z == "touchline" else "spectator"))
    return out


def track_types(tracks: pd.DataFrame, min_share: float = 0.6) -> pd.Series:
    """Per track: the off-pitch type seen in most of its rows (>= ``min_share`` of them off the pitch), else NaN."""
    if "ptype" not in tracks or not len(tracks):
        return pd.Series(dtype=object)
    t = tracks[tracks.track_id >= 0]
    out = {}
    for tid, g in t.groupby("track_id").ptype:
        off = g.dropna()
        if len(off) >= min_share * len(g) and len(off):
            vc = off.value_counts()
            # a hi-vis vest seen in >= 30 % of the off-pitch rows is a steward (the vest is not always visible)
            out[tid] = "steward" if vc.get("steward", 0) >= 0.3 * len(off) else vc.index[0]
    return pd.Series(out, dtype=object)


def people_summary(tracks: pd.DataFrame) -> pd.DataFrame:
    """Detections and distinct tracks per person type (for the report and the console)."""
    if tracks is None or not len(tracks):
        return pd.DataFrame(columns=["type", "detections", "tracks", "per_frame", "description"])
    n_fr = max(tracks.frame.nunique(), 1)
    rows = []
    team = tracks.team if "team" in tracks else pd.Series(0, index=tracks.index)
    for c in ("player", "goalkeeper", "referee", "unknown", "staff", "steward", "spectator", "outside"):
        if c == "player":
            q = tracks[(tracks.cls == "player") & (team >= 0)]
        elif c == "unknown":
            q = tracks[(tracks.cls == "player") & (team < 0)]
        else:
            q = tracks[tracks.cls == c]
        if not len(q) and c in ("outside", "unknown"):
            continue
        rows.append(dict(type=c, detections=int(len(q)), tracks=int(q.loc[q.track_id >= 0, "track_id"].nunique()),
                         per_frame=round(len(q) / n_fr, 2),
                         description=TYPE_DESCRIPTION.get(c, "outside the pitch (older runs)")))
    return pd.DataFrame(rows)
