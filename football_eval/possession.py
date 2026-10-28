"""Ball control per frame -> touches (control segments) -> team possession timeline.

Rules (docs/STATS_RULEBOOK.md §2):
* A player *controls* the ball in a frame if he is the closest player to the ball, within
  ``possession.control_radius_m``, and the ball is slower than ``possession.max_ball_speed_control``
  (a fast ball is in flight — it is passing by, not being played).
* *Duel*: players of both teams within ``possession.duel_radius_m`` of the ball.
* A *touch* is a maximal run of frames controlled by the same track (1-frame gaps bridged).
* *Team possession* switches only when the other team controls the ball for
  ``possession.team_switch_s`` (hysteresis), and is undefined during dead-ball periods.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def frame_control(tracks: pd.DataFrame, ball: pd.DataFrame, cfg) -> pd.DataFrame:
    pc = cfg.possession
    people = tracks[(tracks.team >= 0) & tracks.x.notna()][["frame", "track_id", "team", "cls", "x", "y"]]
    cols = ["frame", "t", "x", "y", "vx", "vy", "speed", "valid"] + [c for c in ("cam_ok", "vis_near_y") if c in ball]
    bl = ball[cols].rename(columns={"x": "bx", "y": "by"})
    m = people.merge(bl[bl.valid], on="frame", how="inner")
    m["d"] = np.hypot(m.x - m.bx, m.y - m.by)
    m = m.sort_values(["frame", "d"])
    near = m.groupby("frame").head(1).set_index("frame")
    # duel: closest player of each team
    per_team = m.groupby(["frame", "team"]).d.min().unstack()
    out = bl.set_index("frame").copy()
    out["ctrl_tid"] = -1
    out["ctrl_team"] = -1
    out["ctrl_d"] = np.nan
    ok = (near.d <= pc.control_radius_m) & (near.speed <= pc.max_ball_speed_control)
    idx = near.index[ok]
    out.loc[idx, "ctrl_tid"] = near.loc[idx, "track_id"].astype(int)
    out.loc[idx, "ctrl_team"] = near.loc[idx, "team"].astype(int)
    out.loc[near.index, "ctrl_d"] = near.d
    duel = pd.Series(False, index=out.index)
    if 0 in per_team and 1 in per_team:
        d = (per_team[0] <= pc.duel_radius_m) & (per_team[1] <= pc.duel_radius_m)
        duel.loc[d.index] = d
    out["duel"] = duel
    return out.reset_index()


def touches(fc: pd.DataFrame, tracks: pd.DataFrame, min_control_s: float = 0.1, bridge: int = 1) -> pd.DataFrame:
    """Control segments. Columns: tid, team, cls, t0, t1, f0, f1, x0, y0, x1, y1 (ball positions)."""
    c = fc[["frame", "t", "bx", "by", "ctrl_tid", "ctrl_team"]].reset_index(drop=True)
    tid = c.ctrl_tid.to_numpy().copy()
    # bridge short gaps (-1) between identical ids
    n = len(tid)
    i = 0
    while i < n:
        if tid[i] == -1:
            j = i
            while j < n and tid[j] == -1:
                j += 1
            if 0 < i and j < n and j - i <= bridge and tid[i - 1] == tid[j]:
                tid[i:j] = tid[j]
            i = j
        else:
            i += 1
    c["tid"] = tid
    seg = (c.tid != c.tid.shift()).cumsum()
    rows = []
    cls_of = tracks.groupby("track_id").cls.agg(lambda s: s.mode().iloc[0]).to_dict()
    team_of = tracks.groupby("track_id").team.agg(lambda s: s.mode().iloc[0]).to_dict()
    dt = float(np.median(np.diff(c.t))) if len(c) > 1 else 0.1
    for _, g in c.groupby(seg):
        k = int(g.tid.iloc[0])
        if k < 0:
            continue
        if len(g) * dt < min_control_s - 1e-9:      # a frame counts for one sampling interval
            continue
        rows.append(dict(tid=k, team=int(team_of.get(k, -1)), cls=cls_of.get(k, "player"),
                         t0=g.t.iloc[0], t1=g.t.iloc[-1], f0=int(g.frame.iloc[0]), f1=int(g.frame.iloc[-1]),
                         x0=g.bx.iloc[0], y0=g.by.iloc[0], x1=g.bx.iloc[-1], y1=g.by.iloc[-1], n=len(g)))
    return pd.DataFrame(rows, columns=["tid", "team", "cls", "t0", "t1", "f0", "f1", "x0", "y0", "x1", "y1", "n"])


def team_possession(fc: pd.DataFrame, dead: np.ndarray, switch_s: float = 0.6) -> np.ndarray:
    """Per-frame team in possession (-1 = none / dead ball) with hysteresis."""
    t = fc.t.to_numpy()
    ct = fc.ctrl_team.to_numpy()
    out = np.full(len(fc), -1, int)
    cur, cand, cand_t0 = -1, -1, None
    for i in range(len(fc)):
        if dead[i]:
            cur, cand, cand_t0 = -1, -1, None
            continue
        if ct[i] >= 0 and ct[i] != cur:
            if cur == -1:
                cur = ct[i]
            elif cand != ct[i]:
                cand, cand_t0 = ct[i], t[i]
            elif t[i] - cand_t0 >= switch_s - 1e-9:
                # switch retroactively from the candidate's first control
                j = i
                while j > 0 and t[j - 1] >= cand_t0:
                    j -= 1
                out[j:i] = cand
                cur, cand, cand_t0 = cand, -1, None
        elif ct[i] == cur:
            cand, cand_t0 = -1, None
        out[i] = cur
    return out
