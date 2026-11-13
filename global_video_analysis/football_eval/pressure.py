"""Attacking phases near goal — offensive play detected even when the ball is not seen.

From one club camera the ball is found in only ~10-25 % of the frames, so ball-based events (passes, shots) are
rare. But the operator keeps the play in the picture, and the players' positions are known on ~90 % of the frames.
An **attack near goal** is a stretch where the play is close to one goal for a few seconds:

per analysed frame (camera calibration usable only)
    play position  = the ball when it is seen (inside the pitch), else the median of the outfield players in view
                     (>= ``min_players_in_view``) — the camera follows the ball, the players crowd around it;
    near goal      = play position within ``zone_m`` of a goal line (left or right goal);
    box crowding   = players of either team inside that penalty area;
    frame vote     = near goal AND (ball seen there OR >= ``min_box_players`` players in the box)

phases: votes smoothed over 1 s, runs of the same goal merged across gaps <= ``merge_gap_s``, kept when they last
>= ``min_duration_s`` and the attacking team (the team attacking that goal in that period, from the setup's direction)
has on average >= ``min_attackers`` players within ``zone_m`` of that goal. Each phase becomes an
``attack_near_goal`` event (start time, duration, evidence) — in the timeline, the replay, and an opportunity
("Attack near goal") when no stronger component (shot, set piece, box entry) of the same team is next to it.

Thresholds: ``config.PressureCfg``; rulebook §3.8.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from .context import MatchContext


def frame_play(tracks: pd.DataFrame, ball: Optional[pd.DataFrame], ctx: MatchContext, cfg) -> pd.DataFrame:
    """Per analysed frame: play position, its source, players in each penalty area, attackers near each goal."""
    pc = cfg.pressure
    p = ctx.pitch
    L, W = p.length, p.width
    pa_d, pa_w = p.penalty_area_depth, p.penalty_area_width / 2
    tr = tracks[tracks.cls.isin(["player", "goalkeeper"]) & tracks.team.isin([0, 1]) & tracks.x.notna()]
    if not len(tr):
        return pd.DataFrame(columns=["frame", "t", "px", "py", "src", "n_view", "box_l", "box_r"])
    out = tr[tr.cls == "player"].groupby("frame").agg(t=("t", "first"), mx=("x", "median"), my=("y", "median"),
                                                     n_view=("x", "size"))
    inside_l = (tr.x <= pa_d) & ((tr.y - W / 2).abs() <= pa_w)
    inside_r = (tr.x >= L - pa_d) & ((tr.y - W / 2).abs() <= pa_w)
    out["box_l"] = inside_l.groupby(tr.frame).sum()
    out["box_r"] = inside_r.groupby(tr.frame).sum()
    near_l = tr.x <= pc.zone_m
    near_r = tr.x >= L - pc.zone_m
    for tm in (0, 1):
        out[f"near_l_{tm}"] = (near_l & (tr.team == tm)).groupby(tr.frame).sum()
        out[f"near_r_{tm}"] = (near_r & (tr.team == tm)).groupby(tr.frame).sum()
    out = out.fillna(0)
    out["px"], out["py"], out["src"] = out.mx, out.my, "players"
    out.loc[out.n_view < pc.min_players_in_view, ["px", "py"]] = np.nan
    out.loc[out.n_view < pc.min_players_in_view, "src"] = ""
    if ball is not None and len(ball) and "x" in ball:
        b = ball.drop_duplicates("frame").set_index("frame")
        ok = b.get("visible", b.get("valid", pd.Series(True, index=b.index))).astype(bool) & b.x.notna() & \
            (b.x > -3) & (b.x < L + 3) & (b.y > -3) & (b.y < W + 3)
        bx = b.x.where(ok).reindex(out.index)
        by = b.y.where(ok).reindex(out.index)
        has = bx.notna()
        out.loc[has, "px"], out.loc[has, "py"], out.loc[has, "src"] = bx[has], by[has], "ball"
    return out.reset_index()


def attack_phases(tracks: pd.DataFrame, ball: Optional[pd.DataFrame], frames: Optional[pd.DataFrame],
                  ctx: MatchContext, cfg) -> pd.DataFrame:
    """``attack_near_goal`` events (columns as in events.csv + t_end, duration, ball_share, box_players)."""
    pc = cfg.pressure
    cols = ["t", "type", "team", "x", "y", "t_end", "duration", "side", "ball_share", "box_players", "attackers",
            "detail", "outcome", "source"]
    fp = frame_play(tracks, ball, ctx, cfg)
    if not len(fp):
        return pd.DataFrame(columns=cols)
    if frames is not None and "cam_ok" in frames:
        ok = frames.drop_duplicates("frame").set_index("frame").cam_ok.astype(bool)
        fp = fp[fp.frame.map(ok).fillna(False).astype(bool)]
    fp = fp.sort_values("t").reset_index(drop=True)
    if not len(fp):
        return pd.DataFrame(columns=cols)
    L = ctx.pitch.length
    dt = float(np.median(np.diff(fp.t))) if len(fp) > 1 else 0.1
    win = max(1, int(round(1.0 / max(dt, 1e-3))))
    out = []
    for side in ("l", "r"):
        near = (fp.px <= pc.zone_m) if side == "l" else (fp.px >= L - pc.zone_m)
        vote = near & ((fp.src == "ball") | (fp[f"box_{side}"] >= pc.min_box_players))
        sm = vote.astype(float).rolling(win, center=True, min_periods=1).mean() >= 0.5
        idx = np.flatnonzero(sm.to_numpy())
        if not len(idx):
            continue
        runs, start, prev = [], idx[0], idx[0]
        for i in idx[1:]:
            if fp.t.iloc[i] - fp.t.iloc[prev] > pc.merge_gap_s or ctx.period_of(fp.t.iloc[i]) != ctx.period_of(fp.t.iloc[prev]):
                runs.append((start, prev))
                start = i
            prev = i
        runs.append((start, prev))
        for a, b in runs:
            t0, t1 = float(fp.t.iloc[a]), float(fp.t.iloc[b])
            if t1 - t0 < pc.min_duration_s:
                continue
            seg = fp.iloc[a:b + 1]
            # the team attacking this goal in this period: attacking right <=> goal on the right
            tm = next((k for k in (0, 1) if ctx.attacking_right(k, (t0 + t1) / 2) == (side == "r")), None)
            if tm is None:
                continue
            att = float(seg[f"near_{side}_{tm}"].mean())
            if att < pc.min_attackers:
                continue
            ball_share = float((seg.src == "ball").mean())
            box = float(seg[f"box_{side}"].mean())
            out.append(dict(t=round(t0, 2), type="attack_near_goal", team=tm, x=round(float(seg.px.median()), 1),
                            y=round(float(seg.py.median()), 1), t_end=round(t1, 2), duration=round(t1 - t0, 1),
                            side="right" if side == "r" else "left", ball_share=round(ball_share, 2),
                            box_players=round(box, 1), attackers=round(att, 1),
                            detail=f"dur={t1 - t0:.1f}s|ball={100 * ball_share:.0f}%|box={box:.1f}|att={att:.1f}",
                            outcome="", source="auto:pressure"))
    if not out:
        return pd.DataFrame(columns=cols)
    return pd.DataFrame(out).sort_values("t").reset_index(drop=True)[cols]
