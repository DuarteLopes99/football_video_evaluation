"""Team and player (tracklet) statistics from events + tracking. Definitions: rulebook §4."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .context import MatchContext


def _count(ev, team, typ, outcome=None, detail=None):
    m = (ev.type == typ) & (ev.team == team)
    if outcome is not None:
        m &= ev.outcome.isin([outcome] if isinstance(outcome, str) else outcome)
    if detail is not None:
        m &= ev.detail.fillna("").str.contains(detail)
    return int(m.sum())


def team_stats(ev: pd.DataFrame, fc: pd.DataFrame, poss: np.ndarray, dead: np.ndarray, tracks_sp: pd.DataFrame,
               ctx: MatchContext, cfg) -> pd.DataFrame:
    ec, pitch = cfg.events, ctx.pitch
    L = pitch.length
    dt = float(np.median(np.diff(fc.t))) if len(fc) > 1 else 0.1
    in_play = ~dead
    rows = {}
    if "cam_ok" in fc:
        in_play = in_play & fc.cam_ok.fillna(True).astype(bool).to_numpy()
    poss_s = {tm: float(((poss == tm) & in_play).sum() * dt) for tm in (0, 1)}
    tot = sum(poss_s.values()) or 1.0
    bx, by, t = fc.bx.to_numpy(), fc.by.to_numpy(), fc.t.to_numpy()
    for tm in (0, 1):
        r = {}
        r["possession_%"] = 100 * poss_s[tm] / tot
        r["ball_in_play_min"] = in_play.sum() * dt / 60
        # field tilt: share of final-third possession time
        ft = {}
        for k in (0, 1):
            m = (poss == k) & in_play & ~np.isnan(bx)
            xs = np.array([ctx.to_attack_frame(k, tt, xx, yy)[0] for tt, xx, yy in zip(t[m], bx[m], by[m])])
            ft[k] = float((xs >= ec.final_third_x).sum()) if len(xs) else 0.0
        r["field_tilt_%"] = 100 * ft[tm] / max(ft[0] + ft[1], 1e-9)
        p_all = _count(ev, tm, "pass")
        p_ok = _count(ev, tm, "pass", "complete")
        r["passes"] = p_all
        r["passes_completed"] = p_ok
        r["pass_accuracy_%"] = 100 * p_ok / p_all if p_all else np.nan
        r["progressive_passes"] = _count(ev, tm, "pass", "complete", "progressive")
        r["final_third_entries_pass"] = _count(ev, tm, "pass", "complete", "final_third_entry")
        r["box_entries_pass"] = _count(ev, tm, "pass", "complete", "box_entry")
        r["crosses"] = _count(ev, tm, "pass", None, "cross")
        r["long_passes"] = _count(ev, tm, "pass", None, "long")
        sh = ev[(ev.type == "shot") & (ev.team == tm)]
        r["shots"] = len(sh)
        r["shots_on_target"] = int(sh.outcome.isin(["goal", "saved"]).sum())
        r["shots_blocked"] = int(sh.outcome.isin(["blocked"]).sum())
        r["xG"] = float(sh.xg.sum()) if len(sh) else 0.0
        r["xG_per_shot"] = float(sh.xg.mean()) if len(sh) else np.nan
        r["big_chances(xG>=0.3)"] = int((sh.xg >= 0.3).sum())
        r["goals"] = int(((ev.type == "goal") & (ev.team == tm)).sum())
        if "source" in ev and (ev.source == "match_sheet").any():     # only known from the match sheet
            r["yellow_cards"] = _count(ev, tm, "yellow")
            r["red_cards"] = _count(ev, tm, "red")
        r["corners"] = _count(ev, tm, "corner")
        r["throw_ins"] = _count(ev, tm, "throw_in")
        r["goal_kicks"] = _count(ev, tm, "goal_kick")
        r["free_kicks_won"] = _count(ev, tm, "free_kick")
        r["penalties_won"] = _count(ev, tm, "penalty")
        r["fouls_committed(proxy)"] = _count(ev, tm, "foul_proxy")
        r["offsides(suspected)"] = _count(ev, tm, "offside_suspected")
        r["interceptions"] = _count(ev, tm, "interception")
        r["ball_recoveries"] = _count(ev, tm, "ball_recovery")
        # PPDA: opponent passes in their own 60 % / our defensive actions in that zone
        opp = 1 - tm
        op = ev[(ev.type == "pass") & (ev.team == opp)]
        opx = np.array([ctx.to_attack_frame(opp, a, b, c)[0] for a, b, c in zip(op.t, op.x, op.y)])
        n_opp = int((opx < L - ec.ppda_zone_x).sum()) if len(opx) else 0
        da = ev[ev.type.isin(["interception", "ball_recovery"]) & (ev.team == tm)]
        dax = np.array([ctx.to_attack_frame(tm, a, b, c)[0] for a, b, c in zip(da.t, da.x, da.y)])
        fo = ev[(ev.type == "foul_proxy") & (ev.team == tm)]
        fox = np.array([ctx.to_attack_frame(tm, a, b, c)[0] for a, b, c in zip(fo.t, fo.x, fo.y)])
        n_def = int((dax >= ec.ppda_zone_x).sum() if len(dax) else 0) + int((fox >= ec.ppda_zone_x).sum() if len(fox) else 0)
        r["PPDA"] = n_opp / n_def if n_def else np.nan
        # physical / shape (tracklets of this team, outfield)
        q = tracks_sp[(tracks_sp.team == tm) & (tracks_sp.cls != "goalkeeper")]
        dist = (q.speed * q.groupby("track_id").t.diff()).sum()
        r["distance_covered_km(tracked)"] = float(dist / 1000)
        r["sprints(>7m/s,>=1s)"] = _sprints(q)
        r["avg_speed_kmh"] = float(q.speed.mean() * 3.6) if q.speed.notna().any() else np.nan
        sh_ = _shape(q, ctx, tm)
        r.update(sh_)
        rows[ctx.team_names[tm] if tm < len(ctx.team_names) else f"Team {tm}"] = r
    return pd.DataFrame(rows)


def _sprints(q: pd.DataFrame, thr=7.0, min_s=1.0) -> int:
    n = 0
    for _, g in q.groupby("track_id"):
        fast = (g.speed > thr).to_numpy()
        tt = g.t.to_numpy()
        m = np.r_[False, fast, False].astype(int)
        d = np.diff(m)
        for a, b in zip(np.where(d == 1)[0], np.where(d == -1)[0]):
            if tt[b - 1] - tt[a] >= min_s:
                n += 1
    return n


def _shape(q: pd.DataFrame, ctx: MatchContext, tm: int) -> dict:
    """Mean team block metrics per frame (attack frame): depth, width, compactness, defensive line height."""
    if len(q) == 0:
        return {}
    xs, ys = [], []
    for (f, tt), g in q[q.x.notna()].groupby(["frame", "t"]):
        if len(g) < 6:
            continue
        x, y = ctx.to_attack_frame(tm, tt, g.x.to_numpy(), g.y.to_numpy())
        xs.append(x)
        ys.append(y)
    if not xs:
        return {}
    depth = np.mean([np.percentile(x, 90) - np.percentile(x, 10) for x in xs])
    width = np.mean([np.percentile(y, 90) - np.percentile(y, 10) for y in ys])
    line = np.mean([np.percentile(x, 15) for x in xs])
    centroid = np.mean([np.mean(x) for x in xs])
    return {"block_depth_m": float(depth), "block_width_m": float(width),
            "def_line_height_m": float(line), "team_centroid_x_m": float(centroid)}


def player_stats(ev: pd.DataFrame, tracks_sp: pd.DataFrame, tch: pd.DataFrame, min_minutes: float = 2.0) -> pd.DataFrame:
    """Per tracklet (track_id). Without jersey OCR / re-ID a real player = several tracklets."""
    q = tracks_sp[tracks_sp.team >= 0]
    if q.empty:                       # e.g. a very short test run before the teams could be separated
        return pd.DataFrame(columns=["team", "role", "first_t", "last_t", "minutes_tracked", "distance_m",
                                     "max_speed_kmh", "avg_x", "avg_y", "touches", "passes", "shots", "xG", "goals"])
    g = q.groupby("track_id")
    base = pd.DataFrame({
        "team": g.team.agg(lambda s: s.mode().iloc[0]),
        "role": g.cls.agg(lambda s: s.mode().iloc[0]),
        "first_t": g.t.min(), "last_t": g.t.max(),
        "minutes_tracked": (g.t.max() - g.t.min()) / 60,
        "distance_m": g.apply(lambda d: float((d.speed * d.t.diff()).sum()), include_groups=False),
        "max_speed_kmh": g.speed.max() * 3.6,
        "avg_x": g.x.mean(), "avg_y": g.y.mean(),
    })
    base = base[base.minutes_tracked >= min_minutes]
    tc = tch.groupby("tid").size().rename("touches")
    e = ev[ev.tid >= 0]
    add = pd.DataFrame({
        "passes": e[e.type == "pass"].groupby("tid").size(),
        "passes_completed": e[(e.type == "pass") & (e.outcome == "complete")].groupby("tid").size(),
        "shots": e[e.type == "shot"].groupby("tid").size(),
        "xG": e[e.type == "shot"].groupby("tid").xg.sum(),
        "goals": e[e.type == "goal"].groupby("tid").size(),
        "interceptions": e[e.type == "interception"].groupby("tid").size(),
        "recoveries": e[e.type == "ball_recovery"].groupby("tid").size(),
    })
    out = base.join(tc).join(add).fillna({c: 0 for c in ["touches", "passes", "passes_completed", "shots", "xG", "goals",
                                                           "interceptions", "recoveries"]})
    out.index.name = "track_id"
    return out.sort_values(["team", "minutes_tracked"], ascending=[True, False])


def possession_timeline(fc: pd.DataFrame, poss: np.ndarray, dead: np.ndarray, window_s: float = 300.0) -> pd.DataFrame:
    """Rolling share of possession of team 0 (dead-ball frames excluded)."""
    d = pd.DataFrame({"t": fc.t.to_numpy(), "p0": (poss == 0).astype(float), "p1": (poss == 1).astype(float)})
    d.loc[dead | (poss < 0), ["p0", "p1"]] = np.nan
    d["bin"] = (d.t // 60).astype(int)
    m = d.groupby("bin")[["p0", "p1"]].sum()
    r = m.rolling(max(1, int(window_s // 60)), min_periods=1).sum()
    share = r.p0 / (r.p0 + r.p1)
    return pd.DataFrame({"minute": m.index, "team0_share": share.to_numpy()})


def numbered_player_stats(ev: pd.DataFrame, ctx: MatchContext) -> pd.DataFrame:
    """Per shirt number, for the events a person attributed in the review or on the match sheet.

    Shirt numbers are not readable by the system at club-stream resolution (players are ~25 px tall, tracklets
    last ~2 s), so this table only covers the events the user labelled — typically goals, shots and cards.
    """
    if "number" not in ev or ev.number.notna().sum() == 0:
        return pd.DataFrame(columns=["team", "number", "player", "goals", "shots", "xG", "yellow", "red"])
    q = ev[ev.number.notna() & ev.team.isin([0, 1])].copy()
    q["number"] = q.number.astype(int)
    rows = []
    for (tm, num), g in q.groupby(["team", "number"]):
        name = g.player.dropna().iloc[0] if "player" in g and g.player.notna().any() else None
        sh = g[g.type == "shot"]
        rows.append(dict(team=ctx.team_names[int(tm)] if int(tm) < len(ctx.team_names) else tm, number=int(num),
                         player=name, goals=int((g.type == "goal").sum()), shots=len(sh),
                         xG=round(float(sh.xg.sum()), 3) if len(sh) else 0.0,
                         yellow=int((g.type == "yellow").sum()), red=int((g.type == "red").sum())))
    return pd.DataFrame(rows).sort_values(["team", "number"])
