"""Rule-based event detection from the pitch-coordinate tracking table.

Adapted from Vidal-Codina et al. (2022) "Automatic event detection in football using tracking
data" (possession / duel zones, set-piece patterns) to the noisier single-camera setting.
Every rule and threshold is documented in docs/STATS_RULEBOOK.md (§3 events). Config keys are
under ``cfg.events``.

Output: a DataFrame of events with columns
    t, frame, period, type, team, tid, x, y, x_end, y_end, outcome, xg, detail, source
``team`` is the team that performed the event (for ``foul_proxy``: the team that committed it).
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd

from .context import MatchContext
from .xg import XGModel, feature_row, freeze_frame_features

EVENT_COLS = ["t", "frame", "period", "type", "team", "tid", "x", "y", "x_end", "y_end", "outcome", "xg",
              "detail", "source"]


# ----------------------------------------------------------------------------- dead ball
def _runs(mask: np.ndarray):
    """Start/end indices (inclusive) of True runs."""
    m = np.r_[False, mask, False].astype(int)
    d = np.diff(m)
    return list(zip(np.where(d == 1)[0], np.where(d == -1)[0] - 1))


def dead_ball(fc: pd.DataFrame, tracks_sp: pd.DataFrame, ctx: MatchContext, cfg) -> tuple[np.ndarray, pd.DataFrame]:
    """Per-frame dead-ball mask + stoppage table (start/end time, reason, exit location)."""
    ec = cfg.events
    pitch = ctx.pitch
    t = fc.t.to_numpy()
    valid = fc.valid.to_numpy(bool)
    bx, by = fc.bx.to_numpy(), fc.by.to_numpy()
    out = valid & ~pitch.in_pitch(np.nan_to_num(bx, nan=-1e3), np.nan_to_num(by, nan=-1e3), ec.out_margin_m)
    out &= valid
    mean_sp = tracks_sp.groupby("frame").speed.mean().reindex(fc.frame).to_numpy()
    mean_sp = pd.Series(mean_sp).rolling(9, center=True, min_periods=1).mean().to_numpy()
    still = np.nan_to_num(mean_sp, nan=0.0) < ec.still_speed_ms
    dt = float(np.median(np.diff(t))) if len(t) > 1 else 0.1
    cand = out.copy()
    cam_ok = fc.cam_ok.to_numpy(bool) if "cam_ok" in fc else np.ones(len(fc), bool)
    # unseen runs: dead if players are still OR the ball left the pitch just before
    for a, b in _runs(~valid):
        if cam_ok[a:b + 1].mean() < 0.5:          # camera calibration lost, not a stoppage
            continue
        dur = (b - a + 1) * dt
        after_out = a > 0 and out[a - 1]
        if dur >= ec.dead_ball_min_s and (still[a:b + 1].mean() > 0.6 or after_out):
            cand[a:b + 1] = True
        elif after_out:
            cand[a:b + 1] = True
    # static ball + still players (foul, injury, free kick preparation)
    # static ball: positional spread over a ~2 s window (robust to detection jitter, unlike speed)
    win = max(3, int(round(2.0 / (float(np.median(np.diff(t))) if len(t) > 1 else 0.1))))
    # median-filter first (1 s) so detection jitter does not hide a ball at rest
    half = max(3, win // 2)
    mx = pd.Series(bx).rolling(half, center=True, min_periods=2).median()
    my = pd.Series(by).rolling(half, center=True, min_periods=2).median()
    spread = np.sqrt(mx.rolling(win, center=True, min_periods=win // 2).var() +
                     my.rolling(win, center=True, min_periods=win // 2).var()).to_numpy()
    static = valid & (np.nan_to_num(spread, nan=99.0) < ec.static_ball_spread_m) & still
    static_any = valid & (np.nan_to_num(spread, nan=99.0) < ec.static_ball_spread_m)
    for a, b in _runs(static):
        if (b - a + 1) * dt >= ec.static_still_min_s:
            cand[a:b + 1] = True
    for a, b in _runs(static_any):             # a ball at rest this long is a set piece, even if players move
        if (b - a + 1) * dt >= ec.static_ball_min_s:
            cand[a:b + 1] = True
    # merge small gaps and drop short candidates (e.g. a lofted ball projected outside the lines)
    gap = int(round(1.5 / dt))
    runs = _runs(cand)
    merged = []
    for a, b in runs:
        if merged and a - merged[-1][1] - 1 <= gap:
            merged[-1] = (merged[-1][0], b)
        else:
            merged.append((a, b))
    dead = np.zeros(len(fc), bool)
    rows = []
    # where the camera's view ends on the near (camera) side, per frame (>= W: the near touchline is visible)
    near_y = fc.vis_near_y.to_numpy(float) if "vis_near_y" in fc else np.full(len(fc), np.nan)
    for a, b in merged:
        dur = (b - a + 1) * dt
        if dur < ec.dead_ball_min_s:
            continue
        dead[a:b + 1] = True
        # exit location: last valid in-pitch ball position before the stoppage, or first out position
        oi = np.where(out[a:b + 1])[0]
        reason = "out" if len(oi) else ("unseen" if (~valid[a:b + 1]).mean() > 0.5 else "static")
        if len(oi):
            k = a + oi[0]
            ex, ey = bx[k], by[k]
        else:
            prev = np.where(valid[:a])[0]
            ex, ey = (bx[prev[-1]], by[prev[-1]]) if len(prev) else (np.nan, np.nan)
        side = None
        ny = near_y[max(0, a - 1)]
        if reason in ("unseen", "static") and np.isfinite(ny) and ny < pitch.width - 0.5 and np.isfinite(ey) \
                and ey >= ny - ec.near_edge_band_m and (~valid[a:b + 1]).mean() > 0.5:
            # the ball left the picture at the bottom while the near touchline is out of view: it probably went out
            # there (throw-in on the camera side) — "unseen" would otherwise look like a foul / injury stoppage
            reason, side = "out_of_view_near", "touchline_near"
        if reason == "out":
            if ex < 0 or ex > pitch.length:
                side = "goal_line_right" if ex > pitch.length else "goal_line_left"
            else:
                side = "touchline"
        rows.append(dict(i0=a, i1=b, t0=t[a], t1=t[b], dur=dur, reason=reason, exit_x=ex, exit_y=ey, side=side))
    return dead, pd.DataFrame(rows, columns=["i0", "i1", "t0", "t1", "dur", "reason", "exit_x", "exit_y", "side"])


# ----------------------------------------------------------------------------- helpers
def _positions_at(tracks: pd.DataFrame, frame: int) -> pd.DataFrame:
    fr = tracks.frame.to_numpy()
    if len(fr) == 0:
        return tracks.iloc[:0]
    uniq = np.unique(fr)
    f = uniq[np.argmin(np.abs(uniq - frame))]
    return tracks[(tracks.frame == f) & tracks.x.notna()]


def _ball_after(fc: pd.DataFrame, t0: float, horizon: float = 0.6):
    w = fc[(fc.t > t0) & (fc.t <= t0 + horizon) & fc.valid]
    if len(w) == 0:
        return None
    k = w.speed.to_numpy().argmax()
    return w.iloc[k]


class EventDetector:
    def __init__(self, cfg, ctx: MatchContext, xg_model: Optional[XGModel] = None):
        self.cfg, self.ctx, self.pitch = cfg, ctx, ctx.pitch
        self.ec = cfg.events
        self.xg = xg_model
        self.events: List[dict] = []

    def _add(self, **kw):
        e = {c: kw.get(c, np.nan) for c in EVENT_COLS}
        e["period"] = self.ctx.period_of(e["t"])
        e["source"] = kw.get("source", "rules")
        self.events.append(e)

    # ---- set pieces ------------------------------------------------------------------
    def classify_restart(self, x, y, t, players: pd.DataFrame, taker_team: int, side: Optional[str] = None,
                         near_edge_y: float = float("nan")) -> str:
        """Set-piece type from where the ball is put back into play (+ which line it left by, if known).

        ``near_edge_y``: pitch y where the camera's view ends on the near side. When the near touchline is out of
        view, a ball that left the picture there and comes back near the bottom edge is a throw-in."""
        p, ec = self.pitch, self.ec
        L, W = p.length, p.width
        if side == "touchline_near" and np.isfinite(near_edge_y) and y >= near_edge_y - ec.near_edge_band_m - 1.0:
            return "throw_in"
        if np.hypot(x - L / 2, y - W / 2) <= ec.kickoff_radius_m:
            if len(players):
                own = []
                for tm in (0, 1):
                    q = players[players.team == tm]
                    if len(q):
                        right = self.ctx.attacking_right(tm, t)
                        m_ = ec.kickoff_half_margin_m
                        own.append(((q.x < L / 2 + m_) if right else (q.x > L / 2 - m_)).mean())
                if own and min(own) >= ec.kickoff_own_half_share:
                    return "kickoff"
            else:
                return "kickoff"
        near_touch = y <= ec.throwin_band_m or y >= W - ec.throwin_band_m
        if side == "touchline" and near_touch:
            return "throw_in"
        for cx, cy in ((0, 0), (0, W), (L, 0), (L, W)):
            if np.hypot(x - cx, y - cy) <= ec.corner_radius_m and side != "touchline":
                return "corner"
        for sx in (p.penalty_spot, L - p.penalty_spot):
            if np.hypot(x - sx, y - W / 2) <= ec.penalty_radius_m:
                # at a penalty only the taker and the goalkeeper are inside the area
                if len(players):
                    inside = p.in_penalty_area(players.x.to_numpy(), players.y.to_numpy(), sx > L / 2)
                    near_ball = np.hypot(players.x.to_numpy() - x, players.y.to_numpy() - y) < 2.0
                    on_line = (np.abs(players.x.to_numpy() - (L if sx > L / 2 else 0)) < 2.5)
                    if (inside & ~near_ball & ~on_line).sum() <= 1:
                        return "penalty"
                else:
                    return "penalty"
        if (x <= p.goal_area_depth + 1 or x >= L - p.goal_area_depth - 1) and abs(y - W / 2) <= p.goal_area_width / 2 + 1:
            return "goal_kick"
        if y <= ec.throwin_band_m or y >= W - ec.throwin_band_m:
            return "throw_in"
        return "free_kick"

    # ---- main ------------------------------------------------------------------------
    def run(self, fc: pd.DataFrame, tch: pd.DataFrame, tracks: pd.DataFrame, stoppages: pd.DataFrame,
            scoreboard_changes: list | None = None) -> pd.DataFrame:
        ec, ctx, pitch = self.ec, self.ctx, self.pitch
        L, W = pitch.length, pitch.width
        tch = tch.sort_values("t0").reset_index(drop=True)
        st = stoppages.sort_values("t0").reset_index(drop=True)

        def stoppage_between(ta, tb):
            s = st[(st.t0 >= ta - 0.05) & (st.t0 <= tb)]
            return s.iloc[0] if len(s) else None

        # ---------------- restarts (after each stoppage) and period kick-offs
        restarts = []
        fct, fcf = fc.t.to_numpy(), fc.frame.to_numpy()
        fbx, fby, fval = fc.bx.to_numpy(), fc.by.to_numpy(), fc.valid.to_numpy(bool)
        fnear = fc.vis_near_y.to_numpy(float) if "vis_near_y" in fc else None
        dtf = float(np.median(np.diff(fct))) if len(fct) > 1 else 0.1
        for _, s in st.iterrows():
            j = int(s.i1) + 1                                   # first live frame
            if j >= len(fc):
                continue
            # where the ball was placed: last valid positions of the stoppage, else first ones after it
            w = int(round(1.0 / dtf))
            idx = [k for k in range(max(int(s.i0), j - w), j) if fval[k]]
            if not idx:
                idx = [k for k in range(j, min(len(fc), j + w)) if fval[k]][:3]
            if not idx:
                continue
            x, y = float(np.median(fbx[idx])), float(np.median(fby[idx]))
            if not pitch.in_pitch(x, y, 1.0):
                continue
            # taken = first frame the ball leaves the spot (within 10 s), else the first live frame
            k_take = j
            for k in range(j, min(len(fc), j + int(round(10.0 / dtf)))):
                if fval[k] and np.hypot(fbx[k] - x, fby[k] - y) > 1.5:
                    k_take = max(j, k - 1)
                    break
            shape = _positions_at(tracks, int(fcf[j]))          # formation check at the end of the stoppage
            shape = shape[shape.team >= 0]
            j = k_take
            players = _positions_at(tracks, int(fcf[j]))
            players = players[players.team >= 0]
            taker_team, taker_tid = -1, -1
            tk = tch[(tch.t0 >= fct[j] - 2.0) & (tch.t0 <= fct[j] + 0.5) &
                     (np.hypot(tch.x0 - x, tch.y0 - y) <= 2.5) & (tch.team >= 0)]
            if len(tk):                                         # the player who actually played it
                taker_team, taker_tid = int(tk.iloc[-1].team), int(tk.iloc[-1].tid)
            elif len(players):
                d = np.hypot(players.x.to_numpy() - x, players.y.to_numpy() - y)
                if d.min() <= 3.0:
                    taker_team, taker_tid = int(players.team.iloc[d.argmin()]), int(players.track_id.iloc[d.argmin()])
            if taker_team < 0:
                nxt = tch[tch.t0 >= s.t1 - 0.5]
                if len(nxt) == 0 or nxt.iloc[0].t0 - s.t1 > 6.0:
                    continue
                taker_team, taker_tid = int(nxt.iloc[0].team), int(nxt.iloc[0].tid)
            kind = self.classify_restart(x, y, fct[j], shape, taker_team, s.side if isinstance(s.side, str) else None,
                                         fnear[j] if fnear is not None else float("nan"))
            restarts.append(dict(t=float(fct[j]), frame=int(fcf[j]), kind=kind, team=taker_team, tid=taker_tid,
                                 x=x, y=y, stop_t0=s.t0, stop_reason=s.reason, side=s.side))
        for a, b in ctx.periods:
            first = tch[(tch.t0 >= a) & (tch.t0 <= b)]
            if len(first):
                k = first.iloc[0]
                if np.hypot(k.x0 - L / 2, k.y0 - W / 2) <= 2 * ec.kickoff_radius_m and \
                        not any(abs(r["t"] - k.t1) < 1 for r in restarts):
                    restarts.append(dict(t=k.t1, frame=int(k.f1), kind="kickoff", team=int(k.team), tid=int(k.tid),
                                         x=k.x0, y=k.y0, stop_t0=a, stop_reason="period_start", side=None))
        restarts.sort(key=lambda r: r["t"])

        # ---------------- touches -> passes, shots, interceptions, recoveries
        shots = []
        last_pass = None
        for i in range(len(tch)):
            a = tch.iloc[i]
            if a.team < 0:
                continue
            b = tch.iloc[i + 1] if i + 1 < len(tch) else None
            stop = stoppage_between(a.t1, b.t0 if b is not None else a.t1 + ec.max_pass_s)
            rel = _ball_after(fc, a.t1)
            ax, ay = ctx.to_attack_frame(a.team, a.t1, a.x1, a.y1)
            # ---- shot?
            is_shot = False
            if rel is not None and rel.speed >= ec.shot_min_speed:
                sgn = 1.0 if ctx.attacking_right(a.team, a.t1) else -1.0   # velocities only flip sign
                vx, vy = sgn * rel.vx, sgn * rel.vy
                dist_goal = np.hypot(L - ax, W / 2 - ay)
                if vx > 0 and dist_goal <= ec.shot_max_dist:
                    y_cross = ay + vy / vx * (L - ax)
                    if abs(y_cross - W / 2) <= pitch.goal_width / 2 + ec.shot_goal_tolerance:
                        is_shot = True
                        # a ball received by a teammate far from goal was a pass (cross / through ball)
                        if b is not None and b.team == a.team and b.tid != a.tid and stop is None and b.t0 - a.t1 < 2.5:
                            bx_, _ = ctx.to_attack_frame(a.team, b.t0, b.x0, b.y0)
                            if bx_ < L - 6:
                                is_shot = False
            if is_shot:
                shots.append(self._shot(a, b, stop, fc, tracks, restarts))
                shots[-1]["detail"] += f"|speed={rel.speed:.1f}"     # kept for the 'why' explanation
                continue
            if b is None:
                continue
            if b.tid == a.tid and stop is None:
                continue                                    # same player: carry continues
            gap = b.t0 - a.t1
            speed_out = rel.speed if rel is not None else 0.0
            if stop is not None:
                if stop.reason == "out" and speed_out > 3.0:
                    self._add(t=a.t1, frame=a.f1, type="pass", team=a.team, tid=a.tid, x=a.x1, y=a.y1,
                              x_end=stop.exit_x, y_end=stop.exit_y, outcome="out", detail="")
                continue
            if gap > ec.max_pass_s:
                continue
            travel = np.hypot(b.x0 - a.x1, b.y0 - a.y1)
            if travel < ec.min_pass_length_m:           # contact flicker at one spot = contest, not a pass
                if b.team != a.team and b.team >= 0 and b.t1 - b.t0 >= ec.recovery_min_s:
                    self._add(t=b.t0, frame=b.f0, type="ball_recovery", team=b.team, tid=b.tid, x=b.x0, y=b.y0,
                              detail="tackle/dispossession")
                continue
            if b.team == a.team:
                d = self._pass_detail(a, b)
                self._add(t=a.t1, frame=a.f1, type="pass", team=a.team, tid=a.tid, x=a.x1, y=a.y1,
                          x_end=b.x0, y_end=b.y0, outcome="complete", detail=d)
                last_pass = (a, b, "complete")
            elif b.team >= 0:
                if speed_out > 4.0:
                    self._add(t=a.t1, frame=a.f1, type="pass", team=a.team, tid=a.tid, x=a.x1, y=a.y1,
                              x_end=b.x0, y_end=b.y0, outcome="intercepted", detail=self._pass_detail(a, b))
                    self._add(t=b.t0, frame=b.f0, type="interception", team=b.team, tid=b.tid, x=b.x0, y=b.y0)
                    last_pass = (a, b, "intercepted")
                elif b.t1 - b.t0 >= ec.recovery_min_s:   # a brief contact in a duel is not a recovery
                    self._add(t=b.t0, frame=b.f0, type="ball_recovery", team=b.team, tid=b.tid, x=b.x0, y=b.y0,
                              detail="tackle/dispossession")

        # ---------------- set-piece events, fouls, offsides, goals from kick-offs
        goal_times = [s["t"] for s in shots if s["outcome"] == "goal"]
        for r in restarts:
            kind = r["kind"]
            if kind == "kickoff":
                self._add(t=r["t"], frame=r["frame"], type="kickoff", team=r["team"], tid=r["tid"], x=r["x"], y=r["y"],
                          detail=r["stop_reason"])
                pa = ctx.periods[ctx.period_of(r["t"])][0]
                n_before = int(((tch.t0 >= pa) & (tch.t0 < r["stop_t0"])).sum())
                if n_before < 3:                    # nothing played yet in this period -> period start
                    self.events[-1]["detail"] = "period_start"
                    continue
                if r["stop_reason"] != "period_start":
                    scorer = 1 - r["team"]
                    near = [g for g in goal_times if r["stop_t0"] - 20 <= g <= r["t"]]
                    stop_len = r["t"] - r["stop_t0"]
                    if not near and ec.goal_min_stoppage_s <= stop_len <= ec.goal_confirm_window_s:
                        self._add(t=r["stop_t0"], frame=r["frame"], type="goal", team=scorer, tid=-1,
                                  x=np.nan, y=np.nan, outcome="inferred_from_kickoff", source="kickoff_rule")
                continue
            if kind in ("free_kick", "penalty") and r["stop_reason"] != "out":
                tx, _ = ctx.to_attack_frame(r["team"], r["t"], r["x"], r["y"])
                offside = False
                if kind == "free_kick" and tx < L / 2 and last_pass is not None:
                    pa, pb, _ = last_pass
                    if pa.team != r["team"] and 0 < r["stop_t0"] - pb.t0 < 8 + (pb.t1 - pb.t0):
                        ox, _ = ctx.to_attack_frame(pa.team, pb.t0, pb.x0, pb.y0)
                        ox0, _ = ctx.to_attack_frame(pa.team, pa.t1, pa.x1, pa.y1)
                        offside = ox > ox0 and ox >= L / 2
                if offside:
                    self._add(t=r["stop_t0"], frame=r["frame"], type="offside_suspected", team=1 - r["team"], tid=-1,
                              x=r["x"], y=r["y"])
                else:
                    self._add(t=r["stop_t0"], frame=r["frame"], type="foul_proxy", team=1 - r["team"], tid=-1,
                              x=r["x"], y=r["y"], detail=kind)
            self._add(t=r["t"], frame=r["frame"], type=kind, team=r["team"], tid=r["tid"], x=r["x"], y=r["y"],
                      detail=r["stop_reason"] or "")

        for s in shots:
            self._add(**s)
            if s["outcome"] == "goal":
                self._add(t=s["t"], frame=s["frame"], type="goal", team=s["team"], tid=s["tid"], x=s["x"], y=s["y"],
                          xg=s["xg"], outcome="from_shot", source="shot_rule")

        # ---------------- scoreboard changes (broadcast overlay) -> confirm / recover goals
        for ch in scoreboard_changes or []:
            tc = ch["t"]
            have = [e for e in self.events if e["type"] == "goal" and tc - 180 <= e["t"] <= tc + 15]
            if have:
                for e in have:
                    e["detail"] = (str(e["detail"]) if isinstance(e["detail"], str) else "") + "|scoreboard_confirmed"
            else:
                self._add(t=tc, frame=-1, type="goal", team=-1, tid=-1, outcome="scoreboard_only", source="scoreboard")

        ev = pd.DataFrame(self.events, columns=EVENT_COLS).sort_values("t").reset_index(drop=True)
        return ev

    # ---- details ---------------------------------------------------------------------
    def _pass_detail(self, a, b) -> str:
        L, W = self.pitch.length, self.pitch.width
        x0, y0 = self.ctx.to_attack_frame(a.team, a.t1, a.x1, a.y1)
        x1, y1 = self.ctx.to_attack_frame(a.team, b.t0, b.x0, b.y0)
        tags = []
        if x1 - x0 >= self.ec.progressive_pass_m:
            tags.append("progressive")
        if x0 < self.ec.final_third_x <= x1:
            tags.append("final_third_entry")
        if self.pitch.in_penalty_area(x1, y1, True) and not self.pitch.in_penalty_area(x0, y0, True):
            tags.append("box_entry")
            if x0 >= L - 35 and abs(y0 - W / 2) > self.pitch.penalty_area_width / 2:
                tags.append("cross")
        if np.hypot(x1 - x0, y1 - y0) >= 30:
            tags.append("long")
        return "|".join(tags)

    def _shot(self, a, b, stop, fc, tracks, restarts) -> dict:
        ec, ctx, pitch = self.ec, self.ctx, self.pitch
        L, W = pitch.length, pitch.width
        t_end = b.t0 if b is not None else a.t1 + 3.0
        if stop is not None:
            t_end = min(t_end, stop.t0 + 0.5)
        traj = fc[(fc.t > a.t1) & (fc.t <= t_end) & fc.valid]
        tx, ty = ctx.to_attack_frame(a.team, a.t1, traj.bx.to_numpy(), traj.by.to_numpy())
        crossed = np.where(tx >= L)[0]
        outcome = "unresolved"
        if len(crossed):
            yc = ty[crossed[0]]
            outcome = "goal" if abs(yc - W / 2) <= pitch.goal_width / 2 + 0.5 else "off_target"
        if b is not None and (stop is None) and b.team != a.team and b.team >= 0:
            bx_, by_ = ctx.to_attack_frame(a.team, b.t0, b.x0, b.y0)
            is_gk = b.cls == "goalkeeper" or (bx_ > L - pitch.goal_area_depth - 1.5 and abs(by_ - W / 2) < 10)
            if is_gk:
                outcome = "saved"
            elif b.t0 - a.t1 <= ec.block_window_s and np.hypot(b.x0 - a.x1, b.y0 - a.y1) <= ec.block_max_dist:
                outcome = "blocked"
            elif outcome == "unresolved":
                outcome = "off_target"
        if stop is not None:
            nxt = [r for r in restarts if r["t"] >= stop.t1 - 0.5]
            if nxt:
                r = nxt[0]
                if r["kind"] == "kickoff" and r["team"] != a.team:
                    outcome = "goal"
                elif outcome in ("unresolved", "goal") and r["kind"] in ("goal_kick", "corner"):
                    outcome = "off_target" if r["kind"] == "goal_kick" else "blocked_or_saved_out"
        # xG with freeze frame
        sx, sy = ctx.to_attack_frame(a.team, a.t1, a.x1, a.y1)
        pos = _positions_at(tracks, int(a.f1))
        opp = pos[pos.team == 1 - a.team]
        ox, oy = ctx.to_attack_frame(a.team, a.t1, opp.x.to_numpy(), opp.y.to_numpy())
        gk = opp[opp.cls == "goalkeeper"]
        if len(gk):
            gx, gy = ctx.to_attack_frame(a.team, a.t1, gk.x.to_numpy()[:1], gk.y.to_numpy()[:1])
            gk_xy = (gx[0], gy[0])
            mask = (opp.cls != "goalkeeper").to_numpy()
        else:                                   # GK not identified: nearest opponent to goal centre in the box
            d = np.hypot(L - ox, W / 2 - oy)
            j = int(np.argmin(d)) if len(d) else -1
            gk_xy = (ox[j], oy[j]) if j >= 0 and d[j] < 12 else None
            mask = np.ones(len(ox), bool)
            if j >= 0 and gk_xy is not None:
                mask[j] = False
        ff = freeze_frame_features((sx, sy), np.c_[ox[mask], oy[mask]], gk_xy, L, W)
        row = feature_row(sx, sy, ff, L, W)
        sit = "open_play"
        for r in restarts:
            if r["tid"] == a.tid and abs(a.t1 - r["t"]) <= 0.3 and r["kind"] in ("penalty", "free_kick"):
                sit = "penalty" if r["kind"] == "penalty" else "direct_free_kick"
        xg = float(self.xg.predict([row], self.cfg.xg.variant, [sit])[0]) if self.xg else np.nan
        return dict(t=a.t1, frame=a.f1, type="shot", team=a.team, tid=a.tid, x=a.x1, y=a.y1,
                    x_end=traj.bx.iloc[-1] if len(traj) else np.nan, y_end=traj.by.iloc[-1] if len(traj) else np.nan,
                    outcome=outcome, xg=xg, detail=f"{sit}|dist={row['dist']:.1f}|def_tri={ff['n_def_tri']}")
