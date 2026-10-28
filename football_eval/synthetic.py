"""Synthetic match generator with ground-truth events.

Purpose: develop and *measure* the analytics rules (possession, events, xG, stats) independently
of the perception models. It produces exactly the tables perception produces (``tracks`` and the
raw ball detections), with configurable noise: position jitter, ball dropouts / occlusions,
track-id fragmentation and unlabelled goalkeepers — the typical single-camera failure modes.

The match engine is deliberately simple (formation block that shifts with the ball, passing
chains, shots with sampled outcomes, set pieces, fouls, throw-ins, goals and kick-offs), but every
event it generates is logged as ground truth in the same schema as the detector output.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pandas as pd

from .ball import clean_ball
from .pitch import Pitch

FORMATION = np.array([(4, 34), (22, 10), (20, 26), (20, 42), (22, 58), (42, 12), (40, 28), (40, 40), (42, 56),
                      (60, 27), (61, 41)], float)


@dataclass
class SynthConfig:
    seed: int = 0
    fps: float = 10.0
    period_s: float = 900.0          # 15-minute halves keep tests fast; use 2700 for a full match
    halftime_s: float = 60.0
    noise_player_m: float = 0.25
    noise_ball_m: float = 0.30
    ball_dropout: float = 0.06       # random missed ball detections
    occlusion_per_min: float = 2.0   # ball occlusion runs (0.4-1.2 s) per minute
    player_dropout: float = 0.03
    fragment_s: Optional[float] = None   # new track id every ~N s (simulates ID switches)
    label_goalkeepers: bool = True
    with_referee: bool = True


class SyntheticMatch:
    def __init__(self, cfg: SynthConfig = SynthConfig(), pitch: Pitch = Pitch()):
        self.c, self.p = cfg, pitch
        self.rng = np.random.default_rng(cfg.seed)
        self.dt = 1.0 / cfg.fps
        L, W = pitch.length, pitch.width
        self.L, self.W = L, W
        # per-frame script buffers
        self.ball: List[np.ndarray] = []
        self.holder: List[int] = []
        self.recv: List[tuple] = []          # (player, target_xy) or (-1, None)
        self.dead: List[bool] = []
        self.kickoff_form: List[bool] = []
        self.presser: List[tuple] = []
        self.period: List[int] = []
        self.gt: List[dict] = []
        self.pos = np.zeros((22, 2))
        self.cur_period = 0
        ph = self.rng.uniform(0, 2 * np.pi, (22, 2, 2))
        per = self.rng.uniform(20, 60, (22, 2, 2))
        self.noise_par = (ph, per)

    # ------------------------------------------------------------------ geometry helpers
    def team(self, i):
        return i // 11

    def attacks_right(self, team, period=None):
        period = self.cur_period if period is None else period
        return (team == 0) == (period % 2 == 0)

    def to_att(self, team, xy):
        xy = np.asarray(xy, float)
        return xy if self.attacks_right(team) else np.array([self.L - xy[0], self.W - xy[1]])

    from_att = to_att  # the transform is an involution

    def base_pos(self, i, ball_xy, t, kickoff=False):
        tm = self.team(i)
        k = i % 11
        b = self.to_att(tm, ball_xy)
        f = FORMATION[k].copy()
        if kickoff:
            f[0] = min(f[0] * 0.9, 50.0)
        else:
            shift = 0.08 if k == 0 else 0.6
            f[0] = np.clip(f[0] + shift * (b[0] - 52.5), 2, 100)
            f[1] = np.clip(f[1] + (0.05 if k == 0 else 0.25) * (b[1] - 34), 2, 66)
            ph, per = self.noise_par
            f += 3.0 * np.array([np.sin(2 * np.pi * t / per[i, d, 0] + ph[i, d, 0]) * 0.6 +
                                 np.sin(2 * np.pi * t / per[i, d, 1] + ph[i, d, 1]) * 0.4 for d in (0, 1)]) * (k > 0)
        return self.from_att(tm, f)

    # ------------------------------------------------------------------ script primitives
    @property
    def t(self):
        return len(self.ball) * self.dt

    def _push(self, ball, holder=-1, recv=(-1, None), dead=False, ko=False, presser=(-1, None)):
        self.ball.append(np.array(ball, float))
        self.holder.append(holder)
        self.recv.append(recv)
        self.dead.append(dead)
        self.kickoff_form.append(ko)
        self.presser.append(presser)
        self.period.append(self.cur_period)

    def _event(self, typ, team, xy, outcome=None, t=None, **kw):
        self.gt.append(dict(t=self.t if t is None else t, type=typ, team=int(team), x=float(xy[0]), y=float(xy[1]),
                            outcome=outcome, period=self.cur_period, **kw))

    def hold(self, i, dur, ball, speed=None, presser=-1):
        tm = self.team(i)
        n = max(1, int(round(dur / self.dt)))
        sp = self.rng.uniform(0.5, 4.0) if speed is None else speed
        d_att = np.array([1.0, self.rng.normal(0, 0.4)])
        d = d_att if self.attacks_right(tm) else -d_att
        d /= np.linalg.norm(d)
        b = np.array(ball, float)
        for _ in range(n):
            b = b + d * sp * self.dt
            b[0] = np.clip(b[0], 1.5, self.L - 1.5)
            b[1] = np.clip(b[1], 1.5, self.W - 1.5)
            self._push(b, holder=i, presser=(presser, b.copy()) if presser >= 0 else (-1, None))
        return b

    def fly(self, a, b, speed, receiver=-1):
        a, b = np.asarray(a, float), np.asarray(b, float)
        n = max(1, int(np.ceil(np.linalg.norm(b - a) / speed / self.dt)))
        for k in range(1, n + 1):
            self._push(a + (b - a) * k / n, recv=(receiver, b) if receiver >= 0 else (-1, None))
        return b

    def stop(self, dur, ball=None, kickoff=False, walker=-1):
        """Dead ball for ``dur`` s. ``walker`` walks to the ball (the restart taker)."""
        n = max(1, int(round(dur / self.dt)))
        for k in range(n):
            self._push(ball if ball is not None else (np.nan, np.nan), dead=True, ko=kickoff and k > n * 0.4,
                       recv=(walker, np.asarray(ball, float)) if walker >= 0 and ball is not None else (-1, None))

    def current_positions(self):
        """Approximate player positions at the current script time (for choosing targets)."""
        b = self.ball[-1] if self.ball and not np.isnan(self.ball[-1][0]) else np.array([self.L / 2, self.W / 2])
        return np.array([self.base_pos(i, b, self.t) for i in range(22)])

    # ------------------------------------------------------------------ match engine
    def kickoff(self, team):
        c = np.array([self.L / 2, self.W / 2])
        kicker = 11 * team + 9
        self.stop(8.0, ball=c, kickoff=True)
        self._event("kickoff", team, c)
        pos = self.current_positions()
        mate = 11 * team + 6
        self.hold(kicker, 0.5, c, speed=0.0)
        self._event("pass", team, self.ball[-1], "complete")
        tgt = pos[mate]
        end = self.fly(self.ball[-1], tgt, 12.0, mate)
        return mate, end

    def chain(self, holder, ball, t_end):
        """Play until a stoppage; returns (next_action, info)."""
        while self.t < t_end:
            tm = self.team(holder)
            ball = self.hold(holder, self.rng.uniform(0.6, 2.0), ball)
            a = self.to_att(tm, ball)
            pos = self.current_positions()
            r = self.rng.random()
            if a[0] > 68 and abs(a[1] - 34) < 24 and r < 0.55:
                # attack the box: carry the ball towards goal, then shoot
                dur = min(self.rng.uniform(0.8, 2.6), max(0.2, (self.rng.uniform(86, 97) - a[0]) / 5.5))
                ball = self.hold(holder, dur, ball, speed=5.5)
                a = self.to_att(tm, ball)
                if a[0] > 77 and abs(a[1] - 34) < 20:
                    return self.shot(holder, ball, self.current_positions())
            r = self.rng.random()
            opps = [j for j in range(11 * (1 - tm), 11 * (1 - tm) + 11)]
            mates = [j for j in range(11 * tm, 11 * tm + 11) if j != holder]
            nearest_opp = min(opps[1:], key=lambda j: np.linalg.norm(pos[j] - ball))
            if r < 0.06:                                         # foul suffered
                ball = self.hold(holder, 1.0, ball, speed=1.0, presser=nearest_opp)
                self._event("foul", 1 - tm, ball, t=self.t)
                self.stop(self.rng.uniform(8, 15), ball=ball)
                return ("restart", holder, ball, "free_kick")
            if r < 0.12:                                         # dispossessed
                ball = self.hold(holder, 1.0, ball, speed=1.0, presser=nearest_opp)
                self._event("ball_recovery", 1 - tm, ball)
                holder = nearest_opp
                continue
            if r < 0.19:                                         # pass out over the touchline
                side = 0.0 - 3 if ball[1] < self.W / 2 else self.W + 3
                tgt = np.array([np.clip(ball[0] + self.rng.normal(0, 10), 3, self.L - 3), side])
                self._event("pass", tm, ball, "out")
                end = self.fly(ball, tgt, 15.0)
                exit_x = end[0]
                self.stop(self.rng.uniform(1.5, 3), ball=end)       # visible out of play
                self.stop(self.rng.uniform(5, 9))                   # ball being fetched (unseen)
                thr = np.array([exit_x, 0.5 if side < 0 else self.W - 0.5])
                taker = min(range(11 * (1 - tm) + 1, 11 * (1 - tm) + 11), key=lambda j: np.linalg.norm(pos[j] - thr))
                return ("restart", taker, thr, "throw_in")
            # pass to a teammate (forward-biased)
            att_pos = np.array([self.to_att(tm, pos[j]) for j in mates])
            w = np.exp(0.10 * (att_pos[:, 0] - a[0])) * (np.linalg.norm(att_pos - a, axis=1) > 6)
            w = w / w.sum()
            mate = mates[self.rng.choice(len(mates), p=w)]
            tgt = pos[mate] + self.rng.normal(0, 1.0, 2)
            tgt = np.clip(tgt, [1, 1], [self.L - 1, self.W - 1])
            speed = self.rng.uniform(12, 20)
            if r < 0.30:                                         # intercepted
                mid = ball + (tgt - ball) * self.rng.uniform(0.4, 0.8)
                icp = min(opps[1:], key=lambda j: np.linalg.norm(pos[j] - mid))
                self._event("pass", tm, ball, "intercepted")
                ball = self.fly(ball, mid, speed, icp)
                self._event("interception", 1 - tm, ball)
                holder = icp
                continue
            self._event("pass", tm, ball, "complete")
            ball = self.fly(ball, tgt, speed, mate)
            holder = mate
        return ("period_end", holder, ball, None)

    def shot(self, holder, ball, pos):
        tm = self.team(holder)
        a = self.to_att(tm, ball)
        o = self.rng.choice(["goal", "saved", "off_target", "blocked", "blocked_corner"], p=[0.12, 0.3, 0.3, 0.18, 0.10])
        gk = 11 * (1 - tm)
        if o == "goal":
            tgt = self.from_att(tm, np.array([self.L + 1.2, 34 + self.rng.uniform(-3, 3)]))
            self._event("shot", tm, ball, "goal")
            self._event("goal", tm, ball)
            self.fly(ball, tgt, 24.0)
            self.stop(2.0, ball=tgt)
            self.stop(self.rng.uniform(25, 40), kickoff=True)   # players walk back to kick-off shape
            return ("kickoff", 1 - tm, None, None)
        if o == "saved":
            tgt = self.from_att(tm, np.array([self.L - 1.0, 34 + self.rng.uniform(-2.5, 2.5)]))
            self._event("shot", tm, ball, "saved")
            end = self.fly(ball, tgt, 22.0, gk)
            end = self.hold(gk, self.rng.uniform(2, 4), end, speed=0.3)
            return ("continue", gk, end, None)
        if o == "off_target":
            sgn = self.rng.choice([-1, 1])
            tgt = self.from_att(tm, np.array([self.L + 3, 34 + sgn * self.rng.uniform(5, 10)]))
            self._event("shot", tm, ball, "off_target")
            self.fly(ball, tgt, 22.0)
            self.stop(self.rng.uniform(2, 3), ball=tgt)
            self.stop(self.rng.uniform(6, 10))
            gkp = self.from_att(tm, np.array([self.L - 5.0, 34 + self.rng.uniform(-4, 4)]))
            return ("restart", gk, gkp, "goal_kick")
        # blocked: defender on the shot line
        goal = self.from_att(tm, np.array([self.L, 34.0]))
        dvec = goal - ball
        blk = ball + dvec / np.linalg.norm(dvec) * self.rng.uniform(3, 7)
        defender = min(range(11 * (1 - tm) + 1, 11 * (1 - tm) + 11), key=lambda j: np.linalg.norm(pos[j] - blk))
        if o == "blocked":
            self._event("shot", tm, ball, "blocked")
            end = self.fly(ball, blk, 20.0, defender)
            return ("continue", defender, end, None)
        self._event("shot", tm, ball, "blocked_or_saved_out")
        self.fly(ball, blk, 20.0)
        sgn = self.rng.choice([-1, 1])
        out = self.from_att(tm, np.array([self.L + 2, 34 + sgn * self.rng.uniform(10, 18)]))
        self.fly(blk, out, 12.0)
        self.stop(2.0, ball=out)
        self.stop(self.rng.uniform(6, 10))
        corner = self.from_att(tm, np.array([self.L - 0.5, 0.5 if sgn < 0 else self.W - 0.5]))
        taker = 11 * tm + (5 if sgn < 0 else 8)
        return ("restart", taker, corner, "corner")

    def restart_play(self, taker, at, kind):
        self.stop(3.0, ball=at, walker=taker)
        self._event(kind, self.team(taker), at)
        # hold at the spot then the chain continues (first touch is the restart)
        return self.hold(taker, 0.6, at, speed=0.0)

    def generate(self) -> dict:
        c = self.c
        kick_team = 0
        for per in range(2):
            self.cur_period = per
            t_end = self.t + c.period_s
            holder, ball = self.kickoff(kick_team if per == 0 else 1 - kick_team)
            while self.t < t_end:
                act, who, at, kind = self.chain(holder, ball, t_end)
                if act == "period_end":
                    break
                if act == "kickoff":
                    holder, ball = self.kickoff(who)
                elif act == "restart":
                    ball = self.restart_play(who, at, kind)
                    holder = who
                else:
                    holder, ball = who, at
            if per == 0:
                n = int(c.halftime_s / self.dt)
                for _ in range(n):
                    self._push((np.nan, np.nan), dead=True)
                    self.period[-1] = -1
        return self._render()

    # ------------------------------------------------------------------ rendering
    def _render(self) -> dict:
        c, rng = self.c, self.rng
        N = len(self.ball)
        t = np.arange(N) * self.dt
        frames = np.arange(N) * int(round(25 / c.fps))           # pretend 25 fps source video
        P = np.zeros((N, 22, 2))
        self.cur_period = 0
        pos = np.array([self.base_pos(i, (self.L / 2, self.W / 2), 0, kickoff=True) for i in range(22)])
        last_ball = np.array([self.L / 2, self.W / 2])
        ko = sorted((e["t"], e["team"]) for e in self.gt if e["type"] == "kickoff")
        ko_t = np.array([x[0] for x in ko])
        for k in range(N):
            per = self.period[k]
            self.cur_period = per if per >= 0 else 1     # half-time: walk to the second-half sides
            b = self.ball[k]
            if not np.isnan(b[0]):
                last_ball = np.clip(b, [0, 0], [self.L, self.W])
            dead = self.dead[k]
            for i in range(22):
                tgt = self.base_pos(i, last_ball, t[k], kickoff=self.kickoff_form[k])
                vmax = 7.0
                if self.holder[k] == i:
                    d = (1 if self.attacks_right(self.team(i)) else -1)
                    tgt = b - np.array([0.45 * d, 0.0])
                    vmax = 12.0
                elif self.recv[k][0] == i:
                    tgt, vmax = self.recv[k][1], 8.5
                elif self.presser[k][0] == i:
                    tgt, vmax = self.presser[k][1] + np.array([0.6, 0.3]), 8.0
                if self.kickoff_form[k] and len(ko) and i == 11 * ko[min(np.searchsorted(ko_t, t[k] - 0.5), len(ko) - 1)][1] + 9:
                    tgt = np.array([self.L / 2 - (0.6 if self.attacks_right(self.team(i)) else -0.6), self.W / 2])
                busy = self.holder[k] == i or self.recv[k][0] == i or self.presser[k][0] == i
                if dead and not self.kickoff_form[k] and not busy:
                    vmax = 1.2
                step = tgt - pos[i]
                n = np.linalg.norm(step)
                if n > vmax * self.dt:
                    step *= vmax * self.dt / n
                pos[i] = pos[i] + step
            P[k] = pos
        # ---- tracks
        rows = []
        half_break = np.array(self.period) < 0
        for i in range(22):
            tm = self.team(i)
            cls = "goalkeeper" if (i % 11 == 0 and c.label_goalkeepers) else "player"
            team = tm if (i % 11 != 0 or c.label_goalkeepers) else -1
            noise = rng.normal(0, c.noise_player_m, (N, 2))
            keep = (rng.random(N) > c.player_dropout) & ~half_break
            if c.fragment_s:
                seg = np.cumsum(rng.random(N) < self.dt / c.fragment_s)
                tid = (i + 1) * 1000 + seg
            else:
                tid = np.full(N, i + 1)
            xy = P[:, i] + noise
            for k in np.where(keep)[0]:
                rows.append((frames[k], t[k], int(tid[k]), cls, 0.9, xy[k, 0], xy[k, 1], team))
        if c.with_referee:
            ref = np.zeros((N, 2))
            rp = np.array([self.L / 2, self.W / 2 + 8])
            lb = rp.copy()
            for k in range(N):
                if not np.isnan(self.ball[k][0]):
                    lb = self.ball[k]
                tgt = lb + np.array([-6, 7])
                step = tgt - rp
                n = np.linalg.norm(step)
                rp = rp + (step * min(1, 6 * self.dt / (n + 1e-9)))
                ref[k] = rp
            for k in np.where(~half_break)[0]:
                rows.append((frames[k], t[k], 999, "referee", 0.9, ref[k, 0], ref[k, 1], -1))
        tracks = pd.DataFrame(rows, columns=["frame", "t", "track_id", "cls", "conf", "x", "y", "team"])
        tracks = tracks.sort_values(["frame", "track_id"]).reset_index(drop=True)
        # ---- ball detections
        B = np.array(self.ball)
        vis = ~np.isnan(B[:, 0])
        vis &= rng.random(N) > c.ball_dropout
        n_occ = int(c.occlusion_per_min * N * self.dt / 60)
        for _ in range(n_occ):
            k0 = rng.integers(0, N)
            vis[k0:k0 + int(rng.uniform(0.4, 1.2) / self.dt)] = False
        Bn = B + rng.normal(0, c.noise_ball_m, B.shape)
        ball_raw = pd.DataFrame({"frame": frames[vis], "t": t[vis], "conf": 0.6, "x": Bn[vis, 0], "y": Bn[vis, 1]})
        ball = clean_ball(ball_raw, frames, t)
        gt = pd.DataFrame(self.gt).sort_values("t").reset_index(drop=True)
        truth = pd.DataFrame({"frame": frames, "t": t, "bx": B[:, 0], "by": B[:, 1], "holder": self.holder,
                              "dead": self.dead, "period": self.period})
        truth["holder_team"] = np.where(truth.holder >= 0, truth.holder // 11, -1)
        periods = []
        per = np.array(self.period)
        for p in (0, 1):
            idx = np.where(per == p)[0]
            periods.append((float(t[idx[0]]), float(t[idx[-1]])))
        return dict(tracks=tracks, ball=ball, ball_raw=ball_raw, gt_events=gt, truth=truth, periods=periods)


SET_PIECE_TOL = {"foul": 4.0, "free_kick": 4.0, "throw_in": 4.0, "corner": 4.0, "goal_kick": 4.0, "kickoff": 4.0,
                 "penalty": 4.0, "offside": 4.0}


def evaluate_events(pred: pd.DataFrame, gt: pd.DataFrame, types=None, tol_s: float = 2.0, match_team: bool = True,
                    tol_by_type: Optional[dict] = None) -> pd.DataFrame:
    """Greedy one-to-one matching per event type within a time tolerance; returns precision / recall / F1.

    Stoppage-based events (set pieces, fouls) have fuzzier timestamps, so they get ``SET_PIECE_TOL``.
    """
    tol_by_type = SET_PIECE_TOL if tol_by_type is None else tol_by_type
    alias = {"foul_proxy": "foul"}
    pred = pred.assign(type=pred.type.replace(alias))
    types = types or sorted(set(gt.type) | set(pred.type))
    rows = []
    for typ in types:
        p = pred[pred.type == typ].sort_values("t")
        g = gt[gt.type == typ].sort_values("t")
        used = set()
        tp = 0
        for _, e in p.iterrows():
            cand = g[(np.abs(g.t - e.t) <= tol_by_type.get(typ, tol_s)) & (~g.index.isin(used))]
            if match_team:
                cand = cand[cand.team == e.team]
            if len(cand):
                used.add((cand.t - e.t).abs().idxmin())
                tp += 1
        prec = tp / len(p) if len(p) else np.nan
        rec = tp / len(g) if len(g) else np.nan
        f1 = 2 * prec * rec / (prec + rec) if (len(p) and len(g) and prec + rec > 0) else np.nan
        rows.append(dict(type=typ, n_gt=len(g), n_pred=len(p), tp=tp, precision=prec, recall=rec, f1=f1))
    return pd.DataFrame(rows).set_index("type")
