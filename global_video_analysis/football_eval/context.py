"""Match context: periods (halves), attacking direction per team, per-frame helper arrays."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .pitch import Pitch


@dataclass
class MatchContext:
    pitch: Pitch
    periods: List[Tuple[float, float]]                      # video seconds
    attacks_right: List[Dict[int, bool]]                    # per period: team -> attacks towards x = L
    team_names: List[str] = field(default_factory=lambda: ["Team A", "Team B"])

    def period_of(self, t: float) -> int:
        for i, (a, b) in enumerate(self.periods):
            if a - 1e-6 <= t <= b + 1e-6:
                return i
        # closest period
        return int(np.argmin([min(abs(t - a), abs(t - b)) for a, b in self.periods]))

    def attacking_right(self, team: int, t: float) -> bool:
        return self.attacks_right[self.period_of(t)][int(team)]

    def to_attack_frame(self, team: int, t: float, x, y):
        """Rotate coordinates so that ``team`` attacks towards +x (x'=L is the opponent goal line)."""
        if self.attacking_right(team, t):
            return np.asarray(x, float), np.asarray(y, float)
        return self.pitch.length - np.asarray(x, float), self.pitch.width - np.asarray(y, float)

    def match_clock(self, t: float, half_len_min: float = 45.0) -> str:
        """Football clock, e.g. 23' or 45+2' (2nd half starts at 45')."""
        p = self.period_of(t)
        start = self.periods[p][0]
        m = max(0.0, (t - start) / 60.0)
        base = p * half_len_min
        if m >= half_len_min:
            return f"{int(base + half_len_min)}+{int(m - half_len_min) + 1}'"
        return f"{int(base + m) + 1}'"


def infer_periods(tracks: pd.DataFrame, pitch: Pitch, halves: Optional[list] = None,
                  attacks_right_first: Optional[bool] = None, min_period_s: float = 300.0) -> MatchContext:
    """Infer halves from the moment the two teams swap sides (needs team labels + pitch x)."""
    p = tracks[(tracks.team >= 0) & tracks.x.notna()]
    t_end = float(tracks.t.max()) if len(tracks) else 0.0
    t_start = float(tracks.t.min()) if len(tracks) else 0.0
    if halves:
        periods = [tuple(map(float, h)) for h in halves]
    else:
        periods = [(t_start, t_end)]
        if len(p):
            # 30-s bins: sign of mean x(team0) - mean x(team1)
            b = (p.t // 30).astype(int)
            m = p.groupby([b, "team"]).x.mean().unstack()
            if 0 in m and 1 in m:
                s = np.sign((m[0] - m[1]).dropna())
                # change point maximising agreement: left of it sign = s0, right = -s0
                best, best_k = -1, None
                vals = s.to_numpy()
                for k in range(1, len(vals)):
                    sc = (vals[:k] == vals[0]).sum() + (vals[k:] == -vals[0]).sum()
                    if sc > best:
                        best, best_k = sc, k
                if best_k is not None and best >= 0.8 * len(vals):
                    tc = float(s.index[best_k] * 30)
                    if tc - t_start >= min_period_s and t_end - tc >= min_period_s:
                        # trim the half-time break: last / first bins with play
                        t1 = float(p[p.t < tc].t.max())
                        t2 = float(p[p.t >= tc].t.min())
                        periods = [(t_start, t1), (t2, t_end)]
    attacks = []
    for i, (a, b) in enumerate(periods):
        q = p[(p.t >= a) & (p.t <= b)]
        if attacks_right_first is not None:
            r0 = attacks_right_first if i % 2 == 0 else not attacks_right_first
        elif len(q) and q.team.nunique() == 2:
            mx = q.groupby("team").x.mean()
            r0 = bool(mx.get(0, 0) < mx.get(1, 0))   # team defending the left half attacks right
        else:
            r0 = (i % 2 == 0)
        attacks.append({0: r0, 1: not r0})
    return MatchContext(pitch, periods, attacks)


def player_speeds(tracks: pd.DataFrame, max_speed: float = 11.0, smooth: int = 5) -> pd.DataFrame:
    """Add vx, vy, speed (m/s) per track using a centred rolling mean of positions."""
    tr = tracks.sort_values(["track_id", "t"]).copy()
    g = tr.groupby("track_id")
    xs = g.x.transform(lambda s: s.rolling(smooth, center=True, min_periods=1).mean())
    ys = g.y.transform(lambda s: s.rolling(smooth, center=True, min_periods=1).mean())
    tr["xs"], tr["ys"] = xs, ys
    dt = g.t.diff()
    dx, dy = tr.groupby("track_id").xs.diff(), tr.groupby("track_id").ys.diff()
    sp = np.hypot(dx, dy) / dt
    sp[(dt > 1.0) | (sp > max_speed)] = np.nan
    tr["speed"] = sp
    return tr
