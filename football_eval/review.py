"""Human review of a processed match: spot-check frames + accept / reject the key events.

After perception + analytics, ``python scripts/match_ui.py review --run outputs/<match>`` shows

1. **Frame checks** — ~16 frames sampled across the match (stratified in time; most of them among the frames
   the automatic camera-quality gate *accepted*, a few among those it rejected) with the projected pitch lines
   and every player box coloured by team. You answer two questions per frame: *do the lines sit on the
   painted lines?* and *are the team colours right?* Those answers are a **measurement** of the perception
   quality on *your* video (precision of the calibration gate with a 95 % interval) instead of a guess.
2. **Key events** — goals, penalties and shots (the events that drive xG and the timeline). For each one a
   short clip around the event; you *confirm*, *reject* or *flip the team*, and may type the shirt number.
3. **Missing events** — goals / shots / cards you saw that the rules missed.

The answers are stored in ``outputs/<match>/review.json`` and applied by the analytics
(:func:`apply_review`); the match sheet in ``match_setup.json`` (final score, goal minutes) is reconciled with
the automatic goals in the same step (:func:`reconcile_match_sheet`). Nothing is deleted silently: rejected
events go to ``events_rejected.csv`` with the reason.
"""
from __future__ import annotations

import datetime as _dt
import json
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from .match_setup import event_time, roster_lookup

REVIEW_TYPES = ("goal", "penalty", "shot")
GOAL_TYPES = ("goal", "own_goal", "penalty_goal")


def _now():
    return _dt.datetime.now().replace(microsecond=0).isoformat()


def new_review(run_dir: str) -> dict:
    return {"version": 1, "run": str(run_dir), "frame_checks": [], "events": [], "added_events": [], "repairs": [],
            "meta": {"created": _now(), "updated": _now()}}


def load_review(path: str | Path) -> dict:
    p = Path(path)
    if not p.exists():
        return new_review(str(p.parent))
    d = json.loads(p.read_text())
    base = new_review(d.get("run", str(p.parent)))
    base.update(d)
    return base


def save_review(review: dict, path: str | Path) -> Path:
    review.setdefault("meta", {})["updated"] = _now()
    p = Path(path)
    p.write_text(json.dumps(review, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
    return p


def event_key(e) -> str:
    team = int(e["team"]) if pd.notna(e["team"]) else -1
    return f"{e['type']}@{float(e['t']):.1f}@{team}"


# ----------------------------------------------------------------------------- frame sampling
def sample_check_frames(frames: pd.DataFrame, n_ok: int = 12, n_rejected: int = 4, seed: int = 0) -> List[dict]:
    """Stratified (in time) sample of analysed frames for the spot check.

    Most frames come from those the automatic quality gate kept (``cam_ok`` / ``has_h``): they feed the
    statistics, so their precision matters. A few come from the rejected ones, to see whether the gate is too
    strict (frames you judge good that were thrown away).
    """
    rng = np.random.default_rng(seed)
    fr = frames.copy()
    ok = fr["cam_ok"].astype(bool) if "cam_ok" in fr else fr["has_h"].astype(bool) if "has_h" in fr else \
        pd.Series(True, index=fr.index)
    out = []
    for flag, n in ((True, n_ok), (False, n_rejected)):
        pool = fr[ok == flag]
        if len(pool) == 0 or n == 0:
            continue
        pool = pool.sort_values("t")
        for chunk in np.array_split(np.arange(len(pool)), min(n, len(pool))):
            r = pool.iloc[int(chunk[int(rng.integers(len(chunk)))])]
            out.append(dict(frame=int(r.frame), t=float(r.t), auto_ok=bool(flag), lines=None, teams=None, note=""))
    return sorted(out, key=lambda d: d["t"])


def candidate_events(events: pd.DataFrame, types=REVIEW_TYPES, max_n: int = 60) -> List[dict]:
    """Events worth a human look, most important first (goals, penalties, then shots by xG)."""
    if events is None or len(events) == 0:
        return []
    q = events[events.type.isin(types)].copy()
    pri = {"goal": 0, "penalty": 1, "shot": 2}
    q["_p"] = q.type.map(pri).fillna(9)
    q["_x"] = -q.xg.fillna(0)
    q = q.sort_values(["_p", "_x"]).head(max_n).sort_values("t")
    cols = [c for c in ("t", "type", "team", "tid", "outcome", "xg", "source", "detail", "why", "frame") if c in q]
    out = []
    for _, r in q[cols].iterrows():
        d = {k: (None if (isinstance(v, float) and np.isnan(v)) else (v.item() if hasattr(v, "item") else v))
             for k, v in r.items()}
        d["key"] = event_key(r)
        out.append(d)
    return out


# ----------------------------------------------------------------------------- statistics of the review
def wilson(k: int, n: int, z: float = 1.96) -> Tuple[float, float, float]:
    """Proportion k/n with its Wilson 95 % interval (sensible for the small samples of a spot check)."""
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    p = k / n
    den = 1 + z ** 2 / n
    c = (p + z ** 2 / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / den
    return p, max(0.0, c - h), min(1.0, c + h)


def _pct(k, n):
    p, lo, hi = wilson(k, n)
    return "n/a" if n == 0 else f"{100 * p:.0f} % ({k}/{n}; 95 % CI {100 * lo:.0f}-{100 * hi:.0f} %)"


def validation_summary(review: dict) -> pd.DataFrame:
    """One row per measured quantity: what you checked, the result, and what it means."""
    rows = []
    fc = review.get("frame_checks", [])
    kept = [c for c in fc if c.get("auto_ok")]
    rej = [c for c in fc if not c.get("auto_ok")]
    judged = [c for c in kept if c.get("lines") in ("good", "bad")]
    k = sum(c["lines"] == "good" for c in judged)
    rows.append(dict(check="pitch lines on frames used for the stats", result=_pct(k, len(judged)),
                     meaning="share of positions you can trust (camera calibration correct)"))
    jr = [c for c in rej if c.get("lines") in ("good", "bad")]
    if jr:
        kr = sum(c["lines"] == "good" for c in jr)
        rows.append(dict(check="frames discarded by the quality gate that were actually good", result=_pct(kr, len(jr)),
                         meaning="high -> the gate (pitch.ptz_max_err_px) is too strict and throws data away"))
    nl = sum(c.get("lines") == "no_lines" for c in fc)
    if nl:
        rows.append(dict(check="frames without visible lines", result=str(nl),
                         meaning="cannot be judged; excluded from the share above"))
    tj = [c for c in fc if c.get("teams") in ("good", "bad")]
    kt = sum(c["teams"] == "good" for c in tj)
    rows.append(dict(check="team colours correct on the frame", result=_pct(kt, len(tj)),
                     meaning="frames where every visible box had the right team"))
    ev = [e for e in review.get("events", []) if e.get("verdict")]
    for typ in sorted({e["type"] for e in ev}):
        q = [e for e in ev if e["type"] == typ]
        real = sum(e["verdict"] in ("confirm", "wrong_team") for e in q)
        rows.append(dict(check=f"{typ}s that really happened", result=_pct(real, len(q)),
                         meaning=f"precision of the automatic {typ} rule on this match"))
        team_j = [e for e in q if e["verdict"] in ("confirm", "wrong_team")]
        if team_j:
            rows.append(dict(check=f"{typ}s with the right team", result=_pct(sum(e['verdict'] == 'confirm' for e in team_j),
                                                                              len(team_j)), meaning="team attribution"))
    add = review.get("added_events", [])
    if add:
        by = pd.Series([a["type"] for a in add]).value_counts().to_dict()
        rows.append(dict(check="events the rules missed (added by hand)", result=", ".join(f"{v} {k}" for k, v in by.items()),
                         meaning="recall gap — these are now in the stats"))
    return pd.DataFrame(rows, columns=["check", "result", "meaning"])


# ----------------------------------------------------------------------------- applying the review
def _basic_xg(xg_model, ctx, team, t, x, y):
    if xg_model is None or pd.isna(x):
        return np.nan
    from .xg import feature_row
    L, W = ctx.pitch.length, ctx.pitch.width
    sx, sy = ctx.to_attack_frame(int(team), t, x, y)
    return float(xg_model.predict([feature_row(float(sx), float(sy), None, L, W)], "basic")[0])


def clicked_xg(xg_model, ctx, tracks: Optional[pd.DataFrame], team: int, t: float, frame: int, shot_xy,
               gk_xy=None, situation: str = "open_play", max_frame_gap: int = 6):
    """xG from a shot location (and goalkeeper) clicked by the user, with the freeze frame of the opponents
    taken from tracking at the nearest analysed frame. Returns (xg, freeze-frame features)."""
    from .xg import freeze_frame_features, feature_row
    if xg_model is None or shot_xy is None or team not in (0, 1):
        return np.nan, {}
    L, W = ctx.pitch.length, ctx.pitch.width
    sx, sy = ctx.to_attack_frame(team, t, shot_xy[0], shot_xy[1])
    opp_xy, gk = np.zeros((0, 2)), None
    if tracks is not None and len(tracks) and frame is not None and frame >= 0:
        fr = tracks.frame.to_numpy()
        j = fr[np.argmin(np.abs(fr - frame))] if len(fr) else None
        if j is not None and abs(int(j) - frame) <= max_frame_gap:
            q = tracks[(tracks.frame == j) & (tracks.team == 1 - team) & tracks.x.notna()]
            out = q[q.cls != "goalkeeper"]
            ox, oy = ctx.to_attack_frame(team, t, out.x.to_numpy(), out.y.to_numpy())
            opp_xy = np.c_[ox, oy]
            g = q[q.cls == "goalkeeper"]
            if gk_xy is None and len(g):
                gx, gy = ctx.to_attack_frame(team, t, g.x.to_numpy()[:1], g.y.to_numpy()[:1])
                gk = (float(gx[0]), float(gy[0]))
    if gk_xy is not None:
        gx, gy = ctx.to_attack_frame(team, t, gk_xy[0], gk_xy[1])
        gk = (float(gx), float(gy))
        if len(opp_xy):                       # the clicked keeper may also be in the outfield list: drop the closest
            d = np.hypot(opp_xy[:, 0] - gk[0], opp_xy[:, 1] - gk[1])
            opp_xy = opp_xy[d > 2.0]
    ff = freeze_frame_features((float(sx), float(sy)), opp_xy, gk, L, W)
    row = feature_row(float(sx), float(sy), ff, L, W)
    return float(xg_model.predict([row], "full", [situation])[0]), dict(ff, dist=row["dist"])


def apply_review(events: pd.DataFrame, review: Optional[dict], setup: Optional[dict], ctx, xg_model=None,
                 tracks: Optional[pd.DataFrame] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Apply the user's verdicts. Returns (events used for the statistics, rejected events with reason).

    Shot locations / goalkeeper positions clicked in the review replace the tracked ones and the xG is recomputed
    with the full model (``detail`` gets ``|location=clicked``)."""
    ev = events.copy()
    for c, v in (("validation", "unreviewed"), ("number", np.nan), ("player", None), ("reject_reason", None),
                 ("location", None)):
        if c not in ev:
            ev[c] = v
    rejected = []

    def relocate(i, d, team):
        if not d.get("shot_xy"):
            return
        sit = "penalty" if "penalty" in str(ev.at[i, "detail"]) else \
            "direct_free_kick" if "direct_free_kick" in str(ev.at[i, "detail"]) else "open_play"
        fr = int(d.get("shot_frame") if d.get("shot_frame") is not None else ev.at[i, "frame"])
        xg, ff = clicked_xg(xg_model, ctx, tracks, team, float(ev.at[i, "t"]), fr, d["shot_xy"], d.get("gk_xy"), sit)
        ev.at[i, "x"], ev.at[i, "y"] = float(d["shot_xy"][0]), float(d["shot_xy"][1])
        ev.at[i, "location"] = "clicked" + ("+gk" if d.get("gk_xy") else "")
        if ev.at[i, "type"] == "shot" and np.isfinite(xg):
            ev.at[i, "xg"] = xg
            ev.at[i, "detail"] = (f"{sit}|dist={ff.get('dist', np.nan):.1f}|def_tri={ff.get('n_def_tri', 0)}"
                                  f"|location=clicked" + ("+gk" if d.get("gk_xy") else ""))

    if review:
        keys = ev.apply(event_key, axis=1) if len(ev) else pd.Series(dtype=str)
        for d in review.get("events", []):
            v = d.get("verdict")
            if not v and not d.get("shot_xy"):
                continue
            idx = ev.index[keys == d["key"]]
            if len(idx) == 0:          # tolerate re-runs where the time moved by a frame
                idx = ev.index[(ev.type == d["type"]) & (np.abs(ev.t - float(d["t"])) < 0.5)]
            if len(idx) == 0:
                continue
            i = idx[0]
            if d.get("number") not in (None, ""):
                ev.at[i, "number"] = float(d["number"])
            if v == "reject":
                ev.at[i, "reject_reason"] = "rejected in review" + (f": {d['note']}" if d.get("note") else "")
                rejected.append(i)
                if ev.at[i, "type"] == "shot":            # the goal derived from this shot goes too
                    g = ev.index[(ev.type == "goal") & (ev.source == "shot_rule") & (np.abs(ev.t - ev.at[i, "t"]) < 0.05)]
                    for j in g:
                        ev.at[j, "reject_reason"] = "its shot was rejected in review"
                    rejected.extend(g)
                continue
            team = int(ev.at[i, "team"])
            if v == "wrong_team":
                team = 1 - team
                ev.at[i, "team"] = team
                ev.at[i, "validation"] = "corrected"
                if ev.at[i, "type"] == "shot" and not d.get("shot_xy"):
                    ev.at[i, "xg"] = _basic_xg(xg_model, ctx, team, ev.at[i, "t"], ev.at[i, "x"], ev.at[i, "y"])
                    ev.at[i, "detail"] = str(ev.at[i, "detail"]) + "|xg=basic(team corrected)"
            elif v == "confirm":
                ev.at[i, "validation"] = "confirmed"
            relocate(i, d, team)
            if d.get("outcome"):
                ev.at[i, "outcome"] = d["outcome"]
            if ev.at[i, "type"] == "shot" and d.get("shot_xy"):      # the goal of this shot follows its location
                g = ev.index[(ev.type == "goal") & (ev.source == "shot_rule") & (np.abs(ev.t - ev.at[i, "t"]) < 0.05)]
                for j in g:
                    ev.at[j, "x"], ev.at[j, "y"], ev.at[j, "xg"] = ev.at[i, "x"], ev.at[i, "y"], ev.at[i, "xg"]
        add = []
        for a in review.get("added_events", []):
            team = int(a["team"]) if a.get("team") not in (None, "") else -1
            xy = a.get("shot_xy")
            xg = np.nan
            if a["type"] in ("shot", "goal", "penalty") and xy:
                sit = "penalty" if a["type"] == "penalty" else "open_play"
                xg, _ = clicked_xg(xg_model, ctx, tracks, team, float(a["t"]), int(a.get("shot_frame") or -1), xy,
                                   a.get("gk_xy"), sit)
            add.append(dict(t=float(a["t"]), frame=int(a.get("shot_frame") or -1), period=ctx.period_of(float(a["t"])),
                            type=a["type"], team=team, tid=-1,
                            x=xy[0] if xy else np.nan, y=xy[1] if xy else np.nan,
                            outcome=a.get("outcome") or ("manual" if a["type"] == "goal" else None),
                            xg=xg if a["type"] == "shot" else np.nan, detail=a.get("note", ""), source="manual",
                            validation="manual", location="clicked" if xy else None,
                            number=float(a["number"]) if a.get("number") not in (None, "") else np.nan))
        if add:
            ev = pd.concat([ev, pd.DataFrame(add)], ignore_index=True)
    rej = ev.loc[sorted(set(rejected))].copy() if rejected else ev.iloc[0:0].copy()
    ev = ev.drop(index=sorted(set(rejected)))
    if setup is not None:
        ev, rej2 = reconcile_match_sheet(ev, setup, ctx)
        rej = pd.concat([rej, rej2], ignore_index=True)
        ev, rej3 = drop_excluded(ev, setup.get("exclusions") or [])
        rej = pd.concat([rej, rej3], ignore_index=True)
        ev["player"] = [roster_lookup(setup, int(t) if pd.notna(t) else -1, n) if pd.notna(n) else None
                        for t, n in zip(ev.team, ev.number)]
    ev = ev.sort_values("t").reset_index(drop=True)
    if "clock" in ev:
        ev["clock"] = [ctx.match_clock(t) for t in ev.t]
    return ev, rej


# ----------------------------------------------------------------------------- excluded stretches
def in_exclusions(t, exclusions) -> np.ndarray:
    t = np.asarray(t, float)
    m = np.zeros(t.shape, bool)
    for e in exclusions or []:
        m |= (t >= float(e["t0"])) & (t <= float(e["t1"]))
    return m


def drop_excluded(ev: pd.DataFrame, exclusions) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Automatic events inside a stretch the user excluded (replay, crowd, injury break...) are removed.
    Events entered by hand or from the match sheet are kept."""
    if not exclusions or len(ev) == 0:
        return ev, ev.iloc[0:0].copy()
    m = in_exclusions(ev.t, exclusions) & ~ev.source.isin(["manual", "match_sheet"]).to_numpy()
    rej = ev[m].copy()
    rej["reject_reason"] = "inside an excluded stretch"
    return ev[~m], rej


def apply_exclusions(tracks: pd.DataFrame, ball: pd.DataFrame, exclusions) -> Tuple[pd.DataFrame, pd.DataFrame, float]:
    """Treat excluded stretches like 'camera lost': no positions, no ball, possession unknown (not dead ball)."""
    if not exclusions:
        return tracks, ball, 0.0
    tracks, ball = tracks.copy(), ball.copy()
    mt = in_exclusions(tracks.t, exclusions)
    tracks.loc[mt, ["x", "y"]] = np.nan
    mb = in_exclusions(ball.t, exclusions)
    ball.loc[mb, ["x", "y", "vx", "vy", "speed"]] = np.nan
    ball.loc[mb, "valid"] = False
    if "cam_ok" not in ball:
        ball["cam_ok"] = True
    ball.loc[mb, "cam_ok"] = False
    return tracks, ball, float(sum(float(e["t1"]) - float(e["t0"]) for e in exclusions))


# ----------------------------------------------------------------------------- camera repair
def lost_stretches(frames: pd.DataFrame, min_s: float = 3.0, merge_gap_s: float = 2.0) -> List[dict]:
    """Stretches where the automatic camera-quality gate rejected the pose (candidates for a repair anchor)."""
    if frames is None or "cam_ok" not in frames or len(frames) == 0:
        return []
    f = frames.sort_values("t")
    bad = ~f.cam_ok.astype(bool).to_numpy()
    t, fr = f.t.to_numpy(), f.frame.to_numpy()
    out = []
    i = 0
    while i < len(bad):
        if bad[i]:
            j = i
            while j + 1 < len(bad) and (bad[j + 1] or (t[j + 1] - t[j] < merge_gap_s and j + 2 < len(bad) and bad[j + 2])):
                j += 1
            if t[j] - t[i] >= min_s:
                out.append(dict(t0=float(t[i]), t1=float(t[j]), frame0=int(fr[i]), frame1=int(fr[j]),
                                dur=round(float(t[j] - t[i]), 1)))
            i = j + 1
        else:
            i += 1
    return out


def repair_window(frames: pd.DataFrame, t_anchor: float, max_s: float = 180.0, settle_s: float = 5.0) -> Tuple[float, float]:
    """Video window to re-process from a repair anchor: from the anchor to the end of the lost stretch it is in
    (+ ``settle_s``), capped at ``max_s``."""
    st = [s for s in lost_stretches(frames, min_s=0.0) if s["t0"] - 1.0 <= t_anchor <= s["t1"] + 1.0]
    end = (st[0]["t1"] if st else t_anchor + 30.0) + settle_s
    return float(t_anchor), float(min(end, t_anchor + max_s))


def reconcile_match_sheet(ev: pd.DataFrame, setup: dict, ctx, window_s: float = 120.0
                          ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Make the goals agree with the match sheet (final score + goal list) of ``match_setup.json``.

    * a sheet goal with a time that matches an automatic / manual goal of the same team (± ``window_s``) confirms
      it (and gives it the scorer's number);
    * a sheet goal with no matching goal is added (``source = match_sheet``) — the system missed it;
    * if the sheet lists *every* goal of the final score, automatic goals left unmatched are rejected
      ("not on the match sheet") — e.g. a long injury stoppage followed by a kick-off-like restart;
    * cards and substitutions are added as events (they cannot be seen reliably from one wide camera).
    """
    ev = ev.copy()
    known = setup.get("known_events", []) or []
    goals_k = [e for e in known if e.get("type") in GOAL_TYPES]
    fs = setup.get("final_score")
    complete = fs is not None and len(goals_k) == sum(fs)
    used = set()
    add = []
    auto_goal_idx = list(ev.index[ev.type == "goal"])
    for g in goals_k:
        t = event_time(setup, g)
        team = int(g["team"]) if g.get("team") not in (None, "") else -1
        num = float(g["number"]) if g.get("number") not in (None, "") else np.nan
        best, best_d = None, np.inf
        if t is not None:
            for i in auto_goal_idx:
                if i in used or int(ev.at[i, "team"]) != team:
                    continue
                d = abs(float(ev.at[i, "t"]) - t)
                if d <= window_s and d < best_d:
                    best, best_d = i, d
        if best is not None:
            used.add(best)
            if ev.at[best, "validation"] in ("unreviewed", "confirmed"):
                ev.at[best, "validation"] = "match_sheet_confirmed"
            if pd.notna(num):
                ev.at[best, "number"] = num
        elif t is not None:
            add.append(dict(t=t, frame=-1, period=ctx.period_of(t), type="goal", team=team, tid=-1, x=np.nan, y=np.nan,
                            outcome=g["type"], xg=np.nan, detail=f"minute {g.get('minute', '')}".strip(),
                            source="match_sheet", validation="match_sheet", number=num))
    rejected = []
    if complete:
        for i in auto_goal_idx:
            if i not in used and ev.at[i, "source"] not in ("manual", "match_sheet"):
                ev.at[i, "reject_reason"] = "not on the match sheet (final score %d-%d)" % tuple(fs)
                rejected.append(i)
    for e in known:
        if e.get("type") in ("yellow", "red", "sub", "penalty_missed"):
            t = event_time(setup, e)
            if t is None:
                continue
            add.append(dict(t=t, frame=-1, period=ctx.period_of(t), type=e["type"],
                            team=int(e["team"]) if e.get("team") not in (None, "") else -1, tid=-1, x=np.nan, y=np.nan,
                            outcome=None, xg=np.nan, detail=e.get("note", ""), source="match_sheet",
                            validation="match_sheet",
                            number=float(e["number"]) if e.get("number") not in (None, "") else np.nan))
    rej = ev.loc[rejected].copy()
    ev = ev.drop(index=rejected)
    if add:
        ev = pd.concat([ev, pd.DataFrame(add)], ignore_index=True)
    return ev, rej


# ----------------------------------------------------------------------------- drawing (UI + notebooks)
class RunGeometry:
    """Everything needed to draw a processed frame: player boxes by team, ball, projected pitch lines."""

    def __init__(self, run_dir: str, cfg, tracks: Optional[pd.DataFrame] = None):
        from .config import resolve_path
        from .pitch import Pitch
        d = Path(run_dir)
        self.dir = d
        self.tracks = tracks if tracks is not None else pd.read_csv(d / "tracks.csv")
        self.ball = pd.read_csv(d / "ball.csv") if (d / "ball.csv").exists() else None
        self.frames = pd.read_csv(d / "frames.csv") if (d / "frames.csv").exists() else None
        self.cam_log = pd.read_csv(d / "camera_log.csv") if (d / "camera_log.csv").exists() else None
        self.cfg = cfg
        meta = json.loads((d / "meta.json").read_text()).get("video", {}) if (d / "meta.json").exists() else {}
        self.size = (int(meta.get("width") or 4000), int(meta.get("height") or 4000))
        self.pitch = Pitch(cfg.pitch.length, cfg.pitch.width)
        self.cam, self.H = None, None
        if cfg.pitch.homography == "ptz" and cfg.pitch.calibration_file:
            from .ptz import load_ptz
            self.cam, _ = load_ptz(resolve_path(cfg.pitch.calibration_file), self.pitch)
            self.pitch = self.cam.pitch
        elif cfg.pitch.homography == "static":
            from .pitch import calibration_from_dict, load_calibration
            if cfg.pitch.calibration_points:
                self.H = calibration_from_dict({"points": cfg.pitch.calibration_points}, self.pitch)
            elif cfg.pitch.calibration_file:
                self.H = load_calibration(resolve_path(cfg.pitch.calibration_file), self.pitch)

    def lines(self, frame: int) -> List[list]:
        from .ptz import template_segments
        A, B = template_segments(self.pitch, 1.0)
        if self.cam is not None and self.cam_log is not None:
            r = self.cam_log[self.cam_log.frame == frame]
            if not len(r) or not np.isfinite(r.pan.iloc[0]):
                return []
            p = (float(r.pan.iloc[0]), float(r.tilt.iloc[0]), float(r.f.iloc[0]))
            a, fa = self.cam.project_ours(A, p)
            b, fb = self.cam.project_ours(B, p)
        elif self.H is not None:
            c = (self.pitch.length / 2, self.pitch.width / 2)
            a, fa = project_h(self.H.H, A, c)
            b, fb = project_h(self.H.H, B, c)
        else:
            return []
        ok = fa & fb & np.all(np.abs(a) < 5e3, 1) & np.all(np.abs(b) < 5e3, 1)
        ok &= visible_segments(a, b, *self.size)
        return np.round(np.c_[a[ok], b[ok]], 1).tolist()

    def homography(self, frame: int, max_gap: int = 6):
        """(Homography | None, camera usable flag, source) for a frame — used to turn clicks into pitch metres."""
        if self.cam is not None and self.cam_log is not None and len(self.cam_log):
            fr = self.cam_log.frame.to_numpy()
            k = int(np.argmin(np.abs(fr - frame)))
            r = self.cam_log.iloc[k]
            if abs(int(r.frame) - frame) > max_gap or not np.isfinite(r.pan):
                return None, False, "no camera pose near this frame"
            ok = True
            if self.frames is not None and "cam_ok" in self.frames:
                q = self.frames[self.frames.frame == int(r.frame)]
                ok = bool(q.cam_ok.iloc[0]) if len(q) else True
            return self.cam.homography((float(r.pan), float(r.tilt), float(r.f))), ok, "tracked camera pose"
        if self.H is not None:
            return self.H, True, "static calibration"
        return None, False, "no calibration"

    def boxes(self, frame: int) -> List[dict]:
        q = self.tracks[self.tracks.frame == frame]
        return [dict(box=[round(float(r.x0), 1), round(float(r.y0), 1), round(float(r.x1), 1), round(float(r.y1), 1)],
                     team=int(r.team), cls=str(r.cls), tid=int(r.track_id)) for _, r in q.iterrows()]

    def ball_at(self, frame: int):
        if self.ball is None or "u" not in self.ball:
            return None
        b = self.ball[(self.ball.frame == frame)]
        if not len(b) or not bool(b.get("valid", pd.Series([True])).iloc[0]) or not np.isfinite(b.u.iloc[0]):
            return None
        return [float(b.u.iloc[0]), float(b.v.iloc[0])]

    def draw(self, img: np.ndarray, frame: int, colors=((214, 120, 42), (59, 116, 232))) -> np.ndarray:
        import cv2
        out = img.copy()
        for x0, y0, x1, y1 in self.lines(frame):
            cv2.line(out, (int(x0), int(y0)), (int(x1), int(y1)), (255, 0, 255), 1, cv2.LINE_AA)
        for b in self.boxes(frame):
            if b["cls"] in ("outside", "spectator", "staff", "steward"):     # not players (football_eval/people.py)
                c = {"staff": (160, 160, 0), "steward": (0, 230, 255)}.get(b["cls"], (120, 80, 140))
            elif b["cls"] == "referee":
                c = (0, 0, 0)
            else:
                c = colors[b["team"]] if b["team"] in (0, 1) else (200, 200, 200)
            x0, y0, x1, y1 = map(int, b["box"])
            cv2.rectangle(out, (x0, y0), (x1, y1), c, 2 if b["cls"] == "goalkeeper" else 1)
        return out


def visible_segments(a: np.ndarray, b: np.ndarray, w: float, h: float, margin: float = 40.0) -> np.ndarray:
    """Segments with at least one end inside the image (+ margin): keeps overlays light."""
    ins = lambda p: (p[:, 0] > -margin) & (p[:, 0] < w + margin) & (p[:, 1] > -margin) & (p[:, 1] < h + margin)
    return ins(a) | ins(b)


def project_h(H: np.ndarray, pts: np.ndarray, centre=(52.5, 34.0)):
    """Project pitch points with a pitch->image homography; returns (uv, in-front mask)."""
    q = np.c_[pts, np.ones(len(pts))] @ np.asarray(H).T
    # the sign of H is arbitrary: the pitch centre must be in front of the camera
    c = np.array([[centre[0], centre[1], 1.0]]) @ np.asarray(H).T
    s = 1.0 if c[0, 2] >= 0 else -1.0
    w = q[:, 2] * s
    uv = q[:, :2] / np.where(np.abs(q[:, 2:3]) < 1e-9, 1e-9, q[:, 2:3])
    return uv, w > 1e-9
