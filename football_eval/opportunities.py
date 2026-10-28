"""Major attacking opportunities: every shot (and the moments around it) flagged for quick access.

A coach watching a 2-hour recording mostly wants to jump to the chances. This module turns the event table into
a short list of **opportunities**, each one an attacking moment of one team with a single timestamp, a video
clip window, a danger score and a ⚑ flag for the ones worth watching first.

Building blocks (rulebook §7c, thresholds in ``config.OpportunitiesCfg``):

=====================  =================================================================================
component              source
=====================  =================================================================================
goal                   ``goal`` events (shot rule, kick-off rule, review, match sheet)
shot                   ``shot`` events with their outcome and xG
penalty                ``penalty`` restarts and ``penalty_missed`` from the match sheet
box entry              a completed pass tagged ``box_entry`` or a carry (touch) that starts outside and ends
                       inside the opponent penalty area
set piece              corners, and free kicks within ``set_piece_max_dist`` m of the opponent goal
=====================  =================================================================================

Components of the same team closer than ``merge_window_s`` in time are one opportunity (a corner -> header ->
rebound shot is one chance, not three). The opportunity is classified by its strongest component:

    Goal > Penalty > Big chance (best shot xG >= big_chance_xg) > Shot on target > Shot
         > Box entry, no shot > Set piece near goal

and **flagged ⚑** when it is a goal, a penalty, a big chance, has a shot on target, or a shot with
xG >= ``flag_min_xg``. Chance value = probability that at least one of its shots is scored,
``1 - prod(1 - xG_i)`` (shots in the same move are not independent chances to add up).
"""
from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from .context import MatchContext
from .video import fmt_clock
from .xg import feature_row

CATEGORIES = ["Goal", "Penalty", "Big chance", "Shot on target", "Shot", "Box entry, no shot", "Set piece near goal"]
RANK = {c: i + 1 for i, c in enumerate(CATEGORIES)}
CHECKED = {"confirmed", "corrected", "match_sheet_confirmed", "manual", "match_sheet"}
OPP_COLS = ["opp_id", "t", "video_time", "clock", "team", "team_name", "category", "rank", "flag", "danger",
            "chance_xg", "best_xg", "n_shots", "n_on_target", "outcome", "player", "build_up", "description",
            "checked", "components", "clip_start", "clip_end", "clip_file", "t_first", "t_last"]


def _num(v) -> Optional[float]:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return None if np.isnan(f) else f


def _player(e) -> str:
    n = _num(e.get("number"))
    if n is None:
        return ""
    name = e.get("player")
    return f"#{int(n)}" + (f" {name}" if isinstance(name, str) and name else "")


def _components(ev: pd.DataFrame, ctx: MatchContext, cfg, xgm=None, touches: Optional[pd.DataFrame] = None) -> list:
    oc = cfg.opportunities
    pitch = ctx.pitch
    L, W = pitch.length, pitch.width
    comps = []
    if ev is None or len(ev) == 0:
        return comps

    def potential_xg(team, t, x, y):
        """Basic (distance + angle) xG from a location: how dangerous a shot from there would have been."""
        if xgm is None or x is None or y is None:
            return np.nan
        ax, ay = ctx.to_attack_frame(int(team), t, x, y)
        return float(xgm.predict([feature_row(float(ax), float(ay), None, L, W)], "basic")[0])

    for i, e in ev.iterrows():
        typ, team = e.type, e.team
        if pd.isna(team) or int(team) not in (0, 1):
            continue
        team = int(team)
        base = dict(t=float(e.t), team=team, idx=i, validation=str(e.get("validation", "") or ""),
                    player=_player(e), why=str(e.get("why", "") or ""))
        if typ == "goal":
            comps.append(dict(base, kind="goal", xg=_num(e.get("xg")), outcome="goal", source=str(e.get("source", ""))))
        elif typ == "shot":
            sit = str(e.get("detail", "")).split("|")[0]
            m = re.search(r"dist=([\d.]+)", str(e.get("detail", "")))
            comps.append(dict(base, kind="shot", xg=_num(e.get("xg")), outcome=str(e.outcome), situation=sit,
                              dist=float(m.group(1)) if m else None))
        elif typ == "penalty":
            comps.append(dict(base, kind="penalty", xg=float(xgm.p.get("penalty_xg", 0.76)) if xgm else 0.76,
                              outcome="penalty awarded"))
        elif typ == "penalty_missed":
            comps.append(dict(base, kind="penalty", xg=float(xgm.p.get("penalty_xg", 0.76)) if xgm else 0.76,
                              outcome="penalty missed"))
        elif oc.include_box_entries and typ == "pass" and "box_entry" in str(e.get("detail", "")) \
                and e.outcome == "complete":
            comps.append(dict(base, kind="box_entry", via="cross" if "cross" in str(e.detail) else "pass",
                              xg=potential_xg(team, e.t, _num(e.x_end), _num(e.y_end))))
        elif oc.include_set_pieces and typ == "corner":
            comps.append(dict(base, kind="set_piece", via="corner", xg=oc.corner_xg_prior))
        elif oc.include_set_pieces and typ == "free_kick" and _num(e.x) is not None:
            ax, ay = ctx.to_attack_frame(team, e.t, e.x, e.y)
            d = float(np.hypot(L - ax, W / 2 - ay))
            if d <= oc.set_piece_max_dist:
                pxg = potential_xg(team, e.t, e.x, e.y)
                comps.append(dict(base, kind="set_piece", via="free kick", dist=d,
                                  xg=pxg * float(xgm.p.get("dfk_factor", 1.0)) if xgm and not np.isnan(pxg)
                                  else oc.free_kick_xg_prior))
    # carries into the box (the pass rule only sees entries by a pass)
    if oc.include_box_entries and touches is not None and len(touches):
        tc = touches[touches.team.isin([0, 1])]
        for _, r in tc.iterrows():
            x0, y0 = ctx.to_attack_frame(int(r.team), r.t0, r.x0, r.y0)
            x1, y1 = ctx.to_attack_frame(int(r.team), r.t1, r.x1, r.y1)
            if pitch.in_penalty_area(x1, y1, True) and not pitch.in_penalty_area(x0, y0, True) \
                    and np.hypot(x1 - x0, y1 - y0) >= oc.carry_min_m:
                comps.append(dict(t=float(r.t1), team=int(r.team), idx=None, validation="", player="", why="",
                                  kind="box_entry", via="carry", xg=potential_xg(int(r.team), r.t1, r.x1, r.y1)))
    comps.sort(key=lambda c: c["t"])
    return comps


def _group(comps: list, window: float) -> list:
    """Same-team components closer than ``window`` seconds -> one opportunity. A component of the other team in
    between closes the group (the ball changed sides)."""
    groups, cur = [], None
    for c in comps:
        if cur is not None and c["team"] == cur[-1]["team"] and c["t"] - cur[-1]["t"] <= window:
            cur.append(c)
            continue
        cur = [c]
        groups.append(cur)
    return groups


def _summarise(g: list, ctx: MatchContext, cfg, video_name: str = "") -> dict:
    oc = cfg.opportunities
    team = g[0]["team"]
    shots = [c for c in g if c["kind"] == "shot"]
    goals = [c for c in g if c["kind"] == "goal"]
    pens = [c for c in g if c["kind"] == "penalty"]
    entries = [c for c in g if c["kind"] == "box_entry"]
    sets = [c for c in g if c["kind"] == "set_piece"]
    xgs = [c["xg"] for c in shots + pens if c.get("xg") is not None and not np.isnan(c["xg"])]
    if not shots and goals and goals[0].get("xg") is not None:      # a goal from the sheet / kick-off rule
        xgs = [goals[0]["xg"]]
    best = max(xgs) if xgs else np.nan
    chance = float(1 - np.prod([1 - x for x in xgs])) if xgs else np.nan
    on_target = [c for c in shots if c["outcome"] in ("goal", "saved")]
    pen_scored = any(c["outcome"] == "penalty awarded" for c in pens) and bool(goals)
    if goals or pen_scored:
        cat = "Goal"
    elif pens:
        cat = "Penalty"
    elif shots and not np.isnan(best) and best >= oc.big_chance_xg:
        cat = "Big chance"
    elif on_target:
        cat = "Shot on target"
    elif shots:
        cat = "Shot"
    elif entries:
        cat = "Box entry, no shot"
    else:
        cat = "Set piece near goal"
    flag = cat in ("Goal", "Penalty", "Big chance", "Shot on target") or \
        (bool(shots) and not np.isnan(best) and best >= oc.flag_min_xg)
    # danger score 0-100: the chance value for shots, a discounted potential for moments without a shot
    if cat == "Goal":
        danger = 100.0
    elif shots or pens:
        danger = 100 * (chance if not np.isnan(chance) else 0.05) + (10 if on_target else 0)
    elif entries:
        px = np.nanmax([c["xg"] for c in entries]) if any(not np.isnan(c["xg"]) for c in entries) else 0.05
        danger = 100 * oc.no_shot_discount * px
    else:
        danger = 100 * max(c["xg"] for c in sets)
    danger = float(min(100.0, danger))
    # key time: first shot / goal, else first component
    key = (shots or goals or pens or g)[0]
    t_first, t_last = g[0]["t"], g[-1]["t"]
    names = ctx.team_names
    tname = names[team] if 0 <= team < len(names) else "?"
    # outcome text
    if shots:
        oc_txt = ", ".join(_outcome_word(c["outcome"]) for c in shots)
    elif goals:
        oc_txt = "goal" + (" (match sheet)" if goals[0].get("source") == "match_sheet" else "")
    elif pens:
        oc_txt = pens[0]["outcome"]
    elif entries:
        oc_txt = "no shot"
    else:
        oc_txt = "set piece"
    build = []
    for c in sets:
        build.append(c["via"])
    for c in entries:
        build.append({"cross": "cross into the box", "pass": "pass into the box", "carry": "carry into the box"}[c["via"]])
    build = list(dict.fromkeys(build))
    player = next((c["player"] for c in shots + goals + pens if c.get("player")), "")
    # description
    if shots:
        d0 = next((c.get("dist") for c in shots if c.get("dist")), None)
        what = (f"{len(shots)} shots ({len(on_target)} on target)" if len(shots) > 1 else "shot") + \
            (f" from {d0:.0f} m" if d0 else "")
        what += f", {oc_txt}" + (f" — best xG {best:.2f}" if not np.isnan(best) else "")
    elif goals:
        what = "goal" + (f" (xG {best:.2f})" if not np.isnan(best) else "")
    elif pens:
        what = pens[0]["outcome"]
    elif entries:
        px = [c["xg"] for c in entries if not np.isnan(c["xg"])]
        what = f"reached the box, no shot" + (f" (a shot from there ≈ xG {max(px):.2f})" if px else "")
    else:
        s = sets[0]
        what = s["via"] + (f" {s['dist']:.0f} m from goal" if s.get("dist") else "")
    desc = f"{cat} — {tname}" + (f" {player}" if player else "") + f": {what}" + \
        (f" (after a {' / '.join(build)})" if build and (shots or goals) else "")
    checked = any(c["validation"] in CHECKED for c in shots + goals + pens)
    clip_start = max(0.0, t_first - oc.clip_before_s)
    clip_end = t_last + oc.clip_after_s
    return dict(t=round(float(key["t"]), 2), video_time=fmt_clock(key["t"]), clock=ctx.match_clock(key["t"]), team=team,
                team_name=tname, category=cat, rank=RANK[cat], flag=bool(flag), danger=round(danger, 1),
                chance_xg=round(chance, 3) if not np.isnan(chance) else np.nan,
                best_xg=round(best, 3) if not np.isnan(best) else np.nan, n_shots=len(shots),
                n_on_target=len(on_target), outcome=oc_txt, player=player, build_up=" / ".join(build),
                description=desc, checked=checked,
                components="|".join(c["kind"] + (":" + c["via"] if c.get("via") else "") for c in g),
                clip_start=round(clip_start, 1), clip_end=round(clip_end, 1), t_first=round(t_first, 2),
                t_last=round(t_last, 2))


def _outcome_word(o: str) -> str:
    return {"goal": "goal", "saved": "saved", "blocked": "blocked", "off_target": "off target",
            "blocked_or_saved_out": "out for a corner", "unresolved": "outcome unclear"}.get(o, o)


def build_opportunities(ev: pd.DataFrame, ctx: MatchContext, cfg, xgm=None,
                        touches: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """One row per attacking opportunity, in time order, with the ⚑ flag, a danger score and a clip window."""
    comps = _components(ev, ctx, cfg, xgm, touches)
    groups = _group(comps, cfg.opportunities.merge_window_s)
    rows = [_summarise(g, ctx, cfg) for g in groups]
    if not rows:
        return pd.DataFrame(columns=OPP_COLS)
    df = pd.DataFrame(rows).sort_values("t").reset_index(drop=True)
    df.insert(0, "opp_id", np.arange(1, len(df) + 1))
    df["clip_file"] = [_clip_name(r) for _, r in df.iterrows()]
    return df[OPP_COLS]


def _clip_name(r) -> str:
    cat = re.sub(r"[^a-z]+", "_", r.category.lower()).strip("_")
    team = re.sub(r"[^A-Za-z0-9]+", "", str(r.team_name))[:16]
    return f"{int(r.opp_id):03d}_{r.video_time.replace(':', '-')}_{team}_{cat}.mp4"


def opportunity_summary(opps: pd.DataFrame, team_names: list) -> pd.DataFrame:
    """Per team: opportunities, flagged, by category, chance value."""
    rows = {}
    for tm, name in enumerate(team_names[:2]):
        q = opps[opps.team == tm] if len(opps) else opps
        r = {"Opportunities": len(q), "Flagged ⚑": int(q.flag.sum()) if len(q) else 0}
        for c in CATEGORIES:
            r[c] = int((q.category == c).sum()) if len(q) else 0
        r["Chance value (Σ xG of moves)"] = round(float(q.chance_xg.fillna(0).sum()), 2) if len(q) else 0.0
        rows[name] = r
    return pd.DataFrame(rows)


def save_opportunities(opps: pd.DataFrame, out_dir: str | Path, video_path: Optional[str] = None,
                       only_flagged: bool = True) -> dict:
    """``opportunities.csv`` / ``.json`` and ``cut_opportunities.sh`` (ffmpeg, stream copy)."""
    out = Path(out_dir)
    opps.to_csv(out / "opportunities.csv", index=False)
    with open(out / "opportunities.json", "w") as f:
        json.dump(json.loads(opps.to_json(orient="records")), f, indent=1, ensure_ascii=False)
    paths = {"csv": out / "opportunities.csv"}
    if video_path:
        sel = opps[opps.flag] if only_flagged else opps
        lines = ["#!/usr/bin/env bash", "# one clip per flagged opportunity (re-run with --all in "
                 "scripts/cut_opportunities.py for every opportunity)", "set -e", f'VIDEO="{video_path}"',
                 'D="$(dirname "$0")/clips/opportunities"', 'mkdir -p "$D"']
        for _, r in sel.iterrows():
            lines.append(f'ffmpeg -loglevel error -y -ss {r.clip_start:.1f} -to {r.clip_end:.1f} -i "$VIDEO" '
                         f'-c copy "$D/{r.clip_file}"')
        p = out / "cut_opportunities.sh"
        p.write_text("\n".join(lines) + "\n")
        p.chmod(0o755)
        paths["script"] = p
    return paths


def cut_clips(opps: pd.DataFrame, video_path: str, out_dir: str | Path, only_flagged: bool = True,
              overwrite: bool = False, log=print) -> list:
    """Cut one clip per opportunity into ``<out_dir>/clips/opportunities``. Uses ffmpeg when installed (fast,
    stream copy), otherwise OpenCV (re-encodes, slower but needs nothing else)."""
    d = Path(out_dir) / "clips" / "opportunities"
    d.mkdir(parents=True, exist_ok=True)
    sel = opps[opps.flag] if only_flagged else opps
    ff = shutil.which("ffmpeg")
    done = []
    for _, r in sel.iterrows():
        p = d / r.clip_file
        if p.exists() and not overwrite:
            done.append(p)
            continue
        if ff:
            # -ss before -i seeks to the keyframe before the start (instant); re-encoding is not needed for review
            cmd = [ff, "-loglevel", "error", "-y", "-ss", f"{r.clip_start:.1f}", "-to", f"{r.clip_end:.1f}",
                   "-i", str(video_path), "-c", "copy", "-movflags", "+faststart", str(p)]
            res = subprocess.run(cmd, capture_output=True, text=True)
            if res.returncode != 0:
                log(f"  ffmpeg failed on {r.clip_file}: {res.stderr.strip()[:200]}")
                continue
        else:
            if not _cut_cv2(video_path, r.clip_start, r.clip_end, p):
                log(f"  could not write {r.clip_file}")
                continue
        done.append(p)
        log(f"  clip {len(done)}/{len(sel)}  {r.video_time}  {r.description[:70]}")
    return done


def link_clips_in_report(run_dir: str | Path) -> int:
    """Turn the clip placeholders of an existing report.html into ▶ links for the clips that now exist."""
    run = Path(run_dir)
    rep = run / "report.html"
    if not rep.exists():
        return 0
    html = rep.read_text()
    n = 0
    for m in set(re.findall(r"<span class='cliplink' data-clip='([^']+)'></span>", html)):
        if (run / "clips" / "opportunities" / m).exists():
            html = html.replace(f"<span class='cliplink' data-clip='{m}'></span>",
                                f"<a href='clips/opportunities/{m}' target='_blank'>▶ clip</a> · ")
            n += 1
    rep.write_text(html)
    return n


def _cut_cv2(video_path: str, t0: float, t1: float, out: Path) -> bool:
    import cv2
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return False
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    wr = None
    for cc in ("avc1", "mp4v"):          # avc1 (H.264) plays in every browser; mp4v is the portable fallback
        wr = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*cc), fps, (w, h))
        if wr.isOpened():
            break
    if wr is None or not wr.isOpened():
        return False
    cap.set(cv2.CAP_PROP_POS_MSEC, t0 * 1000)
    n = int((t1 - t0) * fps)
    for _ in range(n):
        ok, img = cap.read()
        if not ok:
            break
        wr.write(img)
    wr.release()
    cap.release()
    return out.exists() and out.stat().st_size > 0


def evaluate_opportunities(opps: pd.DataFrame, gt: pd.DataFrame, tol_s: float = 3.0,
                           match_minutes: Optional[float] = None) -> dict:
    """Against ground-truth events (synthetic match or a hand-annotated CSV ``t, type, team, outcome``):

    * ``shot_recall``      — true shots inside some opportunity of the same team (time window ± tol)
    * ``goal_recall_flag`` — true goals inside a *flagged* opportunity (must be 1.0)
    * ``on_target_recall_flag`` — true shots on target (goal / saved) inside a flagged opportunity
    * ``flag_precision``   — flagged opportunities that contain a true shot or goal of that team
    * ``watch_min``        — minutes of video to watch to see every flagged clip (overlaps counted once)
    """
    def covered(rows, pool):
        hit = 0
        for _, g in rows.iterrows():
            q = pool[(pool.team == g.team) & (pool.t_first - tol_s <= g.t) & (g.t <= pool.t_last + tol_s)]
            hit += bool(len(q))
        return hit
    shots = gt[gt.type == "shot"]
    goals = gt[gt.type == "goal"]
    ont = shots[shots.outcome.isin(["goal", "saved"])]
    fl = opps[opps.flag] if len(opps) else opps
    real = 0
    for _, o in fl.iterrows():
        q = gt[gt.type.isin(["shot", "goal"]) & (gt.team == o.team) & (gt.t >= o.t_first - tol_s) & (gt.t <= o.t_last + tol_s)]
        real += bool(len(q))
    iv = sorted(zip(fl.clip_start, fl.clip_end)) if len(fl) else []
    watch, cur = 0.0, None
    for a, b in iv:
        if cur is None or a > cur[1]:
            if cur:
                watch += cur[1] - cur[0]
            cur = [a, b]
        else:
            cur[1] = max(cur[1], b)
    if cur:
        watch += cur[1] - cur[0]
    nan = float("nan")
    out = dict(n_opps=len(opps), n_flagged=len(fl), gt_shots=len(shots), gt_goals=len(goals),
               shot_recall=covered(shots, opps) / len(shots) if len(shots) else nan,
               goal_recall_flag=covered(goals, fl) / len(goals) if len(goals) else nan,
               on_target_recall_flag=covered(ont, fl) / len(ont) if len(ont) else nan,
               flag_precision=real / len(fl) if len(fl) else nan, watch_min=watch / 60)
    if match_minutes:
        out["watch_share"] = out["watch_min"] / match_minutes
    return out


def opportunity_plot(opps: pd.DataFrame, ctx: MatchContext):
    """Opportunities over the match: team A above the axis, team B below; height = danger, filled = flagged."""
    import matplotlib.pyplot as plt
    from .report import TEAM_COLORS
    fig, ax = plt.subplots(figsize=(11, 3.4))
    for tm, sgn in ((0, 1), (1, -1)):
        q = opps[opps.team == tm] if len(opps) else opps
        if len(q) == 0:
            continue
        x = q.t.to_numpy() / 60
        y = sgn * q.danger.clip(lower=4).to_numpy()
        ax.vlines(x, 0, y, color=TEAM_COLORS[tm], lw=1.4, alpha=0.7)
        f = q.flag.to_numpy()
        ax.scatter(x[f], y[f], s=55, color=TEAM_COLORS[tm], zorder=3, edgecolors="black", linewidths=0.6)
        ax.scatter(x[~f], y[~f], s=30, facecolors="white", edgecolors=TEAM_COLORS[tm], zorder=3)
        for _, r in q[q.category == "Goal"].iterrows():
            ax.annotate("GOAL", (r.t / 60, sgn * r.danger), xytext=(0, 6 * sgn), textcoords="offset points",
                        ha="center", va="bottom" if sgn > 0 else "top", fontsize=8, fontweight="bold")
    for a, b in ctx.periods:
        ax.axvspan(a / 60, b / 60, color="#000", alpha=0.03, lw=0)
    ax.axhline(0, color="#666", lw=0.8)
    ax.set_ylim(-118, 118)
    ax.set_yticks([-100, -50, 0, 50, 100])
    ax.set_yticklabels(["100", "50", "0", "50", "100"])
    ax.set_ylabel(f"← {ctx.team_names[1]}   danger   {ctx.team_names[0]} →", fontsize=8)
    ax.set_xlabel("video minute (shaded = periods)")
    ax.set_title("Opportunities over the match (height = danger 0-100, filled = flagged ⚑)", fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    return fig
