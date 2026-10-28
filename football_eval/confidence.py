"""How far can each number in the report be trusted *for this recording*? Value x confidence grading.

Every team statistic gets

* a **value** rank: how much it tells a coach about the match
  (1 = headline: result, chances, territory, team shape; 2 = useful detail; 3 = context);
* a **confidence** level, computed for *this* match:

  =============  ===============================================================================================
  High           can be quoted as is (comes from the match sheet, or checked by you)
  Medium         right order of magnitude; good for comparing the two teams / halves of this match
  Low            indicative only; biased by the camera (partial view, 640 x 360) — mention with care
  Experimental   depends on signals that are not reliable on this footage yet (2-5 px ball) — do not use
  =============  ===============================================================================================

The starting level comes from the recording profile (one panning camera, 640 x 360, no score bug, no audio:
docs/RECORDING_PROFILE.md). It then moves up or down with evidence measured on the match (camera coverage,
ball coverage, your review, the match sheet) and every move is written in the ``reason`` column, so the grade is
explainable. Catalogue and rules: docs/REPORT_GUIDE.md.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

LEVELS = ["Experimental", "Low", "Medium", "High"]

# key: (label, group, value rank, base level, how to read it)
CATALOG = {
    "goals": ("Goals", "Result", 1, "Low", "final score"),
    "yellow_cards": ("Yellow cards", "Discipline", 2, "High", "from the match sheet"),
    "red_cards": ("Red cards", "Discipline", 1, "High", "from the match sheet"),
    "shots": ("Shots", "Chances", 1, "Low", "attempts on goal"),
    "shots_on_target": ("Shots on target", "Chances", 1, "Low", "saved or scored"),
    "xG": ("Expected goals (xG)", "Chances", 1, "Low", "quality of the chances created; compare with goals"),
    "big_chances(xG>=0.3)": ("Big chances (xG ≥ 0.3)", "Chances", 1, "Low", "clear-cut chances"),
    "opportunities_flagged": ("Flagged opportunities ⚑", "Chances", 1, "Low",
                              "goals, penalties, big chances, shots on target and shots with xG ≥ 0.10 — "
                              "listed with time and clip in the opportunities section"),
    "xG_per_shot": ("xG per shot", "Chances", 2, "Low", "average shot quality"),
    "shots_blocked": ("Shots blocked", "Chances", 3, "Low", ""),
    "def_line_height_m": ("Defensive line height (m)", "Team shape", 1, "Medium",
                          "distance of the last line from its own goal while in view: high press vs deep block"),
    "block_depth_m": ("Block depth (m)", "Team shape", 1, "Medium", "distance between the deepest and most advanced players: compactness front-to-back"),
    "block_width_m": ("Block width (m)", "Team shape", 1, "Medium", "horizontal spread of the team"),
    "team_centroid_x_m": ("Team centre of mass (m from own goal)", "Team shape", 2, "Medium", "where the team plays on average"),
    "possession_%": ("Possession %", "Ball", 1, "Low", "share of controlled-ball time"),
    "field_tilt_%": ("Field tilt %", "Ball", 1, "Low", "share of possession in the final third"),
    "ball_in_play_min": ("Ball in play (min)", "Ball", 3, "Low", ""),
    "passes": ("Passes", "Passing", 2, "Experimental", ""),
    "passes_completed": ("Passes completed", "Passing", 2, "Experimental", ""),
    "pass_accuracy_%": ("Pass accuracy %", "Passing", 2, "Experimental", ""),
    "progressive_passes": ("Progressive passes", "Passing", 2, "Experimental", ""),
    "final_third_entries_pass": ("Final-third entries (pass)", "Passing", 2, "Experimental", ""),
    "box_entries_pass": ("Box entries (pass)", "Passing", 3, "Experimental", ""),
    "crosses": ("Crosses", "Passing", 3, "Experimental", ""),
    "long_passes": ("Long passes", "Passing", 3, "Experimental", ""),
    "interceptions": ("Interceptions", "Defending", 2, "Experimental", ""),
    "ball_recoveries": ("Ball recoveries", "Defending", 2, "Experimental", ""),
    "PPDA": ("PPDA (pressing)", "Defending", 2, "Experimental", "opponent passes per defensive action; lower = more pressing"),
    "corners": ("Corners", "Set pieces", 2, "Low", ""),
    "throw_ins": ("Throw-ins", "Set pieces", 3, "Low", ""),
    "goal_kicks": ("Goal kicks", "Set pieces", 3, "Low", ""),
    "free_kicks_won": ("Free kicks won", "Set pieces", 3, "Low", ""),
    "penalties_won": ("Penalties won", "Set pieces", 2, "Low", ""),
    "fouls_committed(proxy)": ("Fouls committed (proxy)", "Discipline", 3, "Experimental", "inferred from free-kick restarts"),
    "offsides(suspected)": ("Offsides (suspected)", "Discipline", 3, "Experimental", ""),
    "distance_covered_km(tracked)": ("Distance covered while in view (km)", "Physical", 2, "Low",
                                     "only players in the camera view; compare teams, not absolute values"),
    "avg_speed_kmh": ("Average speed (km/h)", "Physical", 3, "Low", ""),
    "sprints(>7m/s,>=1s)": ("Sprints", "Physical", 3, "Experimental", ""),
}
BALL_BASED = {"possession_%", "field_tilt_%", "ball_in_play_min", "passes", "passes_completed", "pass_accuracy_%",
              "progressive_passes", "final_third_entries_pass", "box_entries_pass", "crosses", "long_passes",
              "interceptions", "ball_recoveries", "PPDA", "corners", "throw_ins", "goal_kicks", "free_kicks_won",
              "fouls_committed(proxy)", "offsides(suspected)", "penalties_won"}
POSITION_BASED = {"def_line_height_m", "block_depth_m", "block_width_m", "team_centroid_x_m",
                  "distance_covered_km(tracked)", "avg_speed_kmh", "sprints(>7m/s,>=1s)"}
DIRECTIONAL = {"def_line_height_m", "team_centroid_x_m", "field_tilt_%", "progressive_passes",
               "final_third_entries_pass", "PPDA", "xG", "shots"}
SHOT_BASED = {"shots", "shots_on_target", "xG", "big_chances(xG>=0.3)", "xG_per_shot", "shots_blocked",
              "opportunities_flagged"}


def _wilson_lo(k, n, z=1.96):
    if n == 0:
        return np.nan
    p = k / n
    den = 1 + z ** 2 / n
    return max(0.0, (p + z ** 2 / (2 * n) - z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2))) / den)


def data_quality(tracks, ball, fc, poss, ev, ev_auto, review: Optional[dict], setup: Optional[dict], ctx,
                 excluded_s: float = 0.0) -> dict:
    """Coverage and validation facts the grading uses (also printed at the top of the report)."""
    dq = {}
    in_per = np.zeros(len(ball), bool)
    for a, b in ctx.periods:
        in_per |= (ball.t >= a).to_numpy() & (ball.t <= b).to_numpy()
    b = ball[in_per] if in_per.any() else ball
    dq["minutes_analysed"] = float(b.t.max() - b.t.min()) / 60 if len(b) else 0.0
    dq["camera_coverage"] = float(b.cam_ok.astype(bool).mean()) if "cam_ok" in b and len(b) else 1.0
    ok = b.cam_ok.astype(bool) if "cam_ok" in b else pd.Series(True, index=b.index)
    vis = b["visible"].astype(bool) if "visible" in b else b["valid"].astype(bool)
    dq["ball_coverage"] = float(vis[ok].mean()) if ok.any() else 0.0
    pl = tracks[tracks.cls.isin(["player", "goalkeeper"]) & tracks.x.notna()]
    dq["players_with_team"] = float((pl.team >= 0).mean()) if len(pl) else 0.0
    dq["excluded_min"] = excluded_s / 60
    s = setup or {}
    prov = s.get("provenance", {})
    src = lambda k: prov.get(k, {}).get("source", "default")
    dq["pitch_manual"] = (s.get("pitch", {}).get("source") == "manual")
    dq["direction_manual"] = s.get("attacks_right_first_half") in (0, 1) and src("direction") == "manual"
    dq["direction_given"] = s.get("attacks_right_first_half") in (0, 1)
    dq["periods_given"] = len([p for p in s.get("periods", []) if p.get("start") is not None and p.get("end") is not None]) >= 1
    kits = s.get("kits", {})
    dq["kits_clicked"] = bool(kits.get("team0")) and bool(kits.get("team1"))
    fs = s.get("final_score")
    g = ev[ev.type == "goal"].groupby("team").size() if len(ev) else pd.Series(dtype=int)
    dq["final_score"] = fs
    dq["score_consistent"] = fs is not None and int(g.get(0, 0)) == fs[0] and int(g.get(1, 0)) == fs[1]
    dq["cards_from_sheet"] = any(e.get("type") in ("yellow", "red") for e in s.get("known_events", []))
    rv = review or {}
    fcks = [c for c in rv.get("frame_checks", []) if c.get("auto_ok") and c.get("lines") in ("good", "bad")]
    k = sum(c["lines"] == "good" for c in fcks)
    dq["calib_checked"] = (k, len(fcks))
    dq["calib_precision"] = k / len(fcks) if fcks else np.nan
    dq["calib_precision_lo"] = _wilson_lo(k, len(fcks))
    tck = [c for c in rv.get("frame_checks", []) if c.get("teams") in ("good", "bad")]
    dq["teams_checked"] = (sum(c["teams"] == "good" for c in tck), len(tck))
    cand = ev_auto[ev_auto.type.isin(["shot", "goal", "penalty"])] if len(ev_auto) else ev_auto
    verd = [e for e in rv.get("events", []) if e.get("verdict")]
    dq["key_events_detected"] = int(len(cand))
    dq["key_events_reviewed"] = len(verd)
    dq["shots_reviewed_share"] = (len([e for e in verd if e["type"] == "shot"]) / max(1, int((cand.type == "shot").sum()))
                                  if len(cand) else 1.0)
    sh = ev[ev.type == "shot"] if len(ev) else ev
    dq["shots_kept"] = int(len(sh))
    dq["shot_locations_clicked"] = float(sh.location.fillna("").str.startswith("clicked").mean()) \
        if len(sh) and "location" in sh else 0.0
    rej = [e for e in verd if e["type"] == "shot"]
    dq["shot_precision"] = (sum(e["verdict"] != "reject" for e in rej) / len(rej)) if rej else np.nan
    dq["missing_events_added"] = len(rv.get("added_events", []))
    dq["repairs"] = len([r for r in rv.get("repairs", []) if r.get("applied")])
    # camera view vs the near (camera-side) touchline: measured per frame from the calibration
    nv = b["vis_near_y"].dropna() if "vis_near_y" in b else pd.Series(dtype=float)
    W = ctx.pitch.width if getattr(ctx, "pitch", None) is not None else 68.0
    dq["near_hidden_share"] = float((nv < W - 0.5).mean()) if len(nv) else np.nan
    dq["near_hidden_m"] = float(W - nv.median()) if len(nv) else np.nan
    dq["near_flag"] = (s.get("pitch", {}) or {}).get("near_touchline")
    return dq


def _grade(key: str, dq: dict):
    label, group, value, base, read = CATALOG[key]
    lvl = LEVELS.index(base)
    why = []

    def move(to: int, reason: str):
        nonlocal lvl
        if to != lvl:
            why.append(reason)
            lvl = to

    cam, ball = dq["camera_coverage"], dq["ball_coverage"]
    if key == "goals":
        if dq["score_consistent"]:
            move(3, "matches the final score on the match sheet")
        else:
            why.append("inferred (no score overlay, no audio) — enter the final score and goal minutes"
                       if dq["final_score"] is None else "does NOT match the match sheet — list every goal with its minute")
    elif key in ("yellow_cards", "red_cards"):
        why.append("from the match sheet" if dq["cards_from_sheet"] else "match sheet has no cards — 0 means 'not entered'")
        if not dq["cards_from_sheet"]:
            move(1, "")
    elif key in SHOT_BASED:
        why.append("ball is 2-5 px: shots come from a weak ball signal")
        if dq["shots_reviewed_share"] >= 0.9 and dq["key_events_detected"]:
            move(2, f"every detected shot was checked by you ({dq['key_events_reviewed']} verdicts)")
            if key in ("xG", "big_chances(xG>=0.3)", "xG_per_shot") and dq["shot_locations_clicked"] < 0.8:
                move(1, f"but only {100 * dq['shot_locations_clicked']:.0f} % of shot locations were clicked: "
                        "xG still uses the tracked position")
            if not dq["missing_events_added"]:
                why.append("rules may still miss shots — add them under 'Missing events'")
        elif dq["key_events_detected"]:
            why.append(f"{100 * dq['shots_reviewed_share']:.0f} % of the shots reviewed — review all of them to reach Medium")
        if np.isfinite(dq.get("shot_precision", np.nan)) and dq["shot_precision"] < 0.5 and dq["shots_reviewed_share"] < 0.9:
            move(0, f"only {100 * dq['shot_precision']:.0f} % of the reviewed shots were real")
    elif key in POSITION_BASED:
        why.append(f"positions only while in view (camera usable {100 * cam:.0f} % of play)")
        if cam < 0.5:
            move(min(lvl, 1), "camera calibration usable on less than half of the play")
        pr, lo = dq["calib_precision"], dq["calib_precision_lo"]
        if np.isfinite(pr):
            n = dq["calib_checked"]
            if pr < 0.6:
                move(min(lvl, 1), f"your check: lines right on only {n[0]}/{n[1]} frames")
            else:
                why.append(f"your check: lines right on {n[0]}/{n[1]} frames")
        else:
            move(min(lvl, 1), "calibration not spot-checked yet (Medium once your frame check confirms the lines)")
        if key in DIRECTIONAL and not dq["direction_given"]:
            move(min(lvl, 1), "direction of play inferred, not given")
        if not dq["kits_clicked"]:
            move(min(lvl, 1), "teams from unsupervised colour clusters (no kit clicks)")
    elif key in BALL_BASED:
        why.append(f"needs the ball: a candidate in {100 * ball:.0f} % of usable frames, some false "
                   "(ball precision not measured)")
        if ball < 0.4 and lvl > 0:
            move(0, "ball seen in less than 40 % of frames")
        if cam < 0.5:
            move(min(lvl, 0), "camera usable on less than half of the play")
    nh = dq.get("near_hidden_share", np.nan)
    if key == "throw_ins" and nh is not None and np.isfinite(nh) and nh > 0.3:
        move(max(0, lvl - 1), f"near touchline out of view in {100 * nh:.0f} % of the frames: near-side throw-ins are "
                              "inferred from the ball leaving the picture, not seen")
    if key in DIRECTIONAL and key not in POSITION_BASED and not dq["direction_given"] and lvl > 1:
        move(1, "direction of play inferred, not given")
    return label, group, value, LEVELS[lvl], "; ".join(w for w in why if w), read


def grade_metrics(team_stats: pd.DataFrame, dq: dict) -> pd.DataFrame:
    """One row per statistic: values per team, value rank, confidence for this match, reason, how to read it."""
    rows = []
    teams = list(team_stats.columns)
    for key in team_stats.index:
        if key not in CATALOG:
            continue
        label, group, value, conf, why, read = _grade(key, dq)
        r = dict(metric=key, label=label, group=group, value_rank=value, confidence=conf, reason=why, how_to_read=read)
        for t in teams:
            r[t] = team_stats.at[key, t]
        rows.append(r)
    out = pd.DataFrame(rows)
    if len(out):
        out["_c"] = out.confidence.map({l: i for i, l in enumerate(LEVELS)})
        out = out.sort_values(["_c", "value_rank"], ascending=[False, True]).drop(columns="_c").reset_index(drop=True)
    return out


def quality_lines(dq: dict) -> list:
    """Plain-language data-quality summary for the top of the report."""
    L = []
    L.append(f"Camera calibration usable on {100 * dq['camera_coverage']:.0f} % of the analysed play "
             f"({dq['minutes_analysed']:.0f} min" + (f", {dq['excluded_min']:.1f} min excluded by you" if dq['excluded_min'] else "") + ").")
    L.append(f"Ball candidate in {100 * dq['ball_coverage']:.0f} % of usable frames. From one wide camera the ball is a few "
             "pixels and some candidates are false: ball-based statistics (possession, passes, set pieces) are Low or Experimental.")
    nh = dq.get("near_hidden_share")
    if nh is not None and np.isfinite(nh) and nh > 0.05:
        L.append(f"Near touchline (camera side) out of view in {100 * nh:.0f} % of the frames (the picture ends ~"
                 f"{dq.get('near_hidden_m', 0):.0f} m before it): balls leaving the picture there are treated as near-side "
                 "throw-ins, players there are not seen — near-side throw-ins and duels are under-counted.")
    k, n = dq["calib_checked"]
    L.append(f"Your frame check: pitch lines right on {k}/{n} frames." if n else
             "Calibration not spot-checked yet (review UI → Frame checks).")
    if dq["key_events_detected"]:
        L.append(f"Key events: {dq['key_events_reviewed']} of {dq['key_events_detected']} goals / shots / penalties reviewed; "
                 f"{100 * dq['shot_locations_clicked']:.0f} % of kept shots have a clicked location.")
    if dq["final_score"] is not None:
        L.append(f"Match sheet: final score {dq['final_score'][0]}-{dq['final_score'][1]} "
                 + ("(goals consistent)." if dq["score_consistent"] else "(goals NOT consistent — list every goal)."))
    else:
        L.append("No match sheet: goals are inferred only.")
    if dq["repairs"]:
        L.append(f"{dq['repairs']} camera stretch(es) repaired from your clicks.")
    return L
