"""Analytics pipeline: tracking table -> possession, events, xG, stats, timeline, report."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from .config import Config
from .context import infer_periods, player_speeds
from .events import EventDetector, dead_ball
from .explain import Provenance, explain_events
from .review import apply_exclusions, apply_review, validation_summary
from .pitch import Pitch
from .possession import frame_control, team_possession, touches
from .stats import numbered_player_stats, player_stats, possession_timeline, team_stats
from .teams import assign_goalkeepers
from .timeline import build_timeline
from .opportunities import build_opportunities, opportunity_summary
from .xg import XGModel


def run_analytics(tracks: pd.DataFrame, ball: pd.DataFrame, cfg: Config, scoreboard_changes: Optional[list] = None,
                  root: Optional[str] = None, setup: Optional[dict] = None, review: Optional[dict] = None,
                  prov: Optional[Provenance] = None) -> dict:
    """``setup`` (match_setup.json) and ``review`` (review.json) are the human inputs: they are applied after the
    automatic event detection (verdicts, match-sheet reconciliation) and logged in the provenance ledger."""
    prov = prov if prov is not None else Provenance()
    pitch = Pitch(cfg.pitch.length, cfg.pitch.width)
    tracks = tracks.copy()
    exclusions = (setup or {}).get("exclusions") or []
    tracks, ball, excl_s = apply_exclusions(tracks, ball, exclusions)
    if exclusions:
        prov.add("setup", "excluded stretches", f"{len(exclusions)} ({excl_s / 60:.1f} min)", "manual",
                 note="; ".join(f"{e.get('reason') or 'excluded'} {int(e['t0'] // 60)}:{int(e['t0'] % 60):02d}"
                                f"-{int(e['t1'] // 60)}:{int(e['t1'] % 60):02d}" for e in exclusions[:8]))
    ctx = infer_periods(tracks, pitch, cfg.match.halves, cfg.match.home_attacks_right_first_half)
    ctx.team_names = list(cfg.teams.team_names)
    # goalkeepers labelled 'other' by the colour clustering -> attach by pitch side, per period
    if (tracks.cls == "goalkeeper").sum() == 0:
        for i, (a, b) in enumerate(ctx.periods):
            sel = (tracks.t >= a) & (tracks.t <= b)
            if not sel.any():                             # e.g. a segment run that covers one period only
                continue
            sub = tracks[sel].copy()
            defends_left = dict(ctx.attacks_right[i])     # attacking right <=> defending the left goal
            sub = assign_goalkeepers(sub, pitch, defends_left)
            tracks.loc[sel, ["team", "cls"]] = sub[["team", "cls"]].to_numpy()
    tracks["team"] = tracks.team.astype(int)
    tr_sp = player_speeds(tracks)
    fc = frame_control(tracks, ball, cfg)
    dead, stoppages = dead_ball(fc, tr_sp, ctx, cfg)
    fc.loc[dead, ["ctrl_tid", "ctrl_team"]] = -1
    tch = touches(fc, tracks, cfg.possession.min_control_s)
    poss = team_possession(fc, dead, cfg.possession.team_switch_s)
    if "cam_ok" in fc:                                   # moving camera lost: possession unknown, not dead
        unknown = ~fc.cam_ok.fillna(True).astype(bool).to_numpy()
        poss[unknown] = -1
        dead = dead & ~unknown
    from .config import resolve_path
    xg_path = Path(root) / cfg.xg.model_file if root else Path(resolve_path(cfg.xg.model_file))
    xgm = XGModel.load(xg_path) if xg_path.exists() else None
    ev_auto = EventDetector(cfg, ctx, xgm).run(fc, tch, tracks, stoppages, scoreboard_changes)
    ev_auto["clock"] = [ctx.match_clock(t) for t in ev_auto.t]
    ev_auto = explain_events(ev_auto, cfg, ctx.team_names)
    # ---- human inputs: review verdicts + match sheet
    ev, rejected = apply_review(ev_auto, review, setup, ctx, xgm, tracks)
    ev = explain_events(ev, cfg, ctx.team_names)
    if len(rejected):
        rejected = explain_events(rejected, cfg, ctx.team_names)
    _log_analytics(prov, ctx, cfg, ev_auto, ev, rejected, review, setup)
    tstats = team_stats(ev, fc, poss, dead, tr_sp, ctx, cfg)
    pstats = player_stats(ev, tr_sp, tch)
    by_number = numbered_player_stats(ev, ctx)
    tl = build_timeline(ev, ctx, final_third_x=cfg.events.final_third_x)
    opps = build_opportunities(ev, ctx, cfg, xgm, tch)
    if len(tstats.columns) >= 2:
        tstats.loc["opportunities_flagged"] = [int(opps[opps.team == k].flag.sum()) if len(opps) else 0
                                               for k in range(len(tstats.columns))]
    pt = possession_timeline(fc, poss, dead)
    vsum = validation_summary(review) if review else None
    from .confidence import grade_metrics, data_quality
    dq = data_quality(tracks, ball, fc, poss, ev, ev_auto, review, setup, ctx, excl_s)
    key = grade_metrics(tstats, dq)
    return dict(ctx=ctx, tracks=tracks, tracks_sp=tr_sp, fc=fc, dead=dead, stoppages=stoppages, touches=tch,
                possession=poss, events=ev, events_auto=ev_auto, events_rejected=rejected, team_stats=tstats,
                player_stats=pstats, player_numbers=by_number, timeline=tl, possession_timeline=pt,
                opportunities=opps, opportunity_summary=opportunity_summary(opps, ctx.team_names),
                opportunities_cfg=cfg.opportunities,
                provenance=prov, validation=vsum, setup=setup, data_quality=dq, key_metrics=key)


def _log_analytics(prov, ctx, cfg, ev_auto, ev, rejected, review, setup):
    n_auto = ev_auto.type.value_counts()
    key = [t for t in ("goal", "shot", "penalty") if t in n_auto.index or t in set(ev.type)]
    for t in key:
        n_a = int(n_auto.get(t, 0))
        q = ev[ev.type == t]
        vc = q.validation.value_counts() if "validation" in q else pd.Series(dtype=int)
        checked = int(vc.get("confirmed", 0) + vc.get("corrected", 0) + vc.get("match_sheet_confirmed", 0))
        added = int(vc.get("manual", 0) + vc.get("match_sheet", 0))
        rej = int((rejected.type == t).sum()) if len(rejected) else 0
        src = "measured" if (checked or rej or added) else "auto"
        prov.add("events", f"{t}s", f"{len(q)} kept of {n_a} detected", src,
                 note=f"{checked} confirmed/corrected, {rej} rejected, {added} added by hand / match sheet, "
                      f"{len(q) - checked - added} unreviewed")
    if setup is not None and setup.get("final_score"):
        fs = setup["final_score"]
        g = ev[ev.type == "goal"].groupby("team").size()
        ok = int(g.get(0, 0)) == fs[0] and int(g.get(1, 0)) == fs[1]
        prov.add("events", "goals vs final score", f"{int(g.get(0, 0))}-{int(g.get(1, 0))} vs sheet {fs[0]}-{fs[1]}",
                 "measured", quality="consistent" if ok else "INCONSISTENT — list every goal on the match sheet")
    if review:
        prov.add("review", "frames spot-checked", sum(1 for c in review.get("frame_checks", []) if c.get("lines")),
                 "manual", note="see the validation table")


def save_analytics(res: dict, out_dir: str, make_report: bool = True, video_path: Optional[str] = None,
                   title: str = "Match report") -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    res["events"].to_csv(out / "events.csv", index=False)
    if res.get("events_rejected") is not None and len(res["events_rejected"]):
        res["events_rejected"].to_csv(out / "events_rejected.csv", index=False)
    elif (out / "events_rejected.csv").exists():
        (out / "events_rejected.csv").unlink()                  # stale file from an earlier run
    res["team_stats"].to_csv(out / "team_stats.csv")
    if res.get("key_metrics") is not None and len(res["key_metrics"]):
        res["key_metrics"].to_csv(out / "key_metrics.csv", index=False)
    if res.get("data_quality") is not None:
        with open(out / "data_quality.json", "w") as f:
            clean = lambda v: None if isinstance(v, float) and np.isnan(v) else (v.item() if hasattr(v, "item") else v)
            json.dump({k: clean(v) for k, v in res["data_quality"].items()}, f, indent=1, default=str)
    res["player_stats"].to_csv(out / "player_stats.csv")
    if res.get("player_numbers") is not None and len(res["player_numbers"]):
        res["player_numbers"].to_csv(out / "player_stats_by_number.csv", index=False)
    res["timeline"].to_csv(out / "timeline.csv", index=False)
    res["stoppages"].to_csv(out / "stoppages.csv", index=False)
    with open(out / "timeline.json", "w") as f:
        json.dump(res["timeline"].to_dict(orient="records"), f, indent=2, default=str)
    ctx = res["ctx"]
    with open(out / "match_context.json", "w") as f:
        json.dump({"periods": ctx.periods, "attacks_right": [{str(k): v for k, v in d.items()} for d in ctx.attacks_right],
                   "team_names": ctx.team_names}, f, indent=2)
    if res.get("provenance") is not None:
        res["provenance"].save(out / "provenance.json")
    if res.get("validation") is not None:
        res["validation"].to_csv(out / "validation_summary.csv", index=False)
    opps = res.get("opportunities")
    clips_dir = None
    if opps is not None:
        from .opportunities import cut_clips, save_opportunities
        save_opportunities(opps, out, video_path)
        oc = res.get("opportunities_cfg")
        if video_path and oc is not None and oc.auto_clips and len(opps) and opps.flag.any():
            print(f"Cutting {int(opps.flag.sum())} clips of flagged opportunities -> {out / 'clips' / 'opportunities'}")
            cut_clips(opps, video_path, out, only_flagged=True, log=lambda m: None)
        if (out / "clips" / "opportunities").exists():
            clips_dir = out / "clips" / "opportunities"
    if video_path:
        from .timeline import export_highlights_script
        export_highlights_script(res["timeline"], video_path, str(out))
    if make_report:
        from .report import heatmaps, html_report, possession_plot, shot_map
        figs = {}
        if opps is not None and len(opps):
            from .opportunities import opportunity_plot
            figs["Opportunities over the match"] = opportunity_plot(opps, ctx)
        figs.update({"Shot map": shot_map(res["events"], ctx), "Possession": possession_plot(res["possession_timeline"], ctx,
                                                                                       res["events"]),
                     "Average positions": heatmaps(res["tracks"], ctx)})
        from .confidence import quality_lines
        dq = res.get("data_quality")
        names = ctx.team_names
        g = res["events"][res["events"].type == "goal"].groupby("team").size()
        score = f"{names[0]} {int(g.get(0, 0))} – {int(g.get(1, 0))} {names[1]}"
        if not (dq and dq.get("score_consistent")):
            score += " <span class='note'>(goals inferred — not confirmed by a match sheet)</span>"
        return html_report(str(out), res["team_stats"], res["timeline"], figs, title,
                           notes="Single-camera analysis with manual validation.",
                           provenance=res.get("provenance"), validation=res.get("validation"),
                           events=res["events"], rejected=res.get("events_rejected"),
                           numbers=res.get("player_numbers"), key_metrics=res.get("key_metrics"),
                           quality=quality_lines(dq) if dq else None, score=score,
                           opportunities=opps, opp_summary=res.get("opportunity_summary"),
                           video_path=video_path, clips_dir=clips_dir)
    return out
