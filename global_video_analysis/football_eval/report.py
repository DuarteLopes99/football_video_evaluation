"""Figures + a self-contained HTML match report."""
from __future__ import annotations

import base64
import io
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from .context import MatchContext  # noqa: E402
from .pitch import draw_pitch  # noqa: E402

TEAM_COLORS = ["#2a78d6", "#e8743b"]


def _b64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def shot_map(ev: pd.DataFrame, ctx: MatchContext, ax=None):
    """All shots drawn towards the goal each team attacked (team A -> right, team B -> left)."""
    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=(9, 6))
    draw_pitch(ax, ctx.pitch)
    L, W = ctx.pitch.length, ctx.pitch.width
    for tm in (0, 1):
        sh = ev[(ev.type == "shot") & (ev.team == tm)]
        for _, s in sh.iterrows():
            x, y = ctx.to_attack_frame(tm, s.t, s.x, s.y)
            if tm == 1:
                x, y = L - x, W - y
            goal = s.outcome == "goal"
            ax.scatter(x, y, s=60 + 900 * (s.xg if pd.notna(s.xg) else 0.05), c=TEAM_COLORS[tm],
                       alpha=0.9 if goal else 0.45, edgecolors="black" if goal else "none", linewidths=1.5, zorder=3)
    ax.set_title("Shot map (size = xG, outlined = goal)  " +
                 f"{ctx.team_names[0]} →   ← {ctx.team_names[1]}", fontsize=10)
    return fig or ax.figure


def heatmaps(tracks: pd.DataFrame, ctx: MatchContext):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    L, W = ctx.pitch.length, ctx.pitch.width
    for tm, ax in zip((0, 1), axes):
        q = tracks[(tracks.team == tm) & tracks.x.notna() & (tracks.cls == "player")]   # outfield only
        xs, ys = [], []
        for (tt), g in q.groupby("t"):
            x, y = ctx.to_attack_frame(tm, tt, g.x.to_numpy(), g.y.to_numpy())
            xs.append(x), ys.append(y)
        draw_pitch(ax, ctx.pitch, color="#888888")
        if xs:
            x, y = np.concatenate(xs), np.concatenate(ys)
            h, xe, ye = np.histogram2d(x, y, bins=[21, 14], range=[[0, L], [0, W]])
            ax.imshow(np.sqrt(h.T), extent=[0, L, W, 0], cmap="Blues" if tm == 0 else "Oranges", alpha=0.8, aspect="auto",
                      zorder=0)
        ax.set_title(f"{ctx.team_names[tm]} — outfield positions (attacking →)", fontsize=10)
    return fig


def possession_plot(pt: pd.DataFrame, ctx: MatchContext, ev: pd.DataFrame | None = None):
    fig, ax = plt.subplots(figsize=(11, 3))
    ax.fill_between(pt.minute, 0.5, pt.team0_share, where=pt.team0_share >= 0.5, color=TEAM_COLORS[0], alpha=0.6,
                    interpolate=True)
    ax.fill_between(pt.minute, pt.team0_share, 0.5, where=pt.team0_share < 0.5, color=TEAM_COLORS[1], alpha=0.6,
                    interpolate=True)
    ax.axhline(0.5, color="#666", lw=0.8)
    if ev is not None:
        for _, g in ev[ev.type == "goal"].iterrows():
            ax.axvline(g.t / 60, color=TEAM_COLORS[int(g.team)] if g.team in (0, 1) else "k", ls="--", lw=1.2)
    ax.set_ylim(0, 1)
    ax.set_xlabel("video minute")
    ax.set_ylabel(f"{ctx.team_names[0]} share")
    ax.set_title("Possession (rolling 5 min); dashed = goals", fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    return fig


BADGE = {"High": "#1f8a4c", "Medium": "#2a78d6", "Low": "#b07100", "Experimental": "#8a8f98"}


def _fmt(v):
    if isinstance(v, (float, np.floating)):
        if np.isnan(v):
            return "–"
        return f"{int(v)}" if float(v).is_integer() else f"{v:.2f}"
    return "" if v is None else str(v)


def _fmt_metric(key: str, v) -> str:
    try:
        v = float(v)
    except (TypeError, ValueError):
        return _fmt(v)
    if np.isnan(v):
        return "–"
    if key.endswith("%"):
        return f"{v:.0f} %"
    if key.startswith("xG"):
        return f"{v:.2f}"
    if key.endswith("_m") or key.endswith("kmh") or key == "PPDA" or key.endswith("_min"):
        return f"{v:.1f}"
    if "km" in key:
        return f"{v:.2f}"
    return f"{int(v)}" if v.is_integer() else f"{v:.1f}"


def _metric_table(km: pd.DataFrame, teams: list) -> str:
    if km is None or len(km) == 0:
        return "<p class='note'>none</p>"
    head = "".join(f"<th>{t}</th>" for t in teams)
    rows = []
    for _, r in km.iterrows():
        vals = "".join(f"<td class='num'>{_fmt_metric(r.metric, r[t])}</td>" for t in teams)
        star = "★" if r.value_rank == 1 else ""
        rows.append(f"<tr><td><b>{r.label}</b> <span class='star'>{star}</span>"
                    f"{'<div class=note>' + r.how_to_read + '</div>' if r.how_to_read else ''}</td>{vals}"
                    f"<td><span class='badge' style='background:{BADGE[r.confidence]}'>{r.confidence}</span></td>"
                    f"<td class='note'>{r.reason}</td></tr>")
    return (f"<table class='km'><tr><th>metric</th>{head}<th>confidence</th><th>why this confidence</th></tr>"
            + "".join(rows) + "</table>")


def _video_href(video_path, out_dir) -> str | None:
    """Link from report.html to the match video (relative when possible, so the folder can move as a whole)."""
    import os
    if not video_path or not Path(str(video_path)).exists():
        return None
    try:
        rel = os.path.relpath(Path(video_path).resolve(), Path(out_dir).resolve())
        return rel.replace(os.sep, "/").replace(" ", "%20")
    except ValueError:                                  # different drive (Windows)
        return Path(video_path).resolve().as_uri()


def _opportunities_html(opps, summary, out_dir, video_path, clips_dir) -> str:
    """'Major opportunities' section: flagged first (with links to the clip and to the moment in the full video),
    the rest collapsed, plus a per-team summary."""
    if opps is None or len(opps) == 0:
        return ("<h2 id='opportunities'>⚑ Major opportunities</h2><p class='note'>No shot, penalty or box entry was "
                "detected. If the match had chances, review the key events (shots are found from the ball track).</p>")
    vhref = _video_href(video_path, out_dir)
    clips = Path(clips_dir) if clips_dir else None

    def row(r):
        if clips is not None and (clips / r.clip_file).exists():
            clip = f"<a href='clips/opportunities/{r.clip_file}' target='_blank'>▶ clip</a> · "
        else:                       # filled in by scripts/cut_opportunities.py once the clip exists
            clip = f"<span class='cliplink' data-clip='{r.clip_file}'></span>"
        full = (f"<a href='{vhref}#t={max(0.0, r.clip_start):.0f}' target='_blank'>full video @ {r.video_time}</a>"
                if vhref else "<span class='note'>video not found</span>")
        cls = "goal" if r.category == "Goal" else ("flag" if r.flag else "")
        xg = f"{r.chance_xg:.2f}" if pd.notna(r.chance_xg) else "–"
        chk = " ✓" if r.checked else ""
        return (f"<tr class='{cls}'><td class='flagc'>{'⚑' if r.flag else ''}</td><td>{int(r.opp_id)}</td>"
                f"<td>{r.clock}</td><td>{r.video_time}</td><td>{r.team_name}</td><td class='cat'>{r.category}{chk}</td>"
                f"<td class='num'>{xg}</td><td class='num'>{r.danger:.0f}</td><td>{r.description}</td>"
                f"<td>{clip}{full}</td></tr>")

    head = ("<tr><th></th><th>#</th><th>clock</th><th>video</th><th>team</th><th>opportunity</th><th>chance xG</th>"
            "<th>danger</th><th>what happened</th><th>watch</th></tr>")
    fl = opps[opps.flag]
    rest = opps[~opps.flag]
    goals = opps[opps.category == "Goal"]
    best = opps[opps.category != "Goal"].sort_values(["danger", "t"], ascending=[False, True]).head(5)
    chip = lambda r: f"<a href='#opp-{int(r.opp_id)}'>{r.clock} {r.team_name}: {r.category}</a>"
    nav = ((f"<b>Goals:</b>{''.join(chip(r) for _, r in goals.iterrows())}" if len(goals) else "")
           + (f"<b>Best chances:</b>{''.join(chip(r) for _, r in best.iterrows())}" if len(best) else ""))
    tbl = lambda q: ("<table class='opp'>" + head + "".join(row(r).replace("<tr ", f"<tr id='opp-{int(r.opp_id)}' ", 1)
                                                          for _, r in q.iterrows()) + "</table>")
    summ = summary.to_html(classes="km", border=0) if summary is not None and len(summary) else ""
    clip_note = ("Clips of the flagged moments are in <code>clips/opportunities/</code> next to this report."
                 if clips is not None else "Clips: run <b>Cut clips</b> in the launcher (or "
                 "<code>python scripts/cut_opportunities.py --run &lt;this folder&gt;</code>) to add ▶ links.")
    return (f"<h2 id='opportunities'>⚑ Major opportunities ({len(fl)} flagged of {len(opps)})</h2>"
            "<p class='note'>Every shot, goal, penalty and box entry, grouped into attacking moves. ⚑ = goal, penalty, "
            "big chance (xG ≥ 0.30), shot on target or shot with xG ≥ 0.10. <b>chance xG</b> = probability that the "
            "move produced a goal; <b>danger</b> 0–100 ranks all moments, including those without a shot. "
            f"✓ = checked in your review. {clip_note} Rules: docs/STATS_RULEBOOK.md §7c.</p>"
            + (f"<div class='nav'>{nav}</div>" if nav else "")
            + (tbl(fl) if len(fl) else "<p class='note'>No flagged opportunity.</p>")
            + (f"<details><summary><b>Other opportunities ({len(rest)})</b> — low-xG shots, box entries without a shot, "
               "set pieces near goal</summary>" + tbl(rest) + "</details>" if len(rest) else "")
            + (f"<details><summary><b>Opportunities per team</b></summary>{summ}</details>" if summ else ""))


def detection_section(tracks: pd.DataFrame, team_names, video_path: str | None = None, n_snapshots: int = 4,
                      width: int = 960) -> dict:
    """What the detector and the classifiers saw: people per type, per team, calibration coverage, and a few
    coloured frames (same colours as the live view). Never empty: a run where positions are missing still shows
    its detections, so 'zero' in the statistics is explained instead of silent."""
    from .people import people_summary
    out = {"table": people_summary(tracks), "lines": [], "images": []}
    if tracks is None or not len(tracks):
        out["lines"].append("No person was detected in the analysed frames — check the video path and the periods "
                            "of the setup (only the game is analysed).")
        return out
    n_fr = tracks.frame.nunique()
    pl = tracks[tracks.cls.isin(["player", "goalkeeper"])]
    with_xy = tracks.groupby("frame").x.agg(lambda s_: s_.notna().any())
    out["lines"].append(f"{len(tracks)} person detections in {n_fr} analysed frames "
                        f"({len(tracks) / max(n_fr, 1):.1f} per frame); "
                        f"{100 * (pl.team >= 0).mean() if len(pl) else 0:.0f} % of the player detections have a team.")
    for k, nm in enumerate(list(team_names)[:2]):
        q = pl[pl.team == k]
        out["lines"].append(f"{nm}: {len(q) / max(n_fr, 1):.1f} players per frame on average, "
                            f"{q.loc[q.track_id >= 0, 'track_id'].nunique()} tracklets.")
    out["lines"].append(f"Pitch positions available on {100 * with_xy.mean():.0f} % of the analysed frames"
                        + ("" if with_xy.mean() > 0 else " — NO camera calibration: every position-based statistic "
                           "(possession, events, xG, distances) is zero. Click >= 4 landmarks on 1-2 frames in the setup."))
    if video_path and Path(str(video_path)).exists() and n_snapshots:
        import cv2
        from .live import annotate_frame
        from .video import fmt_clock
        per = tracks.groupby("frame").size()
        grid = per.index.to_numpy()
        picks = []
        for chunk in np.array_split(grid, n_snapshots):
            if len(chunk):
                picks.append(int(per.loc[chunk].idxmax()))
        cap = cv2.VideoCapture(str(video_path))
        try:
            for fr in picks:
                cap.set(cv2.CAP_PROP_POS_FRAMES, fr)
                ok, img = cap.read()
                if not ok:
                    continue
                rows = tracks[tracks.frame == fr]
                t = float(rows.t.iloc[0])
                v = annotate_frame(img, rows, team_names, title=f"{fmt_clock(t)}  frame {fr}  people {len(rows)}")
                v = cv2.resize(v, (width, int(v.shape[0] * width / v.shape[1])), interpolation=cv2.INTER_AREA)
                ok, buf = cv2.imencode(".jpg", v, [cv2.IMWRITE_JPEG_QUALITY, 80])
                if ok:
                    out["images"].append((f"{fmt_clock(t)} (frame {fr})", base64.b64encode(buf.tobytes()).decode()))
        finally:
            cap.release()
    return out


def _detections_html(d: dict | None) -> str:
    if not d:
        return ""
    h = "<h2>Who was detected</h2><ul>" + "".join(f"<li>{x}</li>" for x in d.get("lines", [])) + "</ul>"
    tb = d.get("table")
    if tb is not None and len(tb):
        h += ("<p class='note'>Every person box is one of these types. Staff, security and fans are kept apart and "
              "never counted as players (docs/STATS_RULEBOOK.md → People in the frame).</p>"
              + tb.to_html(index=False, border=0, classes="prov"))
    if d.get("images"):
        h += "<details open><summary><b>Coloured frames</b> (same colours as the live view)</summary>" + "".join(
            f"<p class='note'>{c}</p><img src='data:image/jpeg;base64,{b}'/>" for c, b in d["images"]) + "</details>"
    return h


def html_report(out_dir: str, team_tbl: pd.DataFrame, timeline: pd.DataFrame, figs: dict, title: str = "Match report",
                notes: str = "", provenance=None, validation: pd.DataFrame | None = None,
                events: pd.DataFrame | None = None, rejected: pd.DataFrame | None = None,
                numbers: pd.DataFrame | None = None, key_metrics: pd.DataFrame | None = None,
                quality: list | None = None, score: str | None = None,
                opportunities: pd.DataFrame | None = None, opp_summary: pd.DataFrame | None = None,
                video_path: str | None = None, clips_dir=None, detections: dict | None = None) -> Path:
    """Self-contained HTML, ordered for a reader: result and data quality -> metrics by confidence -> events ->
    figures -> how it was produced (provenance, your review, rejected events, full table)."""
    imgs = "".join(f'<h3>{k}</h3><img src="data:image/png;base64,{_b64(f)}"/>' for k, f in figs.items())
    teams = list(team_tbl.columns)
    t = team_tbl.copy().map(_fmt)
    tl = timeline[["video_time", "clock", "team", "description"]] if len(timeline) else timeline
    head = f"<h1>{title}</h1>" + (f"<p class='score'>{score}</p>" if score else "") + f"<p class='note'>{notes}</p>"
    legend = ("<div class='legend'><b>How to read it.</b> One panning camera at 640 × 360 sees part of the pitch and a "
              "2-5 px ball, so every number carries a confidence for <i>this</i> match: "
              f"<span class='badge' style='background:{BADGE['High']}'>High</span> quote as is · "
              f"<span class='badge' style='background:{BADGE['Medium']}'>Medium</span> compare the teams / halves · "
              f"<span class='badge' style='background:{BADGE['Low']}'>Low</span> indicative, mention with care · "
              f"<span class='badge' style='background:{BADGE['Experimental']}'>Experimental</span> do not use. "
              "★ = headline metric. The reason column says what raised or lowered the grade "
              "(details: docs/REPORT_GUIDE.md).</div>")
    opp = _opportunities_html(opportunities, opp_summary, out_dir, video_path, clips_dir)
    dq = ("<h2>Data quality</h2><ul>" + "".join(f"<li>{q}</li>" for q in quality) + "</ul>") if quality else ""
    dq += _detections_html(detections)
    metrics = ""
    if key_metrics is not None and len(key_metrics):
        km = key_metrics
        rel = km[km.confidence.isin(["High", "Medium"])]
        low = km[km.confidence == "Low"]
        exp = km[km.confidence == "Experimental"]
        metrics = ("<h2>Key metrics you can rely on</h2>" + _metric_table(rel, teams) +
                   "<h2>Indicative metrics (low confidence)</h2>" + _metric_table(low, teams) +
                   f"<details><summary><b>Experimental metrics ({len(exp)})</b> — computed, but not reliable on this "
                   "footage yet</summary>" + _metric_table(exp, teams) + "</details>")
    else:
        metrics = f"<h2>Team statistics</h2>{t.to_html()}"
    why = ""
    if events is not None and len(events) and "why" in events:
        k = events[events.type.isin(["goal", "shot", "penalty", "red", "yellow"])].copy()
        if len(k):
            from .video import fmt_clock
            k["video_time"] = [fmt_clock(x) for x in k.t]
            cols = [c for c in ("video_time", "clock", "type", "outcome", "xg", "validation", "location", "number",
                                "player", "why") if c in k]
            why = ("<h2>Key events and why they were detected</h2>" + k[cols].to_html(index=False, na_rep="",
                                                                                      float_format=lambda v: f"{v:.2f}"))
    if numbers is not None and len(numbers):
        why += "<h2>Players (by shirt number, from your review / match sheet)</h2>" + numbers.to_html(index=False, na_rep="")
    trust = ""
    if provenance is not None and getattr(provenance, "rows", None):
        trust += ("<h2>How this report was produced</h2><p class='note'>Every input of the pipeline and who provided it: "
                  "<b>manual</b> = entered or checked by you (setup / review UI), <b>measured</b> = verified against your "
                  "review, <b>auto</b> = inferred by the system, <b>default</b> = not provided, <b>assistant</b> = "
                  "pre-filled, not yet confirmed by you.</p>" + provenance.html())
    if validation is not None and len(validation):
        trust += ("<h2>Measured reliability on this match (your review)</h2>"
                  "<p class='note'>Small samples: read the 95 % interval, not only the point estimate.</p>"
                  + validation.to_html(index=False, border=0, classes="val"))
    if rejected is not None and len(rejected):
        cols = [c for c in ("t", "type", "team", "outcome", "source", "reject_reason") if c in rejected]
        trust += ("<h2>Rejected automatic events</h2><p class='note'>Detected by the rules but removed by your review, "
                  "the match sheet or an excluded stretch — kept here for transparency.</p>"
                  + rejected[cols].to_html(index=False, na_rep="", float_format=lambda v: f"{v:.1f}"))
    trust += f"<details><summary><b>Full team statistics table</b></summary>{t.to_html()}</details>"
    html = f"""<!doctype html><html><head><meta charset="utf-8"><title>{title}</title>
<style>body{{font-family:-apple-system,Segoe UI,Roboto,sans-serif;max-width:1100px;margin:24px auto;padding:0 16px;color:#222}}
table{{border-collapse:collapse;font-size:13px;margin-bottom:8px}}td,th{{border-bottom:1px solid #ddd;padding:5px 10px;text-align:right;vertical-align:top}}
th:first-child,td:first-child{{text-align:left}}td:last-child{{text-align:left}}
.prov td,.prov th,.val td,.val th,.km td,.km th{{text-align:left}}.km td.num{{text-align:right;font-variant-numeric:tabular-nums;font-size:14px}}
img{{max-width:100%}}.note{{color:#666;font-size:12px}}.badge{{color:#fff;border-radius:3px;padding:1px 7px;font-size:12px;white-space:nowrap}}
.star{{color:#d4a017}}.score{{font-size:22px;font-weight:600;margin:0}}.legend{{background:#f5f6f8;border-left:4px solid #2a78d6;padding:10px 14px;font-size:13px;line-height:1.7}}
details{{margin:10px 0}}summary{{cursor:pointer;padding:6px 0}}
.opp td,.opp th{{text-align:left}}.opp td:nth-child(5){{white-space:nowrap}}.opp td.num{{text-align:right;font-variant-numeric:tabular-nums}}.opp tr.flag td{{background:#fff8e6}}
.opp tr.goal td{{background:#e9f7ee}}.flagc{{color:#d97706;font-size:15px}}.cat{{font-weight:600;white-space:nowrap}}
.opp a{{color:#2a78d6;text-decoration:none;white-space:nowrap}}.opp a:hover{{text-decoration:underline}}
.nav{{display:flex;flex-wrap:wrap;gap:6px;margin:8px 0 14px}}.nav b{{font-size:12px;padding:3px 2px 3px 6px}}.nav a{{font-size:12px;background:#f1f3f5;border-radius:4px;padding:3px 8px;color:#333;text-decoration:none}}</style></head>
<body>{head}{legend}{opp}{dq}{metrics}<h2>Timeline of major events</h2>{tl.to_html(index=False)}{why}<h2>Figures</h2>{imgs}
{trust}<p class="note">Definitions of every statistic: docs/STATS_RULEBOOK.md · confidence grading: docs/REPORT_GUIDE.md ·
manual inputs and review: docs/MANUAL_VALIDATION.md.</p></body></html>"""
    p = Path(out_dir) / "report.html"
    p.write_text(html)
    return p
