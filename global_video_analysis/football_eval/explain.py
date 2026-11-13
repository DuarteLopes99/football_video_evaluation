"""Explainability: a plain-language *why* for every event and a provenance ledger for every number.

Two questions a coach (or you, three weeks later) asks about any figure in the report:

1. **Why did the system say this happened?** -> :func:`explain_events` adds a ``why`` column to
   ``events.csv``: the rule that fired, the measured values and the thresholds (config keys) they were
   compared with — the same wording as docs/STATS_RULEBOOK.md.
2. **Where does this input come from — did a person check it?** -> :class:`Provenance` records, stage by
   stage, whether each input was *manual* (entered/validated in the setup or review UI), *auto* (inferred)
   or *default*, together with the quality numbers measured on the way. It is saved as ``provenance.json``
   and rendered at the top of ``report.html``.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd


def _f(v, nd=1, unit=""):
    return "?" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.{nd}f}{unit}"


def _detail_kv(detail) -> dict:
    out = {}
    if not isinstance(detail, str):
        return out
    for part in detail.split("|"):
        if "=" in part:
            k, v = part.split("=", 1)
            try:
                out[k] = float(v)
            except ValueError:
                out[k] = v
        elif part:
            out.setdefault("tags", []).append(part)
    return out


def explain_event(e: pd.Series, cfg, team_names: List[str]) -> str:
    ec, pc = cfg.events, cfg.possession
    tn = lambda t: team_names[int(t)] if pd.notna(t) and int(t) in (0, 1) and int(t) < len(team_names) else "unknown team"
    who = f"{tn(e.team)}" + (f" (track {int(e.tid)})" if pd.notna(e.get("tid")) and e.get("tid", -1) >= 0 else "")
    kv = _detail_kv(e.get("detail"))
    tags = kv.get("tags", [])
    src = e.get("source", "rules")
    typ = e.type
    length = np.hypot(e.x_end - e.x, e.y_end - e.y) if pd.notna(e.get("x_end")) and pd.notna(e.x) else np.nan
    if src in ("manual", "match_sheet"):
        base = {"manual": "Added by hand in the event review", "match_sheet": "From the match sheet in match_setup.json"}[src]
        num = f", shirt #{int(e.number)}" if pd.notna(e.get("number")) else ""
        return f"{base}: {typ.replace('_', ' ')} {tn(e.team)}{num}."
    if typ == "pass":
        s = (f"{who} lost control and the next controller was {'a team-mate' if e.outcome == 'complete' else 'an opponent' if e.outcome == 'intercepted' else 'nobody (ball out)'}"
             f"; ball travelled {_f(length)} m (>= events.min_pass_length_m = {ec.min_pass_length_m} m, "
             f"within events.max_pass_s = {ec.max_pass_s} s).")
        expl = {"progressive": f"moved >= {ec.progressive_pass_m} m towards goal", "final_third_entry": "entered the final third",
                "box_entry": "ended in the penalty area", "cross": "from a wide zone into the box", "long": ">= 30 m"}
        if tags:
            s += " Tags: " + "; ".join(f"{t} ({expl.get(t, t)})" for t in tags) + "."
        return s
    if typ == "shot":
        sit = tags[0] if tags else "open_play"
        sp = kv.get("speed")
        s = (f"{who} released the ball at {_f(sp)} m/s (>= events.shot_min_speed = {ec.shot_min_speed}) "
             if sp is not None else f"{who} released the ball faster than events.shot_min_speed = {ec.shot_min_speed} m/s ")
        s += (f"towards the goal mouth (± events.shot_goal_tolerance = {ec.shot_goal_tolerance} m) from "
              f"{_f(kv.get('dist'))} m (<= {ec.shot_max_dist} m). xG {_f(e.xg, 2)}: {sit.replace('_', ' ')}, "
              f"{int(kv.get('def_tri', 0))} outfield defender(s) inside the shot-posts triangle. Outcome '{e.outcome}'")
        why_out = {"goal": "the ball crossed the goal line between the posts, or the opponents kicked off next",
                   "saved": "the next touch was the opposing goalkeeper", "blocked": f"an opponent touched it within "
                   f"{ec.block_window_s} s and {ec.block_max_dist} m", "off_target": "ball went past the goal / next restart was a goal kick",
                   "blocked_or_saved_out": "next restart was a corner", "unresolved": "the ball was lost by the tracker"}
        return s + f" ({why_out.get(e.outcome, e.outcome)})."
    if typ == "goal":
        if src == "shot_rule":
            return f"Goal for {tn(e.team)}: a detected shot whose outcome was 'goal' (ball over the line between the posts or followed by the opponents' kick-off)."
        if src == "kickoff_rule":
            return (f"Goal for {tn(e.team)} INFERRED: no shot was seen, but play stopped for "
                    f">= events.goal_min_stoppage_s = {ec.goal_min_stoppage_s} s and restarted with a kick-off by the other team.")
        if src == "scoreboard":
            return "Goal recovered from a change of the broadcast score overlay (no matching shot / kick-off)."
        return f"Goal ({src})."
    if typ == "interception":
        return f"{who} took the ball off an opponent's pass that left the passer at > 4 m/s."
    if typ == "ball_recovery":
        return (f"{who} won the ball from an opponent in a contest (no pass in between) and kept it >= "
                f"events.recovery_min_s = {ec.recovery_min_s} s.")
    if typ in ("corner", "throw_in", "goal_kick", "free_kick", "penalty", "kickoff"):
        where = {"corner": f"within events.corner_radius_m = {ec.corner_radius_m} m of a corner flag",
                 "throw_in": f"within events.throwin_band_m = {ec.throwin_band_m} m of a touchline after the ball went out over it",
                 "goal_kick": "inside / next to the goal area", "penalty": "on the penalty spot with only the taker and goalkeeper in the box",
                 "free_kick": "anywhere else after a stoppage that was not an out-of-play",
                 "kickoff": f"on the centre spot with >= {int(ec.kickoff_own_half_share * 100)} % of each team in its own half"}[typ]
        reason = f" Stoppage reason: {e.detail}." if isinstance(e.detail, str) and e.detail else ""
        return f"Restart by {who}: ball put back into play {where}.{reason}"
    if typ == "foul_proxy":
        return (f"PROXY — play stopped (ball at rest / players still) and the OTHER team restarted with a "
                f"{e.detail or 'free kick'}; the whistle itself is not observable, so the foul is attributed to {tn(e.team)}.")
    if typ == "offside_suspected":
        return (f"PROXY — a free kick to the defending team in its own half right after a forward pass by {tn(e.team)} "
                "whose receiver was beyond the passer in the opponents' half.")
    if typ in ("yellow", "red", "sub", "card", "substitution"):
        return f"From the match sheet: {typ} {tn(e.team)}."
    return f"{typ} ({src})."


def explain_events(ev: pd.DataFrame, cfg, team_names: List[str]) -> pd.DataFrame:
    ev = ev.copy()
    ev["why"] = [explain_event(r, cfg, team_names) for _, r in ev.iterrows()] if len(ev) else []
    if "validation" in ev:
        lab = {"confirmed": "Confirmed by the user in the event review.", "corrected": "Corrected by the user (team / outcome).",
               "match_sheet_confirmed": "Matches a goal on the match sheet.", "manual": "", "unreviewed": ""}
        ev["why"] = [w + (" " + lab.get(v, "") if lab.get(v) else "") for w, v in zip(ev.why, ev.validation)]
    return ev


# ----------------------------------------------------------------------------- provenance ledger
@dataclass
class Provenance:
    rows: List[dict] = field(default_factory=list)

    def add(self, stage: str, item: str, value, source: str, note: str = "", quality: Optional[str] = None):
        """``source``: manual | assistant | auto | default | measured (= checked against the user's review)."""
        self.rows.append(dict(stage=stage, item=item, value=value if isinstance(value, str) else json.dumps(value, default=str),
                              source=source, quality=quality or "", note=note))

    def frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.rows, columns=["stage", "item", "value", "source", "quality", "note"])

    def save(self, path: str | Path):
        Path(path).write_text(json.dumps(self.rows, indent=1, default=str))

    @classmethod
    def load(cls, path: str | Path) -> "Provenance":
        p = Path(path)
        return cls(json.loads(p.read_text())) if p.exists() else cls()

    def html(self) -> str:
        if not self.rows:
            return ""
        badge = {"manual": "#1f8a4c", "measured": "#1f8a4c", "assistant": "#7a5cc7", "auto": "#b07100", "default": "#b3261e"}
        trs = "".join(
            f"<tr><td>{r['stage']}</td><td>{r['item']}</td><td>{r['value']}</td>"
            f"<td><span style='color:#fff;background:{badge.get(r['source'], '#666')};border-radius:3px;padding:1px 6px'>"
            f"{r['source']}</span></td><td>{r['quality']}</td><td class='note'>{r['note']}</td></tr>" for r in self.rows)
        return ("<table class='prov'><tr><th>stage</th><th>input</th><th>value</th><th>source</th><th>quality</th><th>note</th></tr>"
                + trs + "</table>")


def setup_provenance(prov: Provenance, setup: Optional[dict], cfg, config_changes: Optional[List[str]] = None):
    """Ledger rows for the inputs that come from match_setup.json (or their automatic fallbacks)."""
    from .match_setup import KIT_CLASSES, source_of
    if setup is None:
        prov.add("setup", "match_setup.json", "none", "default", "every input below is inferred or a default; "
                 "run scripts/match_ui.py setup to validate them")
        prov.add("setup", "pitch size", f"{cfg.pitch.length:.1f} x {cfg.pitch.width:.1f} m", "default")
        prov.add("setup", "periods", cfg.match.halves or "inferred (side swap)", "auto" if not cfg.match.halves else "default")
        return
    s = lambda k: source_of(setup, k)
    pc = setup["pitch"]
    psrc = {"manual": "manual", "estimated": "auto"}.get(pc.get("source"), "default")
    prov.add("setup", "pitch size", f"{pc['length']:.1f} x {pc['width']:.1f} m", psrc,
             (pc.get("note") or "") + (" (camera estimate)" if pc.get("source") == "estimated" else ""))
    prov.add("setup", "overlays masked", len(setup.get("overlays", [])), s("overlays"))
    per = setup.get("periods", [])
    prov.add("setup", "periods (video s)", [[p.get("start"), p.get("end")] for p in per] or "inferred", s("periods") if per else "auto")
    d = setup.get("attacks_right_first_half")
    prov.add("setup", "1st-half direction", (setup["teams"][d]["name"] + " attacks right") if d in (0, 1) else "inferred",
             s("direction") if d in (0, 1) else "auto")
    nk = {c: len(setup["kits"].get(c, [])) for c in KIT_CLASSES}
    prov.add("setup", "kit samples", nk, s("kits") if sum(nk.values()) else "auto",
             "seeded team / goalkeeper / referee assignment" if sum(nk.values()) else "unsupervised colour clusters")
    cal = setup.get("calibration", {})
    prov.add("setup", "calibration", f"{cal.get('mode')}; {len(cal.get('landmark_frames', []))} landmark frame(s)",
             s("calibration") if cal.get("landmark_frames") else "auto", cal.get("camera_file") or "")
    fs = setup.get("final_score")
    prov.add("setup", "match sheet", f"final score {fs[0]}-{fs[1]}, {len(setup.get('known_events', []))} events" if fs else "none",
             s("match_sheet") if fs else "default")
    sec_of = [("pitch.length", "pitch"), ("pitch.width", "pitch"), ("video.ignore_regions", "overlays"),
              ("scoreboard", "overlays"), ("match.halves", "periods"), ("match.home_attacks", "direction"),
              ("teams.team_names", "teams"), ("teams:", "kits"), ("pitch", "calibration")]
    for c in config_changes or []:
        sec = next((v for k, v in sec_of if c.startswith(k)), None)
        src = s(sec) if sec else "manual"
        if sec == "pitch":
            src = psrc
        prov.add("config", "applied from setup", c, src if src != "default" else "manual")
