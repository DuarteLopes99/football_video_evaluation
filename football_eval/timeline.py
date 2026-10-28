"""Timeline of major events with video timestamps + football clock, and highlight clip export."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .context import MatchContext
from .video import fmt_clock

PRIORITY = {"goal": 1, "penalty": 1, "red": 1, "shot": 2, "yellow": 2, "sub": 3, "penalty_missed": 2,
            "offside_suspected": 3, "foul_proxy": 3, "corner": 3, "free_kick": 4, "kickoff": 4, "period_end": 4}
CHECK = {"confirmed": " ✓", "match_sheet_confirmed": " ✓ sheet", "corrected": " ✓ corrected", "manual": " (added by hand)",
         "match_sheet": " (match sheet)"}


def _who(e) -> str:
    """' — #9 Rui' when the review / match sheet gave a shirt number (and the roster a name)."""
    num = e.get("number") if hasattr(e, "get") else None
    if num is None or (isinstance(num, float) and np.isnan(num)):
        return ""
    name = e.get("player")
    return f" — #{int(num)}" + (f" {name}" if isinstance(name, str) and name else "")


def build_timeline(ev: pd.DataFrame, ctx: MatchContext, include_minor_free_kicks: bool = False,
                   final_third_x: float = 70.0) -> pd.DataFrame:
    rows = []
    names = ctx.team_names
    tname = lambda tm: names[int(tm)] if (pd.notna(tm) and 0 <= int(tm) < len(names)) else "?"
    for _, e in ev.iterrows():
        typ = e.type
        if typ not in PRIORITY:
            continue
        desc = None
        if typ == "goal":
            desc = f"GOAL {tname(e.team)}{_who(e)}" + (f" (xG {e.xg:.2f})" if pd.notna(e.xg) else "") + f" [{e.outcome}]"
        elif typ == "shot":
            desc = (f"Shot {tname(e.team)}{_who(e)} — {e.outcome}, xG {e.xg:.2f}" if pd.notna(e.xg)
                    else f"Shot {tname(e.team)}{_who(e)}")
        elif typ in ("yellow", "red"):
            desc = f"{'Yellow' if typ == 'yellow' else 'Red'} card {tname(e.team)}{_who(e)}"
        elif typ == "sub":
            desc = f"Substitution {tname(e.team)}{_who(e)}" + (f" ({e.detail})" if isinstance(e.detail, str) and e.detail else "")
        elif typ == "penalty_missed":
            desc = f"Penalty missed {tname(e.team)}{_who(e)}"
        elif typ == "penalty":
            desc = f"Penalty to {tname(e.team)}"
        elif typ == "corner":
            desc = f"Corner {tname(e.team)}"
        elif typ == "foul_proxy":
            desc = f"Foul by {tname(e.team)} (free kick awarded)"
        elif typ == "offside_suspected":
            desc = f"Offside (suspected) {tname(e.team)}"
        elif typ == "kickoff":
            desc = "Kick-off " + ("(period start)" if e.detail == "period_start" else "after goal") + f" — {tname(e.team)}"
        elif typ == "free_kick":
            x, _ = ctx.to_attack_frame(int(e.team), e.t, e.x, e.y)
            if not include_minor_free_kicks and x < final_third_x:
                continue
            desc = f"Free kick in attacking third — {tname(e.team)}"
        desc += CHECK.get(e.get("validation", ""), "") if hasattr(e, "get") else ""
        rows.append(dict(t=e.t, video_time=fmt_clock(e.t), clock=ctx.match_clock(e.t), type=typ,
                         team=tname(e.team), priority=PRIORITY[typ], description=desc,
                         clip_start=max(0.0, e.t - 8.0), clip_end=e.t + 5.0))
    for i, (a, b) in enumerate(ctx.periods):
        rows.append(dict(t=b, video_time=fmt_clock(b), clock=ctx.match_clock(b), type="period_end", team="",
                         priority=4, description=f"End of period {i + 1}", clip_start=b, clip_end=b))
    tl = pd.DataFrame(rows).sort_values("t").reset_index(drop=True) if rows else pd.DataFrame(
        columns=["t", "video_time", "clock", "type", "team", "priority", "description", "clip_start", "clip_end"])
    return tl


def export_highlights_script(tl: pd.DataFrame, video_path: str, out_dir: str, max_priority: int = 2) -> Path:
    """Write a shell script that cuts one clip per major event with ffmpeg (stream copy, fast)."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    lines = ["#!/usr/bin/env bash", "set -e", f'VIDEO="{video_path}"', 'mkdir -p "$(dirname "$0")/clips"']
    for i, r in tl[tl.priority <= max_priority].iterrows():
        name = f"{i:03d}_{r.type}_{r.video_time.replace(':', '-')}.mp4"
        lines.append(f'ffmpeg -loglevel error -y -ss {r.clip_start:.1f} -to {r.clip_end:.1f} -i "$VIDEO" -c copy '
                     f'"$(dirname "$0")/clips/{name}"')
    p = out / "cut_highlights.sh"
    p.write_text("\n".join(lines) + "\n")
    p.chmod(0o755)
    return p
