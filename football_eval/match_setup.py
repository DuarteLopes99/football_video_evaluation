"""Human-validated match setup ("match sheet + camera facts") — one JSON file per match.

Why: a single camera at 640 x 360 cannot reliably *infer* several facts that the person who filmed or watched
the match knows in seconds. Every one of them changes every downstream statistic (see
docs/MANUAL_VALIDATION.md for the evaluation). They are entered once in the setup UI
(``python scripts/match_ui.py setup ...``) and stored here, each with its provenance:

=================  ==============================================  =========================================
section            what the user gives                             what it replaces / fixes
=================  ==============================================  =========================================
pitch              real length x width (m)                         default 105 x 68 or the camera estimate
overlays           rectangles of logos / score bug                 hand-typed pixel boxes in the YAML
periods            kick-off / end of each half (video seconds)     side-swap inference (fooled by warm-up)
direction          team attacking the RIGHT goal in the 1st half   inferred from average positions
teams + kits       names + clicked players of each kit (incl.      unsupervised KMeans(2) + guessed cluster
                   goalkeepers and referee)                        order; heuristic goalkeepers
calibration        clicked pitch landmarks (lines, goal posts)     -> homography / PTZ venue calibration
match sheet        final score, goals (minute, team, shirt no.),   kick-off-rule goals; no scorer, no cards
                   cards, substitutions, roster
=================  ==============================================  =========================================

The file is plain JSON so it can be edited by hand, versioned and re-used (e.g. the venue calibration and the
overlays are the same for every home game).
"""
from __future__ import annotations

import copy
import datetime as _dt
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

SETUP_VERSION = 1
KIT_CLASSES = ("team0", "team1", "gk0", "gk1", "referee")
KIT_LABELS = {"team0": "Team A outfield", "team1": "Team B outfield", "gk0": "Team A goalkeeper",
              "gk1": "Team B goalkeeper", "referee": "Referee"}
EVENT_TYPES = ("goal", "own_goal", "penalty_goal", "yellow", "red", "sub", "penalty_missed", "note")
SOURCES = ("manual", "assistant", "auto", "default")     # who produced a section


def _now() -> str:
    return _dt.datetime.now().replace(microsecond=0).isoformat()


def new_setup(video_path: Optional[str] = None, info: Optional[dict] = None) -> dict:
    return {
        "version": SETUP_VERSION,
        "video": {"path": video_path, **(info or {})},
        "match": {"date": None, "competition": None, "venue": None},
        "teams": [{"name": "Team A", "roster": []}, {"name": "Team B", "roster": []}],
        "pitch": {"length": 105.0, "width": 68.0, "source": "default", "note": "",
                  "near_touchline": "visible"},       # visible | partly | hidden (camera-side touchline out of view)
        "overlays": [], "overlays_confirmed": False, "scoreboard_roi": None,
        "periods": [],
        "attacks_right_first_half": None,      # team index (0/1) attacking the goal on the RIGHT of the camera view
        "kits": {k: [] for k in KIT_CLASSES},
        "calibration": {"mode": "static", "camera_file": None, "landmark_frames": [], "anchors": {}},
        "final_score": None,                   # [goals team 0, goals team 1]
        "known_events": [],                    # match sheet: goals, cards, substitutions
        "provenance": {},                      # section -> {"source", "updated"}
        "meta": {"created": _now(), "updated": _now()},
    }


def _upgrade(d: dict) -> dict:
    base = new_setup()
    out = copy.deepcopy(base)
    for k, v in d.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict) and k not in ("provenance", "anchors"):
            out[k].update(v)
        else:
            out[k] = v
    for k in KIT_CLASSES:
        out["kits"].setdefault(k, [])
    return out


def load_setup(path: str | Path) -> dict:
    with open(path) as f:
        return _upgrade(json.load(f))


def find_setup(video_path: Optional[str], configs_dir: str | Path) -> Optional[Path]:
    """The match setup made for this video (``video.path`` with the same file name), newest first.

    Used by ``run_analysis.py`` when ``--setup`` is not given, so the manual input is never silently ignored (the
    launcher used to start runs without it when an empty choice had been remembered for the video)."""
    if not video_path:
        return None
    name = Path(str(video_path)).name
    hits = []
    for p in Path(configs_dir).glob("match_setup*.json"):
        try:
            vp = (json.loads(p.read_text()).get("video") or {}).get("path")
        except (OSError, ValueError):
            continue
        if vp and Path(str(vp)).name == name:
            hits.append(p)
    return max(hits, key=lambda q: q.stat().st_mtime) if hits else None


def save_setup(setup: dict, path: str | Path) -> Path:
    setup["meta"]["updated"] = _now()
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(setup, indent=1, default=_json_default))
    return p


def _json_default(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))


def mark(setup: dict, section: str, source: str = "manual", note: str = "") -> None:
    """Record who set a section (shown in the report's provenance table)."""
    setup.setdefault("provenance", {})[section] = {"source": source, "updated": _now(), **({"note": note} if note else {})}


def source_of(setup: dict, section: str) -> str:
    return setup.get("provenance", {}).get(section, {}).get("source", "default")


def team_names(setup: dict) -> List[str]:
    return [t.get("name") or f"Team {'AB'[i]}" for i, t in enumerate(setup.get("teams", [])[:2])]


def roster_lookup(setup: dict, team: int, number) -> Optional[str]:
    if number is None or team not in (0, 1):
        return None
    try:
        n = int(number)
    except (TypeError, ValueError):
        return None
    for p in setup["teams"][team].get("roster", []):
        if int(p.get("number", -1)) == n:
            return p.get("name")
    return None


# ----------------------------------------------------------------------------- football clock
_MIN = re.compile(r"^\s*(\d+)\s*(?:\+\s*(\d+))?\s*'?\s*$")


def minute_to_video_t(setup: dict, minute: str | int | float, half_len_min: float = 45.0) -> Optional[float]:
    """Football clock ("23", "45+2", "67'") -> video seconds, using the periods of the setup.

    The minute is the referee's clock (1st half 1-45+, 2nd half 46-90+); the event is placed in the middle of
    that minute. Returns None when periods are unknown.
    """
    periods = setup.get("periods") or []
    if not periods:
        return None
    m = _MIN.match(str(minute))
    if not m:
        return None
    base, extra = int(m.group(1)), int(m.group(2) or 0)
    p = min(0 if base <= half_len_min else 1, len(periods) - 1)
    start = float(periods[p]["start"])
    played_min = (base - 1 - p * half_len_min) + extra          # whole minutes played before this one
    return start + max(0.0, played_min * 60.0 + 30.0)


def event_time(setup: dict, e: dict) -> Optional[float]:
    if e.get("t") not in (None, ""):
        return float(e["t"])
    if e.get("minute") not in (None, ""):
        return minute_to_video_t(setup, e["minute"])
    return None


# ----------------------------------------------------------------------------- checklist (quality gate)
def checklist(setup: dict, root: Optional[Path] = None) -> List[dict]:
    """Status of every setup item: ``ok`` | ``warn`` | ``missing`` (+ why it matters).

    ``blocking`` items stop ``run_analysis.py --setup`` unless ``--force`` is given.
    """
    root = Path(root) if root else Path(__file__).resolve().parents[1]
    out = []

    def add(item, status, detail, impact, blocking=False):
        out.append(dict(item=item, status=status, detail=detail, impact=impact, blocking=blocking and status == "missing"))

    v = setup.get("video", {})
    vp = v.get("path")
    exists = bool(vp) and (Path(vp).exists() or (root / vp).exists())
    if vp and not exists:                          # moved to a video library: found again by file name
        from .storage import find_video
        found = find_video(Path(vp).name)
        exists = bool(found)
        vp = f"{vp} -> {found}" if found else vp
    add("video", "ok" if exists else "missing", vp or "no video path", "everything", blocking=True)

    pc = setup["pitch"]
    src = pc.get("source", "default")              # how the size is known (provenance only says who typed it)
    if src == "manual":
        add("pitch size", "ok", f"{pc['length']:.1f} x {pc['width']:.1f} m ({pc.get('source', src)})",
            "all distances, xG geometry, penalty area / final-third rules")
    elif pc.get("source") == "estimated":
        add("pitch size", "warn", f"{pc['length']:.1f} x {pc['width']:.1f} m estimated by the camera calibration",
            "width estimates varied 52-62 m on the Canedo video — enter the real size")
    else:
        add("pitch size", "warn", "default 105 x 68 m", "lower-league pitches are rarely 105 x 68: positions near the "
            "touchlines and xG angles are biased")

    ov = setup.get("overlays") or []
    add("overlays", "ok" if (ov or setup.get("overlays_confirmed")) else "warn",
        f"{len(ov)} region(s)" + (" (confirmed)" if setup.get("overlays_confirmed") else ""),
        "logos / score bug produce false detections and fake pitch lines")

    per = setup.get("periods") or []
    good = [p for p in per if p.get("start") is not None and p.get("end") is not None and p["end"] > p["start"]]
    partial = [p for p in per if (p.get("start") is None) != (p.get("end") is None)]
    if len(good) >= 2 and not partial:
        add("periods", "ok", ", ".join(f"{_mmss(p['start'])}-{_mmss(p['end'])}" for p in good),
            "football clock, warm-up / half-time excluded, side swap")
    elif good or partial:
        add("periods", "warn", f"{len(good)} complete period(s)" + (f", {len(partial)} with only a start or an end"
                                                                     if partial else ""),
            "incomplete periods are ignored: set kick-off AND final whistle of both halves")
    else:
        add("periods", "warn", "not set — inferred from the side swap",
            "warm-up and half-time are counted as play; clock may be minutes off")

    d = setup.get("attacks_right_first_half")
    add("attacking direction", "ok" if d in (0, 1) else "warn",
        f"{team_names(setup)[d]} attacks right in the 1st half" if d in (0, 1) else "inferred from positions",
        "every directional stat (shots, final third, PPDA, field tilt) flips if wrong")

    names = team_names(setup)
    default_names = names == ["Team A", "Team B"]
    add("team names", "warn" if default_names else "ok", " vs ".join(names), "report labels")

    k = {c: len(setup["kits"].get(c, [])) for c in KIT_CLASSES}
    if k["team0"] >= 3 and k["team1"] >= 3:
        add("kits: outfield", "ok", f"{k['team0']} + {k['team1']} samples",
            "team of every player -> every team statistic")
    elif k["team0"] and k["team1"]:
        add("kits: outfield", "warn", f"{k['team0']} + {k['team1']} samples (>= 3 each, sun AND shade, recommended)",
            "team assignment")
    else:
        add("kits: outfield", "warn", "no samples -> unsupervised colour clusters (team order may be swapped)",
            "a swap attributes EVERY statistic to the wrong team")
    add("kits: goalkeepers", "ok" if (k["gk0"] and k["gk1"]) else "warn", f"{k['gk0']} + {k['gk1']} samples",
        "goalkeepers otherwise guessed from time spent in the box (xG freeze frame, saves)")
    add("kits: referee", "ok" if k["referee"] else "warn", f"{k['referee']} samples",
        "referees otherwise mixed into a team (passes / pressing)")

    cal = setup.get("calibration", {})
    mode = cal.get("mode", "static")
    lf = cal.get("landmark_frames") or []
    n_pts = sum(len(f.get("points", {})) for f in lf)
    if mode == "ptz":
        cf = cal.get("camera_file")
        cf_ok = bool(cf) and (Path(cf).exists() or (root / cf).exists())
        if cf_ok:
            msg = f"venue camera {cf}"
            try:
                cd = json.loads((Path(cf) if Path(cf).exists() else root / cf).read_text())
                dl, dw = cd.get("pitch_length"), cd.get("pitch_width")
                if dl and src == "manual" and (abs(dl - pc["length"]) > 1.0 or abs(dw - pc["width"]) > 1.0):
                    add("calibration", "warn", f"{msg} was solved for a {dl:.1f} x {dw:.1f} pitch, setup says "
                        f"{pc['length']:.1f} x {pc['width']:.1f}", "re-run scripts/calibrate_ptz.py --setup ... "
                        "so the camera is solved with the real pitch size")
                else:
                    add("calibration", "ok", msg + f"; {len(lf)} landmark frame(s), {n_pts} points", "positions")
            except (OSError, ValueError):
                add("calibration", "warn", f"cannot read {cf}", "positions")
        else:
            n4 = sum(len(f.get("points", {})) >= 4 for f in lf)
            add("calibration", "missing" if n_pts < 2 else "warn",
                f"no venue camera yet; {len(lf)} landmark frame(s) with {n_pts} points",
                ("solved automatically from your clicks when the video is processed (football_eval/autocalib.py)"
                 if n4 else "click >= 4 landmarks on at least one frame (or run scripts/calibrate_ptz.py --setup)"),
                blocking=n_pts < 2)
    else:
        best = max(lf, key=lambda f: len(f.get("points", {})), default=None)
        n = len(best["points"]) if best else 0
        if n >= 6:
            add("calibration", "ok", f"{n} landmarks on the frame at {_mmss(best['t'])}", "positions")
        elif n >= 4:
            add("calibration", "warn", f"only {n} landmarks (6-10 well spread recommended)", "positions")
        else:
            add("calibration", "missing", f"{n} landmarks (need >= 4)", "no pitch coordinates at all", blocking=True)
        if mode == "flow" and n >= 4:
            ts = sorted(float(f["t"]) for f in lf if len(f.get("points", {})) >= 4)
            p0 = min([float(p["start"]) for p in setup.get("periods") or [] if p.get("start") is not None] or [0.0])
            gaps = np.diff([p0] + ts) if ts else np.array([])
            if ts and (ts[0] - p0 > 60 or (len(gaps) and gaps.max() > 600)):
                out[-1].update(status="warn", detail=out[-1]["detail"] + f"; panning camera without a venue model: no "
                               f"positions before the first landmark frame ({_mmss(ts[0])}) and drift between frames "
                               f"far apart (largest gap {gaps.max() / 60:.0f} min)",
                               impact="add a landmark frame near each kick-off and every ~5-10 min (or solve a PTZ "
                                      "venue camera once per ground)")

    fs = setup.get("final_score")
    goals = [e for e in setup.get("known_events", []) if e.get("type") in ("goal", "own_goal", "penalty_goal")]
    if fs is None:
        add("match sheet", "warn", "final score not given", "goals are only inferred (no score bug, no audio here)")
    elif sum(fs) != len(goals):
        add("match sheet", "warn", f"score {fs[0]}-{fs[1]} but {len(goals)} goal(s) listed",
            "list every goal (minute, team) so automatic goals can be confirmed or rejected")
    else:
        timed = sum(event_time(setup, e) is not None for e in goals)
        add("match sheet", "ok" if timed == len(goals) else "warn",
            f"score {fs[0]}-{fs[1]}, {len(goals)} goal(s), {timed} with a time", "goals, scorers, timeline")
    n_roster = sum(len(t.get("roster", [])) for t in setup["teams"])
    add("rosters", "ok" if n_roster else "warn", f"{n_roster} players",
        "optional: shirt number -> name in the event review")
    # values pre-filled by someone else (e.g. the assistant) count only once you have looked at them
    for it in out:
        sec = ITEM_SECTION.get(it["item"])
        if sec and source_of(setup, sec) == "assistant" and it["status"] == "ok":
            it["status"] = "warn"
            it["impact"] = "pre-filled, not yet confirmed by you — check it in the setup UI (" + \
                setup["provenance"][sec].get("note", "") + ")"
    return out


ITEM_SECTION = {"pitch size": "pitch", "overlays": "overlays", "periods": "periods", "attacking direction": "direction",
                "team names": "teams", "kits: outfield": "kits", "kits: goalkeepers": "kits", "kits: referee": "kits",
                "calibration": "calibration", "match sheet": "match_sheet"}


def _mmss(t) -> str:
    t = float(t)
    return f"{int(t // 60):d}:{int(t % 60):02d}"


def format_checklist(items: List[dict]) -> str:
    sym = {"ok": "✓", "warn": "!", "missing": "✗"}
    w = max(len(i["item"]) for i in items)
    return "\n".join(f"  {sym[i['status']]} {i['item']:<{w}}  {i['detail']}" + (f"   <- {i['impact']}" if i["status"] != "ok" else "")
                     for i in items)


# ----------------------------------------------------------------------------- setup -> pipeline config
def apply_to_config(setup: dict, cfg) -> List[str]:
    """Copy the validated facts into the pipeline ``Config``. Returns a human-readable change log."""
    log = []

    def setv(obj, key, val, label):
        old = getattr(obj, key)
        if old != val:
            setattr(obj, key, val)
            log.append(f"{label}: {old} -> {val}")

    pc = setup["pitch"]
    if pc.get("source", "default") != "default":
        setv(cfg.pitch, "length", float(pc["length"]), "pitch.length")
        setv(cfg.pitch, "width", float(pc["width"]), "pitch.width")
        cfg.pitch.fixed_size = pc.get("source") == "manual"     # measured -> overrides camera estimates
    if setup.get("overlays") or setup.get("overlays_confirmed"):
        setv(cfg.video, "ignore_regions", [list(map(float, r)) for r in setup.get("overlays", [])], "video.ignore_regions")
    if setup.get("scoreboard_roi"):
        setv(cfg.scoreboard, "roi", list(map(float, setup["scoreboard_roi"])), "scoreboard.roi")
        setv(cfg.scoreboard, "enabled", True, "scoreboard.enabled")
    per = [p for p in setup.get("periods", []) if p.get("start") is not None and p.get("end") is not None]
    if per:
        setv(cfg.match, "halves", [[float(p["start"]), float(p["end"])] for p in per], "match.halves")
    d = setup.get("attacks_right_first_half")
    if d in (0, 1):
        setv(cfg.match, "home_attacks_right_first_half", bool(d == 0), "match.home_attacks_right_first_half")
    names = team_names(setup)
    if names != ["Team A", "Team B"]:
        setv(cfg.teams, "team_names", names, "teams.team_names")
    n_kits = sum(len(setup["kits"].get(c, [])) for c in KIT_CLASSES)
    if n_kits:
        cfg.teams.kit_samples = {c: list(setup["kits"].get(c, [])) for c in KIT_CLASSES}
        cfg.teams.kit_colors = None          # superseded by the clicked samples
        log.append(f"teams: {n_kits} clicked kit samples -> seeded classification "
                   + ", ".join(f"{c}={len(setup['kits'][c])}" for c in KIT_CLASSES))
    cal = setup.get("calibration", {})
    mode = cal.get("mode")
    if mode == "ptz" and cal.get("camera_file"):
        setv(cfg.pitch, "homography", "ptz", "pitch.homography")
        setv(cfg.pitch, "calibration_file", cal["camera_file"], "pitch.calibration_file")
        if cal.get("anchors"):
            cfg.pitch.extra_anchors = {int(k): list(v) for k, v in cal["anchors"].items()}
            log.append(f"pitch: {len(cal['anchors'])} user-validated camera anchors")
    elif mode == "static" and cal.get("landmark_frames"):
        best = max(cal["landmark_frames"], key=lambda f: len(f.get("points", {})))
        if len(best.get("points", {})) >= 4:
            setv(cfg.pitch, "homography", "static", "pitch.homography")
            cfg.pitch.calibration_points = dict(best["points"])
            log.append(f"pitch: static homography from {len(best['points'])} clicked landmarks (t={best['t']:.1f}s)")
    elif mode == "flow" and cal.get("landmark_frames"):
        setv(cfg.pitch, "homography", "flow", "pitch.homography")
        fps = float(setup.get("video", {}).get("fps") or 25.0)
        cfg.pitch.flow_anchor_points = {int(round(f["t"] * fps)): dict(f["points"]) for f in cal["landmark_frames"]
                                        if len(f.get("points", {})) >= 4}
        log.append(f"pitch: flow homography with {len(cfg.pitch.flow_anchor_points)} clicked anchor frames")
    return log
