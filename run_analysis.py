"""Command-line entry point.

Recommended flow (manual + explainable):
    python scripts/match_ui.py setup --video data/videos/match.mp4 --setup configs/match_setup_<match>.json   # ~10 min
    python run_analysis.py --video data/videos/match.mp4 --config configs/<venue>.yaml \
        --setup configs/match_setup_<match>.json --out outputs/<match>
    python scripts/match_ui.py review --run outputs/<match> --setup configs/match_setup_<match>.json        # ~10 min
    python run_analysis.py --tracks outputs/<match> --config configs/<venue>.yaml \
        --setup configs/match_setup_<match>.json --out outputs/<match>      # re-applies review.json in seconds

Other modes:
    python run_analysis.py --tracks outputs/match1 --config configs/default.yaml   # analytics only (tuning rules)
    python run_analysis.py --synthetic --out outputs/synthetic_demo               # demo with ground truth
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from football_eval.analytics import run_analytics, save_analytics  # noqa: E402
from football_eval.config import load_config  # noqa: E402
from football_eval.explain import Provenance, setup_provenance  # noqa: E402
from football_eval.match_setup import apply_to_config, checklist, format_checklist, load_setup  # noqa: E402
from football_eval.perception import load_perception, run_perception, save_perception  # noqa: E402
from football_eval.review import load_review  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description="Single-camera football match analysis")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--video", help="match video (mp4)")
    src.add_argument("--tracks", help="folder with tracks.csv / ball.csv from a previous perception run")
    src.add_argument("--synthetic", action="store_true", help="generate a synthetic match (with ground truth)")
    ap.add_argument("--config", default=None)
    ap.add_argument("--setup", default=None, help="match_setup.json from scripts/match_ui.py setup (validated inputs)")
    ap.add_argument("--review", default=None, help="review.json (default: <out>/review.json if it exists)")
    ap.add_argument("--force", action="store_true", help="run even if the setup checklist has blocking items")
    ap.add_argument("--repair", action="store_true", help="with --tracks: first re-process the camera repairs accepted "
                    "in the review UI (review.json -> repairs), then run the analytics")
    ap.add_argument("--out", default=None)
    ap.add_argument("--max-frames", type=int, default=None, help="debug: stop perception after N analysed frames")
    ap.add_argument("--speed", choices=["fast", "standard", "quality"], default=None,
                    help="processing speed preset (overrides analysis_fps / detector tiles of the config)")
    ap.add_argument("--no-report", action="store_true")
    ap.add_argument("--no-live", action="store_true", help="do not write the live preview (<out>/live/) during perception")
    ap.add_argument("--no-setup", action="store_true", help="do not look for a match setup of this video automatically")
    a = ap.parse_args()

    cfg = load_config(a.config)
    if a.speed:
        from football_eval.config import SPEED_PRESETS, apply_speed
        ch = apply_speed(cfg, a.speed)
        print(f"Speed preset '{a.speed}': {SPEED_PRESETS[a.speed]['note']}" + (" (" + ", ".join(ch) + ")" if ch else ""))
    if a.video:
        from football_eval.config import resolve_path
        a.video = resolve_path(a.video)          # also finds a recording by name in the video libraries
        if not Path(a.video).exists():
            sys.exit(f"video not found: {a.video} (put it in data/videos or a library of configs/storage.yaml)")
    out = Path(a.out or (a.tracks if a.tracks else cfg.output_dir))
    out.mkdir(parents=True, exist_ok=True)
    prov = Provenance()
    setup = None
    if not a.setup and not a.no_setup and (a.video or a.tracks):
        from football_eval.match_setup import find_setup
        found = find_setup(a.video or _video_of_run(a.tracks), ROOT / "configs")
        if found:
            a.setup = str(found.relative_to(ROOT)) if found.is_relative_to(ROOT) else str(found)
            print(f"Match setup found for this video: {a.setup} (your manual input is used; --no-setup to skip it)")
        else:
            print("[warn] no match setup for this video: teams, periods, kits and calibration are all guessed. "
                  "Run the setup first (scripts/match_ui.py setup) for reliable results.")
    if a.setup:
        setup = load_setup(a.setup)
        items = checklist(setup, ROOT)
        print("Setup checklist (" + a.setup + "):\n" + format_checklist(items))
        blocking = [i for i in items if i["blocking"]]
        if blocking and not a.force and not a.tracks:
            sys.exit("\nBlocking setup items: " + ", ".join(i["item"] for i in blocking) +
                     " — complete them in scripts/match_ui.py setup, or pass --force")
        changes = apply_to_config(setup, cfg)
        if a.video:                         # clicked landmarks without a solved venue camera -> solve it now
            from football_eval.autocalib import ensure_calibration
            changes += ensure_calibration(setup, a.setup, cfg, a.video, ROOT)
        if changes:
            print("Applied from the setup:\n  " + "\n  ".join(changes))
        setup_provenance(prov, setup, cfg, changes)
    else:
        setup_provenance(prov, None, cfg)
    sb_changes = []
    video = a.video

    if a.video:
        print(f"[1/2] perception on {a.video}")
        res = run_perception(a.video, cfg, max_frames=a.max_frames, live_dir=None if a.no_live else str(out),
                             team_names=list(cfg.teams.team_names))
        save_perception(res, str(out))
        tracks, ball = res["tracks"], res["ball"]
        sb_changes = res["meta"].get("scoreboard_changes", [])
        _perception_provenance(prov, res)
    elif a.tracks:
        if a.repair:
            from football_eval.perception import repair_run
            from football_eval.review import save_review
            rp0 = Path(a.review) if a.review else Path(a.tracks) / "review.json"
            rv0 = load_review(rp0)
            pending = [r for r in rv0.get("repairs", []) if not r.get("applied")]
            meta0 = json.loads((Path(a.tracks) / "meta.json").read_text())
            vid = meta0.get("video", {}).get("path")
            from football_eval.config import resolve_path
            if vid and not Path(str(vid)).exists():
                vid = resolve_path(vid)                          # moved to / archived in a video library
            if setup is not None and setup.get("video", {}).get("path"):
                vid = resolve_path(setup["video"]["path"]) if not Path(str(vid)).exists() else vid
            vc = meta0.get("config", {}).get("video", {})       # same frame grid as the original run
            cfg.video.analysis_fps = vc.get("analysis_fps", cfg.video.analysis_fps)
            cfg.video.resize_width = vc.get("resize_width", cfg.video.resize_width)
            if not vid or not Path(str(vid)).exists():
                sys.exit("video not found for the repair: give --setup (its video path) or fix meta.json")
            print(f"[repair] {len(pending)} pending camera repair(s) on {vid}")
            if pending:
                rv0["repairs"] = repair_run(vid, cfg, a.tracks, rv0.get("repairs", []))
                save_review(rv0, rp0)
        res = load_perception(a.tracks)
        tracks, ball = res["tracks"], res["ball"]
        sb_changes = res.get("meta", {}).get("scoreboard_changes", [])
        video = res.get("meta", {}).get("video", {}).get("path")
        if video and not Path(str(video)).exists():
            from football_eval.config import resolve_path
            video = resolve_path(video)                         # found again by name in the video libraries
        _perception_provenance(prov, res)
    else:
        from football_eval.synthetic import SynthConfig, SyntheticMatch, evaluate_events
        m = SyntheticMatch(SynthConfig(seed=0)).generate()
        tracks, ball = m["tracks"], m["ball"]
        m["gt_events"].to_csv(out / "gt_events.csv", index=False)
        cfg.match.halves = m["periods"]

    review = None
    rp = Path(a.review) if a.review else out / "review.json"
    if rp.exists():
        review = load_review(rp)
        print(f"Applying the review in {rp}")

    print("[2/2] analytics")
    an = run_analytics(tracks, ball, cfg, sb_changes, root=str(ROOT), setup=setup, review=review, prov=prov)
    rep = save_analytics(an, str(out), make_report=not a.no_report,
                         video_path=video if video and Path(str(video)).exists() else None)
    km = an.get("key_metrics")
    if km is not None and len(km):
        from football_eval.confidence import quality_lines
        print("\nData quality:\n  " + "\n  ".join(quality_lines(an["data_quality"])))
        cols = ["label"] + list(an["team_stats"].columns) + ["confidence"]
        show = km[km.confidence != "Experimental"][cols].copy()
        print("\nKey metrics (Experimental ones are only in report.html / key_metrics.csv):")
        print(show.round(2).to_string(index=False))
    else:
        print(an["team_stats"].round(2).to_string())
    print("\nMajor events:")
    tl = an["timeline"]
    print(tl[tl.priority <= 2][["video_time", "clock", "description"]].to_string(index=False))
    op = an.get("opportunities")
    if op is not None and len(op):
        fl = op[op.flag]
        print(f"\nFlagged opportunities ⚑ ({len(fl)} of {len(op)}; all in opportunities.csv and at the top of the report):")
        if len(fl):
            print(fl[["video_time", "clock", "team_name", "category", "chance_xg", "description"]]
                  .rename(columns={"team_name": "team", "chance_xg": "xG"}).to_string(index=False))
    if an.get("validation") is not None:
        print("\nMeasured reliability (your review):\n" + an["validation"].to_string(index=False))
    if a.synthetic:
        ev = evaluate_events(an["events"], m["gt_events"])
        ev.to_csv(out / "event_detection_eval.csv")
        print("\nEvent detection vs ground truth:\n", ev.round(3).to_string())
    print(f"\nOutputs in {out}  (report: {rep})")
    if (out / "live" / "state.json").exists():                  # tell the launcher's live screen we are done
        st = json.loads((out / "live" / "state.json").read_text() or "{}")
        st.update(status="done", progress=1.0)
        (out / "live" / "state.json").write_text(json.dumps(st))
    if not a.synthetic and review is None:
        print(f"Next: check the key events and a few frames ->  python scripts/match_ui.py review --run {out}"
              + (f" --setup {a.setup}" if a.setup else ""))


def _video_of_run(run_dir: str):
    try:
        return json.loads((Path(run_dir) / "meta.json").read_text()).get("video", {}).get("path")
    except (OSError, ValueError):
        return None


def _perception_provenance(prov: Provenance, res: dict):
    import numpy as np
    meta = res.get("meta", {})
    fr = res.get("frames")
    if fr is not None and len(fr):
        ok = fr.cam_ok.mean() if "cam_ok" in fr else fr.has_h.mean() if "has_h" in fr else np.nan
        prov.add("perception", "frames with a usable camera pose", f"{100 * ok:.0f} % of {len(fr)}", "auto",
                 note="automatic quality gate (line-alignment error); validate it in the review")
    tm = meta.get("teams") or {}
    if tm:
        q = ""
        ta = res.get("team_assignment")
        if ta is not None and len(ta):
            q = f"{100 * (ta.decision == 'majority').mean():.0f} % of tracks by clear majority (>= 60 % of crops)"
        prov.add("perception", "team assignment", tm.get("method", "?"),
                 "manual" if "seeded" in str(tm.get("method")) else "auto", quality=q,
                 note="" if "seeded" in str(tm.get("method")) else str(tm.get("team_order", "")))
    tr = res.get("tracks")
    if tr is not None and len(tr) and "track_id" in tr:
        g = tr[tr.track_id >= 0].groupby("track_id").t.agg(["min", "max"])
        dur = (g["max"] - g["min"])
        prov.add("perception", "tracklets", f"{len(g)} (median {dur.median():.1f} s)", "auto",
                 note="player identity is per tracklet; shirt numbers only where you typed them in the review")


if __name__ == "__main__":
    main()
