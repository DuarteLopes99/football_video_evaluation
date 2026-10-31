"""Regression tests (run: `python -m pytest tests -q` or `python tests/test_pipeline.py`).

They use the synthetic match generator, so they need no video and run in ~1 minute.
Thresholds are deliberately below the current scores so that only real regressions fail.
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from football_eval.analytics import run_analytics  # noqa: E402
from football_eval.config import load_config  # noqa: E402
from football_eval.evaluation import possession_accuracy  # noqa: E402
from football_eval.pitch import Homography, Pitch  # noqa: E402
from football_eval.synthetic import SynthConfig, SyntheticMatch, evaluate_events  # noqa: E402
from football_eval.xg import XGModel, feature_row, freeze_frame_features  # noqa: E402

_cache = {}


def _match(seed=3):
    if seed not in _cache:
        m = SyntheticMatch(SynthConfig(seed=seed, period_s=600)).generate()
        _cache[seed] = (m, run_analytics(m["tracks"], m["ball"], load_config(), root=str(ROOT)))
    return _cache[seed]


def test_homography_roundtrip():
    p = Pitch()
    lm = p.landmarks()
    names = ["corner_tl", "corner_tr", "corner_br", "corner_bl", "centre", "spot_l"]
    H_true = np.array([[12.0, 3.0, 200.0], [0.5, 6.0, 150.0], [0.0005, 0.004, 1.0]])
    img = Homography(H_true).to_image([lm[n] for n in names])
    H = Homography.fit(img, [lm[n] for n in names])
    assert H.reprojection_error(img, [lm[n] for n in names]) < 1e-3
    back = H.to_pitch(img)
    assert np.allclose(back, [lm[n] for n in names], atol=1e-3)


def test_periods_and_direction():
    m, r = _match()
    (a0, b0), (a1, b1) = r["ctx"].periods
    (g0, h0), (g1, h1) = m["periods"]
    assert abs(b0 - h0) < 30 and abs(a1 - g1) < 30
    assert r["ctx"].attacks_right[0][0] != r["ctx"].attacks_right[1][0]


def test_event_detection_quality():
    m, r = _match()
    s = evaluate_events(r["events"], m["gt_events"])
    assert s.loc["pass", "f1"] > 0.9
    assert s.loc["interception", "f1"] > 0.8
    assert s.loc["throw_in", "f1"] > 0.7
    assert s.loc["kickoff", "recall"] >= 0.75
    assert s.loc["shot", "precision"] > 0.8
    assert s.loc["goal", "recall"] == 1.0


def test_possession_accuracy():
    m, r = _match()
    pa = possession_accuracy(r["fc"], r["possession"], m["truth"])
    assert pa["frame_accuracy"] > 0.95
    assert abs(pa["team0_share_pred"] - pa["team0_share_gt"]) < 0.05


def test_xg_model_sanity():
    model = XGModel.load(ROOT / "models/xg_model.json")
    near = feature_row(99, 34, freeze_frame_features((99, 34), np.zeros((0, 2)), (104.5, 34)))
    far = feature_row(75, 34, freeze_frame_features((75, 34), np.zeros((0, 2)), (104.5, 34)))
    crowded = feature_row(99, 34, freeze_frame_features((99, 34), np.array([[101, 34], [102, 33], [100, 35]]), (104.5, 34)))
    p_near, p_far, p_crowd = model.predict([near, far, crowded])
    assert p_near > p_far and p_near > p_crowd
    assert 0.2 < p_near < 0.7 and p_far < 0.06
    assert abs(model.predict([near], situation=["penalty"])[0] - 0.71) < 0.1


def test_ptz_camera_geometry():
    """PTZ model: rays/projection round trip and the rotation+zoom homography between two poses."""
    import cv2
    from football_eval.ptz import PTZCamera, rotation, load_ptz
    cam = PTZCamera(np.array([51.7, -2.6, 3.9]))
    p, q = np.array([0.2, 0.05, 700.0]), np.array([0.35, 0.08, 900.0])
    uv = np.array([[100.0, 50], [600, 300], [320, 180]])
    assert np.allclose(cam.project_dirs(cam.rays(uv, p), p), uv, atol=1e-6)
    X = np.array([[60.0, 20], [80, 40], [95, 10]])
    ua, _ = cam.project_ours(X, p)
    ub, _ = cam.project_ours(X, q)
    H = cam.K(q[2]) @ rotation(q[0], q[1]) @ rotation(p[0], p[1]).T @ np.linalg.inv(cam.K(p[2]))
    assert np.allclose(cv2.perspectiveTransform(ua[None], H)[0], ub, atol=1e-6)
    # the venue file loads and carries the measured pitch size
    cam2, anchors = load_ptz(ROOT / "configs/camera_canedo_ptz.json")
    assert 95 < cam2.pitch.length < 115 and 50 < cam2.pitch.width < 75 and anchors
    # pitch frame <-> image homography is invertible and consistent with project_ours
    h = cam2.homography(next(iter(anchors.values())))
    pts = np.array([[52.7, 30.8], [40, 20]])
    assert np.allclose(h.to_pitch(h.to_image(pts)), pts, atol=1e-6)


# ----------------------------------------------------------------------------- manual setup / review / explainability
def test_setup_checklist_and_config():
    from football_eval.match_setup import (apply_to_config, checklist, load_setup, minute_to_video_t, new_setup,
                                           save_setup)
    import tempfile
    s = new_setup("tests/none.mp4", {"fps": 25.0})
    st = {c["item"]: c["status"] for c in checklist(s, ROOT)}
    assert st["video"] == "missing" and st["pitch size"] == "warn" and st["periods"] == "warn"
    s["pitch"].update(length=100.0, width=60.0, source="manual")
    s["periods"] = [{"start": 300.0, "end": 3050.0}, {"start": 3900.0, "end": 6700.0}]
    s["attacks_right_first_half"] = 1
    s["teams"][0]["name"], s["teams"][1]["name"] = "Home", "Away"
    s["calibration"]["landmark_frames"] = [{"t": 10.0, "frame": 250, "points": {
        "corner_tl": [10, 10], "corner_tr": [600, 12], "corner_br": [630, 340], "corner_bl": [5, 330],
        "centre": [318, 160], "halfway_top": [305, 11]}}]
    p = Path(tempfile.mkdtemp()) / "setup.json"
    save_setup(s, p)
    s2 = load_setup(p)
    st = {c["item"]: c["status"] for c in checklist(s2, ROOT)}
    assert st["pitch size"] == "ok" and st["periods"] == "ok" and st["attacking direction"] == "ok"
    assert st["calibration"] == "ok"
    cfg = load_config()
    log = apply_to_config(s2, cfg)
    assert cfg.pitch.length == 100.0 and cfg.pitch.fixed_size and cfg.match.halves[1] == [3900.0, 6700.0]
    assert cfg.match.home_attacks_right_first_half is False and cfg.teams.team_names == ["Home", "Away"]
    assert cfg.pitch.calibration_points and len(log) >= 5
    # referee's clock -> video time (middle of the minute)
    assert abs(minute_to_video_t(s2, "1") - 330.0) < 1e-6
    assert abs(minute_to_video_t(s2, "45+2") - (300 + 46 * 60 + 30)) < 1e-6
    assert abs(minute_to_video_t(s2, "46") - 3930.0) < 1e-6
    # a section pre-filled by someone else is not "ok" until confirmed
    from football_eval.match_setup import mark
    mark(s2, "direction", "assistant", "guessed")
    st = {c["item"]: c["status"] for c in checklist(s2, ROOT)}
    assert st["attacking direction"] == "warn"


def test_kit_classifier_seeded():
    """Clicked samples fix the team identity (no swap), referee and goalkeeper are recognised by their kit."""
    from football_eval.teams import KitClassifier, vote_roles
    rng = np.random.default_rng(0)
    c = {"team0": np.r_[1.0, 0, 0, 0], "team1": np.r_[0, 1.0, 0, 0], "referee": np.r_[0, 0, 1.0, 0],
         "gk0": np.r_[0, 0, 0, 1.0]}
    n = {"team0": 400, "team1": 400, "referee": 40, "gk0": 30}
    X, y, tid = [], [], []
    for k, (name, m) in enumerate(c.items()):
        X.append(m + 0.12 * rng.standard_normal((n[name], 4)))
        y += [name] * n[name]
        tid += list(k * 1000 + np.arange(n[name]) // 10)          # 10 crops per track
    X = np.vstack(X)
    samples = {name: [m + 0.05 * rng.standard_normal(4) for _ in range(3 if name.startswith("team") else 1)]
               for name, m in c.items()}
    kc = KitClassifier(samples).fit(X)
    codes = kc.predict(X)
    want = np.array([{"team0": 0, "team1": 1, "gk0": 2, "gk1": 3, "referee": 4}[v] for v in y])
    assert (codes == want).mean() > 0.95
    a = vote_roles(np.array(tid), codes)
    assert set(a[a.role == "referee"].track_id // 1000) == {2}
    assert set(a[a.role == "goalkeeper"].team) == {0}
    assert (a.decision == "majority").mean() > 0.95


def test_review_and_match_sheet():
    """Rejected / flipped / added events and the match sheet change the stats; every event is explained."""
    from football_eval.match_setup import new_setup
    from football_eval.review import event_key, validation_summary
    m, _ = _match()
    cfg = load_config()
    base = run_analytics(m["tracks"], m["ball"], cfg, root=str(ROOT))
    ev = base["events"]
    assert ev.why.str.len().min() > 10
    shots = ev[ev.type == "shot"].reset_index(drop=True)
    goals = ev[ev.type == "goal"].reset_index(drop=True)
    assert len(shots) >= 2 and len(goals) >= 1
    review = {"frame_checks": [{"frame": 1, "t": 0.1, "auto_ok": True, "lines": "good", "teams": "good"},
                               {"frame": 2, "t": 0.2, "auto_ok": True, "lines": "bad", "teams": "good"}],
              "events": [{"key": event_key(shots.iloc[0]), "t": float(shots.t[0]), "type": "shot",
                          "team": int(shots.team[0]), "verdict": "reject"},
                         {"key": event_key(shots.iloc[1]), "t": float(shots.t[1]), "type": "shot",
                          "team": int(shots.team[1]), "verdict": "wrong_team", "number": 9}],
              "added_events": [{"t": 100.0, "type": "yellow", "team": 0, "number": 4}]}
    setup = new_setup()
    setup["teams"][1 - int(shots.team[1])]["roster"] = [{"number": 9, "name": "Striker"}]
    gt = goals.groupby("team").size()
    setup["final_score"] = [int(gt.get(0, 0)), int(gt.get(1, 0))]
    setup["periods"] = [{"start": a, "end": b} for a, b in m["periods"]]
    setup["known_events"] = [{"type": "goal", "team": int(g.team), "t": float(g.t) + 20.0} for _, g in goals.iterrows()]
    setup["known_events"].append({"type": "red", "team": 1, "minute": "30", "number": 5})
    r = run_analytics(m["tracks"], m["ball"], cfg, root=str(ROOT), setup=setup, review=review)
    e2 = r["events"]
    n_sh = (e2.type == "shot").sum()
    assert n_sh == len(shots) - 1
    flipped = e2[(e2.type == "shot") & (np.abs(e2.t - shots.t[1]) < 0.05)].iloc[0]
    assert flipped.team == 1 - shots.team[1] and flipped.validation == "corrected" and flipped.player == "Striker"
    assert (e2[e2.type == "goal"].validation == "match_sheet_confirmed").all()
    assert ((e2.type == "red") & (e2.source == "match_sheet")).sum() == 1
    assert ((e2.type == "yellow") & (e2.source == "manual")).sum() == 1
    assert int(r["team_stats"].loc["red_cards"].sum()) == 1
    assert len(r["events_rejected"]) >= 1
    vs = validation_summary(review)
    assert vs.iloc[0].result.startswith("50 %")
    assert any(row["stage"] == "events" for row in r["provenance"].rows)
    # a match sheet that lists every goal rejects automatic goals that are not on it
    setup["final_score"], setup["known_events"] = [0, 0], []
    r3 = run_analytics(m["tracks"], m["ball"], cfg, root=str(ROOT), setup=setup)
    assert (r3["events"].type == "goal").sum() == 0
    assert (r3["events_rejected"].type == "goal").sum() == len(goals)


def test_shot_location_exclusions_and_grading():
    """Clicked shot locations recompute xG; excluded stretches drop events; the confidence grade follows the evidence."""
    import pandas as pd
    from football_eval.confidence import grade_metrics
    from football_eval.match_setup import new_setup
    from football_eval.review import event_key, lost_stretches, repair_window
    m, _ = _match()
    cfg = load_config()
    base = run_analytics(m["tracks"], m["ball"], cfg, root=str(ROOT))
    shots = base["events"][base["events"].type == "shot"].reset_index(drop=True)
    s0 = shots.iloc[0]
    tm = int(s0.team)
    right = base["ctx"].attacking_right(tm, s0.t)
    L, W = base["ctx"].pitch.length, base["ctx"].pitch.width
    spot = [L - 11.0, W / 2] if right else [11.0, W / 2]                 # penalty-spot distance, central
    review = {"events": [{"key": event_key(s0), "t": float(s0.t), "type": "shot", "team": tm, "verdict": "confirm",
                          "shot_xy": spot, "shot_frame": int(s0.frame)} ] + [
                         {"key": event_key(r), "t": float(r.t), "type": "shot", "team": int(r.team), "verdict": "confirm"}
                         for _, r in shots.iloc[1:].iterrows()]}
    setup = new_setup()
    setup["periods"] = [{"start": a, "end": b} for a, b in m["periods"]]
    setup["attacks_right_first_half"] = 0 if base["ctx"].attacks_right[0][0] else 1
    t_ex = float(shots.t.iloc[-1])
    setup["exclusions"] = [{"t0": t_ex - 1.0, "t1": t_ex + 1.0, "reason": "replay / not live"}]
    r = run_analytics(m["tracks"], m["ball"], cfg, root=str(ROOT), setup=setup, review=review)
    e = r["events"]
    moved = e[(e.type == "shot") & (np.abs(e.t - s0.t) < 0.05)].iloc[0]
    assert moved.location == "clicked" and "location=clicked" in moved.detail and 0.08 < moved.xg < 0.4
    assert not ((e.type == "shot") & (np.abs(e.t - t_ex) < 0.05)).any()            # inside the excluded stretch
    assert not ((e.t > t_ex - 1.0) & (e.t < t_ex + 1.0) & ~e.source.isin(["manual", "match_sheet"])).any()
    km = r["key_metrics"].set_index("metric")
    assert km.loc["shots", "confidence"] == "Medium"                               # every shot reviewed
    assert km.loc["xG", "confidence"] == "Low"                                     # but not every location clicked
    assert km.loc["passes", "confidence"] in ("Experimental", "Low")
    assert km.loc["goals", "confidence"] == "Low"                                  # no match sheet
    # the match sheet makes goals High; a failed frame check pulls team shape down
    g = r["events"][r["events"].type == "goal"].groupby("team").size()
    setup["final_score"] = [int(g.get(0, 0)), int(g.get(1, 0))]
    review["frame_checks"] = [{"frame": i, "t": i, "auto_ok": True, "lines": "bad" if i < 4 else "good"} for i in range(6)]
    r2 = run_analytics(m["tracks"], m["ball"], cfg, root=str(ROOT), setup=setup, review=review)
    km2 = r2["key_metrics"].set_index("metric")
    assert km2.loc["goals", "confidence"] == "High"
    assert km2.loc["def_line_height_m", "confidence"] == "Low" and "2/6" in km2.loc["def_line_height_m", "reason"]
    # lost camera stretches and the repair window
    fr = pd.DataFrame({"frame": np.arange(0, 600, 6), "t": np.arange(0, 600, 6) / 30.0})
    fr["cam_ok"] = ~((fr.t >= 5) & (fr.t <= 12))
    ls = lost_stretches(fr)
    assert len(ls) == 1 and abs(ls[0]["t0"] - 5.0) < 0.3 and abs(ls[0]["t1"] - 12.0) < 0.3
    w0, w1 = repair_window(fr, 5.0)
    assert w0 == 5.0 and abs(w1 - 17.0) < 0.3


def test_opportunities_flag_every_goal():
    from football_eval.opportunities import evaluate_opportunities, CATEGORIES
    m, r = _match()
    opps = r["opportunities"]
    assert len(opps) and set(opps.category) <= set(CATEGORIES)
    assert opps.t.is_monotonic_increasing and (opps.clip_end > opps.clip_start).all()
    assert opps[opps.category.isin(["Goal", "Penalty", "Big chance", "Shot on target"])].flag.all()
    e = evaluate_opportunities(opps, m["gt_events"])
    assert e["goal_recall_flag"] == 1.0                      # every true goal is in a flagged opportunity
    assert e["shot_recall"] >= 0.7 and e["flag_precision"] >= 0.8
    # a corner, the shot that follows it and its goal are one move
    from football_eval.opportunities import _group
    g = _group([dict(t=10, team=0), dict(t=13, team=0), dict(t=14, team=1), dict(t=30, team=0)], 6.0)
    assert [len(x) for x in g] == [2, 1, 1]
    assert "opportunities_flagged" in r["team_stats"].index


def test_storage_and_launcher_commands():
    import tempfile
    from football_eval import storage
    from football_eval.ui.launcher import Launcher
    with tempfile.TemporaryDirectory() as d:
        lib = Path(d) / "lib"
        (lib / "season").mkdir(parents=True)
        (lib / "season" / "2026-01-01_A_vs_B.mp4").write_bytes(b"0" * 10)
        (lib / ".2026-01-02_C_vs_D.mp4.icloud").write_bytes(b"0")          # evicted iCloud placeholder
        st_file = Path(d) / "storage.yaml"
        st_file.write_text(f"libraries:\n  - name: Test\n    path: {lib}\narchive: null\n")
        st = storage.load_storage(st_file)
        vids = storage.list_videos(st, include_legacy=False)
        names = {v["name"]: v for v in vids}
        assert set(names) == {"2026-01-01_A_vs_B.mp4", "2026-01-02_C_vs_D.mp4"}
        assert names["2026-01-02_C_vs_D.mp4"]["cloud_only"]
        found = storage.find_video("data/raw/2026-01-01_A_vs_B.mp4", st)     # moved file found by name
        assert found and found.endswith("season/2026-01-01_A_vs_B.mp4")
    spec = Launcher("python").build({"kind": "synthetic"})
    assert spec["cmd"][-3:] == ["--synthetic", "--out", "outputs/synthetic_demo"]
    from football_eval.ui.launcher import preselect_url, write_shortcut
    with tempfile.TemporaryDirectory() as d:
        p = write_shortcut(Path(d), "Analyse my match", ["--video", "a b.mp4", "--start"])
        txt = p.read_text()
        assert "start.command --video 'a b.mp4' --start" in txt and p.stat().st_mode & 0o100
    assert preselect_url("http://x/", {"video_id": "ab", "start": True, "setup": None}) == "http://x/?video_id=ab&start=True"
    assert "Step 1" in Launcher("python").doc("USER_GUIDE")["markdown"]


def test_live_preview_and_match_screen_data():
    import json as _json
    import tempfile
    from types import SimpleNamespace
    from football_eval.live import LivePreview
    from football_eval.perception import save_perception
    from football_eval.ui.server import app_html
    from football_eval.ui.viewer import RunOverlay
    cfg = load_config()
    with tempfile.TemporaryDirectory() as d:
        info = SimpleNamespace(duration_s=600.0)
        lp = LivePreview(d, Pitch(), info, cfg, every_s=0.0, total_frames=10, team_names=["UD Mansores", "Sanguedo"])
        img = np.full((360, 640, 3), 90, np.uint8)
        people = SimpleNamespace(xyxy=np.array([[100, 100, 120, 150.0]]), __len__=lambda self=None: 1)
        people = type("P", (), {"xyxy": np.array([[100, 100, 120, 150.0]]), "__len__": lambda self: 1})()
        H = Homography(np.array([[5.0, 0, 50], [0, 4.0, 40], [0, 0, 1]]))
        lp.update(3, 30, 1.0, img, people, np.array([7]), np.array([[10.0, 20.0]]), H, None, [(300, 200)],
                  labels=["team0"], manual=[True], tiles=[(0, 0, 640, 360)])
        st = _json.loads((Path(d) / "live" / "state.json").read_text())
        assert (Path(d) / "live" / "latest.jpg").stat().st_size > 1000
        assert st["n"] == 3 and st["progress"] == 0.3 and st["n_people"] == 1
        assert st["labelled"]["team0"] == 1 and st["clicked_tracks_in_view"] == 1 and st["n_tiles"] == 1
        assert lp.label_txt["team0"] == "MAN" and lp.label_txt["team1"] == "SAN"          # short names on the boxes
        # replay data of a processed (synthetic) run
        m, _ = _match()
        save_perception({"tracks": m["tracks"], "ball": m["ball"], "meta": {"video": {"width": 1280, "height": 720}}}, d)
        ov = RunOverlay(d)
        w = ov.window(100.0, 102.0)
        assert len(w["frames"]) >= 10 and all("b" in f and "p" in f for f in w["frames"])
        assert ov.info()["size"] == [1280, 720]
    h = app_html("/m").decode()
    assert '"/m/frame/"' in h and 'fetch("/m" + path' in h


def test_tiling_and_seam_merge():
    """Pitch-aware tiling: a 1080p pitch needs 2 detector tiles (was 6); a player split by a seam is joined."""
    from football_eval.detection import merge_cut_boxes, tile_plan
    t = tile_plan(1920, 1080, 1024, 96)
    assert len(t) == 2 and t[0][0] == 0 and t[-1][2] == 1920 and all(b[3] - b[1] == 1080 for b in t)
    assert t[0][2] - t[1][0] >= 96                                        # overlap kept
    r = tile_plan(1920, 1080, 1024, 96, roi=(0, 240, 1920, 1080))
    assert len(r) == 2 and all(b[1] == 240 for b in r)
    assert len(tile_plan(1920, 1080, 800, 96)) > 2                        # quality preset: more, smaller tiles
    # left half (cut at x=1000) + right half (cut at x=990) of one player, plus a whole player elsewhere
    boxes = np.array([[960, 400, 1000, 520], [990, 402, 1030, 522], [200, 300, 240, 400.0]])
    b, s_ = merge_cut_boxes(boxes, np.array([0.6, 0.7, 0.9]), np.array([True, True, False]))
    assert len(b) == 2 and np.allclose(b[0], [960, 400, 1030, 522]) and s_[0] == 0.7
    # a cut fragment inside a whole box is dropped
    b, _ = merge_cut_boxes(np.array([[100, 100, 140, 200], [100, 100, 130, 200.0]]), np.array([0.9, 0.5]),
                           np.array([False, True]))
    assert len(b) == 1


def test_kit_snap_votes_and_green_kit():
    """Loose clicked boxes snap to the detector box; a green kit on a green pitch is still recognised; the clicked
    track keeps its class unless its crop votes strongly disagree."""
    import cv2
    from football_eval.teams import (KitClassifier, TrackVotes, apply_manual, color_descriptor, grass_model,
                                     snap_box, vote_roles)
    dets = np.array([[100, 100, 130, 180], [300, 100, 330, 180.0]])
    assert np.allclose(snap_box([80, 80, 170, 200], dets), dets[0])      # loose box around player 0
    assert np.allclose(snap_box([105, 110, 125, 140], dets), dets[0])    # tiny box on his chest
    assert snap_box([600, 600, 650, 700], dets) is None
    # synthetic frame: artificial turf (hue ~42), green-shirt players (hue ~75, darker) and black/white stripes
    rng = np.random.default_rng(1)
    turf = np.zeros((360, 640, 3), np.uint8)
    turf[:] = cv2.cvtColor(np.uint8([[[42, 120, 130]]]), cv2.COLOR_HSV2BGR)[0, 0]
    turf = np.clip(turf.astype(int) + rng.integers(-8, 8, turf.shape), 0, 255).astype(np.uint8)
    green = cv2.cvtColor(np.uint8([[[75, 160, 90]]]), cv2.COLOR_HSV2BGR)[0, 0]

    def player(img, x, kind, light=1.0):
        crop = img[100:180, x:x + 30]
        if kind == "green":
            crop[12:44] = np.clip(green * light, 0, 255).astype(np.uint8)
        else:
            crop[12:44] = 30
            crop[12:44, ::4] = min(255, int(235 * light))
            crop[12:44, 1::4] = min(255, int(235 * light))
    img = turf.copy()
    g = grass_model(img)
    assert g is not None and abs(g[0] - 42) < 4
    S = {"team0": [], "team1": []}
    for i, light in enumerate((1.0, 0.7, 1.1)):                          # sun, shade, bright
        for cls, kind, x in (("team0", "green", 36 * i), ("team1", "stripes", 300 + 36 * i)):
            player(img, x, kind, light)
            S[cls].append(color_descriptor(img[112:144, x + 6:x + 24], g))
    assert all(d is not None for v in S.values() for d in v)              # the green shirt is not masked as grass
    kc = KitClassifier(S)
    assert kc.report["loo_accuracy"] == 1.0
    # tracks: votes + a clicked track
    tv = TrackVotes()
    tv.add([5, 5, 5, 6], [0, 0, 1, 1])
    assert tv.label(5) == 0 and tv.label(6) == 1 and tv.label(99, fallback=-1) == -1
    tv.manual[6] = 0
    assert tv.label(6) == 0                                               # your click wins in the live view
    a = vote_roles(np.array([5] * 3 + [6] * 12 + [7] * 12), np.array([0, 0, 1] + [0] * 12 + [1] * 12))
    m = apply_manual(a, {6: 1, 7: 1, 8: 4}).set_index("track_id")
    assert m.loc[7, "team"] == 1 and m.loc[7, "decision"] == "clicked in the setup"
    assert m.loc[6, "team"] == 0 and "disagree" in m.loc[6, "decision"]   # 12/12 votes say team 0: identity switch
    assert m.loc[8, "role"] == "referee"


def test_windows_and_near_side_visibility():
    """Only the periods are processed; the view's end on the camera side is measured and a ball leaving the
    picture there becomes a near-side throw-in instead of an 'unseen' stoppage."""
    import pandas as pd
    from football_eval.context import MatchContext
    from football_eval.events import dead_ball
    from football_eval.perception import foot_points, processing_windows, visible_near_y
    cfg = load_config()
    cfg.match.halves = [[300.0, 3000.0], [3010.0, 5800.0]]
    assert processing_windows(cfg, 6000.0) == [(300.0, 3000.0), (3010.0, 5800.0)]   # default: the game only
    cfg.video.period_margin_s = 20.0
    assert processing_windows(cfg, 6000.0) == [(280.0, 5820.0)]          # half-time of 10 s: one window
    cfg.match.halves = [[300.0, 3000.0], [3900.0, 6600.0]]
    assert processing_windows(cfg, 6700.0) == [(280.0, 3020.0), (3880.0, 6620.0)]
    cfg.video.only_periods = False
    assert processing_windows(cfg, 6700.0) == [(0.0, 6700.0)]
    p = Pitch(100, 64)
    # camera on the near side, high up: the bottom of a 1280x720 frame sees y = 50 m (14 m short of the touchline)
    lm = p.landmarks()
    names = ["corner_tl", "corner_tr", "centre", "halfway_top"]
    uv = np.array([[40, 120], [1240, 120], [640, 470], [640, 120.0]])
    H = Homography.fit(uv, [lm[n] for n in names])
    y_end = visible_near_y(H, p, (720, 1280))
    assert 30 < y_end < 64
    f, e = foot_points(np.array([[600, 650, 640, 720], [600, 300, 620, 350.0]]), (720, 1280))
    assert e.tolist() == [True, False] and f[0, 1] > 720                  # feet below the frame are estimated
    # dead-ball rule on a synthetic sequence: ball rolls to the bottom edge, unseen 6 s, players still
    t = np.arange(0, 20, 0.1)
    n = len(t)
    by = np.where(t < 5, y_end - 15 + 2.9 * t, np.nan)                   # last seen 0.6 m before the edge
    by = np.where(t > 11, y_end - 1.0 - 4.0 * (t - 11), by)              # thrown back in, moving away
    fc = pd.DataFrame(dict(frame=np.arange(n), t=t, bx=np.full(n, 50.0), by=by, valid=~np.isnan(by),
                           vis_near_y=np.full(n, y_end)))
    still = (np.repeat(t, 2) > 5) & (np.repeat(t, 2) < 11)
    sp = pd.DataFrame(dict(frame=np.repeat(np.arange(n), 2), speed=np.where(still, 0.5, 3.0)))
    ctx = MatchContext(p, [(0.0, 20.0)], [{0: True, 1: False}])
    dead, st = dead_ball(fc, sp, ctx, cfg)
    assert (st.reason == "out_of_view_near").any() and (st.side == "touchline_near").any()


# ----------------------------------------------------------------------------- manual input always used / people
def test_camera_from_landmark_clicks():
    """Closed-form camera from one frame of clicks finds the camera position; a mislabelled click is dropped; the
    joint solve over two frames reproduces the clicks."""
    from football_eval.ptz import PTZCamera, camera_from_homography, landmark_errors, solve_from_landmarks
    pitch = Pitch(100, 64)
    cam = PTZCamera(np.array([50.5, -3.0, 4.5]), 0.01, pitch=pitch)
    lm = pitch.landmarks()
    frames = []
    for pose in ([-0.75, 0.06, 620.0], [0.8, 0.05, 560.0]):
        uv, front = cam.project_ours(np.array(list(lm.values())), pose)
        ok = front & (uv[:, 0] > 5) & (uv[:, 0] < 635) & (uv[:, 1] > 5) & (uv[:, 1] < 355)
        pts = {n: list(map(float, u)) for n, u, k in zip(lm, uv, ok) if k}
        assert len(pts) >= 6
        frames.append((np.zeros((360, 640, 3), np.uint8), pts))
    C, roll, p, inl, out = camera_from_homography(frames[0][1], pitch, 640, 360)
    assert np.linalg.norm(C - cam.C) < 0.5 and abs(p[2] - 620) < 15 and not out
    bad = dict(frames[0][1])
    k0, k1 = list(bad)[:2]
    bad[k0], bad[k1] = bad[k1], bad[k0]                       # two landmark names swapped by the user
    *_, out = camera_from_homography(bad, pitch, 640, 360)
    assert set(out) & {k0, k1}
    cam2, P, errs, dropped = solve_from_landmarks(frames, pitch)
    assert np.linalg.norm(cam2.C - cam.C) < 1.0
    assert all(np.median(e) < 2.0 for e in errs)
    assert np.median(landmark_errors(cam2, P[1], frames[1][1])) < 2.0


def test_ptz_provider_any_resolution():
    """A venue camera solved at 640 x 360 tracks a 1080p video: the homography is scaled back to video pixels."""
    from football_eval.pitch import Homography as Hm
    from football_eval.ptz import PTZCamera, PTZProvider
    cam = PTZCamera(np.array([52.0, -2.5, 4.0]), pitch=Pitch(100, 64))
    prov = PTZProvider(cam, ignore=[[1500, 900, 1860, 1020]])
    pose = np.array([0.1, 0.06, 600.0])
    seen = {}

    def fake(frame_idx, image, boxes=None):
        seen["shape"], seen["boxes"] = image.shape, boxes
        prov.last_evidence, prov.last_image_motion = None, None
        return cam.homography(pose)
    prov._get = fake
    H = prov.get(0, np.zeros((1080, 1920, 3), np.uint8), np.array([[300.0, 600, 330, 690]]))
    assert seen["shape"][:2] == (360, 640) and np.allclose(seen["boxes"], [[100, 200, 110, 230]])
    assert np.allclose(prov.ignore[0], [500, 300, 620, 340])
    xy = np.array([[50.0, 32.0], [70.0, 10.0]])
    assert np.allclose(H.to_image(xy), 3 * Hm(cam.H_ours(*pose)).to_image(xy), atol=1e-6)


def test_people_types_and_periods():
    """Fans, staff and stewards are separated from the players; analytics keeps only the game periods; the setup
    of a video is found without --setup; stored kit descriptors are used without the video."""
    import json
    import tempfile
    import pandas as pd
    from football_eval.analytics import restrict_to_periods
    from football_eval.match_setup import find_setup, new_setup, save_setup
    from football_eval.people import frame_types, grass_roi, hivis_share, people_summary, track_types, zones
    p = Pitch(100, 64)
    xy = np.array([[50.0, 30.0], [50.0, 66.0], [50.0, -8.0], [50.0, -20.0], [np.nan, np.nan], [np.nan, np.nan]])
    foot = np.array([[0, 0]] * 4 + [[500.0, 300.0], [900.0, 700.0]])
    boxes = np.array([[0, 0, 10, 40]] * 4 + [[490, 200, 510, 300], [890, 600, 910, 700]], float)
    top = np.full(1280, 400.0)                                  # grass starts at row 400
    z = zones(xy, foot, boxes, p, top)
    assert z == ["pitch", "pitch", "touchline", "stands", "stands", "pitch"]
    img = np.zeros((720, 1280, 3), np.uint8)
    img[:, :640] = (0, 230, 255)                                # fluorescent yellow (BGR) on the left half
    assert hivis_share(img, [100, 100, 160, 260]) > 0.9 and hivis_share(img, [900, 100, 960, 260]) < 0.05
    t = frame_types(img, np.array([[100, 100, 160, 260], [900, 100, 960, 260], [900, 100, 960, 260]], float),
                    ["touchline", "touchline", "pitch"])
    assert t == ["steward", "staff", None]
    assert grass_roi(top, (720, 1280)) == (0, 270, 1280, 720)
    tr = pd.DataFrame(dict(frame=[1, 2, 3, 1, 2, 3, 1], t=[0.1, 0.2, 0.3] * 2 + [0.1], track_id=[5, 5, 5, 6, 6, 6, 7],
                           ptype=["spectator", "spectator", None, None, None, "staff", None],
                           cls="player", team=[0, 0, 0, 1, 1, 1, -1]))
    tt = track_types(tr)
    assert tt.to_dict() == {5: "spectator"}
    summ = people_summary(tr).set_index("type")
    assert summ.loc["player", "tracks"] == 2 and summ.loc["unknown", "tracks"] == 1
    ball = pd.DataFrame(dict(frame=[1, 2, 3], t=[0.1, 0.2, 0.3]))
    tr2, b2, cut = restrict_to_periods(tr, ball, [[0.15, 0.35]])
    assert cut == 1 and tr2.t.min() >= 0.15 and len(b2) == 2
    with tempfile.TemporaryDirectory() as d:
        s = new_setup("data/videos/💛 My match_1080p.mp4")
        save_setup(s, Path(d) / "match_setup_x.json")
        assert find_setup("/somewhere/else/💛 My match_1080p.mp4", d).name == "match_setup_x.json"
        assert find_setup("other.mp4", d) is None
    # kit samples clicked outside a cut clip still count (their descriptor is stored in the setup)
    from types import SimpleNamespace
    from football_eval.perception import build_kit_bank
    from football_eval.teams import DESC_DIM, TeamClassifier
    rng = np.random.default_rng(0)
    cfg = load_config()
    cfg.teams.kit_samples = {c: [dict(t=None, frame=-1, box=[0, 0, 10, 30], desc=list(rng.normal(m, 0.05, DESC_DIM)))
                                 for _ in range(3)] for c, m in (("team0", 0.0), ("team1", 1.0))}
    info = SimpleNamespace(width=1920, height=1080, fps=60.0, n_frames=100)
    kit, seeds, rep = build_kit_bank("missing.mp4", cfg, TeamClassifier(), None, info)
    assert kit is not None and rep["stored_descriptor"] == 6 and not seeds
    assert json.dumps(rep)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
