"""Build (and optionally execute) the evaluation notebooks from the cell sources below.

    python scripts/build_notebooks.py            # write notebooks without outputs
    python scripts/build_notebooks.py --execute  # also run them (in-process) and embed outputs

Keeping notebook sources here makes them diff-able and easy to regenerate after code changes.
"""
from __future__ import annotations

import argparse
import ast
import base64
import contextlib
import io
import json
import os
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NB_DIR = ROOT / "notebooks"

SETUP = r'''import sys, time, json, warnings
from pathlib import Path
ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")
import cv2, numpy as np, pandas as pd, matplotlib.pyplot as plt
pd.set_option("display.precision", 3); pd.set_option("display.width", 160)
pd.set_option("display.max_columns", 30); pd.set_option("display.max_rows", 120)
plt.rcParams.update({"figure.dpi": 90, "axes.spines.top": False, "axes.spines.right": False})'''

NB = {}

# ============================================================================ 01 perception
NB["01_perception_eval.ipynb"] = [
("md", r'''# 01 · Perception evaluation — detection, teams, pitch calibration

What this notebook checks, on **one frame** and then on a **short video segment**:

1. **Detection** of people and ball (default: YOLOX-S COCO through `cv2.dnn`, tiled — no PyTorch).
2. **Team assignment** by shirt colour clustering (KMeans on torso descriptors).
3. **Pitch calibration**: homography image ↔ pitch; reprojected lines must sit on the painted lines.
4. Optional **quantitative scores** against a hand-annotated frame (`data/annotations/*.json`):
   detection precision/recall @ IoU 0.5, team accuracy, landmark reprojection error.

Without a match video yet, it falls back to the real frame of the sibling `offiside_code` project
(AFA stream, Anadia FC – SVA) and that project's calibrated camera model.'''),
("code", SETUP + r'''
from football_eval.config import load_config
from football_eval.detection import build_detector
from football_eval.teams import TeamClassifier
from football_eval.pitch import Pitch, load_calibration, draw_pitch
from football_eval.video import read_frame
from football_eval.evaluation import detection_pr, team_accuracy, load_annotation'''),
("code", r'''VIDEO = ROOT / "data/videos/match.mp4"            # <- your full-match video
T_FRAME = 600.0                                  # seconds into the video for the single-frame checks
FALLBACK_IMAGE = ROOT.parent / "offiside_code/data/frame_decision.png"
CONFIG = ROOT / "configs/example_anadia.yaml"    # <- your config (calibration file, overlays, team names)
ANNOTATION = None                                # e.g. ROOT / "data/annotations/frame_0600.json"

cfg = load_config(CONFIG)
img = read_frame(str(VIDEO), T_FRAME) if VIDEO.exists() else cv2.imread(str(FALLBACK_IMAGE))
print("source:", (VIDEO if VIDEO.exists() else FALLBACK_IMAGE).name, "| frame", img.shape)
plt.figure(figsize=(12, 6.5)); plt.imshow(img[..., ::-1]); plt.axis("off"); plt.title("Input frame");'''),
("md", "## 1 · Detection"),
("code", r'''det_model = build_detector(cfg)
t0 = time.time(); det = det_model(img); dt = time.time() - t0
people, balls = det.of("player", "goalkeeper", "referee"), det.of("ball")
print(f"{len(people)} people, {len(balls)} ball candidates in {dt:.2f}s (tile={cfg.detection.tile})")
vis = img.copy()
for (x0, y0, x1, y1), c in zip(people.xyxy.astype(int), people.conf):
    cv2.rectangle(vis, (x0, y0), (x1, y1), (255, 255, 255), 3)
for (x0, y0, x1, y1) in balls.xyxy.astype(int):
    cv2.circle(vis, ((x0 + x1) // 2, (y0 + y1) // 2), 25, (0, 255, 255), 4)
plt.figure(figsize=(12, 6.5)); plt.imshow(vis[..., ::-1]); plt.axis("off"); plt.title("Detections (white) and ball (yellow)");'''),
("md", "## 2 · Team assignment (colour clustering)\nIn a real run the clustering is fitted on ~60 sampled frames and each *track* gets the majority label; here it is fitted on this single frame, which is the hardest case."),
("code", r'''tc = TeamClassifier(cfg.teams.backend)
feats = tc.describe(img, people.xyxy)
ok = [i for i, f in enumerate(feats) if f is not None]
tc.fit(np.array([feats[i] for i in ok]))
labels = np.full(len(people), -1); labels[ok] = tc.predict(np.array([feats[i] for i in ok]))
colors = {0: (255, 90, 30), 1: (0, 140, 255), -1: (200, 200, 200)}
vis = img.copy()
for k, ((x0, y0, x1, y1), l) in enumerate(zip(people.xyxy.astype(int), labels)):
    cv2.rectangle(vis, (x0, y0), (x1, y1), colors[int(l)], 4)
    cv2.putText(vis, str(k), (x0, y0 - 6), 0, 1.0, colors[int(l)], 2)
plt.figure(figsize=(12, 6.5)); plt.imshow(vis[..., ::-1]); plt.axis("off")
plt.title("Team clusters: blue = 0, orange = 1, grey = other (referee / outlier)")
print(pd.Series(labels).value_counts().rename("count").to_frame().T)'''),
("md", "## 3 · Pitch calibration (homography)"),
("code", r'''pitch = Pitch(cfg.pitch.length, cfg.pitch.width)
from football_eval.config import resolve_path
H = load_calibration(resolve_path(cfg.pitch.calibration_file), pitch)
vis = img.copy()
for ln in pitch.lines():
    cv2.polylines(vis, [H.to_image(ln).astype(np.int32)], False, (255, 0, 255), 4)
foot = people.foot
xy = H.to_pitch(foot)
fig, ax = plt.subplots(1, 2, figsize=(16, 5), gridspec_kw={"width_ratios": [1.6, 1]})
ax[0].imshow(vis[..., ::-1]); ax[0].axis("off"); ax[0].set_title("Pitch template re-projected (magenta)")
draw_pitch(ax[1], pitch)
for l in (0, 1, -1):
    m = labels == l
    ax[1].scatter(xy[m, 0], xy[m, 1], s=40, c=["#2a78d6", "#e8743b", "#999999"][l if l >= 0 else 2], label=f"team {l}", zorder=3)
if len(balls):
    bxy = H.to_pitch(balls.foot[:1]); ax[1].scatter(bxy[:, 0], bxy[:, 1], s=60, c="gold", edgecolors="k", zorder=4, label="ball")
ax[1].legend(loc="lower left", fontsize=8); ax[1].set_title("Minimap (pitch metres)")
off = ~pitch.in_pitch(xy[:, 0], xy[:, 1], cfg.pitch.offpitch_margin_m)
print(f"{off.sum()} detections are outside the pitch (bench / staff / crowd) and will be ignored")'''),
("md", "## 4 · Quantitative check against a hand-annotated frame\nCreate an annotation once (start from the template written below, fix it by hand) and set `ANNOTATION`. The scores make detector / clustering / calibration changes comparable."),
("code", r'''tmpl = {"image": "frame.png", "boxes": [{"xyxy": [round(float(v), 1) for v in b], "cls": "player", "team": int(l)}
                                       for b, l in zip(people.xyxy, labels)],
        "ball": [round(float(v), 1) for v in balls.foot[0]] if len(balls) else None,
        "landmarks": {"centre": [0, 0], "halfway_top": [0, 0], "circle_top": [0, 0], "circle_bottom": [0, 0]}}
out = ROOT / "data/annotations/template_from_detections.json"; out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(tmpl, indent=1)); print("annotation template ->", out.relative_to(ROOT))

if ANNOTATION:
    ann = load_annotation(ANNOTATION)
    gt_boxes = np.array([b["xyxy"] for b in ann["boxes"]])
    pr = detection_pr(people.xyxy, gt_boxes)
    print({k: v for k, v in pr.items() if k != "matches"})
    ta = team_accuracy([labels[i] for i, _ in pr["matches"]], [ann["boxes"][j]["team"] for _, j in pr["matches"]])
    print("team accuracy:", ta)
    lm = pitch.landmarks()
    names = [n for n, uv in ann["landmarks"].items() if uv != [0, 0]]
    if names:
        print("landmark reprojection error (px):", round(H.reprojection_error([ann["landmarks"][n] for n in names], [lm[n] for n in names]), 1))
else:
    print("No annotation set — skipping the quantitative scores.")'''),
("md", "## 5 · Short video segment (tracking + ball + pitch coordinates)\nRuns the full perception pipeline on the first `N` analysed frames of the video (if present)."),
("code", r'''from football_eval.perception import run_perception
N = 300   # analysed frames (30 s at 10 fps)
if VIDEO.exists():
    res = run_perception(str(VIDEO), cfg, max_frames=N)
    tr, ball = res["tracks"], res["ball"]
    print(f"{tr.track_id.nunique()} tracks | mean track length {tr.groupby('track_id').size().mean():.1f} frames | "
          f"ball visible {ball.visible.mean():.0%} | homography {res['frames'].has_h.mean():.0%} of frames")
    fig, ax = plt.subplots(figsize=(9, 6)); draw_pitch(ax, pitch)
    for tm, c in ((0, "#2a78d6"), (1, "#e8743b")):
        q = tr[tr.team == tm]; ax.scatter(q.x, q.y, s=2, c=c, alpha=0.3)
    ax.plot(ball.x, ball.y, c="k", lw=0.8); ax.set_title("Positions over the segment (ball in black)")
else:
    print("No video yet: put your match in data/videos/match.mp4 (or change VIDEO) to run this cell.")'''),
("md", r'''## Reading the results / what to improve next

* **Detection** — COCO YOLOX finds almost every player at this resolution, but also people outside the pitch
  (bench, staff, ball boys): they are removed by the pitch-coordinate filter. The ball is found when it is not
  occluded; a football-fine-tuned detector (YOLO11/RT-DETR trained on football data) is the biggest single upgrade.
* **Teams** — colour clustering separates the two kits well in sunlight, but **shadows** push players to "other"
  and a **dark referee kit** can join the team with dark shorts. Fixes, in order of effort: per-track majority vote
  over many frames (already in the pipeline) → SigLIP embeddings (`teams.backend: siglip`) → a small classifier
  trained on a few hundred labelled crops from your club's matches.
* **Calibration** — for a fixed camera, calibrate once per venue (`scripts/calibrate_points.py`, ≥ 6 well spread
  landmarks). If the operator pans/zooms, use `pitch.homography: flow` with a few anchors, or a pitch-keypoint model.'''),
]

# ============================================================================ 02 events
NB["02_event_detection_eval.ipynb"] = [
("md", r'''# 02 · Event-detection & possession evaluation (synthetic ground truth)

The analytics rules (possession, passes, shots, set pieces, fouls…) are evaluated on **synthetic matches**
that come with exact ground truth, so every rule change can be measured.
The generator reproduces single-camera failure modes: position jitter, ball dropouts & occlusions,
track-ID fragmentation and goalkeepers the colour clustering did not label.

Matching: predicted ↔ true events of the same type and team within ±2 s (±4 s for stoppage-based events).
For a real match, annotate events in a CSV (`t, type, team`) and use the same `evaluate_events`.'''),
("code", SETUP + r'''
from football_eval.synthetic import SyntheticMatch, SynthConfig
from football_eval.config import load_config
from football_eval.analytics import run_analytics
from football_eval.evaluation import evaluate_events, possession_accuracy
TYPES = ["pass", "interception", "ball_recovery", "shot", "goal", "kickoff", "throw_in", "corner", "goal_kick", "free_kick", "foul"]

def run(seed=0, period_s=900, cfg=None, **noise):
    m = SyntheticMatch(SynthConfig(seed=seed, period_s=period_s, **noise)).generate()
    r = run_analytics(m["tracks"], m["ball"], cfg or load_config(), root=str(ROOT))
    return m, r'''),
("md", "## 1 · One match in detail"),
("code", r'''t0 = time.time()
m, r = run(seed=0)
print(f"{len(m['tracks']):,} tracking rows, {len(m['gt_events'])} true events, analysed in {time.time()-t0:.1f}s")
ev_score = evaluate_events(r["events"], m["gt_events"], TYPES)
ev_score'''),
("code", r'''pa = possession_accuracy(r["fc"], r["possession"], m["truth"])
print({k: round(v, 3) for k, v in pa.items()})
gt = m["gt_events"]; ev = r["events"]
cmp = pd.DataFrame({
    "passes (true)": gt[gt.type == "pass"].groupby("team").size(),
    "passes (detected)": ev[ev.type == "pass"].groupby("team").size(),
    "completed (true)": gt[(gt.type == "pass") & (gt.outcome == "complete")].groupby("team").size(),
    "completed (detected)": ev[(ev.type == "pass") & (ev.outcome == "complete")].groupby("team").size(),
    "shots (true)": gt[gt.type == "shot"].groupby("team").size(),
    "shots (detected)": ev[ev.type == "shot"].groupby("team").size(),
    "goals (true)": gt[gt.type == "goal"].groupby("team").size(),
    "goals (detected)": ev[ev.type == "goal"].groupby("team").size(),
}).fillna(0).astype(int)
cmp.index = ["Team A", "Team B"][:len(cmp)]
cmp'''),
("md", "## 2 · Benchmark over several matches"),
("code", r'''rows = []
for seed in range(1, 5):
    m_, r_ = run(seed=seed, period_s=600)
    s = evaluate_events(r_["events"], m_["gt_events"], TYPES); s["seed"] = seed; rows.append(s.reset_index())
bench = pd.concat(rows)
agg = bench.groupby("type")[["n_gt", "tp", "n_pred"]].sum()
agg["precision"] = agg.tp / agg.n_pred; agg["recall"] = agg.tp / agg.n_gt
agg["f1"] = 2 * agg.precision * agg.recall / (agg.precision + agg.recall)
agg = agg.loc[[t for t in TYPES if t in agg.index]]
agg'''),
("code", r'''fig, ax = plt.subplots(figsize=(10, 3.6))
x = np.arange(len(agg)); w = 0.38
ax.bar(x - w/2, agg.precision, w, label="precision", color="#2a78d6")
ax.bar(x + w/2, agg.recall, w, label="recall", color="#e8743b")
ax.set_xticks(x, agg.index, rotation=30, ha="right"); ax.set_ylim(0, 1.05); ax.legend(frameon=False, ncol=2)
ax.set_title("Event detection on synthetic matches (4 × 20 min)");'''),
("md", "## 3 · Robustness to perception noise\n`clean` ≈ well-tuned perception; `realistic` ≈ a club stream; `hard` ≈ poor light / low resolution, IDs switching every ~30 s and goalkeepers not recognised."),
("code", r'''levels = {
    "clean":     dict(noise_player_m=0.15, noise_ball_m=0.2, ball_dropout=0.03, occlusion_per_min=1),
    "realistic": dict(noise_player_m=0.4, noise_ball_m=0.5, ball_dropout=0.12, occlusion_per_min=4, fragment_s=60),
    "hard":      dict(noise_player_m=0.8, noise_ball_m=0.9, ball_dropout=0.25, occlusion_per_min=8, fragment_s=30,
                      label_goalkeepers=False),
}
res = {}
for name, kw in levels.items():
    f1s = []
    for seed in (21, 22):
        m_, r_ = run(seed=seed, period_s=600, **kw)
        f1s.append(evaluate_events(r_["events"], m_["gt_events"], TYPES).f1)
        f1s[-1]["possession_acc"] = possession_accuracy(r_["fc"], r_["possession"], m_["truth"])["frame_accuracy"]
    res[name] = pd.concat(f1s, axis=1).mean(axis=1)
rob = pd.DataFrame(res)
rob'''),
("code", r'''ax = rob.drop("possession_acc").plot.bar(figsize=(10, 3.6), color=["#2a78d6", "#7fb0e8", "#e8743b"], width=0.8)
ax.set_ylim(0, 1.05); ax.set_ylabel("F1"); ax.set_title("F1 per event type vs noise level"); ax.legend(frameon=False, ncol=3)
plt.xticks(rotation=30, ha="right");'''),
("md", "## 4 · Sensitivity to the control radius (`possession.control_radius_m`)"),
("code", r'''sens = []
for rad in (0.8, 1.2, 1.5, 2.0, 2.5):
    cfg = load_config(possession=dict(control_radius_m=rad, duel_radius_m=rad))
    m_, r_ = run(seed=31, period_s=600, cfg=cfg, **levels["realistic"])
    s = evaluate_events(r_["events"], m_["gt_events"], ["pass", "interception", "shot"])
    sens.append(dict(radius=rad, pass_f1=s.loc["pass", "f1"], interception_f1=s.loc["interception", "f1"],
                     shot_f1=s.loc["shot", "f1"],
                     possession_acc=possession_accuracy(r_["fc"], r_["possession"], m_["truth"])["frame_accuracy"]))
sens = pd.DataFrame(sens).set_index("radius"); sens'''),
("md", r'''## 5 · Evaluating on a real match
1. Annotate the events you care about while watching (e.g. in a spreadsheet): `t` (video seconds), `type`
   (pass, shot, goal, corner, throw_in, free_kick, foul, …), `team` (0/1 as in `team_stats.csv`).
2. Save as `data/annotations/events_<match>.csv`, then:

```python
gt = pd.read_csv(ROOT / "data/annotations/events_match1.csv")
pred = pd.read_csv(ROOT / "outputs/match1/events.csv")
evaluate_events(pred, gt)
```
Start with goals, shots, corners and throw-ins for one half — 45 minutes of annotation is enough to tune the rules.

**Caveat** — synthetic scores are an *upper bound*: the generator is simpler than real football (no aerial balls,
no deflections, no crowding at set pieces). Their value is to catch regressions and to compare rule variants.'''),
]

# ============================================================================ 03 report
NB["03_match_report_walkthrough.ipynb"] = [
("md", r'''# 03 · Match report walkthrough — team stats, player tracklets, timeline

Runs the analytics on a perception output folder (`outputs/<match>/tracks.csv`, `ball.csv`) — or on a
synthetic match if you have not processed a video yet — and walks through every output table and figure.
Definitions and reliability levels of each statistic: `docs/STATS_RULEBOOK.md`.'''),
("code", SETUP + r'''
from football_eval.config import load_config
from football_eval.perception import load_perception
from football_eval.analytics import run_analytics, save_analytics
from football_eval.report import shot_map, possession_plot, heatmaps

MATCH_DIR = ROOT / "outputs/match1"          # <- output folder of run_analysis.py --video ...
CONFIG = ROOT / "configs/default.yaml"
cfg = load_config(CONFIG)
if (MATCH_DIR / "tracks.csv").exists():
    p = load_perception(str(MATCH_DIR)); tracks, ball = p["tracks"], p["ball"]
    sb = p.get("meta", {}).get("scoreboard_changes", [])
else:
    from football_eval.synthetic import SyntheticMatch, SynthConfig
    m = SyntheticMatch(SynthConfig(seed=7, period_s=1200, fragment_s=90)).generate()
    tracks, ball, sb = m["tracks"], m["ball"], []
    MATCH_DIR = ROOT / "outputs/synthetic_walkthrough"
    print("No perception output found -> using a synthetic match (2 × 20 min)")
r = run_analytics(tracks, ball, cfg, sb, root=str(ROOT))
ctx = r["ctx"]
print("periods (video s):", [(round(a), round(b)) for a, b in ctx.periods], "| attacks right:", ctx.attacks_right)'''),
("md", "## Team statistics"),
("code", r'''r["team_stats"].round(2)'''),
("md", "## Timeline of major events\n`video_time` is the position in the file, `clock` the football clock (2nd half starts at 45')."),
("code", r'''tl = r["timeline"]
tl[tl.priority <= 3][["video_time", "clock", "team", "description"]]'''),
("code", r'''fig = shot_map(r["events"], ctx)'''),
("code", r'''fig = possession_plot(r["possession_timeline"], ctx, r["events"])'''),
("code", r'''fig = heatmaps(r["tracks"], ctx)'''),
("md", "## Player statistics (per tracklet)\nWithout jersey-number recognition a real player is split into several tracklets — use these for physical\nprofiles and as the input to future ID merging, not as final player stats."),
("code", r'''r["player_stats"].head(12).round(2)'''),
("md", "## Event log (first rows) and stoppages"),
("code", r'''r["events"][["clock", "type", "team", "outcome", "x", "y", "xg", "detail"]].head(15)'''),
("code", r'''st = r["stoppages"]
print(f"{len(st)} stoppages; ball in play {100 * (1 - r['dead'].mean()):.0f}% of the analysed time")
st.groupby("reason").dur.describe()[["count", "mean", "50%", "max"]]'''),
("md", "## Save everything (CSV + JSON + HTML report)"),
("code", r'''rep = save_analytics(r, str(MATCH_DIR), make_report=True)
print("report ->", rep.relative_to(ROOT))
print(sorted(p.name for p in MATCH_DIR.iterdir()))'''),
]

# ============================================================================ 04 xG
NB["04_xg_model_eval.ipynb"] = [
("md", r'''# 04 · Expected-goals model — training data, validation, calibration

Logistic-regression xG trained on **StatsBomb open data** (8.8k shots, 351 matches) using only features a
single camera can observe at the moment of the shot: location, goal angle, goalkeeper position, opponents
in the shooting triangle and pressure. Validation is grouped by match (5-fold), and compared with StatsBomb's
own xG on the same shots (which also uses body part, technique and shot height).

Data: StatsBomb open data — https://github.com/statsbomb/open-data (attribution "Data: StatsBomb").'''),
("code", SETUP + r'''
sys.path.insert(0, str(ROOT / "scripts"))
from train_xg import build_table, cv_eval, fit_lr
from football_eval.xg import XGModel, BASIC, FULL, feature_row, freeze_frame_features
from football_eval.pitch import Pitch, draw_pitch
shots_path = ROOT / "data/xg/shots_raw.jsonl"
df = build_table(shots_path)
model = XGModel.load(ROOT / "models/xg_model.json")
print(f"{len(df):,} shots | goal rate {df.goal.mean():.3f}")
df.groupby("type").agg(shots=("goal", "size"), goal_rate=("goal", "mean")).sort_values("shots", ascending=False)'''),
("md", "## Cross-validated performance (grouped by match)"),
("code", r'''op = df[df.type == "Open Play"].reset_index(drop=True)
res, preds = {}, {}
for name, feats in (("basic (dist+angle)", BASIC), ("full (+GK, defenders, pressure)", FULL)):
    m, p = cv_eval(op, feats); res[name] = m; preds[name] = p
sbm = op.dropna(subset=["sb_xg"])
from sklearn.metrics import roc_auc_score, brier_score_loss, log_loss
res["StatsBomb xG (reference)"] = dict(auc=roc_auc_score(sbm.goal, sbm.sb_xg), brier=brier_score_loss(sbm.goal, sbm.sb_xg),
                                       logloss=log_loss(sbm.goal, sbm.sb_xg), mean_pred=sbm.sb_xg.mean(), goal_rate=sbm.goal.mean())
pd.DataFrame(res).T'''),
("code", r'''fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
bins = np.array([0, .03, .06, .1, .15, .2, .3, .45, .7, 1])
for (name, p), c in zip(list(preds.items()) + [("StatsBomb xG", op.sb_xg.to_numpy())], ["#2a78d6", "#e8743b", "#888888"]):
    b = np.digitize(p, bins) - 1
    cal = pd.DataFrame({"b": b, "p": p, "y": op.goal}).groupby("b").agg(p=("p", "mean"), y=("y", "mean"), n=("y", "size"))
    ax[0].plot(cal.p, cal.y, "o-", c=c, label=name)
ax[0].plot([0, .8], [0, .8], "k--", lw=0.8); ax[0].set_xlabel("predicted xG"); ax[0].set_ylabel("observed goal rate")
ax[0].legend(frameon=False, fontsize=8); ax[0].set_title("Calibration (reliability diagram)")
coef = pd.Series(model.p["full"]["coef"], index=model.p["full"]["features"]).sort_values()
ax[1].barh(coef.index, coef.values, color=["#e8743b" if v < 0 else "#2a78d6" for v in coef.values])
ax[1].set_title("Full model — standardised coefficients"); plt.tight_layout()'''),
("md", "## xG surface (no defenders, goalkeeper on his line)"),
("code", r'''pitch = Pitch()
xs, ys = np.linspace(70, 104.5, 70), np.linspace(10, 58, 97)
Z = np.zeros((len(ys), len(xs)))
for i, y in enumerate(ys):
    rows = [feature_row(x, y, freeze_frame_features((x, y), np.zeros((0, 2)), (104.5, 34))) for x in xs]
    Z[i] = model.predict(rows, "full")
fig, ax = plt.subplots(figsize=(7, 6)); draw_pitch(ax, pitch, color="#666666")
im = ax.imshow(Z, extent=[xs[0], xs[-1], ys[-1], ys[0]], cmap="viridis", vmin=0, vmax=0.5, alpha=0.9, zorder=0)
ax.set_xlim(68, 106); ax.set_ylim(60, 8); plt.colorbar(im, ax=ax, shrink=0.7, label="xG")
ax.set_title("Open-play xG by location");'''),
("md", "## Effect of the defensive situation (shot from the penalty spot, 11 m)"),
("code", r'''rows, labels = [], []
spot = (105 - 11, 34)
for n_def in (0, 1, 2, 3):
    opp = np.array([[spot[0] + 3 + k, 34 + (-1) ** k * 0.8 * k] for k in range(n_def)]).reshape(-1, 2)
    for gk_name, gk in (("GK on line", (104.6, 34)), ("GK off line (3 m)", (102.0, 34))):
        rows.append(feature_row(*spot, freeze_frame_features(spot, opp, gk))); labels.append((n_def, gk_name))
eff = pd.Series(model.predict(rows, "full"), index=pd.MultiIndex.from_tuples(labels, names=["defenders in triangle", "goalkeeper"])).unstack()
eff.round(3)'''),
("md", r'''## Notes for video use
* **Penalties** use a constant (empirical conversion in the training data); **direct free kicks** are scaled by
  the empirical ratio stored in `models/xg_model.json`.
* **Headers** cannot be told apart reliably from one wide camera, so the model is trained *without* body part:
  its predictions are averages over feet and headers for each situation.
* The geometric features come from the homography; an airborne ball is projected too far from the camera, but
  shots are taken at foot level, so the shot *location* (ball at the striker's foot) is well estimated.
* Retrain at any time with `python scripts/train_xg.py` (see the script header for re-downloading the data).'''),
]

# ============================================================================ 05 Canedo recording
NB["05_canedo_recording_eval.ipynb"] = [
("md", r'''# 05 · Real match — UD Mansores vs Canedo FC (CFC TV stream)

First real recording in the target format: **one panning/zooming camera, 640 × 360, no score bug, no audio**
(full profile in `docs/RECORDING_PROFILE.md`). This notebook checks, on a processed segment:

1. the **PTZ camera tracker** (is the pitch template on the painted lines? how often? does it drift?),
2. **detection + team assignment** at this resolution,
3. **positions on the pitch** (are players inside the measured pitch?),
4. the **ball** (motion-compensated candidates) and what the analytics can say from this segment.

Inputs are produced by `run_analysis.py --video ... --config configs/canedo_segment.yaml --out outputs/canedo_segment`.'''),
("code", SETUP + r'''
from football_eval.config import load_config, resolve_path
from football_eval.perception import load_perception
from football_eval.ptz import load_ptz, draw_template
from football_eval.pitch import Pitch, draw_pitch
from football_eval.video import read_frame
VIDEO = ROOT / "data/videos/2026-10-18_UD_Mansores_vs_Canedo_FC_J2.mp4"
OUT = ROOT / "outputs/canedo_segment"
cfg = load_config(ROOT / "configs/canedo_segment.yaml")
cam, anchors = load_ptz(resolve_path(cfg.pitch.calibration_file))
pitch = cam.pitch
p = load_perception(str(OUT))
tr, ball, frames = p["tracks"], p["ball"], p["frames"]
camlog = pd.read_csv(OUT / "camera_log.csv")
camlog["t"] = frames.set_index("frame").t.reindex(camlog.frame).to_numpy()
print(f"camera: C = {np.round(cam.C, 2)} m, roll = {np.degrees(cam.roll):.2f} deg, pitch {pitch.length:.1f} x {pitch.width:.1f} m")
print(f"segment {frames.t.min()/60:.1f}-{frames.t.max()/60:.1f} min of the video, {len(frames)} analysed frames, {len(tr):,} person detections")'''),
("md", "## 1 · Camera tracking — template overlay on sampled frames"),
("code", r'''COL = {0: (214, 120, 42), 1: (59, 116, 232), -1: (170, 170, 170)}
ts = np.linspace(frames.t.min() + 5, frames.t.max() - 5, 8)
fig, axes = plt.subplots(4, 2, figsize=(15, 16))
for ax, t in zip(axes.ravel(), ts):
    k = int(np.argmin(np.abs(camlog.t - t)))
    r = camlog.iloc[k]; f = int(r.frame); tt = float(r.t)
    img = read_frame(str(VIDEO), tt)
    v = draw_template(img, cam, (r.pan, r.tilt, r.f)) if np.isfinite(r.pan) else img
    for _, q in tr[tr.frame == f].iterrows():
        c = (90, 90, 90) if q.cls == "outside" else COL.get(int(q.team), COL[-1])
        cv2.rectangle(v, (int(q.x0), int(q.y0)), (int(q.x1), int(q.y1)), c, 1)
    ax.imshow(v[..., ::-1]); ax.axis("off")
    ax.set_title(f"{tt/60:.1f} min - line error {r.line_err:.1f}px - {r.src}", fontsize=9)
plt.tight_layout()'''),
("code", r'''e = camlog.line_err
fig, ax = plt.subplots(3, 1, figsize=(12, 7), sharex=True)
x = camlog.t / 60
ax[0].plot(x, e, ".", ms=2, color="#2a78d6"); ax[0].axhline(3, color="#e8743b", lw=0.8); ax[0].set_ylabel("line error (px)")
ax[0].set_ylim(0, 12)
ax[1].plot(x, np.degrees(camlog.pan), color="#444"); ax[1].set_ylabel("pan (deg)")
ax[2].plot(x, camlog.f, color="#444"); ax[2].set_ylabel("focal (px)"); ax[2].set_xlabel("video minute")
plt.tight_layout()
ec = e.dropna()
summary = pd.Series({"frames": len(camlog), "with line check": len(ec),
                     "median error px": float(ec.median()), "checked frames < 3 px": float((ec < 3).mean()),
                     "checked frames < 5 px": float((ec < 5).mean()),
                     "relocalisations": int((camlog.src == "relocalized").sum()),
                     "global recoveries": int((camlog.src == "global").sum())})
summary.to_frame("camera tracker").T'''),
("md", "## 2 · Detection and teams"),
("code", r'''inside = tr[tr.cls != "outside"]
per_frame = inside.groupby(["frame", "team"]).size().unstack(fill_value=0)
print("people per frame (median) by team label:", per_frame.median().to_dict())
print("share of detections outside the pitch (bench, crowd, staff):", round(float((tr.cls == "outside").mean()), 3))
print(f"tracks with a team label: {inside[inside.team >= 0].track_id.nunique()} of {inside.track_id.nunique()}")
fig, ax = plt.subplots(figsize=(12, 3))
tt = frames.set_index("frame").t.reindex(per_frame.index) / 60
for tm, c in ((0, "#2a78d6"), (1, "#e8743b"), (-1, "#999999")):
    if tm in per_frame:
        ax.plot(tt, per_frame[tm].rolling(10, min_periods=1).mean(), color=c, label=f"team {tm}")
ax.set_ylabel("visible players"); ax.set_xlabel("video minute"); ax.legend(frameon=False, ncol=3);'''),
("md", "## 3 · Positions on the measured pitch"),
("code", r'''fig, axes = plt.subplots(1, 2, figsize=(14, 4.8))
for tm, ax in zip((0, 1), axes):
    q = inside[(inside.team == tm) & inside.x.notna()]
    draw_pitch(ax, pitch)
    ax.hexbin(q.x, q.y, gridsize=(30, 16), extent=(0, pitch.length, 0, pitch.width),
              cmap="Blues" if tm == 0 else "Oranges", mincnt=1, alpha=0.85)
    ax.set_ylim(pitch.width + 4, -4); ax.set_title(f"team {tm}: {len(q):,} positions")
pos = inside[inside.x.notna()]
ok = pitch.in_pitch(pos.x, pos.y, 1.0).mean()
print(f"frames with a usable camera pose: {frames.cam_ok.mean():.0%}" if "cam_ok" in frames else "")
print(f"{ok:.1%} of the player positions kept fall inside the pitch (+1 m)")'''),
("md", "## 4 · Ball"),
("code", r'''print(f"a ball candidate was kept in {ball.visible.mean():.0%} of frames (gap-filled: {ball.valid.mean():.0%}) - "
      "candidates are noisy at 640x360, so the track below is indicative")
fig, ax = plt.subplots(figsize=(8, 5)); draw_pitch(ax, pitch)
b = ball[ball.valid]
ax.scatter(b.x, b.y, s=3, c=b.t, cmap="viridis"); ax.set_title("ball track (colour = time)");'''),
("md", "## 5 · Analytics on the segment (indicative)"),
("code", r'''from football_eval.analytics import run_analytics
r = run_analytics(tr, ball, cfg, root=str(ROOT))
r["team_stats"].loc[["possession_%", "passes", "pass_accuracy_%", "shots", "xG", "throw_ins",
                     "distance_covered_km(tracked)", "block_depth_m", "block_width_m", "team_centroid_x_m"]].round(2)'''),
("md", r'''## Reading the results
* **Camera tracking** is the foundation: when the template sits on the lines (error below ~3 px), positions are good
  to about 1 m in the near half and a few metres near the far touchline (low camera, 640 x 360). Fast pans and long
  spells without visible markings are where drift starts; relocalisation recovers most of it.
* **Teams**: yellow vs green-black separate well in sunlight; shade and the 2nd half (sunset) are the hard part.
* **Ball**: the main limitation at 640 x 360 - candidates are noisy, so possession, passes and shots from this
  recording are *indicative only*. A 720p/1080p download of the same stream would change this the most.
* Position-based statistics (team shape, territory, distance covered while in view) are already usable.'''),
]


# ============================================================================ 06 manual validation
NB["06_manual_validation_eval.ipynb"] = [
("md", r"""# 06 · Which factors are worth doing by hand?

The pipeline can guess everything automatically. This notebook checks, **with numbers from the real Canedo recording**,
where a person's input (setup UI) or check (review UI) pays off — and where it does not. Rule used:
*manual when the automatic estimate is unreliable, an error spreads to many statistics or fails silently, and a person
can give it in seconds or minutes.*

| Section | Factor | Question |
|---|---|---|
| 1 | shirt numbers / player identity | can a person (or OCR) label players per tracklet? |
| 2 | team kits | do a few clicked kit samples beat unsupervised colour clustering? |
| 3 | pitch size and positions | how much does a position error move xG? |
| 4 | direction / periods | how fragile is the automatic inference? |
| 5 | calibration and key events | what does a 10-minute review measure? |
| 6 | the Canedo setup file | what is pre-filled, what is left |

Written up in `docs/MANUAL_VALIDATION.md`."""),
("code", SETUP + r"""
from football_eval.config import load_config
from football_eval.match_setup import load_setup, checklist
from football_eval.review import load_review, validation_summary, RunGeometry
from football_eval.teams import KitClassifier, TeamClassifier, vote, vote_roles, torso_crop, color_descriptor, grass_model
from football_eval.video import iter_frames, read_frame
VIDEO = ROOT / "data/videos/2026-10-18_UD_Mansores_vs_Canedo_FC_J2.mp4"
RUN = ROOT / "outputs/canedo_segment"
SETUP_FILE = ROOT / "configs/match_setup_canedo_J2.json"
tr = pd.read_csv(RUN / "tracks.csv")
setup = load_setup(SETUP_FILE)
HAVE_VIDEO = VIDEO.exists()
print("video available:", HAVE_VIDEO, "| tracking rows:", len(tr))"""),
("md", """## 1 · Shirt numbers: tracklets are short and players are tiny
A per-player statistic needs to know *who* each tracklet is. Tracking at 640 × 360 with a panning camera fragments
identities constantly, and the shirt number is a few pixels tall."""),
("code", r"""q = tr[(tr.track_id >= 0) & (tr.cls != "outside")]
g = q.groupby("track_id").agg(t0=("t", "min"), t1=("t", "max"), n=("t", "size"), h=("y1", "median"))
g["dur"] = g.t1 - g.t0
h_box = (q.y1 - q.y0)
minutes = (q.t.max() - q.t.min()) / 60
cov = g.n.sort_values(ascending=False).cumsum() / g.n.sum()
stats = pd.Series({
    "segment minutes": round(minutes, 1),
    "tracklets": len(g), "tracklets per match (x 9)": int(len(g) * 90 / minutes),
    "median tracklet duration s": round(g.dur.median(), 1), "95th pct duration s": round(g.dur.quantile(.95), 1),
    "player-time covered by 30 longest": f"{100 * cov.iloc[29]:.0f} %",
    "player-time covered by 200 longest": f"{100 * cov.iloc[199]:.0f} %",
    "median box height px": round(h_box.median(), 1),
    "shirt number height px (~15 % of body)": round(0.15 * h_box.median(), 1)})
fig, ax = plt.subplots(1, 3, figsize=(15, 3.4))
ax[0].hist(g.dur.clip(upper=30), bins=30, color="#2a78d6"); ax[0].set_xlabel("tracklet duration (s, clipped at 30)"); ax[0].set_title("Tracklets are short")
ax[1].plot(np.arange(1, len(cov) + 1), cov.to_numpy(), color="#e8743b"); ax[1].set_xscale("log"); ax[1].set_xlabel("K longest tracklets"); ax[1].set_ylabel("share of player-time"); ax[1].set_title("Labelling the longest K")
ax[2].hist(h_box.clip(upper=100), bins=40, color="#666"); ax[2].set_xlabel("person box height (px)"); ax[2].set_title("Players are ~25 px tall")
plt.tight_layout()
stats.to_frame("value")"""),
("md", """**Verdict:** labelling shirt numbers per tracklet is not feasible (≈ 19k tracklets per match, unreadable digits).
Numbers are typed only for **key events** in the review (goals, shots, cards) → `player_stats_by_number.csv`.
Automatic player identity needs higher resolution (720p+) + jersey OCR + tracklet merging (SOTA roadmap)."""),
("md", """## 2 · Team kits: clicks vs unsupervised clustering
96 random tracklets (≥ 3 s) were labelled by eye from crops (`data/annotations/canedo_segment_team_labels.json`;
27 were too blurred / shaded to tell). Descriptors are recomputed from the video for every 5th analysed frame
(same as the perception), then the methods are compared: the stored labels (unsupervised + typed BGR kit colours),
pure unsupervised clustering, and the seeded classifier with the clicked samples of `match_setup_canedo_J2.json`."""),
("code", r"""L = json.load(open(ROOT / "data/annotations/canedo_segment_team_labels.json"))["labels"]
truth = {int(k): v for k, v in L.items()}
def score(team, role=None):
    out = {}
    for c in "YGRK":
        ids = [t for t, y in truth.items() if y == c]
        if c in "YG":
            ok = [team.get(t, -1) == (0 if c == "Y" else 1) for t in ids]
            wrong = [team.get(t, -1) == (1 if c == "Y" else 0) for t in ids]
            out[f"{c} right"] = f"{sum(ok)}/{len(ids)}"; out[f"{c} other team"] = sum(wrong)
        elif c == "R":
            out["referee kept out of the teams"] = f"{sum(team.get(t, -1) == -1 for t in ids)}/{len(ids)}"
        elif role is not None:
            out["goalkeeper recognised"] = f"{sum(role.get(t) == 'goalkeeper' for t in ids)}/{len(ids)}"
    yg = [t for t, y in truth.items() if y in "YG"]
    out["outfield accuracy"] = round(float(np.mean([team.get(t, -1) == (0 if truth[t] == "Y" else 1) for t in yg])), 3)
    return out
res = {"stored run (unsupervised + typed kit colours)": score(tr.groupby("track_id").team.agg(lambda s: s.mode().iloc[0]).to_dict())}
if HAVE_VIDEO:
    fr = sorted(tr.frame.unique()); keep = set(fr[::5]); rows = []
    t0 = time.time()
    from football_eval.video import probe
    fps = probe(str(VIDEO)).fps                      # start on the run's own frame grid (stride = fps / 5)
    for f, t, img in iter_frames(str(VIDEO), 5, (tr.frame.min() + 0.01) / fps, tr.t.max() + 0.1):
        if f not in keep: continue
        g = grass_model(img)                             # pitch colour of this frame (same as the perception)
        for _, r in tr[(tr.frame == f) & (tr.track_id >= 0) & (tr.cls != "outside")].iterrows():
            c = torso_crop(img, [r.x0, r.y0, r.x1, r.y1]); d = color_descriptor(c, g) if c is not None else None
            if d is not None: rows.append((int(r.track_id), d))
    tid = np.array([r[0] for r in rows]); F = np.vstack([r[1] for r in rows])
    print(f"{len(F)} crop descriptors in {time.time() - t0:.0f}s")
    tc = TeamClassifier().fit(F); v = vote(tid, tc.predict(F))
    raw = score(v); res["unsupervised KMeans (raw cluster order)"] = raw
    res["unsupervised KMeans (order fixed by hand)"] = score({k: (1 - x if x in (0, 1) else x) for k, x in v.items()}) \
        if raw["outfield accuracy"] < 0.5 else raw
    samples = {}                                     # clicked samples re-described with the current descriptor
    for k, v_ in setup["kits"].items():
        for s_ in v_:
            im = read_frame(str(VIDEO), s_["t"]); c = torso_crop(im, s_["box"])
            d = color_descriptor(c, grass_model(im)) if c is not None else None
            if d is not None: samples.setdefault(k, []).append(d)
    kc = KitClassifier(samples).fit(F); A = vote_roles(tid, kc.predict(F)).set_index("track_id")
    res["clicked samples (6+6 outfield, 1 GK, 2 referee)"] = score(A.team.to_dict(), A.role.to_dict())
    print("seeded report:", {k: (np.round(v_, 2).tolist() if isinstance(v_, list) else v_) for k, v_ in kc.report.items()})
pd.DataFrame(res).T"""),
("md", """**Verdict:** outfield accuracy is ~85-90 % with any method — the limit is the colour descriptor on blurred,
shaded 25-px players (SigLIP embeddings are the upgrade). What the clicks buy is **certainty**: the team identity
can never swap (unsupervised clusters can come out *swapped*), the **referee** is kept out of the teams, and
goalkeepers are recognised when their kits are clicked (only Canedo's keeper was clicked → add UD Mansores')."""),
("md", "## 3 · Positions and pitch size: how much does a few metres move xG?"),
("code", r"""from football_eval.xg import XGModel, feature_row
xgm = XGModel.load(ROOT / "models/xg_model.json")
Lp, Wp = 105.4, 61.6
spots = {"penalty spot": (Lp - 11, Wp / 2), "11 m out, 8 m wide": (Lp - 11, Wp / 2 - 8),
         "16 m out, 12 m wide": (Lp - 16, Wp / 2 - 12), "6 m out, 5 m wide": (Lp - 6, Wp / 2 - 5)}
errs = np.linspace(-3, 3, 13)
fig, ax = plt.subplots(figsize=(8, 3.6)); rows = {}
for name, (x, y) in spots.items():
    v = [xgm.predict([feature_row(x, y + d, None, Lp, Wp)], "basic")[0] for d in errs]
    ax.plot(errs, v, label=name); rows[name] = {"xG": round(v[6], 3), "xG at -3 m": round(v[0], 3), "xG at +3 m": round(v[-1], 3),
                                                 "relative change": f"±{100 * max(abs(v[0] - v[6]), abs(v[-1] - v[6])) / v[6]:.0f} %"}
ax.set_xlabel("lateral position error (m)"); ax.set_ylabel("xG (basic model)"); ax.legend(frameon=False, fontsize=8)
ax.set_title("xG sensitivity to a lateral position error")
print("width estimates of the venue calibration ranged 52-62 m -> several metres of error near the touchlines")
pd.DataFrame(rows).T"""),
("md", """**Verdict:** a measured pitch size is a 30-second input that removes the largest systematic position error
(the camera cannot see the far touchline well at 640 × 360). Enter it, then re-solve the camera with it fixed
(`calibrate_ptz.py --setup`)."""),
("md", "## 4 · Direction of play and periods: how fragile is the inference?"),
("code", r"""pp = tr[(tr.team >= 0) & tr.x.notna() & (tr.cls == "player")]
b = (pp.t // 30) * 30
m = pp.groupby([b, "team"]).x.mean().unstack()
fig, ax = plt.subplots(figsize=(10, 3))
ax.plot(m.index / 60, m[0], label=setup["teams"][0]["name"], color="#2a78d6"); ax.plot(m.index / 60, m[1], label=setup["teams"][1]["name"], color="#e8743b")
ax.set_xlabel("video minute"); ax.set_ylabel("mean x (m)"); ax.legend(frameon=False); ax.set_title("Mean x per team (30-s bins)")
d = (m[0] - m[1])
print(f"mean x difference team0 - team1: {d.mean():.1f} m on average; sign agrees with the mean in {100 * (np.sign(d) == np.sign(d.mean())).mean():.0f} % of bins")
print("the video also holds", f"{setup['periods'][0]['start'] / 60:.1f} min of warm-up before the kick-off and",
      f"{(setup['periods'][1]['start'] - setup['periods'][0]['end']) / 60:.1f} min of half-time -> the side-swap rule cannot find them")"""),
("md", """**Verdict:** the direction comes from a few metres of average position and flips every directional statistic
if wrong; the periods include long non-play stretches. Both take seconds to give → manual."""),
("md", """## 5 · What a 10-minute review measures (pilot)
Pilot review of the 16 sampled frames and the key event of the segment, done by Claude on 640 × 360 screenshots
(`outputs/canedo_segment/review_pilot_claude.json`). Your own review replaces it."""),
("code", r"""rv = load_review(RUN / "review_pilot_claude.json")
fc = pd.DataFrame(rv["frame_checks"])
if HAVE_VIDEO:
    cfg = load_config(ROOT / "configs/canedo_segment.yaml")
    geo = RunGeometry(str(RUN), cfg)
    pick = fc.iloc[[0, 4, 11, 12]]
    fig, axes = plt.subplots(2, 2, figsize=(14, 8.2))
    for ax, (_, c) in zip(axes.ravel(), pick.iterrows()):
        ax.imshow(geo.draw(read_frame(str(VIDEO), c.t), int(c.frame))[..., ::-1]); ax.axis("off")
        ax.set_title(f"{c.t / 60:.1f} min · gate {'kept' if c.auto_ok else 'discarded'} · lines judged {c.lines}", fontsize=9)
    plt.tight_layout()
validation_summary(rv)"""),
("md", """**Verdict:** the automatic camera-quality gate is useful but not sufficient (wrong poses pass it, good ones
are thrown away), and the only "shot" in the segment did not happen. Both are invisible without a look — a short
review turns them into measured numbers in the report and fixes the events that matter."""),
("md", "## 6 · The Canedo setup file: what is still to do"),
("code", r"""pd.DataFrame(checklist(setup, ROOT))[["item", "status", "detail", "impact"]]"""),
("md", """Sections pre-filled by Claude are shown as *warn* until you confirm them in
`python scripts/match_ui.py setup --video data/videos/2026-10-18_UD_Mansores_vs_Canedo_FC_J2.mp4 --setup configs/match_setup_canedo_J2.json --config configs/canedo_cfctv.yaml`.
Still missing: the real pitch size, UD Mansores' goalkeeper kit, the final score and goal minutes."""),
]


# ---------------------------------------------------------------------------- notebook I/O + executor
# ============================================================================ 07 opportunities
NB["07_opportunities_eval.ipynb"] = [
("md", r"""# 07 · Major opportunities — are the flagged moments the right ones?

`football_eval/opportunities.py` groups shots, goals, penalties, box entries and set pieces into **attacking
moves** and flags (⚑) the ones worth watching first: goals, penalties, big chances (xG ≥ 0.30), shots on target and
shots with xG ≥ 0.10 (rules: `docs/STATS_RULEBOOK.md` §7c). The report lists them at the top with a clip and a link
to the moment in the full video; the launcher plays them next to the list.

Questions, on synthetic matches with ground truth (same generator as notebook 02):

1. **Coverage** — is every true goal inside a flagged opportunity, and every true shot inside some opportunity?
2. **Precision** — does each flagged opportunity contain a real shot or goal?
3. **Time saved** — how many minutes of video do the flagged clips add up to?
4. **Sensitivity** — what the two main settings (`flag_min_xg`, `merge_window_s`) change.
5. **Real recording** — what the list looks like on the 10-minute Canedo segment (2–5 px ball)."""),
("code", SETUP + r"""
import copy
from football_eval.synthetic import SyntheticMatch, SynthConfig
from football_eval.config import load_config
from football_eval.analytics import run_analytics
from football_eval.opportunities import build_opportunities, evaluate_opportunities, opportunity_plot, CATEGORIES
from football_eval.xg import XGModel
XGM = XGModel.load(ROOT / "models/xg_model.json")

def run(seed=0, period_s=900, **noise):
    m = SyntheticMatch(SynthConfig(seed=seed, period_s=period_s, **noise)).generate()
    r = run_analytics(m["tracks"], m["ball"], load_config(), root=str(ROOT))
    return m, r

def rebuild(r, **opp):
    cfg = load_config(opportunities=opp)
    return build_opportunities(r["events"], r["ctx"], cfg, XGM, r["touches"])"""),
("md", "## 1 · One match in detail"),
("code", r"""m, r = run(seed=0)
opps = r["opportunities"]
minutes = sum(b - a for a, b in r["ctx"].periods) / 60
print(f"{len(opps)} opportunities, {int(opps.flag.sum())} flagged, in {minutes:.0f} min of play")
pd.DataFrame(evaluate_opportunities(opps, m["gt_events"], match_minutes=minutes), index=["seed 0"]).round(3)"""),
("code", r"""opps[opps.flag][["video_time", "clock", "team_name", "category", "chance_xg", "danger", "description"]]"""),
("code", r"""opportunity_plot(opps, r["ctx"])"""),
("code", r"""pd.crosstab(opps.category, opps.team_name).reindex(CATEGORIES).fillna(0).astype(int)"""),
("md", r"""## 2 · Benchmark: several matches, default and heavy noise

Heavy noise = 0.8 m position jitter, 25 % ball dropouts, ID switches every 30 s, goalkeepers not labelled
(the stress setting of notebook 02)."""),
("code", r"""rows = []
for label, noise in [("default", {}), ("heavy noise", dict(noise_player_m=0.8, noise_ball_m=0.8, ball_dropout=0.25,
                                                          fragment_s=30, label_goalkeepers=False))]:
    for seed in range(1, 5):
        m_, r_ = run(seed=seed, period_s=600, **noise)
        mins = sum(b - a for a, b in r_["ctx"].periods) / 60
        e = evaluate_opportunities(r_["opportunities"], m_["gt_events"], match_minutes=mins)
        e.update(setting=label, seed=seed, minutes=mins); rows.append(e)
bench = pd.DataFrame(rows)
cols = ["n_opps", "n_flagged", "gt_shots", "gt_goals", "shot_recall", "goal_recall_flag", "on_target_recall_flag",
        "flag_precision", "watch_min", "watch_share"]
summary = bench.groupby("setting")[cols].agg(lambda s: s.sum() if s.name in ("n_opps", "n_flagged", "gt_shots", "gt_goals", "watch_min") else s.mean())
summary.round(3)"""),
("md", r"""## 3 · Sensitivity of the two main settings (seed 0, events unchanged — only the grouping / flag rule)"""),
("code", r"""rows = []
for fx in (0.0, 0.05, 0.10, 0.20, 0.30):
    o = rebuild(r, flag_min_xg=fx)
    e = evaluate_opportunities(o, m["gt_events"], match_minutes=minutes); e["flag_min_xg"] = fx; rows.append(e)
sens_flag = pd.DataFrame(rows).set_index("flag_min_xg")[["n_flagged", "flag_precision", "on_target_recall_flag", "watch_min"]]
rows = []
for w in (0, 3, 6, 10, 15):
    o = rebuild(r, merge_window_s=w)
    e = evaluate_opportunities(o, m["gt_events"], match_minutes=minutes); e["merge_window_s"] = w; rows.append(e)
sens_merge = pd.DataFrame(rows).set_index("merge_window_s")[["n_opps", "n_flagged", "shot_recall", "flag_precision", "watch_min"]]
sens_flag.round(3)"""),
("code", r"""sens_merge.round(3)"""),
("md", r"""## 4 · Real recording: the 10-minute Canedo segment

Shots on this footage come from a 2–5 px ball (`docs/RECORDING_PROFILE.md` §3), so the shot detector rarely
fires. On this segment it finds a single blocked shot from 28 m (xG 0.02, not flagged), and the ball track is too
sparse for box entries. On club recordings, the flagged list depends on the review UI's key-event step. Confirm or
reject each detected shot, and **add the missed chances under "Missing events"**. Each added shot, goal or penalty
becomes a flagged opportunity with its clip on the next analytics run."""),
("code", r"""from football_eval.perception import load_perception
from football_eval.config import load_config
seg = ROOT / "outputs/canedo_segment"
if (seg / "tracks.csv").exists():
    cfg = load_config(ROOT / "configs/canedo_segment.yaml")
    res = load_perception(str(seg))
    rr = run_analytics(res["tracks"], res["ball"], cfg, root=str(ROOT))
    o = rr["opportunities"]
    print(f"{len(o)} opportunities, {int(o.flag.sum())} flagged")
    print(o[["video_time", "team_name", "category", "flag", "danger", "description"]].head(20).to_string(index=False))
else:
    print("outputs/canedo_segment not found — run configs/canedo_segment.yaml first")"""),
("md", r"""## 5 · Take-aways

* **Default noise (4 synthetic matches, 71 true shots, 13 goals):**
  * every true goal and 97 % of the true shots on target fall inside a flagged opportunity;
  * every flagged opportunity contains a real shot or goal (precision 1.0);
  * the flagged clips add up to ~12 % of the playing time, so the chances of a 90-minute game take about 10 minutes to watch.
* **Heavy noise:** coverage drops to 83 % of the goals and 60 % of the shots on target, while precision stays at
  1.0. The list misses chances when the shot detector misses them; it does not invent chances. This is why the
  review UI's "missing events" step matters on real footage.
* **Settings:**
  * `flag_min_xg` between 0.10 and 0.30 changes little, because most flagged moves are goals or shots on target.
    0.05 adds a few long-range attempts, and 0 flags every shot (+75 % watch time).
  * With `merge_window_s` ≥ 10 s, separate attacks start to merge. 6 s keeps a corner → shot → rebound together
    without merging separate attacks."""),
]

# ============================================================================ 08 speed + kits (J3)
NB["08_speed_and_kits_eval.ipynb"] = [
("md", r"""# 08 · Processing speed and clicked kits — J3, UD Mansores vs Sanguedo (1080p)

First evaluation of the full-match workflow on `J3_MansoresSanguedo_1080p.mp4` raised three problems:

1. processing a whole match took ~8.5 h (1.8 analysed frames / s);
2. the players clicked in the setup (kits) did not show in the live boxes;
3. the camera-side touchline is often outside the picture.

This notebook measures (1) and (2) with the code of v0.2. (3) is covered by `test_windows_and_near_side_visibility`
and rulebook §3.1. Needs the J3 video, or the extracts `outputs/_seg_J3_2420.mp4` (video 40:20–42:40) and
`outputs/_kitframes/f<frame>.png` (the frames where kits were clicked)."""),
("code", SETUP + r"""
from football_eval.config import load_config
from football_eval.detection import YoloxOnnxDetector
from football_eval.match_setup import load_setup
from football_eval.teams import KitClassifier, color_descriptor, grass_model, snap_box, torso_crop, person_like, CODE
from football_eval.tracking import iou_matrix
from football_eval.video import iter_frames, probe
VIDEO = ROOT / "data/videos/J3_MansoresSanguedo_1080p.mp4"
SEG = ROOT / "outputs/_seg_J3_2420.mp4"
KF = ROOT / "outputs/_kitframes"
setup = load_setup(ROOT / "configs/match_setup_J3_MansoresSanguedo_1080p.json")
def kit_frame(f):
    p = KF / f"f{f}.png"
    if p.exists():
        return cv2.imread(str(p))
    cap = cv2.VideoCapture(str(VIDEO)); cap.set(cv2.CAP_PROP_POS_FRAMES, f); ok, im = cap.read(); cap.release()
    return im if ok else None
seg = SEG if SEG.exists() else VIDEO
seg_t0 = 0.0 if SEG.exists() else 2420.0
print("speed clip:", seg.name, "| kit frames:", sorted({s["frame"] for v in setup["kits"].values() for s in v}))"""),
("md", """## 1 · Where the time goes, and what the new tiling buys
The detector is ~95 % of the processing time (decoding ~2 %, camera ~3 %, shirt colours < 1 %). Three settings on
the same 14 frames (one every 10 s): the **previous** layout (6 tiles per 1080p frame, `cv2.dnn`), the new
**standard** (2 pitch-aware tiles, onnxruntime) and the new **quality** preset (800 px tiles)."""),
("code", r"""import football_eval.detection as fdet
new_plan = fdet.tile_plan
def legacy_plan(W, H, tile, overlap, roi=None, stretch=1.1):     # v0.1 grid: 960 px tiles, stride 720, whole frame
    T = min(960, H, W)
    ys = sorted(set(list(range(0, max(H - T, 1), 720)) + [H - T])); xs = sorted(set(list(range(0, max(W - T, 1), 720)) + [W - T]))
    return [(x, y, x + T, y + T) for y in ys for x in xs]
M = str(ROOT / "models/yolox.onnx")
ROI = (0, 240, 1920, 1080)        # rows above the far touchline skipped (what the pitch ROI computes on this camera)
dets = {"previous: 960 px grid (6 tiles), cv2.dnn": (YoloxOnnxDetector(M, runtime="cv2"), None, legacy_plan),
        "standard, whole frame": (YoloxOnnxDetector(M, tile=1024, overlap=96), None, new_plan),
        "standard + pitch ROI": (YoloxOnnxDetector(M, tile=1024, overlap=96), ROI, new_plan),
        "quality + pitch ROI": (YoloxOnnxDetector(M, tile=800, overlap=96), ROI, new_plan)}
rows, boxes = {k: [] for k in dets}, {k: [] for k in dets}
for f, t, img in iter_frames(str(seg), 0.1, seg_t0, seg_t0 + 140):
    for k, (D, roi, plan) in dets.items():
        fdet.tile_plan = plan
        t0 = time.perf_counter(); d = D(img, roi=roi); dt = time.perf_counter() - t0
        fdet.tile_plan = new_plan
        rows[k].append((dt, len(d.of("player")), len(d.of("ball")), len(D.last_tiles))); boxes[k].append(d.of("player").xyxy)
ref = list(dets)[0]
res = {}
for k in dets:
    a = np.array(rows[k])
    found = sum(int((iou_matrix(b0, b1).max(1) > 0.5).sum()) if len(b1) and len(b0) else 0 for b0, b1 in zip(boxes[ref], boxes[k]))
    res[k] = {"tiles / frame": a[:, 3].mean(), "ms / frame": 1000 * a[:, 0].mean(), "players / frame": a[:, 1].mean(),
              "ball candidates / frame": a[:, 2].mean(), "previous boxes found again": f"{found}/{sum(len(b) for b in boxes[ref])}"}
R = pd.DataFrame(res).T
R["speed-up"] = (R["ms / frame"].iloc[0] / R["ms / frame"]).round(1)
fig, ax = plt.subplots(figsize=(8, 2.8)); ax.barh(R.index[::-1], R["ms / frame"][::-1], color=["#2a78d6"] * (len(R) - 1) + ["#999"])
ax.set_xlabel("detector time per analysed frame (ms, this machine)"); plt.tight_layout()
R.round(2)"""),
("md", """**Verdict:** the previous grid ran two rows of tiles only 120 px apart on a 1080-px frame. Two tiles that just cover
the pitch, plus onnxruntime, make the detector ~5× faster. The price is resolution: a 1008-px wide tile is shrunk
to the 640-px network input (0.63×, was 0.67×), so a few distant players and ball candidates are lost — the pitch ROI
(skipping the sky and stands) gets part of it back. *Quality* (800-px tiles) is still ~2× faster than before. On the Mac, onnxruntime also tries the CoreML provider and keeps it only if it is faster. Full match at
10 analysed fps ≈ 54k frames: the launcher estimates the time from the speed measured by your previous runs."""),
("md", """## 2 · The boxes you dragged vs the player the detector sees
The kit descriptor looks at the shirt (20–80 % of the box width, 15–55 % of its height). A loose box puts grass —
or a neighbour — in that window."""),
("code", r"""det = dets["standard, whole frame"][0]
S, frames = [], {}
for cls, items in setup["kits"].items():
    for it in items:
        f = it["frame"]
        if f not in frames:
            frames[f] = kit_frame(f)
        img = frames[f]
        if img is None: continue
        P = det(img).of("player").xyxy
        sb = snap_box(it["box"], P)
        iou = float(iou_matrix(np.array([it["box"]], float), P).max()) if len(P) else 0.0
        S.append(dict(cls=cls, frame=f, drawn=np.array(it["box"], float), snapped=sb, iou_with_detection=iou))
K = pd.DataFrame(S)
print(f"{len(K)} clicked samples on {K.frame.nunique()} moments; snapped: {K.snapped.notna().sum()}")
print("IoU of the drawn box with the detected player:", K.iou_with_detection.describe()[["min", "25%", "50%", "75%", "max"]].round(2).to_dict())
f0 = 60548 if 60548 in frames else list(frames)[0]
vis = frames[f0].copy()
for _, r in K[K.frame == f0].iterrows():
    x0, y0, x1, y1 = r.drawn.astype(int); cv2.rectangle(vis, (x0, y0), (x1, y1), (60, 60, 255), 2)
    if r.snapped is not None:
        x0, y0, x1, y1 = np.asarray(r.snapped).astype(int); cv2.rectangle(vis, (x0, y0), (x1, y1), (80, 230, 80), 3)
plt.figure(figsize=(13, 7)); plt.imshow(vis[..., ::-1]); plt.axis("off"); plt.title(f"frame {f0}: drawn (red) vs snapped to the detection (green)");"""),
("md", """## 3 · Can the clicked samples recognise each other?
**Leave-one-out**: each clicked sample is classified by all the others (same rule as the pipeline: mean distance to
the 2 nearest samples of each class). **Leave-one-moment-out**: all samples of one moment are classified by the
samples of the other moments (new lighting / camera position) — the realistic case. Three variants: the drawn boxes
with the previous descriptor (fixed green band removed as "grass"), the snapped boxes with that descriptor, and the
snapped boxes with the v0.2 descriptor (only *this frame's* pitch colour removed, no lightness)."""),
("code", r"""def legacy_descriptor(crop):                     # v0.1: fixed hue band 30-90 removed as grass, Lab quartiles
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    keep = ~((hsv[..., 0] > 30) & (hsv[..., 0] < 90) & (hsv[..., 1] > 50) & (hsv[..., 2] > 40))
    if keep.sum() < 6: return None
    lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)[keep].astype(float)
    q = np.percentile(lab, [25, 50, 75], axis=0).ravel() / 255.0
    hist, _ = np.histogram(hsv[..., 0][keep], bins=12, range=(0, 180), weights=hsv[..., 1][keep] / 255.0 + 0.05)
    br = np.array([(hsv[..., 2][keep] > 170).mean(), (hsv[..., 2][keep] < 60).mean()])
    return np.r_[2.0 * q, hist / (hist.sum() + 1e-9), br]
def describe(r, which):
    img = frames[r.frame]
    box = r.drawn if which == "drawn" else r.snapped
    if box is None: return None
    c = torso_crop(img, box)
    if c is None: return None
    return legacy_descriptor(c) if which != "new" else color_descriptor(c, grass_model(img))
def evaluate(which):
    use = K if which == "drawn" else K[K.snapped.notna()]
    D = [(r.cls, r.frame, describe(r, "drawn" if which == "drawn" else ("new" if which == "new" else "snapped"))) for _, r in use.iterrows()]
    D = [d for d in D if d[2] is not None]
    out = {}
    for name, split in (("leave-one-out", lambda i, j: i != j), ("leave-one-moment-out", lambda i, j: D[i][1] != D[j][1])):
        ok_out = ok_all = n_out = n_all = 0
        for i, (c, f, d) in enumerate(D):
            samples = {}
            for j, (c2, f2, d2) in enumerate(D):
                if split(i, j): samples.setdefault(c2, []).append(d2)
            if "team0" not in samples or "team1" not in samples or c not in samples: continue
            p = KitClassifier(samples).predict(np.array([d]))[0]
            hit = p == CODE[c]; n_all += 1; ok_all += hit
            if c in ("team0", "team1"): n_out += 1; ok_out += hit
        out[name + " (outfield)"] = f"{ok_out}/{n_out}"; out[name + " (all classes)"] = f"{ok_all}/{n_all}"
    return out
pd.DataFrame({"drawn boxes, previous descriptor": evaluate("drawn"),
              "snapped boxes, previous descriptor": evaluate("snapped"),
              "snapped boxes, v0.2 descriptor": evaluate("new")}).T"""),
("md", """**Verdict:** both problems were in the input, not in the voting: loose boxes (grass in the shirt window) and a
"grass" mask that also deleted the green UD Mansores shirts. With snapped boxes and the per-frame pitch colour the
clicked samples recognise each other almost perfectly, also across moments. A class with a single sample (the
Sanguedo goalkeeper) cannot be checked by leave-one-out — click it on 2–3 moments.

The same classifier now drives the live boxes and the final team assignment; the tracks you clicked keep your
label (`team_assignment.csv`, column `clicked`)."""),
]


def to_nb(cells):
    out = []
    for kind, src in cells:
        c = {"cell_type": "markdown" if kind == "md" else "code", "metadata": {}, "source": src.strip("\n").splitlines(True)}
        if kind == "code":
            c.update(execution_count=None, outputs=[])
        out.append(c)
    return {"cells": out, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                        "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}


def execute(nb, cwd):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    g = {"__name__": "__main__"}
    old = os.getcwd()
    os.chdir(cwd)
    n = 0
    try:
        for c in nb["cells"]:
            if c["cell_type"] != "code":
                continue
            n += 1
            src = "".join(c["source"])
            outs = []
            buf = io.StringIO()
            t0 = time.time()
            try:
                tree = ast.parse(src)
                last = None
                if tree.body and isinstance(tree.body[-1], ast.Expr) and not src.rstrip().endswith(";"):
                    last = ast.Expression(tree.body.pop().value)
                with contextlib.redirect_stdout(buf):
                    exec(compile(tree, f"cell{n}", "exec"), g)
                    val = eval(compile(last, f"cell{n}", "eval"), g) if last is not None else None
                if buf.getvalue():
                    outs.append({"output_type": "stream", "name": "stdout", "text": buf.getvalue().splitlines(True)})
                for num in plt.get_fignums():
                    fig = plt.figure(num)
                    b = io.BytesIO()
                    fig.savefig(b, format="png", bbox_inches="tight", dpi=90)
                    outs.append({"output_type": "display_data", "metadata": {},
                                 "data": {"image/png": base64.b64encode(b.getvalue()).decode(), "text/plain": ["<Figure>"]}})
                plt.close("all")
                if val is not None and not hasattr(val, "savefig") and type(val).__name__ not in ("Axes", "AxesSubplot"):
                    data = {"text/plain": repr(val).splitlines(True)}
                    if hasattr(val, "to_html"):
                        data["text/html"] = val.to_html().splitlines(True)
                    outs.append({"output_type": "execute_result", "execution_count": n, "metadata": {}, "data": data})
            except Exception:
                if buf.getvalue():
                    outs.append({"output_type": "stream", "name": "stdout", "text": buf.getvalue().splitlines(True)})
                outs.append({"output_type": "error", "ename": "Error", "evalue": "see traceback",
                             "traceback": traceback.format_exc().splitlines()})
                print(traceback.format_exc(), file=sys.stderr)
            c["outputs"], c["execution_count"] = outs, n
            print(f"  cell {n}: {time.time() - t0:.1f}s", file=sys.stderr)
    finally:
        os.chdir(old)
    return nb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--execute", action="store_true")
    ap.add_argument("--only", default=None)
    a = ap.parse_args()
    NB_DIR.mkdir(exist_ok=True)
    for name, cells in NB.items():
        if a.only and a.only not in name:
            continue
        nb = to_nb(cells)
        if a.execute:
            print("executing", name, file=sys.stderr)
            nb = execute(nb, NB_DIR)
        (NB_DIR / name).write_text(json.dumps(nb, indent=1))
        print("->", NB_DIR / name)


if __name__ == "__main__":
    main()
