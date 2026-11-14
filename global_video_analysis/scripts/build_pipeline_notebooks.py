"""Build (and optionally execute) the step-by-step pipeline notebooks in notebooks/pipeline/.

    python scripts/build_pipeline_notebooks.py                 # write the notebooks without outputs
    python scripts/build_pipeline_notebooks.py --execute       # also run them in order and embed the outputs
    python scripts/build_pipeline_notebooks.py --execute --only 02

The cell sources live here (diff-able, easy to regenerate after a code change), like scripts/build_notebooks.py.
All notebooks read notebooks/pipeline/params.yaml (or $FVE_NB_PARAMS) through football_eval/nbtools.py and pass
their results to the next one through outputs/_pipeline_nb/<run_name>/*.csv.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from build_notebooks import execute, to_nb  # noqa: E402

NB_DIR = ROOT / "notebooks" / "pipeline"

HEAD = r'''import sys, os, time, json, warnings
from pathlib import Path
ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / "football_eval").is_dir())
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")
import cv2, numpy as np, pandas as pd, matplotlib.pyplot as plt
pd.set_option("display.precision", 3); pd.set_option("display.width", 160)
pd.set_option("display.max_columns", 30); pd.set_option("display.max_rows", 60)
plt.rcParams.update({"figure.dpi": 90, "axes.spines.top": False, "axes.spines.right": False})
from football_eval import nbtools as nbt

P = nbt.load_params()            # notebooks/pipeline/params.yaml (or $FVE_NB_PARAMS)
ctx = nbt.build_context(P)       # config <- match setup, restricted to the window
cfg = ctx.cfg'''

NB = {}

# ============================================================================================ 00 setup
NB["00_setup_and_environment.ipynb"] = [
("md", r'''# 00 · Setup, environment and the pipeline map

These notebooks take the pipeline apart, one stage per notebook, so each model / rule can be inspected, measured and
replaced on its own:

| # | notebook | stage | input | output (stage table) |
|---|----------|-------|-------|----------------------|
| 01 | `01_video_to_frames` | decode, sub-sample, grass mask / ROI | mp4 | `frames_index.csv` |
| 02 | `02_player_detection` | YOLOX tiles → boxes (people + ball candidates) | frames | `detections.csv` |
| 03 | `03_pitch_calibration` | homography image ↔ pitch (static / flow / PTZ) | frames + people boxes | `homography.csv` |
| 04 | `04_tracking_and_teams` | IDs across frames, person type, kit → team / role | detections + homography | `tracks.csv` |
| 05 | `05_ball_detection_tracking` | ball tracker (Kalman + zoom + scan) → cleaned trajectory | frames + people | `ball_raw.csv`, `ball.csv` |
| 06 | `06_integrated_perception` | the real loop (`run_perception`) on the same window, cross-checked with 02–05 | mp4 | `perception/` + `benchmarks.csv` |
| 07 | `07_analytics_events_stats` | possession → events → xG → team / player stats | tracks + ball | tables |
| 08 | `08_model_training_cycle` | export data → label → train → evaluate → deploy a detector | a full run | dataset, scores, config |

**Why stages with tables in between:** you can swap one model (e.g. a fine-tuned YOLO in 02) and re-run only the
notebooks downstream on the new table, and 06 tells you whether the integrated loop agrees with the stage-by-stage
result. 06 also appends a row of KPIs to `outputs/_pipeline_nb/benchmarks.csv`, the log of your experiments.

Edit **`notebooks/pipeline/params.yaml`** (video, match setup, time window) once; every notebook reads it.'''),
("code", HEAD),
("md", "## 1 · Where will the models run?\n`accel.describe()` is what the detector sees: the NVIDIA GPU (driver), the onnxruntime execution providers and the PyTorch device. `accel.doctor()` (fix branch) goes further when CUDA is listed but not used."),
("code", r'''from football_eval import accel
print(accel.describe())
try:
    import onnxruntime as ort
    print("\nonnxruntime", ort.__version__, "| device:", ort.get_device(), "| providers:", ort.get_available_providers())
except ImportError:
    print("onnxruntime not installed -> cv2.dnn")'''),
("md", "## 2 · The configuration actually used\nDefaults ← `config` file ← match setup (`apply_to_config`) ← the window from `params.yaml`. These are the values the stages below run with; change them here (e.g. `cfg.detection.tile = 800`) to experiment in one notebook, or in the config file to change all of them."),
("code", r'''nbt.config_table(cfg)'''),
("md", "## 3 · Video, window and expected cost"),
("code", r'''info = ctx.info
n_an = int(sum(b - a for a, b in ctx.windows) * cfg.video.analysis_fps)
print(f"source fps {info.fps:.2f}, analysis fps {cfg.video.analysis_fps:g} -> every {ctx.step}th frame")
print(f"window {ctx.windows} -> {n_an} analysed frames")
full = sum(b - a for a, b in (cfg.match.halves or [(0, info.duration_s)])) * cfg.video.analysis_fps
print(f"whole match: {int(full)} analysed frames; at 5 fps processing that is {full / 5 / 3600:.1f} h, at 20 fps {full / 20 / 3600:.1f} h")
f, t, img = nbt.probe_frame(ctx)
nbt.show(img, f"probe frame {f} (t = {t:.1f} s)");'''),
("md", "## 4 · Match setup facts\nWhat the human told the pipeline (periods, direction, kits, calibration, match sheet). Everything downstream depends on these."),
("code", r'''if ctx.setup:
    s = ctx.setup
    print("teams       :", [t.get("name") for t in s["teams"]])
    print("periods     :", s.get("periods"))
    print("kits        :", {k: len(v) for k, v in s["kits"].items()})
    print("calibration :", s["calibration"].get("mode"), "| landmark frames:",
          [(round(f["t"], 1), len(f["points"])) for f in s["calibration"].get("landmark_frames", [])])
    print("match sheet :", s.get("final_score"), [(e.get("type"), e.get("minute"), e.get("team")) for e in s.get("known_events", [])])
else:
    print("no match setup: run scripts/match_ui.py setup for reliable teams / calibration")'''),
]

# ============================================================================================ 01 frames
NB["01_video_to_frames.ipynb"] = [
("md", r'''# 01 · Video → frames

`football_eval/video.py::iter_frames` decodes the mp4 and keeps every `step = round(fps / analysis_fps)`-th frame, on
**one frame grid** anchored at the start of the first window (so both halves share the grid). Frames that are
skipped are only `grab()`-ed (no decoding to BGR), and decoding runs in a background thread (`prefetch`) so it
overlaps with the detector.

Then, before any model: the **grass mask** (`people.grass_top`) finds the far edge of the pitch in each column; the
detector only looks below it (minus some head room) when there is no calibration yet (`people.grass_roi`).'''),
("code", HEAD),
("md", "## 1 · The frame grid"),
("code", r'''from football_eval.video import iter_frames, read_frame, fmt_clock
rows = [(f, t) for f, t, _ in nbt.frames_in_window(ctx, prefetch=0)]
idx = pd.DataFrame(rows, columns=["frame", "t"])
d = np.diff(idx.frame)
print(f"{len(idx)} analysed frames, frame step {set(d.tolist())}, first {idx.frame.iloc[0]} ({fmt_clock(idx.t.iloc[0])}), "
      f"last {idx.frame.iloc[-1]} ({fmt_clock(idx.t.iloc[-1])})")
assert (d == ctx.step).all(), "the grid should be regular inside one window"
idx.head()'''),
("md", "## 2 · Decoding speed\nSequential decoding with `grab()` for skipped frames vs seeking every frame (`read_frame`), and the effect of the prefetch thread. On a full match decoding is ~5 % of the time; seeking is much slower — never seek per frame."),
("code", r'''def bench(n=60, **kw):
    t0 = time.perf_counter()
    k = sum(1 for _ in nbt.frames_in_window(ctx, max_frames=n, **kw))
    return k / (time.perf_counter() - t0)
seq = bench(prefetch=0); pre = bench(prefetch=8)
t0 = time.perf_counter()
for t in idx.t.iloc[:15]:
    read_frame(ctx.video, float(t))
seek = 15 / (time.perf_counter() - t0)
pd.DataFrame({"frames / s": [seq, pre, seek]}, index=["sequential (grab skipped)", "sequential + prefetch thread", "seek every frame"]).round(1)'''),
("md", "## 3 · What the frames look like across the window"),
("code", r'''imgs, titles = [], []
pick = set(np.linspace(0, len(idx) - 1, 6).round().astype(int).tolist())
for k, (f, t, img) in enumerate(nbt.frames_in_window(ctx)):
    if k in pick:
        imgs.append(img); titles.append(f"frame {f}  t={fmt_clock(t)}")
nbt.grid(imgs, titles, cols=3, size=5.5);'''),
("md", "## 4 · Grass mask and the detector region (before any calibration)\n`grass_top` = far edge of the grass per column (yellow). `grass_roi` = rows the detector will look at (magenta) when no homography is available. With a calibration, the integrated loop uses the *previous frame's* pitch outline instead (`perception.pitch_roi`, see 03)."),
("code", r'''from football_eval.people import grass_top, grass_roi
f, t, img = nbt.probe_frame(ctx)
gt = grass_top(img)
roi = grass_roi(gt, img.shape)
vis = img.copy()
pts = np.c_[np.arange(len(gt)), gt].astype(np.int32)
cv2.polylines(vis, [pts.reshape(-1, 1, 2)], False, (0, 255, 255), 3)
if roi is not None:
    cv2.rectangle(vis, (int(roi[0]), int(roi[1])), (int(roi[2]) - 1, int(roi[3]) - 1), (255, 0, 255), 4)
print("grass ROI (x0, y0, x1, y1):", roi, f"-> skips the top {roi[1] / img.shape[0]:.0%} of the frame" if roi else "")
nbt.show(vis, "far edge of the grass (yellow) and detector ROI (magenta)");'''),
("md", "## 5 · Camera motion over the window\nMean absolute difference between consecutive analysed frames (downscaled): peaks = pans / zooms. The calibration (03) and the ball tracker (05) must follow these."),
("code", r'''prev, md = None, []
for f, t, img in nbt.frames_in_window(ctx):
    g = cv2.cvtColor(cv2.resize(img, (320, 180)), cv2.COLOR_BGR2GRAY).astype(np.float32)
    md.append((f, t, np.nan if prev is None else float(np.abs(g - prev).mean())))
    prev = g
mdf = pd.DataFrame(md, columns=["frame", "t", "mean_abs_diff"])
ax = mdf.plot(x="t", y="mean_abs_diff", figsize=(11, 2.8), legend=False)
ax.set_xlabel("video time (s)"); ax.set_ylabel("mean |Δ| (grey levels)"); ax.set_title("frame-to-frame change (camera motion + players)");'''),
("code", r'''nbt.save_stage(ctx, "frames_index", idx.merge(mdf, on=["frame", "t"]));'''),
]

# ============================================================================================ 02 detection
NB["02_player_detection.ipynb"] = [
("md", r'''# 02 · Player (and ball candidate) detection

Default backend `yolox_onnx` (`football_eval/detection.py`): YOLOX-S trained on COCO, so it only knows *person* and
*sports ball*; roles (goalkeeper / referee) come later from the kits (04). Steps per frame:

1. **tile plan** — the fewest tiles of ≤ `detection.tile` px covering the region of interest, `tile_overlap` px overlap;
2. per tile: **letterbox** to 640×640 → network → 8400 anchors × (box, objectness, 80 class scores) → **decode**;
3. score = objectness × class prob, thresholds `conf_player` / `conf_ball`, **NMS** over all tiles;
4. **merge** people cut by an inner tile border; drop overlay regions and tiny boxes.

This notebook runs each step by hand, then the detector over the window, and shows how to score it against a
hand-annotated frame. To try another model, change `cfg.detection` (e.g. `backend: ultralytics`, a fine-tuned
checkpoint from 08) and re-run from here.'''),
("code", HEAD),
("md", "## 1 · Build the detector\nThe runtime benchmark below is where the GPU is chosen (or not): every execution provider onnxruntime offers is timed on a dummy tile and the fastest is kept. If CUDA is installed but this prints `CUDAExecutionProvider unavailable`, see the fix branch (`python -m football_eval.accel --doctor`)."),
("code", r'''from football_eval.detection import build_detector, tile_plan, merge_cut_boxes
detector = build_detector(cfg)
r = getattr(detector, "runner", None)
if r is not None:
    print(f"runtime: {r.name} | on GPU: {r.on_gpu} | {r.ms or float('nan'):.1f} ms per 640x640 tile (benchmark)")'''),
("md", "## 2 · Tile plan: whole frame vs region of interest"),
("code", r'''from football_eval.people import grass_top, grass_roi
f, t, img = nbt.probe_frame(ctx)
H_, W_ = img.shape[:2]
roi = grass_roi(grass_top(img), img.shape) if cfg.detection.pitch_roi else None
plans = {"whole frame": tile_plan(W_, H_, detector.tile, detector.overlap),
         "grass ROI": tile_plan(W_, H_, detector.tile, detector.overlap, roi)}
fig, axes = plt.subplots(1, 2, figsize=(15, 4.6))
for ax, (name, tiles) in zip(axes, plans.items()):
    vis = img.copy()
    for k, (x0, y0, x1, y1) in enumerate(tiles):
        cv2.rectangle(vis, (x0 + 3, y0 + 3), (x1 - 3, y1 - 3), nbt.COLORS["tile"], 5)
        cv2.putText(vis, f"tile {k}: {x1 - x0}x{y1 - y0}", (x0 + 20, y0 + 60), cv2.FONT_HERSHEY_SIMPLEX, 1.6, (255, 0, 255), 4)
    nbt.show(vis, f"{name}: {len(tiles)} tiles", ax=ax)
fig.tight_layout()'''),
("md", "## 3 · One tile through the network, by hand\nLetterbox (keep aspect, pad with grey 114) → blob (1×3×640×640, BGR, 0–255) → raw output 8400 × 85: `cx, cy, w, h` in grid units, objectness, 80 COCO class probabilities. Anchors come from 3 strides (8, 16, 32): 80² + 40² + 20² = 8400."),
("code", r'''x0, y0, x1, y1 = plans["grass ROI"][0]
tile = img[y0:y1, x0:x1]
S = detector.S
h, w = tile.shape[:2]; rr = min(S / h, S / w)
pad = np.full((S, S, 3), 114, np.uint8)
pad[:int(h * rr), :int(w * rr)] = cv2.resize(tile, (int(w * rr), int(h * rr)), interpolation=cv2.INTER_AREA)
blob = cv2.dnn.blobFromImage(pad, 1.0, (S, S), swapRB=False)
with nbt.Timer() as tm:
    raw = detector.runner(blob).copy()
print(f"tile {w}x{h} -> scale {rr:.3f} -> network input {blob.shape}, raw output {raw.shape} in {tm.s * 1000:.0f} ms")
o = raw.copy()
o[:, :2] = (o[:, :2] + detector.grids) * detector.strides          # grid cell + offset -> pixels (640 space)
o[:, 2:4] = np.exp(o[:, 2:4]) * detector.strides                  # log-size -> pixels
boxes = np.c_[o[:, 0] - o[:, 2] / 2, o[:, 1] - o[:, 3] / 2, o[:, 0] + o[:, 2] / 2, o[:, 1] + o[:, 3] / 2] / rr
s_person, s_ball = o[:, 4] * o[:, 5], o[:, 4] * o[:, 5 + detector.BALL_CLS]
print(f"anchors with person score > {cfg.detection.conf_player}: {(s_person > cfg.detection.conf_player).sum()}, "
      f"ball score > {cfg.detection.conf_ball}: {(s_ball > cfg.detection.conf_ball).sum()} (before NMS)")
# same result as the library's _infer
bp, sp, bb, sb = detector._infer(tile)
assert np.allclose(bp, boxes[s_person > detector.cp]) and np.allclose(sp, s_person[s_person > detector.cp])
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
nbt.show(pad, "letterboxed network input", ax=axes[0])
axes[1].hist([s_person[s_person > 0.01], s_ball[s_ball > 0.01]], bins=40, log=True, label=["person", "sports ball"])
axes[1].axvline(cfg.detection.conf_player, color="C0", ls="--"); axes[1].axvline(cfg.detection.conf_ball, color="C1", ls="--")
axes[1].set_title("anchor scores (> 0.01) and the thresholds"); axes[1].legend();'''),
("md", "## 4 · Threshold sweep and NMS on this tile\nHow many people / ball candidates survive each threshold, before and after non-maximum suppression (IoU 0.45). The plateau after NMS is the number of real people; where it starts to drop is the safe range for `conf_player`."),
("code", r'''def after_nms(b, s, thr):
    k = s > thr
    if not k.any(): return 0
    keep = cv2.dnn.NMSBoxes([[float(a), float(c), float(e - a), float(d - c)] for a, c, e, d in b[k]], s[k].tolist(), thr, 0.45)
    return len(np.array(keep).ravel())
thr = [0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
sweep = pd.DataFrame({"person raw": [(s_person > q).sum() for q in thr], "person NMS": [after_nms(boxes, s_person, q) for q in thr],
                      "ball raw": [(s_ball > q).sum() for q in thr], "ball NMS": [after_nms(boxes, s_ball, q) for q in thr]}, index=thr)
sweep.index.name = "threshold"
sweep'''),
("md", "## 5 · The full detector on the frame\nAll tiles + NMS + merging of people cut by a tile border. Box height vs foot row is a perspective sanity check: far players (top) must be smaller; outliers are usually crowd / staff or merged pairs."),
("code", r'''with nbt.Timer() as tm:
    det = detector(img, roi=roi)
people, balls = det.of("player", "goalkeeper", "referee"), det.of("ball")
print(f"{len(people)} people, {len(balls)} ball candidates, {len(detector.last_tiles)} tiles, {tm.s * 1000:.0f} ms")
vis = nbt.draw_boxes(img, people.xyxy, [f"{c:.2f}" for c in people.conf])
for (a, b, c, d), s in zip(balls.xyxy, balls.conf):
    cv2.circle(vis, (int((a + c) / 2), int((b + d) / 2)), 22, nbt.COLORS["ball"], 3)
nbt.show(vis, "people (white, score) and ball candidates (yellow)")
fig, ax = plt.subplots(figsize=(6, 3.5))
ax.scatter(people.xyxy[:, 3], people.xyxy[:, 3] - people.xyxy[:, 1], c=people.conf, cmap="viridis")
ax.set_xlabel("foot row v (px)"); ax.set_ylabel("box height (px)"); ax.set_title("perspective check");'''),
("md", "## 6 · Over the whole window\nThe same call on every analysed frame (ROI from the grass mask; the integrated loop uses the previous frame's pitch outline once calibrated). Timing per frame tells you the cost of a full match."),
("code", r'''rows, ms = [], []
for fi, ti, im in nbt.frames_in_window(ctx):
    r_ = grass_roi(grass_top(im), im.shape) if cfg.detection.pitch_roi else None
    t0 = time.perf_counter(); d = detector(im, roi=r_); dt = (time.perf_counter() - t0) * 1000
    ms.append(dt)
    for (a, b, c, e), s, k in zip(d.xyxy, d.conf, d.cls):
        rows.append((fi, ti, k, float(s), a, b, c, e, len(detector.last_tiles), dt))
dets = pd.DataFrame(rows, columns=["frame", "t", "cls", "conf", "x0", "y0", "x1", "y1", "n_tiles", "ms"])
per = dets[dets.cls != "ball"].groupby("frame").size()
print(f"{len(ms)} frames: {np.mean(ms):.0f} ms/frame ({1000 / np.mean(ms):.1f} fps), people/frame median {per.median():.0f} "
      f"[{per.min()}-{per.max()}], ball candidate in {dets[dets.cls == 'ball'].frame.nunique() / len(ms):.0%} of the frames")
fig, axes = plt.subplots(1, 3, figsize=(15, 3.2))
per.reset_index(drop=True).plot(ax=axes[0], title="people per frame")
dets[dets.cls != "ball"].conf.hist(bins=30, ax=axes[1]); axes[1].set_title("people scores")
pd.Series(ms).plot(ax=axes[2], title="ms per frame");'''),
("md", r'''## 7 · Scoring against a hand-annotated frame
Format of `data/annotations/<name>.json` (see `football_eval/evaluation.py`): `{"image", "frame", "boxes": [{"xyxy", "cls", "team"}], "ball": [u, v]}`.
The cell writes a **template from the detections** for the probe frame if none exists (plus the image), so you only
correct it (delete false boxes, add missed players, fix the ball) instead of drawing from scratch. Re-run the cell
afterwards to get precision / recall at IoU 0.5 and the ball distance.'''),
("code", r'''from football_eval.evaluation import detection_pr
ann_dir = ROOT / "data" / "annotations"; ann_dir.mkdir(parents=True, exist_ok=True)
ann_path = ann_dir / f"{P.run_name}_f{f}.json"
if not ann_path.exists():
    cv2.imwrite(str(ann_path.with_suffix(".png")), img)
    tpl = {"image": ann_path.with_suffix(".png").name, "video": Path(ctx.video).name, "frame": int(f), "t": float(t),
           "status": "template from detections - CORRECT ME, then set status to 'checked'",
           "boxes": [{"xyxy": [round(float(v), 1) for v in b], "cls": c, "team": -1} for b, c in zip(people.xyxy, people.cls)],
           "ball": [round(float((balls.xyxy[0, 0] + balls.xyxy[0, 2]) / 2), 1), round(float((balls.xyxy[0, 1] + balls.xyxy[0, 3]) / 2), 1)] if len(balls) else None}
    ann_path.write_text(json.dumps(tpl, indent=1), encoding="utf-8")
    print("template written:", ann_path.relative_to(ROOT), "-> correct it (any JSON editor, or LabelMe / CVAT import) and re-run")
ann = json.loads(ann_path.read_text(encoding="utf-8"))
gt = np.array([b["xyxy"] for b in ann["boxes"]], float).reshape(-1, 4)
res = {thr_: detection_pr(people.xyxy, gt, thr_) for thr_ in (0.3, 0.5, 0.7)}
print("annotation status:", ann.get("status", "checked"))
out = pd.DataFrame({k: {m: v[m] for m in ("tp", "fp", "fn", "precision", "recall")} for k, v in res.items()}).T
out.index.name = "IoU threshold"
if ann.get("ball") and len(balls):
    bc = np.c_[(balls.xyxy[:, 0] + balls.xyxy[:, 2]) / 2, (balls.xyxy[:, 1] + balls.xyxy[:, 3]) / 2]
    print(f"ball: closest candidate {np.min(np.hypot(*(bc - np.array(ann['ball'])).T)):.1f} px from the annotated ball")
out'''),
("code", r'''nbt.save_stage(ctx, "detections", dets);'''),
]

# ============================================================================================ 03 pitch
NB["03_pitch_calibration.ipynb"] = [
("md", r'''# 03 · Pitch calibration: image ↔ pitch

Everything in metres (speeds, distances, xG, zones) goes through a homography **H** (pitch plane → image) per frame
(`football_eval/pitch.py`). The provider depends on the camera:

* **static** — fixed camera, one H from ≥ 4 clicked landmarks;
* **flow** — panning camera: H is propagated frame to frame with ORB features on the background (players masked out),
  re-anchored on frames where you clicked landmarks; drift grows between anchors;
* **ptz** — fixed-position pan/tilt/zoom camera model (`ptz.py`): pan / tilt / focal per frame, refined on the white lines;
* **keypoints** — a pose model that finds pitch keypoints in every frame (needs a trained model, see 08).

Here: (1) the quality of the clicked landmarks, (2) the provider over the window, (3) players in pitch coordinates.'''),
("code", HEAD),
("code", r'''from football_eval.pitch import Pitch, Homography, build_provider, draw_pitch
from football_eval.perception import foot_points, pitch_roi, visible_near_y
pitch = Pitch(cfg.pitch.length, cfg.pitch.width)
provider = build_provider(cfg, pitch, ctx.info.fps)
if getattr(provider, "cam", None) is not None:
    pitch = provider.cam.pitch
print(f"pitch {pitch.length} x {pitch.width} m | mode '{cfg.pitch.homography}' -> provider {type(provider).__name__ if provider else None}")
if hasattr(provider, "anchors"):
    an = sorted(provider.anchors)
    print(f"{len(an)} anchor frames at t = {[round(a / ctx.info.fps, 1) for a in an]} s; window {ctx.windows}")
    inside = [a for a in an if any(w0 * ctx.info.fps <= a <= w1 * ctx.info.fps for w0, w1 in ctx.windows)]
    if cfg.pitch.homography == "flow" and not inside:
        print("[!] no anchor inside the window: a flow provider has no pose until it meets one -> put an anchor in the window")'''),
("md", "## 1 · Clicked landmarks: fit and reprojection error\nFor each landmark frame of the setup: H fitted on the clicks (RANSAC), then every click reprojected. Errors > ~5 px on a 1080p frame mean a wrong click or a wrong landmark name; a large error on one side of the pitch usually means lens distortion or a wrong pitch size."),
("code", r'''from football_eval.video import read_frame
L = pitch.landmarks()
frames_cal = (ctx.setup or {}).get("calibration", {}).get("landmark_frames", []) if ctx.setup else []
if not frames_cal and cfg.pitch.calibration_points:
    frames_cal = [{"t": P.probe_t, "points": cfg.pitch.calibration_points}]
cal_rows, cal_vis = [], []
for lf in frames_cal:
    pts = {k: v for k, v in lf["points"].items() if k in L}
    if len(pts) < 4:
        continue
    img_pts = np.array(list(pts.values()), float); pit_pts = np.array([L[k] for k in pts])
    Hc = Homography.fit(img_pts, pit_pts)
    err = np.hypot(*(Hc.to_image(pit_pts) - img_pts).T)
    for k, e in zip(pts, err):
        cal_rows.append(dict(t=round(lf["t"], 1), landmark=k, error_px=round(float(e), 1)))
    try:
        im = read_frame(ctx.video, float(lf["t"]))
    except ValueError:
        continue
    im = nbt.draw_pitch_lines(im, Hc, pitch)
    for (u, v), (pu, pv) in zip(img_pts, Hc.to_image(pit_pts)):
        cv2.circle(im, (int(u), int(v)), 9, (0, 0, 255), 3); cv2.drawMarker(im, (int(pu), int(pv)), (0, 255, 0), cv2.MARKER_CROSS, 22, 3)
    cal_vis.append((im, f"t={lf['t']:.1f}s: {len(pts)} clicks, mean error {err.mean():.1f} px"))
cal = pd.DataFrame(cal_rows)
for im, title in cal_vis[:2]:
    nbt.show(im, title + "  (red = click, green = reprojected, yellow = pitch lines)")
cal.groupby("t").error_px.describe()[["count", "mean", "max"]] if len(cal) else "no clicked landmarks in the setup"'''),
("md", "## 2 · The provider over the window\nThe people boxes of 02 mask the players out of the background features (flow). For a PTZ camera the camera is first tracked from its anchor up to the window, as the real loop does. Projected pitch lines on frames across the window show drift."),
("code", r'''dets = nbt.load_stage(ctx, "detections")
anchor = getattr(provider, "anchor_frame", None)
if anchor is not None and ctx.windows[0][0] > anchor / ctx.info.fps + 1.0:
    print(f"PTZ warm-up from {anchor / ctx.info.fps:.0f}s to {ctx.windows[0][0]:.0f}s ...")
    provider.warmup(ctx.video, anchor / ctx.info.fps, ctx.windows[0][0], cfg.video.analysis_fps)
rows, shots = [], []
pick = None
for k, (f, t, im) in enumerate(nbt.frames_in_window(ctx)):
    people = nbt.detections_from_table(dets, f)
    H = provider.get(f, im, people.xyxy) if provider is not None else None
    row = nbt.homography_table_row(H, f, t)
    M = getattr(provider, "last_image_motion", None)
    if M is not None:
        row.update({f"m{i}{j}": float(np.asarray(M)[i, j]) for i in range(3) for j in range(3)})
    row["vis_near_y"] = visible_near_y(H, pitch, im.shape)
    rows.append(row)
    if k % max(1, int(len(dets.frame.unique()) / 4)) == 0:
        shots.append((nbt.draw_pitch_lines(im, H, pitch), f"frame {f} t={t:.1f}s  {'H ok' if H is not None else 'no H'}"))
hom = pd.DataFrame(rows)
print(f"pose in {hom.has_h.mean():.0%} of the window's frames")
nbt.grid([s[0] for s in shots[:4]], [s[1] for s in shots[:4]], cols=2, size=7);'''),
("md", "## 3 · Stability of H: where do the pitch corners land, frame by frame?\nSmooth curves = camera motion; jumps = a bad estimate (re-anchor, more clicks, or a better provider)."),
("code", r'''corners = np.array([[0, 0], [pitch.length, 0], [pitch.length / 2, pitch.width / 2]])
tr = []
for _, r in hom.iterrows():
    Hr = nbt.homography_from_row(r)
    tr.append(Hr.to_image(corners).ravel() if Hr is not None else np.full(6, np.nan))
tr = np.array(tr)
fig, axes = plt.subplots(1, 2, figsize=(14, 3.2))
for j, name in enumerate(["far-left corner", "far-right corner", "centre spot"]):
    axes[0].plot(hom.t, tr[:, 2 * j], label=name); axes[1].plot(hom.t, tr[:, 2 * j + 1], label=name)
axes[0].set_title("image u (px)"); axes[1].set_title("image v (px)"); axes[0].legend(); axes[1].set_xlabel("t (s)");'''),
("md", "## 4 · Players on the pitch (probe frame)\nFoot point = bottom-centre of the box (estimated when the box is cut by the bottom of the frame) → H⁻¹ → metres. People outside the lines (+ margin) are staff / crowd (04)."),
("code", r'''f, t, img = nbt.probe_frame(ctx)
r = hom.iloc[(hom.frame - f).abs().argmin()]
H = nbt.homography_from_row(r)
people = nbt.detections_from_table(dets, int(r.frame))
foot, edge = foot_points(people.xyxy, img.shape)
xy = H.to_pitch(foot) if H is not None and len(foot) else np.full((len(foot), 2), np.nan)
fig, axes = plt.subplots(1, 2, figsize=(16, 5), gridspec_kw={"width_ratios": [1.6, 1]})
nbt.show(nbt.draw_pitch_lines(nbt.draw_boxes(img, people.xyxy), H, pitch), f"frame {int(r.frame)}", ax=axes[0])
draw_pitch(axes[1], pitch)
axes[1].scatter(xy[:, 0], xy[:, 1], c=["C3" if e else "k" for e in edge], s=25)
axes[1].set_title("feet in pitch coordinates (red = foot estimated)")
roi = pitch_roi(H, pitch, img.shape) if H is not None else None
print("detector ROI from this pose (next frame):", roi, "| visible near touchline y:", r.vis_near_y)'''),
("code", r'''nbt.save_stage(ctx, "homography", hom);'''),
]

# ============================================================================================ 04 tracking + teams
NB["04_tracking_and_teams.ipynb"] = [
("md", r'''# 04 · Tracking, person type and teams

Per analysed frame (same order as `perception.run_perception`):

1. **tracker** (`tracking.IoUTracker`: Hungarian on IoU + centre distance with a constant-velocity prediction;
   `bytetrack` optional) → track id per box, `-1` until a track is confirmed (2 hits);
2. **zone** of each person (pitch / touchline / stands, from pitch metres or the grass mask) and **person type**
   (staff, spectator, steward = hi-vis) for people off the pitch (`people.py`);
3. **kit descriptor** of the torso (every 2nd frame): chroma quartiles + hue histogram after removing this frame's
   grass colour (`teams.color_descriptor`);
4. **team / role**: with the players clicked in the setup → `KitClassifier` (nearest clicked samples per class:
   team0, team1, gk0, gk1, referee), else KMeans(2) → **vote per track** (`vote_roles`).'''),
("code", HEAD),
("code", r'''from football_eval.tracking import build_tracker
from football_eval.pitch import Pitch, draw_pitch
from football_eval.perception import foot_points, build_kit_bank
from football_eval.people import grass_top, zones, frame_types, track_types, people_summary
from football_eval.teams import TeamClassifier, KitClassifier, vote, vote_roles, CODE_NAME, torso_crop
dets = nbt.load_stage(ctx, "detections"); hom = nbt.load_stage(ctx, "homography").set_index("frame")
pitch = Pitch(cfg.pitch.length, cfg.pitch.width)
tracker = build_tracker(cfg, cfg.video.analysis_fps)
teams = TeamClassifier(cfg.teams.backend, cfg.teams.outlier_quantile, cfg.teams.outlier_factor, cfg.detection.device)
print(type(tracker).__name__, vars(tracker) if hasattr(tracker, "max_age") else "")'''),
("md", "## 1 · The per-frame loop (tracker + zones + descriptors)"),
("code", r'''rows, feats, feat_rows, crops = [], [], [], []
tm = nbt.Timer().__enter__()
for n, (f, t, img) in enumerate(nbt.frames_in_window(ctx)):
    people = nbt.detections_from_table(dets, f); tm.lap("read")
    ids = tracker.update(people); tm.lap("tracker")
    H = nbt.homography_from_row(hom.loc[f]) if f in hom.index else None
    foot, edge = foot_points(people.xyxy, img.shape)
    xy = H.to_pitch(foot) if (H is not None and len(foot)) else np.full((len(foot), 2), np.nan)
    gtop = grass_top(img)
    zone = zones(xy, foot, people.xyxy, pitch, gtop, cfg.pitch.offpitch_margin_m, cfg.pitch.staff_band_m)
    ptype = frame_types(img, people.xyxy, zone, cfg.pitch.hivis_min_share); tm.lap("zones / person type")
    desc = teams.describe(img, people.xyxy) if (n % 2 == 0 and len(people)) else [None] * len(people); tm.lap("kit descriptor")
    for k in range(len(people)):
        rows.append((f, t, int(ids[k]), people.cls[k], float(people.conf[k]), *people.xyxy[k], *foot[k], *xy[k], int(edge[k]), zone[k], ptype[k]))
        if desc[k] is not None and ptype[k] is None and ids[k] >= 0:
            feats.append(desc[k]); feat_rows.append(len(rows) - 1)
            if len(crops) < 400:
                crops.append((len(rows) - 1, torso_crop(img, people.xyxy[k]), nbt.crop(img, people.xyxy[k])))
tm.__exit__()
cols = ["frame", "t", "track_id", "cls", "conf", "x0", "y0", "x1", "y1", "u", "v", "x", "y", "edge", "zone", "ptype"]
tracks = pd.DataFrame(rows, columns=cols)
print(f"{len(tracks)} person rows, {tracks.track_id.nunique() - (tracks.track_id < 0).any()} confirmed tracks, "
      f"{(tracks.track_id < 0).mean():.1%} rows unconfirmed, {len(feats)} kit descriptors")
tm.table(n + 1)'''),
("md", "## 2 · Track quality\nWithout ground truth, proxies: **fragmentation** = tracks per person actually on the pitch (1.0 ideal; > 1.5 = IDs break often), track length distribution, and ID births per frame. Short tracks are what the team vote cannot fix."),
("code", r'''tt = tracks[tracks.track_id >= 0]
life = tt.groupby("track_id").agg(first=("t", "min"), last=("t", "max"), n=("t", "size"))
life["dur_s"] = life["last"] - life["first"] + 1 / cfg.video.analysis_fps
on_pitch = tracks[tracks.zone == "pitch"].groupby("frame").size().median()
print(f"median people on the pitch per frame: {on_pitch:.0f}; tracks touching the pitch: "
      f"{tt[tt.zone == 'pitch'].track_id.nunique()} -> fragmentation {tt[tt.zone == 'pitch'].track_id.nunique() / max(on_pitch, 1):.2f}")
fig, axes = plt.subplots(1, 2, figsize=(14, 3.6))
life.dur_s.hist(bins=30, ax=axes[0]); axes[0].set_title("track duration (s)")
for i, (tid, g) in enumerate(tt.groupby("track_id")):
    axes[1].plot(g.t, np.full(len(g), i), "|", ms=3)
axes[1].set_title("track timelines (one row per id)"); axes[1].set_xlabel("t (s)");'''),
("md", "## 3 · Team / role classifier"),
("code", r'''F = np.asarray(feats)
if cfg.teams.kit_samples and cfg.teams.backend == "color":
    from football_eval.detection import build_detector
    kit, seeds, kit_report = build_kit_bank(ctx.video, cfg, teams, build_detector(cfg), ctx.info)
else:
    kit, seeds, kit_report = None, {}, {}
if kit is not None:
    kit.fit(F)
    codes, margin = kit.predict(F, return_margin=True)
    print("kit classifier:", {k: v for k, v in kit.report.items() if k in ("loo_accuracy", "loo_n", "thr", "ignored_samples")})
else:
    teams.fit(F); lab = teams.predict(F)
    codes = np.where(lab >= 0, lab, -1); margin = np.full(len(codes), np.nan)
    print("unsupervised KMeans(2): team order is arbitrary (no clicked kits)")
pd.Series([CODE_NAME.get(int(c)) or "unknown" for c in codes]).value_counts().to_frame("crops")'''),
("md", "### Crops per predicted class\nScan these: a wrong class is usually an occluded player (two shirts in one crop) or shade / motion blur."),
("code", r'''by = {}
for (ri, torso, full), c in zip(crops, codes[:len(crops)]):
    by.setdefault(CODE_NAME.get(int(c)) or "unknown", []).append(full)
for name, ims in by.items():
    ims = ims[:12]
    fig, axes = plt.subplots(1, 12, figsize=(14, 1.6))
    for ax, k in zip(axes, range(12)):
        ax.axis("off")
        if k < len(ims): ax.imshow(ims[k][..., ::-1])
    axes[0].set_title(f"{name} ({len(by[name])} crops shown max 12)", loc="left")'''),
("md", "### Descriptor space (PCA)\nTwo tight, separated clouds = easy kits. Overlap = the descriptor cannot separate them (try `teams.backend: siglip`, or click more samples in different light)."),
("code", r'''from sklearn.decomposition import PCA
pca = PCA(2).fit(F)
Z = pca.transform(F)
fig, ax = plt.subplots(figsize=(7, 5))
for c in sorted(set(codes.tolist())):
    m = codes == c
    ax.scatter(Z[m, 0], Z[m, 1], s=8, alpha=0.6, label=CODE_NAME.get(int(c)) or "unknown")
if kit is not None:
    for cls_, S_ in kit.samples.items():
        zs = pca.transform(S_); ax.scatter(zs[:, 0], zs[:, 1], marker="*", s=180, edgecolor="k", label=f"clicked {cls_}")
ax.legend(fontsize=7); ax.set_title(f"kit descriptors (PCA, {pca.explained_variance_ratio_.sum():.0%} of variance)");'''),
("md", "## 4 · Vote per track → team and role"),
("code", r'''assignment = vote_roles(tracks.loc[feat_rows, "track_id"].to_numpy(), codes)
a = assignment.set_index("track_id")
tracks["team"] = tracks.track_id.map(a.team).fillna(-1).astype(int)
role = tracks.track_id.map(a.role)
sel = role.isin(["goalkeeper", "referee"])
tracks.loc[sel, "cls"] = role[sel]
tt_ = track_types(tracks)
off = tracks.track_id.map(tt_); sel = off.notna()
tracks.loc[sel, "cls"] = off[sel]; tracks.loc[sel, "team"] = -1
print(assignment.decision.value_counts().to_dict())
people_summary(tracks)'''),
("code", r'''f, t, img = nbt.probe_frame(ctx)
g = tracks[tracks.frame == tracks.frame.iloc[(tracks.frame - f).abs().argmin()]]
names = list(cfg.teams.team_names)
lab = [f"{r.track_id} {names[r.team][:8] if r.team >= 0 else r.cls}" for r in g.itertuples()]
col = [nbt.COLORS[int(r.team)] if r.team >= 0 else (nbt.COLORS["referee"] if r.cls == "referee" else (160, 160, 160)) for r in g.itertuples()]
fig, axes = plt.subplots(1, 2, figsize=(16, 5), gridspec_kw={"width_ratios": [1.6, 1]})
nbt.show(nbt.draw_boxes(img, g[["x0", "y0", "x1", "y1"]].to_numpy(), lab, col), "track id + team", ax=axes[0])
draw_pitch(axes[1], pitch)
for tm_, c in ((0, "C3"), (1, "C0"), (-1, "0.5")):
    q = g[g.team == tm_]; axes[1].scatter(q.x, q.y, c=c, s=30, label=names[tm_] if tm_ >= 0 else "other")
axes[1].legend(fontsize=7);'''),
("code", r'''nbt.save_stage(ctx, "tracks", tracks); nbt.save_stage(ctx, "team_assignment", assignment);'''),
]

# ============================================================================================ 05 ball
NB["05_ball_detection_tracking.ipynb"] = [
("md", r'''# 05 · Ball: candidates → tracker → clean trajectory

The ball is 6–15 px on a 1080p wide shot and becomes 4–7 px in a 1024-px tile shrunk to 640: the COCO detector alone
finds it in a small share of the frames. `balltrack.PitchBallTracker` (default `ball.detector: tracker`):

1. **Kalman filter** in image pixels (camera motion applied when the provider gives it — PTZ);
2. **zoomed re-detection**: 320-px window around the prediction, upscaled ×2;
3. **acquisition scan** at native resolution along the grass band while lost; **motion blobs** at 640 px;
4. gates: on / just above the grass, not on a player's upper body, static objects erased after 2 s.

Then `ball.clean_ball`: in pitch metres, one candidate per frame closest to the previous accepted one (speed gate
40 m/s), gaps ≤ 1 s interpolated, Savitzky–Golay smoothing, speed. Single camera: an airborne ball projects too
far (ground-plane homography), see `docs/SOTA_REVIEW.md`.'''),
("code", HEAD),
("code", r'''from football_eval.detection import build_detector
from football_eval.balltrack import PitchBallTracker
from football_eval.ball import clean_ball
from football_eval.people import grass_top, grass_roi
from football_eval.pitch import Pitch, draw_pitch
pitch = Pitch(cfg.pitch.length, cfg.pitch.width)
dets = nbt.load_stage(ctx, "detections"); hom = nbt.load_stage(ctx, "homography").set_index("frame")
tracks = nbt.load_stage(ctx, "tracks", required=False)
detector = build_detector(cfg)'''),
("md", "## 1 · What the plain detector gives (from 02)"),
("code", r'''bd = dets[dets.cls == "ball"]
nfr = dets.frame.nunique()
print(f"ball candidates (conf >= {cfg.detection.conf_ball}) in {bd.frame.nunique()}/{nfr} frames ({bd.frame.nunique() / max(nfr, 1):.0%}); "
      f"{(bd.groupby('frame').size() > 1).sum()} frames with >1 candidate")
ims, titles = [], []
for r in bd.sort_values("conf", ascending=False).head(10).itertuples():
    im = nbt.read_frame_idx(ctx.video, int(r.frame))
    cx, cy = (r.x0 + r.x1) / 2, (r.y0 + r.y1) / 2
    ims.append(nbt.crop(im, [cx - 40, cy - 40, cx + 40, cy + 40], pad=0, size=160)); titles.append(f"f{int(r.frame)} {r.conf:.2f}")
nbt.grid(ims, titles, cols=5, size=2.4) if ims else print("no ball candidate from the detector in this window");'''),
("md", "## 2 · The tracker, frame by frame\nThe detector's ball threshold is lowered to `ball.tracker_conf` (weak hits are gated by the track), exactly as in the integrated loop. `source` tells which mechanism measured the ball."),
("code", r'''if hasattr(detector, "cb"):
    detector.cb = min(detector.cb, cfg.ball.tracker_conf)
sc_ = (cfg.video.resize_width / ctx.info.width) if cfg.video.resize_width else 1.0
bt = PitchBallTracker(detector, cfg.video.analysis_fps, int(cfg.video.resize_width or ctx.info.width), scan_every=cfg.ball.scan_every,
                      ignore=[[v * sc_ for v in r] for r in (cfg.video.ignore_regions or [])])
hits, frames_seen, keep_last = [], [], {}
with nbt.Timer() as tm:
    for n, (f, t, img) in enumerate(nbt.frames_in_window(ctx)):
        gtop = grass_top(img)
        det = detector(img, roi=grass_roi(gtop, img.shape) if cfg.detection.pitch_roi else None)
        people = det.of("player", "goalkeeper", "referee"); b = det.of("ball")
        M = None
        if f in hom.index and "m00" in hom.columns and np.isfinite(hom.loc[f, "m00"]):
            M = hom.loc[f, [f"m{i}{j}" for i in range(3) for j in range(3)]].to_numpy(float).reshape(3, 3)
        hit = bt.step(img, b.xyxy, b.conf, people.xyxy, gtop, M)
        if bt.erase:                                   # the last frames' "ball" was a static object
            hits = [h for h in hits if h[-1] < n - bt.erase]; bt.erase = 0
        if hit is not None:
            hits.append((f, t, *hit, n))
        frames_seen.append((f, t))
        if n in (5, 60, 120): keep_last[n] = (f, img.copy(), hit)
H_ = pd.DataFrame(hits, columns=["frame", "t", "u", "v", "radius", "source", "k"])
print(f"{len(frames_seen)} frames in {tm.s:.0f}s ({len(frames_seen) / tm.s:.1f} fps incl. detector) | ball measured in {len(H_)} frames "
      f"({len(H_) / len(frames_seen):.0%})")
pd.Series(bt.stats).to_frame("count").T'''),
("code", r'''fig, axes = plt.subplots(2, 1, figsize=(12, 4.6), sharex=True)
for s_, g in H_.groupby("source"):
    axes[0].plot(g.t, g.u, ".", label=s_); axes[1].plot(g.t, g.v, ".", label=s_)
axes[0].set_ylabel("u (px)"); axes[1].set_ylabel("v (px)"); axes[0].legend(ncol=5, fontsize=8); axes[1].set_xlabel("t (s)")
axes[0].set_title("measured ball position by source");'''),
("md", "## 3 · Pitch coordinates and cleaning"),
("code", r'''raw = []
for r in H_.itertuples():
    Hm = nbt.homography_from_row(hom.loc[r.frame]) if r.frame in hom.index else None
    x, y = Hm.to_pitch(np.array([[r.u, r.v]]))[0] if Hm is not None else (np.nan, np.nan)
    conf = {"full": 0.6, "zoom": 0.5, "scan": 0.45, "motion": 0.3, "blob": 0.2}.get(r.source, 0.3)
    raw.append((r.frame, r.t, conf, r.u, r.v, x, y))
ball_raw = pd.DataFrame(raw, columns=["frame", "t", "conf", "u", "v", "x", "y"])
fr = pd.DataFrame(frames_seen, columns=["frame", "t"])
ball = clean_ball(ball_raw, fr.frame.to_numpy(), fr.t.to_numpy(), cfg.ball.max_speed_ms, cfg.ball.max_gap_s, cfg.ball.smooth_window)
print(f"visible {ball.visible.mean():.0%} | interpolated {ball.interp.mean():.0%} | usable {ball.valid.mean():.0%} | "
      f"rejected by the speed gate: {ball_raw.x.notna().sum() - ball.visible.sum()}")
fig, axes = plt.subplots(1, 2, figsize=(15, 3.8))
axes[0].plot(ball_raw.t, ball_raw.x, ".", c="0.6", label="raw x"); axes[0].plot(ball.t, ball.x, label="clean x")
axes[0].plot(ball_raw.t, ball_raw.y, ".", c="0.8", label="raw y"); axes[0].plot(ball.t, ball.y, label="clean y")
axes[0].legend(fontsize=8); axes[0].set_title("ball (m)")
axes[1].plot(ball.t, ball.speed); axes[1].set_title("ball speed (m/s)"); axes[1].set_ylim(0, 45);'''),
("code", r'''fig, ax = plt.subplots(figsize=(8, 5.4))
draw_pitch(ax, pitch)
v = ball[ball.valid]
ax.scatter(v.x, v.y, c=v.t, cmap="viridis", s=10)
if tracks is not None:
    q = tracks[(tracks.frame == tracks.frame.median()) & tracks.x.notna()]
    ax.scatter(q.x, q.y, c=["C3" if t_ == 0 else "C0" if t_ == 1 else "0.5" for t_ in q.team], s=25)
ax.set_title("ball trajectory over the window (colour = time)");'''),
("code", r'''nbt.save_stage(ctx, "ball_raw", ball_raw); nbt.save_stage(ctx, "ball", ball);'''),
]

# ============================================================================================ 06 integrated
NB["06_integrated_perception.ipynb"] = [
("md", r'''# 06 · The integrated perception loop on the same window

`perception.run_perception` is what `run_analysis.py` runs. It does 02–05 in **one loop with feedback** the
notebooks above do not have: the detector ROI comes from the *previous* frame's pose, the clicked kits seed the
tracker, the PTZ camera motion drives the ball filter, off-pitch people are typed per track, bad camera frames are
masked. Comparing it with the stage-by-stage tables catches integration bugs (a stage that works alone but not in
the loop) and gives the KPIs to log per experiment.'''),
("code", HEAD),
("code", r'''from football_eval.perception import run_perception, save_perception, load_perception
ctx2 = nbt.build_context(P, verbose=False)               # fresh config: run_perception mutates it
with nbt.Timer() as tm:
    res = run_perception(ctx2.video, ctx2.cfg, progress=True, live_dir=None, team_names=list(ctx2.cfg.teams.team_names))
save_perception(res, str(ctx.out / "perception"))
m = res["meta"]
print(f"\n{m['n_frames_analysed']} frames in {tm.s:.0f}s -> {m['speed']['fps']} fps | detector {m['speed']['detector']} | GPU {m['speed']['gpu']} | "
      f"tiles/frame {m['speed']['tiles_per_frame']}")'''),
("md", "## 1 · Integrated vs stage by stage"),
("code", r'''S = {k: nbt.load_stage(ctx, k, required=False) for k in ("detections", "homography", "tracks", "ball")}
T, B, Fr = res["tracks"], res["ball"], res["frames"]
def kpis_stage():
    d, h, tr, b = S["detections"], S["homography"], S["tracks"], S["ball"]
    return {"people / frame (median)": d[d.cls != "ball"].groupby("frame").size().median() if d is not None else np.nan,
            "frames with pose": h.has_h.mean() if h is not None else np.nan,
            "confirmed tracks": tr[tr.track_id >= 0].track_id.nunique() if tr is not None else np.nan,
            "team-labelled rows": (tr.team >= 0).mean() if tr is not None else np.nan,
            "ball visible": b.visible.mean() if b is not None else np.nan,
            "ball usable (with gaps filled)": b.valid.mean() if b is not None else np.nan}
k_int = {"people / frame (median)": Fr.n_people.median(), "frames with pose": Fr.has_h.mean(),
         "confirmed tracks": T[T.track_id >= 0].track_id.nunique(), "team-labelled rows": (T.team >= 0).mean(),
         "ball visible": B.visible.mean(), "ball usable (with gaps filled)": B.valid.mean()}
cmp_ = pd.DataFrame({"stage notebooks (02-05)": kpis_stage(), "integrated loop": k_int})
cmp_["diff"] = cmp_.iloc[:, 1] - cmp_.iloc[:, 0]
cmp_.round(3)'''),
("md", "Expected differences: more people / frame in the loop when the pitch ROI (from the pose) is wider than the grass ROI; fewer confirmed tracks when the kit seeds keep IDs; a different ball share because 05 used the grass ROI and no seeds. A large gap on *frames with pose* or *team-labelled rows* is worth a look."),
("md", "## 2 · Look at a frame of the loop's output"),
("code", r'''from football_eval.pitch import Pitch, draw_pitch
f, t, img = nbt.probe_frame(ctx)
fr_ = T.frame.iloc[(T.frame - f).abs().argmin()]
g = T[T.frame == fr_]; names = list(ctx2.cfg.teams.team_names)
col = [nbt.COLORS[int(r.team)] if r.team >= 0 else (160, 160, 160) for r in g.itertuples()]
lab = [f"{r.track_id} {r.cls[:3]}" for r in g.itertuples()]
vis = nbt.draw_boxes(nbt.read_frame_idx(ctx.video, int(fr_)), g[["x0", "y0", "x1", "y1"]].to_numpy(), lab, col)
bb = res["ball_raw"]; bb = bb[bb.frame == fr_]
for r in bb.itertuples():
    cv2.circle(vis, (int(r.u), int(r.v)), 18, nbt.COLORS["ball"], 3)
nbt.show(vis, f"integrated loop, frame {fr_}");'''),
("md", "## 3 · Log the experiment\nOne row per run in `outputs/_pipeline_nb/benchmarks.csv` (commit, detector, thresholds, KPIs). Give it a label when you change something (`label=\"yolo11s finetuned, conf 0.25\"`)."),
("code", r'''kp = {k.split(" (")[0].replace(" / ", "_per_").replace(" ", "_"): round(float(v), 3) for k, v in k_int.items()}
kp.update(fps=m["speed"]["fps"], runtime=m["speed"]["detector"], gpu=m["speed"]["gpu"])
bench = nbt.log_benchmark(ctx2, kp, label="baseline")
bench.tail(8)'''),
]

# ============================================================================================ 07 analytics
NB["07_analytics_events_stats.ipynb"] = [
("md", r'''# 07 · From tracks to events and statistics

`analytics.run_analytics`, unrolled. Input: `tracks` (people in pitch metres, team, role) + `ball` (clean trajectory).
Uses the full-match run of `params.yaml → full_run_dir` when it exists (events need a whole match), else the window
of 06. The rules and thresholds are documented in `docs/STATS_RULEBOOK.md`; each cell maps to one of them.'''),
("code", HEAD),
("code", r'''from football_eval.perception import load_perception
from football_eval.analytics import restrict_to_periods, run_analytics
from football_eval.context import infer_periods, player_speeds
from football_eval.teams import assign_goalkeepers
from football_eval.pitch import Pitch, draw_pitch
from football_eval.possession import frame_control, touches, team_possession
from football_eval.events import EventDetector, dead_ball
from football_eval.xg import XGModel
from football_eval.stats import team_stats, player_stats, possession_timeline
from football_eval.timeline import build_timeline
src = Path(P.full_run_dir) if P.full_run_dir else None
src = (ROOT / src) if src is not None and not src.is_absolute() else src
if src is None or not (src / "tracks.csv").exists():
    src = ctx.out / "perception"
print("analytics input:", src)
res = load_perception(str(src))
tracks, ball = res["tracks"], res["ball"]
print(f"{len(tracks)} person rows, {tracks.track_id.nunique()} track ids, {len(ball)} ball rows "
      f"({ball.t.min() / 60:.1f}-{ball.t.max() / 60:.1f} min), ball usable {ball.valid.mean():.0%}")'''),
("md", "## 1 · Periods, playing directions, goalkeepers"),
("code", r'''pitch = Pitch(cfg.pitch.length, cfg.pitch.width)
if cfg.match.halves and cfg.video.only_periods:
    tracks, ball, cut = restrict_to_periods(tracks, ball, cfg.match.halves)
    print(f"outside the periods: {cut} frames dropped")
mctx = infer_periods(tracks, pitch, cfg.match.halves, cfg.match.home_attacks_right_first_half)
mctx.team_names = list(cfg.teams.team_names)
if (tracks.cls == "goalkeeper").sum() == 0:
    for i, (a, b) in enumerate(mctx.periods):
        sel = (tracks.t >= a) & (tracks.t <= b)
        if sel.any():
            sub = assign_goalkeepers(tracks[sel].copy(), pitch, dict(mctx.attacks_right[i]))
            tracks.loc[sel, ["team", "cls"]] = sub[["team", "cls"]].to_numpy()
tracks["team"] = tracks.team.astype(int)
print("periods (video s):", mctx.periods, "| attacks right:", mctx.attacks_right)
tr_sp = player_speeds(tracks)
tr_sp.groupby("team").speed.describe()[["mean", "50%", "max"]]'''),
("md", "## 2 · Ball control per frame → dead ball → touches → possession"),
("code", r'''fc = frame_control(tracks, ball, cfg)
dead, stoppages = dead_ball(fc, tr_sp, mctx, cfg)
fc.loc[dead, ["ctrl_tid", "ctrl_team"]] = -1
tch = touches(fc, tracks, cfg.possession.min_control_s)
poss = team_possession(fc, dead, cfg.possession.team_switch_s)
if "cam_ok" in fc:
    unknown = ~fc.cam_ok.fillna(True).astype(bool).to_numpy(); poss[unknown] = -1; dead = dead & ~unknown
live = ~dead
print(f"frames: {len(fc)} | dead ball {dead.mean():.0%} | {len(stoppages)} stoppages | {len(tch)} touches")
print("possession (live frames with a team):", {mctx.team_names[k]: f"{(poss[live] == k).sum() / max((poss[live] >= 0).sum(), 1):.0%}" for k in (0, 1)},
      f"| unknown {(poss[live] < 0).mean():.0%}")
pt = possession_timeline(fc, poss, dead)
pt.head()'''),
("md", "## 3 · Events and xG"),
("code", r'''xg_path = ROOT / cfg.xg.model_file
xgm = XGModel.load(xg_path) if xg_path.exists() else None
ev = EventDetector(cfg, mctx, xgm).run(fc, tch, tracks, stoppages, res.get("meta", {}).get("scoreboard_changes", []))
ev["clock"] = [mctx.match_clock(t) for t in ev.t]
print(ev.type.value_counts().to_dict())
ev.head(12)'''),
("code", r'''fig, axes = plt.subplots(1, 2, figsize=(15, 5))
draw_pitch(axes[0], pitch)
for k, c in ((0, "C3"), (1, "C0")):
    p_ = ev[(ev.type == "pass") & (ev.team == k)]
    axes[0].quiver(p_.x, p_.y, p_.x_end - p_.x, p_.y_end - p_.y, angles="xy", scale_units="xy", scale=1, color=c, width=0.002, alpha=0.6, label=mctx.team_names[k])
axes[0].legend(fontsize=8); axes[0].set_title("passes")
draw_pitch(axes[1], pitch)
sh = ev[ev.type == "shot"]
if len(sh):
    axes[1].scatter(sh.x, sh.y, s=800 * sh.xg.fillna(0.02) + 20, c=["C3" if t == 0 else "C0" for t in sh.team], alpha=0.7)
axes[1].set_title(f"shots ({len(sh)}), size = xG");'''),
("md", "## 4 · Statistics and the timeline"),
("code", r'''tstats = team_stats(ev, fc, poss, dead, tr_sp, mctx, cfg)
pstats = player_stats(ev, tr_sp, tch)
tl = build_timeline(ev, mctx, final_third_x=cfg.events.final_third_x)
tstats'''),
("code", r'''pstats.sort_values(pstats.columns[-1], ascending=False).head(10) if len(pstats) else pstats'''),
("md", "## 5 · Cross-check with `run_analytics` (and the match sheet)\nThe library run also applies the match sheet / review verdicts and the pressure phases, so it can have *more* events; the automatic counts should match the cells above."),
("code", r'''full = run_analytics(res["tracks"], res["ball"], cfg, res.get("meta", {}).get("scoreboard_changes", []), root=str(ROOT), setup=ctx.setup)
c = pd.DataFrame({"unrolled (this notebook)": ev.type.value_counts(), "run_analytics: automatic": full["events_auto"].type.value_counts(),
                  "run_analytics: final (setup / review applied)": full["events"].type.value_counts()}).fillna(0).astype(int)
if ctx.setup:
    print("match sheet:", [(e.get("type"), e.get("minute"), e.get("team")) for e in ctx.setup.get("known_events", [])])
c'''),
]

# ============================================================================================ 08 training
NB["08_model_training_cycle.ipynb"] = [
("md", r'''# 08 · The model cycle: data → labels → training → evaluation → deployment

The weakest model is the detector: COCO YOLOX has no *goalkeeper / referee* class and misses small balls. The cycle
to improve it on **your** footage (same camera, same pitches):

```
 full-match runs ──► 1 export frames + pseudo-labels (YOLO format, split by time)
                         │
                         ▼
                     2 correct the labels (CVAT / Label Studio / Roboflow)  ◄── the human step, the one that matters
                         │
                         ▼
                     3 train  (ultralytics YOLO, imgsz 1280, GPU)
                         │
                         ▼
                     4 evaluate on held-out *corrected* frames: per-class P/R, ball recall, ms / frame
                         │            vs the current detector
                         ▼
                     5 deploy: config override → 06 on the same window → benchmarks.csv → full run
```

Other trainable parts, same cycle: **kit classifier** (your clicks = training data; leave-one-out in 04),
**xG model** (`scripts/train_xg.py`, `notebooks/04_xg_model_eval.ipynb`), **pitch keypoints** for the
`keypoints` homography (same export with landmarks instead of boxes).'''),
("code", HEAD),
("code", r'''from football_eval.detection import build_detector
from football_eval.evaluation import detection_pr
from football_eval import accel
CLASSES = ["player", "goalkeeper", "referee", "ball"]          # YOLO class ids 0..3
DS = ROOT / "data" / "datasets" / f"{Path(ctx.video).stem}_det"
run_dir = Path(P.full_run_dir) if P.full_run_dir else None
run_dir = (ROOT / run_dir) if run_dir is not None and not run_dir.is_absolute() else run_dir
tracks_full = pd.read_csv(run_dir / "tracks.csv", low_memory=False) if run_dir is not None and (run_dir / "tracks.csv").exists() else None
print("dataset folder:", DS)
print("roles from a full run:", run_dir if tracks_full is not None else "none (all people exported as 'player')")'''),
("md", r'''## 1 · Export frames + pseudo-labels
Frames every `EVERY_S` seconds inside the periods. Boxes from the current detector; the role (goalkeeper / referee)
from the full run's `tracks.csv` when the box matches a track box on that frame (IoU ≥ 0.5); staff / fans dropped.
**Split by time block** (alternating 5-minute blocks → train / val): neighbouring frames are near-duplicates, a random
split would leak and inflate the scores.'''),
("code", r'''EVERY_S, BLOCK_S, MAX_FRAMES = 20.0, 300.0, 40          # raise MAX_FRAMES for a real dataset (300-1000 frames)
from football_eval.tracking import iou_matrix
halves = cfg.match.halves or [[0, ctx.info.duration_s]]
times = np.concatenate([np.arange(a + 5, min(b, ctx.info.duration_s) - 5, EVERY_S) for a, b in halves])
times = times[np.linspace(0, len(times) - 1, min(MAX_FRAMES, len(times))).round().astype(int)]
detector = build_detector(cfg)
for sp in ("train", "val"):
    (DS / "images" / sp).mkdir(parents=True, exist_ok=True); (DS / "labels" / sp).mkdir(parents=True, exist_ok=True)
manifest = []
cap = cv2.VideoCapture(ctx.video)
for t in times:
    f = int(round(t * ctx.info.fps)); cap.set(cv2.CAP_PROP_POS_FRAMES, f); ok, img = cap.read()
    if not ok: continue
    d = detector(img)
    cls = list(d.cls)
    if tracks_full is not None:
        g = tracks_full[(tracks_full.frame - f).abs() <= ctx.step]
        if len(g):
            g = g[g.frame == g.frame.iloc[(g.frame - f).abs().argmin()]]
            people = [i for i, c in enumerate(cls) if c != "ball"]
            if people:
                iou = iou_matrix(d.xyxy[people], g[["x0", "y0", "x1", "y1"]].to_numpy(float))
                for a_, i in enumerate(people):
                    j = int(iou[a_].argmax())
                    if iou[a_, j] >= 0.5:
                        cls[i] = str(g.cls.iloc[j])
    split = "train" if int(t // BLOCK_S) % 2 == 0 else "val"
    name = f"{Path(ctx.video).stem}_f{f:07d}"
    cv2.imwrite(str(DS / "images" / split / f"{name}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
    h, w = img.shape[:2]; lines = []
    for (x0, y0, x1, y1), c in zip(d.xyxy, cls):
        if c not in CLASSES: continue                      # staff / spectator / steward are not labelled
        lines.append(f"{CLASSES.index(c)} {(x0 + x1) / 2 / w:.6f} {(y0 + y1) / 2 / h:.6f} {(x1 - x0) / w:.6f} {(y1 - y0) / h:.6f}")
    (DS / "labels" / split / f"{name}.txt").write_text("\n".join(lines), encoding="utf-8")
    manifest.append(dict(name=name, split=split, frame=f, t=round(t, 1), n_boxes=len(lines), **{c: cls.count(c) for c in CLASSES}))
cap.release()
(DS / "data.yaml").write_text(f"path: {DS.as_posix()}\ntrain: images/train\nval: images/val\nnames: {dict(enumerate(CLASSES))}\n", encoding="utf-8")
man = pd.DataFrame(manifest); man.to_csv(DS / "manifest.csv", index=False)
man.groupby("split")[["n_boxes"] + CLASSES].sum().assign(frames=man.groupby("split").size())'''),
("md", r'''## 2 · Correct the labels (outside the notebook)
The pseudo-labels carry the current detector's mistakes; training on them uncorrected only teaches the model to copy
itself. Import `data/datasets/<name>_det/` as a **YOLO** dataset in CVAT, Label Studio or Roboflow, fix the boxes
(missed far players, the ball, goalkeepers / referees), export back as YOLO into the same folder. Mark progress in
`manifest.csv` (`checked` column) — the evaluation below only trusts checked frames for the val split.'''),
("code", r'''if "checked" not in man:
    man["checked"] = False; man.to_csv(DS / "manifest.csv", index=False)
man = pd.read_csv(DS / "manifest.csv")
print(f"checked frames: train {man[(man.split == 'train')].checked.sum()}/{(man.split == 'train').sum()}, "
      f"val {man[(man.split == 'val')].checked.sum()}/{(man.split == 'val').sum()}")'''),
("md", "## 3 · Train\nNeeds `pip install ultralytics` (and the CUDA build of PyTorch on the NVIDIA machine). Set `TRAIN = True`. Small objects: train at `imgsz=1280` (or on 1024-px tiles), keep the mosaic augmentation, start from `yolo11s.pt`; ~50 epochs on a few hundred frames is a useful first model."),
("code", r'''TRAIN = False
BASE, EPOCHS, IMGSZ = "yolo11s.pt", 50, 1280
dev = accel.torch_device("auto")
cmd = f'yolo detect train data="{DS / "data.yaml"}" model={BASE} imgsz={IMGSZ} epochs={EPOCHS} batch=-1 device={dev.replace("cuda:", "")} project="{ROOT / "models" / "runs"}" name=det_{Path(ctx.video).stem}'
print("device:", dev, "\nequivalent CLI:\n ", cmd)
if TRAIN:
    from ultralytics import YOLO
    model = YOLO(BASE)
    r = model.train(data=str(DS / "data.yaml"), imgsz=IMGSZ, epochs=EPOCHS, batch=-1, device=dev,
                    project=str(ROOT / "models" / "runs"), name=f"det_{Path(ctx.video).stem}")
    print("best weights:", Path(r.save_dir) / "weights" / "best.pt")'''),
("md", "## 4 · Evaluate: current detector vs candidate on the val split\nPer class precision / recall at IoU 0.5 against the labels, ball recall within 10 px, ms per frame. Until the val labels are corrected the baseline is scored against its own output (≈ 100 %) — only the *checked* frames give a real number."),
("code", r'''CANDIDATE = None          # e.g. ROOT / "models/runs/det_<video>/weights/best.pt"
def load_labels(p, w, h):
    out = []
    for ln in p.read_text(encoding="utf-8").splitlines():
        c, cx, cy, bw, bh = ln.split(); cx, cy, bw, bh = float(cx) * w, float(cy) * h, float(bw) * w, float(bh) * h
        out.append((CLASSES[int(c)], [cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2]))
    return out
def evaluate(det_fn, rows):
    acc, ms = {c: dict(tp=0, fp=0, fn=0) for c in CLASSES}, []
    ball_hit = ball_n = 0
    for r in rows.itertuples():
        img = cv2.imread(str(DS / "images" / r.split / f"{r.name}.jpg")); h, w = img.shape[:2]
        gt = load_labels(DS / "labels" / r.split / f"{r.name}.txt", w, h)
        t0 = time.perf_counter(); d = det_fn(img); ms.append((time.perf_counter() - t0) * 1000)
        for c in CLASSES:
            G = np.array([b for k, b in gt if k == c], float).reshape(-1, 4); Pd = d.of(c).xyxy
            if c == "ball":
                for gb in G:
                    ball_n += 1; gc = (gb[:2] + gb[2:]) / 2
                    pc = (Pd[:, :2] + Pd[:, 2:]) / 2 if len(Pd) else np.zeros((0, 2))
                    ball_hit += bool(len(pc) and np.hypot(*(pc - gc).T).min() <= 10)
            s = detection_pr(Pd, G, 0.5)
            for k in ("tp", "fp", "fn"): acc[c][k] += s[k]
    tab = pd.DataFrame(acc).T
    tab["precision"] = tab.tp / (tab.tp + tab.fp).clip(lower=1); tab["recall"] = tab.tp / (tab.tp + tab.fn).clip(lower=1)
    tab.loc["ball", "recall_10px"] = ball_hit / max(ball_n, 1)
    tab["ms_per_frame"] = np.mean(ms)
    return tab
val = man[man.split == "val"]
val_checked = val[val.checked] if val.checked.any() else val
print(f"scoring on {len(val_checked)} val frames ({'checked' if val.checked.any() else 'UNCHECKED pseudo-labels'})")
scores = {"current (" + cfg.detection.backend + ")": evaluate(detector, val_checked)}
if CANDIDATE and Path(CANDIDATE).exists():
    from football_eval.detection import UltralyticsDetector
    cand = UltralyticsDetector(str(CANDIDATE), {c: c for c in CLASSES}, cfg.detection.conf_player, cfg.detection.conf_ball, 1280, "auto")
    scores["candidate"] = evaluate(cand, val_checked)
pd.concat(scores, names=["model", "class"]).round(3)'''),
("md", "## 5 · Deploy\nA config override that switches the backend; run notebook 06 with it (`config:` in `params.yaml`) to log the KPIs next to the baseline in `benchmarks.csv`, then a full `run_analysis.py --config ...`."),
("code", r'''override = ROOT / "configs" / "candidate_detector.yaml"
if CANDIDATE:
    import yaml
    base = yaml.safe_load(open(ROOT / P.config, encoding="utf-8")) or {}
    base.setdefault("detection", {}).update(backend="ultralytics", model_path=str(Path(CANDIDATE).relative_to(ROOT)) if Path(CANDIDATE).is_relative_to(ROOT) else str(CANDIDATE),
                                            class_map={c: c for c in CLASSES}, imgsz=1280)
    override.write_text(yaml.safe_dump(base, sort_keys=False), encoding="utf-8")
    print("written", override, "\nnext: params.yaml -> config: configs/candidate_detector.yaml, run 06, compare in benchmarks.csv")
    print(f'full run: python run_analysis.py --video "{P.video}" --config configs/candidate_detector.yaml --out outputs/{Path(ctx.video).stem}_candidate')
else:
    print("set CANDIDATE to the trained weights to write configs/candidate_detector.yaml")'''),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--execute", action="store_true")
    ap.add_argument("--only", default=None, help="substring of the notebook name, e.g. 02")
    a = ap.parse_args()
    NB_DIR.mkdir(parents=True, exist_ok=True)
    for name, cells in NB.items():
        if a.only and a.only not in name:
            continue
        nb = to_nb(cells)
        if a.execute:
            print("executing", name, file=sys.stderr)
            nb = execute(nb, NB_DIR)
        (NB_DIR / name).write_text(json.dumps(nb, indent=1), encoding="utf-8")
        print("->", NB_DIR / name)


if __name__ == "__main__":
    main()
