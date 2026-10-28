# football_video_evaluation

Automatic analysis of a **full football match filmed from a single camera** (one viewpoint, e.g. a club
live-stream or a tripod at the halfway line): team statistics, expected goals, set pieces, fouls (proxy) and a
**timeline of the major events** with timestamps and highlight clips.

```
match.mp4 ──▶ PERCEPTION ─────────────────────────────────▶ tracks.csv / ball.csv (pitch metres)
              detect (YOLOX/YOLO) → track → teams (colour/SigLIP)
              → homography (static | flow | keypoints) → ball cleaning
                                   │
                                   ▼
              ANALYTICS (rules, docs/STATS_RULEBOOK.md) ──▶ events.csv, team_stats.csv, player_stats.csv,
              possession → stoppages → set pieces → passes      timeline.csv/json, report.html,
              → shots (+xG) → goals → fouls (proxy) → stats     cut_highlights.sh,
              → OPPORTUNITIES (⚑ flagged chances)               opportunities.csv + clips/opportunities/*.mp4
```

* State of the art and design rationale: [`docs/SOTA_REVIEW.md`](docs/SOTA_REVIEW.md)
* Every statistic, rule, threshold and reliability level: [`docs/STATS_RULEBOOK.md`](docs/STATS_RULEBOOK.md)
* What you validate by hand, why, and how it is used: [`docs/MANUAL_VALIDATION.md`](docs/MANUAL_VALIDATION.md)
* What the report contains, which metrics matter and how far to trust each one: [`docs/REPORT_GUIDE.md`](docs/REPORT_GUIDE.md)
* Where the recordings live and how to keep a season without filling the disk: [`docs/VIDEO_STORAGE.md`](docs/VIDEO_STORAGE.md)

## Start here: the launcher

```bash
python scripts/launcher.py          # or double-click "Football Video Analysis" on the Desktop / start.command
python scripts/launcher.py --video 2026-10-18_UD_Mansores_vs_Canedo_FC_J2.mp4 --config configs/canedo_cfctv.yaml --start
```

**How to use it, step by step:** [`docs/USER_GUIDE.md`](docs/USER_GUIDE.md), also in the launcher's **Guide** tab (with
this README and every other guide).

| Option | Effect |
|---|---|
| `--video <file>` | that recording selected (file name in any video library, or a path); its match setup and the config recorded in it are pre-selected |
| `--config`, `--setup`, `--run` | override the config / setup / output folder |
| `--open setup\|review\|screen\|live\|results\|guide` | screen to open first |
| `--start` (`--force`) | start processing at once (and ignore red checklist items); the Match screen opens in *Live* |

If the launcher is already running, a second start just opens it in the browser with the new choice. The
**Desktop shortcut** button (Analyse tab) writes `~/Desktop/Analyse <video>.command` with these options, optionally
`--start`. **Quit** (top right) stops the launcher and its jobs. On macOS, processing jobs run under `caffeinate`,
so the Mac does not idle-sleep in the middle of a match.

One local web page for the whole workflow, with no extra dependencies (standard-library server):

* **Video library** (left) — every recording in `data/videos/` and in the other libraries of
  `configs/storage.yaml` (external drive, Google Drive / OneDrive / iCloud folder). Drop an `.mp4` on it to add one.
  **Move to archive** sends an analysed match to the archive library.
* **Analyse** — choose the venue config and the match setup, then run the four steps: *Setup* → *Process the
  video* (with a "quick test: first N frames" option) → *Review* → *Re-run analytics*. Each step shows the exact
  command it runs, and a live log with progress and time left.
* **Manual decisions** — the setup and the review, inside the launcher (no separate window or port). *Setup*
  covers pitch size, periods and direction, overlays, kit clicks, pitch landmarks, the match sheet and the
  checklist. *Review* covers frame checks, camera repair, key events (confirm / reject / flip team / shirt number /
  shot location), missing events and excluded stretches. Everything is saved as you go to `configs/match_setup_*.json`
  and `outputs/<match>/review.json`; *Apply review* re-runs the analytics. Same screens as `scripts/match_ui.py`.
* **Match screen** — the game with every analysis overlay drawn on it:
  * *Live (processing)*: while step 2 runs, the frame being analysed right now, with the projected pitch lines,
    every detected player (box + track id, in his team's shirt colour once the kits are clicked), ball candidates,
    a minimap and the camera status, refreshed every second. Progress, speed, time left and the log are on the side
    (`football_eval/live.py` writes `outputs/<match>/live/`; `--no-live` turns it off).
  * *Replay (analysed)*: the full video plays in the browser and the analysis is drawn on top: pitch lines from the
    camera pose of each frame, players by team, goalkeepers and referee, the ball, event banners (shots with xG,
    goals, set pieces), a minimap with the passes and shots of the last seconds, and a red band where the camera
    calibration was lost. A timeline shows the processed stretch, shots, goals and corners — click to jump; `space`,
    `←/→`, `n/p` for play, ±5 s, next / previous event. Each overlay can be switched off.
* **Jobs** — every command started in this session with its log.
* **Guide** — the user guide, this README and the other guides, rendered with a table of contents.

## Major opportunities ⚑

Every run builds `opportunities.csv` (`football_eval/opportunities.py`, rulebook §7c). Shots, goals, penalties, box
entries and set pieces are grouped into attacking moves. Each move gets a category (Goal, Penalty, Big chance, Shot
on target, Shot, Box entry with no shot, Set piece near goal), a *chance xG* and a 0–100 *danger*. **Flagged ⚑** =
goal, penalty, big chance (xG ≥ 0.30), shot on target, or any shot with xG ≥ 0.10. These are the first section of
`report.html`. Each flagged move has a **▶ clip** (`outputs/<match>/clips/opportunities/`, cut automatically after
each run) and a **full video @ mm:ss** link. On synthetic matches, every true goal and 97 % of the shots on
target fall inside a flagged move, and the flagged clips take ~12 % of the playing time to watch
(`notebooks/07_opportunities_eval.ipynb`).

```bash
python scripts/cut_opportunities.py --run outputs/match1 [--all]     # (re)cut clips; adds the ▶ links to the report
```

## Manual, explainable workflow (recommended)

A single 640 × 360 camera cannot reliably *infer* some facts that you know in seconds, and a few wrong guesses
change every statistic: pitch size, which team is which kit, direction of play, when the halves start, the score.
So the pipeline has two short **human-in-the-loop** steps in a local browser UI, and every output says
**why** and **who said so**:

```
 SETUP UI (~10 min)                      PROCESS (hours, CPU)                 REVIEW UI (~10 min)
 pitch size · periods · direction   ──▶  perception + analytics       ──▶     ~16 frame checks (lines / teams)
 overlays · kit clicks · landmarks       (uses match_setup.json)              goals & shots: confirm / reject /
 match sheet (score, goals, cards)       checklist gate before running        flip team, shirt numbers, missing events
        │                                                                              │
        └──────────── match_setup.json ─────────────▶ ANALYTICS ◀──── review.json ────┘
                                             events.csv (+ why, validation), events_rejected.csv,
                                             provenance.json, validation_summary.csv, report.html
```

```bash
python scripts/match_ui.py setup  --video data/videos/match.mp4 --setup configs/match_setup_match1.json --config configs/<venue>.yaml
python run_analysis.py --video data/videos/match.mp4 --config configs/<venue>.yaml --setup configs/match_setup_match1.json --out outputs/match1
python scripts/match_ui.py review --run outputs/match1 --setup configs/match_setup_match1.json --config configs/<venue>.yaml
python run_analysis.py --tracks outputs/match1 --repair --config configs/<venue>.yaml --setup configs/match_setup_match1.json --out outputs/match1
```

The report opens with the score, a **data-quality** summary and the metrics **graded by confidence for this match**
(High / Medium / Low / Experimental, with the reason). On this kind of recording, after the setup and review: result,
scorers and cards are High; shots, shots on target, xG (with clicked shot locations) and team shape (defensive line
height, block depth / width) are Medium; possession and set pieces are Low; passing and pressing are Experimental.

What was evaluated and made manual (details and numbers in `docs/MANUAL_VALIDATION.md`):

| Manual | Not manual (and why) |
|---|---|
| pitch size, overlays, periods, 1st-half direction | ball positions, player boxes (thousands of frames) |
| kit clicks for both teams, both goalkeepers and the referee (seeded classification, no team swap) | shirt number per tracklet: ~19k tracklets per match, median 1.8 s, players ~25 px tall → numbers are typed only on key events |
| pitch landmarks with a live overlay and per-point error | camera position (solved robustly from the landmarks) |
| match sheet: score, goal minutes, cards, substitutions, rosters | |
| review: ~16 sampled frames + every goal / shot | |
| camera repair on lost stretches (2–6 landmark clicks, only those windows re-processed) | |
| shot and keeper location (2 clicks per shot → xG recomputed) | |
| excluded stretches (replays, crowd, injury breaks) | |

## Quick start (macOS / Linux, CPU — no GPU or PyTorch needed)

```bash
cd football_video_evaluation
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python scripts/get_models.py                 # YOLOX detector (copied from ../offiside_code if present)

# 0) sanity check without any video: synthetic match with ground truth
python run_analysis.py --synthetic --out outputs/synthetic_demo
python tests/test_pipeline.py

# 1) calibrate the camera once per venue (click >= 6 pitch landmarks on a frame)
python scripts/calibrate_points.py --video data/videos/match.mp4 --t 600 --out configs/calibration_myvenue.json

# 2) copy configs/default.yaml -> configs/mymatch.yaml and set
#    teams.team_names, pitch.calibration_file, video.ignore_regions (score bug), scoreboard.roi
# 3) run the full match. CPU cost is dominated by detection. Since v0.2: onnxruntime (CoreML on a Mac when
#    faster) + 2 pitch-aware tiles per 1080p frame instead of 6 -> ~6x faster (1.6 s -> 0.27 s per frame on 2
#    vCPUs; a 90-min match at 10 fps = 54k frames). Only the periods of the setup are processed.
#    --speed fast (5 fps) halves it again; --speed quality uses smaller tiles. Test first: --max-frames 600
python run_analysis.py --video data/videos/match.mp4 --config configs/mymatch.yaml --out outputs/match1

# re-run only the analytics after changing a rule threshold (seconds, not hours)
python run_analysis.py --tracks outputs/match1 --config configs/mymatch.yaml --out outputs/match1
```

Outputs in `outputs/<match>/`: `report.html` (⚑ opportunities + tables + shot map + possession + positions),
`opportunities.csv` / `clips/opportunities/`, `team_stats.csv`,
`player_stats.csv` (per tracklet), `events.csv`, `timeline.csv/json`, `stoppages.csv`, the raw tracking tables and
`cut_highlights.sh` (ffmpeg clips of goals & shots).

## Club recordings: one panning/zooming camera (first match: UD Mansores vs Canedo FC)

The target footage is filmed by **one operator-driven camera** next to the halfway line that pans, tilts and zooms
to follow play, streamed at 640 × 360 without score bug or audio (`docs/RECORDING_PROFILE.md`). The pipeline was
adapted to it:

* `football_eval/ptz.py` — pan-tilt-zoom camera model with a fixed camera centre: frame-to-frame rotation/zoom
  from background features + alignment of the pitch template to the white lines and to the far edge of the grass,
  with local re-localisation. Processing can start anywhere (the camera is tracked from the calibration anchor).
* `scripts/calibrate_ptz.py` — one-time venue calibration from a few clicked landmarks (camera position, roll,
  **pitch size**, grass margins). Canedo: `configs/camera_canedo_ptz.json` (pitch ≈ 105.4 × 61.6 m,
  camera ≈ 2.6 m behind the near touchline, ≈ 3.9 m high) from `configs/landmarks_canedo.json`.
* `football_eval/ballcv.py` — motion-compensated ball candidates (the ball is 2–5 px at 640 × 360).
* `configs/canedo_cfctv.yaml` (whole match) and `configs/canedo_segment.yaml` (10-min development segment).
* `scripts/render_debug_video.py` — annotated MP4 (boxes by team, projected pitch lines, minimap) to check by eye.
* `notebooks/05_canedo_recording_eval.ipynb` — camera-tracking quality, detection/teams, positions, ball, stats.

```bash
python run_analysis.py --video data/videos/2026-10-18_UD_Mansores_vs_Canedo_FC_J2.mp4 \
    --config configs/canedo_cfctv.yaml --out outputs/canedo_full          # ~0.6-0.9 s per analysed frame on CPU
python scripts/render_debug_video.py --video data/videos/2026-10-18_UD_Mansores_vs_Canedo_FC_J2.mp4 \
    --tracks outputs/canedo_full --config configs/canedo_cfctv.yaml --start 900 --end 960
```
Status on a 10-minute segment: camera pose usable in 63 % of frames, players placed inside the pitch, teams
separated; the ball (2–5 px) is the main open problem — see `docs/RECORDING_PROFILE.md` §3.
New venue: `python scripts/calibrate_ptz.py --video <match> --click <t>` on 2–4 frames (centre circle / halfway line
on one, goal-post bases on frames looking at each goal), then run it with `--out configs/camera_<venue>_ptz.json`.

## Folder structure

| Path | Content |
|---|---|
| `scripts/launcher.py`, `start.command` | **front end**: video library, run the steps, live logs, results with the flagged opportunities and a video player |
| `football_eval/ui/launcher.py`, `launcher.html` | the launcher (stdlib HTTP server with video range requests, job runner; mounts the setup / review UI at `/m/`) |
| `football_eval/ui/viewer.py` | Match screen replay data: per analysed frame homography, players, ball; events, opportunities |
| `football_eval/live.py` | live preview of the frame being processed (`outputs/<match>/live/latest.jpg`, `state.json`) |
| `football_eval/opportunities.py` | major opportunities: grouping, categories, ⚑ flags, danger, clips, evaluation |
| `scripts/cut_opportunities.py` | cut clips of the flagged (or all) opportunities and link them in the report |
| `football_eval/storage.py`, `configs/storage.yaml` | video libraries: listing, cloud-only detection, lookup by file name, archive |
| `data/videos/` | **the match recordings** (git-ignored); more libraries in `configs/storage.yaml` |
| `docs/VIDEO_STORAGE.md` | storage options: cloud folder on demand, external drive, keep only outputs + clips |
| `notebooks/07_opportunities_eval.ipynb` | coverage / precision / time saved of the flagged opportunities, threshold sensitivity |
| `run_analysis.py` | CLI: `--video` (full pipeline), `--tracks` (analytics only), `--synthetic` (demo + evaluation); `--setup` / `--review` apply the manual inputs |
| `scripts/match_ui.py` | browser UI for the manual steps: `setup` (before processing) and `review` (after) |
| `football_eval/ui/` | the UI: `server.py` (stdlib HTTP server + OpenCV frame access), `app.html` |
| `football_eval/match_setup.py` | `match_setup.json` schema, checklist (quality gate), setup → config |
| `football_eval/review.py` | frame sampling, event candidates, applying verdicts, match-sheet reconciliation, measured reliability (Wilson CI) |
| `football_eval/confidence.py` | metric catalogue (value ★), confidence grade per statistic for this match + data-quality summary → `key_metrics.csv` |
| `football_eval/explain.py` | `why` sentence per event + provenance ledger (manual / measured / auto / default / assistant) |
| `configs/match_setup_canedo_J2.json` | pre-filled setup for UD Mansores vs Canedo FC (sections marked `assistant` until you confirm them) |
| `football_eval/config.py` | all settings and thresholds (dataclasses, overridable from YAML) |
| `football_eval/video.py` | sub-sampled frame reader for long videos |
| `football_eval/detection.py` | detectors: YOLOX ONNX via onnxruntime / CoreML / cv2.dnn, pitch-aware tiling with seam merging, or any ultralytics model |
| `football_eval/tracking.py` | built-in IoU/Hungarian tracker or ByteTrack (supervision) |
| `football_eval/teams.py` | team clustering (colour descriptor or SigLIP), track voting, goalkeeper attribution; `KitClassifier` seeded by your kit clicks |
| `football_eval/pitch.py` | IFAB pitch model, landmarks, homography fitting, static / flow / keypoint providers, minimap drawing |
| `football_eval/ball.py` | ball trajectory cleaning (speed gating, gap interpolation, smoothing) |
| `football_eval/scoreboard.py` | score-bug change detector (goal confirmation), optional OCR |
| `football_eval/perception.py` | perception orchestration → `tracks.csv`, `ball.csv` |
| `football_eval/context.py` | periods, attacking direction, football clock, player speeds |
| `football_eval/possession.py` | ball control, touches, team possession |
| `football_eval/events.py` | stoppages, restarts, passes, interceptions, recoveries, shots, goals, fouls/offsides (proxy) |
| `football_eval/xg.py` | xG features + model (JSON coefficients) |
| `football_eval/stats.py` | team and per-tracklet statistics, PPDA, shape, physical |
| `football_eval/timeline.py` | major-events timeline + highlight script |
| `football_eval/report.py` | figures + HTML report |
| `football_eval/synthetic.py` | synthetic match generator with ground-truth events + event evaluation |
| `football_eval/evaluation.py` | detection / team / possession metrics for the notebooks |
| `notebooks/01_perception_eval.ipynb` | detection, team clustering, calibration on a real frame (+ annotated-frame scoring) |
| `notebooks/02_event_detection_eval.ipynb` | event P/R/F1 vs ground truth, robustness to noise, threshold sensitivity |
| `notebooks/03_match_report_walkthrough.ipynb` | every output table and figure, explained |
| `notebooks/04_xg_model_eval.ipynb` | xG training data, CV metrics vs StatsBomb, calibration, xG surface |
| `notebooks/06_manual_validation_eval.ipynb` | which factors are worth doing by hand: fragmentation / shirt numbers, kit clicks vs clustering, xG sensitivity to position error, the pilot review |
| `scripts/calibrate_points.py` | click landmarks → calibration json (static camera or flow anchors) |
| `scripts/calibrate_ptz.py` | venue calibration for a pan/tilt/zoom camera (camera position, roll, pitch size) |
| `scripts/render_debug_video.py` | annotated MP4 for checking perception by eye |
| `football_eval/ptz.py` | PTZ camera model, background-motion tracker, line/grass-edge alignment, bundle adjustment |
| `football_eval/ballcv.py` | motion-compensated ball candidates for low-resolution video |
| `docs/RECORDING_PROFILE.md` | how the club matches are filmed, what it changes, recommendations |
| `scripts/train_xg.py` | (re)train the xG model on StatsBomb open data |
| `scripts/build_notebooks.py` | regenerate / execute the notebooks from source |
| `scripts/get_models.py` | fetch the default detector |
| `configs/default.yaml`, `configs/example_anadia.yaml` | default settings; example re-using the offside project's camera model |
| `models/xg_model.json` | trained xG coefficients |
| `data/xg/shots_raw.jsonl` | 8.8k StatsBomb shots used for training (Data: StatsBomb, CC BY-NC-SA) |
| `tests/test_pipeline.py` | regression tests (synthetic match) |

## Current performance (v0.1)

**Analytics on synthetic matches** (`notebooks/02`, 4 × 20 min, default synthetic noise; ±2 s, ±4 s for set pieces):

| Event | Precision | Recall |
|---|---|---|
| Passes | 0.99 | 0.99 |
| Interceptions | 1.00 | 1.00 |
| Shots | 1.00 | 0.85 |
| Goals | 1.00 | 1.00 |
| Kick-offs / throw-ins / goal kicks / corners | 1.00 / 0.95 / 1.00 / 0.90 | 1.00 / 0.98 / 1.00 / 0.90 |
| Free kicks / fouls (proxy) | 1.00 / 0.96 | 0.94 / 0.90 |
| Possession (per frame) | 98 % accuracy | |

With heavy noise (0.8 m jitter, 25 % ball dropout, ID switches every 30 s, unlabelled goalkeepers) pass F1
stays at 0.88 and goal F1 at 0.97, while interceptions (0.72), shots (0.74) and recoveries (0.41) degrade.
These are **upper bounds**: the synthetic engine has no aerial balls, deflections or crowded set pieces.
**xG** (StatsBomb open data, grouped 5-fold CV): AUC 0.78, Brier 0.080 vs StatsBomb's own xG 0.81 / 0.076.
**Perception** on a real lower-league stream frame: 24 people + ball detected in 1.3 s on CPU, camera model
re-projects onto the painted lines, kits separated except players in shadow and the referee
(see `notebooks/01`).

## Limitations and next steps
See the roadmap in `docs/SOTA_REVIEW.md` §5. In short: fine-tune a football detector, SigLIP team embeddings,
per-frame keypoint calibration for panning streams, tracklet merging with jersey numbers, a TrackNet-style ball
model, then learned action spotting (T-DEED) trained on a few annotated matches.
