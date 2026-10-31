# Temporary Global Repository Analysis

This is a working map for individual analysis, not a replacement for the rulebook or user guide. It captures the current repository structure and the main processing paths as inspected on 2026-10-30.

## 1. Repository at a glance

This project processes single-camera football video into player and ball tracks, football events, statistics, xG, opportunity clips, and a reviewable HTML report. It also provides a local browser launcher for match setup, processing, review, and replay.

The main workflow is:

```text
video + YAML config + optional match setup
  -> run_analysis.py
  -> perception (sample frames, detect, track, classify, project to pitch, clean ball)
  -> reusable tracking tables
  -> analytics (context, possession, events, xG, statistics, opportunities, confidence)
  -> CSV/JSON products, clips, provenance, report.html
  -> optional human review and analytics rerun
```

The repository also has a separate quick visual tracker (`simple_track.py` and `football_eval/simple/`), a standard-library web UI (`football_eval/ui/`), eight evaluation/walkthrough notebooks, and synthetic test data.

## 2. Main entrypoints and run modes

| Entry | Role |
|---|---|
| [`run_analysis.py`](run_analysis.py) | Main CLI and pipeline coordinator. `--video` runs perception then analytics; `--tracks` reruns analytics from saved tracks; `--synthetic` generates a synthetic run; setup/review inputs and repair options alter those paths. |
| [`scripts/launcher.py`](scripts/launcher.py) | Starts the local launcher workflow and can select a video/config/setup or start a run. |
| [`start.command`](start.command) | macOS double-click launcher wrapper. |
| [`scripts/match_ui.py`](scripts/match_ui.py) | Standalone setup/review UI entrypoint. |
| [`simple_track.py`](simple_track.py), [`debug_simple.py`](debug_simple.py) | Short visual tracking and debugging workflow, separate from the full analytics pipeline. |
| [`scripts/calibrate_points.py`](scripts/calibrate_points.py), [`scripts/calibrate_ptz.py`](scripts/calibrate_ptz.py) | Static/flow camera and PTZ venue calibration tools. |
| [`scripts/render_debug_video.py`](scripts/render_debug_video.py) | Renders annotated video from saved tracks for visual inspection. |
| [`scripts/cut_opportunities.py`](scripts/cut_opportunities.py) | Cuts clips for flagged or all opportunities and can add links to the report. |
| [`scripts/train_xg.py`](scripts/train_xg.py) | Rebuilds the xG model from the shot dataset. |
| [`scripts/get_models.py`](scripts/get_models.py) | Retrieves/copies the detector model used by the pipeline. |
| [`scripts/build_notebooks.py`](scripts/build_notebooks.py) | Notebook build/execution helper. |

## 3. Core processing map

| Stage | Main functions/classes | File(s) | Responsibility |
|---|---|---|---|
| Configuration | Config dataclasses and YAML loader | [`football_eval/config.py`](football_eval/config.py) | Defaults, thresholds, nested configuration, and YAML overrides. |
| Video sampling | `processing_windows` and video reader | [`football_eval/perception.py`](football_eval/perception.py), [`football_eval/video.py`](football_eval/video.py) | Select configured periods/windows and sample frames for analysis. |
| Detection | `YoloxOnnxDetector`, `tile_plan`, `merge_cut_boxes`, `build_detector` | [`football_eval/detection.py`](football_eval/detection.py) | Person/object inference, ONNX/runtime backends, pitch-aware tiling, and detector selection. |
| Player tracking | `IoUTracker`, `ByteTrackWrapper` | [`football_eval/tracking.py`](football_eval/tracking.py) | Associate detections over time and produce track IDs. |
| Team assignment | `TeamClassifier`, `KitClassifier`, `TrackVotes`, `assign_goalkeepers`, `build_kit_bank` | [`football_eval/teams.py`](football_eval/teams.py), [`football_eval/perception.py`](football_eval/perception.py) | Classify kits, combine track-level votes, and identify goalkeeper/team assignments. |
| Camera and pitch | `Pitch`, `Homography`, `build_provider`; `PTZCamera`, `PTZProvider`, `bundle_adjust`, `track_video` | [`football_eval/pitch.py`](football_eval/pitch.py), [`football_eval/ptz.py`](football_eval/ptz.py) | Convert image coordinates into pitch coordinates with static, flow, keypoint, or PTZ camera models. |
| Ball | `MotionBallDetector`, `clean_ball` | [`football_eval/ballcv.py`](football_eval/ballcv.py), [`football_eval/ball.py`](football_eval/ball.py) | Generate ball candidates and clean/gap-fill/smooth the resulting trajectory. |
| Perception orchestration | `run_perception`, `save_perception`, `repair_run`, `_perception_provenance` | [`football_eval/perception.py`](football_eval/perception.py), [`run_analysis.py`](run_analysis.py) | Coordinate frame processing, write reusable tables, and selectively repair camera/perception segments. |
| Match context | `infer_periods`, `player_speeds` | [`football_eval/context.py`](football_eval/context.py) | Match clock/period interpretation, attack direction context, and speed estimates. |
| Possession | `frame_control`, `touches`, `team_possession`, `dead_ball` | [`football_eval/possession.py`](football_eval/possession.py) | Estimate control, touches, possession sequences, and dead-ball stretches. |
| Event detection | `EventDetector.run` | [`football_eval/events.py`](football_eval/events.py) | Derive stoppages, restarts, passes, interceptions, recoveries, shots, goals, and proxy fouls/offsides. |
| xG | `XGModel`, `feature_row` | [`football_eval/xg.py`](football_eval/xg.py) | Turn shot context/location into xG using the JSON model coefficients. |
| Statistics | `team_stats`, `player_stats` | [`football_eval/stats.py`](football_eval/stats.py) | Aggregate team and per-tracklet performance, shape, physical, and PPDA metrics. |
| Opportunities | `build_opportunities`, `evaluate_opportunities` | [`football_eval/opportunities.py`](football_eval/opportunities.py) | Group attacking moves, assign categories/danger/flags, evaluate coverage, and support clip creation. |
| Confidence and explanation | `grade_metrics`, `data_quality`; event explanation/provenance helpers | [`football_eval/confidence.py`](football_eval/confidence.py), [`football_eval/explain.py`](football_eval/explain.py) | Qualify metric reliability and state why events/values exist and whether they are manual, measured, automatic, default, or assistant-provided. |
| Analytics orchestration | `run_analytics`, `save_analytics` | [`football_eval/analytics.py`](football_eval/analytics.py) | Connect context, possession, event detection, xG, statistics, timeline, opportunities, validation, and exports. |
| Timeline/report | `build_timeline`, `html_report` | [`football_eval/timeline.py`](football_eval/timeline.py), [`football_eval/report.py`](football_eval/report.py) | Build event timeline/highlight commands, plots, tables, and `report.html`. |

## 4. Human-in-the-loop setup and review

The manual workflow supplies high-impact facts that are unreliable to infer from a single camera: pitch and camera calibration, periods/direction, team kit examples, match sheet, event corrections, and excluded stretches.

| Component | Key symbols / role | File(s) |
|---|---|---|
| Setup schema and gate | `checklist`, `apply_to_config`; setup JSON validation and transfer into runtime config | [`football_eval/match_setup.py`](football_eval/match_setup.py) |
| Review and reconciliation | Frame sampling, event candidates, `apply_review`, `reconcile_match_sheet`, reliability calculations | [`football_eval/review.py`](football_eval/review.py) |
| UI routes and frame access | Setup/review endpoints for frames, detection, geometry, kits, fitting, review, and image-to-pitch operations | [`football_eval/ui/server.py`](football_eval/ui/server.py), [`scripts/match_ui.py`](scripts/match_ui.py) |
| Launcher and live/replay views | Job and video APIs, live state, overlay data, docs, and browser UI | [`football_eval/ui/launcher.py`](football_eval/ui/launcher.py), [`football_eval/ui/launcher.html`](football_eval/ui/launcher.html), [`football_eval/ui/viewer.py`](football_eval/ui/viewer.py), [`football_eval/ui/app.html`](football_eval/ui/app.html), [`football_eval/live.py`](football_eval/live.py) |

Typical sequence: setup UI writes `configs/match_setup_*.json`; full video processing writes perception tables; review writes `outputs/<match>/review.json`; rerunning with `--tracks` applies review and recomputes analytics without repeating detector work. Camera repair can require targeted perception processing before analytics reruns.

## 5. Important data products

| Product | Typical location | Meaning |
|---|---|---|
| `tracks.csv` | `outputs/<match>/` | Player detections/tracks and pitch coordinates. |
| `ball_raw.csv`, `ball.csv` | `outputs/<match>/` | Ball candidates and cleaned per-frame ball positions. |
| `frames.csv`, `camera_log.csv`, `team_assignment.csv` | `outputs/<match>/` | Frame-level processing, camera pose/status, and team assignment when available. |
| `events.csv`, `events_rejected.csv` | `outputs/<match>/` | Accepted event records and rejected/corrected candidates. |
| `team_stats.csv`, `player_stats.csv`, `key_metrics.csv` | `outputs/<match>/` | Aggregated statistics and confidence-graded headline metrics. |
| `timeline.csv`, `timeline.json`, `stoppages.csv` | `outputs/<match>/` | Time-indexed match events and dead-ball intervals. |
| `opportunities.csv`, opportunity clips | `outputs/<match>/` | Grouped attacking moves, flags, and review clips. |
| `meta.json`, `match_context.json`, `provenance.json`, validation outputs | `outputs/<match>/` | Run settings, match facts, data lineage, and validation status. |
| `report.html`, `cut_highlights.sh` | `outputs/<match>/` | Human-facing report and generated highlight commands. |

For field definitions and reliability caveats, read [`docs/STATS_RULEBOOK.md`](docs/STATS_RULEBOOK.md) and [`docs/REPORT_GUIDE.md`](docs/REPORT_GUIDE.md) alongside the code that writes each table.

## 6. Configs, data, and model files

- [`configs/`](configs/) contains default/preview/example/venue settings, storage libraries, PTZ calibration/landmarks, and match setup JSON. Treat configuration values as active algorithm inputs, not merely documentation.
- [`data/annotations/`](data/annotations/) contains example/manual labels used by evaluation workflows.
- [`data/xg/shots_raw.jsonl`](data/xg/shots_raw.jsonl) is the shot dataset consumed by xG training; [`models/xg_model.json`](models/xg_model.json) is the runtime model.
- `models/yolox.onnx` is a binary detector model. Video files, model weights, archives, clips, image previews, and large generated output tables were inventoried by role, not decoded or reviewed row-by-row.
- `data/raw/` is present but empty in the inspected tree. Video storage behavior is documented in [`docs/VIDEO_STORAGE.md`](docs/VIDEO_STORAGE.md) and implemented in [`football_eval/storage.py`](football_eval/storage.py).

## 7. Tests and notebooks

| File(s) | Analysis purpose |
|---|---|
| [`tests/test_pipeline.py`](tests/test_pipeline.py) | Synthetic pipeline regression tests plus geometry, xG, PTZ, setup/review, opportunities, storage/UI, tiling, kits, and processing-window behavior. |
| [`tests/test_simple.py`](tests/test_simple.py) | Tests for the separate simple tracker. |
| [`notebooks/01_perception_eval.ipynb`](notebooks/01_perception_eval.ipynb) | Detection, team clustering, calibration, annotated-frame scoring. |
| [`notebooks/02_event_detection_eval.ipynb`](notebooks/02_event_detection_eval.ipynb) | Synthetic event/possession quality, noise robustness, threshold sensitivity. |
| [`notebooks/03_match_report_walkthrough.ipynb`](notebooks/03_match_report_walkthrough.ipynb) | Walkthrough of report outputs. |
| [`notebooks/04_xg_model_eval.ipynb`](notebooks/04_xg_model_eval.ipynb) | xG training/evaluation and calibration. |
| [`notebooks/05_canedo_recording_eval.ipynb`](notebooks/05_canedo_recording_eval.ipynb) | PTZ, detection, teams, positions, ball, and stats on Canedo footage. |
| [`notebooks/06_manual_validation_eval.ipynb`](notebooks/06_manual_validation_eval.ipynb) | Value/sensitivity of human inputs and review. |
| [`notebooks/07_opportunities_eval.ipynb`](notebooks/07_opportunities_eval.ipynb) | Opportunity coverage, precision, time saved, and threshold sensitivity. |
| [`notebooks/08_speed_and_kits_eval.ipynb`](notebooks/08_speed_and_kits_eval.ipynb) | Runtime and kit evaluation. |

The related written rationale is in [`docs/`](docs/): user guide, recording profile, stats rulebook, report guide, manual validation, SOTA review, storage, and simple tracker guide. Notebook cell code was inspected; notebook results should be treated as saved exploratory evidence, not proof that every current code path passes.

## 8. Suggested order for individual analysis

1. Read [`run_analysis.py`](run_analysis.py), then [`football_eval/perception.py`](football_eval/perception.py) and [`football_eval/analytics.py`](football_eval/analytics.py). This reveals the main orchestration contracts and files passed between stages.
2. Follow one perception path: [`football_eval/video.py`](football_eval/video.py) -> [`football_eval/detection.py`](football_eval/detection.py) -> [`football_eval/tracking.py`](football_eval/tracking.py) -> [`football_eval/teams.py`](football_eval/teams.py) -> [`football_eval/pitch.py`](football_eval/pitch.py) / [`football_eval/ptz.py`](football_eval/ptz.py) -> ball modules.
3. Follow one event path: [`football_eval/context.py`](football_eval/context.py) -> [`football_eval/possession.py`](football_eval/possession.py) -> [`football_eval/events.py`](football_eval/events.py) -> [`football_eval/xg.py`](football_eval/xg.py) -> [`football_eval/stats.py`](football_eval/stats.py).
4. Compare the computed facts against [`football_eval/review.py`](football_eval/review.py), [`football_eval/confidence.py`](football_eval/confidence.py), and [`football_eval/explain.py`](football_eval/explain.py) to understand correction and trust boundaries.
5. Trace exports through [`football_eval/analytics.py`](football_eval/analytics.py), [`football_eval/opportunities.py`](football_eval/opportunities.py), [`football_eval/timeline.py`](football_eval/timeline.py), and [`football_eval/report.py`](football_eval/report.py).
6. Use the focused test(s) and corresponding evaluation notebook to inspect assumptions and known quality limits; consult the rulebook before interpreting a metric as ground truth.
7. Analyze the launcher/UI and simple tracker separately; neither is the core event/stat computation path.

## 9. Inspection boundaries and follow-up checks

- The text-based Python modules, scripts, tests, docs, notebooks, UI HTML, and project configuration/data files were covered at architecture/symbol level. There are 53 Python files in the inventory: 39 package modules, nine scripts, three root-level Python entrypoints, and two test modules; there are eight notebooks.
- Generated output directories were classified by file/product role; large CSV contents were not exhaustively audited row-by-row. Binary videos, model weights, archives, and embedded base64 kit thumbnails were not decoded.
- The documented Canedo PTZ run appears tied to a 640x360 source filename absent from the current `data/videos/` listing, while the available Canedo video has different dimensions/name. The PTZ provider rejects a width mismatch; verify the intended source/calibration pairing before treating that run as reproducible.
- The J3 output includes `--max-frames 600` in launcher metadata, so interpret it as a short smoke run rather than full-match validation.
- No tests were run as part of this repository map. Run the relevant focused test before relying on behavior-level conclusions.

This file is intentionally temporary and can be removed after the individual code reviews are complete.