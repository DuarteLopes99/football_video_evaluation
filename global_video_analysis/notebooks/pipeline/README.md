# Pipeline notebooks: every stage, cell by cell

One notebook per stage of the pipeline, so each model / rule can be inspected, measured and replaced on its own.

| notebook | stage | writes (`outputs/_pipeline_nb/<run_name>/`) |
|---|---|---|
| `00_setup_and_environment` | where the models run (GPU / CPU), the config actually used, the match setup | |
| `01_video_to_frames` | decoding, frame grid, decode speed, grass mask / detector ROI, camera motion | `frames_index.csv` |
| `02_player_detection` | tiles → letterbox → raw YOLOX output → decode → thresholds → NMS → merge; window run; scoring vs an annotated frame | `detections.csv` |
| `03_pitch_calibration` | clicked landmarks + reprojection error, homography provider over the window, players in metres | `homography.csv` |
| `04_tracking_and_teams` | tracker, track quality, zones / person types, kit descriptors, kit classifier, vote per track | `tracks.csv`, `team_assignment.csv` |
| `05_ball_detection_tracking` | detector candidates, ball tracker step by step (source of each measurement), cleaning in metres | `ball_raw.csv`, `ball.csv` |
| `06_integrated_perception` | `run_perception` on the same window, compared with 02–05; KPIs appended to `benchmarks.csv` | `perception/`, `../benchmarks.csv` |
| `07_analytics_events_stats` | `run_analytics` unrolled: periods, possession, dead ball, events, xG, stats, timeline | |
| `08_model_training_cycle` | export frames + pseudo-labels (YOLO), label, train (ultralytics), evaluate vs current detector, deploy via a config override | `data/datasets/<video>_det/` |

## Use

1. Edit `params.yaml` (video, match setup, a 10–60 s window, `full_run_dir` of a finished run for 07 / 08).
2. Run the notebooks in order (each reads the previous stage's table). After changing one stage (a new model in 02,
   a tracker setting in 04) re-run that notebook and the ones after it.
3. 06 logs one row per experiment in `outputs/_pipeline_nb/benchmarks.csv`: give it a `label`.

Another parameter file: `FVE_NB_PARAMS=path/to/params.yaml jupyter lab`.

The notebooks are generated from `scripts/build_pipeline_notebooks.py` (edit the cells there, then
`python scripts/build_pipeline_notebooks.py`; `--execute` runs them and embeds the outputs).
