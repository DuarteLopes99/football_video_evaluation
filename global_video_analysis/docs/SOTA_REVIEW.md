# State of the art — automatic match analysis from a single camera

*Review for `football_video_evaluation`, October 2026.*

## 1. The problem in one picture

```
mp4 (one viewpoint) ─▶ detect people + ball ─▶ track ─▶ team / role / identity
                         │                                     │
                         └──▶ camera calibration (homography) ─┴─▶ 2-D "game state" (x, y per player & ball, per frame)
                                                                         │
                                         events (rules or learned) ◀─────┘─▶ stats, xG, timeline
```

The research community calls the middle step **Game State Reconstruction (GSR)**: turning video into
minimap coordinates with role, team and jersey number for every athlete. Events can then be spotted
either *directly from video* (learned action spotting) or *from the reconstructed tracking data*
(rule-based or learned on trajectories).

## 2. Benchmarks and what wins them (SoccerNet)

SoccerNet runs yearly challenges on broadcast football; the 2026 edition had 427 teams and 1,129 submissions.

| Task | What it measures | Best published approach | Score |
|---|---|---|---|
| **Game State Reconstruction** (2024) | athletes on the minimap with role / team / jersey | Constructor Tech: YOLOv5m detector; SegFormer camera-parameter regression + ResNet-18 pitch-keypoint refinement; DeepSORT **in pitch coordinates**; OSNet team embeddings; 2-head jersey-digit ResNet-18; tracklet merging by jersey + Re-ID | GS-HOTA **63.8** (baseline 23.4) |
| **Team Ball Action Spotting** (2025) | 12 ball actions + which team, 1-s tolerance | T-DEED re-implementation with a joint action-team head, mirroring augmentation with label swap | Team-mAP@1 **60.0** |
| **Player-Centric Ball Action Spotting** (2026) | action + team + jersey of the player | TAAD temporal transformer over ROI-aligned X3D player features, 4-model ensemble | F1 **58.9** (baseline 46.4) |
| **Ball Action Anticipation** (2026) | predict the next action 5 s ahead | FAANTRA + RegNetY-GSF backbone | mAP **24.1** |
| **SynLoc** (2026) | localise athletes on the pitch from one calibrated static 4K image | YOLO26-L with adaptive tiling + RTMPose-X ground-point, ray casting | mAP-LocSim **97.7** |
| **Multi-view foul recognition** (2025) | foul / severity from 2–4 synchronised replays | — | combined BA **52.2 %** |

Take-aways for us:
* **Calibration and post-processing drive GSR quality**, not the detector: the 2024 winner attributes its
  detection gains to pitch localisation and its association gains to tracklet post-processing.
* **Tracking in pitch metres** (not pixels) makes motion gating physical (no 180° turns, max speed).
* **Learned action spotting is good for ball actions (≈60 mAP) but needs labelled broadcast video** and
  GPUs; fouls are still hard even with multiple replay views (52 % balanced accuracy) — from one wide camera
  they must be inferred from the *restart* (free kick), as we do.
* With a **static** camera, ground localisation is essentially solved (SynLoc 97.7) once the camera is calibrated.

## 3. Component by component

### Detection
* YOLO family (v5 → v8 → YOLO11 → YOLO26) and DETR-style detectors (RT-DETR, RF-DETR) fine-tuned on football
  data with classes {player, goalkeeper, referee, ball}. Off-the-shelf COCO models detect `person` and
  `sports ball` reasonably but cannot separate roles.
* Small objects: tile the frame (SAHI-style) for the ball; players are 30–80 px tall in a wide lower-league shot.
* A 2026 study (YOLOv5x/v8x/v11x + SAM2) reports F1 ≈ 0.85 for players on a single fixed camera without
  fine-tuning (17/22 players found), with ball boys/referees as false positives.

### Tracking & identity
* ByteTrack / BoT-SORT / Deep-EIoU are the standard baselines; GSR winners add Re-ID embeddings and **jersey
  number recognition** to merge tracklets across the match.
* SAM2 mask propagation is an alternative for short clips (robust but heavy).

### Team / role assignment
* Unsupervised and per match: crop the torso → embedding → cluster into two teams. Roboflow's widely used
  recipe is **SigLIP embeddings → UMAP → KMeans(2)**; simple colour histograms work in good light but fail
  with shadows. Goalkeepers/referees are outliers handled by position (penalty area) or a third cluster.
* Left/right team side is decided by majority vote of mean x over frames.

### Camera calibration
* Fixed camera: calibrate once from ≥ 4 pitch landmarks (DLT / RANSAC homography) or a full pinhole model.
* Moving broadcast camera: per-frame **pitch keypoint detection** (e.g. 32-point YOLOv8-pose model from
  Roboflow `sports`, 74-point models in GSR) or line-based calibrators (NBJW-Calib, PnLCalib), temporal
  smoothing (Savitzky–Golay) of camera parameters.
* Expected error on a well calibrated single camera: ≈ 0.5 m average projection error.

### Ball
* Hardest object: tiny, motion-blurred, occluded, airborne. TrackNet (V2–V4, heat-map regression over frame
  stacks) is the reference for tiny fast balls; trajectory cleaning with physical speed limits + interpolation.
* An airborne ball cannot be placed on the ground plane from one view without physics (parabolic) fitting.

### Events from tracking data
* Vidal-Codina et al. (2022) detect events from tracking using **possession zones around players** (0.5–1 m),
  **duel zones**, ball direction/speed changes and **set-piece spatial patterns** (kick-off, corner, throw-in,
  goal kick, penalty; otherwise free kick). Against manual event data: **passes > 90 % precision & recall**,
  set pieces > 90 %, shots 78 % precision / 53 % recall; tackles not detectable without limb tracking.
  This is the backbone of our rule engine, adapted to noisier single-camera tracks.

### Expected goals
* Location (distance, angle) explains most of xG; StatsBomb 360 "freeze frames" add goalkeeper position,
  defenders in the shooting triangle and pressure (Soccermatics lesson 7 uses exactly these features with a
  shallow network). Body part and shot height add the rest — not observable from one wide camera.

## 4. What this means for a single, fixed/panning lower-league camera

| Aspect | Broadcast (SoccerNet) | Our setting |
|---|---|---|
| Viewpoint | cuts, replays, close-ups | one continuous wide view — **no shot-boundary problem** |
| Calibration | per frame | **once per venue** if the camera is fixed; light pan → flow + anchors |
| Player size | 50–200 px | 25–80 px → tiling, high resolution |
| Jersey numbers | readable in close-ups | rarely readable → player identity = tracklets |
| Labelled data | 500+ games | none → **unsupervised clustering + rules + synthetic evaluation** |
| Compute | GPU | laptop CPU / Apple-Silicon (MPS) |

**Design decision**: a *modular two-layer system* — learned perception where pretrained models exist
(detection, embeddings, calibration) and a transparent **rule engine** on pitch-coordinate tracks for events
and stats (no labelled video needed, every rule documented and testable). The learned action-spotting route
(T-DEED / TAAD) is the upgrade path once a few matches are annotated.

## 5. Roadmap

1. **Now (v0.1, this repo)** — YOLOX COCO detection (cv2, no torch), IoU tracker, colour clustering,
   static / flow / keypoint homography, rule-based events, StatsBomb-trained xG, synthetic benchmark.
2. **Perception upgrade** — `pip install ultralytics`; fine-tune YOLO11/RF-DETR on a football dataset
   (Roboflow *football-players-detection*, SoccerNet-Tracking) + 200 frames of your own venue;
   `teams.backend: siglip`; ByteTrack.
3. **Calibration** — pitch-keypoint pose model for panning streams; Savitzky–Golay smoothing.
4. **Identity** — tracklet merging by Re-ID + jersey OCR (PARSeq / two-digit classifier) → true player stats.
5. **Ball** — TrackNet-style heat-map model trained on your footage; parabolic fit for aerial balls.
6. **Learned events** — annotate 2–3 matches (Team-BAS label format) and fine-tune T-DEED; keep rules as
   fallback / weak labels.
7. **Scoreboard OCR** — exact goal times and match clock from the broadcast overlay.

## 6. Adaptation to the actual club recordings (Oct 2026)

The first real match (UD Mansores vs Canedo FC, CFC TV) is filmed by **one camera that pans/tilts/zooms from a
fixed spot**, at 640 × 360. In the SoTA vocabulary this is a *PTZ broadcast camera without cuts*, so per-frame
camera calibration is required (as in SoccerNet GSR) — but because the optical centre is fixed we use a
**PTZ model** instead of a free homography per frame: background-feature rotation tracking (classic panorama /
sports-camera registration) + chamfer alignment to pitch lines and the grass edge + bundle adjustment of the venue
(camera position and the real pitch size). Ball detection at this resolution is the open problem
(TrackNet-style heat-map models are the reference); see `docs/RECORDING_PROFILE.md` for details and recording
recommendations (≥ 720p is the single biggest improvement).

## 7. Human-in-the-loop instead of more models (Oct 2026)

Professional providers (StatsBomb, Opta, Wyscout) combine automatic tracking with **human operators** who validate
events and identities; academic Game State Reconstruction systems are still evaluated with ground-truth team sides,
pitch size and periods. For a single 640 × 360 stream the cheapest accuracy gain is the same: let the person who
knows the match supply the few facts that the camera cannot see well (pitch size, kits, direction, periods, score)
and spot-check the outputs, then *measure* the quality instead of assuming it. Implemented as the setup / review UI
and the provenance ledger — evaluation and numbers in `docs/MANUAL_VALIDATION.md`.

## Sources
* SoccerNet 2026 Challenges Results — https://arxiv.org/pdf/2607.07320
* SoccerNet 2025 Challenges Results — https://arxiv.org/html/2508.19182v1
* SoccerNet Game State Reconstruction (Somers et al., CVPRW 2024) — https://arxiv.org/abs/2404.11335v1
* From Broadcast to Minimap: SoTA SoccerNet GSR (2025) — https://arxiv.org/html/2504.06357v1
* AI Driven Soccer Analysis Using Computer Vision (2026) — https://arxiv.org/pdf/2604.08722
* Vidal-Codina et al., Automatic event detection in football using tracking data — https://arxiv.org/pdf/2202.00804
* Roboflow sports (player detection, SigLIP team clustering, pitch keypoints) — https://blog.roboflow.com/camera-calibration-sports-computer-vision/
* Roboflow football field keypoint model — https://huggingface.co/Simon9/football-field-detection-roboflow
* TrackNet / TrackNetV4 — https://arxiv.org/pdf/1907.03698 , https://arxiv.org/pdf/2409.14543
* Soccermatics — xG with tracking (freeze-frame) data — https://soccermatics.readthedocs.io/en/latest/_sources/gallery/lesson7/plot_xG_tracking.rst.txt
* StatsBomb open data — https://github.com/statsbomb/open-data
