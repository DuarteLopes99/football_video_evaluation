# Simple tracker: coloured players + ball

A small, separate pipeline (`football_eval/simple/`, about 650 lines) that draws every player in his team's colour
and follows the ball, on any match video, **with no calibration and no clicking**. It came from two LinkedIn demos:
Shabahang Shayegan's NBA Finals broadcast (outlined players with shirt numbers, ball tracking, 2D court map) and
Syco Coders' football broadcast (player + ball tracking with live team ball control % on screen).

```bash
python simple_track.py data/videos/J3_MansoresSanguedo_1080p.mp4 --start 24:38 --duration 30 --roi-top 0.2
python simple_track.py data/videos/2026-10-18_UD_Mansores_vs_Canedo_FC_J2.mp4 --start 10:00 --duration 30 \
       --roi-top 0.5 --upscale 3 --stride 2
```

Output in `outputs/simple/<video>/`: `annotated.mp4`, `tracks.csv` (frame, t, track_id, box, team) and
`ball.csv` (frame, t, x, y, r, state = det / interp).

| Option | Use |
|---|---|
| `--start`, `--duration` | the part of the match to process (`24:38`, `1:02:30`; duration `0` = to the end) |
| `--roi-top 0.2` | skip the top 20 % of the frame (sky, trees): faster, fewer false hits |
| `--stride 2` | every 2nd frame: twice as fast, output video at half the frame rate |
| `--upscale 3` | low-resolution streams (640x360): enlarge before detection, so far more players are found |

## How it works (one file per step)

| Step | File | What it does |
|---|---|---|
| Detect | `detection.py` (existing) | YOLOX COCO, `person` + `sports ball` (ball threshold 0.05) |
| Camera pan | `run.py` · `CameraMotion` | phase correlation between frames: the shift is applied to every track and to the ball filter, so fast pans do not break the tracks |
| Pitch | `segment.py` · `field_mask` | grass region of the frame. People must stand on it, and the ball must be on or just above it (no "balls" in the trees) |
| Track people | `tracking.py` (existing) | IoU tracker |
| Teams | `teams.py` | shirt chroma relative to the *local* grass → 2-component Gaussian mixture → **one colour per track for the whole clip** (majority vote, no flicker). Goalkeepers and referee = "other". The drawing colour has the hue of the real kit |
| Ball | `ball.py` | **Kalman filter** (rejects impossible jumps) + **zoomed re-detection** (2x crop around the prediction, so a 6–12 px ball becomes 15–25 px) + white-blob fallback + **static-object filter** (a "ball" that stays put for 1.5 s with nobody near is a tree or a post: dropped, and that spot is ignored afterwards), then gap interpolation and smoothing |
| Silhouettes | `segment.py` · `player_mask` | "simple segmentation": pixels in the box whose colour differs from the box border (= local grass, sun and shade), biggest blob, holes filled, then **grown ~10 % of the box width and rounded** so the whole player is always inside the outline (not pixel-exact on purpose). No SAM |
| Ball control | `run.py` · `_carrier` | the player whose feet are closest to the ball (within 1.5 box widths) has it; the team keeps control while the ball travels, until a player of the other team gets it. Share of frames per team = the "ball control" bar |
| Draw | `draw.py` | translucent fill + outline in team colour, ball circle + fading trail, triangle over the ball carrier, legend + ball control bar |

It runs in two passes. Pass 1 analyses and collects shirt colours. The teams are then decided once for the whole
clip, and pass 2 renders. That is why the colours are stable from the first frame.

## Results on the club videos (20 s clips, 30 October 2026)

Earlier versions reported 94 % and then 61 % on J3. The difference was objects tracked as the ball (a tree behind the
goal, something above the far wall). The static-object filter now removes them, including the frames before
they were recognised, so the remaining number is lower but honest.


| Video | Ball found | Teams |
|---|---|---|
| J3 Mansores–Sanguedo, 1080p | **~45 % of frames** (37 % detected, 8 % interpolated). Missing: a long goal kick in the air + a fast camera pan (~7 s) and the ball far away at the end (~3.5 s) | green vs striped correct; GK + referee = other |
| J2 Mansores–Canedo, 640x360 stream (`--upscale 3`) | ~30 % of frames (the ball is 2–5 px: the limit of a COCO detector) | yellow vs dark kit mostly correct |

## Versus the LinkedIn demo

| Demo | Here |
|---|---|
| Broadcast HD, players large | single club camera, players often < 60 px |
| Pixel-accurate masks (most likely SAM-2 type) | colour-difference silhouettes (cheap; rougher with motion blur and on far players, where it falls back to an ellipse) |
| Shirt numbers | not done (track ids only) |
| 2D court map | not here. The main pipeline (`perception.py` + `pitch.py`) already has the pitch homography |

## Limits and next steps

* **Low-resolution ball**: at 360p the COCO detector almost never sees the ball. A ball model fine-tuned on a few hundred
  labelled frames of these venues is the real fix (see `docs/SOTA_REVIEW.md`, and `ballcv.py` for the motion-based
  candidates already in the main pipeline).
* **Track ids fragment** when players cross or leave the frame (IoU tracker). ByteTrack or Re-ID would help.
* **Referee / goalkeeper with a kit between the two team kits** in colour can join a team.
* Better masks: SAM-2 prompted with the detection box, if a GPU is available.
* 2D mini-map: project the foot points with the homography from the main pipeline.

## Debugging

`debug_simple.py` runs every step on **one moment** of a video and saves an image per step
(`01_frame`, `02_field_mask`, `03_detections`, `04_masks`, `05_teams`, `06_ball_zoom`, `07_overlay`) in
`outputs/simple/debug/<video>_<time>/`, printing what each step found. Its header is the map of the main functions
in the order the real run calls them.

```bash
python debug_simple.py data/videos/J3_MansoresSanguedo_1080p.mp4 --time 24:40 --roi-top 0.2
```

Tests: `python -m pytest tests/test_simple.py -q` (synthetic images, a few seconds).
