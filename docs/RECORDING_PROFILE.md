# Recording profile — how the club matches are filmed, and what it changes

Reference recording: `data/videos/2026-10-18_UD_Mansores_vs_Canedo_FC_J2.mp4`
(UD Mansores vs Canedo FC, jornada 2, CFC TV stream from Canedo's ground).

## 1 · What the footage looks like

| Property | Value | Consequence for the pipeline |
|---|---|---|
| Cameras | **one**, near side, ~at the halfway line, ≈ 2.5 m behind the near touchline, ≈ 4 m high | low viewing angle (≈ 6–7° onto the centre circle): strong foreshortening, players overlap in depth |
| Camera motion | operator **pans ±60°, tilts and zooms** (focal ≈ 500–1100 px) to follow the ball | the image ↔ pitch mapping changes every frame → new **PTZ camera tracker** (`football_eval/ptz.py`) |
| Field of view | usually ⅓–½ of the pitch; the far side is always visible, the near touchline only at the extremes | players outside the view are simply missing → team-level stats only for the visible part of play |
| Resolution | **640 × 360**, 30 fps, H.264 ~320 kb/s | players 8–50 px tall, the **ball 2–5 px** → COCO detectors do not see it; pitch lines are faint |
| Duration | 2 h 03 min: warm-up, line-up, two halves, half-time | periods set in the config (`match.halves`): kick-offs at 7:10 and 71:02, ends ≈ 55:15 and ≈ 122:50 |
| Overlays | club crest (top-left), "CFC TV" (bottom-right); **no score bug / clock** | overlays masked (`video.ignore_regions`); the scoreboard-based goal check is not available |
| Audio | track present but **silent** | no referee-whistle detection possible |
| Light | sunny afternoon; **tree shadows** on the near half, growing until the 2nd half is mostly in shade | line detector uses brightness *and* low chroma (rejects sunlit strips); team colours must cope with shade |
| Pitch | artificial turf, measured **≈ 105.4 × 61.6 m** (not 105 × 68) | pitch size is estimated together with the camera; using 68 m would push every position off |
| Kits (this match) | yellow shirts / blue shorts vs green-black; goalkeepers pink/red; referee light blue | colour clustering separates them well in sunlight |
| Other | spectator leaning into the frame near the end, a short odd view at 122:12 | tracker holds/relocalises; frames without pitch evidence are skipped |

## 2 · What changed in the method

1. **PTZ camera model** (`football_eval/ptz.py`): the camera centre `C`, roll and the pitch size are fixed per
   venue; each frame only needs pan, tilt and focal length.
   * *Prediction*: rotation + zoom between consecutive frames from ORB features on the whole background
     (stands, trees, houses — valid because the camera rotates about a fixed point), players masked.
   * *Correction*: guarded, coarse-to-fine chamfer alignment of the pitch template to (a) white line pixels
     (shadow-robust detector) and (b) the far edge of the grass, which is visible in every frame.
   * *Recovery*: local re-localisation when the alignment error stays high.
   * *Start anywhere*: the camera is tracked from the calibration anchor to the requested start time.
2. **Venue calibration** (`scripts/calibrate_ptz.py`, once per filming spot): a few clicked landmarks
   (halfway line / centre circle, goal-post bases) + tracking + bundle adjustment of `C`, roll, grass margins,
   **pitch length and width** and keyframe poses. Canedo's result: `configs/camera_canedo_ptz.json`.
3. **Low-resolution detection settings**: YOLOX on two 360-px tiles (each upscaled to the 640-px network input).
4. **Ball**: motion-compensated difference blobs (`football_eval/ballcv.py`) instead of the detector's ball class.
5. **Match config**: `configs/canedo_cfctv.yaml` (overlays, halves, pitch size, PTZ, ball detector).
6. **Debug video** (`scripts/render_debug_video.py`): boxes by team + projected pitch lines + minimap, to check
   any segment by eye.

## 3 · Where it stands (10-minute development segment, video minutes 15–25)

| Check | Result |
|---|---|
| Venue calibration | camera ≈ 2.6 m behind the near touchline at the halfway line, ≈ 3.9 m high; pitch ≈ 105.4 × 61.6 m |
| Camera pose usable (rolling median line error ≤ 6 px) | **63 % of frames** — locked during normal play; lost spells after very fast pans to the corners or strong zooms, recovered by re-localisation / global search |
| Line alignment when checked | median 2.95 px, 51 % of checks < 3 px |
| Player detections | ~13 people per frame (5–7 per team visible); 11 % outside the pitch (bench, crowd) filtered out |
| Team assignment | yellow (Canedo) vs green-black (Mansores) separated; colours mapped to names via `teams.kit_colors` |
| Positions | 99 % of the kept player positions fall inside the measured pitch |
| Ball | a candidate in 37 % of frames, many false (moving feet) → possession / passes / shots are **not reliable yet** |
| Speed | ~0.65 s per analysed frame on 2 CPU cores (detection ≈ 70 %); a full match at 5 fps ≈ 5–6 h here, less on a recent Mac |

Positions-based outputs (team shape, block depth/width, territory, distance covered while in view) are usable for
the frames with a good camera pose; ball-based statistics need a better ball detector or higher resolution.

## 4 · Recommendations for future recordings (biggest wins first)

1. **Download / record at 1280 × 720 or higher.** Streams are often available at 720p/1080p; 640 × 360 is the
   main limit for the ball and for distant players. Everything else in the pipeline works unchanged.
2. **Keep the same filming spot at home games** — the venue calibration is then re-used as is (only a new anchor
   is needed, `scripts/calibrate_ptz.py --click`). Away grounds need one calibration each (~5 min of clicking).
3. **Zoom out a little more** when play is near the far touchline or a goal: the far edge of the grass and at
   least one marked line in view keep the camera tracker locked.
4. **Keep a score/clock overlay if the streaming tool allows it** — it gives exact goal times for free
   (`scoreboard.enabled`).
5. **Record audio** — the referee's whistle is a strong, cheap signal for stoppages, fouls and goals.
