# Manual validation: what you check by hand, and why

The pipeline is fully automatic, but on a single 640 × 360 club stream several inputs are guessed badly, and
a few wrong guesses quietly change every number in the report. This document does two things:

1. It **evaluates each factor** that a person could provide or check (pitch lines and goals, shirt numbers,
   pitch dimensions, kits and so on). Each one is judged on how reliable the automatic estimate is on the
   Canedo recording, what a mistake costs downstream, and how much of your time a manual input takes.
2. It describes the **manual, explainable workflow** that came out of that evaluation:
   a setup UI before processing, a review UI after it, and a provenance and "why" trail in every output.

The rule used to decide is simple. A factor is made manual when **(a)** the automatic estimate is unreliable on
this footage, **(b)** an error spreads to many statistics or fails without any warning, and **(c)** a person can
give it in seconds or minutes. If a factor needs thousands of labels (every ball position, every player box,
every tracklet's identity), it is *not* made manual. Those labels are only worth collecting as an evaluation set.

---

## 1 · Evaluation: factor by factor

Measured on `2026-10-18_UD_Mansores_vs_Canedo_FC_J2.mp4`, mostly on the 10-minute development segment
(video 15:00–25:00) whose outputs ship in `outputs/canedo_segment/`.

| # | Factor | Automatic today (measured) | Cost of an error | Manual cost | Verdict |
|---|---|---|---|---|---|
| 1 | **Pitch dimensions** | the PTZ venue calibration *estimates* L × W; the width came out between **52 and 62 m** across runs (far touchline barely visible) | every metre-based number: positions, distances, final third, penalty-area rules, xG. A **3 m lateral error** moves xG by **−24 / +27 %** for a shot 11 m out and 8 m wide (0.092 → 0.070 / 0.117), and by **−37 / +45 %** for a close, angled shot (0.183 → 0.116 / 0.265) | 30 s (club / federation page, or the satellite-map distance tool) | **Manual — required** |
| 2 | **Overlays** (logos, score bug) | none: pixel boxes typed into the YAML | false "people" and false "lines" in the logo; a score bug, if present, gives exact goal times | 30 s, drag rectangles | **Manual** |
| 3 | **Periods** (kick-off and final whistle of each half) | inferred from the moment the teams swap sides. The file holds 7 min of warm-up, a 16-min half-time and an end tail, and the side swap cannot find where the warm-up stops | the clock is wrong by minutes, and warm-up and half-time count as play | 1–2 min of scrubbing | **Manual — required** |
| 4 | **Attacking direction** (1st half) | inferred from each team's mean x. On the segment the gap was only **3–5 m**, and it pointed the wrong way in **22 %** of the 30-s bins | **every** directional stat is mirrored: shots, xG, final-third entries, PPDA, field tilt | 10 s | **Manual — required** |
| 5 | **Team kits** (who is who) | 59 tracklets labelled by hand from crops: unsupervised colour clusters came out **swapped (20 % correct)**, 80 % once swapped back. Typed BGR kit colours: 90 % on outfield players but **referee 0/5** (always put in a team). **Seeded by clicks: 81–88 % outfield (spread across runs), referee 5/5, a swap is impossible** | a swap gives *every* team statistic to the wrong side. A referee counted as a player adds fake passes and pressing | 1–2 min: click 3+ players per kit, in sun and shade | **Manual — required** |
| 6 | **Goalkeepers** | heuristic: an "other" track that spends 70 % of its time in a box. With only the Canedo goalkeeper clicked, **1 of 5** goalkeeper tracklets was recognised; the rest wore the other keeper's kit | xG freeze frame (goalkeeper position), saves vs blocks | 20 s, one click per goalkeeper | **Manual** |
| 7 | **Pitch lines and goals** (calibration landmarks) | already manual, but in a bare OpenCV window with no feedback | **silent**: players simply land in the wrong place | 3–5 min. The UI now fits and draws the pitch on the frame, shows the error of each clicked point, and solves a PTZ pose in about 2 s | **Manual (improved)** |
| 8 | **Calibration check** after processing | the camera grades itself with a line-alignment error, and 63 % of frames pass the gate | unknown until somebody looks | 2–3 min: about 16 sampled frames, good / bad | **Manual review → a measurement** |
| 9 | **Shirt numbers per player** | not possible: 2,102 tracklets in 10 minutes (about 19k per match), **median 1.8 s**, 95 % shorter than 15 s; the 200 longest cover only **44 %** of player-time. Players are a **median 25 px tall**, so a digit is 3–5 px and unreadable by a person or by OCR | only per-tracklet player stats | hours, and at this resolution impossible | **Not manual per tracklet.** Numbers are typed only on **key events** (goals, shots, cards), and a roster maps them to names |
| 10 | **Match sheet** (score, goal minutes, cards, substitutions) | goals are only *inferred* (no score bug, silent audio). Cards and substitutions cannot be seen | false or missed goals; no scorers, no discipline | 1–2 min from the official match sheet | **Manual — required** |
| 11 | **Key events** (goals, penalties, shots) | rules on a weak ball signal (a ball candidate in 37 % of frames, many of them false). In the pilot below, the **only shot** detected in the segment was a **false positive** | xG and the timeline are built from these few events | 5–10 min: watch a 7-s clip each, confirm / reject / flip the team | **Manual review** |
| 12 | Ball position per frame | motion blobs | possession, passes | thousands of frames | **Not manual** (annotate about 200 frames later as a test set) |
| 13 | Player boxes / tracking | detector + tracker | — | prohibitive | **Not manual** |
| 14 | Camera position | multi-start fit + bundle adjustment, stable | — | — | **Not needed** |

### Pilot review of the segment (by Claude, not by you)

To check that the review produces useful numbers, Claude went through the 16 sampled frames and the one key event
of `outputs/canedo_segment` in the review UI. This was a pilot on 640 × 360 screenshots, so it is no substitute
for your own review. It is stored in `outputs/canedo_segment/review_pilot_claude.json`:

| Check | Result |
|---|---|
| pitch lines right on frames the stats **used** | 75 % (6/8; 95 % CI 41–93 %), plus 4 frames where no lines were visible to judge |
| frames the quality gate **discarded** that were actually fine | 1 of 3 |
| frames where **every** box had the right team (old unsupervised colour labels) | 20 % (3/15). With ~10 players in view, 90 % per-player accuracy gives 0.9¹⁰ ≈ 35 % |
| shots that really happened | 0/1: the 18:02 "shot from 28 m" happens while play is at the centre circle |

The intervals are wide, which is expected for a 16-frame check. Their value is that the report now says
*"positions right on 6 of 8 checked frames"* instead of implying that every position is right.

---

## 2 · The workflow

**In the launcher** (`python scripts/launcher.py`, or `start.command`): choose the recording, then *Analyse →
Open setup*, *Process the video*, *Open review* and *Apply review*. The setup and the review open in the
**Manual decisions** tab. They are the same screens and files as the commands below. While step 2 runs, the
**Match screen** tab shows the frame being analysed with its overlays. Afterwards, it replays the whole match with
every overlay, so a wrong camera pose, team swap or missed shot is visible before you review.

```bash
# 0) once per match (~10 min): validate what the camera cannot know
python scripts/match_ui.py setup --video data/videos/match.mp4 --setup configs/match_setup_<match>.json \
       --config configs/<venue>.yaml
#    1 match & pitch size · 2 periods & direction · 3 overlays · 4 kits · 5 pitch landmarks
#    6 match sheet (score, goals, cards, rosters) · 7 checklist (+ the exact commands to run next)

# 0b) panning camera at a new ground: solve the venue camera from your clicks and the REAL pitch size
python scripts/calibrate_ptz.py --video data/videos/match.mp4 --setup configs/match_setup_<match>.json \
       --config configs/<venue>.yaml --out configs/camera_<venue>_ptz.json

# 1) process (hours on a laptop CPU); the setup checklist is printed, and blocking items stop the run
python run_analysis.py --video data/videos/match.mp4 --config configs/<venue>.yaml \
       --setup configs/match_setup_<match>.json --out outputs/<match>

# 2) review (~15 min): 1 frame checks · 2 camera repair · 3 key events (+ shot / keeper location clicks)
#    4 missing events · 5 exclude stretches · 6 summary
python scripts/match_ui.py review --run outputs/<match> --setup configs/match_setup_<match>.json \
       --config configs/<venue>.yaml

# 3) apply: --repair first re-processes only the camera stretches you repaired (minutes), then the analytics
#    apply review.json, the excluded stretches and the match sheet (seconds)
python run_analysis.py --tracks outputs/<match> --repair --config configs/<venue>.yaml \
       --setup configs/match_setup_<match>.json --out outputs/<match>
```

The UI runs locally (Python standard library + OpenCV, no new dependencies) at <http://127.0.0.1:8765>.
Everything is saved to JSON as you go. Shortcuts: <kbd>space</kbd> play, <kbd>←</kbd>/<kbd>→</kbd> ±0.2 s,
<kbd>shift</kbd>+arrows ±1 s; frame checks <kbd>g</kbd>/<kbd>b</kbd>/<kbd>n</kbd> (lines) and
<kbd>1</kbd>/<kbd>2</kbd> (teams); events <kbd>c</kbd> confirm, <kbd>x</kbd> reject, <kbd>t</kbd> other team.

### What each manual input changes in the code

| Input (`match_setup.json`) | Used by | Effect |
|---|---|---|
| `pitch` (L, W, source) | `match_setup.apply_to_config` → `pitch.length/width`; `source: manual` sets `pitch.fixed_size` | positions, all rules. `calibrate_ptz.py --setup` solves the camera with that size **fixed** (`bundle_adjust(fit_pitch=False)`) |
| `overlays`, `scoreboard_roi` | `video.ignore_regions`, `scoreboard.*` | detections and line evidence masked; score-bug goal check |
| `periods`, `attacks_right_first_half` | `match.halves`, `match.home_attacks_right_first_half` | `context.infer_periods` stops guessing: clock, attack frame |
| `kits` (clicked boxes) | `teams.kit_samples` → `teams.KitClassifier` (perception) | seeded k-means for the two teams (centres start at your samples, identity checked after fitting, so they cannot swap), nearest-sample check for goalkeepers / referee, ties → *unknown*. Writes `team_assignment.csv` (votes per track) |
| `calibration.landmark_frames` | static: `pitch.calibration_points`; flow: `pitch.flow_anchor_points`; PTZ: `calibrate_ptz.py --setup` | the homography / camera model |
| `calibration.anchors` (PTZ poses you accepted in the UI) | `pitch.extra_anchors` → `PTZProvider` start pose | the camera tracker can start or recover near any validated frame |
| `final_score`, `known_events`, `teams[].roster` | `review.reconcile_match_sheet` (analytics) | a goal within ±120 s of a sheet goal of the same team is confirmed (and gets the scorer); missing sheet goals are added; if the sheet lists every goal, unmatched automatic goals are rejected. Cards and substitutions are added as events; `yellow_cards` / `red_cards` appear in `team_stats` |
| `review.json → repairs` (landmarks clicked on a frame of a lost camera stretch → pose) | `perception.repair_run` (`run_analysis.py --repair`) | the PTZ tracker restarts from your pose at the first analysed frame after it and runs to the end of the lost stretch + 5 s (max 3 min, `review.repair_window`); those frames are replaced in tracks / ball / camera log, the ball is re-cleaned. Canedo test: a 40-s stretch went from **14 % → 100 %** usable frames, lines visibly on the painted ones |
| `review.json → events[].shot_xy / gk_xy` (clicks on the frame of the strike) | `review.clicked_xg` | the clicked spot replaces the tracked one (2-5 px ball); xG recomputed with the **full** model, opponents from tracking at that frame, the keeper from your click (`location = clicked(+gk)` in `events.csv`). Also gives an xG to shots you add by hand |
| `exclusions` (stretches marked in the review, saved in `match_setup.json`) | `review.apply_exclusions` / `drop_excluded` | positions and ball blanked and the frames treated like "camera lost" (possession unknown, not dead ball → no fake stoppages, fouls or set pieces); automatic events inside are dropped; time reported in *Data quality* |
| `review.json` (frame checks, verdicts, added events) | `review.apply_review` + `review.validation_summary` | rejected events → `events_rejected.csv` (with the reason); flipped team → shot xG recomputed (basic model); shirt numbers → `player_stats_by_number.csv`; measured reliability → report + `validation_summary.csv` |

---

### Second round of manual inputs (camera repair, shot locations, excluded stretches)

| Input | Weak point it fixes | Effort | Effect measured / expected |
|---|---|---|---|
| **Camera repair**: on a lost stretch, click 2–6 landmarks, check the overlay, accept | 31–37 % of frames had no usable camera pose: every position-based number ignored them | ~1 min per stretch; the longest few matter most | Canedo test: a 40-s stretch from 14 % to 100 % usable; overall 63 % → 69 % after one repair |
| **Shot and keeper location**: two clicks on the frame of the strike | xG is built on the shot location, which came from a 2–5 px ball | ~5 s per shot | xG becomes Medium confidence once ≥ 80 % of shots are located (see `docs/REPORT_GUIDE.md`) |
| **Excluded stretches**: start / end + reason | replays, crowd shots, long injury breaks read as stoppages or play | ~1–2 min per match | no fake set pieces / fouls / ball-in-play time from them |

### Third round: making the manual inputs reach the processing (J3, UD Mansores vs Sanguedo, 1080p)

Measured on `J3_MansoresSanguedo_1080p.mp4` (notebook `08_speed_and_kits_eval.ipynb`).

| Weak point found | Measured | Fix | Effect |
|---|---|---|---|
| **Processing speed**: 1.8 analysed frames/s on the Mac → ~8.5 h for a full match | detector = 95 % of the time; a 1080p frame was cut into **6** tiles (960 px, stride 720: two rows only 120 px apart) and run through `cv2.dnn` | tiles laid over the calibrated pitch only, fewest tiles of ≤ 1024 px with 96 px overlap (**2** per frame), people cut by a seam joined; YOLOX through **onnxruntime** (CoreML on the Mac when faster); decoding in a background thread; only the periods are processed; speed presets | 14 frames, 2 vCPUs: **2.0 s → 0.36 s per frame (5.6×)**, 94–96 % of the previous boxes found again, same ball-candidate rate; *quality* (800-px tiles + pitch ROI) is still **3.9×** faster and finds more players (15.5 vs 15.1 per frame) and ball candidates (0.43 vs 0.29) |
| **Clicked kits not used by the live boxes** | the live view coloured each box by the nearest *stored* descriptor, with drab shirt colours (#525441 vs #6f787b) that look like "unknown" grey | one classifier (`KitClassifier`) for the live view and the final assignment; each **track** shows its running vote; the tracks you clicked are seeded with your label (`*`) on their frame; fixed palette + legend + team short names | J3 segment: 15/15 clicked players matched to their track on the clicked frame; labels stable across frames |
| **Dragged kit boxes are loose** | IoU with the detected player: median **0.50** (quartiles 0.42–0.66): the "torso" crop was mostly grass | boxes are snapped to the detector's box (setup UI and pipeline) | 53 clicked samples on 5 moments; leave-one-out (outfield): **25/46 → 35/45** (drawn → snapped) |
| **Green kit masked as grass** | the fixed "grass" hue band (30–90) also deleted the UD Mansores green shirts | the pitch colour is measured on every frame and only *that* colour is removed; lightness left out of the descriptor (sun / shade) | leave-one-out: **35/45 → 44/45** outfield, **49/50** with goalkeepers and referee; leaving out a whole moment (new light): **28/46 → 40/45** outfield. The old seeded k-means also drifted further than the gap between the two kits (J3 run: drift 1.39); the nearest-samples rule cannot drift |
| **Flow anchors ignored** | landmark frames on odd frame numbers were never analysed (24 fps → every 2nd frame), so a *flow* run had **no calibration at all** | an anchor is used on the first analysed frame within one step | J3 test window: calibration on 97 % of frames instead of 0 % |
| **Near touchline out of view** | J3: the picture often ends before the camera-side touchline | setup: say so in *Match & pitch*, place near-side landmarks in a grey margin outside the picture; processing: `vis_near_y` per frame, `out_of_view_near` stoppages → near-side throw-ins, feet of cut-off players estimated, data-quality line and lower confidence for throw-ins | see rulebook §3.1 |

## 3 · Explainability: every number can answer "why?" and "who said so?"

* **`events.csv → why`**: one sentence per event naming the rule, the measured values and the thresholds
  (config keys), worded like the rulebook. For example: *"Canedo FC (track 1242) released the ball at 18.6 m/s
  (>= events.shot_min_speed = 11.0) towards the goal mouth (± 3.0 m) from 28.2 m (<= 35.0 m). xG 0.02: open
  play, 0 outfield defenders inside the shot-posts triangle. Outcome 'blocked' (an opponent touched it within 0.8 s
  and 12.0 m)."* The report lists the *why* of every goal, shot and card.
* **`events.csv → validation`**: `confirmed`, `corrected`, `match_sheet_confirmed`, `manual` (added in the review),
  `match_sheet`, or `unreviewed`. The timeline marks them as ✓, ✓ sheet, (added by hand) and (match sheet).
* **`provenance.json`** and the top of `report.html`: a ledger of every input (pitch, periods, direction, kits,
  calibration, match sheet, the camera-quality share, tracklets, events kept vs detected, review) with its
  **source**:
  * `manual`: you entered or validated it.
  * `measured`: checked against your review.
  * `auto`: inferred.
  * `default`: never provided.
  * `assistant`: pre-filled by someone else and not yet confirmed. The checklist keeps these amber until you
    press *"I checked this"* in the UI.
* **`key_metrics.csv`** and the top of the report: every statistic with its **value** rank (★ headline) and a
  **confidence** for this match (High / Medium / Low / Experimental) with the reason. See `docs/REPORT_GUIDE.md`.
* **`validation_summary.csv`** and *Measured reliability* in the report: what your review measured, each figure
  with its 95 % Wilson interval.
* **`team_assignment.csv`**: for every track, the crop votes per class, the winner, its share and the decision
  (`majority`, `plurality (ambiguous)`, `tie -> unknown`).

---

## 4 · Canedo FC vs UD Mansores (J2): what is pre-filled and what is left for you

`configs/match_setup_canedo_J2.json` was pre-filled from the earlier work (configs and landmarks) plus about 15 kit
clicks made by Claude. Every pre-filled section is marked `assistant`. To finish it:

```bash
python scripts/match_ui.py setup --video data/videos/2026-10-18_UD_Mansores_vs_Canedo_FC_J2.mp4 \
       --setup configs/match_setup_canedo_J2.json --config configs/canedo_cfctv.yaml
```

1. **Pitch size**: enter the real size of Campo das Valadas (it is 105.4 × 61.6 *estimated*). Then re-solve the
   camera with it fixed: `python scripts/calibrate_ptz.py --video ... --setup configs/match_setup_canedo_J2.json
   --config configs/canedo_cfctv.yaml --out configs/camera_canedo_ptz.json`.
2. **Periods** (7:10–55:15, 71:02–122:50) and **direction** (Canedo attacks right in the 1st half): check both
   and confirm them.
3. **Kits**: confirm the samples, add **UD Mansores' goalkeeper** (no sample yet), and check that the pink/red
   keeper at the left goal at 43:20 really is Canedo's.
4. **Match sheet**: final score, goal minutes and teams, scorers' numbers, cards; add rosters if you want names.
5. After the full-match run: **review** the sampled frames and every goal and shot.
