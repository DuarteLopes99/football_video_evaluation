# Statistics rulebook

How every event and statistic is computed from one camera, which thresholds drive it (config key in
`configs/*.yaml` → `football_eval/config.py`) and how far you can trust it. Code references point to the
module that implements the rule. Version 0.1 — rules are deliberately simple and measurable
(`notebooks/02_event_detection_eval.ipynb`).

---

## 0 · Conventions

| Item | Definition |
|---|---|
| Pitch frame | metres, origin at the top-left corner of the minimap; `x ∈ [0, L]` along the touchline, `y ∈ [0, W]` (far touchline at `y = 0`). Default 105 × 68 (`pitch.length`, `pitch.width`); **club pitches differ** — the PTZ venue calibration measures them (Canedo ≈ 105.4 × 61.6) and overrides the config. Every rule below that says 105/68 means L/W. |
| Attack frame | for team *T* at time *t*: coordinates rotated so that *T* attacks towards `x' = 105` (`MatchContext.to_attack_frame`). All "final third", "progressive", PPDA and shot rules use it. |
| Time | `t` = seconds in the video file; `clock` = football clock (`23'`, `45+2'`; second half starts at 45'). |
| Periods | given in `match.halves`, else inferred as the moment the two teams **swap sides** (sign change of mean-x difference, 30-s bins, ≥ 80 % agreement). |
| Attacking direction | `match.home_attacks_right_first_half`, else the team whose players average the lower x in a period attacks right. |
| Teams | `0` and `1` = Team A / Team B of `match_setup.json` when kit samples were clicked (identity fixed); otherwise colour clusters whose order must be checked after the first run. `-1` = referee / staff / unknown. |
| Analysis rate | `video.analysis_fps` (default 10 fps). All durations below are in seconds, independent of the rate. |

### Reliability levels

| Level | Meaning | Typical use |
|---|---|---|
| **A** — reliable | robust with decent tracking; synthetic F1 ≥ 0.9 | report as is |
| **B** — approximate | right order of magnitude, biased by occlusions / ID switches | compare teams, trends within a match |
| **C** — proxy | inferred from a correlated signal (e.g. fouls from free-kick restarts) | indicative only, say "proxy" |
| **D** — not available | cannot be observed from one wide camera today | do not report |

---

## 1 · From perception to the game state

| Step | Rule | Keys |
|---|---|---|
| Ground point | bottom-centre of each person box (feet) and of the ball box, projected with the homography | `pitch.*` |
| Off-pitch people | further than `offpitch_margin_m` outside the lines → `cls = outside` (bench, staff, ball boys) and ignored | `pitch.offpitch_margin_m` |
| Team per track (kit clicks) | **with `match_setup.json` kits**: colour descriptor per crop every 5th frame → k-means with the two centres *seeded* at the clicked samples of each team (samples weighted ×20, identity re-checked after fitting) → goalkeeper / referee if a crop is closer to one of their clicked samples than 0.8 × the team distance and within the 75th-percentile in-team spread → vote per track; a tie between the top classes = unknown. Votes saved in `team_assignment.csv` | `teams.kit_samples`, `teams.KitClassifier` |
| Team per track (no clicks) | colour descriptor per crop every 5th frame → KMeans(2) fitted on the first `fit_frames` → majority vote per track (≥ 60 %); cluster order arbitrary unless `teams.kit_colors` is set | `teams.*` |
| Referee / other | crops farther than `quantile × factor` from both team centroids | `teams.outlier_quantile`, `teams.outlier_factor` |
| Goalkeeper | a track labelled *other* that spends ≥ 70 % of its frames within 18 m of a goal line and 22 m of the centre line → goalkeeper of the team defending that goal (per period) | `teams.assign_goalkeepers` |
| Ball track | per frame keep the candidate consistent with the previous position (`≤ ball.max_speed_ms`), interpolate gaps `≤ ball.max_gap_s`, Savitzky–Golay smoothing, velocity by finite differences | `ball.*` |
| Player speed | rolling mean of 5 positions, speed from consecutive samples; values > 11 m/s or across gaps > 1 s discarded | `context.player_speeds` |

---

## 2 · Ball control and possession (`possession.py`)

**Control (per frame).** Player *P* controls the ball if
1. *P* is the closest player to the ball,
2. distance ≤ `possession.control_radius_m` (1.5 m — larger than tracking-data practice (0.5–1 m) because of single-camera position noise),
3. ball speed ≤ `possession.max_ball_speed_control` (9 m/s — a faster ball is in flight).

**Duel.** At least one player of each team within `possession.duel_radius_m` of the ball.

**Touch.** Maximal run of frames controlled by the same track (single-frame gaps bridged); runs shorter than
`possession.min_control_s` are dropped. Each touch has start/end time and ball position.

**Team possession (per frame).** The team of the last controller, switched only after the other team controls
the ball for `possession.team_switch_s` (0.6 s hysteresis; the switch is back-dated to the first control).
Undefined (–1) during dead-ball periods.

| Stat | Definition | Level |
|---|---|---|
| **Possession %** | live-ball time with team possession = T ÷ live-ball time with any team possession | **A** |
| **Ball in play (min)** | analysed time not inside a stoppage | **B** (stoppage edges ±2 s) |
| **Field tilt %** | possession frames of T with the ball in T's attacking third (`x' ≥ events.final_third_x`) ÷ same for both teams | **B** |

---

## 3 · Events (`events.py`)

### 3.1 Dead ball and stoppages
A frame is a **stoppage candidate** if any of:
* **Out** — the ball is detected beyond a line by `events.out_margin_m`;
* **Unseen & still** — the ball is missing for ≥ `events.dead_ball_min_s` and the mean player speed is
  < `events.still_speed_ms` for most of that time, or the ball disappeared right after being out;
* **At rest & still** — the ball position spread over 2 s is < `events.static_ball_spread_m` and players are still,
  for ≥ `events.static_still_min_s` (foul, injury);
* **At rest** — the same without the player condition for ≥ `events.static_ball_min_s` (set-piece preparation).

Candidates closer than 1.5 s are merged; merged intervals shorter than `dead_ball_min_s` are discarded (a lofted
ball projected outside the lines for a moment is *not* a stoppage). Each stoppage stores its reason
(`out` / `unseen` / `static`), exit location and line (`touchline`, `goal_line_left/right`).

**Near touchline out of view.** Every analysed frame records `vis_near_y` (frames.csv / ball.csv): the pitch y
seen at the bottom edge of the picture, i.e. where the camera's view ends on the camera side (≥ pitch width = the
near touchline is in the picture). When a stoppage is mostly *unseen* and the ball was last seen within
`events.near_edge_band_m` (3 m) of that edge while the near touchline is out of view, the reason becomes
`out_of_view_near` and the line `touchline_near`: the ball very probably went out on the camera side. Without this
rule such stoppages looked like fouls / injuries. People whose box is cut by the bottom edge (`tracks.edge = 1`) get
their feet estimated below the frame (box width / 0.4) instead of being placed on the edge.

### 3.2 Restarts (set pieces)
At the end of each stoppage: **placement** = median ball position in the last second of the stoppage (else first
seconds after); **taken** = first frame the ball moves > 1.5 m from the spot; **taker** = player whose touch starts
at the spot (else the nearest player within 3 m). Classification, in order:

| Restart | Rule | Keys |
|---|---|---|
| **Kick-off** | ball within `kickoff_radius_m` of the centre spot **and** ≥ `kickoff_own_half_share` (90 %) of each team in its own half at the end of the stoppage | `events.kickoff_*` |
| **Throw-in** | the ball left over the touchline and is put back within `throwin_band_m` of it | `events.throwin_band_m` |
| **Throw-in** (near side, not seen) | stoppage `out_of_view_near` and the ball comes back within `near_edge_band_m` + 1 m of the bottom edge of the picture | `events.near_edge_band_m` |
| **Corner** | within `corner_radius_m` of a corner flag (and the ball did not leave over the touchline) | `events.corner_radius_m` |
| **Penalty** | within `penalty_radius_m` of a penalty spot and ≤ 1 player (besides taker and the goalkeeper on his line) inside the area | `events.penalty_radius_m` |
| **Goal kick** | inside a goal area (+1 m tolerance) | — |
| **Throw-in** (fallback) | within `throwin_band_m` of a touchline | — |
| **Free kick** | anything else | — |

The **first kick-off of each period** is recognised when fewer than 3 touches happened in the period before it.

| Stat | Definition | Level |
|---|---|---|
| Corners, throw-ins, goal kicks, kick-offs | restarts of that type taken by T | **A** |
| Free kicks won | free-kick restarts taken by T | **B** |
| Penalties won | penalty restarts taken by T | **B** (rare event; check the clip) |

### 3.3 Passes, interceptions, recoveries
For consecutive touches *a → b* (no stoppage in between, gap ≤ `events.max_pass_s`, ball travel
≥ `events.min_pass_length_m`):

| Event | Rule |
|---|---|
| **Pass — complete** | *b* is a different player of the same team |
| **Pass — intercepted** + **Interception** (for *b*'s team) | *b* is an opponent and the ball left *a* faster than 4 m/s |
| **Ball recovery** (tackle / dispossession) | *b* is an opponent, slow ball, and *b* keeps control ≥ `events.recovery_min_s`; also when travel < `min_pass_length_m` (contest at one spot) |
| **Pass — out** | a stoppage of reason `out` follows *a*'s release and the ball left faster than 3 m/s |
| Carry (no event) | same player regains the ball |

Pass tags (attack frame): **progressive** (x' gain ≥ `events.progressive_pass_m`), **final_third_entry**
(starts before `final_third_x`, ends after), **box_entry** (ends inside the penalty area), **cross** (box entry
from the wide zone `x' ≥ 70`, `|y'−34| > 20.16`), **long** (≥ 30 m).

| Stat | Definition | Level |
|---|---|---|
| Passes / completed / accuracy % | counts of pass events and `complete / all` | **A** (team), **B** (one-touch passes merge) |
| Progressive passes, final-third entries, box entries | completed passes with that tag | **B** |
| Crosses, long passes | passes with that tag (any outcome) | **B** (aerial balls are mis-projected) |
| Interceptions | interception events | **A** |
| Ball recoveries | ball_recovery events | **B** |

### 3.4 Shots
At the release of touch *a* by team T, using the fastest ball sample in the next 0.6 s:
1. ball speed ≥ `events.shot_min_speed` (11 m/s),
2. moving towards the opponent goal (`v'x > 0`),
3. release point within `events.shot_max_dist` (35 m) of the goal centre,
4. the ball's straight line crosses the goal line within the posts ± `events.shot_goal_tolerance` (3 m),
5. *not* received within 2.5 s by a team-mate farther than 6 m from the goal line (that was a pass/cross).

**Outcome** (first rule that applies):

| Outcome | Rule |
|---|---|
| **goal** | the tracked ball crosses the goal line between the posts (+0.5 m), **or** the next restart is a kick-off by the opponents |
| **saved** | next controller is the opposing goalkeeper (or an opponent inside the goal area +1.5 m) |
| **blocked** | next controller is an outfield opponent within `block_window_s` and `block_max_dist` of the shot |
| **off_target** | ball crosses the goal line outside the posts, or next restart is a goal kick, or an opponent collects it later |
| **blocked_or_saved_out** | next restart is a corner |
| **unresolved** | none of the above |

| Stat | Definition | Level |
|---|---|---|
| Shots | shot events | **A**/**B** (synthetic recall 0.7–0.9; headers and deflections are missed) |
| Shots on target | outcome ∈ {goal, saved} | **B** |
| Shots blocked | outcome = blocked | **B** |

### 3.5 Goals — three independent sources
1. **Shot rule** — a shot with outcome *goal* (`source = shot_rule`).
2. **Kick-off rule** — a kick-off that is not a period start, after a stoppage of ≥ `events.goal_min_stoppage_s`
   (20 s) and ≤ `events.goal_confirm_window_s`, with no detected goal in the 20 s before: goal for the team
   that did **not** kick off (`source = kickoff_rule`).
3. **Scoreboard** — when `scoreboard.enabled`, a persistent change of the score-bug pixels confirms a goal within
   the previous 180 s (`detail |scoreboard_confirmed`) or creates one (`source = scoreboard`, team unknown).

Goals: **A** with the scoreboard, **B** without.

### 3.6 Fouls and offsides (proxies)
Fouls are not visible as such from one wide camera; the referee's decision is visible through the **restart**:
* **foul_proxy** — a free kick or penalty after a stoppage that was **not** caused by the ball going out;
  attributed to the team that did **not** take the free kick.
* **offside_suspected** — instead of a foul when the free kick is taken in the taker's own half, shortly after an
  opponent's forward pass that was received in the taker's half.

| Stat | Level |
|---|---|
| Fouls committed (proxy) | **C** — also counts handballs, dangerous play, drop-ball-like restarts |
| Offsides (suspected) | **C** |
| Yellow / red cards | **D** |

### 3.7 PPDA (pressing)
`PPDA(T) = opponent passes with x'(opponent) < L − events.ppda_zone_x  ÷  (interceptions + ball recoveries + fouls)
of T with x'(T) ≥ events.ppda_zone_x`. Lower = more intense pressing. **B**.

---

## 4 · Expected goals (`xg.py`, `models/xg_model.json`)

`xG = sigmoid(β₀ + Σ βᵢ zᵢ)` with standardised features (attack frame, goal centre `(105, 34)`):

| Feature | Definition |
|---|---|
| `dist`, `log_dist` | distance shot → goal centre (m) |
| `angle` | goal-mouth angle seen from the shot, `atan2(7.32·dx, dx² + dy² − 3.66²)` |
| `n_def_tri` | outfield opponents inside the triangle shot / left post / right post |
| `gk_in_tri` | goalkeeper inside that triangle |
| `gk_dist` | goalkeeper → shot distance (capped at 40 m) |
| `n_press` | opponents within 3 m of the shooter |

* Trained on 8,152 open-play shots (StatsBomb open data, 351 matches). Grouped 5-fold CV: **AUC 0.78, Brier
  0.080** (full); 0.74 / 0.085 (distance + angle only). StatsBomb's own xG on the same shots: 0.81 / 0.076.
* Without a goalkeeper track, the opponent nearest to the goal centre within 12 m is used; without any, the
  goalkeeper is assumed on his line.
* **Penalty** → constant empirical conversion (0.71). **Direct free kick** (shot from a free-kick restart by the
  taker) → model × empirical factor.
* Header / foot is **not** a feature (not observable reliably) → predictions average both.

| Stat | Definition | Level |
|---|---|---|
| xG | sum of shot xG | **B** (inherits shot detection) |
| xG per shot | mean | **B** |
| Big chances | shots with xG ≥ 0.30 | **B** |

---

## 5 · Physical and shape statistics

| Stat | Definition | Level |
|---|---|---|
| Distance covered (tracked) | Σ speed × Δt over a team's outfield tracklets — **only the time players were visible**; scale by `90 / minutes_tracked` for estimates | **B** (fixed wide camera) / **C** (panning) |
| Sprints | runs > 7 m/s (25 km/h) lasting ≥ 1 s | **B** |
| Average speed | mean of valid speed samples | **B** |
| Block depth / width | mean over frames of the 10–90 % percentile range of x' / y' of a team (≥ 6 players visible) | **B** |
| Defensive line height | mean 15th percentile of x' | **B** |
| Team centroid | mean x' | **A** |

---

## 6 · Player statistics (per tracklet)

Until jersey numbers / Re-ID merge tracklets, a real player is split over several `track_id`s. Per tracklet:
minutes tracked, distance, max speed, average position, touches, passes (completed), shots, xG, goals,
interceptions, recoveries. Use for physical profiles and spotting key actors; **do not** publish as final
player stats (**C**).

---

## 7 · Timeline of major events (`timeline.py`)

Included: goals and penalties (priority 1), shots (2), corners, fouls (proxy), suspected offsides (3), period
kick-offs / ends and free kicks in the attacking third (4). Each row has `video_time`, `clock`, team,
description and a clip window `[t − 8 s, t + 5 s]`; `outputs/<match>/cut_highlights.sh` cuts priority ≤ 2 clips
with ffmpeg.

---

## 7c · Major opportunities (`opportunities.py`)

The list of chances you watch first. It is built from the kept events (after your review and the match sheet).
Thresholds: `opportunities.*` in the config.

**Components**

| Component | Rule |
|---|---|
| goal | `goal` event (shot rule, kick-off rule, review, match sheet) |
| shot | `shot` event, with its outcome and xG (§3.4, §4) |
| penalty | `penalty` restart, or `penalty_missed` from the match sheet; chance value = the model's penalty xG |
| box entry | a completed pass tagged `box_entry`, or a touch (carry) that starts outside and ends inside the opponent penalty area, moving the ball ≥ `carry_min_m` (3 m). It includes `include_box_entries` |
| set piece | corners, and free kicks within `set_piece_max_dist` (30 m) of the opponent goal centre; `include_set_pieces` |

**Grouping.** Components of the same team are one opportunity when each follows the previous one by ≤
`merge_window_s` (6 s). A component of the other team in between starts a new one. This turns corner → header → rebound into
one move.

**Category** (strongest component wins):

| Category | Rule | Flag ⚑ |
|---|---|---|
| Goal | contains a goal | yes |
| Penalty | contains a penalty | yes |
| Big chance | best shot xG ≥ `big_chance_xg` (0.30, same as the *big chances* stat) | yes |
| Shot on target | a shot with outcome goal / saved | yes |
| Shot | other shots | yes if best xG ≥ `flag_min_xg` (0.10) |
| Box entry, no shot | the ball reached the box, no shot followed | no |
| Set piece near goal | corner / close free kick, no shot followed | no |

**Numbers per opportunity**

| Column | Definition |
|---|---|
| `chance_xg` | probability that the move produced a goal: `1 − Π(1 − xG_i)` over its shots (two shots in one move are not two independent chances, so they are not summed) |
| `danger` (0–100) | goal = 100; with shots: `100 × chance_xg` (+10 if a shot was on target); box entry: `100 × no_shot_discount (0.5) × xG of a shot from the entry point` (basic distance + angle model); set piece: `100 × corner_xg_prior (0.03)` or the free kick's direct xG. It ranks all moments, including those without a shot |
| `t`, `video_time`, `clock` | first shot / goal of the move (else its first component) |
| `clip_start`, `clip_end` | `[first component − clip_before_s (10 s), last component + clip_after_s (6 s)]` |
| `checked` | a shot / goal of the move was confirmed or corrected in your review or by the match sheet |

**Outputs.** These are `opportunities.csv` / `.json`, the ⚑ section at the top of `report.html` (flagged first, the
rest collapsed, per-team summary, links to the clip and to the moment in the full video), and the *Opportunities over
the match* figure. They also include the `opportunities_flagged` team statistic, graded like the shots, and
`clips/opportunities/*.mp4`. Clips are cut automatically after each run when the video is available
(`auto_clips`; ffmpeg stream copy, else OpenCV). `scripts/cut_opportunities.py --all` cuts every opportunity.

**Reliability.** The list inherits the shot detector's: level **B** on good tracking, and **C** on the 640 × 360 club
streams until the shots are reviewed. Box entries depend on the ball track and are **C**, which is why they are not
flagged. Evaluation: `notebooks/07_opportunities_eval.ipynb`. On synthetic matches, every true goal and 97 % of the
true shots on target fall inside a flagged opportunity, with flag precision 1.0 and ~12 % of the playing time to
watch. Under heavy noise, coverage drops to 83 % / 60 %.

---

## 7b · Low-resolution club streams (640 × 360, panning camera)
* Positions come from the PTZ camera tracker; each frame logs its alignment error (`camera_log.csv`). Frames
  with a median line error above `pitch.ptz_max_err_px` should be treated as unreliable.
* Players outside the current view are missing: distance covered, shape and territory describe **the visible part
  of play** (the camera follows the ball, so this is where the action is).
* The ball is 2–5 px: possession, passes, shots and xG from such recordings are **level C (indicative)** until the
  footage is ≥ 720p or a dedicated ball model is trained.

## 8 · Not available from one wide camera (level D)
Cards, offside lines at the moment of the pass (see the sibling `offiside_code` project for single plays),
headers vs. foot, shot height / placement, pass height, tackles vs. clearances, player names (without jersey OCR).

## 8b · Manual inputs and the review (`match_setup.py`, `review.py`, `explain.py`)

Why each input is manual, with measurements: `docs/MANUAL_VALIDATION.md`.

| Rule | Definition |
|---|---|
| Setup gate | `run_analysis.py --setup` prints the checklist; **blocking** items (no video, no calibration) stop a `--video` run unless `--force` is given. Sections pre-filled by someone else (`provenance.source = assistant`) stay amber until confirmed in the UI |
| Match-sheet minute → video time | referee's clock `m` (or `45+x`, `90+x`) → start of that half + (m − 1 − 45·half + x) min + 30 s |
| Sheet goal ↔ detected goal | same team and \|Δt\| ≤ 120 s → `validation = match_sheet_confirmed` (+ the scorer's number); an unmatched sheet goal is added (`source = match_sheet`); when the sheet lists **every** goal of the final score, unmatched automatic goals are moved to `events_rejected.csv` |
| Review verdicts | `confirm` → `validation = confirmed`; `reject` → removed (a goal derived from a rejected shot goes too) and kept in `events_rejected.csv`; `wrong_team` → team flipped, `validation = corrected`, shot xG recomputed with the *basic* model (freeze frame of the wrong team is invalid) |
| Added events | typed in the review at the current video time → `source = manual`, counted in the stats like any other event |
| Shirt numbers | only those typed in the review / match sheet → `player_stats_by_number.csv` (+ roster names). Per-tracklet stats stay in `player_stats.csv` |
| Measured reliability | frame checks among frames the camera gate kept → precision of the calibration with a 95 % Wilson interval; frames the gate discarded but you judged good → how strict the gate is; event verdicts → precision of each rule and of its team attribution |
| `why` | every event gets a sentence with the rule, the measured values and the thresholds (`explain.explain_event`) |
| Clicked shot location | a shot / goal / added shot with `shot_xy` (pitch metres from the click and that frame's camera pose) gets x, y replaced and xG recomputed with the full model: opponents (attack frame) from tracking at the nearest analysed frame (≤ 6 source frames), goalkeeper from `gk_xy` if clicked, else the tracked goalkeeper; `location = clicked` / `clicked+gk` |
| Excluded stretch | tracks and ball blanked inside `[t0, t1]`, `cam_ok = False` → possession unknown, not dead ball; automatic events inside are dropped (`events_rejected.csv`, reason *inside an excluded stretch*) |
| Camera repair | user pose on frame *f* → re-run perception from the first analysed frame ≥ *f* to the end of the lost stretch + 5 s (max 180 s); rows of those frames replaced; ball re-cleaned over the run |
| Confidence grade | per statistic for the match: start level from the recording profile, moved by camera / ball coverage, your frame check, event review completeness, shot-location clicks, match-sheet consistency (rules in `docs/REPORT_GUIDE.md` §4, code `confidence.py`) |

## 9 · Tuning guide
| Symptom | Likely cause | Adjust |
|---|---|---|
| Too many short passes / interceptions | control flicker in crowds | ↑ `min_pass_length_m`, ↑ `possession.min_control_s`, ↓ `control_radius_m` |
| Passes missing | ball lost behind players | ↑ `ball.max_gap_s`, ↑ `control_radius_m` |
| Stoppages during slow build-up | static-ball rule too eager | ↑ `static_still_min_s`, ↓ `static_ball_spread_m` |
| Throw-ins classed as free kicks | touchline badly calibrated | recalibrate; ↑ `throwin_band_m` |
| Shots missing | ball speed under-estimated at 10 fps | ↑ `video.analysis_fps` to 25, ↓ `shot_min_speed` |
| Phantom goals | long stoppage + centre restart | ↑ `goal_min_stoppage_s`, enable `scoreboard` |
