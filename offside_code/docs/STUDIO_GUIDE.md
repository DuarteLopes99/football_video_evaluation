# Offside tab — guide

The **Offside** tab of the dashboard (`football_video_evaluation/start.command` → tab *Offside*) is the web
front end of `offside_code`: give it a set of images of one camera, calibrate the camera semi-manually, then
judge **offside / onside** on a chosen frame — with or without the offside lines and the metrics.

It runs the same code as the command line (`calibrate_camera.py`, `run_offside.py`); the tab only collects the
manual inputs and shows the results. It also runs on its own: `python offside_code/offside_studio.py`
(http://127.0.0.1:8775).

---

## Venues

A **venue** is one camera set-up (e.g. *Mansores main camera*, *Phone from the stand at Macieira*). It lives in
`offside_code/venues/<name>/`:

| File | What |
|---|---|
| `venue.json` | the config — same format as `configs/*.json` (so the CLIs work on it) + a `studio` block with every frame and its clicked markings |
| `images/` | the images you dropped in the tab |
| `camera_model.json` | the calibrated camera (+ `camera_model_background.png` for a stationary camera) |
| `results/<frame>/` | decision images + `offside_result.json` of each judged frame |

**+ New venue** asks for a name and the camera type:

* **Stationary camera** — never moves (club camera on a pole, fixed wide shot). The markings are clicked once on
  the ★ reference frame and hold for every frame. Frames of *other plays* are pooled: the per-pixel median
  removes the players (complete lines) and every player becomes a ruler of the perspective. **40+ frames**,
  including stoppages with players standing, are worth far more than one frame (README, table in §3).
* **Panning camera** — turns / zooms from one place (phone held from a seat, PTZ). Every frame has its own view,
  so every frame used for the calibration needs its own markings. Frames showing *other* markings (centre
  circle, a penalty area) share the camera position and fix the perspective. Only frames that had markings when
  the camera was calibrated can be judged.

**Import config** turns an existing `configs/*.json` (e.g. `anadia_example.json`, `macieira.json`) into a venue;
its images stay where they are, and an existing stationary-camera model is copied, so you can decide at once.

---

## Step 1 — Frames & markings

**Add frames** on the left: drop PNG/JPG screenshots of the match video (same camera), or *Add from
offside_code/ folders…* for images already in `data/`. Choose *calibration* (used to calibrate) or *play to judge*
(only judged). The first frame becomes the **★ reference** (the decision frame of the config); any frame can be
made the reference.

**Click the pitch markings** you can see (toolbar → *✎ Marking* + element):

| Marking | How |
|---|---|
| straight line (halfway line, touchlines, goal lines, penalty / goal area edges) | 2 clicks **on** the painted line, near its two ends (Shift-click for more points) |
| arc (penalty arc, part of the centre circle) | 3+ clicks along it, then **Enter** (or double-click / *Finish*) |
| whole centre circle | drag a box around it |
| point (centre spot, penalty spot, corner) | 1 click |

*band* = tolerance around your clicks in image pixels (wider for thick, near lines; ~12 for thin far lines);
tick *thin/far* for the far touchline. "Left/right" = goal on the left/right of the picture, "far/near" =
touchline away from / close to the camera. Drag any dot to move it; ✕ in the list removes a marking.

Also:

* **＋ Turf points** — 3–5 clicks on grass *inside* the field along the bottom of the picture (tells the
  calibration that the near touchline is not visible — it stabilises the perspective a lot).
* **▭ Ignore box** — score bug, logos, crowd: no line pixels and no players are taken from it.
* **▭ Occluding box** — an overlay that can hide players' feet: those players are not used as rulers.
* **⤒ Far-touchline limit** — click just above the far touchline: people above it (bench, fans) are ignored.

**Check markings** shows the white-line pixels each marking picks up (count per marking). A marking with very few
pixels (⚠) is mis-clicked: click closer to the line or widen the band. Everything is saved automatically.

Good set of markings: lines in **both directions** (e.g. halfway line + far touchline) **and** a circle or arc.

## Step 2 — Calibrate

The checklist shows what is there; **▶ Calibrate camera** starts `calibrate_camera.py --config
venues/<name>/venue.json` in the background with a live log (≈ 1–3 min for a stationary camera, a few minutes
for several panning frames; *quick (30)* uses fewer Monte-Carlo draws for a first try). Afterwards:

* the **calibrated pitch** is drawn in magenta on every frame — it must sit on the painted lines (switch frames on
  the left; tick *line pixels found* to compare);
* the panel lists the pitch size found, focal length / field of view, camera position, the residual of every
  marking (a few px is normal; ≫ 6 px = a wrong marking), how many players were used as rulers and frames left
  out (moved camera / other size);
* if you change markings later, the badge says *markings changed since — calibrate again*.

## Step 3 — Decide

1. **Choose the frame** on the left (✓ judge = the camera model fits it).
2. **Teams** — type a team name, press **🎨** and click one of its shirts (or *shorts*): an HSV colour rule is
   made from the clicked colour (white / black / grey kits get a brightness rule; red wraps around the hue circle
   automatically). Rules are checked top to bottom; a team without rules takes everyone not matched above; `REF`
   is the referee. *edit the colour rules (JSON)* for fine-tuning.
3. **Attacking direction** — one frame cannot tell who attacks where, so each line is one hypothesis:
   attacking team, → right / ← left, defending team, *GK in view* (if the defending goalkeeper is in the picture the
   line goes through the 2nd-deepest defender; otherwise the deepest visible defender). *both directions* judges
   the two hypotheses.
4. **Ball** — detected automatically, clicked by you (● *Ball* tool, click its contact point), or ignored (line =
   second-last defender / halfway line).
5. **Show** — *offside line & player lines* (red line through the 2nd-last defender with its 90 % band, the
   defender's "curtain", a line per attacker), *metrics* (distance to the line in cm, P(offside), 90 % band of the
   line), *calibrated pitch markings*. Toggling them only re-draws the images (≈ 0.1 s).
6. **▶ Decide on this frame** (≈ 1.5 s the first time on a laptop; detections are then re-used, so changing teams,
   ball or directions is instant).

Results: the verdict per direction in the toolbar; the tabs *Frame* (players found, their team, the ball),
*→ / ← team* (the decision image), *Bird's-eye*; in the panel the attackers near the line with their distance and
P(offside), which defender sets the line, and every player with a team selector — **change a wrongly coloured
player's team** there (or *⇄ Player team* + click him), the decision is re-computed at once. Everything is saved
to `venues/<name>/results/<frame>/` (*full resolution ↗* opens the PNG).

---

## Command line equivalents

```bash
cd offside_code
python calibrate_camera.py --config venues/<name>/venue.json          # step 2
python run_offside.py --config venues/<name>/venue.json --image <frame> [--metrics] [--no-lines] [--no-ball]
python offside_studio.py                                              # the tab on its own page
python tests/test_studio_synthetic.py                                 # back-end self-test (synthetic, ~1 min)
```

## Limits (see README → Limitations)

A single frame is not the moment of the pass; players outside the picture are not known; a stationary-camera model
is only valid while the camera does not move (frames that moved are detected and left out); team colour rules are
per match. The decision is geometric — it does not judge *involvement in play*.
