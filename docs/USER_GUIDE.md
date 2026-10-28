# User guide — analysing a match, step by step

This guide is for using the system day to day. Technical details are in the README and the other guides
(*Statistics rulebook*, *Report guide*, *Manual validation*, *Video storage*), all readable from the selector above.

## 0 · Start the launcher

| How | When |
|---|---|
| Double-click **Football Video Analysis** on the Desktop | normal use: opens the launcher in the browser |
| Double-click a **match shortcut** on the Desktop (made with *Desktop shortcut* in the Analyse tab) | opens the launcher with that recording, config and setup already chosen |
| `python scripts/launcher.py` in the project folder | the same, from a terminal |
| `python scripts/launcher.py --video <file> --start` | starts processing that recording right away (see §7) |

The first start creates the Python environment, which takes a few minutes once. A Terminal window stays open
while the launcher runs. Closing it, or the **Quit** button at the top right, stops the launcher and every job
it started. If the launcher is already running, a shortcut just opens it in the browser with the chosen video.

## 1 · Add the recording

* Drop the `.mp4` on the drop zone at the bottom left, or copy it into `data/videos/`.
* Name it `YYYY-MM-DD_Home_vs_Away_round.mp4` so the list sorts by date.
* Short of disk space? Add an external drive or a cloud folder as a library, then archive analysed matches there
  (*Video storage* guide).

Click the recording in the list. The **Analyse** tab shows four steps.

## 2 · Choose the config and the setup

| Field | What to pick |
|---|---|
| **Venue / match config** | the YAML of the ground: camera type, detector, overlays. For Canedo use `configs/canedo_cfctv.yaml` (whole match) or `canedo_segment.yaml` (10-minute test). A new ground needs a new config (README, *Club recordings*). |
| **Match setup** | `configs/match_setup_<video>.json`. It is created on the first *Open setup*. Recordings with a setup show **setup ✓**. |
| **Output folder** | where the results go (`outputs/<name>`). The default is the video name. |

The launcher remembers these choices per recording.

## 3 · Step 1 — Setup (≈10 min, before processing)

*Open setup* shows the setup in the **Manual decisions** tab. Go through the steps on the left. Each is saved as
you type:

1. **Match & pitch:** team names, date, the real pitch size (measure it or ask the club; it changes every statistic).
   Also say whether the **near touchline (camera side)** is in the picture. If it is not, the pitch size you typed
   still holds: the pitch simply continues below the frame (see step 5).
2. **Periods & direction:** stop the video at each kick-off and half end and press *set*. Then say which way Team A attacks in the 1st half.
3. **Overlays:** draw boxes over the club crest and the stream logo so that they are ignored.
4. **Kits:** press *Detect players on this frame*, then click **inside** the boxes: 3–6 players of each team, in sun
   and in shade, on 3–4 different moments, plus both goalkeepers and the referee. If you drag a box yourself it is
   snapped to the detected player (a loose box is mostly grass and spoils the shirt colour).
   These clicks drive the boxes you see during processing: every player is matched to the nearest clicked samples,
   the tracks you clicked keep your label (marked `*` on the live screen).
5. **Pitch landmarks:** click the points you can see (centre spot, corners of the box, …), then check the overlay.
   **Near touchline not in the picture?** Tick *Show the area outside the picture*: a grey margin appears around
   the frame. Follow the painted lines and click where a near-side landmark *would* be (e.g. `halfway_bottom`, a box
   corner, a near corner flag). After *Fit*, the cyan model continues into the margin and the panel tells you where
   the near touchline ends up (and warns if that contradicts what you said in step 1).
   Panning camera (*flow*): add a landmark frame near each kick-off and every 5–10 minutes — before the first one
   there are no positions, and the camera estimate drifts between frames far apart.
6. **Match sheet:** final score, goal minutes, cards, substitutions.
7. **Checklist:** red items block processing. Amber items are allowed, but they lower the confidence of the numbers they feed.

Sections filled in by the assistant show **Pre-filled, not yet confirmed**. Check them and press *I checked this*.

## 4 · Step 2 — Process the video (~1–2 h on a laptop)

* Choose the **Speed**: *standard* (10 analysed frames per second), *fast* (5 per second, about 2× faster, misses
  more short passes / fast shots) or *quality* (smaller detector tiles, finds more distant players and balls, 2–3×
  slower). The line under it estimates the time from the speed measured on this computer by your previous runs.
* Only the periods you set in the setup are processed (+ 20 s around them): warm-up, half-time and the walk-off cost
  nothing.
* Tick **quick test: first 600 frames** the first time on a new ground. Check the result, then run the whole match.
* Press **▶ Start analysis**. Then press **● Watch live** (or open the **Match screen** tab, *Live*). You see the
  frame being analysed with pitch lines, players in their shirt colours, ball candidates and the minimap, plus
  progress, speed and time left.
* What to look for live:
  * lines that sit on the painted lines (camera OK);
  * boxes on every player (the dashed rectangles are the detector tiles — the sky above the pitch is skipped);
  * the two teams in two colours: blue = Team A, orange = Team B, purple / pink = goalkeepers, black = referee,
    grey = not decided yet. Labels show the short team name + track number; `*` = a player you clicked in the setup.
  * if the near touchline is out of view, the status bar says how far before it the picture ends and the minimap
    darkens that strip.
  
  If the lines drift for a long time, note the minute. You fix it in the review (*Camera repair*).
* The Mac must stay awake. The launcher asks macOS not to sleep while a job runs, but keep the lid open and the
  charger connected.
* At the end, the report is written and the clips of the flagged opportunities are cut.

## 5 · Step 3 — Review (≈10 min) and step 4 — Apply

*Open review* shows the review in the **Manual decisions** tab:

1. **Frame checks:** for ~16 random frames, say whether the lines and the teams are right. This measures how far the numbers can be trusted.
2. **Camera repair:** for stretches where the camera was lost, click 2–6 landmarks. Only those minutes are re-processed.
3. **Key events:** confirm or reject each goal and shot, flip the team if wrong, type the shirt number, click the shot location for a better xG.
4. **Missing events:** add the shots and goals the system missed. Each one becomes a flagged opportunity.
5. **Exclude stretches:** replays, injuries, crowd shots.

Then press **Apply review → re-run analytics** (seconds), or step 4 in the Analyse tab.

## 6 · Look at the results

| Tab | Use it to |
|---|---|
| **Match screen → Replay** | watch the match with every overlay. Click the timeline or *event ▶* to jump to shots and goals. Keys: `space` play, `←/→` ±5 s, `n/p` next / previous event. |
| **Results & opportunities** | see the score, the key metrics with their confidence, and the ⚑ flagged chances. Click one to play it; `n/p` moves between them; **overlays ⧉** opens it on the Match screen. Here you also open the full report and cut clips. |
| **Report** (*Open full report*) | read everything, with the confidence of each number and why. Send it with the `clips/` folder. |

Read the confidence badges:

| Badge | Meaning |
|---|---|
| **High** | quote it |
| **Medium** | compare the teams |
| **Low** | indicative |
| **Experimental** | do not use |

## 7 · Process a chosen recording straight away

When the setup is done and you just want the match processed (overnight, for example):

```bash
cd ~/Desktop/DS_ML/football_video_evaluation
./start.command --video 2026-10-18_UD_Mansores_vs_Canedo_FC_J2.mp4 --config configs/canedo_cfctv.yaml --start
```

The launcher opens with that recording selected and starts step 2 at once. The config and setup are taken from
the command, else from the match setup of that video, else from the last ones used. Follow it live on the Match
screen. A **match shortcut** on the Desktop does the same with a double-click (*Desktop shortcut* button). Tick
*start processing when opened* to make it start step 2 too.

## 8 · After the match

* **Archive:** *Move to archive* sends the recording to the archive library. The results, clips and replay keep
  working while that library is reachable.
* Everything about a match is in `outputs/<name>/`: `report.html`, `opportunities.csv`, `clips/`, `events.csv`,
  the stats tables.

## Troubleshooting

| Problem | What to do |
|---|---|
| The browser does not open | go to http://127.0.0.1:8770 |
| "address already in use" | the launcher is already running: use the open tab, or press *Quit* there first |
| Lines far from the painted lines | review → *Camera repair*; for a new ground, calibrate it first (README) |
| Teams swapped or mixed | add kit clicks in sun **and** shade on 3–4 moments (setup step 4), then process again. The run prints how many clicked samples are recognised by the others ("leave-one-out"): below ~90 % the kits are hard to tell apart — add samples |
| Processing too slow | Speed *fast*; make sure `onnxruntime` is installed (`.venv/bin/pip install onnxruntime`, done automatically by start.command); set the periods so half-time is skipped |
| Many throw-ins on the camera side missing | the near touchline is out of view: they are inferred when the ball leaves the bottom of the picture (rulebook §3.1) — check them in the review |
| No shots found | expected on 640 × 360 streams (2–5 px ball): add them in the review, *Missing events* |
| A job failed | the **Jobs** tab has its full log; the last lines say why |
