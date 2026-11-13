"""Generates notebooks/offside_demo.ipynb (kept as code so the notebook stays reviewable)."""
import json
import os

cells = []


def md(s):
    cells.append(dict(cell_type='markdown', metadata={}, source=s.strip('\n').splitlines(keepends=True)))


def code(s):
    cells.append(dict(cell_type='code', metadata={}, execution_count=None, outputs=[],
                      source=s.strip('\n').splitlines(keepends=True)))


md('''
# Offside lines from broadcast frames — demo

This notebook walks through the `offside` package step by step:

1. find the pitch markings in the **decision frame** (hints or automatic candidates),
2. calibrate the camera against the FIFA pitch template,
3. detect players, their 33 body keypoints and their team,
4. lift the body keypoints to 3-D and decide **OFFSIDE / ONSIDE** for each attacker,
5. calibrate the **stationary camera once** from frames of *other plays* (clean background + many
   players used as rulers) and save a camera model,
6. judge any new image of that camera in ~1–2 s with `OffsidePredictor` (ball found automatically).

Run the cells top to bottom. Everything is configured by `configs/anadia_example.json`.
''')

code('''
import os, sys, json, copy
import numpy as np, cv2
import matplotlib.pyplot as plt
%matplotlib inline

ROOT = os.path.abspath('..') if os.path.basename(os.getcwd()) == 'notebooks' else os.getcwd()
sys.path.insert(0, ROOT)
import offside as off

def show(img, title=None, size=(16, 9)):
    plt.figure(figsize=size); plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)); plt.axis('off')
    if title: plt.title(title)
    plt.show()

print('project root:', ROOT)
''')

md('''
## 0. Models
The detector (YOLOX) and pose model (MediaPipe BlazePose) are ONNX files from the OpenCV Model Zoo.
They are downloaded into `models/` the first time (≈ 41 MB).
''')
code('''
if not all(os.path.exists(os.path.join(ROOT, 'models', m)) for m in ('yolox.onnx', 'mp_pose.onnx')):
    import runpy; runpy.run_path(os.path.join(ROOT, 'scripts', 'download_models.py'))
print(os.listdir(os.path.join(ROOT, 'models')))
''')

md('## 1. Decision frame and configuration')
code('''
cfg = json.load(open(os.path.join(ROOT, 'configs', 'anadia_example.json'), encoding="utf-8"))
dfc = cfg['decision_frame']
img = cv2.imread(os.path.join(ROOT, dfc['image']))
print(img.shape)
show(img, 'decision frame')
''')

md('''
## 2. Pitch markings

**Option A — automatic candidates.** Sequential RANSAC finds straight-line and ellipse candidates on the
white-line mask. Look at the ids and map them to template elements
(`off.all_element_names()` lists them).
''')
code('''
rng = np.random.default_rng(0)
cands = off.find_candidates(img, dfc['ignore_regions'], rng=rng)
print({k: len(v) for k, v in cands.items()})
show(off.draw_candidates(img, cands), 'automatic candidates')
print(off.all_element_names())
''')
code('''
# Map the ids you trust (check them in the picture above — they can differ between frames).
# Speckle in deep shadow can produce junk 'L' candidates: simply don't map them.
# Thin, far-away lines (the far touchline) are often missed here — add them with a hint (Option B).
feats_auto = off.features_from_candidates(cands, {'halfway_line': 'L0', 'centre_circle': 'E0'})
feats_auto.update(off.extract_features(img, [s for s in dfc['features'] if s['element'] == 'touchline_far'],
                                       dfc['ignore_regions'], rng))
show(off.draw_features(img, feats_auto), 'features from candidates (+ touchline hint)')
''')

md('''
**Option B — hints (used in the config).** Two clicks per straight line and a bounding box for the centre
circle; the package keeps the white pixels in a band around each hint and cleans them with RANSAC,
re-centring on the whole painted stripe.
''')
code('''
feats = off.extract_features(img, dfc['features'], dfc['ignore_regions'], rng)
print({k: len(v) for k, v in feats.items()})
show(off.draw_features(img, feats), 'labelled pitch-marking pixels')
''')

md('''
## 3. Players, pose and teams
Detections inside `ignore_regions` (score bug, logos) are dropped; players whose feet fall in an
`occluding_regions` box (the video player's play button here) are flagged with `*` and not used as a
calibration cue.
''')
code('''
det = off.PlayerDetector(os.path.join(ROOT, cfg['models_dir']), **cfg.get('detector', {}))
players = det(img, dfc['ignore_regions'], dfc['occluding_regions'], dfc['player_min_y'])
players = off.assign_teams(img, players, cfg['teams'], cfg.get('team_overrides'))
vis = img.copy()
for q in players:
    x0, y0, x1, y1 = q['box'].astype(int)
    col = {'AFC': (255, 200, 60), 'SVA': (60, 200, 255), 'REF': (0, 0, 0)}.get(q['team'], (0, 0, 255))
    cv2.rectangle(vis, (x0, y0), (x1, y1), col, 3)
    cv2.putText(vis, f"#{q['id']} {q['team']}{'*' if q['occluded'] else ''}", (x0, y0 - 8), 0, 0.9, col, 2)
    for k in (11, 12, 23, 24, 25, 26, 27, 28):
        cv2.circle(vis, tuple(q['kp'][k].astype(int)), 4, (0, 0, 255), -1)
show(vis, 'players, teams and a few keypoints')
''')

md('''
## 4. Single-frame calibration
Unknowns: focal length, rotation, camera position, pitch length/width, principal-point offset.
`inside_pitch` encodes the fact that the **near touchline is not in the frame** (bottom-row pixels are
inside the field of play) — without it the solution drifts to physically impossible cameras.
''')
code('''
frame = off.Frame('decision', (img.shape[1], img.shape[0]), feats, 'main', np.array(dfc['inside_pitch'], float),
                  people=off.players.calibration_people(players))
priors = off.pipeline.priors_from(cfg)
calib = off.calibrate([frame], priors, rng)
cam = calib['cams']['decision']
print(json.dumps(calib['diagnostics'], indent=1, default=float))
print('pitch %.1f x %.1f m' % (2 * calib['half_length'], 2 * calib['half_width']))
chk = img.copy(); off.draw_pitch_model(chk, cam, calib['half_length'], calib['half_width'], list(feats))
show(chk, 'calibrated template re-projected (magenta)')
''')

md('''
## 5. Decisions (+ Monte-Carlo uncertainty)
Each allowed keypoint (head, body, feet — not arms) is intersected with the player's vertical plane and
dropped onto the pitch. With the goalkeeper off-frame, the deepest visible defender is the second-last
opponent. The Monte-Carlo loop re-runs calibration with perturbed inputs; increase `n` for final results.
''')
code('''
scen = cfg['scenarios']
res, ball, base = off.decide(players, cam, dfc['ball_pixel'], scen)
draws = off.monte_carlo([frame], calib, players, dfc['ball_pixel'], scen, n=40, priors=priors)
summary, mc = off.summarise(base, draws)
for name, s in summary.items():
    print(name)
    for i, a in sorted(s['attackers'].items(), key=lambda t: -t[1]['margin_m'])[:3]:
        print(f"   attacker #{i}: {a['decision']}   (P(offside) = {a['p_offside']:.2f})")
for sc in scen:
    name = off.decision.scenario_name(sc)
    show(off.render_scenario(img, cam, calib['half_length'], calib['half_width'], res, summary[name],
                             sc['attacking'], sc['defending'], sc['direction'], list(feats), show_metrics=False), name)
show(off.render_birds_eye(cam, calib['half_length'], calib['half_width'], res, ball, summary, frame.size, ['AFC', 'SVA']),
     "bird's-eye view", size=(12, 8))
''')

md('''
## 6. Calibrate the stationary camera once, from frames of other plays

The camera never moves, so every frame of the match has exactly the same calibration. Frames of other
plays add no new markings, but they add:

* a **clean background** — the per-pixel median of the frames removes players, so every line pixel is
  visible;
* **many players at many depths** — scale-free height ratios, and for standing/walking players the body
  axis points at the vertical vanishing point. That is what fixes the perspective which the halfway line,
  far touchline and centre circle leave almost undetermined.

Put the frames (same resolution, same camera; any moment of the match — stoppages with standing players
are best) in `data/calibration_frames/` and run the cell (or `python calibrate_camera.py`).
Frames whose content is shifted (camera bumped / zoomed) are detected and left out.
''')
code('''
import glob
from offside.venue import calibrate_fixed_camera, save_camera_model
paths = [os.path.join(ROOT, dfc['image'])] + sorted(glob.glob(os.path.join(ROOT, 'data', 'calibration_frames', '*.png')))
images = [cv2.imread(p) for p in paths]
print(len(images), 'frame(s)')
venue = calibrate_fixed_camera(images, dfc, det, priors, n_draws=40)       # 150+ draws for the real model
model_path = os.path.join(ROOT, 'calibration', 'notebook_camera.json')
save_camera_model(model_path, venue, (img.shape[1], img.shape[0]))
show(venue['background'], 'median background (players removed when there are several frames)')
print(venue['meta'])
''')

md('''
## 7. Judge a play in ~1–2 seconds

With the camera model saved, judging an image is: detection (+ ball) → pose → teams → geometry →
uncertainty from the stored camera draws. No calibration at prediction time.
''')
code('''
from offside.predict import OffsidePredictor
pred = OffsidePredictor(model_path, cfg, os.path.join(ROOT, 'models'), detector=det)
out = pred.predict(img)                     # pass ball_pixel=(x, y) to override the automatic ball
print('ball:', out['ball_source'], out['ball_pixel'])
print('timings (s):', out['timings'])
for sc, d in out['decisions'].items():
    print(sc, {i: v for i, v in d.items() if out['summary'][sc]['attackers'][i]['margin_m'] > -3})
for name, im in out['images'].items():
    show(im, name)
''')

md('''
### Why the extra frames help — synthetic check with known ground truth
Same view as the real frame (halfway line, far touchline, centre circle painted 1.3 % out of round). Each
extra frame adds ~15 players at random places (30 % standing, the rest running). See
`tests/test_fixed_camera_synthetic.py` for the full table.
''')
code('''
from tests.test_multiframe_synthetic import make_frames, errors
from tests.test_fixed_camera_synthetic import synth_people
syn_frames, syn_cams, syn_imgs = make_frames()
show(syn_imgs['decision'], 'synthetic view (ground truth known)', size=(10, 5.6))
gt, dec = syn_cams['decision'], syn_frames[0]
r = np.random.default_rng(0)
for n_frames in (0, 40):
    people = sum((synth_people(gt, 15, r) for _ in range(n_frames)), [])
    e = errors(off.calibrate([off.Frame('decision', dec.size, dec.features, 'main', dec.inside_pitch, people)]), gt)
    print(f"{n_frames:3d} other-play frames -> max player-position error {e['max_X_err_m']:.2f} m")
''')

md('''
## Appendix — if the camera does move (pan / zoom)
For a broadcast camera that pans and zooms (same position), frames showing other markings — e.g. a
penalty area — can be added as `aux_frames` and calibrated jointly with `off.run(cfg)`; see the README
and `tests/test_multiframe_synthetic.py`. Not needed for this match (single stationary camera).
''')

nb = dict(cells=cells, metadata=dict(kernelspec=dict(display_name='Python 3', language='python', name='python3'),
                                     language_info=dict(name='python')), nbformat=4, nbformat_minor=5)
dst = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'notebooks', 'offside_demo.ipynb')
json.dump(nb, open(dst, 'w', encoding="utf-8"), indent=1)
print('wrote', os.path.normpath(dst), len(cells), 'cells')
