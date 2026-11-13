"""Time each stage of judging one image with a pre-calibrated camera.
    python scripts/benchmark.py [--camera calibration/anadia_camera.json] [--repeat 3]"""
import argparse, json, os, sys, time
import cv2
import numpy as np
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.insert(0, ROOT)
from offside.predict import OffsidePredictor          # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('--config', default='configs/anadia_example.json')
ap.add_argument('--camera', default=None)
ap.add_argument('--repeat', type=int, default=3)
ap.add_argument('--tile', type=int, default=None)
ap.add_argument('--stride', type=int, default=None)
a = ap.parse_args()
cfg = json.load(open(os.path.join(ROOT, a.config)))
if a.tile:
    cfg.setdefault('detector', {}).update(tile=a.tile, stride=a.stride or int(a.tile * 0.85))
cam_path = os.path.join(ROOT, a.camera or cfg['camera']['camera_model'])
img = cv2.imread(os.path.join(ROOT, cfg['decision_frame']['image']))
t0 = time.perf_counter()
pred = OffsidePredictor(cam_path, cfg, os.path.join(ROOT, cfg['models_dir']))
print(f'load models + camera model: {time.perf_counter() - t0:.2f} s (once)')
rows = []
for k in range(a.repeat):
    out = pred.predict(img)
    rows.append(out['timings'])
    print(f"run {k + 1}: " + '  '.join(f'{s}={v:.2f}s' for s, v in out['timings'].items()))
med = {s: float(np.median([r[s] for r in rows])) for s in rows[0]}
print('median  : ' + '  '.join(f'{s}={v:.2f}s' for s, v in med.items()))
print('players :', len(out['players']), ' decisions:', out['decisions'])
