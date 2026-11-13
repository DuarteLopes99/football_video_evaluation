#!/usr/bin/env python3
"""Judge a play:  python run_offside.py [--image play.png] [--ball X Y] [--metrics] [--no-lines]

Uses the camera model of the config (camera.camera_model). If it does not exist yet it is built first from the
config's frames (same as calibrate_camera.py). With a panning-camera model the image must be one of the
calibrated frames (decision_frame / aux_frames)."""
import argparse
import json
import os
import subprocess
import sys

import cv2

from offside.predict import OffsidePredictor

ap = argparse.ArgumentParser(description='Offside decision for one image of a calibrated camera.')
ap.add_argument('--config', default='configs/anadia_example.json')
ap.add_argument('--image', help='image to judge (default: config decision_frame.image)')
ap.add_argument('--ball', nargs=2, type=float, help='ball contact pixel x y (default: detected automatically)')
ap.add_argument('--no-ball', action='store_true', help='ignore the ball (line = 2nd-last defender / halfway line)')
ap.add_argument('--metrics', action='store_true', help='write distances / probabilities on the images')
ap.add_argument('--no-lines', action='store_true', help='decisions only: no offside / attacker lines')
ap.add_argument('--no-pitch', action='store_true', help='do not draw the calibrated pitch template')
ap.add_argument('--no-uncertainty', action='store_true')
ap.add_argument('--out', help='output folder (overrides config)')
ap.add_argument('--recalibrate', action='store_true')
a = ap.parse_args()
root = os.path.dirname(os.path.abspath(__file__))
cfg_path = a.config if os.path.isabs(a.config) else os.path.join(root, a.config)
cfg = json.load(open(cfg_path))
cam_cfg = cfg.get('camera', {})
model = os.path.join(root, cam_cfg.get('camera_model', 'calibration/camera_model.json'))
if a.recalibrate or not os.path.exists(model):
    subprocess.run([sys.executable, os.path.join(root, 'calibrate_camera.py'), '--config', cfg_path], check=True)
if a.metrics:
    cfg['show_metrics'] = True
img_path = a.image or os.path.join(root, cfg['decision_frame']['image'])
img = cv2.imread(img_path)
if img is None:
    sys.exit(f'cannot read {img_path}')

# panning camera: the frame's own camera and markings config
frame, frame_cfg = None, None
if (cam_cfg.get('mode') or ('panning' if cfg.get('aux_frames') else 'fixed')) == 'panning':
    rel = os.path.relpath(os.path.abspath(img_path), root)
    for fc in [cfg['decision_frame']] + list(cfg.get('aux_frames', [])):
        if os.path.normpath(fc['image']) == os.path.normpath(rel):
            frame, frame_cfg = fc['image'], fc
    if frame is None:
        sys.exit(f'{rel} is not a calibrated frame of this panning camera: add it to aux_frames and recalibrate')

pred = OffsidePredictor(model, cfg, os.path.join(root, cfg.get('models_dir', 'models')), frame=frame)
out = pred.predict(img, ball_pixel=a.ball, uncertainty=not a.no_uncertainty, frame_cfg=frame_cfg,
                   auto_ball=not a.no_ball, show_lines=not a.no_lines, show_pitch=not a.no_pitch)
out_dir = os.path.join(root, a.out or cfg.get('output_dir', 'outputs'))
os.makedirs(out_dir, exist_ok=True)
for name, im in out['images'].items():
    fn = 'birds_eye.png' if name == 'birds_eye' else 'offside_' + name.replace(' attacks ', '_attacks_') + '.png'
    cv2.imwrite(os.path.join(out_dir, fn), im)
print('players:', ', '.join(f"#{q['id']}:{q['team']}" for q in out['players']))
for sc, d in out['decisions'].items():
    print(f'{sc}: ' + ', '.join(f'#{i} {v}' for i, v in d.items() if out['summary'][sc]['attackers'][i]['margin_m'] > -3))
for sc, why in out['skipped'].items():
    print(f'{sc}: skipped ({why})')
print('ball:', out['ball_source'], out['ball_pixel'])
print('timings (s):', out['timings'])
json.dump(dict(decisions=out['decisions'], summary=out['summary'], timings=out['timings']),
          open(os.path.join(out_dir, 'offside_result.json'), 'w'), indent=1, default=float)
