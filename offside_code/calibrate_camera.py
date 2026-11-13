#!/usr/bin/env python3
"""Calibrate the camera ONCE from many frames and save the camera model.

    python calibrate_camera.py --config configs/anadia_example.json
    python calibrate_camera.py --frames "data/calibration_frames/*.png" --draws 200

Two camera set-ups (config ``camera.mode``; default: "panning" when the config has ``aux_frames``, else "fixed"):

* fixed   — the camera never moves. Frames: the decision frame + camera.calibration_frames (+ --frames). They must
            come from the same, unmoved camera at the same resolution; shifted frames are detected and left out.
            The markings (hints) of the decision frame are valid for every frame. Best frames: stoppages / walking
            players (upright bodies), players spread over the whole visible pitch.
* panning — a camera that pans / zooms from one position (phone held from one seat, PTZ). The decision frame and
            every ``aux_frames`` entry have their own markings; they are calibrated jointly (shared camera centre)
            and the model keeps one camera per frame, so any of those frames can then be judged.

The dashboard's Offside tab runs exactly this script (venues/<name>/venue.json is a config of the same format).
"""
import argparse
import json
import os
import sys

import cv2

from offside import PlayerDetector
from offside.pipeline import priors_from
from offside.venue import calibrate_fixed_camera, calibrate_panning_camera, save_camera_model, expand_paths


def camera_mode(cfg):
    return cfg.get('camera', {}).get('mode') or ('panning' if cfg.get('aux_frames') else 'fixed')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default='configs/anadia_example.json')
    ap.add_argument('--frames', nargs='*', default=[], help='extra calibration frames (globs ok; fixed camera only)')
    ap.add_argument('--draws', type=int, help='Monte-Carlo camera draws stored in the model')
    ap.add_argument('--out', help='camera model path (default: config camera.camera_model)')
    a = ap.parse_args()
    root = os.path.dirname(os.path.abspath(__file__))
    cfg = json.load(open(a.config if os.path.isabs(a.config) else os.path.join(root, a.config)))
    cam_cfg = cfg.get('camera', {})
    mode = camera_mode(cfg)
    n_draws = a.draws or cam_cfg.get('n_draws', 150)
    det = PlayerDetector(os.path.join(root, cfg.get('models_dir', 'models')), **cfg.get('detector', {}))
    out = a.out or cam_cfg.get('camera_model', 'calibration/camera_model.json')
    out = out if os.path.isabs(out) else os.path.join(root, out)

    if mode == 'panning':
        fcs = [cfg['decision_frame']] + list(cfg.get('aux_frames', []))
        names = [fc['image'] for fc in fcs]
        paths = [fc['image'] if os.path.isabs(fc['image']) else os.path.join(root, fc['image']) for fc in fcs]
        images = [cv2.imread(p) for p in paths]
        missing = [p for p, im in zip(paths, images) if im is None]
        if missing:
            sys.exit(f'cannot read: {missing}')
        print(f'[venue] panning camera, {len(images)} frame(s) with markings: '
              + ', '.join(os.path.basename(p) for p in paths), flush=True)
        res = calibrate_panning_camera(images, fcs, names, det, priors_from(cfg), n_draws=n_draws)
        save_camera_model(out, res, (images[0].shape[1], images[0].shape[0]), dict(frames=names))
        print(f'[venue] camera model -> {os.path.relpath(out, root)}  ({len(names)} calibrated frames)', flush=True)
        return

    paths = [os.path.join(root, cfg['decision_frame']['image'])]
    paths += expand_paths(cam_cfg.get('calibration_frames', []) + a.frames, root)
    paths = list(dict.fromkeys(paths))
    images = [cv2.imread(p) for p in paths]
    missing = [p for p, im in zip(paths, images) if im is None]
    if missing:
        sys.exit(f'cannot read: {missing}')
    print(f'[venue] fixed camera, {len(images)} frame(s): ' + ', '.join(os.path.basename(p) for p in paths), flush=True)
    res = calibrate_fixed_camera(images, cfg['decision_frame'], det, priors_from(cfg), n_draws=n_draws)
    save_camera_model(out, res, (images[0].shape[1], images[0].shape[0]),
                      dict(frames=[os.path.relpath(p, root) for p in paths]))
    bg = os.path.splitext(out)[0] + '_background.png'
    cv2.imwrite(bg, res['background'])
    print(f'[venue] camera model -> {os.path.relpath(out, root)}   background -> {os.path.relpath(bg, root)}', flush=True)


if __name__ == '__main__':
    main()
