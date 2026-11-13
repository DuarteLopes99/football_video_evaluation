"""Calibrating a STATIONARY camera once, from many frames of different plays.

The camera never moves, so every frame shares exactly the same K, R, C. Frames from other plays add no
new pitch markings (the view is identical), but they add two things that a single frame lacks:

  1. a clean view of the markings: the per-pixel temporal median of N frames removes the players
     (and averages out moving shadows), so the line/circle pixels are complete;
  2. many players at many depths, i.e. many "rulers" standing on the pitch: scale-free height ratios
     and, for upright players (stoppages, walking), the direction of the vertical. Together they fix the
     perspective that the halfway line + touchline + centre circle leave almost undetermined.

The result is saved as a *camera model* (JSON) together with Monte-Carlo camera draws, so judging a new
play only needs detection + pose + geometry (see predict.py).

A camera that PANS / ZOOMS from one position (a phone held from one seat, a PTZ) is calibrated with
``calibrate_panning_camera``: every frame has its own markings, rotation and focal length, the camera centre
is shared. The camera model then also stores one camera (+ draws) per calibrated frame, and any of those
frames can be judged.
"""
import glob
import json
import os
import time

import cv2
import numpy as np

from .calibration import Frame, Priors, calibrate
from .features import extract_features
from .players import calibration_people
from .camera import make_camera, vanishing_point

# ------------------------------------------------------------------ frames


def expand_paths(patterns, base='.'):
    out = []
    for p in patterns:
        p = p if os.path.isabs(p) else os.path.join(base, p)
        out += sorted(glob.glob(p)) if any(c in p for c in '*?[') else [p]
    return out


def alignment_shift(ref, img):
    """Sub-pixel translation between two frames (phase correlation on grey levels). A stationary camera
    should give ~0; a large value means the camera was moved/zoomed and the frame must not be pooled."""
    a = cv2.cvtColor(cv2.resize(ref, None, fx=0.5, fy=0.5), cv2.COLOR_BGR2GRAY).astype(np.float32)
    b = cv2.cvtColor(cv2.resize(img, None, fx=0.5, fy=0.5), cv2.COLOR_BGR2GRAY).astype(np.float32)
    win = cv2.createHanningWindow(a.shape[::-1], cv2.CV_32F)
    (dx, dy), resp = cv2.phaseCorrelate(a, b, win)
    return 2 * float(np.hypot(dx, dy)), float(resp)


def median_background(images):
    """Per-pixel temporal median — players and the ball disappear, painted lines stay."""
    if len(images) == 1:
        return images[0]
    return np.median(np.stack(images), axis=0).astype(np.uint8)


# ------------------------------------------------------------------ calibration


def calibrate_fixed_camera(images, frame_cfg, detector, priors=None, max_shift_px=3.0, n_draws=150,
                           seed=0, verbose=True, perturb_sigma=(0.005, 0.008, 0.008), pp_sigma=40.0):
    """images[0] is the reference frame (e.g. the decision frame); the rest are frames from other plays.
    frame_cfg holds the hints (`features`), `inside_pitch`, `ignore_regions`, `occluding_regions`,
    `player_min_y` — valid for every frame because the view never changes."""
    log = print if verbose else (lambda *a, **k: None)
    rng = np.random.default_rng(seed)
    t0 = time.time()
    ref = images[0]
    used, rejected = [ref], []
    for k, im in enumerate(images[1:], 1):
        if im.shape != ref.shape:
            rejected.append((k, 'size'))
            continue
        shift, _ = alignment_shift(ref, im)
        if shift <= max_shift_px:
            used.append(im)
        else:
            rejected.append((k, f'shift {shift:.1f}px'))
    if rejected:
        log(f'[venue] {len(rejected)} frame(s) not pooled (camera moved / different size): {rejected}')
    bg = median_background(used)
    ign = frame_cfg.get('ignore_regions', [])
    feats = extract_features(bg, frame_cfg['features'], ign, rng)
    people = []
    n_up = 0
    for im in used:
        pl = detector(im, ign, frame_cfg.get('occluding_regions', []), frame_cfg.get('player_min_y', 0))
        obs = calibration_people(pl)
        n_up += sum(o['upright'] for o in obs)
        people += obs
    log(f'[venue] {len(used)} frames pooled, {len(people)} player observations ({n_up} upright)')
    H, W = ref.shape[:2]
    frame = Frame('camera', (W, H), feats, 'main',
                  np.array(frame_cfg['inside_pitch'], float) if frame_cfg.get('inside_pitch') else None, people)
    priors = priors or Priors()
    calib = calibrate([frame], priors, rng)
    draws = camera_draws([frame], calib, priors, n_draws, rng, perturb_sigma, pp_sigma, verbose)
    vps = np.array([vanishing_point(c, (0, 1, 0))[1] for c in draws])
    meta = dict(mode='fixed', n_frames=len(used), n_rejected=len(rejected), n_people=len(people), n_upright=n_up,
                seconds=round(time.time() - t0, 1), vp_row_ci90=np.percentile(vps, [5, 95]).tolist())
    log(f"[venue] f={calib['cams']['camera']['f']:.0f}px  goal-line vanishing row 90% CI "
        f"{np.round(meta['vp_row_ci90'])}  ({meta['seconds']} s)")
    return dict(calib=calib, draws=draws, background=bg, features=feats, meta=meta)


def calibrate_panning_camera(images, frame_cfgs, names, detector=None, priors=None, n_draws=150, seed=0,
                             verbose=True, perturb_sigma=(0.005, 0.008, 0.008), pp_sigma=40.0):
    """A camera that pans / zooms from ONE position. images[0] / frame_cfgs[0] is the reference frame.

    Each frame config has its own hints (`features`), `inside_pitch`, `ignore_regions`, `occluding_regions`,
    `player_min_y`. Players found in every frame are added as calibration cues of that frame (needs `detector`).
    Returns the same dict as calibrate_fixed_camera plus `ref`, `draws_by_frame` and `sizes` (per frame)."""
    log = print if verbose else (lambda *a, **k: None)
    rng = np.random.default_rng(seed)
    t0 = time.time()
    frames, n_people, n_up = [], 0, 0
    for name, im, fc in zip(names, images, frame_cfgs):
        ign = fc.get('ignore_regions', [])
        feats = extract_features(im, fc.get('features', []), ign, rng)
        obs = []
        if detector is not None:
            pl = detector(im, ign, fc.get('occluding_regions', []), fc.get('player_min_y', 0))
            obs = calibration_people(pl)
        n_people += len(obs)
        n_up += sum(o['upright'] for o in obs)
        H, W = im.shape[:2]
        inside = np.array(fc['inside_pitch'], float) if fc.get('inside_pitch') else None
        frames.append(Frame(name, (W, H), feats, 'main', inside, obs))
        log(f'[venue] {name}: ' + ', '.join(f'{n}={len(P)}' for n, P in feats.items()) + f'; {len(obs)} players')
    priors = priors or Priors()
    calib = calibrate(frames, priors, rng)
    log(f"[venue] calibrated {len(frames)} frames (shared camera centre {np.round(calib['cams'][names[0]]['C'], 1)})")
    by_frame = camera_draws(frames, calib, priors, n_draws, rng, perturb_sigma, pp_sigma, verbose, all_frames=True)
    ref = names[0]
    vps = np.array([vanishing_point(c, (0, 1, 0))[1] for c in by_frame[ref]])
    meta = dict(mode='panning', n_frames=len(frames), n_rejected=0, n_people=n_people, n_upright=n_up,
                seconds=round(time.time() - t0, 1), vp_row_ci90=np.percentile(vps, [5, 95]).tolist())
    log(f"[venue] f={calib['cams'][ref]['f']:.0f}px (reference)  goal-line vanishing row 90% CI "
        f"{np.round(meta['vp_row_ci90'])}  ({meta['seconds']} s)")
    return dict(calib=calib, draws=by_frame[ref], draws_by_frame=by_frame, ref=ref, background=images[0],
                features=frames[0].features, meta=meta, sizes={f.name: f.size for f in frames})


def camera_draws(frames, calib, priors, n, rng, circle_sigma=(0.005, 0.008, 0.008), pp_sigma=40.0, verbose=False,
                 width_sigma=2.5, length_sigma=3.0, all_frames=False):
    """Monte-Carlo cameras: recalibrate with bootstrap-resampled pixels/players and perturbed priors.
    Returns the draws of frames[0] (list), or with all_frames=True {frame name: list} for every frame."""
    out = []
    name = frames[0].name
    by_frame = {f.name: [] for f in frames}
    for it in range(n):
        pr = Priors(half_length=(rng.normal(priors.half_length[0], length_sigma), priors.half_length[1]),
                    half_width=(rng.normal(priors.half_width[0], width_sigma), priors.half_width[1]),
                    shoulder_h=(rng.normal(priors.shoulder_h[0], 0.08), priors.shoulder_h[1]),
                    pp_sigma=priors.pp_sigma, pp_offset=(rng.normal(0, pp_sigma), rng.normal(0, pp_sigma)))
        perturb = dict(circle=(rng.normal(0, circle_sigma[0]), rng.normal(0, circle_sigma[1]), rng.uniform(0, 2 * np.pi),
                               rng.normal(0, circle_sigma[2]), rng.uniform(0, 2 * np.pi)))
        cb = calibrate(frames, pr, rng, x0=calib['params'], perturb=perturb, boot=True, max_nfev=1500)
        out.append(cb['cams'][name])
        for f in frames:
            by_frame[f.name].append(cb['cams'][f.name])
        if verbose and ((it + 1) % 25 == 0 or it + 1 == n):
            print(f'  camera draws {it + 1}/{n}', flush=True)
    return by_frame if all_frames else out


# ------------------------------------------------------------------ camera model file


def _cam_to_json(c):
    return dict(f=c['f'], rvec=cv2.Rodrigues(c['R'])[0].ravel().tolist(), C=c['C'].tolist(), cx=c['cx'], cy=c['cy'])


def _cam_from_json(d):
    return make_camera(d['f'], d['rvec'], d['C'], d['cx'], d['cy'])


def save_camera_model(path, result, frame_size, extra=None):
    c = result['calib']
    ref = result.get('ref', 'camera')
    js = dict(created=time.strftime('%Y-%m-%d %H:%M'), frame_size=list(frame_size),
              mode=result['meta'].get('mode', 'fixed'),
              camera=_cam_to_json(c['cams'][ref]), half_length=c['half_length'], half_width=c['half_width'],
              diagnostics=c['diagnostics'], draws=[_cam_to_json(d) for d in result['draws']], meta=result['meta'],
              **(extra or {}))
    if result.get('draws_by_frame'):                 # panning camera: one camera (+ draws) per calibrated frame
        js['frame_cameras'] = {n: dict(frame_size=list(result['sizes'][n]), camera=_cam_to_json(c['cams'][n]),
                                draws=[_cam_to_json(d) for d in result['draws_by_frame'][n]])
                        for n in result['draws_by_frame']}
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, 'w', encoding="utf-8") as fh:
        json.dump(js, fh, indent=1, default=float)
    return path


def load_camera_model(path, frame=None):
    """frame: for a panning-camera model, the calibrated frame (its image path as in the config) whose camera is
    wanted; None = the reference frame. A stationary-camera model has one camera for every frame."""
    js = json.load(open(path, encoding="utf-8"))
    src = js
    fc = js.get('frame_cameras') or {}
    if frame is not None and fc:
        if frame not in fc:
            raise KeyError(f'{frame} is not one of the calibrated frames of this panning-camera model: add its '
                           f'markings and calibrate again (calibrated: {sorted(fc)})')
        src = fc[frame]
    return dict(cam=_cam_from_json(src['camera']), draws=[_cam_from_json(d) for d in src['draws']],
                half_length=js['half_length'], half_width=js['half_width'],
                frame_size=tuple(src.get('frame_size', js['frame_size'])), mode=js.get('mode', 'fixed'),
                frame_cameras=sorted(fc), meta=js.get('meta', {}), diagnostics=js.get('diagnostics', {}),
                raw=js)
