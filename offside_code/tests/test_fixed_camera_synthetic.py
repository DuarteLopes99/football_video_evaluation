"""Stationary camera: does pooling players from OTHER plays fix the calibration?

Ground truth as in the real frame: one fixed camera, only the halfway line, far touchline and a centre
circle painted 1.3 % out of round are visible. Every extra frame of another play adds ~15 players at
random places (heights N(1.80, 0.07) m; 30 % standing/walking, the rest running: crouched and leaning,
keypoint noise 2 px). No extra pitch markings — the view never changes.

    python tests/test_fixed_camera_synthetic.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from offside import Frame, calibrate, ground, project                         # noqa: E402
from tests.test_multiframe_synthetic import make_frames, errors, SIZE         # noqa: E402


def synth_people(cam, n, rng, upright_frac=0.3, kp_noise=2.0):
    W, H = SIZE
    px = np.c_[rng.uniform(40, W - 40, n), rng.uniform(320, H - 60, n)]
    G = ground(px, cam)
    out = []
    for g in G:
        h = rng.normal(1.80, 0.07)
        up = rng.random() < upright_frac
        if up:
            top = g + [rng.normal(0, 0.03), rng.normal(0, 0.03), 0.818 * h]
        else:   # running: lower shoulders, body leaning in a random direction
            d = rng.normal(0, 1, 2)
            d /= np.linalg.norm(d)
            top = g + [*(d * rng.uniform(0.05, 0.30)), 0.818 * h * rng.uniform(0.82, 0.97)]
        f, s = project(np.array([g, top]), cam)
        out.append(dict(foot=f + rng.normal(0, kp_noise, 2), shoulder=s + rng.normal(0, kp_noise, 2), upright=up))
    return out


def main():
    rng = np.random.default_rng(0)
    frames, cams, _ = make_frames()
    dec, gt = frames[0], cams['decision']
    print('n_frames  people  upright | max player-X error   vanishing-row error   focal error')
    results = {}
    for n_frames in (0, 1, 10, 40):
        people = []
        for _ in range(n_frames):
            people += synth_people(gt, 15, rng)
        fr = Frame('decision', dec.size, dec.features, 'main', dec.inside_pitch, people)
        e = errors(calibrate([fr]), gt)
        results[n_frames] = e
        print(f"{n_frames:8d} {len(people):7d} {sum(p['upright'] for p in people):8d} | {e['max_X_err_m']:10.2f} m"
              f" {e['vp_row_err']:17.0f} px {e['f_err_pct']:14.1f} %")
    assert results[40]['max_X_err_m'] < results[0]['max_X_err_m'] / 3, 'players from other plays should help'
    print('OK')


def test_fixed_camera_synthetic():                 # pytest entry point
    main()


if __name__ == '__main__':
    main()
