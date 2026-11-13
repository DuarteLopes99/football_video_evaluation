"""Synthetic check that auxiliary frames improve the decision-frame calibration.

Ground truth: 105 x 68 m pitch, main camera 60 m from the halfway line, 10.5 m high. The centre circle
is painted 1.3 % out of round (like real pitches). The decision frame sees only the halfway line, the
far touchline and the centre circle (exactly like the Anadia frame); the auxiliary frame is the same
camera panned/zoomed onto the right penalty area.

Typical output (2026-10, this repo):
  [perfect circle]            single frame: player X error up to ~9.7 m   multi frame: ~0.04 m
  [circle 1.3 % out of round] single frame: ~10.4 m                        multi frame: ~0.9 m
i.e. halfway line + far touchline + centre circle alone leave the perspective almost undetermined
(the cost valley is nearly flat); one extra frame of the same camera fixes it.

    python tests/test_multiframe_synthetic.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from offside import Frame, calibrate, extract_features, ground, vanishing_point          # noqa: E402
from offside.synthetic import camera_looking_at, render_frame, hint_specs, visible_elements  # noqa: E402

SIZE = (2930, 1578)
HL, HW = 52.5, 34.0
C = np.array([0.0, -60.0, 10.5])
CIRCLE_ERR = (0.013, 0.7)


def make_frames(seed=0, circle_err=CIRCLE_ERR):
    dec = camera_looking_at(C, (2.0, 6.0, 0), 5000, SIZE)
    aux = camera_looking_at(C, (40.0, 6.0, 0), 6500, SIZE)
    frames, cams, imgs = [], {}, {}
    for name, cam, elems in [('decision', dec, ['halfway_line', 'touchline_far', 'centre_circle']),
                             ('aux_penalty_right', aux, None)]:
        img = render_frame(cam, SIZE, HL, HW, circle_err, seed=seed)
        elems = elems or [e for e in visible_elements(cam, SIZE, HL, HW) if e != 'centre_circle']
        feats = extract_features(img, hint_specs(cam, SIZE, elems, HL, HW, seed=seed))
        bottom = np.c_[np.linspace(0, SIZE[0] - 1, 9), np.full(9, SIZE[1] - 15.0)]
        g = ground(bottom, cam)
        inside = bottom[(np.abs(g[:, 1]) < HW) & (np.abs(g[:, 0]) < HL)]
        frames.append(Frame(name, SIZE, feats, 'main', inside if len(inside) else None))
        cams[name], imgs[name] = cam, img
    return frames, cams, imgs


def errors(cal, gt):
    cam = cal['cams']['decision']
    probe = np.array([[2600.0, 1000.0], [2300.0, 300.0], [700.0, 900.0]])     # where players would stand
    dx = np.abs(ground(probe, cam)[:, 0] - ground(probe, gt)[:, 0])
    vp = vanishing_point(cam, (0, 1, 0))[1] - vanishing_point(gt, (0, 1, 0))[1]
    return dict(vp_row_err=float(vp), max_X_err_m=float(dx.max()), C_err_m=float(np.linalg.norm(cam['C'] - gt['C'])),
                f_err_pct=float(100 * (cam['f'] / gt['f'] - 1)))


def main():
    for label, ce in [('perfect circle', (0.0, 0.0)), ('circle 1.3 % out of round', CIRCLE_ERR)]:
        frames, cams, _ = make_frames(circle_err=ce)
        single = calibrate(frames[:1])
        multi = calibrate(frames)
        es, em = errors(single, cams['decision']), errors(multi, cams['decision'])
        print(f'[{label}]')
        print('   single frame :', {k: round(v, 3) for k, v in es.items()})
        print('   multi frame  :', {k: round(v, 3) for k, v in em.items()})
        if ce == (0.0, 0.0):
            assert em['max_X_err_m'] < 0.10, 'perfect data: multi-frame should localise players to < 10 cm'
        assert em['max_X_err_m'] < es['max_X_err_m'] / 5, 'auxiliary frame should cut the error > 5x'
    print('OK')


def test_multiframe_synthetic():                 # pytest entry point
    main()


if __name__ == '__main__':
    main()
