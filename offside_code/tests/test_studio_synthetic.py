"""Synthetic check of the dashboard's Offside tab back end (offside/studio.py) and of the panning-camera model.

1. calibrate_panning_camera + save/load_camera_model: one camera per calibrated frame, each close to the truth,
   and a clear error for a frame that was not calibrated.
2. Studio venue workflow without the browser: create a venue, add frames, send clicked markings, "Check" finds the
   white-line pixels, venue.json compiles to the CLI format (decision_frame / aux_frames / camera), the re-projected
   template of the saved model lands on the painted lines.

No detector needed (players are not used here). Runs in ~1 min:

    python tests/test_studio_synthetic.py
"""
import json
import os
import shutil
import sys
import tempfile

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from offside import ground                                                             # noqa: E402
from offside.synthetic import camera_looking_at, render_frame, hint_specs, visible_elements  # noqa: E402
from offside.venue import calibrate_panning_camera, save_camera_model, load_camera_model  # noqa: E402
from offside import studio as S                                                        # noqa: E402

SIZE = (2930, 1578)
HL, HW = 52.5, 34.0
C = np.array([0.0, -60.0, 10.5])


def synthetic_frames():
    dec = camera_looking_at(C, (2.0, 6.0, 0), 5000, SIZE)
    aux = camera_looking_at(C, (40.0, 6.0, 0), 6500, SIZE)
    out = []
    for name, cam, elems in [('decision', dec, ['halfway_line', 'touchline_far', 'centre_circle']), ('aux', aux, None)]:
        img = render_frame(cam, SIZE, HL, HW, (0.0, 0.0), seed=0)
        elems = elems or [e for e in visible_elements(cam, SIZE, HL, HW) if e != 'centre_circle']
        specs = hint_specs(cam, SIZE, elems, HL, HW, seed=0)
        bottom = np.c_[np.linspace(0, SIZE[0] - 1, 9), np.full(9, SIZE[1] - 15.0)]
        g = ground(bottom, cam)
        inside = bottom[(np.abs(g[:, 1]) < HW) & (np.abs(g[:, 0]) < HL)].round().tolist()
        out.append(dict(name=name, cam=cam, img=img, cfg=dict(features=specs, inside_pitch=inside)))
    return out


def x_error(cam, gt):
    probe = np.array([[2600.0, 1000.0], [2300.0, 300.0], [700.0, 900.0]])
    return float(np.abs(ground(probe, cam)[:, 0] - ground(probe, gt)[:, 0]).max())


def check_panning_model(fr, tmp):
    names = [f['name'] + '.png' for f in fr]
    res = calibrate_panning_camera([f['img'] for f in fr], [f['cfg'] for f in fr], names, detector=None,
                                   n_draws=3, verbose=False)
    path = os.path.join(tmp, 'model.json')
    save_camera_model(path, res, SIZE, dict(frames=names))
    for f, n in zip(fr, names):
        m = load_camera_model(path, n)
        err = x_error(m['cam'], f['cam'])
        print(f'  panning model, frame {n}: max player-X error {err:.2f} m, {len(m["draws"])} draws')
        assert err < 0.5, f'{n}: calibrated camera too far from the truth ({err:.2f} m)'
        assert m['frame_size'] == SIZE and m['mode'] == 'panning'
    try:
        load_camera_model(path, 'not_calibrated.png')
        raise AssertionError('an uncalibrated frame must be refused')
    except KeyError:
        pass
    assert load_camera_model(path)['cam']['f'] == load_camera_model(path, names[0])['cam']['f']   # default = reference
    js = json.load(open(path))
    assert isinstance(js['frames'], list) and set(js['frame_cameras']) == set(names)
    return path


def check_studio(fr, model_path):
    st = S.Studio()
    name = f'selftest_{os.getpid()}'
    vdir = st._vdir(name)
    try:
        st.create(name, mode='panning')
        img_dir = vdir / 'images'
        rels = []
        for f in fr:
            p = img_dir / (f['name'] + '.png')
            cv2.imwrite(str(p), f['img'])
            rels.append(S._rel(p))
        st.add_existing(name, rels)
        v = st.venue(name)
        assert v['reference'] == rels[0] and len(v['frames']) == 2 and v['model'] is None
        # the clicked markings, as the page sends them
        v = st.update(name, dict(hints={r: dict(features=f['cfg']['features'], inside_pitch=f['cfg']['inside_pitch'])
                                        for r, f in zip(rels, fr)}))
        assert [x['n_features'] for x in v['frames']] == [len(f['cfg']['features']) for f in fr]
        chk = st.check(name, rels[1])
        print('  check (aux frame):', chk['counts'])
        assert all(n > 40 for n in chk['counts'].values()), chk['counts']
        cfg = json.loads((vdir / 'venue.json').read_text())
        assert cfg['camera']['mode'] == 'panning' and cfg['decision_frame']['image'] == rels[0]
        assert [a['image'] for a in cfg['aux_frames']] == [rels[1]]
        # pretend the calibration job ran: the model with this venue's frame names
        js = json.load(open(model_path))
        js['frame_cameras'] = {r: js['frame_cameras'][f['name'] + '.png'] for r, f in zip(rels, fr)}
        (vdir / 'camera_model.json').write_text(json.dumps(js))
        v = st.venue(name)
        assert all(x['judgeable'] for x in v['frames']), 'both calibrated frames must be judgeable'
        rp = st.reprojection(name, rels[1])
        # the re-projected template must sit on the painted (synthetic) lines of the aux frame
        mask = cv2.cvtColor(fr[1]['img'], cv2.COLOR_BGR2GRAY) > 200
        dist = cv2.distanceTransform((~mask).astype(np.uint8), cv2.DIST_L2, 3)
        d = []
        for segs in rp['elements'].values():
            for s in segs:
                for x, y in s:
                    if 0 <= x < SIZE[0] and 0 <= y < SIZE[1]:
                        d.append(dist[int(y), int(x)])
        print(f'  re-projection on the aux frame: median {np.median(d):.1f} px from a painted line ({len(d)} points)')
        assert np.median(d) < 3.0
        sample = st.sample(name, rels[0], 1000, 1200)
        assert sample['rules'] and sample['rules'][0]['region'] == 'torso'
    finally:
        shutil.rmtree(vdir, ignore_errors=True)


def main():
    frames = synthetic_frames()
    tmp = tempfile.mkdtemp()
    try:
        mp = check_panning_model(frames, tmp)
        check_studio(frames, mp)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print('OK')


def test_studio_synthetic():          # pytest entry point (the checks take arguments, so they are not tests)
    main()


if __name__ == '__main__':
    main()
