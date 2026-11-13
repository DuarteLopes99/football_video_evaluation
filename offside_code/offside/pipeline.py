"""One-shot pipeline: calibrate on the given frame(s) and judge the decision frame in one go.

For the stationary-camera workflow (calibrate once, judge many plays fast) use calibrate_camera.py +
offside.predict.OffsidePredictor instead. This module is kept for one-off frames and for cameras that
pan/zoom (`aux_frames` sharing the camera centre)."""
import json
import os
import time

import cv2
import numpy as np

from .features import extract_features, find_candidates, features_from_candidates
from .calibration import Frame, Priors, calibrate
from .players import PlayerDetector, assign_teams, calibration_people
from .decision import decide, monte_carlo, summarise, analyse, scenario_name
from .render import render_scenario, render_birds_eye


def _p(base, path):
    return path if os.path.isabs(path) else os.path.join(base, path)


def load_frame(fc, base, name, rng):
    img = cv2.imread(_p(base, fc['image']))
    if img is None:
        raise FileNotFoundError(_p(base, fc['image']))
    ign = fc.get('ignore_regions', [])
    if 'features' in fc:
        feats = extract_features(img, fc['features'], ign, rng)
    elif 'candidates_mapping' in fc:
        feats = features_from_candidates(find_candidates(img, ign, rng=rng), fc['candidates_mapping'])
    else:
        raise ValueError(f'{name}: give "features" (hints) or "candidates_mapping"')
    frame = Frame(name=name, size=(img.shape[1], img.shape[0]), features=feats, group=fc.get('group', 'main'),
                  inside_pitch=np.array(fc['inside_pitch'], float) if fc.get('inside_pitch') else None)
    return img, frame


def priors_from(cfg):
    p = cfg.get('priors', {})
    return Priors(half_length=tuple(p.get('half_length', (52.5, 2.5))), half_width=tuple(p.get('half_width', (34.0, 2.0))),
                  shoulder_h=tuple(p.get('shoulder_h', (1.40, 0.25))), pp_sigma=p.get('pp_sigma', 40.0))


def run(cfg, base_dir='.', verbose=True, detector=None):
    t0 = time.time()
    log = print if verbose else (lambda *a, **k: None)
    rng = np.random.default_rng(cfg.get('seed', 0))
    out_dir = _p(base_dir, cfg.get('output_dir', 'outputs'))
    os.makedirs(out_dir, exist_ok=True)

    # 1. features in every frame ---------------------------------------------------------------
    dfc = cfg['decision_frame']
    img, dframe = load_frame(dfc, base_dir, 'decision', rng)
    frames = [dframe]
    for k, afc in enumerate(cfg.get('aux_frames', [])):
        frames.append(load_frame(afc, base_dir, afc.get('name', f'aux{k}'), rng)[1])
    for f in frames:
        log(f'[features] {f.name}: ' + ', '.join(f'{n}={len(P)}' for n, P in f.features.items()))

    # 2. players (needed as a weak calibration cue as well) ---------------------------------------
    det = detector or PlayerDetector(_p(base_dir, cfg.get('models_dir', 'models')))
    players = det(img, dfc.get('ignore_regions', []), dfc.get('occluding_regions', []), dfc.get('player_min_y', 0))
    players = assign_teams(img, players, cfg['teams'], cfg.get('team_overrides'))
    log('[players] ' + ', '.join(f"#{q['id']}:{q['team']}" + ('*' if q['occluded'] else '') for q in players))
    dframe.people = calibration_people(players)

    # 3. calibration -------------------------------------------------------------------------------
    priors = priors_from(cfg)
    calib = calibrate(frames, priors, rng)
    cam = calib['cams']['decision']
    log(f"[calibration] pitch {2 * calib['half_length']:.1f} x {2 * calib['half_width']:.1f} m, "
        f"cost {calib['cost']:.0f}")
    for n, d in calib['diagnostics'].items():
        log(f"   {n}: f={d['f_px']:.0f}px C={d['camera_xyz_m']} vp_goal_dir={d['goal_line_vanishing_point']} "
            f"rms={ {k: round(v, 2) for k, v in d['rms_px'].items()} }")

    # 4. decisions + Monte-Carlo -------------------------------------------------------------------
    scen = cfg['scenarios']
    res, ball, base = decide(players, cam, dfc.get('ball_pixel'), scen)
    mc = cfg.get('monte_carlo', {})
    draws = monte_carlo(frames, calib, players, dfc.get('ball_pixel'), scen, n=mc.get('n', 150),
                        priors=priors, seed=mc.get('seed', 1), progress=verbose)
    summary, mc_diag = summarise(base, draws)
    log(f"[monte-carlo] goal-line vanishing point row 90% CI {np.round(mc_diag['vp_row_ci90'])}")
    for name, s in summary.items():
        log(f"  {name}:")
        for i, a in sorted(s['attackers'].items(), key=lambda t: -t[1]['margin_m'])[:3]:
            log(f"     attacker #{i}: {a['decision']}  (P(offside)={a['p_offside']:.2f})")

    # 5. images ------------------------------------------------------------------------------------
    teams = [t['team'] for t in cfg['teams'] if t['team'] not in ('REF',)]
    files = []
    hl, hw = calib['half_length'], calib['half_width']
    for sc in scen:
        name = scenario_name(sc)
        im = render_scenario(img, cam, hl, hw, res, summary[name], sc['attacking'], sc['defending'], sc['direction'],
                             cfg.get('draw_elements', list(dframe.features)), cfg.get('show_metrics', False))
        fn = os.path.join(out_dir, f"offside_{sc['attacking']}_attacks_{sc['direction']}.png")
        cv2.imwrite(fn, im)
        files.append(fn)
    bev = render_birds_eye(cam, hl, hw, res, ball, summary, dframe.size, teams)
    fn = os.path.join(out_dir, 'birds_eye.png')
    cv2.imwrite(fn, bev)
    files.append(fn)

    result = dict(
        calibration=dict(pitch_length_m=2 * hl, pitch_width_m=2 * hw, frames=calib['diagnostics'], **mc_diag),
        ball_XY=None if ball is None else ball[:2].tolist(),
        players={q['id']: dict(team=q['team'], box=[round(float(v)) for v in q['box']], occluded=q['occluded'],
                               **({k: v for k, v in res[q['id']].items() if k in ('xmin', 'xmax', 'y')}
                                  if q['id'] in res else {})) for q in players},
        scenarios=summary, files=files, seconds=round(time.time() - t0, 1))
    with open(os.path.join(out_dir, 'offside_result.json'), 'w', encoding="utf-8") as fh:
        json.dump(result, fh, indent=1, default=lambda v: float(v) if np.isscalar(v) else str(v))
    log(f'[done] {len(files)} images in {out_dir} ({result["seconds"]} s)')
    return dict(result=result, calib=calib, players=players, res=res, ball=ball, frames=frames, image=img)
