"""Offside geometry: lift body keypoints to 3-D, find the offside line, decide, propagate uncertainty.

Law 11 (IFAB): a player is in an offside position if any part of the head, body or feet is in the
opponents' half (excluding the halfway line) and nearer to the opponents' goal line than both the
ball and the second-last opponent. Hands and arms (up to the bottom of the armpit) do not count.
"""
import numpy as np

from .camera import rays, ground
from .calibration import calibrate, Priors

# BlazePose indices that count for offside: head (0-10), shoulders (11-12), hips, legs, feet (23-32)
ALLOWED = list(range(0, 13)) + list(range(23, 33))


def lift_player(player, cam, rng=None, kp_sigma=0.0, depth_sigma=0.0, ankle_h=0.07):
    """3-D keypoints: each keypoint's viewing ray intersected with the vertical plane that contains the
    support foot and faces the camera. Returns (len(ALLOWED), 3) world points."""
    kp = player['kp'][ALLOWED]
    if kp_sigma:
        kp = kp + rng.normal(0, kp_sigma, kp.shape)
    v = rays(player['foot'][None], cam)[0]
    root = cam['C'] + (ankle_h - cam['C'][2]) / v[2] * v
    root[2] = 0
    n = root - cam['C']
    n[2] = 0
    n /= np.linalg.norm(n)
    if depth_sigma:
        root = root + n * rng.normal(0, depth_sigma)
    r = rays(kp, cam)
    s = ((root - cam['C']) @ n) / (r @ n)
    return cam['C'] + s[:, None] * r


def analyse(players, cam, ball_px, rng=None, kp_sigma=0.0, depth_sigma=0.0):
    res = {}
    for q in players:
        if q['team'] in ('REF', 'UNKNOWN'):
            continue
        X3 = lift_player(q, cam, rng, kp_sigma, depth_sigma)
        res[q['id']] = dict(team=q['team'], X3=X3, xmax=float(X3[:, 0].max()), xmin=float(X3[:, 0].min()),
                            kx=int(X3[:, 0].argmax()), kn=int(X3[:, 0].argmin()), y=float(np.median(X3[:, 1])))
    ball = ground(np.asarray(ball_px, float)[None], cam)[0] if ball_px is not None else None
    return res, ball


def offside_line(res, ball, attacking, defending, sign, gk_visible=False):
    """sign=+1: attackers attack the +X (right) goal. Returns (defender X, line X, {attacker id: margin m}),
    margin > 0 means in an offside position. With the goalkeeper off-frame, the deepest visible
    defending player is taken as the second-last opponent."""
    key = 'xmax' if sign > 0 else 'xmin'
    defs = sorted((r[key] * sign for r in res.values() if r['team'] == defending), reverse=True)
    if not defs:
        return None, None, {}
    xd = defs[1] if gk_visible and len(defs) > 1 else defs[0]
    line = max(xd, (ball[0] * sign) if ball is not None else -np.inf, 0.0)
    margins = {i: r[key] * sign - line for i, r in res.items() if r['team'] == attacking}
    return xd * sign, line * sign, margins


def decide(players, cam, ball_px, scenarios):
    res, ball = analyse(players, cam, ball_px)
    out = {}
    for sc in scenarios:
        sign = 1 if sc['direction'] == 'right' else -1
        xd, line, m = offside_line(res, ball, sc['attacking'], sc['defending'], sign, sc.get('gk_visible', False))
        out[scenario_name(sc)] = dict(defender_X=xd, line_X=line, margins=m)
    return res, ball, out


def scenario_name(sc):
    return f"{sc['attacking']} attacks {sc['direction']}"


def monte_carlo(frames, calib, players, ball_px, scenarios, n=150, priors=None, seed=1,
                circle_sigma=(0.005, 0.008, 0.008), pp_sigma=40.0, kp_sigma=2.0, depth_sigma=0.35,
                width_sigma=2.5, length_sigma=3.0, progress=False):
    """Propagate every uncertain input through calibration + lifting + decision.
    Perturbs: feature pixels (bootstrap), pitch dimensions, principal point, shoulder-height prior,
    centre-circle painting (2nd/3rd harmonic radius errors), keypoints, body depth."""
    rng = np.random.default_rng(seed)
    priors = priors or Priors()
    ref = frames[0].name
    draws = []
    for it in range(n):
        pr = Priors(half_length=(rng.normal(priors.half_length[0], length_sigma), priors.half_length[1]),
                    half_width=(rng.normal(priors.half_width[0], width_sigma), priors.half_width[1]),
                    shoulder_h=(rng.normal(priors.shoulder_h[0], 0.08), priors.shoulder_h[1]),
                    pp_sigma=priors.pp_sigma,
                    pp_offset=(rng.normal(0, pp_sigma), rng.normal(0, pp_sigma)))
        perturb = dict(circle=(rng.normal(0, circle_sigma[0]), rng.normal(0, circle_sigma[1]), rng.uniform(0, 2 * np.pi),
                               rng.normal(0, circle_sigma[2]), rng.uniform(0, 2 * np.pi)))
        cb = calibrate(frames, pr, rng, x0=calib['params'], perturb=perturb, boot=True, max_nfev=1500)
        cam = cb['cams'][ref]
        res, ball = analyse(players, cam, ball_px, rng, kp_sigma, depth_sigma)
        d = {}
        for sc in scenarios:
            sign = 1 if sc['direction'] == 'right' else -1
            d[scenario_name(sc)] = offside_line(res, ball, sc['attacking'], sc['defending'], sign, sc.get('gk_visible', False))
        draws.append(dict(lines=d, vp=cb['diagnostics'][ref]['goal_line_vanishing_point'], f=cam['f']))
        if progress and (it + 1) % 25 == 0:
            print(f'  MC {it + 1}/{n}')
    return draws


def summarise(base, draws):
    """P(offside position) and 90 % intervals per scenario and attacker."""
    summary = {}
    for name, b in base.items():
        lines = np.array([d['lines'][name][1] for d in draws])
        att = {}
        for i, m in b['margins'].items():
            mb = np.array([d['lines'][name][2][i] for d in draws])
            p = float((mb > 0).mean())
            att[i] = dict(margin_m=float(m), p_offside=p, ci90_m=np.percentile(mb, [5, 95]).tolist(),
                          decision='OFFSIDE' if m > 0 else 'ONSIDE')
        summary[name] = dict(defender_X=b['defender_X'], line_X=b['line_X'],
                             line_ci90=np.percentile(lines, [5, 95]).tolist(), attackers=att)
    vps = np.array([d['vp'][1] for d in draws])
    return summary, dict(vp_row_ci90=np.percentile(vps, [5, 95]).tolist(),
                         f_ci90=np.percentile([d['f'] for d in draws], [5, 95]).tolist())
