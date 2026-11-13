"""Fast per-play judgement with a pre-calibrated camera.

    pred = OffsidePredictor('calibration/anadia_camera.json', cfg)
    out = pred.predict(cv2.imread('play.png'))          # ball found automatically, or pass ball_pixel=(x, y)
    out['decisions']  ->  {'AFC attacks right': {14: 'OFFSIDE', 16: 'ONSIDE', ...}, ...}
    out['timings']    ->  seconds per stage
    imgs = pred.render(image, out, show_lines=False, show_metrics=True)   # re-draw without re-computing

Only detection + pose + geometry run per image. The uncertainty uses the camera draws stored in the
camera model (computed once at calibration time), so it costs milliseconds instead of re-calibrating.
For a panning-camera model pass ``frame=`` (the calibrated frame's image path as in the config).
"""
import copy
import time

import numpy as np

from .players import PlayerDetector, assign_teams
from .decision import analyse, offside_line, scenario_name, summarise
from .render import render_scenario, render_birds_eye
from .venue import load_camera_model
from .camera import vanishing_point


class OffsidePredictor:
    def __init__(self, camera_model, cfg, models_dir='models', detector=None, frame=None):
        self.model = load_camera_model(camera_model, frame) if isinstance(camera_model, str) else camera_model
        self.cfg = cfg
        fc = cfg['decision_frame']
        self.fc = fc
        self.det = detector or PlayerDetector(models_dir, **cfg.get('detector', {}))

    def detect(self, image, frame_cfg=None):
        """Detection + pose (≈ 90 % of the time). The result can be passed back to predict(detections=...) to
        judge the same image again with other teams / ball / scenarios without re-running the networks."""
        fc = frame_cfg or self.fc
        t0 = time.perf_counter()
        boxes, scores = self.det.detect(image, fc.get('ignore_regions', []), fc.get('player_min_y', 0))
        t_det = time.perf_counter() - t0
        ball = self.det.last_ball
        t0 = time.perf_counter()
        players = self.det(image, fc.get('ignore_regions', []), fc.get('occluding_regions', []),
                           fc.get('player_min_y', 0), boxes=(boxes, scores))
        return dict(players=players, ball=ball, timings=dict(detection=t_det, pose=time.perf_counter() - t0))

    def predict(self, image, ball_pixel=None, scenarios=None, render=True, uncertainty=True,
                kp_sigma=2.0, depth_sigma=0.35, seed=0, frame_cfg=None, detections=None, auto_ball=True,
                show_lines=True, show_pitch=True, show_metrics=None):
        t = {}
        T0 = time.perf_counter()
        cfg, m = self.cfg, self.model
        H, W = image.shape[:2]
        if (W, H) != tuple(m['frame_size']):
            raise ValueError(f'image is {W}x{H} but the camera model was calibrated on {m["frame_size"]}')
        scen = scenarios or cfg['scenarios']

        if detections is None:
            detections = self.detect(image, frame_cfg)
            t.update(detections['timings'])
        else:
            t.update(detection=0.0, pose=0.0)
        players = copy.deepcopy(detections['players'])
        ball_src = 'given'
        if ball_pixel is None and auto_ball:                # automatic: YOLOX 'sports ball' class
            sb = detections['ball']
            if sb is not None and sb[0] >= cfg.get('ball_min_score', 0.15):
                ball_pixel, ball_src = np.asarray(sb[1], float).tolist(), f'detected (score {sb[0]:.2f})'
            else:
                ball_src = 'not found - ball ignored'
        elif ball_pixel is None:
            ball_src = 'not used'
        t0 = time.perf_counter()
        players = assign_teams(image, players, cfg['teams'], cfg.get('team_overrides'))
        t['teams'] = time.perf_counter() - t0

        t0 = time.perf_counter()
        cam = m['cam']
        res, ball = analyse(players, cam, ball_pixel)
        base, skipped = {}, {}
        for sc in scen:
            sign = 1 if sc['direction'] == 'right' else -1
            xd, line, mg = offside_line(res, ball, sc['attacking'], sc['defending'], sign, sc.get('gk_visible', False))
            name = scenario_name(sc)
            if line is None:
                skipped[name] = f"no {sc['defending']} player found in the picture"
                continue
            base[name] = dict(defender_X=xd, line_X=line, margins=mg)
        valid = [sc for sc in scen if scenario_name(sc) in base]
        t['geometry'] = time.perf_counter() - t0

        t0 = time.perf_counter()
        draws = []
        if uncertainty and m['draws'] and base:
            rng = np.random.default_rng(seed)
            for c in m['draws']:
                rb, bb = analyse(players, c, ball_pixel, rng, kp_sigma, depth_sigma)
                lines = {}
                for sc in valid:
                    sign = 1 if sc['direction'] == 'right' else -1
                    lines[scenario_name(sc)] = offside_line(rb, bb, sc['attacking'], sc['defending'], sign,
                                                            sc.get('gk_visible', False))
                if all(v[1] is not None for v in lines.values()):
                    draws.append(dict(lines=lines, vp=vanishing_point(c, (0, 1, 0)).tolist(), f=c['f']))
        if not draws:
            draws = [dict(lines={k: (b['defender_X'], b['line_X'], b['margins']) for k, b in base.items()},
                          vp=[0, 0], f=cam['f'])]
        summary, _ = summarise(base, draws) if base else ({}, None)
        t['uncertainty'] = time.perf_counter() - t0
        t['decision_total'] = time.perf_counter() - T0

        out = dict(summary=summary, players=players, res=res, ball=ball, scenarios=valid, skipped=skipped,
                   ball_pixel=ball_pixel, ball_source=ball_src, detections=detections)
        out['decisions'] = {k: {i: a['decision'] for i, a in s['attackers'].items()} for k, s in summary.items()}
        out['images'] = {}
        if render:
            t0 = time.perf_counter()
            out['images'] = self.render(image, out, show_lines=show_lines, show_pitch=show_pitch,
                                        show_metrics=show_metrics)
            t['render'] = time.perf_counter() - t0
        t['total'] = time.perf_counter() - T0
        out['timings'] = {k: round(v, 3) for k, v in t.items()}
        return out

    def render(self, image, out, show_lines=True, show_pitch=True, show_metrics=None, birds_eye=True,
               draw_elements=None):
        """Annotated images of a predict() result: {scenario name: image, 'birds_eye': image}."""
        cfg, m = self.cfg, self.model
        cam, hl, hw = m['cam'], m['half_length'], m['half_width']
        show_metrics = cfg.get('show_metrics', False) if show_metrics is None else show_metrics
        elements = cfg.get('draw_elements', []) if draw_elements is None else draw_elements
        images = {}
        for sc in out['scenarios']:
            name = scenario_name(sc)
            images[name] = render_scenario(image, cam, hl, hw, out['res'], out['summary'][name], sc['attacking'],
                                           sc['defending'], sc['direction'], elements, show_metrics,
                                           show_lines=show_lines, show_pitch=show_pitch)
        if birds_eye:
            teams = [x['team'] for x in cfg['teams'] if x['team'] != 'REF']
            teams = list(dict.fromkeys(teams))
            H, W = image.shape[:2]
            images['birds_eye'] = render_birds_eye(cam, hl, hw, out['res'], out['ball'], out['summary'], (W, H), teams)
        return images
