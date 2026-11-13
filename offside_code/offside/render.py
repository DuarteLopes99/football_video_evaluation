"""Drawing: offside lines on the broadcast frame and a bird's-eye reconstruction."""
import cv2
import numpy as np

from .camera import project, ground
from .template import polyline, line_elements, circle_elements

RED, YELLOW, GREEN, MAGENTA, WHITE = (0, 0, 255), (0, 255, 255), (80, 255, 80), (255, 0, 255), (255, 255, 255)
TEAM_BGR = {0: (255, 200, 60), 1: (60, 200, 255)}


def _seg(X, y0, y1, n=200):
    t = np.linspace(y0, y1, n)
    return np.c_[np.full(n, X), t, 0 * t]


def _poly(img, P, col, th):
    P = P[np.all(np.isfinite(P), axis=1) & (np.abs(P).max(1) < 1e5)]
    if len(P) > 1:
        cv2.polylines(img, [np.round(P * 16).astype(np.int32)], False, col, th, cv2.LINE_AA, 4)


def _label(img, txt, org, col, scale=0.8):
    H, W = img.shape[:2]
    (w, h), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_DUPLEX, scale, 2)
    x = int(min(max(org[0], 12), W - w - 12))
    y = int(min(max(org[1], h + 14), H - 14))
    cv2.rectangle(img, (x - 6, y - h - 8), (x + w + 6, y + 8), (20, 20, 20), -1)
    cv2.putText(img, txt, (x, y), cv2.FONT_HERSHEY_DUPLEX, scale, col, 2, cv2.LINE_AA)


def draw_pitch_model(img, cam, hl, hw, elements, col=MAGENTA):
    """Re-project the calibrated template (sanity check: it should sit on the painted lines)."""
    H = img.shape[0]
    y_bot = ground(np.array([[img.shape[1] / 2, H - 1.0]]), cam)[0][1] - 10
    for name in elements:
        P = polyline(name, hl, hw, 400)
        if name in line_elements(hl, hw) and line_elements(hl, hw)[name][0] == 'X':
            P = P[P[:, 1] > y_bot]
        _poly(img, project(P, cam), col, 2)


def render_scenario(img, cam, hl, hw, res, summary_sc, attacking, defending, direction,
                    pitch_elements=(), show_metrics=False, show_lines=True, show_pitch=True):
    """Annotated frame for one attacking direction.

    show_metrics=False: only the decisions are written (no distances / probabilities).
    show_lines=False:   no offside line, confidence band, defender "curtain" or attacker lines — only a dot on the
                        decisive body part and the OFFSIDE / ONSIDE labels.
    show_pitch=False:   the calibrated pitch template (magenta) is not drawn."""
    img = img.copy()
    H, W = img.shape[:2]
    sign = 1 if direction == 'right' else -1
    xk, key = ('xmax', 'kx') if sign > 0 else ('xmin', 'kn')
    y_bot = ground(np.array([[W / 2, H - 1.0]]), cam)[0][1] - 10

    def gline(X, col, th):
        _poly(img, project(_seg(X, y_bot, hw), cam), col, th)

    def drop(i, col):
        r = res[i]
        X = r['X3'][r[key]]
        a = project(X[None], cam)[0]
        b = project(np.r_[X[:2], 0][None], cam)[0]
        if show_lines:
            cv2.line(img, tuple(np.round(a).astype(int)), tuple(np.round(b).astype(int)), col, 2, cv2.LINE_AA)
        cv2.circle(img, tuple(np.round(a).astype(int)), 7, col, -1, cv2.LINE_AA)
        if show_lines:
            cv2.circle(img, tuple(np.round(b).astype(int)), 6, WHITE, 2, cv2.LINE_AA)

    if show_pitch:
        draw_pitch_model(img, cam, hl, hw, pitch_elements)
    lo, hi = summary_sc['line_ci90']
    line_X = summary_sc['line_X']
    if show_lines:
        A = project(_seg(lo, y_bot, hw, 60), cam)
        B = project(_seg(hi, hw, y_bot, 60), cam)
        ov = img.copy()
        cv2.fillPoly(ov, [np.round(np.r_[A, B]).astype(np.int32)], RED)
        cv2.addWeighted(ov, 0.22, img, 0.78, 0, img)
        gline(line_X, RED, 4)

    defenders = [i for i, r in res.items() if r['team'] == defending]
    d = max(defenders, key=lambda i: res[i][xk] * sign)
    if abs(res[d][xk] - line_X) < 1e-6 and show_lines:   # line set by a defender (not the ball / halfway line)
        drop(d, RED)
        Yc = res[d]['y']
        Q = project(np.array([[line_X, Yc - 2.5, 0], [line_X, Yc + 2.5, 0], [line_X, Yc + 2.5, 2.0],
                              [line_X, Yc - 2.5, 2.0]]), cam).round().astype(np.int32)
        ov = img.copy()
        cv2.fillPoly(ov, [Q], RED)
        cv2.addWeighted(ov, 0.18, img, 0.82, 0, img)
        cv2.polylines(img, [Q], True, RED, 2, cv2.LINE_AA)
        P = project(np.array([[line_X, Yc, 2.6]]), cam)[0]
        _label(img, '2nd-last defender', (P[0] - 120, P[1] - 10), RED)

    for i, a in summary_sc['attackers'].items():
        if a['margin_m'] < -3:                       # far behind the line: nothing to show
            continue
        off = a['margin_m'] > 0
        col = YELLOW if off else GREEN
        if show_lines:
            gline(res[i][xk], col, 2)
        drop(i, col)
        P = project(np.array([[res[i][xk], res[i]['y'], 2.3]]), cam)[0]
        txt = 'OFFSIDE' if off else 'ONSIDE'
        if show_metrics:
            txt += f"  {a['margin_m'] * 100:+.0f} cm  P={a['p_offside']:.2f}"
        _label(img, txt, (P[0] - 60, P[1] - 10), col, 0.9)

    arrow = 'right -->' if sign > 0 else '<-- left'
    banner = f'If {attacking} attack {arrow}   offside line through the 2nd-last defender'
    if not show_lines:
        banner = f'If {attacking} attack {arrow}'
    if show_metrics:
        banner = f"If {attacking} attack {arrow}   line X={line_X:+.2f} m  90% CI [{lo:+.2f}, {hi:+.2f}]"
    _label(img, banner, (40, H - 40), WHITE, 1.0)
    return img


def render_birds_eye(cam, hl, hw, res, ball, summary, frame_size, teams, scale=10):
    Lx, Ly = 60, hw + 3
    bev = np.full((int(2 * Ly * scale) + 80, int(2 * Lx * scale) + 80, 3), (60, 120, 60), np.uint8)
    tp = lambda X, Y: (int(40 + (X + Lx) * scale), int(40 + (Ly - Y) * scale))
    cv2.line(bev, tp(0, -hw), tp(0, hw), WHITE, 2)
    cv2.circle(bev, tp(0, 0), int(9.15 * scale), WHITE, 2)
    for yy in (hw, -hw):
        cv2.line(bev, tp(-Lx, yy), tp(Lx, yy), WHITE, 2)
    W, H = frame_size
    G = ground(np.array([[0, 0.14 * H], [W - 1, 0.14 * H], [W - 1, H - 1], [0, H - 1]], float), cam)[:, :2]
    cv2.polylines(bev, [np.array([tp(*g) for g in G], np.int32)], True, (210, 210, 210), 1, cv2.LINE_AA)
    col = {t: TEAM_BGR[k % 2] for k, t in enumerate(teams)}
    for i, r in res.items():
        c = col.get(r['team'], WHITE)
        cv2.line(bev, tp(r['xmin'], r['y']), tp(r['xmax'], r['y']), c, 4)
        cv2.circle(bev, tp((r['xmin'] + r['xmax']) / 2, r['y']), 9, c, -1)
        cv2.putText(bev, f'#{i}', tp(r['xmax'] + 0.6, r['y'] + 0.7), 0, 0.5, WHITE, 1, cv2.LINE_AA)
    if ball is not None:
        cv2.circle(bev, tp(*ball[:2]), 6, WHITE, -1)
    for s in summary.values():
        if s['line_X'] is not None:
            cv2.line(bev, tp(s['line_X'], -hw), tp(s['line_X'], hw), RED, 2, cv2.LINE_AA)
    names = ' / '.join(f'{t} {"blue" if k % 2 == 0 else "orange"}' for k, t in enumerate(teams))
    cv2.putText(bev, f'frame footprint (grey) / {names} / offside lines red', (50, bev.shape[0] - 15),
                0, 0.55, WHITE, 1, cv2.LINE_AA)
    return bev
