"""Pitch template (IFAB Law 1 markings) parameterised by the pitch half-length and half-width.

Every marking that can be used for calibration is a named *element*:
  * straight lines  -> (fixed axis, value, (lo, hi) range of the other axis)
  * circles / arcs  -> (centre, radius, optional angular validity test)
  * points          -> (x, y)
"Left"/"right" refer to the goals on the left/right of the main broadcast camera (−X / +X),
"far"/"near" to the touchline away from / close to that camera (+Y / −Y).
"""
import numpy as np

R_CIRCLE = 9.15          # centre circle and penalty arcs
PA_DEPTH, PA_HALF = 16.5, 20.16   # penalty area: 16.5 m deep, 40.32 m wide
GA_DEPTH, GA_HALF = 5.5, 9.16     # goal area: 5.5 m deep, 18.32 m wide
PEN_SPOT = 11.0
DEFAULT_HALF_LENGTH, DEFAULT_HALF_WIDTH = 52.5, 34.0


def line_elements(hl, hw):
    E = {
        'halfway_line': ('X', 0.0, (-hw, hw)),
        'touchline_far': ('Y', hw, (-hl, hl)),
        'touchline_near': ('Y', -hw, (-hl, hl)),
        'goal_line_left': ('X', -hl, (-hw, hw)),
        'goal_line_right': ('X', hl, (-hw, hw)),
    }
    for side, s in (('left', -1), ('right', 1)):
        g = s * hl
        E[f'penalty_area_{side}_front'] = ('X', g - s * PA_DEPTH, (-PA_HALF, PA_HALF))
        E[f'penalty_area_{side}_far'] = ('Y', PA_HALF, tuple(sorted((g, g - s * PA_DEPTH))))
        E[f'penalty_area_{side}_near'] = ('Y', -PA_HALF, tuple(sorted((g, g - s * PA_DEPTH))))
        E[f'goal_area_{side}_front'] = ('X', g - s * GA_DEPTH, (-GA_HALF, GA_HALF))
        E[f'goal_area_{side}_far'] = ('Y', GA_HALF, tuple(sorted((g, g - s * GA_DEPTH))))
        E[f'goal_area_{side}_near'] = ('Y', -GA_HALF, tuple(sorted((g, g - s * GA_DEPTH))))
    return E


def circle_elements(hl, hw):
    E = {'centre_circle': ((0.0, 0.0), R_CIRCLE)}
    for side, s in (('left', -1), ('right', 1)):
        E[f'penalty_arc_{side}'] = ((s * (hl - PEN_SPOT), 0.0), R_CIRCLE)
    return E


def point_elements(hl, hw):
    E = {'centre_spot': (0.0, 0.0), 'halfway_far': (0.0, hw), 'halfway_near': (0.0, -hw)}
    for side, s in (('left', -1), ('right', 1)):
        E[f'penalty_spot_{side}'] = (s * (hl - PEN_SPOT), 0.0)
        E[f'corner_far_{side}'] = (s * hl, hw)
        E[f'corner_near_{side}'] = (s * hl, -hw)
    return E


def kind_of(name):
    if name in line_elements(1, 1):
        return 'line'
    if name in circle_elements(1, 1):
        return 'circle'
    if name in point_elements(1, 1):
        return 'point'
    raise KeyError(f'unknown pitch element {name!r}; valid: {sorted(all_element_names())}')


def all_element_names():
    return list(line_elements(1, 1)) + list(circle_elements(1, 1)) + list(point_elements(1, 1))


def polyline(name, hl=DEFAULT_HALF_LENGTH, hw=DEFAULT_HALF_WIDTH, n=200):
    """World polyline (N,3) of an element — used for drawing and for synthetic tests."""
    k = kind_of(name)
    if k == 'line':
        axis, v, (lo, hi) = line_elements(hl, hw)[name]
        t = np.linspace(lo, hi, n)
        return np.c_[np.full(n, v), t, 0 * t] if axis == 'X' else np.c_[t, np.full(n, v), 0 * t]
    if k == 'circle':
        (cx, cy), r = circle_elements(hl, hw)[name]
        a = np.linspace(-np.pi, np.pi, n) if name == 'penalty_arc_left' else np.linspace(0, 2 * np.pi, n)
        P = np.c_[cx + r * np.cos(a), cy + r * np.sin(a), 0 * a]
        if name.startswith('penalty_arc'):                     # only the part outside the penalty area
            P = P[np.abs(P[:, 0]) < hl - PA_DEPTH]
        return P
    x, y = point_elements(hl, hw)[name]
    return np.array([[x, y, 0.0]])


def anchor(name, hl=DEFAULT_HALF_LENGTH, hw=DEFAULT_HALF_WIDTH):
    return polyline(name, hl, hw).mean(0)
