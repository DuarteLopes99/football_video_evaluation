"""Pinhole camera model used everywhere in the package.

World frame (metres): origin at the centre spot, X along the touchlines (positive towards the
goal on the right of the main camera), Y across the pitch (positive towards the far touchline),
Z up. A camera is a dict  {f, R, C, cx, cy}  with  x ~ K R (X - C),  K = [[f,0,cx],[0,f,cy],[0,0,1]]
(square pixels, zero skew, no lens distortion).
"""
import cv2
import numpy as np


def rodrigues(rvec):
    return cv2.Rodrigues(np.asarray(rvec, float).reshape(3, 1))[0]


def make_camera(f, rvec, C, cx, cy):
    return dict(f=float(f), R=rodrigues(rvec), C=np.asarray(C, float), cx=float(cx), cy=float(cy))


def look_at_rvec(C, target):
    """Rotation vector of a camera at C looking at `target` with no roll (image y points down)."""
    z = np.asarray(target, float) - np.asarray(C, float)
    z /= np.linalg.norm(z)
    x = np.cross(z, [0, 0, 1.0])
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    return cv2.Rodrigues(np.vstack([x, y, z]))[0].ravel()


def project(X, cam):
    """World points (N,3) -> pixels (N,2)."""
    Xc = (np.atleast_2d(X) - cam['C']) @ cam['R'].T
    return np.c_[cam['f'] * Xc[:, 0] / Xc[:, 2] + cam['cx'], cam['f'] * Xc[:, 1] / Xc[:, 2] + cam['cy']]


def rays(P, cam):
    """Pixels (N,2) -> world-frame viewing-ray directions (N,3)."""
    P = np.atleast_2d(P)
    return np.c_[(P[:, 0] - cam['cx']) / cam['f'], (P[:, 1] - cam['cy']) / cam['f'], np.ones(len(P))] @ cam['R']


def ground(P, cam, z=0.0):
    """Back-project pixels onto the horizontal plane Z = z."""
    v = rays(P, cam)
    t = (z - cam['C'][2]) / v[:, 2]
    return cam['C'] + t[:, None] * v


def vanishing_point(cam, direction):
    """Image of the point at infinity in `direction` (e.g. (0,1,0): direction of the goal lines)."""
    d = cam['R'] @ np.asarray(direction, float)
    if abs(d[2]) < 1e-12:
        return np.array([np.inf, np.inf])
    return np.array([cam['f'] * d[0] / d[2] + cam['cx'], cam['f'] * d[1] / d[2] + cam['cy']])


def homography(cam):
    """Pitch-plane (X, Y, 1) -> image homography H = K [r1 r2 -R C]."""
    K = np.array([[cam['f'], 0, cam['cx']], [0, cam['f'], cam['cy']], [0, 0, 1.0]])
    R = cam['R']
    return K @ np.c_[R[:, 0], R[:, 1], -R @ cam['C']]
