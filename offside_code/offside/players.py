"""Player detection (YOLOX, tiled), 33-keypoint body pose (MediaPipe BlazePose) and team assignment.

Both networks are the ONNX exports from the OpenCV Model Zoo and run with cv2.dnn (no PyTorch). When OpenCV was
built with CUDA and sees an NVIDIA GPU they run on it automatically (set OFFSIDE_DEVICE=cpu to force the CPU).
Swap in a stronger detector / pose model (RT-DETR, ViTPose, SAM-3D-Body...) by returning the same
player dicts: {'box', 'score', 'kp' (33,2 BlazePose order), 'conf'}.
"""
import os

import cv2
import numpy as np

KP_NAMES = ['nose', 'l_eye_in', 'l_eye', 'l_eye_out', 'r_eye_in', 'r_eye', 'r_eye_out', 'l_ear', 'r_ear',
            'mouth_l', 'mouth_r', 'l_shoulder', 'r_shoulder', 'l_elbow', 'r_elbow', 'l_wrist', 'r_wrist',
            'l_pinky', 'r_pinky', 'l_index', 'r_index', 'l_thumb', 'r_thumb', 'l_hip', 'r_hip', 'l_knee',
            'r_knee', 'l_ankle', 'r_ankle', 'l_heel', 'r_heel', 'l_foot_index', 'r_foot_index']


def _cuda_devices():
    try:
        return int(cv2.cuda.getCudaEnabledDeviceCount())
    except Exception:  # noqa: BLE001 — OpenCV without the cuda module
        return 0


def _to_gpu(net):
    """cv2.dnn network -> CUDA when available (and not disabled with OFFSIDE_DEVICE=cpu). Returns the device name."""
    if os.environ.get('OFFSIDE_DEVICE', 'auto').lower() == 'cpu' or _cuda_devices() == 0:
        return 'cpu'
    try:
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
        return 'cuda'
    except Exception:  # noqa: BLE001
        return 'cpu'


class PlayerDetector:
    def __init__(self, models_dir='models', input_size=640, tile=900, stride=600, score_thr=0.25):
        self.det = cv2.dnn.readNet(os.path.join(models_dir, 'yolox.onnx'))
        self.pose_net = cv2.dnn.readNet(os.path.join(models_dir, 'mp_pose.onnx'))
        self.device = _to_gpu(self.det)
        _to_gpu(self.pose_net)
        self.S, self.tile, self.stride, self.thr = input_size, tile, stride, score_thr
        g_, e_ = [], []
        for s in (8, 16, 32):
            g = input_size // s
            yv, xv = np.meshgrid(np.arange(g), np.arange(g), indexing='ij')
            g_.append(np.stack([xv, yv], -1).reshape(-1, 2))
            e_.append(np.full((g * g, 1), s))
        self.grids, self.strides = np.concatenate(g_), np.concatenate(e_)

    # ---------------------------------------------------------------- detection
    def _yolox(self, tile):
        S = self.S
        h, w = tile.shape[:2]
        r = min(S / h, S / w)
        pad = np.full((S, S, 3), 114, np.uint8)
        pad[:int(h * r), :int(w * r)] = cv2.resize(tile, (int(w * r), int(h * r)))
        self.det.setInput(cv2.dnn.blobFromImage(pad, 1.0, (S, S), swapRB=False))
        o = self.det.forward()[0].copy()
        o[:, :2] = (o[:, :2] + self.grids) * self.strides
        o[:, 2:4] = np.exp(o[:, 2:4]) * self.strides
        sc = o[:, 4] * o[:, 5]                     # objectness x P(person)
        k = sc > self.thr
        b = o[k]
        sb = o[:, 4] * o[:, 5 + 32]                # COCO class 32 = sports ball
        j = int(np.argmax(sb))
        ball = (float(sb[j]), np.array([o[j, 0], o[j, 1] + o[j, 3] / 2]) / r)   # bottom of the ball = contact point
        return np.c_[b[:, 0] - b[:, 2] / 2, b[:, 1] - b[:, 3] / 2, b[:, 0] + b[:, 2] / 2, b[:, 1] + b[:, 3] / 2] / r, sc[k], ball

    def detect(self, img, ignore_regions=(), min_y=0):
        H, W = img.shape[:2]
        T, ST = min(self.tile, H, W), self.stride
        boxes, scores = [], []
        y_first = int(min(max(0, min_y - 250), H - T))         # nothing to find above the far touchline
        ys = list(range(y_first, max(H - T, y_first), ST)) + [H - T]
        xs = list(range(0, max(W - T, 0), ST)) + [W - T]
        self.last_ball = None                          # (score, pixel) of the best ball candidate
        for y0 in ys:
            for x0 in xs:
                b, s, (bs, bp) = self._yolox(img[y0:y0 + T, x0:x0 + T])
                b += [x0, y0, x0, y0]
                if self.last_ball is None or bs > self.last_ball[0]:
                    self.last_ball = (bs, bp + [x0, y0])
                ok = (((b[:, 0] > x0 + 3) | (x0 == 0)) & ((b[:, 2] < x0 + T - 3) | (x0 + T == W)) &
                      ((b[:, 1] > y0 + 3) | (y0 == y_first)) & ((b[:, 3] < y0 + T - 3) | (y0 + T == H)))
                boxes.append(b[ok])
                scores.append(s[ok])
        boxes, scores = np.concatenate(boxes), np.concatenate(scores)
        keep = np.array(cv2.dnn.NMSBoxes([[float(a), float(b), float(c - a), float(d - b)] for a, b, c, d in boxes],
                                         scores.tolist(), self.thr, 0.45)).ravel().astype(int)
        boxes, scores = boxes[keep], scores[keep]
        sel = boxes[:, 3] > min_y
        for x0, y0, x1, y1 in ignore_regions:   # drop detections fully inside an overlay (score bug etc.)
            sel &= ~((boxes[:, 0] >= x0) & (boxes[:, 2] <= x1) & (boxes[:, 1] >= y0) & (boxes[:, 3] <= y1))
        return boxes[sel], scores[sel]

    # ---------------------------------------------------------------- pose
    def pose(self, img, box):
        cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
        side = 1.35 * max(box[3] - box[1], box[2] - box[0])
        s = 256 / side
        M = np.array([[s, 0, 128 - cx * s], [0, s, 128 - cy * s]])
        crop = cv2.cvtColor(cv2.warpAffine(img, M, (256, 256)), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.
        self.pose_net.setInput(crop[None])
        outs = self.pose_net.forward(self.pose_net.getUnconnectedOutLayersNames())
        lm = [o for o in outs if o.size == 195][0].reshape(39, 5)[:33]
        conf = float([o for o in outs if o.size == 1][0].ravel()[0])
        return (lm[:, :2] - [128 - cx * s, 128 - cy * s]) / s, conf

    def __call__(self, img, ignore_regions=(), occluding_regions=(), min_y=0, min_pose_conf=0.3, boxes=None):
        boxes, scores = boxes if boxes is not None else self.detect(img, ignore_regions, min_y)
        players = []
        for b, s in zip(boxes, scores):
            kp, conf = self.pose(img, b)
            if conf >= min_pose_conf:
                players.append(dict(box=b, score=float(s), kp=kp, conf=conf))
        players = dedupe(players)
        for k, q in enumerate(players):
            ank = q['kp'][[27, 28]]
            q['foot'] = ank[np.argmax(ank[:, 1])]                 # lower ankle = support foot
            q['shoulder'] = (q['kp'][11] + q['kp'][12]) / 2
            q['upright'] = is_upright(q['kp'])
            q['occluded'] = any(x0 < q['foot'][0] < x1 and y0 < q['foot'][1] < y1
                                for x0, y0, x1, y1 in list(occluding_regions) + list(ignore_regions))
            q['id'] = k
        return players


def _angle(a, b, c):
    u, v = a - b, c - b
    return np.degrees(np.arccos(np.clip(u @ v / (np.linalg.norm(u) * np.linalg.norm(v) + 1e-9), -1, 1)))


def is_upright(kp, knee_min=160.0, lean_max=0.15):
    """Standing / walking straight: both legs extended, hips over ankles, shoulders over hips.
    Such bodies are good 'vertical rulers' for calibration."""
    hip, ank, sh = (kp[23] + kp[24]) / 2, (kp[27] + kp[28]) / 2, (kp[11] + kp[12]) / 2
    leg = abs(ank[1] - hip[1]) + 1e-6
    torso = abs(hip[1] - sh[1]) + 1e-6
    return bool(_angle(kp[23], kp[25], kp[27]) > knee_min and _angle(kp[24], kp[26], kp[28]) > knee_min
                and abs(hip[0] - ank[0]) < lean_max * leg and abs(sh[0] - hip[0]) < lean_max * torso
                and abs(kp[27][0] - kp[28][0]) < 0.35 * leg)


def calibration_people(players):
    """Player observations usable as calibration objects (team-agnostic: referees count too)."""
    out = []
    for q in players:
        if q.get('occluded'):
            continue
        foot = (q['kp'][27] + q['kp'][28]) / 2 if q['upright'] else q['foot']
        out.append(dict(foot=foot, shoulder=q['shoulder'], upright=q['upright']))
    return out


def dedupe(players):
    """Remove partial duplicate detections of the same body."""
    def inside(a, b):
        ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
        iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
        return ix * iy / ((a[2] - a[0]) * (a[3] - a[1]))
    out = []
    for q in sorted(players, key=lambda q: -q['score']):
        sh = (q['kp'][11] + q['kp'][12]) / 2
        if all(np.linalg.norm(sh - (u['kp'][11] + u['kp'][12]) / 2) > 25 and inside(q['box'], u['box']) < 0.5
               for u in out):
            out.append(q)
    return sorted(out, key=lambda q: q['box'][0])


# ---------------------------------------------------------------- teams
def _region_polys(kp):
    d = (kp[25] + kp[26]) / 2 - (kp[23] + kp[24]) / 2
    return dict(torso=np.array([kp[11], kp[12], kp[24], kp[23]]),
                shorts=np.array([kp[23], kp[24], kp[24] + 0.6 * d, kp[23] + 0.6 * d]))


def colour_fraction(hsv, poly, lo, hi):
    """Fraction of pixels inside `poly` whose HSV lies in [lo, hi] (works on the polygon's bounding box only)."""
    poly = np.round(poly).astype(np.int32)
    x0, y0 = np.maximum(poly.min(0), 0)
    x1, y1 = np.minimum(poly.max(0) + 1, [hsv.shape[1], hsv.shape[0]])
    if x1 <= x0 or y1 <= y0:
        return 0.0
    m = np.zeros((y1 - y0, x1 - x0), np.uint8)
    cv2.fillConvexPoly(m, poly - [x0, y0], 1)
    px = hsv[y0:y1, x0:x1][m > 0]
    if not len(px):
        return 0.0
    return float(np.all((px >= np.array(lo)) & (px <= np.array(hi)), axis=1).mean())


def assign_teams(img, players, team_rules, overrides=None):
    """team_rules: ordered list; the first team whose rules all pass wins, a rule-less entry is the default.
       [{'team': 'AFC', 'rules': [{'region': 'shorts', 'hsv_lo': [90,60,60], 'hsv_hi': [125,255,255], 'min': 0.2}]},
        {'team': 'REF', 'rules': [...]},
        {'team': 'SVA', 'rules': []}]
       overrides: {player_id: team} for manual fixes."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    for q in players:
        polys = _region_polys(q['kp'])
        q['colour_stats'] = {}
        for t in team_rules:
            ok = True
            for r in t['rules']:
                f = colour_fraction(hsv, polys[r['region']], r['hsv_lo'], r['hsv_hi'])
                q['colour_stats'][f"{t['team']}:{r['region']}"] = round(f, 3)
                ok &= (f >= r.get('min', 0.0)) and (f <= r.get('max', 1.0))
            if ok:
                q['team'] = t['team']
                break
        else:
            q['team'] = 'UNKNOWN'
    for pid, team in (overrides or {}).items():
        for q in players:
            if q['id'] == int(pid):
                q['team'] = team
    return players
