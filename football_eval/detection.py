"""Object detection: players / goalkeepers / referees / ball.

Backends
--------
* ``yolox_onnx``  — YOLOX (COCO, OpenCV Model Zoo). Runs through **onnxruntime** when installed (≈2x faster than
  ``cv2.dnn`` on CPU; on a Mac the CoreML provider is tried and kept only if it is faster), else ``cv2.dnn``.
  No PyTorch needed. COCO only knows ``person`` and ``sports ball``, so role (GK / referee) comes later from the
  team classification. Tiled inference keeps the ball (often < 15 px) detectable.

Speed: the detector is ~95 % of the processing time, so the number of tiles per frame is what matters. Tiles are
laid out over the *pitch region* of the frame (``roi``: the calibrated pitch + a margin, given by the perception loop
from the previous frame's camera pose), with the fewest tiles of at most ``tile`` px that cover it with
``tile_overlap`` px of overlap. A 1920 x 1080 frame went from 6 tiles (960 px, stride 720) to 2. People cut by an
inner tile border are merged with their other half, or dropped when a neighbouring tile saw them whole.
* ``ultralytics`` — any YOLOv8/YOLO11/YOLO26 or RT-DETR checkpoint. Recommended upgrade path:
  fine-tune on a football dataset with classes {ball, goalkeeper, player, referee}
  (e.g. Roboflow "football-players-detection", SoccerNet-Tracking) — see docs/SOTA_REVIEW.md.

Every backend returns :class:`Detections` in the canonical classes.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import time
from typing import List, Sequence

import cv2
import numpy as np

CANONICAL = ("player", "goalkeeper", "referee", "ball")


@dataclass
class Detections:
    xyxy: np.ndarray = field(default_factory=lambda: np.zeros((0, 4)))
    conf: np.ndarray = field(default_factory=lambda: np.zeros(0))
    cls: List[str] = field(default_factory=list)

    def __len__(self):
        return len(self.conf)

    def select(self, mask) -> "Detections":
        mask = np.asarray(mask, bool)
        return Detections(self.xyxy[mask], self.conf[mask], [c for c, m in zip(self.cls, mask) if m])

    def of(self, *classes) -> "Detections":
        return self.select([c in classes for c in self.cls])

    @property
    def foot(self) -> np.ndarray:
        """Bottom-centre of each box = ground contact point (players) / ball contact point."""
        return np.c_[(self.xyxy[:, 0] + self.xyxy[:, 2]) / 2, self.xyxy[:, 3]]


def _drop_regions(det: Detections, regions: Sequence, min_h: float) -> Detections:
    if len(det) == 0:
        return det
    keep = (det.xyxy[:, 3] - det.xyxy[:, 1]) >= min_h
    keep |= np.array([c == "ball" for c in det.cls])
    cx, cy = (det.xyxy[:, 0] + det.xyxy[:, 2]) / 2, (det.xyxy[:, 1] + det.xyxy[:, 3]) / 2
    for x0, y0, x1, y1 in regions or []:
        keep &= ~((cx >= x0) & (cx <= x1) & (cy >= y0) & (cy <= y1))
    return det.select(keep)


def tile_plan(W: int, H: int, tile: int, overlap: int, roi=None, stretch: float = 1.1) -> List[tuple]:
    """Fewest tiles (x0, y0, x1, y1) of at most ``tile`` px (``stretch`` x tile if that saves a row / column) that
    cover ``roi`` (default: the whole frame) with at least ``overlap`` px of overlap."""
    x0, y0, x1, y1 = (0, 0, W, H) if roi is None else (max(0, int(roi[0])), max(0, int(roi[1])),
                                                         min(W, int(np.ceil(roi[2]))), min(H, int(np.ceil(roi[3]))))
    if x1 - x0 < 32 or y1 - y0 < 32:
        x0, y0, x1, y1 = 0, 0, W, H

    def axis(a, b):
        L = b - a
        if L <= tile * stretch:
            return [(a, b)]
        n = int(np.ceil((L - overlap) / max(tile - overlap, 1)))
        size = int(np.ceil((L + (n - 1) * overlap) / n))
        starts = np.linspace(a, b - size, n).round().astype(int)
        return [(int(s), int(s) + size) for s in starts]
    return [(xa, ya, xb, yb) for ya, yb in axis(y0, y1) for xa, xb in axis(x0, x1)]


def _contained(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Share of each box of ``a`` inside each box of ``b`` (n_a, n_b)."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    x0 = np.maximum(a[:, None, 0], b[None, :, 0]); y0 = np.maximum(a[:, None, 1], b[None, :, 1])
    x1 = np.minimum(a[:, None, 2], b[None, :, 2]); y1 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x1 - x0, 0, None) * np.clip(y1 - y0, 0, None)
    return inter / np.maximum(((a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]))[:, None], 1e-9)


def merge_cut_boxes(boxes: np.ndarray, scores: np.ndarray, cut: np.ndarray):
    """Boxes cut by an inner tile border: drop them when a whole box contains them, else join the two halves of a
    person split between two tiles (same rows, touching / overlapping columns, or the transpose)."""
    boxes, scores, cut = boxes.copy(), scores.copy(), cut.copy()
    keep = np.ones(len(boxes), bool)
    whole = np.where(~cut)[0]
    ci = np.where(cut)[0]
    if len(ci) and len(whole):
        c = _contained(boxes[ci], boxes[whole]).max(1)
        keep[ci[c >= 0.5]] = False
    ci = [i for i in np.where(cut & keep)[0]]
    for a_ in range(len(ci)):
        i = ci[a_]
        if not keep[i]:
            continue
        for j in ci[a_ + 1:]:
            if not keep[j]:
                continue
            bi, bj = boxes[i], boxes[j]
            ov = lambda lo1, hi1, lo2, hi2: max(0.0, min(hi1, hi2) - max(lo1, lo2)) / max(min(hi1 - lo1, hi2 - lo2), 1e-9)
            gap_x = max(bi[0], bj[0]) - min(bi[2], bj[2])
            gap_y = max(bi[1], bj[1]) - min(bi[3], bj[3])
            same_rows = ov(bi[1], bi[3], bj[1], bj[3]) >= 0.6 and gap_x <= 6
            same_cols = ov(bi[0], bi[2], bj[0], bj[2]) >= 0.6 and gap_y <= 6
            if same_rows or same_cols:
                boxes[i] = [min(bi[0], bj[0]), min(bi[1], bj[1]), max(bi[2], bj[2]), max(bi[3], bj[3])]
                scores[i] = max(scores[i], scores[j])
                keep[j] = False
    return boxes[keep], scores[keep]


class _OnnxRunner:
    """onnxruntime session (CoreML on a Mac when it is measurably faster) or cv2.dnn as the fallback."""

    def __init__(self, model_path: str, size: int, runtime: str = "auto", threads: int = 0):
        self.size, self.sess, self.net, self.name = size, None, None, "cv2.dnn"
        if runtime in ("auto", "onnxruntime", "ort", "coreml"):
            try:
                import onnxruntime as ort
                so = ort.SessionOptions()
                if threads:
                    so.intra_op_num_threads = int(threads)
                avail = ort.get_available_providers()
                cands = [["CPUExecutionProvider"]]
                if "CoreMLExecutionProvider" in avail and runtime in ("auto", "coreml"):
                    cands.insert(0, ["CoreMLExecutionProvider", "CPUExecutionProvider"])
                best = None
                x = np.random.default_rng(0).uniform(0, 255, (1, 3, size, size)).astype(np.float32)
                for prov in cands:
                    try:
                        sess = ort.InferenceSession(model_path, so, providers=prov)
                        nm = sess.get_inputs()[0].name
                        sess.run(None, {nm: x})                       # warm-up (CoreML compiles here)
                        t0 = time.perf_counter()
                        for _ in range(3):
                            sess.run(None, {nm: x})
                        dt = (time.perf_counter() - t0) / 3
                        if best is None or dt < best[0]:
                            best = (dt, sess, nm, prov[0])
                    except Exception as e:  # noqa: BLE001 — a provider that fails is simply skipped
                        print(f"[detector] {prov[0]} unavailable: {e}")
                if best is not None:
                    self.ms, self.sess, self.inp, self.name = best[0] * 1000, best[1], best[2], "onnxruntime/" + best[3]
            except ImportError:
                if runtime != "auto":
                    print("[detector] onnxruntime not installed -> cv2.dnn (pip install onnxruntime for ~2x speed)")
        if self.sess is None:
            self.net = cv2.dnn.readNet(model_path)

    def __call__(self, blob: np.ndarray) -> np.ndarray:
        if self.sess is not None:
            return self.sess.run(None, {self.inp: blob})[0][0]
        self.net.setInput(blob)
        return self.net.forward()[0]


class YoloxOnnxDetector:
    """YOLOX-S COCO with pitch-aware tiling (see the module docstring)."""

    BALL_CLS = 32  # COCO "sports ball"

    def __init__(self, model_path: str, conf_player=0.3, conf_ball=0.15, tile=1024, overlap=96,
                 input_size=640, ignore_regions=(), min_box_h_frac=0.012, runtime: str = "auto", threads: int = 0):
        self.runner = _OnnxRunner(model_path, input_size, runtime, threads)
        self.S, self.tile, self.overlap = input_size, int(tile), int(overlap)
        self.cp, self.cb = conf_player, conf_ball
        self.ignore, self.min_h_frac = list(ignore_regions or []), min_box_h_frac
        self.last_tiles: List[tuple] = []
        g_, e_ = [], []
        for s in (8, 16, 32):
            g = input_size // s
            yv, xv = np.meshgrid(np.arange(g), np.arange(g), indexing="ij")
            g_.append(np.stack([xv, yv], -1).reshape(-1, 2))
            e_.append(np.full((g * g, 1), s))
        self.grids, self.strides = np.concatenate(g_), np.concatenate(e_)
        print(f"[detector] YOLOX via {self.runner.name}, tiles <= {self.tile} px (overlap {self.overlap})")

    def _infer(self, tile):
        S = self.S
        h, w = tile.shape[:2]
        r = min(S / h, S / w)
        pad = np.full((S, S, 3), 114, np.uint8)
        pad[:int(h * r), :int(w * r)] = cv2.resize(tile, (int(w * r), int(h * r)), interpolation=cv2.INTER_AREA)
        o = self.runner(cv2.dnn.blobFromImage(pad, 1.0, (S, S), swapRB=False)).copy()
        o[:, :2] = (o[:, :2] + self.grids) * self.strides
        o[:, 2:4] = np.exp(o[:, 2:4]) * self.strides
        boxes = np.c_[o[:, 0] - o[:, 2] / 2, o[:, 1] - o[:, 3] / 2, o[:, 0] + o[:, 2] / 2, o[:, 1] + o[:, 3] / 2] / r
        sp = o[:, 4] * o[:, 5]
        sb = o[:, 4] * o[:, 5 + self.BALL_CLS]
        kp, kb = sp > self.cp, sb > self.cb
        return boxes[kp], sp[kp], boxes[kb], sb[kb]

    def __call__(self, img: np.ndarray, roi=None) -> Detections:
        """``roi`` = (x0, y0, x1, y1) region worth looking at (the pitch); None = whole frame."""
        H, W = img.shape[:2]
        tiles = tile_plan(W, H, self.tile, self.overlap, roi)
        self.last_tiles = tiles
        P, PS, PC, B, BS = [], [], [], [], []
        for x0, y0, x1, y1 in tiles:
            p, ps, b, bs = self._infer(img[y0:y1, x0:x1])
            tw, th = x1 - x0, y1 - y0
            # touching an inner tile border (not the frame border) = possibly cut
            cut = (((p[:, 0] <= 3) & (x0 > 0)) | ((p[:, 2] >= tw - 3) & (x1 < W)) |
                   ((p[:, 1] <= 3) & (y0 > 0)) | ((p[:, 3] >= th - 3) & (y1 < H)))
            off = np.array([x0, y0, x0, y0])
            P.append(p + off), PS.append(ps), PC.append(cut), B.append(b + off), BS.append(bs)
        out = []
        for boxes, scores, name, thr, cut in ((np.concatenate(P), np.concatenate(PS), "player", self.cp, np.concatenate(PC)),
                                              (np.concatenate(B), np.concatenate(BS), "ball", self.cb, None)):
            if len(boxes) == 0:
                continue
            keep = np.array(cv2.dnn.NMSBoxes([[float(a), float(b), float(c - a), float(d - b)] for a, b, c, d in boxes],
                                             scores.tolist(), thr, 0.45)).ravel().astype(int)
            boxes, scores = boxes[keep], scores[keep]
            if cut is not None and len(tiles) > 1:
                boxes, scores = merge_cut_boxes(boxes, scores, cut[keep])
            out.append(Detections(boxes, scores, [name] * len(scores)))
        det = Detections(np.concatenate([d.xyxy for d in out]) if out else np.zeros((0, 4)),
                         np.concatenate([d.conf for d in out]) if out else np.zeros(0),
                         sum([d.cls for d in out], []))
        return _drop_regions(det, self.ignore, self.min_h_frac * H)


class UltralyticsDetector:
    def __init__(self, model_path: str, class_map: dict, conf_player=0.3, conf_ball=0.15, imgsz=1280,
                 device="cpu", ignore_regions=(), min_box_h_frac=0.012):
        from ultralytics import YOLO  # optional dependency
        self.model = YOLO(model_path)
        self.names = self.model.names
        self.class_map = class_map
        self.cp, self.cb, self.imgsz, self.device = conf_player, conf_ball, imgsz, device
        self.ignore, self.min_h_frac = list(ignore_regions or []), min_box_h_frac
        wanted = [i for i, n in self.names.items() if n in class_map]
        self.classes = wanted or None

    def __call__(self, img: np.ndarray, roi=None) -> Detections:
        r = self.model.predict(img, conf=min(self.cp, self.cb), imgsz=self.imgsz, device=self.device,
                               classes=self.classes, verbose=False)[0]
        b = r.boxes
        if b is None or len(b) == 0:
            return Detections()
        xyxy, conf = b.xyxy.cpu().numpy(), b.conf.cpu().numpy()
        cls = [self.class_map.get(self.names[int(c)], "other") for c in b.cls.cpu().numpy()]
        thr = np.array([self.cb if c == "ball" else self.cp for c in cls])
        keep = (conf >= thr) & np.array([c in CANONICAL for c in cls])
        det = Detections(xyxy, conf, cls).select(keep)
        return _drop_regions(det, self.ignore, self.min_h_frac * img.shape[0])


def build_detector(cfg):
    from .config import resolve_path
    d = cfg.detection
    d.model_path = resolve_path(d.model_path)
    if d.backend == "yolox_onnx":
        overlap = d.tile - d.tile_stride if (d.tile_stride and d.tile_stride < d.tile) else d.tile_overlap
        return YoloxOnnxDetector(d.model_path, d.conf_player, d.conf_ball, d.tile, overlap,
                                 ignore_regions=cfg.video.ignore_regions, min_box_h_frac=d.min_box_h_frac,
                                 runtime=d.runtime, threads=d.threads)
    if d.backend == "ultralytics":
        return UltralyticsDetector(d.model_path, d.class_map, d.conf_player, d.conf_ball, d.imgsz, d.device,
                                   cfg.video.ignore_regions, d.min_box_h_frac)
    raise ValueError(f"Unknown detection backend {d.backend}")
