"""Multi-object tracking of people (players, goalkeepers, referees).

* ``iou`` (default): built-in tracker — constant-velocity prediction + Hungarian matching on
  (1 - IoU) with a centre-distance gate. No extra dependency; good enough at 10 fps.
* ``bytetrack``: ``supervision.ByteTrack`` (Zhang et al., 2022) — the de-facto baseline in the
  SoccerNet tracking / GSR challenges. Swap for Deep-EIoU / BoT-SORT + Re-ID for long-term identity.

Identity across the whole match (re-identification after players leave the frame) is NOT solved
here: track ids fragment. Player-level stats are therefore reported per *tracklet* until jersey
number recognition / Re-ID merging is added (see docs/SOTA_REVIEW.md, "Game State Reconstruction").
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import numpy as np
from scipy.optimize import linear_sum_assignment

from .detection import Detections


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    x0 = np.maximum(a[:, None, 0], b[None, :, 0])
    y0 = np.maximum(a[:, None, 1], b[None, :, 1])
    x1 = np.minimum(a[:, None, 2], b[None, :, 2])
    y1 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x1 - x0, 0, None) * np.clip(y1 - y0, 0, None)
    area = lambda r: (r[:, 2] - r[:, 0]) * (r[:, 3] - r[:, 1])
    return inter / (area(a)[:, None] + area(b)[None, :] - inter + 1e-9)


@dataclass
class _Track:
    tid: int
    box: np.ndarray
    vel: np.ndarray = field(default_factory=lambda: np.zeros(4))
    age: int = 0          # frames since last match
    hits: int = 1
    cls: str = "player"


class IoUTracker:
    def __init__(self, max_age: int = 10, min_iou: float = 0.2, max_center_dist: float = 80.0, min_hits: int = 2):
        self.max_age, self.min_iou, self.max_dist, self.min_hits = max_age, min_iou, max_center_dist, min_hits
        self.tracks: List[_Track] = []
        self._next = 1

    def update(self, det: Detections) -> np.ndarray:
        """Returns track ids aligned with ``det`` (-1 for unconfirmed detections)."""
        pred = np.array([t.box + t.vel for t in self.tracks]).reshape(-1, 4)
        boxes = det.xyxy
        ids = np.full(len(boxes), -1, int)
        if len(pred) and len(boxes):
            iou = iou_matrix(pred, boxes)
            cp = (pred[:, :2] + pred[:, 2:]) / 2
            cd = (boxes[:, :2] + boxes[:, 2:]) / 2
            dist = np.linalg.norm(cp[:, None] - cd[None], axis=2)
            h = (pred[:, 3] - pred[:, 1])[:, None]
            cost = (1 - iou) + 0.5 * np.clip(dist / (self.max_dist + 1e-9), 0, 2)
            gate = (iou < self.min_iou) & (dist > np.minimum(self.max_dist, 1.2 * h))
            cost[gate] = 1e6
            r, c = linear_sum_assignment(cost)
            matched_t, matched_d = set(), set()
            for i, j in zip(r, c):
                if cost[i, j] >= 1e6:
                    continue
                t = self.tracks[i]
                t.vel = 0.6 * t.vel + 0.4 * (boxes[j] - t.box)
                t.box, t.age, t.hits = boxes[j].copy(), 0, t.hits + 1
                t.cls = det.cls[j]
                matched_t.add(i), matched_d.add(j)
                if t.hits >= self.min_hits:
                    ids[j] = t.tid
        else:
            matched_t, matched_d = set(), set()
        for i, t in enumerate(self.tracks):
            if i not in matched_t:
                t.age += 1
                t.box = t.box + t.vel
                t.vel *= 0.7
        for j in range(len(boxes)):
            if j not in matched_d:
                self.tracks.append(_Track(self._next, boxes[j].copy(), cls=det.cls[j]))
                self._next += 1
        self.tracks = [t for t in self.tracks if t.age <= self.max_age]
        return ids


    def id_for(self, box, min_iou: float = 0.5) -> int:
        """Id of the (possibly not yet confirmed) track whose current box overlaps ``box`` best, else -1."""
        if not self.tracks:
            return -1
        iou = iou_matrix(np.asarray(box, float).reshape(1, 4), np.array([t.box for t in self.tracks]))[0]
        k = int(np.argmax(iou))
        return self.tracks[k].tid if iou[k] >= min_iou else -1


class ByteTrackWrapper:
    def __init__(self, fps: float, max_age_s: float = 1.0):
        import supervision as sv  # optional dependency
        self.sv = sv
        self.tracker = sv.ByteTrack(frame_rate=int(round(fps)), lost_track_buffer=int(max_age_s * fps))

    def update(self, det: Detections) -> np.ndarray:
        if len(det) == 0:
            self.tracker.update_with_detections(self.sv.Detections.empty())
            return np.zeros(0, int)
        d = self.sv.Detections(xyxy=det.xyxy.astype(np.float32), confidence=det.conf.astype(np.float32),
                               class_id=np.zeros(len(det), int))
        out = self.tracker.update_with_detections(d)
        ids = np.full(len(det), -1, int)
        if len(out):
            iou = iou_matrix(det.xyxy, out.xyxy)
            for j in range(len(out)):
                i = int(np.argmax(iou[:, j]))
                if iou[i, j] > 0.7:
                    ids[i] = int(out.tracker_id[j])
        return ids


def build_tracker(cfg, fps: float):
    t = cfg.tracking
    if t.backend == "bytetrack":
        return ByteTrackWrapper(fps, t.max_age_s)
    return IoUTracker(max_age=max(1, int(t.max_age_s * fps)), min_iou=t.min_iou, max_center_dist=t.max_center_dist_px)
