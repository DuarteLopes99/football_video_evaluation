"""Evaluation helpers used by the notebooks.

Perception: detection precision/recall (IoU matching), team-assignment accuracy (label-permutation
invariant), homography reprojection error. Analytics: event P/R/F1 (see synthetic.evaluate_events),
per-frame possession accuracy against ground truth.

Annotation format for real frames (``data/annotations/<name>.json``)::

    {"image": "frame_0600.png",
     "boxes": [{"xyxy": [x0, y0, x1, y1], "cls": "player", "team": 0}, ...],
     "ball": [u, v],
     "landmarks": {"centre": [u, v], "halfway_top": [u, v], ...}}
"""
from __future__ import annotations

import json
from itertools import permutations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from .synthetic import evaluate_events  # noqa: F401  (re-export)
from .tracking import iou_matrix


def load_annotation(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def detection_pr(pred_xyxy: np.ndarray, gt_xyxy: np.ndarray, iou_thr: float = 0.5) -> dict:
    pred_xyxy, gt_xyxy = np.asarray(pred_xyxy).reshape(-1, 4), np.asarray(gt_xyxy).reshape(-1, 4)
    if len(pred_xyxy) == 0 or len(gt_xyxy) == 0:
        return dict(tp=0, fp=len(pred_xyxy), fn=len(gt_xyxy), precision=np.nan, recall=0.0 if len(gt_xyxy) else np.nan,
                    matches=[])
    iou = iou_matrix(pred_xyxy, gt_xyxy)
    r, c = linear_sum_assignment(-iou)
    m = [(i, j) for i, j in zip(r, c) if iou[i, j] >= iou_thr]
    tp = len(m)
    return dict(tp=tp, fp=len(pred_xyxy) - tp, fn=len(gt_xyxy) - tp, precision=tp / len(pred_xyxy),
                recall=tp / len(gt_xyxy), mean_iou=float(np.mean([iou[i, j] for i, j in m])) if m else np.nan,
                matches=m)


def team_accuracy(pred: np.ndarray, gt: np.ndarray) -> dict:
    """Cluster ids are arbitrary: score the best mapping {0,1} -> {0,1}. -1 (other) kept as is."""
    pred, gt = np.asarray(pred), np.asarray(gt)
    best = None
    for perm in permutations([0, 1]):
        mp = {0: perm[0], 1: perm[1], -1: -1}
        acc = float(np.mean([mp.get(int(p), -1) == int(g) for p, g in zip(pred, gt)])) if len(gt) else np.nan
        if best is None or acc > best[0]:
            best = (acc, perm)
    return dict(accuracy=best[0], mapping=best[1], n=len(gt))


def possession_accuracy(fc: pd.DataFrame, poss: np.ndarray, truth: pd.DataFrame) -> dict:
    """Per-frame team possession vs the synthetic ground truth (live-ball frames only)."""
    t = truth.set_index("frame").reindex(fc.frame)
    live = (~t.dead.fillna(True).astype(bool)).to_numpy() & (t.holder_team.to_numpy() >= 0)
    # ground-truth team possession: last holder team (carry over flights)
    gt_team = pd.Series(np.where(t.holder_team.to_numpy() >= 0, t.holder_team.to_numpy(), np.nan)).ffill().to_numpy()
    m = live & (poss >= 0)
    acc = float((poss[m] == gt_team[m]).mean()) if m.any() else np.nan
    cov = float(m.sum() / max(live.sum(), 1))
    share_pred = float((poss[m] == 0).mean()) if m.any() else np.nan
    share_gt = float((gt_team[live] == 0).mean()) if live.any() else np.nan
    return dict(frame_accuracy=acc, coverage=cov, team0_share_pred=share_pred, team0_share_gt=share_gt)
