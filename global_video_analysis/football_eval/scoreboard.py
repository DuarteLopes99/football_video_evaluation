"""Broadcast scoreboard watcher — a cheap, very reliable goal signal for streamed matches.

Most club live-streams (e.g. federation streams) burn a score bug into the video. When the score
digits change and stay changed for ``min_persist_s``, a goal almost certainly happened in the
preceding ~10-60 s (the operator updates the bug after the celebration). We use this to confirm
or recover goals detected by the event rules. OCR is optional (pytesseract) — change detection
alone needs nothing.
"""
from __future__ import annotations

from typing import List, Optional

import cv2
import numpy as np


class ScoreboardWatcher:
    def __init__(self, roi, change_threshold: float = 18.0, min_persist_s: float = 3.0, ocr: bool = False):
        self.roi = [int(v) for v in roi]
        self.thr, self.persist = change_threshold, min_persist_s
        self.ref: Optional[np.ndarray] = None
        self.pending = None                   # (t_first_change, patch)
        self.changes: List[dict] = []
        self.ocr = None
        if ocr:
            try:
                import pytesseract
                self.ocr = pytesseract
            except ImportError:
                self.ocr = None

    def _patch(self, img):
        x0, y0, x1, y1 = self.roi
        g = cv2.cvtColor(img[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY)
        return cv2.resize(g, (max(8, (x1 - x0) // 2), max(8, (y1 - y0) // 2))).astype(np.float32)

    def read_text(self, img) -> Optional[str]:
        if self.ocr is None:
            return None
        x0, y0, x1, y1 = self.roi
        g = cv2.cvtColor(img[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY)
        g = cv2.resize(g, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
        return self.ocr.image_to_string(g, config="--psm 7 -c tessedit_char_whitelist=0123456789-:").strip()

    def update(self, t: float, img: np.ndarray):
        p = self._patch(img)
        if self.ref is None:
            self.ref = p
            return
        diff = float(np.abs(p - self.ref).mean())
        if diff > self.thr:
            if self.pending is None:
                self.pending = (t, p)
            elif t - self.pending[0] >= self.persist:
                # changed and stable w.r.t. the new patch -> commit
                if float(np.abs(p - self.pending[1]).mean()) < self.thr:
                    self.changes.append({"t": self.pending[0], "diff": diff, "text": self.read_text(img)})
                    self.ref = p
                self.pending = None
        else:
            self.pending = None
