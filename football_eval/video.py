"""Frame access for long match videos (a full game is ~100 min = 150k frames at 25 fps)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, List, Optional, Tuple

import cv2
import numpy as np


@dataclass
class VideoInfo:
    path: str
    fps: float
    n_frames: int
    width: int
    height: int

    @property
    def duration_s(self) -> float:
        return self.n_frames / self.fps if self.fps else 0.0


def probe(path: str) -> VideoInfo:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video {path}")
    info = VideoInfo(path, cap.get(cv2.CAP_PROP_FPS) or 25.0, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
                     int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    cap.release()
    return info


def iter_frames(path: str, analysis_fps: float = 10.0, start_s: float = 0.0, end_s: Optional[float] = None,
                resize_width: Optional[int] = None, windows: Optional[List[Tuple[float, float]]] = None,
                prefetch: int = 8) -> Iterator[Tuple[int, float, np.ndarray]]:
    """Yield ``(source_frame_idx, t_seconds, bgr_image)`` sub-sampled to ``analysis_fps``.

    ``windows``: list of (start_s, end_s) to read (e.g. the two halves, skipping half-time); default one window
    ``[start_s, end_s]``. All windows share one frame grid (every ``step``-th frame from the first window's start). ``prefetch``: decode in a background thread, ``prefetch`` frames ahead, so decoding overlaps
    with the detector (OpenCV releases the GIL while decoding).
    """
    gen = _frames(path, analysis_fps, start_s, end_s, resize_width, windows)
    if prefetch <= 0:
        yield from gen
        return
    import queue
    import threading
    q: "queue.Queue" = queue.Queue(maxsize=prefetch)
    stop = threading.Event()
    DONE = object()

    def work():
        try:
            for item in gen:
                while not stop.is_set():
                    try:
                        q.put(item, timeout=0.2)
                        break
                    except queue.Full:
                        continue
                if stop.is_set():
                    break
        except Exception as e:  # noqa: BLE001 — re-raised in the consumer
            q.put(e)
        finally:
            q.put(DONE)

    th = threading.Thread(target=work, daemon=True)
    th.start()
    try:
        while True:
            item = q.get()
            if item is DONE:
                break
            if isinstance(item, Exception):
                raise item
            yield item
    finally:
        stop.set()
        while not q.empty():                     # unblock the worker
            try:
                q.get_nowait()
            except Exception:  # noqa: BLE001
                break


def _frames(path, analysis_fps, start_s, end_s, resize_width, windows):
    info = probe(path)
    step = max(1, int(round(info.fps / analysis_fps)))
    wins = sorted((float(a or 0.0), None if b is None else float(b)) for a, b in (windows or [(start_s, end_s)]))
    to_f = lambda s_: int(round(s_ * info.fps)) if abs(s_ * info.fps - round(s_ * info.fps)) < 0.05 else int(np.ceil(s_ * info.fps))
    base = to_f(wins[0][0])                     # one frame grid for all windows (anchored at the first start)
    on_grid = lambda f_: base + int(np.ceil((f_ - base) / step)) * step
    cap = cv2.VideoCapture(path)
    pos = 0                                      # next frame the decoder will return
    try:
        for a, b in wins:
            first = on_grid(to_f(a))
            last = info.n_frames if b is None else min(info.n_frames, int(b * info.fps))
            if first < pos:
                first = on_grid(pos)
            if first >= last:
                continue
            if first != pos and not (0 < first - pos <= 3 * step):
                cap.set(cv2.CAP_PROP_POS_FRAMES, first)
                pos = first
            idx = pos
            while idx < last:
                if idx >= first and (idx - base) % step == 0:
                    ok, img = cap.read()
                    if not ok:
                        return
                    if resize_width and img.shape[1] != resize_width:
                        sc = resize_width / img.shape[1]
                        img = cv2.resize(img, (resize_width, int(round(img.shape[0] * sc))), interpolation=cv2.INTER_AREA)
                    yield idx, idx / info.fps, img
                else:
                    if not cap.grab():
                        return
                idx += 1
            pos = idx
    finally:
        cap.release()


def read_frame(path: str, t_s: float) -> np.ndarray:
    cap = cv2.VideoCapture(path)
    cap.set(cv2.CAP_PROP_POS_MSEC, t_s * 1000)
    ok, img = cap.read()
    cap.release()
    if not ok:
        raise ValueError(f"Cannot read frame at {t_s}s")
    return img


def fmt_clock(t_s: float) -> str:
    m, s = divmod(int(round(t_s)), 60)
    h, m = divmod(m, 60)
    return f"{h:d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"
