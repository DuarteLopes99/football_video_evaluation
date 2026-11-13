"""Simple tracker: coloured players + ball tracking on a match video (no calibration, no clicking).

    detection (YOLOX)  ->  people tracking (IoU)  ->  team colours (KMeans per track)
                       ->  ball tracking (Kalman + zoomed re-detection)
                       ->  simple segmentation (local grass subtraction)  ->  annotated video

Files: ``teams.py``, ``ball.py``, ``segment.py``, ``draw.py``, ``run.py``. Entry point: ``simple_track.py``.
"""
from .run import run  # noqa: F401
