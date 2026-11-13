"""Coloured players + ball tracking on a match video.

Examples
--------
    python simple_track.py data/videos/J3_MansoresSanguedo_1080p.mp4 --start 24:38 --duration 30
    python simple_track.py data/videos/2026-10-18_UD_Mansores_vs_Canedo_FC_J2.mp4 --start 10:00 --duration 60 --roi-top 0.5 --upscale 3

Writes outputs/simple/<video name>/annotated.mp4 + tracks.csv + ball.csv.
"""
import argparse
import os

from football_eval.simple import run


def seconds(s: str) -> float:
    """'90', '1:30' or '1:02:30' -> seconds."""
    parts = [float(p) for p in s.split(":")]
    return sum(p * 60 ** i for i, p in enumerate(reversed(parts)))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video")
    ap.add_argument("--start", default="0", help="start time, e.g. 24:38")
    ap.add_argument("--duration", default="30", help="seconds to process (0 = until the end)")
    ap.add_argument("--stride", type=int, default=1, help="use every N-th frame (2 = twice as fast)")
    ap.add_argument("--roi-top", type=float, default=0.0,
                    help="ignore the top part of the frame (sky / stands), as a fraction of the height")
    ap.add_argument("--upscale", type=float, default=1.0,
                    help="enlarge low-resolution video before detection (e.g. 3 for 640x360 streams; slower)")
    ap.add_argument("--out", default=None, help="output folder (default outputs/simple/<video name>)")
    ap.add_argument("--model", default="models/yolox.onnx")
    a = ap.parse_args()
    name = os.path.splitext(os.path.basename(a.video))[0]
    run(a.video, a.out or os.path.join("outputs", "simple", name), start_s=seconds(a.start),
        duration_s=seconds(a.duration) or None, stride=a.stride, model=a.model, roi_top=a.roi_top,
        upscale=a.upscale)
