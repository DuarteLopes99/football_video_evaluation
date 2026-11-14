"""Plumbing of the step-by-step pipeline notebooks (football_eval/nbtools.py).

    python -m pytest tests/test_nbtools.py -q
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from football_eval import nbtools  # noqa: E402
from football_eval.pitch import Homography  # noqa: E402


def test_params_defaults_and_override(tmp_path):
    p = tmp_path / "params.yaml"
    p.write_text("video: data/videos/x.mp4\nt_start: 100\nduration_s: 10\n", encoding="utf-8")
    P = nbtools.load_params(p)
    assert P.probe_t == 105.0 and P.run_name == "x_t100" and P.analysis_fps == 10.0
    assert nbtools.load_params(p, run_name="r").run_name == "r"


def test_homography_row_round_trip():
    H = Homography(np.array([[10.0, 1, 50], [0.5, 8, 30], [0.001, 0.002, 1]]))
    row = nbtools.homography_table_row(H, 7, 0.7)
    H2 = nbtools.homography_from_row(row)
    pts = np.array([[0, 0], [50, 30], [100, 64]], float)
    assert np.allclose(H.to_image(pts), H2.to_image(pts))
    assert nbtools.homography_from_row(nbtools.homography_table_row(None, 8, 0.8)) is None


def test_draw_helpers():
    img = np.zeros((100, 200, 3), np.uint8)
    out = nbtools.draw_boxes(img, [[10, 10, 50, 80]], ["a"], [(255, 255, 255)])
    assert out.sum() > 0 and img.sum() == 0
    assert nbtools.crop(img, [10, 10, 50, 80]).shape == (96, 96, 3)
