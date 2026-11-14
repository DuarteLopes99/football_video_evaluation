"""Helpers for the step-by-step pipeline notebooks (``notebooks/pipeline/``).

The notebooks show every stage of the pipeline separately (video -> frames -> detection -> pitch -> tracking / teams
-> ball -> integrated perception -> analytics -> model training cycle). This module only holds the plumbing they
share, so the cells themselves show the pipeline code:

* :func:`load_params` / :func:`build_context` — one parameter file (``notebooks/pipeline/params.yaml``) for all the
  notebooks: which video, which match setup, which time window. The config is built exactly like ``run_analysis.py``
  does (config file <- match setup), then restricted to the window.
* :func:`save_stage` / :func:`load_stage` — each notebook writes its output table to
  ``outputs/_pipeline_nb/<run_name>/``; the next notebook reads it. A stage can therefore be replaced (new model,
  new tracker) and the downstream notebooks re-run on its output without touching the rest.
* drawing helpers (:func:`show`, :func:`draw_boxes`, :func:`grid`) and a :class:`Timer`.

Point the notebooks at another parameter file with the environment variable ``FVE_NB_PARAMS``.
"""
from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Iterable, Optional, Sequence

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PARAMS = ROOT / "notebooks" / "pipeline" / "params.yaml"

DEFAULTS = dict(
    video="data/videos/match.mp4",
    setup="auto",            # match_setup json, "auto" = configs/match_setup_<video stem>.json, null = none
    config="configs/default.yaml",
    t_start=600.0,           # seconds into the video
    duration_s=20.0,         # length of the analysed window
    analysis_fps=10.0,
    probe_t=None,            # single frame for the per-frame checks (default: middle of the window)
    run_name=None,           # default: <video stem>_t<t_start>
    runtime=None,            # detector runtime override: auto | cpu | cuda | directml | coreml ...
    autocalib=True,          # solve the venue camera from clicked landmarks if needed (as run_analysis.py)
    full_run_dir=None,       # a finished run_analysis.py output folder (analytics + training notebooks)
)

# BGR colours used by every notebook
COLORS = {0: (60, 60, 230), 1: (230, 160, 40), -1: (200, 200, 200), "ball": (0, 230, 255),
          "player": (255, 255, 255), "goalkeeper": (0, 200, 0), "referee": (0, 0, 0), "tile": (255, 0, 255)}


# ----------------------------------------------------------------------------- parameters and context
def load_params(path: Optional[str | Path] = None, **override) -> SimpleNamespace:
    import yaml
    path = Path(path or os.environ.get("FVE_NB_PARAMS") or DEFAULT_PARAMS)
    p = dict(DEFAULTS)
    if path.exists():
        with open(path, encoding="utf-8") as f:
            p.update({k: v for k, v in (yaml.safe_load(f) or {}).items() if v is not None or k in p})
    p.update(override)
    p["params_file"] = str(path)
    if p["probe_t"] is None:
        p["probe_t"] = float(p["t_start"]) + float(p["duration_s"]) / 2
    if not p["run_name"]:
        p["run_name"] = f"{Path(p['video']).stem}_t{int(float(p['t_start']))}"
    return SimpleNamespace(**p)


def _abs(p: Optional[str]) -> Optional[Path]:
    if not p:
        return None
    q = Path(p).expanduser()
    return q if q.is_absolute() else ROOT / q


def build_context(P: SimpleNamespace, verbose: bool = True) -> SimpleNamespace:
    """Config + match setup + video info for the window ``[t_start, t_start + duration_s]`` (what ``run_analysis.py``
    does before the perception, restricted to the window)."""
    from .config import load_config, resolve_path
    from .match_setup import apply_to_config, checklist, find_setup, format_checklist, load_setup
    from .perception import processing_windows
    from .video import probe

    video = resolve_path(str(_abs(P.video)))
    if not Path(video).exists():
        raise FileNotFoundError(f"video not found: {video} -> set 'video' in {P.params_file}")
    cfg = load_config(_abs(P.config))
    setup, setup_path, changes, items = None, None, [], []
    if P.setup == "auto":
        setup_path = find_setup(video, ROOT / "configs")
    elif P.setup:
        setup_path = _abs(P.setup)
    if setup_path:
        setup = load_setup(setup_path)
        items = checklist(setup, ROOT)
        changes = apply_to_config(setup, cfg)
        if P.autocalib:
            try:
                from .autocalib import ensure_calibration
                changes += ensure_calibration(setup, str(setup_path), cfg, video, ROOT)
            except Exception as e:  # noqa: BLE001 — the notebooks still run without a solved camera
                changes.append(f"autocalib skipped: {e}")
    if P.runtime:
        cfg.detection.runtime = P.runtime
    cfg.video.analysis_fps = float(P.analysis_fps)
    cfg.video.start_s = float(P.t_start)
    cfg.video.end_s = float(P.t_start) + float(P.duration_s)
    info = probe(video)
    windows = processing_windows(cfg, info.duration_s)
    out = ROOT / "outputs" / "_pipeline_nb" / P.run_name
    out.mkdir(parents=True, exist_ok=True)
    ctx = SimpleNamespace(P=P, cfg=cfg, setup=setup, setup_path=setup_path, setup_changes=changes,
                          checklist=items, video=video, info=info, windows=windows, out=out,
                          step=max(1, int(round(info.fps / cfg.video.analysis_fps))))
    if verbose:
        print(f"video     : {video}\n            {info.width}x{info.height} @ {info.fps:.2f} fps, "
              f"{info.duration_s / 60:.1f} min")
        print(f"setup     : {setup_path or 'none (teams, periods, kits and calibration are guessed)'}")
        if items:
            print(format_checklist(items))
        for c in changes:
            print("  applied:", c)
        print(f"window    : {', '.join(f'{a:.1f}-{b:.1f}s' for a, b in windows)} "
              f"(1 frame in {ctx.step} -> {info.fps / ctx.step:.1f} analysed fps)")
        print(f"stage dir : {out}")
    return ctx


# ----------------------------------------------------------------------------- stages
def save_stage(ctx, name: str, df: pd.DataFrame) -> Path:
    p = ctx.out / f"{name}.csv"
    df.to_csv(p, index=False)
    print(f"[stage] {name}: {len(df)} rows -> {p.relative_to(ROOT) if p.is_relative_to(ROOT) else p}")
    return p


def load_stage(ctx, name: str, required: bool = True) -> Optional[pd.DataFrame]:
    p = ctx.out / f"{name}.csv"
    if not p.exists():
        if required:
            raise FileNotFoundError(f"stage '{name}' not found ({p}); run the notebook that produces it first")
        return None
    return pd.read_csv(p)


def frames_in_window(ctx, max_frames: Optional[int] = None, prefetch: int = 8):
    """``(frame_idx, t, image)`` of the analysed window, on the same frame grid as the pipeline."""
    from .video import iter_frames
    for k, item in enumerate(iter_frames(ctx.video, ctx.cfg.video.analysis_fps, windows=ctx.windows,
                                         resize_width=ctx.cfg.video.resize_width, prefetch=prefetch)):
        if max_frames is not None and k >= max_frames:
            return
        yield item


def read_frame_idx(video: str, frame: int) -> np.ndarray:
    cap = cv2.VideoCapture(video)
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame))
    ok, img = cap.read()
    cap.release()
    if not ok:
        raise ValueError(f"cannot read frame {frame}")
    return img


def probe_frame(ctx) -> tuple:
    """(frame_idx, t, image) of the analysed frame closest to ``probe_t``."""
    f = int(round(ctx.P.probe_t * ctx.info.fps))
    base = int(round(ctx.windows[0][0] * ctx.info.fps))
    f = base + int(round((f - base) / ctx.step)) * ctx.step
    img = read_frame_idx(ctx.video, f)
    if ctx.cfg.video.resize_width and img.shape[1] != ctx.cfg.video.resize_width:
        sc = ctx.cfg.video.resize_width / img.shape[1]
        img = cv2.resize(img, (ctx.cfg.video.resize_width, int(round(img.shape[0] * sc))), interpolation=cv2.INTER_AREA)
    return f, f / ctx.info.fps, img


def detections_from_table(df: pd.DataFrame, frame: int, classes: Sequence[str] = ("player", "goalkeeper", "referee")):
    from .detection import Detections
    g = df[(df.frame == frame) & df.cls.isin(classes)]
    return Detections(g[["x0", "y0", "x1", "y1"]].to_numpy(float), g.conf.to_numpy(float), g.cls.tolist())


def homography_table_row(H, frame: int, t: float) -> dict:
    row = dict(frame=int(frame), t=float(t), has_h=H is not None)
    M = H.H / H.H[2, 2] if H is not None else np.full((3, 3), np.nan)
    row.update({f"h{i}{j}": float(M[i, j]) for i in range(3) for j in range(3)})
    return row


def homography_from_row(row):
    from .pitch import Homography
    if not bool(row["has_h"]) or not np.isfinite(row["h00"]):
        return None
    return Homography(np.array([[row[f"h{i}{j}"] for j in range(3)] for i in range(3)], float))


# ----------------------------------------------------------------------------- drawing
def show(img: np.ndarray, title: str = "", size: float = 12.0, ax=None):
    import matplotlib.pyplot as plt
    if ax is None:
        h, w = img.shape[:2]
        _, ax = plt.subplots(figsize=(size, size * h / w))
    ax.imshow(img[..., ::-1] if img.ndim == 3 else img, cmap=None if img.ndim == 3 else "gray")
    ax.set_title(title)
    ax.axis("off")
    return ax


def draw_boxes(img: np.ndarray, boxes, labels: Optional[Iterable] = None, colors: Optional[Iterable] = None,
               thickness: int = 2, font: float = 0.5) -> np.ndarray:
    out = img.copy()
    boxes = np.asarray(boxes, float).reshape(-1, 4)
    labels = list(labels) if labels is not None else [None] * len(boxes)
    colors = list(colors) if colors is not None else [(255, 255, 255)] * len(boxes)
    for (x0, y0, x1, y1), lab, col in zip(boxes.astype(int), labels, colors):
        cv2.rectangle(out, (x0, y0), (x1, y1), col, thickness)
        if lab is not None and lab != "":
            cv2.putText(out, str(lab), (x0, max(y0 - 4, 10)), cv2.FONT_HERSHEY_SIMPLEX, font, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(out, str(lab), (x0, max(y0 - 4, 10)), cv2.FONT_HERSHEY_SIMPLEX, font, col, 1, cv2.LINE_AA)
    return out


def draw_pitch_lines(img: np.ndarray, H, pitch, color=(0, 255, 255), thickness: int = 2) -> np.ndarray:
    """Pitch lines projected into the image with ``H`` (a :class:`football_eval.pitch.Homography`)."""
    out = img.copy()
    if H is None:
        return out
    for line in pitch.lines():
        pts = H.to_image(line)
        if not np.isfinite(pts).all():
            continue
        cv2.polylines(out, [np.round(pts).astype(np.int32).reshape(-1, 1, 2)], False, color, thickness, cv2.LINE_AA)
    return out


def grid(images: Sequence[np.ndarray], titles: Optional[Sequence[str]] = None, cols: int = 3, size: float = 5.0):
    import matplotlib.pyplot as plt
    n = len(images)
    if n == 0:
        print("(nothing to show)")
        return None
    rows = int(np.ceil(n / cols))
    h, w = images[0].shape[:2]
    fig, axes = plt.subplots(rows, cols, figsize=(cols * size, rows * size * h / w + 0.4), squeeze=False)
    for k, ax in enumerate(axes.ravel()):
        if k < n:
            show(images[k], titles[k] if titles else "", ax=ax)
        else:
            ax.axis("off")
    fig.tight_layout()
    return fig


def crop(img: np.ndarray, box, pad: float = 0.15, size: Optional[int] = 96) -> np.ndarray:
    x0, y0, x1, y1 = np.asarray(box, float)
    w, h = x1 - x0, y1 - y0
    x0, x1 = int(max(0, x0 - pad * w)), int(min(img.shape[1], x1 + pad * w))
    y0, y1 = int(max(0, y0 - pad * h)), int(min(img.shape[0], y1 + pad * h))
    c = img[y0:y1, x0:x1]
    if size and c.size:
        s = size / max(c.shape[:2])
        c = cv2.resize(c, (max(1, int(c.shape[1] * s)), max(1, int(c.shape[0] * s))), interpolation=cv2.INTER_CUBIC)
        c = cv2.copyMakeBorder(c, 0, size - c.shape[0], 0, size - c.shape[1], cv2.BORDER_CONSTANT, value=(40, 40, 40))
    return c


# ----------------------------------------------------------------------------- bookkeeping
class Timer:
    """``with Timer() as tm: ...`` then ``tm.s``; or ``tm.lap('name')`` inside a loop and ``tm.table()``."""

    def __init__(self):
        self.s, self.laps, self._t = 0.0, {}, None

    def __enter__(self):
        self._t = self._l = time.perf_counter()
        return self

    def lap(self, name: str):
        now = time.perf_counter()
        self.laps[name] = self.laps.get(name, 0.0) + (now - self._l)
        self._l = now

    def __exit__(self, *exc):
        self.s = time.perf_counter() - self._t

    def table(self, n: int = 1) -> pd.DataFrame:
        tot = sum(self.laps.values()) or 1e-9
        return pd.DataFrame([dict(stage=k, total_s=round(v, 2), ms_per_frame=round(1000 * v / max(n, 1), 1),
                                  share=round(v / tot, 3)) for k, v in self.laps.items()])


def git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True,
                              timeout=5).stdout.strip() or "unknown"
    except Exception:  # noqa: BLE001
        return "unknown"


def log_benchmark(ctx, kpis: dict, label: str = "") -> pd.DataFrame:
    """Append a row of perception KPIs to ``outputs/_pipeline_nb/benchmarks.csv`` (one row per experiment), so
    model / parameter changes can be compared over time. Returns the whole table."""
    p = ROOT / "outputs" / "_pipeline_nb" / "benchmarks.csv"
    cfg = ctx.cfg
    row = dict(when=time.strftime("%Y-%m-%d %H:%M"), commit=git_commit(), label=label, run=ctx.P.run_name,
               video=Path(ctx.video).name, window=f"{ctx.windows[0][0]:.0f}-{ctx.windows[-1][1]:.0f}s",
               detector=f"{cfg.detection.backend}:{Path(cfg.detection.model_path).name}",
               tile=cfg.detection.tile, conf_player=cfg.detection.conf_player, conf_ball=cfg.detection.conf_ball,
               tracker=cfg.tracking.backend, ball=cfg.ball.detector, **kpis)
    df = pd.read_csv(p) if p.exists() else pd.DataFrame()
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    df.to_csv(p, index=False)
    return df


def config_table(cfg, sections: Sequence[str] = ("video", "detection", "tracking", "teams", "pitch", "ball")) -> pd.DataFrame:
    d = cfg.to_dict()
    rows = []
    for s in sections:
        for k, v in (d.get(s) or {}).items():
            if isinstance(v, (dict, list)) and len(json.dumps(v, default=str)) > 80:
                v = f"<{type(v).__name__} of {len(v)}>"
            rows.append(dict(section=s, key=k, value=v))
    return pd.DataFrame(rows)
