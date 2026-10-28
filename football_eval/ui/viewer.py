"""Data for the launcher's *Match screen* (replay mode): everything the analysis produced, per analysed frame, so
the browser can draw it on top of the playing video.

Per analysed frame: the pitch -> image homography of that frame (PTZ pose or static calibration; sign-normalised
so that ``w > 0`` means "in front of the camera"), camera-quality flag, every person box with team / role / track
id, the people in pitch metres (minimap) and the ball in pitch metres. Once per run: pitch size and lines, the
analysed image size, team names, events and opportunities. Loaded once per run and cached.
"""
from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from ..config import load_config, resolve_path
from ..pitch import Pitch

EVENT_TYPES = ("goal", "shot", "penalty", "corner", "free_kick", "throw_in", "goal_kick", "kickoff", "foul_proxy",
               "offside_suspected", "pass", "interception", "ball_recovery", "yellow", "red", "sub")


class RunOverlay:
    def __init__(self, run_dir: str | Path):
        d = Path(run_dir)
        self.dir = d
        self.meta = _json(d / "meta.json") or {}
        rc = self.meta.get("config", {})
        cfg = load_config()
        for sec in ("pitch", "video", "teams"):
            for k, v in (rc.get(sec) or {}).items():
                if hasattr(getattr(cfg, sec), k):
                    setattr(getattr(cfg, sec), k, v)
        self.cfg = cfg
        self.tracks = pd.read_csv(d / "tracks.csv") if (d / "tracks.csv").exists() else pd.DataFrame()
        self.ball = pd.read_csv(d / "ball.csv") if (d / "ball.csv").exists() else None
        self.frames = pd.read_csv(d / "frames.csv") if (d / "frames.csv").exists() else None
        self.cam_log = pd.read_csv(d / "camera_log.csv") if (d / "camera_log.csv").exists() else None
        self.pitch = Pitch(cfg.pitch.length, cfg.pitch.width)
        self.cam, self.H = None, None
        if cfg.pitch.homography == "ptz" and cfg.pitch.calibration_file and self.cam_log is not None:
            from ..ptz import load_ptz
            self.cam, _ = load_ptz(resolve_path(cfg.pitch.calibration_file), self.pitch)
            self.pitch = self.cam.pitch
        elif cfg.pitch.homography == "static":
            from ..pitch import calibration_from_dict, load_calibration
            try:
                if cfg.pitch.calibration_points:
                    self.H = calibration_from_dict({"points": cfg.pitch.calibration_points}, self.pitch)
                elif cfg.pitch.calibration_file:
                    self.H = load_calibration(resolve_path(cfg.pitch.calibration_file), self.pitch)
            except Exception:  # noqa: BLE001 — no lines, the rest still draws
                self.H = None
        v = self.meta.get("video", {})
        rw = (rc.get("video") or {}).get("resize_width")
        W, Hh = int(v.get("width") or 0), int(v.get("height") or 0)
        self.size = [int(rw), int(round(Hh * rw / W))] if (rw and W) else [W, Hh]
        if self.tracks is not None and len(self.tracks):
            self.tracks = self.tracks.sort_values("frame").reset_index(drop=True)
        if self.frames is not None and len(self.frames):
            self.fr = self.frames.sort_values("frame").reset_index(drop=True)
        elif len(self.tracks):
            g = self.tracks.groupby("frame").t.first().reset_index()
            self.fr = g
        else:
            self.fr = pd.DataFrame(columns=["frame", "t"])
        self._has_boxes = all(c in self.tracks for c in ("x0", "y0", "x1", "y1"))   # synthetic runs: positions only
        self._tf = self.tracks.frame.to_numpy() if len(self.tracks) else np.array([], int)
        self._ball = self.ball.set_index("frame") if self.ball is not None and len(self.ball) else None
        self._cam = self.cam_log.drop_duplicates("frame").set_index("frame") if self.cam_log is not None else None
        self.ctx = _json(d / "match_context.json") or {}
        self.lock = threading.Lock()

    # ---------------------------------------------------------------- static info
    def info(self) -> dict:
        p = self.pitch
        ev = self.events()
        opps = _json(self.dir / "opportunities.json") or []
        t = self.fr.t.to_numpy() if len(self.fr) else np.array([0.0])
        return dict(size=self.size, pitch=[float(p.length), float(p.width)],
                    lines=[np.round(ln, 2).tolist() for ln in p.lines(24)],
                    team_names=self.ctx.get("team_names") or list(self.cfg.teams.team_names),
                    periods=self.ctx.get("periods"), t_range=[float(t.min()), float(t.max())],
                    analysis_fps=float(self.cfg.video.analysis_fps), has_camera=self.cam is not None or self.H is not None,
                    n_frames=int(len(self.fr)), events=ev, opportunities=opps,
                    ignore_regions=self.cfg.video.ignore_regions or [])

    def events(self) -> list:
        f = self.dir / "events.csv"
        if not f.exists():
            return []
        e = pd.read_csv(f)
        e = e[e.type.isin(EVENT_TYPES)]
        out = []
        for _, r in e.iterrows():
            row = dict(t=round(float(r.t), 2), type=str(r.type), team=int(r.team) if pd.notna(r.team) else -1)
            for k in ("x", "y", "x_end", "y_end", "xg"):
                if k in r and pd.notna(r[k]):
                    row[k] = round(float(r[k]), 2 if k != "xg" else 3)
            for k in ("outcome", "clock", "validation"):
                if k in r and pd.notna(r[k]):
                    row[k] = str(r[k])
            out.append(row)
        return out

    # ---------------------------------------------------------------- per frame
    def _H(self, frame: int) -> Optional[list]:
        if self.cam is not None and self._cam is not None:
            if frame not in self._cam.index:
                return None
            r = self._cam.loc[frame]
            if not np.isfinite(r.pan):
                return None
            M = np.asarray(self.cam.homography((float(r.pan), float(r.tilt), float(r.f))).H, float)
        elif self.H is not None:
            M = np.asarray(self.H.H, float)
        else:
            return None
        c = np.array([self.pitch.length / 2, self.pitch.width / 2, 1.0]) @ M.T
        if c[2] < 0:
            M = -M
        return np.round(M / (np.abs(M).max() or 1.0), 9).tolist()

    def window(self, t0: float, t1: float) -> dict:
        with self.lock:
            fr = self.fr[(self.fr.t >= t0) & (self.fr.t <= t1)]
            frames = []
            for _, r in fr.iterrows():
                f = int(r.frame)
                a, b = np.searchsorted(self._tf, f), np.searchsorted(self._tf, f, side="right")
                q = self.tracks.iloc[a:b]
                boxes = np.c_[q[["x0", "y0", "x1", "y1"]].to_numpy().round(1), q.team.to_numpy(),
                              q.track_id.to_numpy()].tolist() if len(q) and self._has_boxes else []
                cls = q.cls.astype(str).tolist() if len(q) and self._has_boxes else []
                pxy = q[["x", "y"]].to_numpy()
                pts = [[round(float(x), 1), round(float(y), 1), int(tm), c[:1]] for (x, y), tm, c in
                       zip(pxy, q.team.to_numpy(), q.cls.astype(str).tolist()) if np.isfinite(x) and np.isfinite(y) and c != "outside"]
                ball = None
                if self._ball is not None and f in self._ball.index:
                    br = self._ball.loc[f]
                    if isinstance(br, pd.DataFrame):
                        br = br.iloc[0]
                    if bool(br.get("valid", True)) and np.isfinite(br.x) and np.isfinite(br.y):
                        ball = [round(float(br.x), 2), round(float(br.y), 2), bool(br.get("interp", False))]
                cam_ok = bool(r.cam_ok) if "cam_ok" in r and pd.notna(r.cam_ok) else True
                frames.append(dict(f=f, t=round(float(r.t), 3), H=self._H(f), ok=cam_ok, b=boxes, c=cls, p=pts,
                                   ball=ball))
            return dict(t0=t0, t1=t1, frames=frames)


_cache: dict = {}
_cache_lock = threading.Lock()


def get_overlay(run_dir: Path) -> RunOverlay:
    """Cached per run; reloaded when tracks.csv / events.csv change (a re-run of the analytics)."""
    key = str(run_dir)
    stamp = tuple((run_dir / f).stat().st_mtime if (run_dir / f).exists() else 0
                  for f in ("tracks.csv", "events.csv", "opportunities.json", "camera_log.csv"))
    with _cache_lock:
        hit = _cache.get(key)
        if hit is None or hit[0] != stamp:
            _cache[key] = (stamp, RunOverlay(run_dir))
        return _cache[key][1]


def _json(p: Path):
    try:
        return json.loads(p.read_text())
    except (OSError, json.JSONDecodeError):
        return None
