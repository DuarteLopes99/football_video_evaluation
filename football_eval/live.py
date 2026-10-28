"""Live view of the processing: the frame being analysed with everything the pipeline sees drawn on it.

While ``run_perception`` works through the video, :class:`LivePreview` writes (about once per second of wall
time, so it costs < 1 % of the run):

``<out>/live/latest.jpg``   the current analysed frame with
                            * projected pitch lines (camera calibration of this frame), magenta,
                            * every detected person: box + track id, coloured by team / role. The label is the
                              *track's* running vote of the same classifier that makes the final assignment (the
                              players clicked in the setup, :class:`football_eval.teams.KitClassifier`), so it does
                              not flicker; ``*`` = a track you clicked yourself in the setup,
                            * the detector tiles of this frame (thin dashed), the unseen near-side strip on the
                              minimap when the camera does not show the near touchline,
                            * ball candidates of this frame (yellow), ignored overlay regions (hatched),
                            * a minimap with the people in pitch metres and a status bar.
``<out>/live/state.json``   frame, video time, progress, speed, ETA, camera status, counts.

The launcher's *Match screen* (Live mode) polls these two files. Turn it off with ``run_analysis.py --no-live``.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

TEAM_BGR = {"team0": (226, 144, 74), "team1": (60, 138, 239), "gk0": (255, 107, 155), "gk1": (162, 95, 255),
            "referee": (25, 25, 25), None: (190, 190, 190)}          # same colours as the setup / review UI


class LivePreview:
    def __init__(self, out_dir: str | Path, pitch, info, cfg, every_s: float = 1.0, total_frames: Optional[int] = None,
                 kit_samples: Optional[dict] = None, team_names=None, kit_report: Optional[dict] = None):
        self.dir = Path(out_dir) / "live"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.pitch, self.info, self.cfg = pitch, info, cfg
        self.every_s = every_s
        self.total = total_frames
        self.last = 0.0
        self.t0 = time.time()
        self.team_names = list(team_names or ["Team A", "Team B"])
        ab = [_short(n, "AB"[i]) for i, n in enumerate(self.team_names[:2])]
        self.label_txt = {"team0": ab[0], "team1": ab[1], "gk0": "GK " + ab[0], "gk1": "GK " + ab[1], "referee": "REF"}
        self.colors = dict(TEAM_BGR)
        self.kit_report = kit_report or {}
        self.lines = [ln for ln in pitch.lines(24)]
        self.n_written = 0
        self._state({"status": "starting", "started": self.t0})

    def begin(self):
        """Start of the frame loop (after the camera warm-up): speed and ETA are measured from here."""
        self.t0 = time.time()
        self.last = 0.0

    # ---------------------------------------------------------------- per frame
    def due(self) -> bool:
        return time.time() - self.last >= self.every_s

    def update(self, n: int, f: int, t: float, img: np.ndarray, people, ids, xy: np.ndarray, H, provider=None,
               balls_uv: Optional[list] = None, teams=None, labels: Optional[list] = None,
               manual: Optional[list] = None, tiles: Optional[list] = None):
        """Call every analysed frame; draws and writes only when ``every_s`` has elapsed. ``labels``: per person
        team0 | team1 | gk0 | gk1 | referee | None (the track's current vote)."""
        if not self.due():
            return
        self.last = time.time()
        try:
            v = self.draw(f, t, img, people, ids, xy, H, provider, balls_uv or [], labels, manual, tiles)
            ok, buf = cv2.imencode(".jpg", v, [cv2.IMWRITE_JPEG_QUALITY, 82])
            if ok:
                _atomic_write(self.dir / "latest.jpg", buf.tobytes())
            el = time.time() - self.t0
            fps = n / el if el > 0 else 0.0
            cam = _camera_status(provider)
            lab = [l for l in (labels or []) if l]
            st = dict(status="perception", frame=int(f), t=round(float(t), 2), n=int(n), fps=round(fps, 2),
                      elapsed_s=round(el, 1), n_people=int(len(people)), n_ball=len(balls_uv or []),
                      has_calibration=H is not None, camera=cam, width=int(v.shape[1]), height=int(v.shape[0]),
                      video_duration_s=round(self.info.duration_s, 1), updated=time.time(),
                      n_tiles=len(tiles) if tiles else None,
                      labelled={k: lab.count(k) for k in ("team0", "team1", "gk0", "gk1", "referee")},
                      clicked_tracks_in_view=int(sum(bool(m) for m in (manual or []))))
            if self.total:
                st["progress"] = round(min(1.0, n / self.total), 4)
                st["eta_s"] = round((self.total - n) / fps, 0) if fps > 0 else None
            self._state(st)
            self.n_written += 1
        except Exception as e:  # noqa: BLE001 — the preview must never stop the analysis
            self._state({"status": "perception", "preview_error": str(e), "updated": time.time()})

    def finish(self, status: str = "perception done", **extra):
        self._state(dict(status=status, updated=time.time(), elapsed_s=round(time.time() - self.t0, 1), **extra))

    # ---------------------------------------------------------------- drawing
    def draw(self, f, t, img, people, ids, xy, H, provider, balls_uv, labels=None, manual=None, tiles=None) -> np.ndarray:
        v = img.copy()
        h, w = v.shape[:2]
        sc = max(1.0, h / 720)                                      # text / line size for 1080p and up
        # ignored overlay regions (score bug, logos)
        for r in (self.cfg.video.ignore_regions or []):
            x0, y0, x1, y1 = map(int, r)
            sub = v[max(0, y0):y1, max(0, x0):x1]
            if sub.size:
                v[max(0, y0):y1, max(0, x0):x1] = (sub * 0.45).astype(np.uint8)
            cv2.rectangle(v, (x0, y0), (x1, y1), (120, 120, 120), 1)
        # detector tiles of this frame
        for x0, y0, x1, y1 in (tiles or []):
            _dashed_rect(v, (int(x0) + 1, int(y0) + 1), (int(x1) - 2, int(y1) - 2), (0, 200, 255), 18)
        # pitch lines of this frame's calibration
        if H is not None:
            from .review import project_h
            c = (self.pitch.length / 2, self.pitch.width / 2)
            for ln in self.lines:
                uv, front = project_h(H.H, ln, c)
                for k in range(len(ln) - 1):
                    if front[k] and front[k + 1] and np.all(np.abs(uv[k:k + 2]) < 4 * max(w, h)):
                        cv2.line(v, tuple(np.int32(uv[k])), tuple(np.int32(uv[k + 1])), (255, 0, 255), 1, cv2.LINE_AA)
        # people (unknown first, so team boxes stay on top)
        boxes = people.xyxy if len(people) else np.zeros((0, 4))
        cls = list(labels) if labels is not None else [None] * len(boxes)
        man = list(manual) if manual is not None else [False] * len(boxes)
        fs = 0.42 * sc
        for k in sorted(range(len(boxes)), key=lambda i: cls[i] is not None):
            x0, y0, x1, y1 = map(int, boxes[k])
            col = self.colors.get(cls[k], TEAM_BGR[None])
            cv2.rectangle(v, (x0, y0), (x1, y1), col, 2 if cls[k] else 1)
            lab = (self.label_txt.get(cls[k], "") + " " if cls[k] else "") + (str(int(ids[k])) if ids[k] >= 0 else "")
            lab = (lab.strip() + ("*" if man[k] else "")).strip()
            if lab:
                (tw, th), _ = cv2.getTextSize(lab, cv2.FONT_HERSHEY_SIMPLEX, fs, 1)
                yb = max(th + 4, y0 - 2)
                cv2.rectangle(v, (x0, yb - th - 4), (x0 + tw + 4, yb + 1), col, -1)
                txt_col = (255, 255, 255) if sum(col) < 380 else (20, 20, 20)
                cv2.putText(v, lab, (x0 + 2, yb - 2), cv2.FONT_HERSHEY_SIMPLEX, fs, txt_col, 1, cv2.LINE_AA)
        for (u, vv) in balls_uv:
            cv2.circle(v, (int(u), int(vv)), int(7 * sc), (0, 255, 255), 2, cv2.LINE_AA)
        # minimap (+ unseen near-side strip)
        near_y = None
        if H is not None:
            from .perception import visible_near_y
            near_y = visible_near_y(H, self.pitch, img.shape)
        mm = self.minimap(xy, cls, near_y=near_y, w=int(170 * sc))
        mh, mw = mm.shape[:2]
        if mh < h and mw < w:
            v[h - mh - 4:h - 4, 4:4 + mw] = cv2.addWeighted(v[h - mh - 4:h - 4, 4:4 + mw], 0.25, mm, 0.75, 0)
        self._legend(v, sc)
        # status bar
        cam = _camera_status(provider)
        txt = f"{_clock(t)}  frame {f}  people {len(boxes)}  ball cand. {len(balls_uv)}"
        if tiles:
            txt += f"  tiles {len(tiles)}"
        if cam:
            txt += f"  camera {cam.get('src', '')} err {cam.get('line_err', '–')}"
        elif H is None:
            txt += "  NO CALIBRATION"
        if near_y is not None and np.isfinite(near_y) and near_y < self.pitch.width - 0.5:
            txt += f"  near touchline out of view (view ends {self.pitch.width - near_y:.0f} m before it)"
        bh = int(18 * sc)
        cv2.rectangle(v, (0, 0), (w, bh), (0, 0, 0), -1)
        cv2.putText(v, txt, (5, bh - int(5 * sc)), cv2.FONT_HERSHEY_SIMPLEX, 0.42 * sc, (220, 255, 220), 1, cv2.LINE_AA)
        return v

    def _legend(self, v, sc):
        items = [("team0", self.team_names[0]), ("team1", self.team_names[1] if len(self.team_names) > 1 else "Team B"),
                 ("gk0", "GK " + self.label_txt["team0"]), ("gk1", "GK " + self.label_txt["team1"]),
                 ("referee", "referee"), (None, "unknown")]
        fs = 0.42 * sc
        x = v.shape[1] - int(170 * sc)
        y = int(30 * sc)
        cv2.rectangle(v, (x - 8, y - int(14 * sc)), (v.shape[1] - 4, y + int(17 * sc) * len(items) + int(6 * sc)),
                      (0, 0, 0), -1)
        for k, (c, name) in enumerate(items):
            yy = y + int(17 * sc) * k
            cv2.rectangle(v, (x, yy - int(9 * sc)), (x + int(12 * sc), yy + int(2 * sc)), self.colors.get(c, TEAM_BGR[None]), -1)
            cv2.putText(v, str(name)[:18], (x + int(18 * sc), yy), cv2.FONT_HERSHEY_SIMPLEX, fs, (235, 235, 235), 1, cv2.LINE_AA)
        cv2.putText(v, "* = clicked by you", (x, y + int(17 * sc) * len(items)), cv2.FONT_HERSHEY_SIMPLEX, 0.36 * sc,
                    (200, 200, 200), 1, cv2.LINE_AA)

    def minimap(self, xy, cls, w: int = 170, near_y: Optional[float] = None) -> np.ndarray:
        p = self.pitch
        s = w / (p.length + 8)
        h = int((p.width + 8) * s)
        m = np.full((h, w, 3), (40, 105, 48), np.uint8)
        to = lambda x, y: (int((x + 4) * s), int((y + 4) * s))
        if near_y is not None and np.isfinite(near_y) and near_y < p.width:
            a, b = to(0, max(0.0, near_y)), to(p.length, p.width + 4)
            ov = m.copy()
            cv2.rectangle(ov, a, b, (25, 45, 30), -1)
            m = cv2.addWeighted(ov, 0.7, m, 0.3, 0)
        for ln in self.lines:
            cv2.polylines(m, [np.array([to(x, y) for x, y in ln], np.int32)], False, (225, 225, 225), 1)
        for k in range(len(xy)):
            x, y = xy[k]
            if np.isfinite(x) and np.isfinite(y):
                cv2.circle(m, to(x, y), 3, self.colors.get(cls[k] if k < len(cls) else None, TEAM_BGR[None]), -1)
        return m

    def _state(self, d: dict):
        _atomic_write(self.dir / "state.json", json.dumps(d, default=float).encode())


def _short(name: str, fallback: str) -> str:
    """Short team label for the boxes: initials of a multi-word name, else the first 3 letters."""
    ws = [w for w in str(name or "").replace(".", " ").split() if w[:1].isalnum()]
    if not ws:
        return fallback
    if len(ws) >= 2 and ws[0].upper() in ("UD", "FC", "SC", "CD", "AD", "GD", "SL", "CF", "AC", "US", "CS", "GS"):
        return ws[1][:3].upper()
    return ws[0][:3].upper()


def _dashed_rect(img, p0, p1, col, dash: int = 16):
    (x0, y0), (x1, y1) = p0, p1
    for a, b, fixed, horiz in ((x0, x1, y0, True), (x0, x1, y1, True), (y0, y1, x0, False), (y0, y1, x1, False)):
        for s_ in range(int(a), int(b), 2 * dash):
            e = min(int(b), s_ + dash)
            if horiz:
                cv2.line(img, (s_, fixed), (e, fixed), col, 1)
            else:
                cv2.line(img, (fixed, s_), (fixed, e), col, 1)


def _camera_status(provider) -> Optional[dict]:
    log = getattr(provider, "log", None)
    if not log:
        return None
    r = log[-1]
    le = r.get("line_err")
    return dict(src=str(r.get("src", "")), line_err=None if le is None or not np.isfinite(le) else round(float(le), 1))


def _clock(t: float) -> str:
    m, s = divmod(int(t), 60)
    h, m = divmod(m, 60)
    return f"{h:d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def _atomic_write(p: Path, data: bytes):
    tmp = p.with_name(p.name + ".tmp")
    with open(tmp, "wb") as fh:
        fh.write(data)
    os.replace(tmp, p)
