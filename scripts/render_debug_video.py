"""Render an annotated MP4 to check perception by eye: boxes coloured by team / role (same colours and legend as the
live view: staff, security and fans in their own colours), ball, projected pitch lines (camera calibration) and a
minimap of the people who are playing. Team names, pitch and camera come from the run itself (meta.json).

    python scripts/render_debug_video.py --video data/videos/match.mp4 --tracks outputs/match1 \
        --start 900 --end 960 --config configs/canedo_cfctv.yaml --out outputs/match1/debug_900_960.mp4
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from football_eval.config import load_config, resolve_path  # noqa: E402
from football_eval.pitch import Pitch  # noqa: E402
from football_eval.video import iter_frames, probe  # noqa: E402

from football_eval.live import TEAM_BGR, annotate_frame, row_label  # noqa: E402


def minimap(pitch, players, ball, w=200):
    s = w / (pitch.length + 8)
    h = int((pitch.width + 8) * s)
    m = np.full((h, w, 3), (40, 110, 50), np.uint8)
    to = lambda x, y: (int((x + 4) * s), int((y + 4) * s))
    for ln in pitch.lines(40):
        cv2.polylines(m, [np.array([to(x, y) for x, y in ln], np.int32)], False, (230, 230, 230), 1)
    for _, r in players.iterrows():
        if np.isfinite(r.x):
            cv2.circle(m, to(r.x, r.y), 3, TEAM_BGR.get(row_label(str(r.cls), int(r.team)), TEAM_BGR[None]), -1)
    if ball is not None and np.isfinite(ball[0]):
        cv2.circle(m, to(*ball), 3, (0, 255, 255), -1)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--tracks", required=True, help="perception output folder (tracks.csv, ball.csv, camera_log.csv)")
    ap.add_argument("--config", default=None, help="default: the configuration recorded by the run (meta.json)")
    ap.add_argument("--start", type=float, default=0)
    ap.add_argument("--end", type=float, default=60)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    from football_eval.config import resolve_path as _rp
    a.video = _rp(a.video) if getattr(a, "video", None) else a.video   # also finds it in the video libraries
    d = Path(a.tracks)
    import json
    meta = json.loads((d / "meta.json").read_text()) if (d / "meta.json").exists() else {}
    cfg = load_config(a.config)
    if a.config is None and meta.get("config"):          # same pitch / camera / fps as the run itself
        from football_eval.config import _merge
        try:
            _merge(cfg, meta["config"])
        except KeyError as e:                              # run made by an older version
            print(f"[warn] {e}: using --config / defaults for that key")
    names = ["Team A", "Team B"]
    if (d / "match_context.json").exists():
        names = json.loads((d / "match_context.json").read_text()).get("team_names") or names
    elif cfg.teams.team_names:
        names = list(cfg.teams.team_names)
    pitch = Pitch(cfg.pitch.length, cfg.pitch.width)
    tr = pd.read_csv(d / "tracks.csv")
    ball = pd.read_csv(d / "ball.csv") if (d / "ball.csv").exists() else None
    cam_log = pd.read_csv(d / "camera_log.csv") if (d / "camera_log.csv").exists() else None
    cam = None
    if cfg.pitch.homography == "ptz" and cam_log is not None:
        from football_eval.ptz import draw_template, load_ptz
        cam, _ = load_ptz(resolve_path(cfg.pitch.calibration_file), pitch)
    info = probe(a.video)
    out = a.out or str(d / f"debug_{int(a.start)}_{int(a.end)}.mp4")
    vw = None
    frames_done = set(tr.frame.unique())
    for f, t, img in iter_frames(a.video, cfg.video.analysis_fps, a.start, a.end):
        if f not in frames_done:
            continue
        v = img.copy()
        if cam is not None:
            r = cam_log[cam_log.frame == f]
            if len(r) and np.isfinite(r.pan.iloc[0]):
                v = draw_template(v, cam, (r.pan.iloc[0], r.tilt.iloc[0], r.f.iloc[0]), (255, 0, 255))
        q = tr[tr.frame == f]
        bxy, buv = None, None
        if ball is not None:
            b = ball[ball.frame == f]
            if len(b) and bool(b.valid.iloc[0]):
                bxy = (b.x.iloc[0], b.y.iloc[0])
                if "u" in b:
                    buv = (b.u.iloc[0], b.v.iloc[0])
        v = annotate_frame(v, q, names, ball_uv=buv, title=f"{int(t // 60):02d}:{t % 60:04.1f}  frame {f}  people {len(q)}")
        mm = minimap(pitch, q[q.cls.isin(["player", "goalkeeper", "referee"])], bxy)
        v[v.shape[0] - mm.shape[0]:, :mm.shape[1]] = mm
        if vw is None:
            vw = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*"mp4v"), cfg.video.analysis_fps, (v.shape[1], v.shape[0]))
        vw.write(v)
    if vw is not None:
        vw.release()
    print("->", out)


if __name__ == "__main__":
    main()
