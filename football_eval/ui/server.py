"""Local web UI for the manual steps: match setup (before processing) and review (after processing).

Standard library HTTP server + OpenCV — no extra dependency. Run it through ``scripts/match_ui.py``; it opens
http://127.0.0.1:8765 in the browser. Everything is saved to plain JSON files as you go
(``match_setup.json`` / ``<run>/review.json``).
"""
from __future__ import annotations

import json
import threading
import traceback
from collections import OrderedDict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Optional
from urllib.parse import parse_qs, urlparse

import cv2
import numpy as np

from ..config import load_config, resolve_path
from ..match_setup import (KIT_CLASSES, KIT_LABELS, apply_to_config, checklist, load_setup, new_setup, save_setup)
from ..pitch import Homography, Pitch
from ..video import probe

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


class FrameSource:
    """Random access to video frames with a small cache; reads forward instead of seeking for short jumps
    (H.264 seeks are slow, and clip playback is sequential)."""

    def __init__(self, path: str, cache: int = 400):
        self.info = probe(path)
        self.cap = cv2.VideoCapture(path)
        self.pos = 0
        self.lock = threading.Lock()
        self.cache: OrderedDict = OrderedDict()
        self.max = cache

    def get(self, idx: int) -> Optional[np.ndarray]:
        idx = int(max(0, min(self.info.n_frames - 1, idx)))
        with self.lock:
            if idx in self.cache:
                self.cache.move_to_end(idx)
                return self.cache[idx]
            if not (0 <= idx - self.pos <= 90):
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                self.pos = idx
            while self.pos < idx:
                self.cap.grab()
                self.pos += 1
            ok, img = self.cap.read()
            self.pos += 1
            if not ok:
                return None
            self.cache[idx] = img
            if len(self.cache) > self.max:
                self.cache.popitem(last=False)
            return img

    def jpeg(self, idx: int, quality: int = 88) -> Optional[bytes]:
        img = self.get(idx)
        if img is None:
            return None
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
        return buf.tobytes() if ok else None


class App:
    def __init__(self, mode: str, video: str, setup_path: str, config: Optional[str] = None,
                 run_dir: Optional[str] = None):
        self.mode = mode
        self.video_path = video
        self.frames = FrameSource(video)
        self.setup_path = Path(setup_path)
        self.cfg_path = config
        info = self.frames.info
        vinfo = dict(fps=info.fps, n_frames=info.n_frames, width=info.width, height=info.height,
                     duration_s=info.duration_s)
        if self.setup_path.exists():
            self.setup = load_setup(self.setup_path)
            self.setup["video"].update(vinfo)
        else:
            self.setup = new_setup(_rel(video), vinfo)
            if config:
                self.setup["meta"]["config"] = _rel(config)
        self.cfg = load_config(config)
        apply_to_config(self.setup, self.cfg)
        self._detector = None
        self._cam = None
        self.run_dir = Path(run_dir) if run_dir else None
        self.review = None
        if mode == "review":
            self._init_review()

    # ------------------------------------------------------------------ helpers
    @property
    def pitch(self) -> Pitch:
        return Pitch(float(self.setup["pitch"]["length"]), float(self.setup["pitch"]["width"]))

    def detector(self):
        if self._detector is None:
            from ..detection import build_detector
            self._detector = build_detector(self.cfg)
        return self._detector

    def camera(self):
        cf = self.setup.get("calibration", {}).get("camera_file") or \
            (self.cfg.pitch.calibration_file if self.cfg.pitch.homography == "ptz" else None)
        if not cf:
            return None
        if self._cam is None or self._cam[0] != cf:
            from ..ptz import load_ptz
            cam, _ = load_ptz(resolve_path(cf), self.pitch)
            self._cam = (cf, cam)
        return self._cam[1]

    def state(self) -> dict:
        p = self.pitch
        st = dict(mode=self.mode, setup=self.setup, setup_path=str(self.setup_path), video=self.setup["video"],
                  video_name=Path(self.video_path).name,
                  landmarks=p.landmarks(), pitch_lines=[ln.round(2).tolist() for ln in p.lines(30)],
                  pitch_size=[p.length, p.width], kit_classes=list(KIT_CLASSES), kit_labels=KIT_LABELS,
                  checklist=checklist(self.setup, ROOT), config=self.cfg_path)
        if self.mode == "review":
            from ..review import lost_stretches
            fr = self.geo.frames
            st.update(review=self.review, review_path=str(self.run_dir / "review.json"), run=str(self.run_dir),
                      candidates=self.candidates, team_names=self.team_names,
                      analysis_fps=self.cfg.video.analysis_fps, lost=lost_stretches(fr),
                      camera_coverage=float(fr.cam_ok.mean()) if fr is not None and "cam_ok" in fr else None,
                      has_ptz=self.geo.cam is not None, has_calibration=self.geo.cam is not None or self.geo.H is not None)
        return st

    # ------------------------------------------------------------------ setup actions
    def kit_sample(self, frame: int, box) -> dict:
        """A clicked / dragged player -> kit sample. A dragged box is snapped to the detector's box of that player
        (dragged boxes are usually loose: grass in the shirt crop made samples unrecognisable)."""
        from ..teams import color_descriptor, grass_model, snap_box, torso_crop
        img = self.frames.get(frame)
        drawn = [round(float(v), 1) for v in box]
        snapped = False
        try:
            dets = self._detections(frame)
            sb = snap_box(box, np.array([d["box"] for d in dets]).reshape(-1, 4)) if dets else None
            if sb is not None:
                snapped = bool(np.abs(np.asarray(sb) - np.asarray(box, float)).max() > 1.0)
                box = [float(v) for v in sb]
        except Exception:  # noqa: BLE001 — no detector: keep the box as drawn
            pass
        crop = torso_crop(img, box)
        if crop is None:
            raise ValueError("box too small")
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        grass = (hsv[..., 0] > 30) & (hsv[..., 0] < 90) & (hsv[..., 1] > 50) & (hsv[..., 2] > 40)
        px = crop[~grass] if (~grass).sum() >= 6 else crop.reshape(-1, 3)
        b, g, r = np.median(px, 0).astype(int)
        desc = color_descriptor(crop, grass_model(img))
        ok, buf = cv2.imencode(".png", cv2.resize(crop, (24, int(24 * crop.shape[0] / max(crop.shape[1], 1)) or 24),
                                                  interpolation=cv2.INTER_NEAREST))
        import base64
        return dict(t=round(frame / self.frames.info.fps, 3), frame=int(frame), box=[round(float(v), 1) for v in box],
                    drawn=drawn if snapped else None,
                    rgb=f"#{r:02x}{g:02x}{b:02x}", desc=None if desc is None else np.round(desc, 4).tolist(),
                    thumb="data:image/png;base64," + base64.b64encode(buf.tobytes()).decode() if ok else None)

    def _detections(self, frame: int) -> list:
        if not hasattr(self, "_det_cache"):
            self._det_cache = OrderedDict()
        if frame not in self._det_cache:
            img = self.frames.get(frame)
            det = self.detector()(img).of("player", "goalkeeper", "referee")
            self._det_cache[frame] = [dict(box=[round(float(v), 1) for v in b], conf=round(float(c), 2))
                                      for b, c in zip(det.xyxy, det.conf)]
            while len(self._det_cache) > 30:
                self._det_cache.popitem(last=False)
        return self._det_cache[frame]

    def detect(self, frame: int) -> list:
        return self._detections(frame)

    def fit(self, frame: int, points: dict, pad: float = 40.0) -> dict:
        """Preview of the calibration on one frame from the clicked landmarks, with per-point residuals.

        ``pad``: px around the frame where the projected pitch model is still drawn (the UI's "outside the picture"
        margin, where off-screen landmarks can be placed)."""
        from ..ptz import template_segments
        from ..review import project_h
        p = self.pitch
        lm = p.landmarks()
        names = [n for n in points if n in lm]
        uv = np.array([points[n] for n in names], float)
        X = np.array([lm[n] for n in names], float)
        A, B = template_segments(p, 1.0)
        out = {"n": len(names)}
        cam = self.camera() if (self.setup["calibration"].get("mode") == "ptz" or self.cfg.pitch.homography == "ptz") \
            else None
        if cam is not None and len(names) >= 2:
            from ..ptz import Evidence, pose_from_landmarks
            img = self.frames.get(frame)
            ev = Evidence(img, None, self.cfg.video.ignore_regions)
            # fast: multi-start on the clicked points only, then one refinement that also uses the painted lines
            pose, _ = pose_from_landmarks(cam, None, {n: points[n] for n in names}, cam.pitch, pan_step=0.1,
                                          use_lines=False)
            pose = _refine_pose(cam, ev, X, uv, pose)
            proj, _ = cam.project_ours(X, pose)
            a, fa = cam.project_ours(A, pose)
            b, fb = cam.project_ours(B, pose)
            from ..ptz import Aligner
            line_err = float(Aligner(cam).mean_line_error(pose, ev)) if ev.n_line >= 50 else None
            out.update(method="ptz", pose=[float(v) for v in pose], line_err=line_err)
        elif len(names) >= 4:
            if _max_collinear(X) > len(X) - 2:
                return dict(out, method=None, message="too many of your points lie on ONE pitch line (e.g. the halfway "
                            "line): a homography needs >= 4 points with no 3 on a line — add points off that line")
            H = Homography.fit(uv, X)
            proj = H.to_image(X)
            c = (p.length / 2, p.width / 2)
            a, fa = project_h(H.H, A, c)
            b, fb = project_h(H.H, B, c)
            out.update(method="homography", H=H.H.tolist())
        else:
            out.update(method=None, message="click >= 4 landmarks (>= 2 with a solved PTZ venue camera)")
            return out
        from ..review import visible_segments
        info = self.frames.info
        ok = fa & fb & np.all(np.abs(a) < 5e3, 1) & np.all(np.abs(b) < 5e3, 1) & \
            visible_segments(a, b, info.width, info.height, margin=max(40.0, float(pad)))
        res = np.linalg.norm(proj - uv, axis=1)
        out["near_check"] = self._near_check(out, p, info)
        out["off_screen"] = [n for n in names if not (0 <= points[n][0] <= info.width and 0 <= points[n][1] <= info.height)]
        out.update(lines=np.round(np.c_[a[ok], b[ok]], 1).tolist(),
                   residuals={n: round(float(r), 2) for n, r in zip(names, res)},
                   mean_err=round(float(res.mean()), 2), max_err=round(float(res.max()), 2))
        return out

    def _near_check(self, out: dict, p: Pitch, info) -> Optional[str]:
        """Where does this calibration put the near touchline? Compared with what the user said in Match & pitch."""
        if out.get("method") != "homography":
            return None
        H = Homography(np.asarray(out["H"], float))
        xs = np.linspace(0, p.length, 41)
        uv = H.to_image(np.c_[xs, np.full_like(xs, p.width)])
        inside = (uv[:, 0] >= 0) & (uv[:, 0] <= info.width)
        if not inside.any():
            return None
        v = float(np.median(uv[inside, 1]))
        said = self.setup.get("pitch", {}).get("near_touchline", "visible")
        if v > info.height + 2:
            msg = f"With this fit the near touchline runs {v - info.height:.0f} px below the picture (out of view)."
            if said == "visible":
                msg += " You said it is visible: check the pitch width in Match & pitch, or the landmarks."
            return msg
        msg = f"With this fit the near touchline is inside the picture ({info.height - v:.0f} px above the bottom edge)."
        if said == "hidden":
            msg += " You said it is out of view: the width or a near-side point is probably wrong."
        return msg

    def save_setup(self, setup: dict) -> dict:
        self.setup = setup
        save_setup(setup, self.setup_path)
        self.cfg = load_config(self.cfg_path)
        apply_to_config(self.setup, self.cfg)
        return dict(saved=str(self.setup_path), checklist=checklist(self.setup, ROOT))

    # ------------------------------------------------------------------ review
    def _init_review(self):
        import pandas as pd
        from ..review import RunGeometry, candidate_events, load_review, sample_check_frames
        d = self.run_dir
        self.review = load_review(d / "review.json")
        meta = json.loads((d / "meta.json").read_text()) if (d / "meta.json").exists() else {}
        rc = meta.get("config", {})
        # the run's own pitch / camera settings drive the overlay
        for sec in ("pitch", "video", "teams"):
            for k, v in (rc.get(sec) or {}).items():
                if hasattr(getattr(self.cfg, sec), k):
                    setattr(getattr(self.cfg, sec), k, v)
        apply_to_config(self.setup, self.cfg)
        self.geo = RunGeometry(str(d), self.cfg)
        frames = self.geo.frames if self.geo.frames is not None else pd.DataFrame(
            {"frame": sorted(self.geo.tracks.frame.unique())}).assign(t=lambda q: q.frame / self.frames.info.fps)
        if not self.review.get("frame_checks"):
            self.review["frame_checks"] = sample_check_frames(frames)
        ev = pd.read_csv(d / "events.csv") if (d / "events.csv").exists() else None
        auto = ev[~ev.source.isin(["manual", "match_sheet"])] if ev is not None and "source" in ev else ev
        self.candidates = candidate_events(auto)
        ctxp = d / "match_context.json"
        self.team_names = json.loads(ctxp.read_text()).get("team_names") if ctxp.exists() else \
            [t["name"] for t in self.setup["teams"]]

    def geometry(self, frame: int) -> dict:
        # tracks are stored only for analysed frames: use the nearest analysed one
        fr = self.geo.tracks.frame.to_numpy()
        if len(fr):
            u = np.unique(fr)
            j = int(u[np.argmin(np.abs(u - frame))])
            near = abs(j - frame) <= max(1, int(self.frames.info.fps / max(self.cfg.video.analysis_fps, 1)))
        else:
            j, near = frame, False
        return dict(frame=j, exact=bool(j == frame), lines=self.geo.lines(j) if near else [],
                    boxes=self.geo.boxes(j) if near else [], ball=self.geo.ball_at(j) if near else None)

    def to_pitch(self, frame: int, uv: list, pose=None) -> dict:
        """Clicked image points -> pitch metres, with the camera pose of that frame (or a repair pose)."""
        if pose is not None and self.geo.cam is not None:
            H, ok, src = self.geo.cam.homography(tuple(pose)), True, "your repair pose"
        else:
            H, ok, src = self.geo.homography(frame)
        if H is None:
            return dict(xy=None, ok=False, source=src)
        xy = H.to_pitch(np.asarray(uv, float).reshape(-1, 2))
        return dict(xy=np.round(xy, 2).tolist(), ok=bool(ok), source=src)

    def save_review(self, review: dict) -> dict:
        from ..review import save_review, validation_summary
        self.review = review
        save_review(review, self.run_dir / "review.json")
        return dict(saved=str(self.run_dir / "review.json"),
                    summary=validation_summary(review).to_dict(orient="records"))


def _max_collinear(X: np.ndarray, tol: float = 0.3) -> int:
    """Largest number of pitch points on one straight line (metres)."""
    best = 2 if len(X) >= 2 else len(X)
    for i in range(len(X)):
        for j in range(i + 1, len(X)):
            d = X[j] - X[i]
            n = np.linalg.norm(d)
            if n < 1e-6:
                continue
            v = X - X[i]
            dist = np.abs(d[0] * v[:, 1] - d[1] * v[:, 0]) / n
            best = max(best, int((dist < tol).sum()))
    return best


def _refine_pose(cam, ev, X, uv, p0):
    from scipy.optimize import least_squares
    from ..ptz import Aligner
    al = Aligner(cam)

    def res(p):
        q, front = cam.project_ours(X, p)
        r = np.where(front[:, None], q - uv, 500.0).ravel() / 2.0
        return np.r_[r, 0.3 * al.residuals(p, ev)] if ev.n_line >= 50 else r
    try:
        s = least_squares(res, p0, bounds=([-1.7, -0.3, 150], [1.7, 0.7, 8000]), x_scale=[0.01, 0.01, 20],
                          diff_step=1e-4, loss="soft_l1", max_nfev=80)
        return s.x
    except ValueError:
        return np.asarray(p0, float)


def _rel(p: str) -> str:
    try:
        return str(Path(p).resolve().relative_to(ROOT))
    except ValueError:
        return str(p)


def _dumps(obj) -> bytes:
    return json.dumps(obj, default=lambda o: o.item() if hasattr(o, "item") else str(o)).encode()


def app_html(prefix: str = "") -> bytes:
    """The setup / review page; ``prefix`` mounts it under a sub-path (the launcher serves it at /m/)."""
    html = (HERE / "app.html").read_text(encoding="utf-8")
    if prefix:
        html = html.replace("const r = await fetch(path,", f"const r = await fetch(\"{prefix}\" + path,")
        html = html.replace('"/frame/"', f'"{prefix}/frame/"')
    return html.encode("utf-8")


def route_get(app: App, path: str, q: dict):
    """GET routes of the manual UI -> (status, body bytes, content type, cacheable)."""
    if path.startswith("/frame/"):
        idx = int(path.split("/")[-1].split(".")[0])
        b = app.frames.jpeg(idx)
        return (200, b, "image/jpeg", True) if b else (404, b"", "text/plain", False)
    if path == "/api/state":
        return 200, _dumps(app.state()), "application/json", False
    if path == "/api/checklist":
        return 200, _dumps(checklist(app.setup, ROOT)), "application/json", False
    if path == "/api/detect":
        return 200, _dumps(app.detect(int(q["frame"]))), "application/json", False
    if path == "/api/geometry":
        return 200, _dumps(app.geometry(int(q["frame"]))), "application/json", False
    return 404, b"not found", "text/plain", False


def route_post(app: App, path: str, body: dict):
    """POST routes of the manual UI -> JSON-able result (raises KeyError for an unknown route)."""
    if path == "/api/setup":
        return app.save_setup(body)
    if path == "/api/kit_sample":
        return app.kit_sample(int(body["frame"]), body["box"])
    if path == "/api/fit":
        return app.fit(int(body["frame"]), body["points"], float(body.get("pad") or 40.0))
    if path == "/api/review":
        return app.save_review(body)
    if path == "/api/to_pitch":
        return app.to_pitch(int(body["frame"]), body["uv"], body.get("pose"))
    raise KeyError(path)


def make_handler(app: App):
    html = app_html()

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, code, body: bytes, ctype="application/json", cache=False):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "max-age=3600" if cache else "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, obj, code=200):
            self._send(code, _dumps(obj))

        def do_GET(self):
            try:
                u = urlparse(self.path)
                q = {k: v[0] for k, v in parse_qs(u.query).items()}
                if u.path in ("/", "/index.html"):
                    return self._send(200, html, "text/html; charset=utf-8")
                code, body, ctype, cache = route_get(app, u.path, q)
                self._send(code, body, ctype, cache)
            except Exception as e:  # noqa: BLE001 — report to the UI
                traceback.print_exc()
                self._json({"error": str(e)}, 500)

        def do_POST(self):
            try:
                n = int(self.headers.get("Content-Length", 0))
                body = json.loads(self.rfile.read(n) or b"{}")
                return self._json(route_post(app, urlparse(self.path).path, body))
            except KeyError as e:
                if str(e).strip("'").startswith("/"):
                    return self._send(404, b"not found", "text/plain")
                traceback.print_exc()
                self._json({"error": str(e)}, 500)
            except Exception as e:  # noqa: BLE001
                traceback.print_exc()
                self._json({"error": str(e)}, 500)

    return H


def serve(app: App, host: str = "127.0.0.1", port: int = 8765, open_browser: bool = True):
    srv = ThreadingHTTPServer((host, port), make_handler(app))
    url = f"http://{host}:{port}/"
    print(f"{app.mode} UI on {url}  (Ctrl+C to stop; everything is saved as you go)")
    if open_browser:
        import webbrowser
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")
    finally:
        srv.server_close()
