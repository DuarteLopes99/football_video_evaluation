"""Offside studio — the web workflow behind the dashboard's **Offside** tab (also runs on its own:
``python offside_studio.py``).

A *venue* is one camera set-up, kept in ``venues/<name>/``:

    venue.json         the config — the same format as configs/*.json (so calibrate_camera.py / run_offside.py
                       work on it) plus a "studio" block with every frame and its clicked markings
    images/            frames added in the dashboard (calibration frames and plays to judge)
    camera_model.json  the calibrated camera (+ camera_model_background.png for a stationary camera)
    results/<frame>/   decision images + offside_result.json of each judged frame

Workflow (one tab, three steps):
  1. Frames & markings — add images; on each, click the pitch markings you can see (2 clicks per straight line,
     3+ per arc, or a box around the centre circle), a few turf points inside the field, overlay boxes to ignore.
     "Check" shows which white-line pixels the hints pick up.
  2. Calibrate — background job running ``calibrate_camera.py`` on venue.json (stationary or panning camera);
     afterwards the calibrated pitch template is drawn on every frame as a check.
  3. Decide — choose a frame, teams (kit colours sampled with a click), attacking direction, ball (auto / click /
     ignore) → OFFSIDE / ONSIDE per attacker, with or without the lines and the metrics (re-drawn instantly).

Every heavy step runs the same code as the CLIs; nothing here changes the geometry.
"""
from __future__ import annotations

import copy
import hashlib
import json
import mimetypes
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import traceback
import uuid
from collections import OrderedDict
from pathlib import Path
from typing import Optional
from urllib.parse import parse_qs, unquote, urlparse

import cv2
import numpy as np

from .camera import project
from .template import circle_elements, line_elements, point_elements, polyline

# Windows reads MIME types from the registry, which is often wrong (.js as text/plain) or missing (.mp4):
# pin the types this server sends so the dashboard and the video player work the same on every OS.
for _ext, _type in ((".js", "text/javascript"), (".mjs", "text/javascript"), (".css", "text/css"),
                    (".html", "text/html"), (".json", "application/json"), (".svg", "image/svg+xml"),
                    (".png", "image/png"), (".jpg", "image/jpeg"), (".mp4", "video/mp4"), (".m4v", "video/mp4"),
                    (".mov", "video/quicktime"), (".webm", "video/webm"), (".md", "text/markdown")):
    mimetypes.add_type(_type, _ext)

ROOT = Path(__file__).resolve().parents[1]          # offside_code/
VENUES = ROOT / "venues"
HTML = Path(__file__).resolve().parent / "ui" / "studio.html"
IMG_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
HINT_KEYS = ("features", "inside_pitch", "ignore_regions", "occluding_regions", "player_min_y")
DEFAULT_TEAMS = [
    {"team": "HOME", "rules": [{"region": "torso", "hsv_lo": [100, 80, 60], "hsv_hi": [130, 255, 255], "min": 0.3}]},
    {"team": "REF", "rules": [{"region": "torso", "hsv_lo": [0, 0, 0], "hsv_hi": [180, 255, 60], "min": 0.6}]},
    {"team": "AWAY", "rules": []},
]


# ----------------------------------------------------------------------------- small helpers
def _rel(p) -> str:
    p = Path(p).resolve()
    try:
        return p.relative_to(ROOT).as_posix()
    except ValueError:
        return str(p)


def _abs(rel: str) -> Path:
    p = Path(rel)
    return (p if p.is_absolute() else ROOT / p).resolve()


def _inside_root(p: Path) -> bool:
    try:
        p.resolve().relative_to(ROOT)
        return True
    except ValueError:
        return False


def _safe(s: str, n: int = 60) -> str:
    s = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(s)).strip("._")
    return s[:n] or "venue"


def _jsonable(o):
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def _dumps(o) -> bytes:
    return json.dumps(o, default=_jsonable).encode()


def _scenario_name(sc) -> str:
    return f"{sc['attacking']} attacks {sc['direction']}"


def _frame_entry(fc: dict, use: bool = True) -> dict:
    return dict(use=use, features=copy.deepcopy(fc.get("features", [])),
                inside_pitch=copy.deepcopy(fc.get("inside_pitch", [])),
                ignore_regions=copy.deepcopy(fc.get("ignore_regions", [])),
                occluding_regions=copy.deepcopy(fc.get("occluding_regions", [])),
                player_min_y=fc.get("player_min_y", 0))


def _hints(entry: dict) -> dict:
    return {k: copy.deepcopy(entry.get(k, [] if k != "player_min_y" else 0)) for k in HINT_KEYS}


class _Images:
    """Full-resolution frames (small LRU) and scaled JPEG previews for the browser."""

    def __init__(self, n_full: int = 4, n_jpeg: int = 60):
        self.full: OrderedDict = OrderedDict()
        self.jpg: OrderedDict = OrderedDict()
        self.sizes: dict = {}
        self.n_full, self.n_jpeg = n_full, n_jpeg
        self.lock = threading.Lock()

    @staticmethod
    def _mtime(p: Path) -> float:
        return p.stat().st_mtime

    def get(self, p: Path) -> np.ndarray:
        key = (str(p), self._mtime(p))
        with self.lock:
            if key in self.full:
                self.full.move_to_end(key)
                return self.full[key]
        img = cv2.imread(str(p))
        if img is None:
            raise FileNotFoundError(f"cannot read image {_rel(p)}")
        with self.lock:
            self.full[key] = img
            self.sizes[key] = (img.shape[1], img.shape[0])
            while len(self.full) > self.n_full:
                self.full.popitem(last=False)
        return img

    def size(self, p: Path) -> Optional[tuple]:
        if not p.exists():
            return None
        key = (str(p), self._mtime(p))
        if key not in self.sizes:
            try:
                self.get(p)                                     # decodes once; cached by mtime
            except FileNotFoundError:
                return None
        return self.sizes[key]

    def jpeg(self, p: Path, w: int) -> bytes:
        key = (str(p), self._mtime(p), int(w))
        with self.lock:
            if key in self.jpg:
                self.jpg.move_to_end(key)
                return self.jpg[key]
        img = self.get(p)
        s = min(1.0, w / img.shape[1]) if w else 1.0
        small = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else img
        ok, buf = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 85])
        b = buf.tobytes()
        with self.lock:
            self.jpg[key] = b
            while len(self.jpg) > self.n_jpeg:
                self.jpg.popitem(last=False)
        return b


class Job:
    """A background command (the calibration) with its log, like the launcher's jobs."""

    def __init__(self, cmd: list, title: str, venue: str, on_done=None):
        self.id = uuid.uuid4().hex[:8]
        self.cmd, self.title, self.venue = cmd, title, venue
        self.lines = ["$ " + " ".join(f'"{c}"' if " " in str(c) else str(c) for c in cmd)]
        self.status, self.rc = "running", None
        self.started, self.ended = time.time(), None
        self.on_done = on_done
        env = dict(os.environ, PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8", MPLBACKEND="Agg")
        self.proc = subprocess.Popen(cmd, cwd=str(ROOT), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                     bufsize=1, env=env, encoding="utf-8", errors="replace")
        threading.Thread(target=self._pump, daemon=True).start()
        if sys.platform == "darwin" and shutil.which("caffeinate"):
            subprocess.Popen(["caffeinate", "-i", "-w", str(self.proc.pid)], stdout=subprocess.DEVNULL,
                             stderr=subprocess.DEVNULL)

    def _pump(self):
        for line in self.proc.stdout:
            self.lines.append(line.rstrip("\n"))
        self.rc = self.proc.wait()
        if self.status == "running":
            self.status = "done" if self.rc == 0 else "failed"
        self.ended = time.time()
        if self.on_done:
            try:
                self.on_done(self)
            except Exception:  # noqa: BLE001
                traceback.print_exc()

    def stop(self):
        if self.proc.poll() is None:
            self.status = "stopped"
            self.proc.terminate()

    def progress(self) -> Optional[float]:
        for ln in reversed(self.lines[-30:]):
            m = re.search(r"camera draws (\d+)/(\d+)", ln)
            if m:
                return int(m.group(1)) / max(1, int(m.group(2)))
        return None

    def summary(self) -> dict:
        return dict(id=self.id, title=self.title, venue=self.venue, status=self.status, rc=self.rc,
                    started=self.started, ended=self.ended, elapsed=(self.ended or time.time()) - self.started,
                    progress=self.progress(), n_lines=len(self.lines))


# ----------------------------------------------------------------------------- the studio
class Studio:
    def __init__(self, python: Optional[str] = None):
        self.python = python or sys.executable
        self.images = _Images()
        self.jobs: dict = {}
        self._detector = None
        self._det_key = None
        self._det_lock = threading.Lock()
        self._detections: OrderedDict = OrderedDict()
        self._last: dict = {}                       # (venue, image) -> (predictor, out, options)
        VENUES.mkdir(exist_ok=True)

    # ------------------------------------------------------------------ venues on disk
    def _vdir(self, name: str) -> Path:
        d = VENUES / _safe(name)
        if d.parent != VENUES:
            raise ValueError("bad venue name")
        return d

    def load(self, name: str) -> dict:
        p = self._vdir(name) / "venue.json"
        if not p.exists():
            raise FileNotFoundError(f"venue {name}")
        cfg = json.loads(p.read_text(encoding="utf-8"))
        if "studio" not in cfg:
            self._decompile(cfg)
        return cfg

    def save(self, name: str, cfg: dict) -> dict:
        self._compile(name, cfg)
        d = self._vdir(name)
        d.mkdir(parents=True, exist_ok=True)
        tmp = d / "venue.json.tmp"
        tmp.write_text(json.dumps(cfg, indent=1, default=_jsonable), encoding="utf-8")
        tmp.replace(d / "venue.json")
        return cfg

    def _decompile(self, cfg: dict):
        """A plain config (configs/*.json) -> studio block (frames + markings)."""
        from .venue import expand_paths
        cam = cfg.get("camera", {})
        mode = cam.get("mode") or ("panning" if cfg.get("aux_frames") else "fixed")
        dfc = cfg["decision_frame"]
        ref = dfc["image"]
        frames = OrderedDict()
        frames[ref] = _frame_entry(dfc)
        for afc in cfg.get("aux_frames", []):
            frames[afc["image"]] = _frame_entry(afc)
        ref_size = self.images.size(_abs(ref))
        for p in expand_paths(cam.get("calibration_frames", []), str(ROOT)):
            r = _rel(p)
            if r not in frames and Path(p).suffix.lower() in IMG_EXT:
                same = self.images.size(Path(p)) == ref_size
                frames[r] = _frame_entry({}, use=bool(same and mode == "fixed"))
        cfg["studio"] = dict(mode=mode, reference=ref, frames=frames)

    def _compile(self, name: str, cfg: dict):
        """studio block -> decision_frame / aux_frames / camera (the format the CLIs read)."""
        st = cfg["studio"]
        fr = st["frames"]
        ref = st.get("reference")
        if ref not in fr:
            ref = next(iter(fr), None)
            st["reference"] = ref
        cam = cfg.setdefault("camera", {})
        cam["mode"] = st.get("mode", "fixed")
        cam.setdefault("camera_model", f"venues/{_safe(name)}/camera_model.json")
        cam.setdefault("n_draws", 150)
        cfg.setdefault("models_dir", "models")
        cfg["output_dir"] = f"venues/{_safe(name)}/results"
        cfg.setdefault("teams", copy.deepcopy(DEFAULT_TEAMS))
        cfg.setdefault("team_overrides", {})
        if not cfg.get("scenarios"):
            t = [x["team"] for x in cfg["teams"] if x["team"] != "REF"]
            t = list(dict.fromkeys(t)) + ["HOME", "AWAY"]
            cfg["scenarios"] = [dict(attacking=t[0], defending=t[1], direction="right")]
        cfg.setdefault("priors", {"half_length": [52.5, 2.5], "half_width": [34.0, 2.0], "shoulder_h": [1.4, 0.25],
                                  "pp_sigma": 40})
        cfg.setdefault("monte_carlo", {"n": 150, "seed": 1})
        cfg.setdefault("show_metrics", False)
        if ref is None:
            cfg.pop("decision_frame", None)
            return
        cfg["decision_frame"] = dict(image=ref, **_hints(fr[ref]))
        others = [k for k, v in fr.items() if k != ref and v.get("use")]
        if cam["mode"] == "panning":
            cfg["aux_frames"] = [dict(name=Path(k).stem, image=k, **_hints(fr[k])) for k in others if fr[k].get("features")]
            cam["calibration_frames"] = []
        else:
            cfg.pop("aux_frames", None)
            cam["calibration_frames"] = others
        if not cfg.get("draw_elements"):
            cfg["draw_elements"] = sorted({f["element"] for f in fr[ref].get("features", [])})

    @staticmethod
    def _calib_hash(cfg: dict) -> str:
        st = cfg["studio"]
        used = {k: _hints(v) for k, v in st["frames"].items() if v.get("use") or k == st.get("reference")}
        key = dict(mode=st.get("mode"), ref=st.get("reference"), frames=used, priors=cfg.get("priors"),
                   detector=cfg.get("detector"))
        return hashlib.sha1(json.dumps(key, sort_keys=True, default=_jsonable).encode()).hexdigest()[:12]

    def venues(self) -> list:
        out = []
        for p in sorted(VENUES.glob("*/venue.json")):
            try:
                cfg = json.loads(p.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            model = _abs(cfg.get("camera", {}).get("camera_model", f"venues/{p.parent.name}/camera_model.json"))
            st = cfg.get("studio", {})
            out.append(dict(name=p.parent.name, title=cfg.get("name", p.parent.name),
                            mode=st.get("mode") or cfg.get("camera", {}).get("mode", "fixed"),
                            n_frames=len(st.get("frames", {})), calibrated=model.exists(),
                            modified=p.stat().st_mtime))
        return out

    def state(self) -> dict:
        return dict(venues=self.venues(),
                    configs=sorted(_rel(p) for p in (ROOT / "configs").glob("*.json")),
                    elements=dict(lines=list(line_elements(1, 1)), circles=list(circle_elements(1, 1)),
                                  points=list(point_elements(1, 1))),
                    models_ok=(ROOT / "models" / "yolox.onnx").exists() and (ROOT / "models" / "mp_pose.onnx").exists(),
                    root=str(ROOT), python=self.python,
                    jobs=[j.summary() for j in self.jobs.values()])

    def create(self, name: str, mode: str = "fixed", title: Optional[str] = None) -> dict:
        n = _safe(name)
        d = self._vdir(n)
        if (d / "venue.json").exists():
            raise FileExistsError(f"venue {n} already exists")
        (d / "images").mkdir(parents=True, exist_ok=True)
        cfg = dict(name=title or name, studio=dict(mode=mode if mode in ("fixed", "panning") else "fixed",
                                                   reference=None, frames=OrderedDict()))
        self.save(n, cfg)
        return self.venue(n)

    def import_config(self, config: str, name: Optional[str] = None) -> dict:
        src = _abs(config)
        if not src.exists() or not _inside_root(src):
            raise FileNotFoundError(config)
        cfg = json.loads(src.read_text(encoding="utf-8"))
        n = _safe(name or src.stem)
        d = self._vdir(n)
        if (d / "venue.json").exists():
            raise FileExistsError(f"venue {n} already exists")
        (d / "images").mkdir(parents=True, exist_ok=True)
        self._decompile(cfg)
        cam = cfg.setdefault("camera", {})
        old_model = _abs(cam["camera_model"]) if cam.get("camera_model") else None
        cam["camera_model"] = f"venues/{n}/camera_model.json"
        if old_model is not None and old_model.exists() and cfg["studio"]["mode"] == "fixed":
            shutil.copy2(old_model, d / "camera_model.json")          # already calibrated: usable at once
            bg = old_model.with_name(old_model.stem + "_background.png")
            if bg.exists():
                shutil.copy2(bg, d / "camera_model_background.png")
            cfg["studio"]["calibrated_hash"] = self._calib_hash(cfg)
        cfg["studio"]["imported_from"] = _rel(src)
        self.save(n, cfg)
        return self.venue(n)

    # ------------------------------------------------------------------ venue state for the page
    def _model_path(self, cfg: dict) -> Path:
        return _abs(cfg["camera"]["camera_model"])

    def venue(self, name: str) -> dict:
        cfg = self.load(name)
        self._compile(name, cfg)
        st = cfg["studio"]
        ref = st.get("reference")
        ref_size = self.images.size(_abs(ref)) if ref else None
        model_p = self._model_path(cfg)
        model = None
        frame_cams = []
        if model_p.exists():
            try:
                js = json.loads(model_p.read_text(encoding="utf-8"))
                frame_cams = sorted((js.get("frame_cameras") or {}).keys())
                model = dict(path=_rel(model_p), created=js.get("created"), mode=js.get("mode", "fixed"),
                             frame_size=js.get("frame_size"), pitch=[2 * js["half_length"], 2 * js["half_width"]],
                             meta=js.get("meta", {}), diagnostics=js.get("diagnostics", {}), frame_cameras=frame_cams,
                             stale=st.get("calibrated_hash") not in (None, self._calib_hash(cfg)),
                             background=_rel(model_p.with_name(model_p.stem + "_background.png"))
                             if model_p.with_name(model_p.stem + "_background.png").exists() else None)
            except (OSError, ValueError, KeyError) as e:
                model = dict(path=_rel(model_p), error=str(e))
        frames = []
        for k, v in st["frames"].items():
            p = _abs(k)
            size = self.images.size(p)
            if st.get("mode") == "panning":
                judgeable = model is not None and k in frame_cams
            else:
                judgeable = model is not None and size is not None and model.get("frame_size") == list(size)
            frames.append(dict(image=k, name=Path(k).name, exists=p.exists(), size=size, use=bool(v.get("use")),
                               reference=k == ref, same_size=size == ref_size,
                               n_features=len(v.get("features", [])), n_inside=len(v.get("inside_pitch", [])),
                               n_ignore=len(v.get("ignore_regions", [])), judgeable=judgeable,
                               mtime=p.stat().st_mtime if p.exists() else 0,
                               result=(self._vdir(name) / "results" / _safe(Path(k).stem, 80) / "offside_result.json").exists()))
        job = next((j.summary() for j in reversed(list(self.jobs.values())) if j.venue == name), None)
        return dict(name=name, title=cfg.get("name", name), mode=st.get("mode", "fixed"), reference=ref,
                    frames=frames, hints={k: _hints(v) for k, v in st["frames"].items()},
                    teams=cfg["teams"], team_overrides=cfg.get("team_overrides", {}), scenarios=cfg["scenarios"],
                    priors=cfg.get("priors"), detector=cfg.get("detector", {}), n_draws=cfg["camera"].get("n_draws", 150),
                    draw_elements=cfg.get("draw_elements", []), show_metrics=cfg.get("show_metrics", False),
                    model=model, job=job, imported_from=st.get("imported_from"),
                    config_path=f"venues/{_safe(name)}/venue.json")

    # ------------------------------------------------------------------ edits
    def update(self, name: str, b: dict) -> dict:
        """Replace whole keys sent by the page: frames' markings, reference, mode, teams, scenarios..."""
        cfg = self.load(name)
        st = cfg["studio"]
        if "mode" in b and b["mode"] in ("fixed", "panning"):
            st["mode"] = b["mode"]
        if "reference" in b and b["reference"] in st["frames"]:
            st["reference"] = b["reference"]
        for img, h in (b.get("hints") or {}).items():
            if img in st["frames"]:
                for k in HINT_KEYS:
                    if k in h:
                        st["frames"][img][k] = h[k]
        for img, u in (b.get("use") or {}).items():
            if img in st["frames"]:
                st["frames"][img]["use"] = bool(u)
        for img in b.get("remove") or []:
            st["frames"].pop(img, None)
        if "teams" in b:
            teams = b["teams"]
            if not isinstance(teams, list) or not all(isinstance(t, dict) and "team" in t for t in teams):
                raise ValueError("teams: a list of {team, rules}")
            for t in teams:
                t.setdefault("rules", [])
                for r in t["rules"]:
                    if r.get("region") not in ("torso", "shorts") or len(r.get("hsv_lo", [])) != 3 \
                            or len(r.get("hsv_hi", [])) != 3:
                        raise ValueError(f"team {t['team']}: each rule needs region torso|shorts, hsv_lo, hsv_hi")
            cfg["teams"] = teams
        if "team_overrides" in b:
            cfg["team_overrides"] = {str(k): v for k, v in (b["team_overrides"] or {}).items() if v}
        if "scenarios" in b:
            seen, sc = set(), []
            for s in b["scenarios"]:
                if s.get("attacking") and s.get("defending") and s.get("direction") in ("left", "right"):
                    n = _scenario_name(s)
                    if n not in seen:
                        seen.add(n)
                        sc.append(dict(attacking=s["attacking"], defending=s["defending"], direction=s["direction"],
                                       gk_visible=bool(s.get("gk_visible"))))
            cfg["scenarios"] = sc
        if "n_draws" in b:
            cfg.setdefault("camera", {})["n_draws"] = int(max(5, min(500, int(b["n_draws"]))))
        if "draw_elements" in b:
            cfg["draw_elements"] = [e for e in b["draw_elements"] if isinstance(e, str)]
        if "show_metrics" in b:
            cfg["show_metrics"] = bool(b["show_metrics"])
        if "title" in b and b["title"]:
            cfg["name"] = str(b["title"])
        if "detector" in b and isinstance(b["detector"], dict):
            cfg["detector"] = {k: int(v) for k, v in b["detector"].items() if k in ("tile", "stride") and v}
        if "priors" in b and isinstance(b["priors"], dict):
            cfg["priors"] = b["priors"]
        self.save(name, cfg)
        return self.venue(name)

    def upload(self, name: str, filename: str, rfile, length: int, role: str = "calibration") -> dict:
        cfg = self.load(name)
        fn = Path(unquote(filename)).name
        if Path(fn).suffix.lower() not in IMG_EXT:
            raise ValueError(f"not an image: {fn}")
        d = self._vdir(name) / "images"
        d.mkdir(parents=True, exist_ok=True)
        stem, ext = _safe(Path(fn).stem, 80), Path(fn).suffix.lower()
        dst = d / f"{stem}{ext}"
        k = 1
        while dst.exists():
            dst = d / f"{stem}_{k}{ext}"
            k += 1
        tmp = dst.with_name("." + dst.name + ".part")
        left = length
        with open(tmp, "wb") as f:
            while left > 0:
                chunk = rfile.read(min(4 << 20, left))
                if not chunk:
                    break
                f.write(chunk)
                left -= len(chunk)
        if left:
            tmp.unlink(missing_ok=True)
            raise IOError("upload interrupted")
        tmp.replace(dst)
        if cv2.imread(str(dst), cv2.IMREAD_REDUCED_GRAYSCALE_8) is None:
            raise ValueError(f"{fn}: not a readable image")
        r = _rel(dst)
        st = cfg["studio"]
        st["frames"][r] = _frame_entry({}, use=role != "play")
        if not st.get("reference"):
            st["reference"] = r
        self.save(name, cfg)
        return dict(ok=True, image=r)

    def add_existing(self, name: str, paths: list, role: str = "calibration") -> dict:
        """Frames already inside offside_code/ (e.g. data/*.png) — referenced, not copied."""
        cfg = self.load(name)
        st = cfg["studio"]
        added = []
        for rp in paths:
            p = _abs(rp)
            if not p.exists() or not _inside_root(p) or p.suffix.lower() not in IMG_EXT:
                continue
            r = _rel(p)
            if r not in st["frames"]:
                st["frames"][r] = _frame_entry({}, use=role != "play")
                added.append(r)
            if not st.get("reference"):
                st["reference"] = r
        self.save(name, cfg)
        return dict(ok=True, added=added)

    def browse(self) -> list:
        """Images inside offside_code/ that can be added to a venue without uploading."""
        out = []
        for p in sorted(ROOT.rglob("*")):
            if p.suffix.lower() in IMG_EXT and p.is_file():
                r = _rel(p)
                if r.startswith(("outputs/", ".venv/", "notebooks/")) or "/results/" in r or r.endswith("_background.png"):
                    continue
                out.append(r)
        return out[:500]

    # ------------------------------------------------------------------ step 1: check the markings
    def check(self, name: str, image: str) -> dict:
        """White-line pixels picked up by the hints of one frame (what the calibration will use)."""
        from .features import extract_features
        cfg = self.load(name)
        st = cfg["studio"]
        if image not in st["frames"]:
            raise FileNotFoundError(image)
        mode = st.get("mode", "fixed")
        src = image if mode == "panning" else st.get("reference") or image
        h = _hints(st["frames"][src])
        if not h["features"]:
            return dict(features={}, counts={}, note="no markings yet")
        img = self.images.get(_abs(image))
        t0 = time.time()
        feats = extract_features(img, h["features"], h["ignore_regions"], np.random.default_rng(0))
        out, counts = {}, {}
        for el, P in feats.items():
            counts[el] = int(len(P))
            if len(P) > 1500:
                P = P[np.random.default_rng(1).choice(len(P), 1500, replace=False)]
            out[el] = np.round(P, 1).tolist()
        warn = [el for el, n in counts.items() if n < 40 and not el.startswith(("penalty_spot", "centre_spot", "corner", "halfway_"))]
        return dict(features=out, counts=counts, seconds=round(time.time() - t0, 2), weak=warn,
                    note=("markings of the reference frame (stationary camera: the same for every frame)"
                          if mode == "fixed" and src != image else None))

    # ------------------------------------------------------------------ step 2: calibration
    def calibrate(self, name: str, n_draws: Optional[int] = None) -> dict:
        for j in self.jobs.values():
            if j.venue == name and j.status == "running":
                raise RuntimeError("a calibration of this venue is already running")
        cfg = self.load(name)
        st = cfg["studio"]
        ref = st.get("reference")
        if not ref:
            raise ValueError("add a frame first")
        if not st["frames"][ref].get("features"):
            raise ValueError("the reference frame has no markings: click at least two pitch markings on it")
        if n_draws:
            cfg["camera"]["n_draws"] = int(max(5, min(500, int(n_draws))))
        self.save(name, cfg)
        h = self._calib_hash(cfg)
        cmd = [self.python, "-u", str(ROOT / "calibrate_camera.py"), "--config", f"venues/{_safe(name)}/venue.json"]

        def done(job):
            if job.status == "done":
                c = self.load(name)
                c["studio"]["calibrated_hash"] = h
                self.save(name, c)
                self._last = {k: v for k, v in self._last.items() if k[0] != name}

        j = Job(cmd, f"Calibrate {name}", name, done)
        self.jobs[j.id] = j
        return j.summary()

    def job_log(self, jid: str, since: int = 0) -> dict:
        j = self.jobs.get(jid)
        if j is None:
            raise FileNotFoundError(f"job {jid}")
        return dict(job=j.summary(), lines=j.lines[since:], next=len(j.lines))

    def reprojection(self, name: str, image: str) -> dict:
        """The calibrated pitch template drawn on a frame (image pixels), to check the calibration by eye."""
        from .venue import load_camera_model
        cfg = self.load(name)
        mode = cfg["studio"].get("mode", "fixed")
        mp = self._model_path(cfg)
        if not mp.exists():
            raise FileNotFoundError("not calibrated yet")
        m = load_camera_model(str(mp), image if mode == "panning" else None)
        cam, hl, hw = m["cam"], m["half_length"], m["half_width"]
        W, H = m["frame_size"]
        out = {}
        names = list(line_elements(hl, hw)) + list(circle_elements(hl, hw))
        for n in names:
            P = polyline(n, hl, hw, 160)
            depth = (P - cam["C"]) @ cam["R"][2]
            uv = project(P, cam)
            ok = (depth > 0.5) & (uv[:, 0] > -0.25 * W) & (uv[:, 0] < 1.25 * W) & (uv[:, 1] > -0.25 * H) & (uv[:, 1] < 1.25 * H)
            if ok.sum() < 2:
                continue
            segs, cur = [], []
            for good, q in zip(ok, uv):
                if good:
                    cur.append([round(float(q[0]), 1), round(float(q[1]), 1)])
                elif cur:
                    segs.append(cur)
                    cur = []
            if cur:
                segs.append(cur)
            out[n] = [s for s in segs if len(s) > 1]
        pts = {}
        for n, (x, y) in point_elements(hl, hw).items():
            P = np.array([[x, y, 0.0]])
            if ((P - cam["C"]) @ cam["R"][2])[0] > 0.5:
                q = project(P, cam)[0]
                if 0 <= q[0] < W and 0 <= q[1] < H:
                    pts[n] = [round(float(q[0]), 1), round(float(q[1]), 1)]
        return dict(elements=out, points=pts, frame_size=[W, H], pitch=[2 * hl, 2 * hw],
                    f_px=round(float(cam["f"]), 1), camera_xyz=np.round(cam["C"], 2).tolist())

    # ------------------------------------------------------------------ step 3: decide
    def sample(self, name: str, image: str, x: float, y: float, r: int = 6, region: str = "torso") -> dict:
        """Median colour around a clicked pixel -> suggested HSV rule for a team's kit."""
        img = self.images.get(_abs(image))
        H, W = img.shape[:2]
        x, y, r = int(round(x)), int(round(y)), int(max(2, min(30, r)))
        patch = img[max(0, y - r):min(H, y + r + 1), max(0, x - r):min(W, x + r + 1)]
        if patch.size == 0:
            raise ValueError("outside the image")
        hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV).reshape(-1, 3).astype(float)
        ang = hsv[:, 0] * np.pi / 90.0                         # OpenCV hue 0..180 -> angle
        h = (np.degrees(np.arctan2(np.sin(ang).mean(), np.cos(ang).mean())) / 2.0) % 180
        s, v = float(np.median(hsv[:, 1])), float(np.median(hsv[:, 2]))
        b, g, rr = np.median(patch.reshape(-1, 3), 0).astype(int)
        rules = []
        if s < 45:                                             # white / grey / black kits: brightness only
            if v >= 160:
                rules.append(([0, 0, 150], [180, 70, 255], "white"))
            elif v <= 70:
                rules.append(([0, 0, 0], [180, 255, 70], "black / dark"))
            else:
                rules.append(([0, 0, 70], [180, 55, 170], "grey"))
        else:
            lo_s, lo_v = int(max(40, 0.5 * s)), int(max(40, 0.45 * v))
            lo_h, hi_h = h - 10, h + 10
            if lo_h < 0:
                rules += [([0, lo_s, lo_v], [int(hi_h), 255, 255], "hue"), ([int(180 + lo_h), lo_s, lo_v], [180, 255, 255], "hue (wraps)")]
            elif hi_h > 180:
                rules += [([int(lo_h), lo_s, lo_v], [180, 255, 255], "hue"), ([0, lo_s, lo_v], [int(hi_h - 180), 255, 255], "hue (wraps)")]
            else:
                rules.append(([int(lo_h), lo_s, lo_v], [int(hi_h), 255, 255], "hue"))
        return dict(hsv=[round(h, 1), round(s), round(v)], rgb="#%02x%02x%02x" % (rr, g, b),
                    rules=[dict(region=region, hsv_lo=lo, hsv_hi=hi, min=0.3) for lo, hi, _ in rules],
                    kind=rules[0][2])

    def _get_detector(self, cfg: dict):
        from .players import PlayerDetector
        key = (cfg.get("models_dir", "models"), json.dumps(cfg.get("detector", {}), sort_keys=True))
        if self._detector is None or self._det_key != key:
            self._detector = PlayerDetector(str(_abs(cfg.get("models_dir", "models"))), **cfg.get("detector", {}))
            self._det_key = key
            self._detections.clear()
        return self._detector

    def predict(self, name: str, b: dict) -> dict:
        from .predict import OffsidePredictor
        from .venue import load_camera_model
        cfg = self.load(name)
        self._compile(name, cfg)
        st = cfg["studio"]
        image = b["image"]
        if image not in st["frames"]:
            raise FileNotFoundError(image)
        mode = st.get("mode", "fixed")
        mp = self._model_path(cfg)
        if not mp.exists():
            raise ValueError("calibrate the camera first (step 2)")
        frame_cfg = _hints(st["frames"][image]) if mode == "panning" else dict(cfg["decision_frame"])
        try:
            model = load_camera_model(str(mp), image if mode == "panning" else None)
        except KeyError as e:
            raise ValueError(str(e).strip('"'))
        img = self.images.get(_abs(image))
        if (img.shape[1], img.shape[0]) != tuple(model["frame_size"]):
            raise ValueError(f"this frame is {img.shape[1]}x{img.shape[0]} but the camera was calibrated on "
                             f"{model['frame_size'][0]}x{model['frame_size'][1]} — a stationary camera model only fits "
                             f"frames of the same size and view")
        if b.get("scenarios") is not None:
            cfg = self._with(cfg, scenarios=b["scenarios"])
        if b.get("teams") is not None:
            cfg["teams"] = b["teams"]
        if b.get("team_overrides") is not None:
            cfg["team_overrides"] = {str(k): v for k, v in b["team_overrides"].items() if v}
        if not cfg.get("scenarios"):
            raise ValueError("add at least one attacking direction (scenario)")
        with self._det_lock:
            det = self._get_detector(cfg)
            pred = OffsidePredictor(model, cfg, detector=det)
            dkey = (image, _abs(image).stat().st_mtime, json.dumps(
                {k: frame_cfg.get(k) for k in ("ignore_regions", "occluding_regions", "player_min_y")}, sort_keys=True))
            detections = self._detections.get(dkey)
            if detections is None:
                detections = pred.detect(img, frame_cfg)
                self._detections[dkey] = detections
                while len(self._detections) > 12:
                    self._detections.popitem(last=False)
                cached = False
            else:
                cached = True
            ball = b.get("ball") or {"mode": "auto"}
            ball_px = ball.get("xy") if ball.get("mode") == "click" and ball.get("xy") else None
            opts = dict(show_lines=bool(b.get("show_lines", True)), show_pitch=bool(b.get("show_pitch", True)),
                        show_metrics=bool(b.get("show_metrics", False)))
            out = pred.predict(img, ball_pixel=ball_px, uncertainty=bool(b.get("uncertainty", True)),
                               frame_cfg=frame_cfg, detections=detections, auto_ball=ball.get("mode") == "auto",
                               **opts)
        if cached:
            out["timings"]["detection_cached"] = True
        else:
            out["timings"].update({k: round(v, 3) for k, v in detections["timings"].items()})
        self._last[(name, image)] = (pred, out)
        files = self._write_results(name, image, out, opts)
        return self._result_json(out, files, model, opts)

    @staticmethod
    def _with(cfg, **kw):
        c = dict(cfg)
        c.update(kw)
        return c

    def render(self, name: str, b: dict) -> dict:
        key = (name, b["image"])
        if key not in self._last:
            return self.predict(name, b)
        pred, out = self._last[key]
        img = self.images.get(_abs(b["image"]))
        opts = dict(show_lines=bool(b.get("show_lines", True)), show_pitch=bool(b.get("show_pitch", True)),
                    show_metrics=bool(b.get("show_metrics", False)))
        t0 = time.perf_counter()
        out["images"] = pred.render(img, out, **opts)
        out["timings"]["render"] = round(time.perf_counter() - t0, 3)
        files = self._write_results(name, b["image"], out, opts)
        return self._result_json(out, files, pred.model, opts)

    def _write_results(self, name: str, image: str, out: dict, opts: dict) -> dict:
        d = self._vdir(name) / "results" / _safe(Path(image).stem, 80)
        d.mkdir(parents=True, exist_ok=True)
        files = {}
        for n, im in out["images"].items():
            fn = "birds_eye.png" if n == "birds_eye" else "offside_" + _safe(n.replace(" attacks ", "_attacks_")) + ".png"
            cv2.imwrite(str(d / fn), im)
            files[n] = _rel(d / fn)
        res = dict(image=image, created=time.strftime("%Y-%m-%d %H:%M:%S"), options=opts, decisions=out["decisions"],
                   summary=out["summary"], skipped=out["skipped"], ball_pixel=out["ball_pixel"],
                   ball_source=out["ball_source"], timings=out["timings"],
                   players=[dict(id=q["id"], team=q["team"], box=np.round(q["box"], 1).tolist(),
                                 foot=np.round(q["foot"], 1).tolist(), upright=bool(q["upright"]),
                                 occluded=bool(q["occluded"])) for q in out["players"]], files=files)
        (d / "offside_result.json").write_text(json.dumps(res, indent=1, default=_jsonable), encoding="utf-8")
        files["json"] = _rel(d / "offside_result.json")
        return files

    @staticmethod
    def _result_json(out: dict, files: dict, model: dict, opts: dict) -> dict:
        scen = []
        for sc in out["scenarios"]:
            n = _scenario_name(sc)
            s = out["summary"][n]
            att = []
            for i, a in s["attackers"].items():
                r = out["res"][i]
                att.append(dict(id=int(i), team=r["team"], decision=a["decision"], margin_m=round(a["margin_m"], 3),
                                p_offside=round(a["p_offside"], 3), ci90_m=[round(v, 3) for v in a["ci90_m"]],
                                relevant=a["margin_m"] > -3))
            att.sort(key=lambda a: -a["margin_m"])
            line_by = None
            key = "xmax" if sc["direction"] == "right" else "xmin"
            for i, r in out["res"].items():
                if r["team"] == sc["defending"] and abs(r[key] - s["line_X"]) < 1e-6:
                    line_by = f"defender #{i}"
            if line_by is None:
                ball = out["ball"]
                line_by = "the ball" if ball is not None and abs(ball[0] - s["line_X"]) < 1e-6 else "the halfway line"
            scen.append(dict(name=n, attacking=sc["attacking"], defending=sc["defending"], direction=sc["direction"],
                             line_X=round(s["line_X"], 3), line_ci90=[round(v, 3) for v in s["line_ci90"]],
                             line_by=line_by, attackers=att,
                             n_offside=sum(a["decision"] == "OFFSIDE" for a in att)))
        players = []
        for q in out["players"]:
            r = out["res"].get(q["id"])
            players.append(dict(id=int(q["id"]), team=q["team"], box=np.round(q["box"], 1).tolist(),
                                foot=np.round(q["foot"], 1).tolist(), upright=bool(q["upright"]),
                                occluded=bool(q["occluded"]), x_m=None if r is None else round((r["xmin"] + r["xmax"]) / 2, 2),
                                y_m=None if r is None else round(r["y"], 2)))
        v = int(time.time() * 1000)
        return dict(scenarios=scen, skipped=out["skipped"], players=players,
                    ball_pixel=None if out["ball_pixel"] is None else [round(float(c), 1) for c in out["ball_pixel"]],
                    ball_source=out["ball_source"], timings=out["timings"], options=opts,
                    images={n: dict(file=f, v=v) for n, f in files.items() if n != "json"}, result_json=files.get("json"),
                    pitch=[2 * model["half_length"], 2 * model["half_width"]])

    def last_result(self, name: str, image: str) -> Optional[dict]:
        p = self._vdir(name) / "results" / _safe(Path(image).stem, 80) / "offside_result.json"
        if not p.exists():
            return None
        return json.loads(p.read_text(encoding="utf-8"))

    def stop_all(self):
        for j in self.jobs.values():
            j.stop()


# ----------------------------------------------------------------------------- HTTP routing (used by the launcher)
def page(base: str) -> bytes:
    """The studio page; `base` is the URL prefix it is served under ('' standalone, '/off' in the launcher)."""
    html = HTML.read_text(encoding="utf-8")
    return html.replace("__BASE__", base).encode()


def route_get(studio: Studio, path: str, q: dict):
    """-> (code, body, content type, extra headers)."""
    path = unquote(path)
    if path in ("", "/", "/index.html"):
        return 200, None, "text/html; charset=utf-8", None            # caller serves page(base)
    if path == "/api/state":
        return 200, _dumps(studio.state()), "application/json", None
    if path == "/api/browse":
        return 200, _dumps(dict(images=studio.browse())), "application/json", None
    if path.startswith("/api/venue/"):
        parts = path.split("/")
        name = parts[3]
        if len(parts) == 4:
            return 200, _dumps(studio.venue(name)), "application/json", None
        if parts[4] == "reprojection":
            return 200, _dumps(studio.reprojection(name, q["image"])), "application/json", None
        if parts[4] == "result":
            return 200, _dumps(dict(result=studio.last_result(name, q["image"]))), "application/json", None
    if path.startswith("/api/job/"):
        return 200, _dumps(studio.job_log(path.split("/")[3], int(q.get("since", 0)))), "application/json", None
    if path == "/img":
        p = _abs(q["p"])
        if not _inside_root(p) or p.suffix.lower() not in IMG_EXT:
            return 403, b"forbidden", "text/plain", None
        return 200, studio.images.jpeg(p, int(q.get("w", 1600))), "image/jpeg", {"Cache-Control": "max-age=600"}
    if path == "/file":
        p = _abs(q["p"])
        if not _inside_root(p) or not p.is_file():
            return 404, b"not found", "text/plain", None
        ctype = mimetypes.guess_type(p.name)[0] or "application/octet-stream"
        extra = {"Content-Disposition": f'inline; filename="{p.name}"'}
        return 200, p.read_bytes(), ctype, extra
    return 404, b"not found", "text/plain", None


def route_post(studio: Studio, path: str, q: dict, rfile, length: int):
    """-> (code, json-able dict). Image uploads stream from rfile; everything else is a JSON body."""
    path = unquote(path)
    if path.startswith("/api/venue/") and path.endswith("/upload"):
        name = path.split("/")[3]
        return 200, studio.upload(name, q.get("name", "frame.png"), rfile, length, q.get("role", "calibration"))
    b = json.loads(rfile.read(length) or b"{}") if length else {}
    if path == "/api/venues":
        if b.get("action") == "import":
            return 200, studio.import_config(b["config"], b.get("name"))
        return 200, studio.create(b["name"], b.get("mode", "fixed"), b.get("title"))
    if path.startswith("/api/job/") and path.endswith("/stop"):
        j = studio.jobs.get(path.split("/")[3])
        if j:
            j.stop()
        return 200, dict(ok=bool(j))
    if path.startswith("/api/venue/"):
        parts = path.split("/")
        name, action = parts[3], parts[4] if len(parts) > 4 else ""
        if action == "save":
            return 200, studio.update(name, b)
        if action == "add":
            return 200, studio.add_existing(name, b.get("paths", []), b.get("role", "calibration"))
        if action == "check":
            return 200, studio.check(name, b["image"])
        if action == "calibrate":
            return 200, studio.calibrate(name, b.get("n_draws"))
        if action == "sample":
            return 200, studio.sample(name, b["image"], b["x"], b["y"], b.get("r", 6), b.get("region", "torso"))
        if action == "predict":
            return 200, studio.predict(name, b)
        if action == "render":
            return 200, studio.render(name, b)
    return 404, dict(error="not found")


def handle(handler, studio: Studio, method: str, base: str):
    """Serve one request on a BaseHTTPRequestHandler whose path starts with `base` (shared by the launcher and the
    standalone server)."""
    u = urlparse(handler.path)
    q = {k: v[0] for k, v in parse_qs(u.query).items()}
    sub = u.path[len(base):] if base and u.path.startswith(base) else u.path

    def send(code, body, ctype, extra=None):
        handler.send_response(code)
        handler.send_header("Content-Type", ctype)
        handler.send_header("Content-Length", str(len(body)))
        if not (extra and "Cache-Control" in extra):
            handler.send_header("Cache-Control", "no-store")
        for k, v in (extra or {}).items():
            handler.send_header(k, v)
        handler.end_headers()
        if handler.command != "HEAD":
            handler.wfile.write(body)

    try:
        if method == "GET":
            code, body, ctype, extra = route_get(studio, sub, q)
            if body is None:
                body = page(base)
            return send(code, body, ctype, extra)
        n = int(handler.headers.get("Content-Length", 0))
        code, obj = route_post(studio, sub, q, handler.rfile, n)
        return send(code, _dumps(obj), "application/json")
    except (ValueError, RuntimeError, FileExistsError, FileNotFoundError, KeyError, IOError) as e:
        msg = str(e) if not isinstance(e, KeyError) else f"missing: {e}"
        return send(400, _dumps(dict(error=msg)), "application/json")
    except Exception as e:  # noqa: BLE001
        traceback.print_exc()
        return send(500, _dumps(dict(error=f"{type(e).__name__}: {e}")), "application/json")


def serve(host: str = "127.0.0.1", port: int = 8775, open_browser: bool = True, python: Optional[str] = None):
    """Stand-alone: the studio on its own page (the dashboard serves the same thing in its Offside tab)."""
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    studio = Studio(python)

    class H(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *a):
            pass

        def do_GET(self):
            handle(self, studio, "GET", "")

        def do_HEAD(self):
            handle(self, studio, "GET", "")

        def do_POST(self):
            handle(self, studio, "POST", "")

    srv = ThreadingHTTPServer((host, port), H)
    srv.daemon_threads = True
    url = f"http://{host}:{port}/"
    print(f"Offside studio on {url}  (Ctrl+C to stop) — venues in {VENUES}")
    if open_browser:
        import webbrowser
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nstopping…")
    finally:
        studio.stop_all()
        srv.server_close()
