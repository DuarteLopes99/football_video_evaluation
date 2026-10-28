"""Launcher: one local web page to run everything — video library, analysis jobs with a live log, the setup and
review UIs, and the results with the flagged opportunities next to a video player.

Standard library only (``http.server`` + ``subprocess``). Start it with ``python scripts/launcher.py`` (or
double-click ``start.command`` on macOS); it opens http://127.0.0.1:8770. Every button runs the same command you
could type yourself, and the page shows that command.
"""
from __future__ import annotations

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
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Optional
from urllib.parse import parse_qs, unquote, urlparse

from ..storage import (VIDEO_EXT, disk_free_gb, find_video, library, list_videos, load_storage, move_video,
                       primary_library)

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUTPUTS = ROOT / "outputs"
UI_PORT = 8765                     # scripts/match_ui.py
PROGRESS = re.compile(r"video t=\s*([\d.]+) min")


# ----------------------------------------------------------------------------- jobs
class Job:
    def __init__(self, kind: str, title: str, cmd: list, run: Optional[str] = None, video: Optional[str] = None,
                 duration_min: Optional[float] = None, url: Optional[str] = None, start_min: float = 0.0):
        self.id = uuid.uuid4().hex[:8]
        self.kind, self.title, self.cmd, self.run, self.video, self.url = kind, title, cmd, run, video, url
        self.duration_min, self.start_min = duration_min, start_min
        self.lines: list = []
        self.status = "running"
        self.rc: Optional[int] = None
        self.started, self.ended = time.time(), None
        self.progress: Optional[float] = None
        env = dict(os.environ, PYTHONUNBUFFERED="1", MPLBACKEND="Agg", PYTHONIOENCODING="utf-8")
        self.lines.append("$ " + " ".join(_q(c) for c in cmd))
        self.proc = subprocess.Popen(cmd, cwd=str(ROOT), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                     text=True, bufsize=1, env=env, encoding="utf-8", errors="replace")
        threading.Thread(target=self._pump, daemon=True).start()
        if kind in ("analyse", "repair") and sys.platform == "darwin" and shutil.which("caffeinate"):
            # keep the Mac from idle-sleeping while this job runs (hours); caffeinate exits with the job
            subprocess.Popen(["caffeinate", "-i", "-w", str(self.proc.pid)], stdout=subprocess.DEVNULL,
                             stderr=subprocess.DEVNULL)

    def _pump(self):
        for line in self.proc.stdout:
            line = line.rstrip("\n")
            self.lines.append(line)
            m = PROGRESS.search(line)
            if m and self.duration_min:
                span = max(1e-6, self.duration_min - self.start_min)
                self.progress = max(0.0, min(1.0, (float(m.group(1)) - self.start_min) / span))
            if len(self.lines) > 20000:
                del self.lines[1:5000]
        self.rc = self.proc.wait()
        if self.status == "running":
            self.status = "done" if self.rc == 0 else "failed"
        if self.status == "done" and self.kind in ("analyse", "analytics", "repair", "synthetic"):
            self.progress = 1.0
        self.ended = time.time()

    def stop(self):
        if self.proc.poll() is None:
            self.status = "stopped"
            self.proc.terminate()
            try:
                self.proc.wait(5)
            except subprocess.TimeoutExpired:
                self.proc.kill()

    def summary(self) -> dict:
        return dict(id=self.id, kind=self.kind, title=self.title, status=self.status, rc=self.rc, run=self.run,
                    video=self.video, url=self.url, started=self.started, ended=self.ended, progress=self.progress,
                    elapsed=(self.ended or time.time()) - self.started, n_lines=len(self.lines),
                    command=self.lines[0][2:] if self.lines else "")


def _q(s: str) -> str:
    s = str(s)
    return f'"{s}"' if (" " in s or not s) else s


def _rel(p) -> str:
    try:
        return str(Path(p).resolve().relative_to(ROOT))
    except ValueError:
        return str(p)


def _safe_name(s: str) -> str:
    s = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(s)).strip("._")
    return s[:80] or "match"


# ----------------------------------------------------------------------------- app state
DOCS = {"USER_GUIDE": "docs/USER_GUIDE.md", "README": "README.md", "STATS_RULEBOOK": "docs/STATS_RULEBOOK.md",
        "REPORT_GUIDE": "docs/REPORT_GUIDE.md", "MANUAL_VALIDATION": "docs/MANUAL_VALIDATION.md",
        "VIDEO_STORAGE": "docs/VIDEO_STORAGE.md", "RECORDING_PROFILE": "docs/RECORDING_PROFILE.md",
        "SOTA_REVIEW": "docs/SOTA_REVIEW.md"}


class Launcher:
    def __init__(self, python: Optional[str] = None, preselect: Optional[dict] = None):
        self.python = python or sys.executable
        self.preselect = preselect or {}         # --video / --config / --setup / --open / --start from the command line
        self.server = None
        self.jobs: dict = {}
        self.lock = threading.Lock()
        self.manual = None                 # the setup / review App currently open in the "Manual decisions" tab
        self.manual_info: dict = {}
        self._dur_cache: dict = {}

    # ---- listings
    def state(self) -> dict:
        st = load_storage()
        vids = list_videos(st)
        runs = self.runs(vids)
        by_video = {}
        for r in runs:
            if r.get("video_id"):
                by_video.setdefault(r["video_id"], []).append(r["name"])
        for v in vids:
            v["runs"] = by_video.get(v["id"], [])
            v["setup"] = self._setup_for(v["name"])
            v["config"] = self._config_for(v["setup"]) if v["setup"] else None
        libs = []
        for lib in st["libraries"]:
            libs.append(dict(lib, free_gb=disk_free_gb(lib["path"]) if lib["exists"] else None,
                             n_videos=sum(1 for v in vids if v["library"] == lib["name"]),
                             size_gb=round(sum((v["size_mb"] or 0) for v in vids if v["library"] == lib["name"]) / 1e3, 2)))
        return dict(root=str(ROOT), python=self.python, ffmpeg=bool(shutil.which("ffmpeg")), libraries=libs,
                    archive=st.get("archive"), storage_file=_rel(st["file"]), videos=vids,
                    configs=sorted(_rel(p) for p in (ROOT / "configs").glob("*.yaml") if p.name != "storage.yaml"),
                    setups=sorted(_rel(p) for p in (ROOT / "configs").glob("match_setup*.json")),
                    runs=runs, jobs=[j.summary() for j in sorted(self.jobs.values(), key=lambda j: -j.started)],
                    preselect=self.preselect, platform=sys.platform, docs=list(DOCS))

    def _setup_for(self, video_name: str) -> Optional[str]:
        for p in sorted((ROOT / "configs").glob("match_setup*.json")):
            try:
                vp = json.loads(p.read_text()).get("video", {}).get("path")
            except (OSError, json.JSONDecodeError):
                continue
            if vp and Path(vp).name == video_name:
                return _rel(p)
        return None

    @staticmethod
    def _config_for(setup_rel: str) -> Optional[str]:
        """The venue config recorded in a match setup (setup.meta.config), if it exists."""
        st = _read_json(ROOT / setup_rel) or {}
        c = (st.get("meta") or {}).get("config")
        return c if c and (ROOT / c).exists() else None

    def runs(self, vids: Optional[list] = None) -> list:
        vids = list_videos() if vids is None else vids
        by_name = {}
        for v in vids:                                   # local copies first, cloud-only last
            if v["name"] not in by_name or by_name[v["name"]]["cloud_only"]:
                by_name[v["name"]] = v
        out = []
        if not OUTPUTS.exists():
            return out
        for d in sorted(OUTPUTS.iterdir()):
            if not d.is_dir() or not ((d / "events.csv").exists() or (d / "tracks.csv").exists()
                                      or (d / "live" / "state.json").exists()):
                continue
            meta = _read_json(d / "meta.json") or {}
            vname = Path(str(meta.get("video", {}).get("path") or "")).name or None
            vpath = by_name[vname]["path"] if vname in by_name else None
            la = _read_json(d / "launcher_args.json") or {}
            opps = _read_json(d / "opportunities.json") or []
            mc = _read_json(d / "match_context.json") or {}
            out.append(dict(name=d.name, mtime=max((f.stat().st_mtime for f in d.iterdir() if f.is_file()), default=0),
                            has_report=(d / "report.html").exists(), has_tracks=(d / "tracks.csv").exists(),
                            has_review=(d / "review.json").exists(), video=vname,
                            has_events=(d / "events.csv").exists(),
                            video_id=_vid(vpath) if vpath else None, video_found=bool(vpath),
                            config=la.get("config"), setup=la.get("setup"), teams=mc.get("team_names"),
                            n_opps=len(opps), n_flagged=sum(1 for o in opps if o.get("flag")),
                            score=_score(d, mc.get("team_names"))))
        out.sort(key=lambda r: -r["mtime"])
        return out

    def run_detail(self, name: str) -> dict:
        d = self._run_dir(name)
        meta = _read_json(d / "meta.json") or {}
        mc = _read_json(d / "match_context.json") or {}
        vname = Path(str(meta.get("video", {}).get("path") or "")).name or None
        vpath = find_video(vname) if vname else None
        clips_dir = d / "clips" / "opportunities"
        clips = sorted(p.name for p in clips_dir.glob("*.mp4")) if clips_dir.exists() else []
        opps = _read_json(d / "opportunities.json") or []
        for o in opps:
            o["clip"] = o.get("clip_file") in clips
        km = []
        if (d / "key_metrics.csv").exists():
            import pandas as pd
            k = pd.read_csv(d / "key_metrics.csv")
            teams = mc.get("team_names") or []
            for _, r in k[k.confidence.isin(["High", "Medium", "Low"])].head(14).iterrows():
                km.append(dict(label=r.label, confidence=r.confidence,
                               values=[None if pd.isna(r.get(t)) else float(r.get(t)) for t in teams if t in r]))
        return dict(name=name, teams=mc.get("team_names"), periods=mc.get("periods"), score=_score(d, mc.get("team_names")),
                    video=vname, video_id=_vid(vpath) if vpath else None, video_path=vpath,
                    has_report=(d / "report.html").exists(), opportunities=opps, clips=clips, key_metrics=km,
                    launcher_args=_read_json(d / "launcher_args.json"), has_review=(d / "review.json").exists(),
                    has_tracks=(d / "tracks.csv").exists())

    def _run_dir(self, name: str) -> Path:
        d = (OUTPUTS / _safe_name(name)).resolve()
        if OUTPUTS.resolve() not in d.parents or not d.is_dir():
            raise FileNotFoundError(name)
        return d

    def video_by_id(self, vid: str) -> Optional[str]:
        for v in list_videos():
            if v["id"] == vid:
                return v["path"]
        return None

    def duration_min(self, path: str) -> Optional[float]:
        if path in self._dur_cache:
            return self._dur_cache[path]
        try:
            from ..video import probe
            d = probe(path).duration_s / 60
        except Exception:  # noqa: BLE001
            d = None
        self._dur_cache[path] = d
        return d

    # ---- commands
    def build(self, b: dict) -> dict:
        """Request from the page -> command line (also used for the preview)."""
        kind = b.get("kind")
        py = [self.python, "-u"]
        video = self.video_by_id(b["video_id"]) if b.get("video_id") else None
        run = _safe_name(b["run"]) if b.get("run") else (_safe_name(Path(video).stem) if video else None)
        cfg, setup = b.get("config") or None, b.get("setup") or None
        out = f"outputs/{run}" if run else None
        title, url = kind, None
        if kind == "analyse":
            if not video:
                raise ValueError("choose a video")
            cmd = py + ["run_analysis.py", "--video", _rel(video), "--out", out]
            cmd += ["--config", cfg] if cfg else []
            cmd += ["--setup", setup] if setup else []
            if b.get("max_frames"):
                cmd += ["--max-frames", str(int(b["max_frames"]))]
            if b.get("speed") in ("fast", "quality", "standard") and b.get("speed") != "standard":
                cmd += ["--speed", b["speed"]]
            if b.get("force"):
                cmd += ["--force"]
            title = f"Analyse {Path(video).name}" + (f" (first {b['max_frames']} frames)" if b.get("max_frames") else "")
        elif kind in ("analytics", "repair"):
            if not run:
                raise ValueError("choose a run")
            cmd = py + ["run_analysis.py", "--tracks", out, "--out", out]
            cmd += ["--config", cfg] if cfg else []
            cmd += ["--setup", setup] if setup else []
            cmd += ["--repair"] if kind == "repair" else []
            title = ("Repair camera + analytics" if kind == "repair" else "Re-run analytics") + f" — {run}"
        elif kind == "synthetic":
            run = "synthetic_demo"
            cmd = py + ["run_analysis.py", "--synthetic", "--out", "outputs/synthetic_demo"]
            title = "Synthetic demo match (no video needed)"
        elif kind in ("setup_ui", "review_ui"):
            mode = kind.split("_")[0]
            if not setup:
                stem = _safe_name(Path(video).stem if video else run or "match")
                setup = f"configs/match_setup_{stem}.json"
            cmd = py + ["scripts/match_ui.py", mode, "--setup", setup, "--no-browser", "--port", str(UI_PORT)]
            cmd += ["--config", cfg] if cfg else []
            if mode == "setup":
                if not video:
                    raise ValueError("choose a video")
                cmd += ["--video", _rel(video)]
                title = f"Setup UI — {Path(video).name}"
            else:
                if not run:
                    raise ValueError("choose a run to review")
                cmd += ["--run", out]
                if video:
                    cmd += ["--video", _rel(video)]
                title = f"Review UI — {run}"
            url = f"http://127.0.0.1:{UI_PORT}/"
        elif kind == "clips":
            if not run:
                raise ValueError("choose a run")
            cmd = py + ["scripts/cut_opportunities.py", "--run", out] + (["--all"] if b.get("all") else [])
            if video:
                cmd += ["--video", _rel(video)]
            title = f"Cut {'all' if b.get('all') else 'flagged'} opportunity clips — {run}"
        elif kind == "tests":
            cmd = py + ["tests/test_pipeline.py"]
            title = "Regression tests"
        else:
            raise ValueError(f"unknown job {kind}")
        return dict(kind=kind, cmd=cmd, title=title, run=run, video=video, url=url, config=cfg, setup=setup,
                    command=" ".join(_q(c) for c in cmd))

    def estimate(self, b: dict, spec: dict) -> Optional[dict]:
        """Rough processing time: minutes of video to analyse (periods only when the setup has them) x analysed fps
        / the speed measured on this computer by earlier runs with the same preset (meta.json -> speed)."""
        try:
            from ..config import SPEED_PRESETS, apply_speed, load_config
            cfg = load_config(str(ROOT / spec["config"]) if spec.get("config") else None)
            preset = b.get("speed") or "standard"
            apply_speed(cfg, preset)
            minutes = self.duration_min(spec["video"]) if spec.get("video") else None
            if spec.get("setup") and (ROOT / spec["setup"]).exists():
                per = [p for p in (_read_json(ROOT / spec["setup"]) or {}).get("periods", [])
                       if p.get("start") is not None and p.get("end") is not None]
                if per and cfg.video.only_periods:
                    m_ = sum(p["end"] - p["start"] + 2 * cfg.video.period_margin_s for p in per) / 60
                    minutes = min(m_, minutes) if minutes else m_
            frames = (minutes or 0) * 60 * cfg.video.analysis_fps
            if b.get("max_frames"):
                frames = min(frames or b["max_frames"], int(b["max_frames"]))
            fps, src = None, None
            for d in sorted(OUTPUTS.glob("*/meta.json"), key=lambda q: -q.stat().st_mtime):
                m = _read_json(d) or {}
                sp, mc = m.get("speed") or {}, (m.get("config") or {})
                if sp.get("fps") and (mc.get("detection") or {}).get("tile") == cfg.detection.tile:
                    fps, src = float(sp["fps"]), d.parent.name
                    break
            out = dict(preset=preset, note=SPEED_PRESETS[preset]["note"], minutes=round(minutes, 1) if minutes else None,
                       frames=int(frames) if frames else None, fps=fps, measured_on=src)
            if fps and frames:
                out["hours"] = round(frames / fps / 3600, 2)
            return out
        except Exception as e:  # noqa: BLE001 — an estimate must never break the page
            return {"error": str(e)}

    def start(self, b: dict) -> dict:
        spec = self.build(b)
        with self.lock:
            if spec["kind"] in ("analyse", "analytics", "repair") and any(
                    j.status == "running" and j.run == spec["run"] and j.kind in ("analyse", "analytics", "repair")
                    for j in self.jobs.values()):
                raise RuntimeError(f"a job is already writing outputs/{spec['run']}")
            if spec["kind"] in ("setup_ui", "review_ui"):         # one manual UI at a time (fixed port)
                for j in self.jobs.values():
                    if j.kind in ("setup_ui", "review_ui") and j.status == "running":
                        j.stop()
                time.sleep(0.3)
            dur = self.duration_min(spec["video"]) if spec["kind"] == "analyse" and spec["video"] else None
            start_min = 0.0
            if spec["kind"] == "analyse" and b.get("max_frames") and dur:
                dur = None                                          # unknown share of the video: no bar
            job = Job(spec["kind"], spec["title"], spec["cmd"], spec["run"], spec["video"], dur, spec["url"], start_min)
            self.jobs[job.id] = job
        if spec["kind"] in ("analyse", "analytics", "repair") and spec["run"]:
            d = OUTPUTS / spec["run"]
            d.mkdir(parents=True, exist_ok=True)
            (d / "launcher_args.json").write_text(json.dumps(
                {"config": spec["config"], "setup": spec["setup"], "video": _rel(spec["video"]) if spec["video"] else None,
                 "command": spec["command"], "at": time.strftime("%Y-%m-%d %H:%M")}, indent=1))
        return job.summary()

    def job_log(self, jid: str, since: int = 0) -> dict:
        j = self.jobs.get(jid)
        if j is None:
            raise FileNotFoundError(jid)
        lines = j.lines[since:]
        return dict(job=j.summary(), lines=lines, next=since + len(lines))

    # ---- manual decisions (setup / review UI mounted at /m/)
    def open_manual(self, b: dict) -> dict:
        from .server import App
        mode = b.get("mode", "setup")
        video = self.video_by_id(b["video_id"]) if b.get("video_id") else None
        run = _safe_name(b["run"]) if b.get("run") else None
        cfg = b.get("config") or None
        setup = b.get("setup") or None
        if mode == "review":
            if not run or not (OUTPUTS / run / "tracks.csv").exists():
                raise ValueError("review needs a processed match: run step 2 first (outputs/<run>/tracks.csv)")
            meta = _read_json(OUTPUTS / run / "meta.json") or {}
            if not video:
                video = find_video(Path(str(meta.get("video", {}).get("path") or "")).name)
            la = _read_json(OUTPUTS / run / "launcher_args.json") or {}
            cfg = cfg or la.get("config")
            setup = setup or la.get("setup")
        if not video or not Path(video).exists():
            raise ValueError("choose a recording (the video is needed to show the frames)")
        if not setup:
            setup = f"configs/match_setup_{_safe_name(Path(video).stem)}.json"
        with self.lock:
            self.manual = App(mode, str(video), str(ROOT / setup) if not Path(setup).is_absolute() else setup,
                              str(ROOT / cfg) if cfg and not Path(cfg).is_absolute() else cfg,
                              str(OUTPUTS / run) if (mode == "review" and run) else None)
            self.manual_info = dict(mode=mode, video=Path(video).name, setup=setup, config=cfg, run=run,
                                    opened=time.time())
        return dict(ok=True, **self.manual_info)

    # ---- live screen + replay overlays
    def live(self, name: str) -> dict:
        d = self._run_dir_any(name)
        st = _read_json(d / "live" / "state.json") or {}
        running = [j.summary() for j in self.jobs.values() if j.run == name and j.status == "running"]
        st["job"] = running[0] if running else None
        st["has_image"] = (d / "live" / "latest.jpg").exists()
        return st

    def _run_dir_any(self, name: str) -> Path:
        d = (OUTPUTS / _safe_name(name)).resolve()
        if OUTPUTS.resolve() not in d.parents:
            raise FileNotFoundError(name)
        return d

    def overlay_info(self, name: str) -> dict:
        from .viewer import get_overlay
        d = self._run_dir(name)
        if not (d / "tracks.csv").exists():
            raise FileNotFoundError(f"{name} has no tracking yet")
        info = get_overlay(d).info()
        det = self.run_detail(name)
        info.update(video_id=det.get("video_id"), video=det.get("video"), name=name)
        la = _read_json(d / "launcher_args.json") or {}
        sp = la.get("setup") or (self._setup_for(det["video"]) if det.get("video") else None)
        info["team_colors"] = _kit_colors(ROOT / sp) if sp else None
        return info

    def overlay_window(self, name: str, t0: float, t1: float) -> dict:
        from .viewer import get_overlay
        return get_overlay(self._run_dir(name)).window(t0, min(t1, t0 + 120))

    # ---- guide, desktop shortcuts, quit
    def doc(self, name: str) -> dict:
        if name not in DOCS:
            raise FileNotFoundError(name)
        p = ROOT / DOCS[name]
        return dict(name=name, path=DOCS[name], markdown=p.read_text(encoding="utf-8") if p.exists() else "(missing)")

    def shortcut(self, b: dict) -> dict:
        """Write a double-clickable file on the Desktop that opens the launcher with this recording (and its config /
        setup) already chosen, optionally starting the processing."""
        desk = Path.home() / "Desktop"
        if not desk.exists():
            raise FileNotFoundError("no Desktop folder")
        video = self.video_by_id(b["video_id"]) if b.get("video_id") else None
        args = []
        if video:
            args += ["--video", Path(video).name]
        for k in ("config", "setup", "run"):
            if b.get(k):
                args += [f"--{k}", str(b[k])]
        if b.get("start"):
            args += ["--start"]
        name = b.get("name") or (("Analyse " + Path(video).stem) if video else "Football Video Analysis")
        return dict(ok=True, path=str(write_shortcut(desk, name, args)))

    def quit(self):
        for j in self.jobs.values():
            j.stop()
        if self.server is not None:
            threading.Timer(0.3, self.server.shutdown).start()
        return dict(ok=True)

    # ---- library actions
    def upload(self, name: str, rfile, length: int, lib_name: Optional[str]) -> dict:
        name = Path(unquote(name)).name
        if Path(name).suffix.lower() not in VIDEO_EXT:
            raise ValueError(f"not a video file: {name}")
        st = load_storage()
        lib = (library(lib_name, st) if lib_name else None) or primary_library(st)
        dst_dir = Path(lib["path"])
        dst_dir.mkdir(parents=True, exist_ok=True)
        dst = dst_dir / _safe_name(name)
        if dst.exists():
            raise FileExistsError(f"{dst.name} is already in {lib['name']}")
        tmp = dst.with_name("." + dst.name + ".part")
        left = length
        with open(tmp, "wb") as f:
            while left > 0:
                chunk = rfile.read(min(8 << 20, left))
                if not chunk:
                    break
                f.write(chunk)
                left -= len(chunk)
        if left:
            tmp.unlink(missing_ok=True)
            raise IOError("upload interrupted")
        tmp.rename(dst)
        return dict(ok=True, path=_rel(dst), library=lib["name"])

    def archive(self, vid: str, target: Optional[str]) -> dict:
        st = load_storage()
        target = target or st.get("archive")
        if not target:
            raise ValueError("no archive library: set 'archive:' in configs/storage.yaml")
        path = self.video_by_id(vid)
        if not path:
            raise FileNotFoundError("video not found")
        for j in self.jobs.values():
            if j.status == "running" and j.video == path:
                raise RuntimeError("a job is using this video")
        return dict(ok=True, path=move_video(path, target, st))

    def open_folder(self, what: str) -> dict:
        st = load_storage()
        if what.startswith("run:"):
            p = self._run_dir(what[4:])
        elif what == "outputs":
            p = OUTPUTS
        else:
            lib = library(what, st)
            if lib is None:
                raise FileNotFoundError(what)
            p = Path(lib["path"])
            p.mkdir(parents=True, exist_ok=True)
        opener = "open" if sys.platform == "darwin" else ("explorer" if os.name == "nt" else "xdg-open")
        if shutil.which(opener) or os.name == "nt":
            subprocess.Popen([opener, str(p)])
            return dict(ok=True, path=str(p))
        return dict(ok=False, path=str(p))


def _kit_colors(setup_path: Path) -> Optional[list]:
    """Average shirt colour of the kit samples clicked in the setup, per team (for the overlays)."""
    st = _read_json(setup_path) or {}
    out = []
    for c in ("team0", "team1"):
        rgb = [s.get("rgb") for s in (st.get("kits") or {}).get(c, []) if s.get("rgb")]
        if not rgb:
            return None
        v = [[int(h[i:i + 2], 16) for i in (1, 3, 5)] for h in rgb]
        m = [int(sum(x[i] for x in v) / len(v)) for i in range(3)]
        out.append("#%02x%02x%02x" % tuple(m))
    return out


def _vid(path: str) -> str:
    from ..storage import video_id
    return video_id(path)


def _read_json(p: Path):
    try:
        return json.loads(p.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def _score(d: Path, teams) -> Optional[list]:
    f = d / "events.csv"
    if not f.exists():
        return None
    try:
        import pandas as pd
        e = pd.read_csv(f, usecols=["type", "team"])
        g = e[e.type == "goal"].groupby("team").size()
        return [int(g.get(0, 0)), int(g.get(1, 0))]
    except Exception:  # noqa: BLE001
        return None


# ----------------------------------------------------------------------------- HTTP
def make_handler(app: Launcher):
    html_path = HERE / "launcher.html"

    class H(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt, *args):
            pass

        def _send(self, code, body: bytes, ctype="application/octet-stream", extra: Optional[dict] = None):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            if not (extra and "Cache-Control" in extra):
                self.send_header("Cache-Control", "no-store")
            for k, v in (extra or {}).items():
                self.send_header(k, v)
            self.end_headers()
            if self.command != "HEAD":
                self.wfile.write(body)

        def _json(self, obj, code=200):
            self._send(code, json.dumps(obj, default=lambda o: o.item() if hasattr(o, "item") else str(o)).encode(),
                       "application/json")

        def _file(self, p: Path):
            """Static file with HTTP Range support (the browser's video player seeks with it)."""
            if not p.exists() or not p.is_file():
                return self._send(404, b"not found", "text/plain")
            size = p.stat().st_size
            ctype = mimetypes.guess_type(p.name)[0] or "application/octet-stream"
            if p.suffix.lower() in (".mp4", ".m4v", ".mov"):
                ctype = "video/mp4"
            if ctype.startswith("text/") or ctype in ("application/json",):
                ctype += "; charset=utf-8"
            rng = self.headers.get("Range")
            start, end = 0, size - 1
            code = 200
            if rng and rng.startswith("bytes="):
                a, _, b = rng[6:].split(",")[0].partition("-")
                if a:
                    start = int(a)
                    end = int(b) if b else size - 1
                else:
                    start = max(0, size - int(b))
                end = min(end, size - 1, start + (16 << 20) - 1)      # at most 16 MB per response
                if start >= size:
                    self.send_response(416)
                    self.send_header("Content-Range", f"bytes */{size}")
                    self.send_header("Content-Length", "0")
                    self.end_headers()
                    return
                code = 206
            n = end - start + 1
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Content-Length", str(n))
            if "/live/" in str(p):
                self.send_header("Cache-Control", "no-store")
            if code == 206:
                self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
            self.end_headers()
            if self.command == "HEAD":
                return
            try:
                with open(p, "rb") as f:
                    f.seek(start)
                    left = n
                    while left > 0:
                        chunk = f.read(min(1 << 20, left))
                        if not chunk:
                            break
                        self.wfile.write(chunk)
                        left -= len(chunk)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def do_HEAD(self):
            self.do_GET()

        def do_GET(self):
            try:
                u = urlparse(self.path)
                q = {k: v[0] for k, v in parse_qs(u.query).items()}
                path = unquote(u.path)
                if path in ("/", "/index.html"):
                    return self._send(200, html_path.read_bytes(), "text/html; charset=utf-8")
                if path == "/api/state":
                    return self._json(app.state())
                if path == "/m" or path == "/m/":
                    if app.manual is None:
                        return self._send(200, NO_MANUAL, "text/html; charset=utf-8")
                    from .server import app_html
                    return self._send(200, app_html("/m"), "text/html; charset=utf-8")
                if path.startswith("/m/"):
                    if app.manual is None:
                        return self._json({"error": "no setup / review open"}, 409)
                    from .server import route_get
                    code, body, ctype, cache = route_get(app.manual, path[2:], q)
                    return self._send(code, body, ctype, {"Cache-Control": "max-age=3600"} if cache else None)
                if path == "/api/manual":
                    return self._json(dict(open=app.manual is not None, **app.manual_info))
                if path.startswith("/api/doc/"):
                    return self._json(app.doc(path.split("/")[3]))
                if path.startswith("/api/live/"):
                    return self._json(app.live(path.split("/")[3]))
                if path.startswith("/api/overlay/"):
                    parts = path.split("/")
                    if len(parts) > 4 and parts[4] == "window":
                        return self._json(app.overlay_window(parts[3], float(q.get("t0", 0)), float(q.get("t1", 30))))
                    return self._json(app.overlay_info(parts[3]))
                if path.startswith("/api/run/"):
                    return self._json(app.run_detail(path[len("/api/run/"):]))
                if path.startswith("/api/jobs/"):
                    return self._json(app.job_log(path.split("/")[3], int(q.get("since", 0))))
                if path.startswith("/media/video/"):
                    v = app.video_by_id(path.split("/")[3])
                    return self._file(Path(v)) if v else self._send(404, b"video not found", "text/plain")
                if path.startswith("/runs/"):                       # report.html, clips, csv of a run
                    parts = path.split("/", 3)
                    d = app._run_dir_any(parts[2])
                    f = (d / (parts[3] if len(parts) > 3 and parts[3] else "report.html")).resolve()
                    if d not in f.parents:
                        return self._send(403, b"forbidden", "text/plain")
                    if f.name == "report.html" and f.exists():
                        return self._send(200, _report_for_browser(f, app, parts[2]), "text/html; charset=utf-8")
                    return self._file(f)
                self._send(404, b"not found", "text/plain")
            except FileNotFoundError as e:
                self._json({"error": f"not found: {e}"}, 404)
            except Exception as e:  # noqa: BLE001
                traceback.print_exc()
                self._json({"error": str(e)}, 500)

        def do_POST(self):
            try:
                u = urlparse(self.path)
                q = {k: v[0] for k, v in parse_qs(u.query).items()}
                n = int(self.headers.get("Content-Length", 0))
                if u.path == "/api/upload":
                    return self._json(app.upload(q.get("name", "video.mp4"), self.rfile, n, q.get("library")))
                body = json.loads(self.rfile.read(n) or b"{}")
                if u.path.startswith("/m/"):
                    if app.manual is None:
                        return self._json({"error": "no setup / review open"}, 409)
                    from .server import route_post
                    return self._json(route_post(app.manual, u.path[2:], body))
                if u.path == "/api/manual":
                    return self._json(app.open_manual(body))
                if u.path == "/api/shortcut":
                    return self._json(app.shortcut(body))
                if u.path == "/api/quit":
                    return self._json(app.quit())
                if u.path == "/api/preselect_done":
                    app.preselect = {}
                    return self._json({"ok": True})
                if u.path == "/api/preview":
                    spec = app.build(body)
                    est = app.estimate(body, spec) if spec["kind"] == "analyse" else None
                    return self._json({"command": spec["command"], "run": spec["run"], "estimate": est})
                if u.path == "/api/jobs":
                    return self._json(app.start(body))
                if u.path.startswith("/api/jobs/") and u.path.endswith("/stop"):
                    j = app.jobs.get(u.path.split("/")[3])
                    if j:
                        j.stop()
                    return self._json({"ok": bool(j)})
                if u.path == "/api/archive":
                    return self._json(app.archive(body["video_id"], body.get("library")))
                if u.path == "/api/open":
                    return self._json(app.open_folder(body["what"]))
                self._send(404, b"not found", "text/plain")
            except (ValueError, RuntimeError, FileExistsError, FileNotFoundError) as e:
                self._json({"error": str(e)}, 400)
            except Exception as e:  # noqa: BLE001
                traceback.print_exc()
                self._json({"error": str(e)}, 500)

    return H


NO_MANUAL = ("<!doctype html><meta charset=utf-8><body style='background:#121417;color:#9aa1a9;font:14px sans-serif;"
             "padding:40px'>Nothing open yet: choose a recording in the launcher and click <b>Setup</b> or "
             "<b>Review</b>.</body>").encode()


def _report_for_browser(f: Path, app: Launcher, run: str) -> bytes:
    """Serve report.html through the launcher: links to the full video point at the launcher's video endpoint
    (a relative file link does not work over http), clip links stay relative."""
    html = f.read_text(encoding="utf-8")
    det = app.run_detail(run)
    if det.get("video_id"):
        html = re.sub(r"href='[^']*?#t=(\d+)'", lambda m: f"href='/media/video/{det['video_id']}#t={m.group(1)}'", html)
    return html.encode()


def _autostart(app: Launcher):
    """--start: begin step 2 for the chosen video with the given / remembered config and setup."""
    pre = app.preselect
    try:
        st = app.state()
        v = next((x for x in st["videos"] if x["id"] == pre.get("video_id")), None)
        if v is None:
            print("--start: video not found in the libraries")
            return
        job = app.start(dict(kind="analyse", video_id=v["id"], config=pre.get("config") or v.get("config"),
                             setup=pre.get("setup") or v.get("setup"), run=pre.get("run"), force=pre.get("force")))
        pre["job"] = job["id"]
        print(f"Started: {job['title']}  — {job['command']}")
    except Exception as e:  # noqa: BLE001
        print(f"--start failed: {e}")


def write_shortcut(folder: Path, name: str, args: list) -> Path:
    """macOS: a ``.command`` file (double-click -> Terminal -> launcher). Windows: ``.bat``. Linux: ``.sh``."""
    import shlex
    safe = re.sub(r"[^A-Za-z0-9 _.()-]+", "_", name).strip() or "Football Video Analysis"
    a = " ".join(shlex.quote(x) for x in args)
    if os.name == "nt":
        p = folder / f"{safe}.bat"
        p.write_text(f'@echo off\r\ncd /d "{ROOT}"\r\n".venv\\Scripts\\python.exe" scripts\\launcher.py {a}\r\n')
        return p
    p = folder / (f"{safe}.command" if sys.platform == "darwin" else f"{safe}.sh")
    p.write_text("#!/bin/bash\n# Opens the football_video_evaluation launcher" + (f" with: {a}" if a else "") + "\n"
                 f"cd {shlex.quote(str(ROOT))} || {{ echo 'Project folder not found: {ROOT}'; read -r; exit 1; }}\n"
                 f"exec ./start.command {a}\n")
    p.chmod(0o755)
    return p


def preselect_url(base: str, pre: dict) -> str:
    from urllib.parse import urlencode
    q = {k: v for k, v in pre.items() if v not in (None, False, "")}
    return base + ("?" + urlencode(q) if q else "")


def serve(app: Launcher, host: str = "127.0.0.1", port: int = 8770, open_browser: bool = True):
    url = f"http://{host}:{port}/"
    try:
        srv = ThreadingHTTPServer((host, port), make_handler(app))
    except OSError:
        # already running (e.g. a second shortcut): hand the choice over to the open launcher and stop here
        import urllib.request
        try:
            urllib.request.urlopen(url + "api/state", timeout=3)
        except Exception:  # noqa: BLE001
            raise SystemExit(f"port {port} is busy with another program: use --port")
        print(f"The launcher is already running on {url} — opening it")
        if open_browser:
            import webbrowser
            webbrowser.open(preselect_url(url, app.preselect))
        return
    srv.daemon_threads = True
    app.server = srv
    print(f"Football video evaluation — launcher on {url}  (Ctrl+C to stop)")
    print(f"Python used for the jobs: {app.python}")
    if open_browser:
        import webbrowser
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    if app.preselect.get("start"):
        threading.Timer(0.2, lambda: _autostart(app)).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nstopping…")
    finally:
        for j in app.jobs.values():
            j.stop()
        srv.server_close()
