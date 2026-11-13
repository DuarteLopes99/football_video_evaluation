"""Where the match recordings live: one or more *video libraries* (folders) declared in ``configs/storage.yaml``.

A library is just a folder: the project's own ``data/videos`` (default, git-ignored), an external drive, or a
cloud-synced folder that downloads files on demand (Google Drive "Stream files", iCloud "Optimise Mac Storage",
OneDrive "Files On-Demand", Dropbox "online-only"). The pipeline only needs a video while it is being processed or
reviewed; the outputs of an analysed match (~10-20 MB with the opportunity clips) are kept in ``outputs/``, so a
full recording can be moved to an archive library afterwards. See docs/VIDEO_STORAGE.md.

Videos are referenced by file name: a path that no longer exists (e.g. ``data/raw/match.mp4`` after the file was
moved to ``data/videos`` or archived) is found again by name in the libraries (:func:`find_video`).
"""
from __future__ import annotations

import hashlib
import os
import shutil
from pathlib import Path
from typing import Optional

import yaml

from .config import PROJECT_ROOT

VIDEO_EXT = {".mp4", ".mov", ".m4v", ".mkv", ".avi"}
STORAGE_FILE = PROJECT_ROOT / "configs" / "storage.yaml"
DEFAULT = {"libraries": [{"name": "Local", "path": "data/videos"}], "archive": None}
LEGACY = [PROJECT_ROOT / "data" / "raw"]           # searched by name too (older runs and setups point there)


def _abs(p: str) -> Path:
    q = Path(os.path.expandvars(str(p))).expanduser()
    return q if q.is_absolute() else (PROJECT_ROOT / q)


def load_storage(path: Optional[str | Path] = None) -> dict:
    p = Path(path) if path else STORAGE_FILE
    st = DEFAULT
    if p.exists():
        st = yaml.safe_load(p.read_text()) or DEFAULT
    libs = []
    for i, lib in enumerate(st.get("libraries") or DEFAULT["libraries"]):
        if isinstance(lib, str):
            lib = {"name": Path(lib).name or f"Library {i + 1}", "path": lib}
        root = _abs(lib["path"])
        libs.append(dict(name=str(lib.get("name") or root.name), path=str(root), configured=str(lib["path"]),
                         exists=root.exists(), primary=(i == 0), note=str(lib.get("note", "") or "")))
    return {"libraries": libs, "archive": st.get("archive"), "file": str(p)}


def video_id(path: str | Path) -> str:
    return hashlib.sha1(str(Path(path).resolve()).encode()).hexdigest()[:12]


def _placeholder_target(p: Path) -> Optional[Path]:
    """iCloud keeps evicted files as '.<name>.icloud' placeholders."""
    if p.name.startswith(".") and p.name.endswith(".icloud"):
        return p.with_name(p.name[1:-len(".icloud")])
    return None


def _cloud_only(p: Path) -> bool:
    """Heuristic: a cloud-synced file whose bytes are not on this disk has (almost) no allocated blocks."""
    try:
        s = p.stat()
        return hasattr(s, "st_blocks") and s.st_size > 1_000_000 and s.st_blocks * 512 < 0.05 * s.st_size
    except OSError:
        return False


def list_videos(st: Optional[dict] = None, include_legacy: bool = True) -> list:
    st = st or load_storage()
    roots = [(lib["name"], Path(lib["path"])) for lib in st["libraries"] if lib["exists"]]
    if include_legacy:
        roots += [("data/raw (old)", r) for r in LEGACY if r.exists() and r not in [x[1] for x in roots]]
    out, seen = [], set()
    for name, root in roots:
        for p in sorted(root.rglob("*")):
            if any(part.startswith(".") and part not in (".", "..") for part in p.relative_to(root).parts[:-1]):
                continue
            target = _placeholder_target(p)
            real = target or p
            if real.suffix.lower() not in VIDEO_EXT or not (p.is_file()):
                continue
            if p.name.startswith(".") and target is None:
                continue
            key = str(real.resolve())
            if key in seen:
                continue
            seen.add(key)
            s = p.stat()
            out.append(dict(id=video_id(real), name=real.name, path=str(real), library=name,
                            rel=str(real.relative_to(root)), size_mb=round(s.st_size / 1e6, 1) if target is None else None,
                            mtime=s.st_mtime, cloud_only=bool(target is not None or _cloud_only(p))))
    out.sort(key=lambda v: (-v["mtime"], v["name"]))
    return out


def find_video(name_or_path: Optional[str], st: Optional[dict] = None) -> Optional[str]:
    """Return an existing path for a video given its path or just its file name (searched in every library)."""
    if not name_or_path:
        return name_or_path
    p = Path(str(name_or_path)).expanduser()
    for cand in (p, PROJECT_ROOT / p):
        if cand.exists():
            return str(cand)
    for v in list_videos(st):
        if v["name"] == p.name and not v["cloud_only"]:
            return v["path"]
    for v in list_videos(st):                         # cloud-only: opening it downloads it on demand
        if v["name"] == p.name:
            return v["path"]
    return None


def library(name: Optional[str], st: Optional[dict] = None) -> Optional[dict]:
    st = st or load_storage()
    for lib in st["libraries"]:
        if lib["name"] == name:
            return lib
    return None


def primary_library(st: Optional[dict] = None) -> dict:
    st = st or load_storage()
    return st["libraries"][0]


def move_video(path: str, target_lib: str, st: Optional[dict] = None) -> str:
    """Move a recording to another library (e.g. the archive). Outputs keep working: runs find it by name."""
    lib = library(target_lib, st)
    if lib is None:
        raise ValueError(f"unknown library {target_lib!r} (configs/storage.yaml)")
    dst_dir = Path(lib["path"])
    dst_dir.mkdir(parents=True, exist_ok=True)
    src = Path(path)
    dst = dst_dir / src.name
    if dst.exists():
        raise FileExistsError(f"{dst} already exists")
    shutil.move(str(src), str(dst))
    return str(dst)


def disk_free_gb(path: str) -> Optional[float]:
    try:
        return round(shutil.disk_usage(path).free / 1e9, 1)
    except OSError:
        return None
