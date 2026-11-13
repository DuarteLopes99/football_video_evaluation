"""Model files that are not in git (too big): fetched on demand, checked, never left half-downloaded.

``models/*.onnx`` is git-ignored, so a fresh clone (a new PC) has no detector. ``setup_env.py`` fetches it at the
first start, and :func:`ensure` runs again before every analysis: a missing file is copied from the sibling project
(``../offside_code/models``) or downloaded from the OpenCV Model Zoo. A file below 1 MB (a git-LFS pointer or an
interrupted download) is replaced. A different but valid model you put there yourself is never overwritten.

    python scripts/get_models.py
"""
from __future__ import annotations

import hashlib
import shutil
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ZOO = "https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/"
MODELS = {
    "yolox.onnx": dict(url=ZOO + "object_detection_yolox/object_detection_yolox_2022nov.onnx",
                       size=35_858_002, sha256="c5c2d13e59ae883e6af3b45daea64af4833a4951c92d116ec270d9ddbe998063",
                       siblings=[ROOT.parent / "offside_code" / "models" / "yolox.onnx"]),
}
MIN_BYTES = 1_000_000


class ModelMissing(FileNotFoundError):
    pass


def usable(p: Path) -> bool:
    """Present and not a placeholder (git-LFS pointer files are ~130 bytes)."""
    return p.is_file() and p.stat().st_size >= MIN_BYTES


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download(url: str, dst: Path, log=print, tries: int = 3, timeout: float = 60.0) -> None:
    """To ``dst.part`` first, then renamed: an interrupted download never looks like a model."""
    tmp = dst.with_name(dst.name + ".part")
    last = None
    for attempt in range(1, tries + 1):
        try:
            with urllib.request.urlopen(url, timeout=timeout) as r, open(tmp, "wb") as f:
                total = int(r.headers.get("Content-Length") or 0)
                done, step = 0, max(total // 5, 1)
                while True:
                    chunk = r.read(1 << 20)
                    if not chunk:
                        break
                    f.write(chunk)
                    done += len(chunk)
                    if total and done // step != (done - len(chunk)) // step:
                        log(f"  {dst.name}: {100 * done // total} %")
            if tmp.stat().st_size < MIN_BYTES:
                raise IOError(f"only {tmp.stat().st_size} bytes received")
            tmp.replace(dst)
            return
        except Exception as e:  # noqa: BLE001 — network errors of every kind: retry, then report
            last = e
            tmp.unlink(missing_ok=True)
            if attempt < tries:
                time.sleep(2 * attempt)
    raise ModelMissing(f"download of {dst.name} failed ({last})")


def ensure(name: str = "yolox.onnx", dst: Path | None = None, log=print) -> Path:
    """Path of a usable model ``name``: as it is, copied from the sibling project, or downloaded."""
    spec = MODELS[name]
    dst = Path(dst) if dst else ROOT / "models" / name
    if usable(dst):
        return dst
    dst.parent.mkdir(parents=True, exist_ok=True)
    for s in spec.get("siblings", []):
        if usable(s):
            shutil.copy2(s, dst)
            log(f"[models] {name}: copied from {s}")
            return dst
    log(f"[models] {name} missing: downloading {spec['size'] // 1_000_000} MB from the OpenCV Model Zoo ...")
    try:
        download(spec["url"], dst, log)
    except ModelMissing as e:
        raise ModelMissing(
            f"{e}.\nThe detector model {dst} is not in git (too big). With internet, run "
            f"'python scripts/get_models.py' (inside global_video_analysis/), or download\n  {spec['url']}\n"
            f"and save it as {dst}.") from None
    if spec.get("sha256") and sha256(dst) != spec["sha256"]:
        log(f"[models] warning: {name} checksum differs from the tested file (kept; results may differ slightly)")
    log(f"[models] {name}: {dst.stat().st_size // 1_000_000} MB -> {dst}")
    return dst


def resolve_model(model_path: str | None, log=print) -> str:
    """Detector path from the config -> an existing file. The default YOLOX is fetched when missing; any other
    missing file is a clear error instead of OpenCV's 'Can't read ONNX file'."""
    p = Path(model_path or "models/yolox.onnx")
    if usable(p) or usable(ROOT / p):
        return str(p if usable(p) else ROOT / p)
    if p.name in MODELS:
        return str(ensure(p.name, ROOT / "models" / p.name, log))
    raise ModelMissing(f"detector model not found: {model_path} (looked in the current folder and {ROOT}). "
                       f"Check detection.model_path in the config.")
