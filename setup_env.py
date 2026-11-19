#!/usr/bin/env python3
"""Create / repair the project's Python environment, the same on every computer.

    python setup_env.py                 # check and install what is needed (fast when nothing changed)
    python setup_env.py --run [args]    # ... then start the dashboard (start.command / start.bat do this)
    python setup_env.py --check         # only report
    python setup_env.py --force         # reinstall every package
    python setup_env.py --cpu           # CPU detector runtime even on an NVIDIA computer
    python setup_env.py --python PATH   # build the environment with this interpreter

What it guarantees:

1. **A supported Python** (3.12, 3.13 or 3.14, 64-bit; 3.13 preferred). The interpreter running this script
   only bootstraps: it can be any Python 3.8+ (the Mac's system 3.9, a brand-new 3.15 ...). A supported one is
   looked up (``py -3.13`` on Windows, ``python3.13`` / Homebrew / python.org on macOS, ...); when none exists the
   script says how to install one and, if you agree, does it (Python install manager / winget / Homebrew).
2. **One environment per computer** in ``.venv`` (``.venv-windows`` / ``.venv-posix`` if the folder is synced
   between a Mac and a PC and ``.venv`` belongs to the other one). An environment built with an unsupported
   Python is rebuilt.
3. **The same package versions everywhere**: ``requirements.txt`` + the exact pins of ``constraints.txt``.
   A stamp of those files, the Python version and the detector runtime skips the install when nothing changed.
4. **The right detector runtime**: ``onnxruntime-gpu[cuda,cudnn]`` when ``nvidia-smi`` finds an NVIDIA GPU, else
   ``onnxruntime`` (CoreML on a Mac); never both. If it cannot be installed the detector uses cv2.dnn.
5. **The model files** (``*.onnx``, not in git): fetched from the OpenCV Model Zoo when missing, at every start.
6. **A check** (``football_eval.envcheck``) that what is installed matches the pins; ``pip freeze`` is saved in
   the environment folder for comparison between computers.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

SUPPORTED = ((3, 12), (3, 14))           # inclusive; keep in sync with global_video_analysis/football_eval/envcheck.py
PREFERRED = [(3, 13), (3, 12), (3, 14)]  # 3.13 = the version the tests run on
ROOT = Path(__file__).resolve().parent
IS_WIN = os.name == "nt"
STAMP_FILES = ["requirements.txt", "constraints.txt", "global_video_analysis/requirements.txt",
               "offside_code/requirements.txt", "setup_env.py"]
ENV = dict(os.environ, PYTHONUTF8="1", PYTHONIOENCODING="utf-8", PIP_DISABLE_PIP_VERSION_CHECK="1")


def say(msg: str = "") -> None:
    print(msg, flush=True)


def ver_str(v) -> str:
    return f"{v[0]}.{v[1]}"


def supported(v) -> bool:
    return SUPPORTED[0] <= tuple(v) <= SUPPORTED[1]


# ------------------------------------------------------------------ interpreters
def probe(cmd: list):
    """(major, minor), bits, executable of an interpreter command, or None."""
    code = "import sys,struct;print(sys.version_info[0],sys.version_info[1],struct.calcsize('P')*8,sys.executable)"
    try:
        r = subprocess.run(list(cmd) + ["-c", code], capture_output=True, text=True, timeout=30, env=ENV)
    except (OSError, subprocess.TimeoutExpired):
        return None
    if r.returncode != 0:
        return None
    try:
        a, b, bits, exe = r.stdout.strip().split(" ", 3)
        return (int(a), int(b)), int(bits), exe
    except ValueError:
        return None


def candidate_commands() -> list:
    cands = [[sys.executable]]
    names = [ver_str(v) for v in PREFERRED]
    if IS_WIN:
        for n in names:                                   # classic launcher and the Python install manager
            cands += [["py", f"-{n}"], ["py", f"-V:{n}"]]
        local = Path(os.environ.get("LOCALAPPDATA", ""))
        for n in names:
            tag = n.replace(".", "")
            cands += [[str(local / "Programs" / "Python" / f"Python{tag}" / "python.exe")],
                      [str(local / "Python" / f"pythoncore-{n}-64" / "python.exe")],
                      [rf"C:\Program Files\Python{tag}\python.exe"]]
        cands.append(["python"])
    else:
        dirs = ["/opt/homebrew/bin", "/usr/local/bin", "/usr/bin", str(Path.home() / ".local" / "bin")]
        for n in names:
            w = shutil.which(f"python{n}")
            if w:
                cands.append([w])
            cands += [[f"{d}/python{n}"] for d in dirs]
            cands.append([f"/Library/Frameworks/Python.framework/Versions/{n}/bin/python{n}"])
            cands += [[str(p / "bin" / "python3")] for p in sorted((Path.home() / ".pyenv" / "versions").glob(f"{n}.*"))]
        cands.append(["python3"])
    seen, out = set(), []
    for c in cands:
        key = tuple(c)
        if key in seen:
            continue
        seen.add(key)
        if len(c) == 1 and os.sep in c[0] and not Path(c[0]).exists():
            continue
        out.append(c)
    return out


def find_interpreters() -> list:
    """Every interpreter found: [(version, bits, exe)], unique by executable."""
    found, seen = [], set()
    for c in candidate_commands():
        p = probe(c)
        if p and p[2] not in seen:
            seen.add(p[2])
            found.append(p)
    return found


def pick(found: list):
    ok = [f for f in found if supported(f[0]) and f[1] == 64]
    for v in PREFERRED:
        for f in ok:
            if f[0] == v:
                return f
    return ok[0] if ok else None


def offer_install(assume_yes: bool) -> bool:
    """No supported Python: explain, and install Python 3.13 when a package manager is there and the user agrees."""
    want = ver_str(PREFERRED[0])
    if IS_WIN:
        if shutil.which("pymanager"):
            cmd = ["pymanager", "install", want]
        elif shutil.which("winget"):
            cmd = ["winget", "install", "-e", "--id", f"Python.Python.{want}",
                   "--accept-package-agreements", "--accept-source-agreements"]
        else:
            cmd = None
        manual = (f"Install Python {want} (64-bit) from https://www.python.org/downloads/windows/ "
                  f"and run start.bat again.")
    elif sys.platform == "darwin":
        cmd = ["brew", "install", f"python@{want}"] if shutil.which("brew") else None
        manual = f"Install Python {want} from https://www.python.org/downloads/macos/ and run start.command again."
    else:
        cmd = None
        manual = (f"Install Python {want}, e.g. 'sudo apt install python{want} python{want}-venv' or "
                  f"'uv python install {want}', and run ./start.command again.")
    say(manual)
    if cmd is None:
        return False
    say(f"It can be installed now with:  {' '.join(cmd)}")
    if not assume_yes:
        if not sys.stdin or not sys.stdin.isatty():
            return False
        try:
            if input("Install it now? [Y/n] ").strip().lower() not in ("", "y", "yes", "s", "sim"):
                return False
        except EOFError:
            return False
    say(f"$ {' '.join(cmd)}")
    return subprocess.call(cmd) == 0


# ------------------------------------------------------------------ environment folder
def venv_python(d: Path) -> Path:
    return d / ("Scripts/python.exe" if IS_WIN else "bin/python")


def foreign_venv(d: Path) -> bool:
    """A venv made on the other OS (folder synced between Mac and PC): never delete it."""
    cfg = d / "pyvenv.cfg"
    if not cfg.exists():
        return False
    home = next((ln.split("=", 1)[1].strip() for ln in cfg.read_text(encoding="utf-8", errors="replace").splitlines()
                 if ln.strip().lower().startswith("home")), "")
    windows_home = bool(re.match(r"^[A-Za-z]:\\", home))
    return windows_home != IS_WIN


def venv_dir() -> Path:
    d = ROOT / ".venv"
    if d.exists() and foreign_venv(d):
        return ROOT / (".venv-windows" if IS_WIN else ".venv-posix")
    return d


def remove_tree(d: Path) -> None:
    def onerror(func, path, _exc):
        os.chmod(path, 0o700)
        func(path)
    try:
        shutil.rmtree(d, onerror=onerror)
    except OSError as e:
        sys.exit(f"Cannot remove the old environment {d}: {e}\nClose every program that uses it "
                 f"(the dashboard, Jupyter, VS Code) and run again.")


def ensure_venv(d: Path, python_cmd, force_python: bool):
    """Return (venv python path, version). Rebuild when missing, broken or made with an unsupported Python."""
    vp = venv_python(d)
    cur = probe([str(vp)]) if vp.exists() else None
    if cur and supported(cur[0]) and cur[1] == 64 and not force_python:
        return vp, cur[0]
    if Path(sys.prefix).resolve() == d.resolve():
        sys.exit(f"This script is running from {d}, which must be rebuilt: run it with another Python "
                 f"(start.command / start.bat do).")
    chosen = python_cmd
    if d.exists():
        why = (f"made with Python {ver_str(cur[0])} ({cur[1]}-bit)" if cur else "broken") if not force_python \
            else "rebuilt with the requested interpreter"
        say(f"Removing the environment {d.name} ({why}).")
        remove_tree(d)
    say(f"Creating {d.name} with Python {ver_str(chosen[0])} ({chosen[2]}) ...")
    r = subprocess.call([chosen[2], "-m", "venv", str(d)], env=ENV)
    if r != 0 or not vp.exists():
        sys.exit("Could not create the virtual environment (on Linux: install the python3.X-venv package).")
    return vp, chosen[0]


# ------------------------------------------------------------------ packages
def nvidia_gpu() -> bool:
    exe = shutil.which("nvidia-smi")
    if not exe:
        return False
    try:
        r = subprocess.run([exe, "-L"], capture_output=True, text=True, timeout=15)
    except (OSError, subprocess.TimeoutExpired):
        return False
    return r.returncode == 0 and "GPU" in r.stdout


def pip(vp: Path, *args, check: bool = True) -> int:
    cmd = [str(vp), "-m", "pip", *args]
    say("$ pip " + " ".join(args))
    r = subprocess.call(cmd, cwd=str(ROOT), env=ENV)
    if check and r != 0:
        sys.exit(f"pip failed (exit {r}). See the messages above; run 'python setup_env.py --force' to retry.")
    return r


def installed(vp: Path, names) -> dict:
    code = ("import json,importlib.metadata as m\nout={}\nfor n in %r:\n"
            "    try: out[n]=m.version(n)\n    except Exception: out[n]=None\nprint(json.dumps(out))" % (list(names),))
    r = subprocess.run([str(vp), "-c", code], capture_output=True, text=True, env=ENV)
    try:
        return json.loads(r.stdout)
    except ValueError:
        return {n: None for n in names}


def install_runtime(vp: Path, gpu: bool) -> str:
    """onnxruntime-gpu (NVIDIA) or onnxruntime, never both. Returns what is installed ('' = none, cv2.dnn)."""
    have = installed(vp, ["onnxruntime", "onnxruntime-gpu"])
    want = "onnxruntime-gpu" if gpu else "onnxruntime"
    other = "onnxruntime" if gpu else "onnxruntime-gpu"
    if have.get(other):                                  # the two share files: remove both, then install one
        pip(vp, "uninstall", "-y", "-q", "onnxruntime", "onnxruntime-gpu", check=False)
    spec = "onnxruntime-gpu[cuda,cudnn]" if gpu else "onnxruntime"
    if pip(vp, "install", "-q", "-c", "constraints.txt", spec, check=False) == 0:
        return want
    if gpu:
        say("onnxruntime-gpu could not be installed: using the CPU runtime.")
        pip(vp, "uninstall", "-y", "-q", "onnxruntime-gpu", check=False)
        if pip(vp, "install", "-q", "-c", "constraints.txt", "onnxruntime", check=False) == 0:
            return "onnxruntime"
    say("onnxruntime is not available for this computer (on a Mac it needs macOS 14+): "
        "the detector uses cv2.dnn instead (slower, same results).")
    return ""


def stamp(pyver, runtime_kind: str) -> str:
    h = hashlib.sha256()
    for f in STAMP_FILES:
        p = ROOT / f
        h.update(f.encode() + b"\0" + (p.read_bytes() if p.exists() else b""))
    h.update(f"py{ver_str(pyver)}|{runtime_kind}|{sys.platform}".encode())
    return h.hexdigest()


MODEL_FILES = {   # file -> fetch script (each also copies from the sibling project before downloading)
    "global_video_analysis/models/yolox.onnx": "global_video_analysis/scripts/get_models.py",
    "offside_code/models/yolox.onnx": "offside_code/scripts/download_models.py",
    "offside_code/models/mp_pose.onnx": "offside_code/scripts/download_models.py",
}


def ensure_models(vp: Path) -> bool:
    """Run the fetch scripts of the models that are missing (or < 1 MB: a git-LFS pointer). Never fatal."""
    missing = [f for f in MODEL_FILES if not ((ROOT / f).is_file() and (ROOT / f).stat().st_size >= 1_000_000)]
    ok = True
    for script in dict.fromkeys(MODEL_FILES[f] for f in missing):
        say(f"Fetching the models of {script.split('/')[0]} (not in git: ~40 MB, once) ...")
        if subprocess.call([str(vp), str(ROOT / script)], cwd=str((ROOT / script).parent.parent), env=ENV) != 0:
            ok = False
            say("The models could not be downloaded (no internet?). The dashboard starts, but processing a video "
                "needs them: run start.command / start.bat again when online.")
    return ok


def envcheck(vp: Path) -> int:
    return subprocess.call([str(vp), "-m", "football_eval.envcheck"], cwd=str(ROOT / "global_video_analysis"), env=ENV)


# ------------------------------------------------------------------ main
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true", help="only report, change nothing")
    ap.add_argument("--force", action="store_true", help="reinstall every package")
    ap.add_argument("--cpu", action="store_true", help="CPU detector runtime even with an NVIDIA GPU")
    ap.add_argument("--python", default=None, help="interpreter to build the environment with")
    ap.add_argument("--yes", action="store_true", help="install a supported Python without asking")
    ap.add_argument("--run", nargs=argparse.REMAINDER, default=None,
                    help="after the setup, start the dashboard with these arguments")
    a = ap.parse_args()
    os.chdir(ROOT)
    d = venv_dir()
    lo, hi = SUPPORTED

    if a.check:
        vp = venv_python(d)
        if not vp.exists():
            say(f"No environment yet ({d.name}). Run: python setup_env.py")
            return 1
        return envcheck(vp)

    # 1. a supported interpreter (only needed to build the environment)
    vp = venv_python(d)
    cur = probe([str(vp)]) if vp.exists() else None
    need_build = a.python is not None or not (cur and supported(cur[0]) and cur[1] == 64)
    chosen = None
    if need_build:
        if a.python:
            chosen = probe([a.python])
            if not chosen or not supported(chosen[0]) or chosen[1] != 64:
                sys.exit(f"{a.python}: need a 64-bit Python {ver_str(lo)}-{ver_str(hi)}.")
        else:
            found = find_interpreters()
            chosen = pick(found)
            if chosen is None:
                say(f"No supported Python found. The project needs Python {ver_str(lo)}-{ver_str(hi)} (64-bit); "
                    f"{ver_str(PREFERRED[0])} recommended.")
                if found:
                    say("Found: " + ", ".join(f"{ver_str(v)} {b}-bit ({e})" for v, b, e in found))
                    if any(v > hi for v, _, _ in found):
                        say(f"Python {ver_str((hi[0], hi[1] + 1))} and newer are not supported yet: onnxruntime has "
                            f"no wheels for them (the error 'No matching distribution found for onnxruntime').")
                if offer_install(a.yes):
                    chosen = pick(find_interpreters())
                if chosen is None:
                    say("After installing it, run start.command / start.bat (or python setup_env.py) again.")
                    return 2
    vp, pyver = ensure_venv(d, chosen, force_python=a.python is not None)

    # 2. packages, pinned
    gpu = nvidia_gpu() and not a.cpu
    kind = "gpu" if gpu else "cpu"
    st_file = d / ".setup_stamp"
    st = stamp(pyver, kind)
    if a.force or not st_file.exists() or st_file.read_text(encoding="utf-8").strip() != st:
        say(f"Installing the pinned packages into {d.name} (Python {ver_str(pyver)}, "
            f"{'NVIDIA GPU' if gpu else 'CPU / Apple GPU'}) ...")
        pip(vp, "install", "-q", "--upgrade", "pip")
        pip(vp, "install", "-q", *(["--force-reinstall"] if a.force else []), "-r", "requirements.txt")
        rt = install_runtime(vp, gpu)
        pip(vp, "check", check=False)
        freeze = subprocess.run([str(vp), "-m", "pip", "freeze"], capture_output=True, text=True, env=ENV).stdout
        (d / "pip-freeze.txt").write_text(freeze, encoding="utf-8")
        st_file.write_text(st, encoding="utf-8")
        say(f"Environment ready: Python {ver_str(pyver)}, detector runtime {rt or 'cv2.dnn'}.")
        if envcheck(vp) != 0:
            say("(the dashboard still starts; the differences above may change results slightly)")
    # 3. the model files (every start: cheap when they are there)
    ensure_models(vp)
    # 4. start the dashboard
    if a.run is not None:
        return subprocess.call([str(vp), str(ROOT / "global_video_analysis" / "scripts" / "launcher.py"), *a.run],
                               cwd=str(ROOT), env=ENV)
    return 0


if __name__ == "__main__":
    sys.exit(main())
