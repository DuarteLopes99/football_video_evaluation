"""Hardware acceleration: use the GPU automatically when the computer has one.

Nothing has to be configured. With ``detection.runtime: auto`` and ``detection.device: auto`` (the defaults):

* **YOLOX (onnxruntime)** tries every GPU execution provider the installed onnxruntime offers, in this order,
  benchmarks each on a dummy tile and keeps the fastest (CPU is always a candidate, so a slow or broken GPU
  provider never makes things worse):

  ======================  =========================================  ===================================
  provider                hardware                                   package
  ======================  =========================================  ===================================
  CUDAExecutionProvider   NVIDIA (Linux / Windows)                   ``onnxruntime-gpu[cuda,cudnn]``
  ROCMExecutionProvider   AMD (Linux)                                onnxruntime ROCm build
  DmlExecutionProvider    any DirectX 12 GPU (Windows)               ``onnxruntime-directml``
  CoreMLExecutionProvider Apple Silicon (Neural Engine / GPU)        ``onnxruntime`` (default wheel)
  OpenVINOExecutionProvider Intel iGPU / CPU                         ``onnxruntime-openvino``
  ======================  =========================================  ===================================

  TensorRT is the fastest on NVIDIA but builds an engine for minutes on the first run, so it is only used when
  asked for (``runtime: tensorrt``); the engine is cached in ``models/trt_cache/``.

* **cv2.dnn** (fallback when onnxruntime is missing) runs on CUDA when OpenCV was built with it.
* **PyTorch** models (ultralytics detectors / pitch keypoints, SigLIP team embeddings) get ``cuda:0`` on NVIDIA,
  ``mps`` on Apple Silicon, else ``cpu``.

``start.command`` / ``start.bat`` install ``onnxruntime-gpu`` instead of ``onnxruntime`` when ``nvidia-smi`` finds
an NVIDIA GPU. Force the CPU with ``--device cpu`` (CLI) or ``detection.runtime: cpu`` / ``detection.device: cpu``.

Check what this computer will use::

    python -m football_eval.accel             # summary
    python -m football_eval.accel --doctor    # CUDA listed but not used? full diagnosis + the fix for this machine

**"CUDAExecutionProvider" listed is not "CUDA works".** onnxruntime lists the CUDA provider as soon as the
``onnxruntime-gpu`` package is installed; whether the GPU can run it is only known when a session is created. A
broken CUDA set-up used to print a long ``EP Error ... CUDA failure 801`` and fall back to the CPU silently. Now
the CUDA provider is first tried in a **separate process** (:func:`probe_cuda`, a few seconds, once per run): when it
fails the detector skips CUDA with one line that says why and what to install, and tries the next GPU provider.

Most common cause on Windows since onnxruntime-gpu 1.27 (PyPI default build moved from CUDA 12 to CUDA 13): CUDA 13
needs an NVIDIA driver >= 580 and a GPU of compute capability >= 7.5 (RTX 20xx / GTX 16xx or newer). An older
driver or a GTX 10xx / 900 card fails at ``cudaSetDevice`` (error 801 / 35 / 100). Fix: update the driver, or install
the CUDA 12 build (``onnxruntime-gpu[cuda,cudnn]>=1.21,<1.27``, driver >= 525), or DirectML (any DirectX 12 GPU, no
CUDA at all). :func:`recommend_onnxruntime` picks the right one from ``nvidia-smi``; ``start.bat`` uses it.
"""
from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path
from typing import Dict, List, Optional

# onnxruntime GPU providers worth trying, fastest family first (TensorRT only on request, see the docstring)
GPU_PROVIDERS = ["CUDAExecutionProvider", "ROCMExecutionProvider", "MIGraphXExecutionProvider",
                 "DmlExecutionProvider", "CoreMLExecutionProvider", "OpenVINOExecutionProvider"]
# runtime value -> providers it allows
RUNTIME_ALIASES = {
    "cuda": ["CUDAExecutionProvider"], "nvidia": ["CUDAExecutionProvider"],
    "tensorrt": ["TensorrtExecutionProvider", "CUDAExecutionProvider"], "trt": ["TensorrtExecutionProvider", "CUDAExecutionProvider"],
    "rocm": ["ROCMExecutionProvider", "MIGraphXExecutionProvider"], "amd": ["ROCMExecutionProvider", "MIGraphXExecutionProvider"],
    "directml": ["DmlExecutionProvider"], "dml": ["DmlExecutionProvider"],
    "coreml": ["CoreMLExecutionProvider"], "openvino": ["OpenVINOExecutionProvider"],
}
CPU_ONLY = ("cpu", "none", "off")
_PRELOADED = False


def _preload_cuda_dlls(ort) -> None:
    """onnxruntime-gpu >= 1.21 can load the CUDA / cuDNN libraries installed by pip (``onnxruntime-gpu[cuda,cudnn]``),
    so no system CUDA toolkit is needed. Harmless when they are missing or already loaded."""
    global _PRELOADED
    if _PRELOADED or not hasattr(ort, "preload_dlls") or os.environ.get("FVE_NO_PRELOAD") == "1":
        return
    _PRELOADED = True
    try:
        ort.preload_dlls()
    except Exception:  # noqa: BLE001 — system CUDA (if any) is used instead
        pass


def ort_available() -> List[str]:
    try:
        import onnxruntime as ort
    except ImportError:
        return []
    if "CUDAExecutionProvider" in ort.get_available_providers():
        _preload_cuda_dlls(ort)
    return list(ort.get_available_providers())


def provider_candidates(runtime: str = "auto", available: Optional[List[str]] = None,
                        trt_cache: Optional[str] = None, gpu_id: int = 0) -> List[list]:
    """Provider lists to benchmark, GPU first, CPU always last. ``runtime``: auto | gpu | cpu | cuda | tensorrt |
    rocm | directml | coreml | openvino (also the legacy ``onnxruntime`` / ``ort`` = auto)."""
    avail = ort_available() if available is None else list(available)
    rt = (runtime or "auto").lower()
    if rt in CPU_ONLY:
        return [["CPUExecutionProvider"]]
    wanted = RUNTIME_ALIASES.get(rt, GPU_PROVIDERS)
    if rt in ("auto", "gpu", "onnxruntime", "ort"):
        wanted = GPU_PROVIDERS
    cands = []
    for p in wanted:
        if p not in avail:
            continue
        if p == "TensorrtExecutionProvider":
            cache = trt_cache or str(Path(__file__).resolve().parents[1] / "models" / "trt_cache")
            os.makedirs(cache, exist_ok=True)
            cands.append([(p, {"trt_engine_cache_enable": True, "trt_engine_cache_path": cache, "trt_fp16_enable": True,
                               "device_id": int(gpu_id)}),
                          ("CUDAExecutionProvider", {"device_id": int(gpu_id)}), "CPUExecutionProvider"])
        elif p == "CUDAExecutionProvider":
            cands.append([(p, {"device_id": int(gpu_id)}), "CPUExecutionProvider"])
        else:
            cands.append([p, "CPUExecutionProvider"])
    cands.append(["CPUExecutionProvider"])
    return cands


def provider_name(prov) -> str:
    p = prov[0]
    return p[0] if isinstance(p, tuple) else p


def cv2_cuda_devices() -> int:
    try:
        import cv2
        return int(cv2.cuda.getCudaEnabledDeviceCount())
    except Exception:  # noqa: BLE001 — OpenCV without the cuda module
        return 0


def use_cv2_cuda(net, runtime: str = "auto") -> bool:
    """Move a cv2.dnn network to the GPU when OpenCV was built with CUDA. Returns True if it did."""
    if (runtime or "auto").lower() in CPU_ONLY or cv2_cuda_devices() == 0:
        return False
    import cv2
    try:
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
        return True
    except Exception:  # noqa: BLE001
        return False


def torch_device(pref: Optional[str] = "auto") -> str:
    """'auto' -> 'cuda:0' (NVIDIA / ROCm build), 'mps' (Apple Silicon) or 'cpu'. Any other value is returned as is."""
    p = (pref or "auto").lower()
    if p not in ("auto", "gpu"):
        return pref
    try:
        import torch
    except ImportError:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda:0"
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return "mps"
    return "cpu"


def nvidia_gpus() -> List[str]:
    """Names of the NVIDIA GPUs the driver reports (empty without nvidia-smi)."""
    exe = shutil.which("nvidia-smi")
    if not exe:
        return []
    try:
        out = subprocess.run([exe, "--query-gpu=name", "--format=csv,noheader"], capture_output=True, text=True,
                             timeout=10).stdout
    except Exception:  # noqa: BLE001
        return []
    return [ln.strip() for ln in out.splitlines() if ln.strip()]


# ----------------------------------------------------------------------------- CUDA diagnosis
CUDA_ERRORS = {
    35: "the NVIDIA driver is too old for this CUDA runtime",
    100: "no CUDA-capable GPU visible to this process (CUDA_VISIBLE_DEVICES, driver not loaded, remote session)",
    209: "no kernel image for this GPU: the onnxruntime build does not support this GPU generation",
    222: "the driver is too old for the PTX in this build",
    801: "operation not supported: this CUDA runtime cannot use this GPU (driver too old for CUDA 13, GPU older than "
         "compute capability 7.5 with a CUDA 13 build, a virtual / partial GPU, or a TCC/MCDM driver mode)",
    999: "unknown CUDA error (often after a driver update without a reboot)",
}


def nvidia_info() -> List[Dict[str, object]]:
    """Per NVIDIA GPU: name, driver version, compute capability, memory (MiB). Empty without nvidia-smi."""
    exe = shutil.which("nvidia-smi")
    if not exe:
        return []
    out = []
    for fields in ("name,driver_version,compute_cap,memory.total", "name,driver_version,memory.total"):
        try:
            txt = subprocess.run([exe, f"--query-gpu={fields}", "--format=csv,noheader,nounits"], capture_output=True,
                                 text=True, timeout=10).stdout
        except Exception:  # noqa: BLE001
            return []
        rows = [[c.strip() for c in ln.split(",")] for ln in txt.splitlines() if ln.strip()]
        if rows and all(len(r) == len(fields.split(",")) and "not supported" not in ",".join(r).lower() for r in rows):
            for r in rows:
                d = dict(zip(fields.replace("memory.total", "memory_mib").split(","), r))
                out.append(dict(name=d.get("name"), driver=d.get("driver_version"),
                                compute_cap=_float(d.get("compute_cap")), memory_mib=_float(d.get("memory_mib"))))
            return out
    return out


def _float(v) -> Optional[float]:
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def driver_major(driver: Optional[str]) -> Optional[int]:
    try:
        return int(str(driver).split(".")[0])
    except (TypeError, ValueError):
        return None


def onnxruntime_packages() -> Dict[str, str]:
    """Installed onnxruntime distributions. More than one = files of one build overwrite the other's (broken)."""
    from importlib import metadata
    names = ("onnxruntime", "onnxruntime-gpu", "onnxruntime-directml", "onnxruntime-openvino", "onnxruntime-silicon",
             "onnxruntime-gpu-cu12", "onnxruntime-qnn")
    out = {}
    for n in names:
        try:
            out[n] = metadata.version(n)
        except metadata.PackageNotFoundError:
            pass
    return out


def nvidia_wheels() -> Dict[str, str]:
    """CUDA / cuDNN libraries installed by pip (``onnxruntime-gpu[cuda,cudnn]``, PyTorch)."""
    from importlib import metadata
    out = {}
    for d in metadata.distributions():
        n = (d.metadata.get("Name") or "").lower()
        if n.startswith("nvidia-") and any(k in n for k in ("cuda-runtime", "cudnn", "cublas", "cu12", "cu13")):
            out[n] = d.version
    return dict(sorted(out.items()))


def ort_cuda_build() -> Optional[int]:
    """CUDA major version the installed onnxruntime-gpu was built for (12 / 13), None for a CPU / other build."""
    try:
        from onnxruntime.capi import build_and_package_info as b
    except Exception:  # noqa: BLE001
        return None
    cv = getattr(b, "cuda_version", None)
    if cv:
        try:
            return int(str(cv).split(".")[0])
        except ValueError:
            pass
    pk = onnxruntime_packages()
    v = pk.get("onnxruntime-gpu")
    if not v:
        return None
    wheels = " ".join(nvidia_wheels())
    if "cu13" in wheels or "nvidia-cuda-runtime " in wheels:
        return 13
    if "cu12" in wheels:
        return 12
    try:
        major, minor = (int(x) for x in v.split(".")[:2])
        return 13 if (major, minor) >= (1, 27) else 12      # PyPI default build: CUDA 12 since 1.19, CUDA 13 since 1.27
    except ValueError:
        return None


def recommend_onnxruntime(gpus: Optional[List[Dict[str, object]]] = None, build_cuda: Optional[int] = None,
                          windows: Optional[bool] = None) -> Dict[str, object]:
    """Which onnxruntime package this machine should run, from the driver and the GPU generation.

    Returns ``{"package": pip spec, "cuda": 12 | 13 | None, "reason": text, "mismatch": bool}`` where ``mismatch``
    = the installed build (``build_cuda``) cannot work here."""
    gpus = nvidia_info() if gpus is None else gpus
    windows = (os.name == "nt") if windows is None else windows
    fallback = "onnxruntime-directml" if windows else "onnxruntime"
    if not gpus:
        return dict(package=fallback if windows else "onnxruntime", cuda=None, mismatch=False,
                    reason="no NVIDIA GPU / driver found (nvidia-smi)")
    g = gpus[0]
    drv, cc = driver_major(g.get("driver")), g.get("compute_cap")
    if drv is not None and drv >= 580 and (cc is None or cc >= 7.5):
        rec = dict(package="onnxruntime-gpu[cuda,cudnn]>=1.27", cuda=13,
                   reason=f"driver {g.get('driver')} >= 580 and compute capability {cc} >= 7.5: CUDA 13 build")
    elif drv is not None and drv >= 525 and (cc is None or cc >= 5.0):
        why = [f"driver {g.get('driver')} < 580"] if drv < 580 else []
        if cc is not None and cc < 7.5:
            why.append(f"compute capability {cc} < 7.5 ({g.get('name')} is older than CUDA 13 supports)")
        rec = dict(package="onnxruntime-gpu[cuda,cudnn]>=1.21,<1.27", cuda=12,
                   reason=" and ".join(why) + ": CUDA 12 build (onnxruntime-gpu < 1.27)")
    else:
        rec = dict(package=fallback, cuda=None,
                   reason=f"driver {g.get('driver')} is too old for CUDA 12 (needs >= 525): update the NVIDIA driver "
                          f"(https://www.nvidia.com/drivers), meanwhile {fallback}")
    rec["mismatch"] = bool(build_cuda and rec["cuda"] and build_cuda != rec["cuda"]) or bool(build_cuda and rec["cuda"] is None)
    return rec


_PROBE_SRC = r"""
import json, sys, time
res = {"ok": False}
try:
    import numpy as np, onnxruntime as ort
    if hasattr(ort, "preload_dlls") and "--no-preload" not in sys.argv:
        try:
            ort.preload_dlls()
        except Exception as e:
            res["preload_error"] = str(e)[:300]
    so = ort.SessionOptions(); so.log_severity_level = 3
    t0 = time.time()
    s = ort.InferenceSession(sys.argv[1], so, providers=[("CUDAExecutionProvider", {"device_id": int(sys.argv[2])})])
    res["providers"] = s.get_providers()
    if s.get_providers()[0] != "CUDAExecutionProvider":
        raise RuntimeError("session fell back to " + s.get_providers()[0])
    i = s.get_inputs()[0]
    shape = [d if isinstance(d, int) and d > 0 else 1 for d in i.shape]
    s.run(None, {i.name: np.zeros(shape, np.float32)})
    res.update(ok=True, seconds=round(time.time() - t0, 2))
except Exception as e:
    res["error"] = str(e)
print("@@PROBE@@" + json.dumps(res))
"""


def _first_line(text, n: int = 200) -> str:
    lines = [ln for ln in str(text or "").splitlines() if ln.strip()]
    return lines[0][:n] if lines else ""


def _cuda_code(text: str) -> Optional[int]:
    import re
    m = re.search(r"CUDA failure (\d+)", text or "") or re.search(r"cuda(?:Error)?\D{0,20}(\d{2,3})\b", text or "")
    return int(m.group(1)) if m else None


def probe_cuda(model_path: Optional[str] = None, device_id: int = 0, timeout: float = 180.0,
               preload: bool = True) -> Dict[str, object]:
    """Create a CUDA session and run it once **in a separate process** (a broken CUDA set-up can print pages of
    errors, poison the process or hang). Returns ``{"ok", "error", "code", "meaning", "seconds", "output"}``."""
    import json
    import sys
    model = model_path or str(Path(__file__).resolve().parents[1] / "models" / "yolox.onnx")
    if not Path(model).exists():
        return dict(ok=False, error=f"model not found: {model}", code=None, meaning=None)
    args = [sys.executable, "-c", _PROBE_SRC, model, str(int(device_id))] + ([] if preload else ["--no-preload"])
    try:
        p = subprocess.run(args, capture_output=True, text=True, timeout=timeout)
        out = (p.stdout or "") + (p.stderr or "")
    except subprocess.TimeoutExpired:
        return dict(ok=False, error=f"no answer after {timeout:.0f} s", code=None, meaning="CUDA initialisation hangs")
    res = {"ok": False}
    for ln in out.splitlines():
        if ln.startswith("@@PROBE@@"):
            res = json.loads(ln[len("@@PROBE@@"):])
    if "error" not in res and not res.get("ok"):
        res["error"] = "probe process crashed (exit code %s)" % p.returncode
    noise = "\n".join(ln for ln in out.splitlines() if not ln.startswith("@@PROBE@@"))
    if not res.get("ok"):                     # onnxruntime fell back to the CPU: the real reason is in its log
        why = [ln.strip() for ln in noise.splitlines() if "CUDA failure" in ln or "LoadLibrary failed" in ln
               or "cannot open shared object" in ln]
        if why:
            res["error"] = why[0][:400] + "\n" + str(res.get("error", ""))
    code = _cuda_code(res.get("error", "") + "\n" + noise)
    res.update(code=code, meaning=CUDA_ERRORS.get(code) if code else None, output=noise[-3000:])
    return res


_CUDA_OK: Dict[tuple, Dict[str, object]] = {}


def cuda_usable(model_path: Optional[str] = None, device_id: int = 0) -> Dict[str, object]:
    """:func:`probe_cuda`, cached for the process. Skipped (ok) with ``FVE_SKIP_CUDA_PROBE=1``."""
    if os.environ.get("FVE_SKIP_CUDA_PROBE") == "1":
        return dict(ok=True, skipped=True)
    key = (model_path, int(device_id))
    if key not in _CUDA_OK:
        _CUDA_OK[key] = probe_cuda(model_path, device_id)
    return _CUDA_OK[key]


def cuda_hint(res: Optional[Dict[str, object]] = None) -> str:
    """One line: why CUDA failed and what to install on this machine."""
    rec = recommend_onnxruntime(build_cuda=ort_cuda_build())
    why = ((res or {}).get("meaning") or _first_line((res or {}).get("error"), 160)) if res else "CUDA not usable"
    pk = onnxruntime_packages()
    if len(pk) > 1:
        return (f"{why}; several onnxruntime packages are installed together ({', '.join(pk)}): they overwrite each "
                f"other -> pip uninstall -y {' '.join(pk)} && pip install \"{rec['package']}\"")
    fix = f"pip uninstall -y {' '.join(pk) or 'onnxruntime-gpu'} && pip install \"{rec['package']}\""
    return f"{why}. This machine: {rec['reason']} -> {fix}   (details: python -m football_eval.accel --doctor)"


def doctor(model_path: Optional[str] = None, device_id: int = 0) -> str:
    """Everything that decides whether the detector runs on the NVIDIA GPU, the probe result and the fix."""
    import platform
    import sys
    L = []
    add = L.append
    add(f"python        : {sys.version.split()[0]} ({sys.executable})")
    add(f"system        : {platform.platform()}")
    gpus = nvidia_info()
    for g in gpus or [{}]:
        add(f"NVIDIA GPU    : {g.get('name', 'none (nvidia-smi not found)')} | driver {g.get('driver')} | compute "
            f"capability {g.get('compute_cap')} | {g.get('memory_mib')} MiB")
    pk = onnxruntime_packages()
    add(f"onnxruntime   : {', '.join(f'{k} {v}' for k, v in pk.items()) or 'not installed'}"
        + ("   <-- MORE THAN ONE: they overwrite each other's files" if len(pk) > 1 else ""))
    bc = ort_cuda_build()
    add(f"built for CUDA: {bc or '-'}")
    try:
        import onnxruntime as ort
        add(f"providers     : {', '.join(ort.get_available_providers())}")
    except Exception as e:  # noqa: BLE001
        add(f"providers     : import failed: {e}")
    nw = nvidia_wheels()
    add(f"CUDA wheels   : {', '.join(f'{k} {v}' for k, v in nw.items()) or 'none (system CUDA / PATH is used)'}")
    for var in ("CUDA_PATH", "CUDA_VISIBLE_DEVICES", "CUDA_DEVICE_ORDER"):
        if os.environ.get(var) is not None:
            add(f"{var:<14}: {os.environ[var]}")
    try:
        import torch
        add(f"PyTorch       : {torch.__version__} (CUDA {torch.version.cuda}) cuda available: {torch.cuda.is_available()}")
    except ImportError:
        add("PyTorch       : not installed")
    rec = recommend_onnxruntime(gpus, bc)
    add(f"recommended   : {rec['package']}  ({rec['reason']})")
    if "CUDAExecutionProvider" in ort_available():
        add("")
        add("probe: CUDA session in a separate process ...")
        r = probe_cuda(model_path, device_id)
        if r.get("ok"):
            add(f"  OK - CUDA works ({r.get('seconds')} s to create + run). The detector will use it.")
        else:
            add(f"  FAILED: {_first_line(r.get('error'), 300)}")
            if r.get("code"):
                add(f"  CUDA error {r['code']}: {r.get('meaning') or 'see the CUDA runtime API error codes'}")
            r2 = probe_cuda(model_path, device_id, preload=False)
            add(f"  without the pip CUDA libraries (system CUDA on PATH): {'OK' if r2.get('ok') else 'fails too'}")
            add("")
            add("fix:")
            if len(pk) > 1:
                add(f"  1. pip uninstall -y {' '.join(pk)}")
                add(f"  2. pip install \"{rec['package']}\"")
            elif r2.get("ok"):
                add("  the system CUDA works but the pip one does not: pip uninstall the nvidia-* CUDA wheels above, "
                    "or set FVE_NO_PRELOAD=1")
            else:
                add(f"  pip uninstall -y {' '.join(pk) or 'onnxruntime-gpu'} && pip install \"{rec['package']}\"")
                if rec["cuda"] == 13 and r.get("code") in (801, 35, 999):
                    add("  (driver and GPU look right for CUDA 13: reboot after a driver update; check that the GPU is "
                        "not in TCC / MCDM mode and that the session is not a remote desktop on a virtual GPU)")
            add("  no CUDA at all (any DirectX 12 GPU, Windows): pip uninstall -y onnxruntime-gpu && pip install "
                "onnxruntime-directml   (the detector picks DirectML automatically)")
            add("  start.bat does this choice automatically on the next start.")
    elif gpus:
        add("")
        add(f"NVIDIA GPU but no CUDA provider: pip uninstall -y {' '.join(pk) or 'onnxruntime'} && "
            f"pip install \"{rec['package']}\"")
    return "\n".join(L)


def summary() -> Dict[str, object]:
    avail = ort_available()
    gpu_ort = [p for p in avail if p in GPU_PROVIDERS or p == "TensorrtExecutionProvider"]
    s = {"nvidia_gpus": nvidia_gpus(), "onnxruntime_providers": avail, "onnxruntime_gpu": gpu_ort,
         "cv2_cuda_devices": cv2_cuda_devices(), "torch_device": torch_device("auto")}
    hint = None
    if s["nvidia_gpus"] and "CUDAExecutionProvider" not in avail:
        hint = ("NVIDIA GPU found but onnxruntime has no CUDA provider: pip uninstall -y onnxruntime && "
                "pip install 'onnxruntime-gpu[cuda,cudnn]'  (start.command / start.bat do this automatically)")
    s["hint"] = hint
    return s


def describe() -> str:
    s = summary()
    gpu = ", ".join(s["nvidia_gpus"]) or "none"
    lines = [f"NVIDIA GPU         : {gpu}",
             f"onnxruntime GPU    : {', '.join(s['onnxruntime_gpu']) or 'none'}  (all: {', '.join(s['onnxruntime_providers']) or 'onnxruntime not installed'})",
             f"OpenCV CUDA devices: {s['cv2_cuda_devices']}",
             f"PyTorch device     : {s['torch_device']}"]
    if s["hint"]:
        lines.append("hint: " + s["hint"])
    return "\n".join(lines)


def _main(argv=None) -> int:
    import argparse
    import json
    ap = argparse.ArgumentParser(description="Which device the models run on (football_eval/accel.py)")
    ap.add_argument("--doctor", action="store_true", help="full CUDA diagnosis + the fix for this machine")
    ap.add_argument("--probe-cuda", action="store_true",
                    help="exit code 0 if a CUDA session works, 1 if not, 2 if it cannot be tested (used by start.bat)")
    ap.add_argument("--recommend", action="store_true", help="print the onnxruntime pip spec for this machine")
    ap.add_argument("--gpu-id", type=int, default=0)
    a = ap.parse_args(argv)
    if a.probe_cuda:
        if "CUDAExecutionProvider" not in ort_available():
            print("CUDA FAILED: onnxruntime has no CUDA provider (CPU / DirectML build installed)")
            return 1
        r = probe_cuda(device_id=a.gpu_id)
        if not r.get("ok") and str(r.get("error", "")).startswith("model not found"):
            print("CUDA UNKNOWN: " + r["error"] + " (python scripts/get_models.py)")
            return 2
        print("CUDA OK" if r.get("ok") else f"CUDA FAILED: {r.get('meaning') or _first_line(r.get('error'))}")
        return 0 if r.get("ok") else 1
    if a.recommend:
        r = recommend_onnxruntime(build_cuda=ort_cuda_build())
        print(r["package"])
        print(json.dumps(r), file=__import__("sys").stderr)
        return 0
    print(doctor(device_id=a.gpu_id) if a.doctor else describe())
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
