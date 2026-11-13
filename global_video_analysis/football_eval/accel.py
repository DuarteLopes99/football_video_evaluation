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

    python -m football_eval.accel
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
    if _PRELOADED or not hasattr(ort, "preload_dlls"):
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
                        trt_cache: Optional[str] = None) -> List[list]:
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
            cands.append([(p, {"trt_engine_cache_enable": True, "trt_engine_cache_path": cache, "trt_fp16_enable": True}),
                          "CUDAExecutionProvider", "CPUExecutionProvider"])
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


if __name__ == "__main__":
    print(describe())
