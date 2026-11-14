# GPU troubleshooting: "CUDA exists but is not used"

## The symptom (Windows, 2026-11-13)

```
***** EP Error *****
EP Error ... cuda_call.cc:154 onnxruntime::CudaCall CUDA failure 801: (null) ; GPU=-1 ; ...
expr=cudaSetDevice(info.device_id);
 when using ['CUDAExecutionProvider', 'CPUExecutionProvider']
Falling back to ['CPUExecutionProvider'] and retrying.
[detector] CUDAExecutionProvider unavailable: not usable here (session runs on CPUExecutionProvider)
[detector] CPUExecutionProvider: 99.6 ms / tile
```

`nvidia-smi` works and onnxruntime **lists** `CUDAExecutionProvider`, but the first CUDA call of a session
(`cudaSetDevice`) fails with error **801 `cudaErrorNotSupported`** and onnxruntime silently runs on the CPU.

## Why

*Listed* only means the `onnxruntime-gpu` package is installed. Whether CUDA can run depends on three things that
must match:

| onnxruntime-gpu (PyPI) | built for | needs NVIDIA driver | needs GPU (compute capability) |
|---|---|---|---|
| 1.19 – 1.26 | CUDA 12 | ≥ 525 | ≥ 5.0 (GTX 900 and newer) |
| ≥ 1.27 (current default) | CUDA 13 | ≥ 580 | ≥ 7.5 (GTX 16xx / RTX 20xx and newer) |

`start.bat` used to install `onnxruntime-gpu[cuda,cudnn]>=1.21`, which today resolves to a **CUDA 13** build. On a
machine with a driver older than 580, or a GTX 10xx / 900 card, CUDA 13 cannot use the GPU → 801 (or 35 / 100), and
the old check (`'CUDAExecutionProvider' in get_available_providers()`) still passed, so nothing was repaired.
Other causes of the same error: `onnxruntime` and `onnxruntime-gpu` installed together (they overwrite each other's
files — the requirements update used to do that), a driver update without a reboot, a GPU in TCC / MCDM mode, a
remote desktop on a virtual / partial GPU.

## What changed (branch `fix/cuda-provider-fallback`)

* `football_eval/accel.py`
  * `probe_cuda()` creates a CUDA session and runs it once **in a separate process** (a broken CUDA set-up can hang
    or print pages of errors); `cuda_usable()` caches it per run.
  * `recommend_onnxruntime()` reads driver + compute capability from `nvidia-smi` and returns the package that can
    work here (CUDA 13 build, CUDA 12 build `>=1.21,<1.27`, or DirectML / CPU).
  * `doctor()` / `python -m football_eval.accel --doctor`: GPU, driver, compute capability, installed onnxruntime
    packages (warns when more than one), CUDA build, pip CUDA wheels, PyTorch, the probe (with and without the pip
    CUDA libraries) and the exact commands to fix this machine.
  * CLI: `--probe-cuda` (exit 0 ok / 1 fails / 2 cannot test), `--recommend` (pip spec), `--gpu-id N`.
  * CUDA / TensorRT providers get an explicit `device_id` (`detection.gpu_id`, default 0).
* `football_eval/detection.py`: before trying CUDA / TensorRT the detector runs the probe; when it fails it prints
  **one line** with the reason and the fix and moves on to the next GPU provider (DirectML, ...) or the CPU. No more
  EP Error page, no CUDA session created in the main process.
* `start.bat` / `start.command`: a real CUDA session decides, not the provider list. If it fails, the build that
  matches the driver / GPU is installed (after uninstalling every onnxruntime package); if CUDA still fails on
  Windows, `onnxruntime-directml` is installed instead (uses the NVIDIA GPU through DirectX 12, no CUDA needed).
  Tried once (`.venv\.gpu_setup_tried`; delete it to retry); reset after a requirements update.

## Fix it now (on the Windows machine)

```bat
cd <your clone>\global_video_analysis
..\.venv\Scripts\python -m football_eval.accel --doctor
```

Then run the two commands it prints under **fix:** — typically:

```bat
..\.venv\Scripts\python -m pip uninstall -y onnxruntime onnxruntime-gpu onnxruntime-directml
..\.venv\Scripts\python -m pip install "onnxruntime-gpu[cuda,cudnn]>=1.21,<1.27"
..\.venv\Scripts\python -m football_eval.accel --probe-cuda
```

(or update the NVIDIA driver to ≥ 580 and keep the CUDA 13 build; or `pip install onnxruntime-directml`). Or just
start `start.bat`: it does the same automatically. The run log then shows
`[detector] CUDAExecutionProvider: x ms / tile` and `YOLOX via onnxruntime/CUDAExecutionProvider`.
