"""Automatic GPU selection (football_eval/accel.py). Runs without a GPU: the provider lists are faked.

    python -m pytest tests/test_accel.py -q
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from football_eval import accel  # noqa: E402
from football_eval.config import load_config  # noqa: E402

NVIDIA = ["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"]
MAC = ["CoreMLExecutionProvider", "CPUExecutionProvider"]
WINDOWS_DML = ["DmlExecutionProvider", "CPUExecutionProvider"]


def names(cands):
    return [accel.provider_name(c) for c in cands]


def test_auto_tries_gpu_first_and_cpu_last():
    assert names(accel.provider_candidates("auto", NVIDIA)) == ["CUDAExecutionProvider", "CPUExecutionProvider"]
    assert names(accel.provider_candidates("auto", MAC)) == ["CoreMLExecutionProvider", "CPUExecutionProvider"]
    assert names(accel.provider_candidates("auto", WINDOWS_DML)) == ["DmlExecutionProvider", "CPUExecutionProvider"]
    assert names(accel.provider_candidates("auto", ["CPUExecutionProvider"])) == ["CPUExecutionProvider"]
    # legacy config values mean "auto"
    assert names(accel.provider_candidates("onnxruntime", NVIDIA)) == names(accel.provider_candidates("auto", NVIDIA))


def test_tensorrt_only_on_request(tmp_path):
    assert "TensorrtExecutionProvider" not in names(accel.provider_candidates("auto", NVIDIA))
    c = accel.provider_candidates("tensorrt", NVIDIA, trt_cache=str(tmp_path / "trt"))
    assert names(c)[:2] == ["TensorrtExecutionProvider", "CUDAExecutionProvider"]
    assert c[0][0][1]["trt_engine_cache_path"] == str(tmp_path / "trt") and (tmp_path / "trt").is_dir()


def test_cpu_forced_and_missing_provider():
    assert accel.provider_candidates("cpu", NVIDIA) == [["CPUExecutionProvider"]]
    assert accel.provider_candidates("cuda", MAC) == [["CPUExecutionProvider"]]   # asked for CUDA on a Mac


def test_torch_device_and_defaults():
    assert accel.torch_device("cpu") == "cpu" and accel.torch_device("cuda:1") == "cuda:1"
    assert accel.torch_device("auto") in ("cpu", "mps", "cuda:0")
    cfg = load_config(Path(__file__).resolve().parents[1] / "configs" / "default.yaml")
    assert cfg.detection.runtime == "auto" and cfg.detection.device == "auto"
    s = accel.summary()
    assert {"nvidia_gpus", "onnxruntime_providers", "cv2_cuda_devices", "torch_device"} <= set(s)
