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


# --------------------------------------------------------------------- CUDA listed but not usable (error 801)
USER_LOG = (r"EP Error N:\work\1\s\onnxruntime\core\providers\cuda\cuda_call.cc:154 onnxruntime::CudaCall "
            r"CUDA failure 801: (null) ; GPU=-1 ; hostname=PC ; file=cuda_execution_provider.cc ; "
            r"line=345 ; expr=cudaSetDevice(info.device_id);")


def test_cuda_error_code_is_parsed():
    assert accel._cuda_code(USER_LOG) == 801
    assert "not supported" in accel.CUDA_ERRORS[801]
    assert accel._cuda_code("all good") is None


def test_recommendation_follows_driver_and_gpu_generation():
    rec = lambda name, drv, cc, build=13: accel.recommend_onnxruntime([dict(name=name, driver=drv, compute_cap=cc)],
                                                                    build, windows=True)
    new = rec("RTX 4070", "591.44", 8.9)
    assert new["cuda"] == 13 and new["package"].endswith(">=1.27") and not new["mismatch"]
    old_driver = rec("RTX 3060", "572.16", 8.6)               # CUDA 13 build needs a 580+ driver
    assert old_driver["cuda"] == 12 and "<1.27" in old_driver["package"] and old_driver["mismatch"]
    pascal = rec("GTX 1060", "581.08", 6.1)                   # CUDA 13 dropped compute capability < 7.5
    assert pascal["cuda"] == 12 and "7.5" in pascal["reason"]
    ancient = rec("GTX 750", "472.12", 5.0)
    assert ancient["package"] == "onnxruntime-directml" and ancient["cuda"] is None
    assert accel.recommend_onnxruntime([], None, windows=False)["package"] == "onnxruntime"


def test_cuda_candidates_carry_the_gpu_id():
    c = accel.provider_candidates("auto", NVIDIA, gpu_id=1)
    assert c[0][0] == ("CUDAExecutionProvider", {"device_id": 1})


def test_detector_skips_a_broken_cuda_without_creating_a_session(monkeypatch, capsys):
    """CUDA listed + probe fails -> one clear line, CPU (or the next GPU provider) used, no EP Error page."""
    pytest = __import__("pytest")
    ort = pytest.importorskip("onnxruntime")
    model = Path(__file__).resolve().parents[1] / "models" / "yolox.onnx"
    if not model.exists():
        pytest.skip("models/yolox.onnx not downloaded")
    from football_eval import detection
    monkeypatch.setattr(accel, "ort_available", lambda: ["CUDAExecutionProvider", "CPUExecutionProvider"])
    monkeypatch.setattr(accel, "probe_cuda", lambda *a, **k: dict(ok=False, error=USER_LOG, code=801,
                                                                   meaning=accel.CUDA_ERRORS[801]))
    accel._CUDA_OK.clear()
    created = []
    real = ort.InferenceSession

    def spy(path, so=None, providers=None, **kw):
        created.append(accel.provider_name(providers))
        return real(path, so, providers=["CPUExecutionProvider"], **kw)
    monkeypatch.setattr(ort, "InferenceSession", spy)
    r = detection._OnnxRunner(str(model), 640, "auto")
    out = capsys.readouterr().out
    assert "CUDAExecutionProvider skipped" in out and "--doctor" in out
    assert created == ["CPUExecutionProvider"] and r.name.endswith("CPUExecutionProvider")
    accel._CUDA_OK.clear()


def test_probe_cli_without_cuda_provider_says_so(capsys):
    if "CUDAExecutionProvider" in accel.ort_available():
        return
    assert accel._main(["--probe-cuda"]) == 1
    assert "no CUDA provider" in capsys.readouterr().out
