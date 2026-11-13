"""Package normalisation: setup_env.py (interpreter choice, foreign venvs, stamp) and football_eval.envcheck.

    python -m pytest tests/test_environment.py -q
"""
import importlib.util
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))

from football_eval import envcheck  # noqa: E402

TOP = HERE.parent
spec = importlib.util.spec_from_file_location("setup_env", TOP / "setup_env.py")
setup_env = importlib.util.module_from_spec(spec)
spec.loader.exec_module(setup_env)


def test_supported_range_consistent():
    assert setup_env.SUPPORTED == envcheck.SUPPORTED_PYTHON == ((3, 12), (3, 14))
    assert setup_env.PREFERRED[0] == envcheck.RECOMMENDED_PYTHON
    assert not setup_env.supported((3, 15)) and not setup_env.supported((3, 11)) and setup_env.supported((3, 13))
    assert (TOP / ".python-version").read_text().strip() == "3.13"


def test_pick_prefers_313_and_skips_315_and_32bit():
    found = [((3, 15), 64, "py315"), ((3, 12), 64, "py312"), ((3, 13), 32, "py313-32"), ((3, 14), 64, "py314")]
    assert setup_env.pick(found)[2] == "py312"            # 3.13 only in 32-bit -> next preferred, 3.12
    assert setup_env.pick(found + [((3, 13), 64, "py313")])[2] == "py313"
    assert setup_env.pick([((3, 15), 64, "py315"), ((3, 9), 64, "py39")]) is None   # the PC / old Mac case


def test_foreign_venv(tmp_path):
    d = tmp_path / ".venv"
    d.mkdir()
    (d / "pyvenv.cfg").write_text("home = C:\\Users\\x\\AppData\\Local\\Python\\pythoncore-3.13-64\nversion = 3.13.9\n")
    assert setup_env.foreign_venv(d) == (not setup_env.IS_WIN)
    (d / "pyvenv.cfg").write_text("home = /opt/homebrew/opt/python@3.13/bin\nversion = 3.13.9\n")
    assert setup_env.foreign_venv(d) == setup_env.IS_WIN
    assert not setup_env.foreign_venv(tmp_path / "missing")


def test_stamp_changes_with_python_and_runtime():
    a = setup_env.stamp((3, 13), "cpu")
    assert a == setup_env.stamp((3, 13), "cpu")
    assert a != setup_env.stamp((3, 12), "cpu") and a != setup_env.stamp((3, 13), "gpu")


def test_constraints_pins_and_requirements():
    pins = envcheck.read_constraints()
    for name in ("numpy", "scipy", "pandas", "opencv-python", "scikit-learn", "matplotlib", "pyyaml",
                 "onnxruntime", "onnxruntime-gpu"):
        assert name in pins, name
    assert pins["onnxruntime"] == pins["onnxruntime-gpu"]            # GPU and CPU builds of the same release
    req = (TOP / "requirements.txt").read_text()
    assert "-c constraints.txt" in req and "global_video_analysis/requirements.txt" in req
    # every pinned core package is required by one of the projects (no stale pins)
    projects = ((HERE / "requirements.txt").read_text() + (TOP / "offside_code" / "requirements.txt").read_text()).lower()
    for name in ("numpy", "scipy", "pandas", "opencv-python", "scikit-learn", "matplotlib", "pyyaml"):
        assert name in projects, name


def test_compare_reports_mismatch_missing_and_both_runtimes():
    pins = {"numpy": "2.5.3", "pandas": "3.0.5", "onnxruntime": "1.29.0", "onnxruntime-gpu": "1.29.0", "jupyter": "1.1.1"}
    ok = {"numpy": "2.5.3", "pandas": "3.0.5", "onnxruntime": "1.29.0", "onnxruntime-gpu": None, "jupyter": None}
    assert envcheck.compare(pins, ok) == []
    bad = dict(ok, numpy="1.26.4", pandas=None, **{"onnxruntime-gpu": "1.29.0"})
    probs = envcheck.compare(pins, bad)
    assert any("numpy 1.26.4" in p for p in probs) and any("pandas is not installed" in p for p in probs)
    assert any("both onnxruntime and onnxruntime-gpu" in p for p in probs)
    assert envcheck.canonical("PyYAML") == "pyyaml" and envcheck.canonical("scikit_learn") == "scikit-learn"
