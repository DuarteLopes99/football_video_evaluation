"""Download the two ONNX models (YOLOX, MediaPipe pose) from the OpenCV Model Zoo into ./models when missing.
setup_env.py runs this at the first start; PlayerDetector checks again before loading them (offside/models.py)."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from offside.models import MODELS, ensure_models  # noqa: E402

if __name__ == '__main__':
    try:
        d = ensure_models()
    except FileNotFoundError as e:
        sys.exit(str(e))
    for name in MODELS:
        print('ok      ', os.path.join(d, name))
