"""Fetch the default detector (YOLOX-S COCO, ONNX, ~36 MB) into ./models.

The model is not in git (too big). setup_env.py runs this at the first start, and every analysis checks it again
(football_eval/models.py): copied from ../offside_code/models/yolox.onnx when present, else downloaded from the
OpenCV Model Zoo (GitHub LFS), into a temporary file first, size-checked and compared with the tested checksum.

Better detectors (recommended once you start fine-tuning, see docs/SOTA_REVIEW.md):
    pip install ultralytics
    # then in the config: detection.backend: ultralytics, detection.model_path: yolo11m.pt (COCO)
    # or a football fine-tuned checkpoint with classes {ball, goalkeeper, player, referee}
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from football_eval.models import MODELS, ModelMissing, ensure  # noqa: E402

if __name__ == "__main__":
    try:
        for name in MODELS:
            print("ok", ensure(name))
    except ModelMissing as e:
        sys.exit(str(e))
