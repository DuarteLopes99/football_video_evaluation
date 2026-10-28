"""Fetch the default detector (YOLOX-S COCO, ONNX, ~36 MB) into ./models.

1. Re-uses the copy from the sibling project ../offiside_code/models/yolox.onnx when present.
2. Otherwise downloads it from the OpenCV Model Zoo (GitHub LFS).

Better detectors (recommended once you start fine-tuning, see docs/SOTA_REVIEW.md):
    pip install ultralytics
    # then in the config: detection.backend: ultralytics, detection.model_path: yolo11m.pt (COCO)
    # or a football fine-tuned checkpoint with classes {ball, goalkeeper, player, referee}
"""
import shutil
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DST = ROOT / "models" / "yolox.onnx"
SIBLING = ROOT.parent / "offiside_code" / "models" / "yolox.onnx"
URL = ("https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/"
       "object_detection_yolox/object_detection_yolox_2022nov.onnx")

DST.parent.mkdir(exist_ok=True)
if DST.exists() and DST.stat().st_size > 1_000_000:
    print("ok", DST)
elif SIBLING.exists():
    shutil.copy(SIBLING, DST)
    print("copied", SIBLING, "->", DST)
else:
    print("download", URL)
    urllib.request.urlretrieve(URL, DST)
    print(DST, DST.stat().st_size // 1_000_000, "MB")
