"""Download the two ONNX models from the OpenCV Model Zoo (GitHub LFS) into ./models."""
import os
import urllib.request

BASE = 'https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/'
MODELS = {
    'yolox.onnx': BASE + 'object_detection_yolox/object_detection_yolox_2022nov.onnx',
    'mp_pose.onnx': BASE + 'pose_estimation_mediapipe/pose_estimation_mediapipe_2023mar.onnx',
}
root = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'models')
os.makedirs(root, exist_ok=True)
for name, url in MODELS.items():
    dst = os.path.join(root, name)
    if os.path.exists(dst) and os.path.getsize(dst) > 1_000_000:
        print('ok      ', name)
        continue
    print('download', name, '<-', url)
    urllib.request.urlretrieve(url, dst)
    print('        ', os.path.getsize(dst) // 1_000_000, 'MB')
