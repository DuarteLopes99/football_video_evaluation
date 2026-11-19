"""The two ONNX models (not in git: too big), fetched when missing. Same rules as
global_video_analysis/football_eval/models.py: copied from the sibling project when it has them, else downloaded
from the OpenCV Model Zoo into a temporary file, size-checked; a file < 1 MB (git-LFS pointer, interrupted
download) is replaced; a valid file you put there yourself is never overwritten.

    python scripts/download_models.py
"""
import hashlib
import os
import shutil
import time
import urllib.request

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))            # offside_code/
ZOO = 'https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/'
MODELS = {
    'yolox.onnx': dict(url=ZOO + 'object_detection_yolox/object_detection_yolox_2022nov.onnx', size=35_858_002,
                       sha256='c5c2d13e59ae883e6af3b45daea64af4833a4951c92d116ec270d9ddbe998063',
                       sibling=os.path.join(HERE, '..', 'global_video_analysis', 'models', 'yolox.onnx')),
    'mp_pose.onnx': dict(url=ZOO + 'pose_estimation_mediapipe/pose_estimation_mediapipe_2023mar.onnx',
                         size=5_557_238,
                         sha256='9d89c599319a18fb7d2e28451a883476164543182bafca5f09eb2cf767ed2f3f'),
}
MIN_BYTES = 1_000_000


def usable(p):
    return os.path.isfile(p) and os.path.getsize(p) >= MIN_BYTES


def _download(url, dst, log, tries=3):
    tmp = dst + '.part'
    last = None
    for attempt in range(1, tries + 1):
        try:
            with urllib.request.urlopen(url, timeout=60) as r, open(tmp, 'wb') as f:
                shutil.copyfileobj(r, f, 1 << 20)
            if os.path.getsize(tmp) < MIN_BYTES:
                raise IOError('only %d bytes received' % os.path.getsize(tmp))
            os.replace(tmp, dst)
            return
        except Exception as e:  # noqa: BLE001 — retry, then report
            last = e
            if os.path.exists(tmp):
                os.remove(tmp)
            if attempt < tries:
                time.sleep(2 * attempt)
    raise FileNotFoundError(
        'download of %s failed (%s).\nThe model is not in git (too big). With internet, run '
        '"python scripts/download_models.py" inside offside_code/, or download\n  %s\nand save it as %s'
        % (os.path.basename(dst), last, url, dst))


def ensure_models(models_dir=None, log=print):
    """Make ``models_dir`` hold usable yolox.onnx and mp_pose.onnx; returns the directory."""
    d = os.path.abspath(models_dir or os.path.join(HERE, 'models'))
    os.makedirs(d, exist_ok=True)
    for name, spec in MODELS.items():
        dst = os.path.join(d, name)
        if usable(dst):
            continue
        sib = spec.get('sibling')
        if sib and usable(sib):
            shutil.copy2(sib, dst)
            log('[models] %s: copied from %s' % (name, os.path.normpath(sib)))
            continue
        log('[models] %s missing: downloading %d MB from the OpenCV Model Zoo ...' % (name, spec['size'] // 1_000_000))
        _download(spec['url'], dst, log)
        h = hashlib.sha256(open(dst, 'rb').read()).hexdigest()
        if h != spec['sha256']:
            log('[models] warning: %s checksum differs from the tested file (kept)' % name)
        log('[models] %s: %d MB' % (name, os.path.getsize(dst) // 1_000_000))
    return d
