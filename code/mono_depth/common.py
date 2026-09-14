"""step 스크립트들이 같이 쓰는 함수 모음."""
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

HERE = Path(__file__).parent

# 카메라 좌표계는 Y가 아래, Z가 앞이라 Open3D 창에 그대로 그리면 위아래가 뒤집혀 보인다.
# 점군에 이 행렬을 곱하면 화면 방향과 맞는다.
FLIP = [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]

RELATIVE_MODEL = "depth-anything/Depth-Anything-V2-Small-hf"                 # 상대 깊이 (값이 클수록 가까움)
METRIC_MODEL = "depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf"      # 실내용 절대 깊이 (미터)


def load_camera():
    d = np.load(HERE / "calibration/camera_params.npz")
    return d["K"], d["dist"]


class ImageCap:
    """--image 옵션: 사진 한 장을 웹캠처럼 계속 돌려준다 (카메라 없을 때 연습용)."""
    def __init__(self, path):
        self.img = cv2.imread(str(path))
        assert self.img is not None, f"사진을 못 읽음: {path}"

    def read(self):
        return True, self.img.copy()

    def release(self):
        pass


def open_source(argv):
    if "--image" in argv:
        return ImageCap(argv[argv.index("--image") + 1])
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)    # K는 640x480에서 구했으므로 같은 크기로 맞춘다
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    ok, frame = cap.read()
    assert ok and frame.shape[:2] == (480, 640), "웹캠이 640x480이 아니면 K가 맞지 않는다"
    return cap


def load_depth_model(metric=True):
    import torch
    from transformers import pipeline   # 무거운 라이브러리라 필요할 때만 불러온다
    device = "cuda" if torch.cuda.is_available() else "cpu"   # GPU 없으면 느리지만 동작은 한다
    return pipeline("depth-estimation", model=METRIC_MODEL if metric else RELATIVE_MODEL, device=device)


def predict_depth(pipe, bgr):
    """BGR 사진 → (H, W) float32 깊이 배열. metric 모델이면 단위는 미터."""
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    depth = pipe(Image.fromarray(rgb))["predicted_depth"].cpu().numpy().astype(np.float32)
    if depth.shape != bgr.shape[:2]:   # 라이브러리 버전에 따라 모델 입력 크기로 나올 때가 있다
        depth = cv2.resize(depth, (bgr.shape[1], bgr.shape[0]))
    return depth


def depth_to_points(depth, K, rgb, step=1, zmin=0.1, zmax=10.0):
    """깊이 + K → (N, 3) 점 좌표와 (N, 3) 색(0~1). step=2면 2픽셀마다 하나만 쓴다."""
    depth, rgb = depth[::step, ::step], rgb[::step, ::step]
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    v, u = np.indices(depth.shape) * step   # 건너뛴 만큼 원래 픽셀 좌표로 되돌린다
    valid = np.isfinite(depth) & (depth > zmin) & (depth < zmax)
    z = depth[valid]
    x = (u[valid] - cx) * z / fx
    y = (v[valid] - cy) * z / fy
    return np.column_stack((x, y, z)), rgb[valid] / 255.0


if __name__ == "__main__":   # 자체 확인: python3 common.py
    K = np.array([[100.0, 0, 1], [0, 100.0, 1], [0, 0, 1]])
    d = np.array([[0.5, 0.5, 0.0], [0.5, 0.5, 0.5], [np.nan, 0.5, 0.5]])
    pts, col = depth_to_points(d, K, np.full((3, 3, 3), 255))
    assert pts.shape == (7, 3) and np.allclose(pts[0], [-0.005, -0.005, 0.5]) and np.allclose(col, 1.0)
    pts2, _ = depth_to_points(np.full((4, 4), 1.0), K, np.zeros((4, 4, 3)), step=2)
    assert len(pts2) == 4 and np.allclose(pts2[-1], [0.01, 0.01, 1.0])   # 픽셀 (2,2) → x=(2-1)*1/100
    print("common.py OK")
