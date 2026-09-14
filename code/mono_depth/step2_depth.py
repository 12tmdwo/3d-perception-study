"""Step 2: test.jpg → 깊이 배열. 상대 모델과 절대 모델을 둘 다 돌려서 비교한다."""
import cv2
import numpy as np

from common import HERE, load_depth_model, predict_depth

bgr = cv2.imread(str(HERE / "test.jpg"))
h, w = bgr.shape[:2]
for name, metric in [("relative", False), ("metric", True)]:
    depth = predict_depth(load_depth_model(metric), bgr)
    print(f"[{name}] shape={depth.shape} dtype={depth.dtype} "
          f"min={depth.min():.3f} max={depth.max():.3f} center={depth[h // 2, w // 2]:.3f}")
    vis = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)   # 보기용. 계산에는 depth를 쓴다
    cv2.imwrite(str(HERE / f"depth_{name}.png"), cv2.applyColorMap(vis, cv2.COLORMAP_INFERNO))
print("relative: 값이 클수록 가까움(역깊이) / metric: 값 = 미터, 작을수록 가까움")
