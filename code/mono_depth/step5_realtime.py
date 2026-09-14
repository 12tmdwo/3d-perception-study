"""Step 5: 실시간. RGB 창 + 깊이 창 + 3D 점군 창. ESC = 종료.
카메라 없으면: --image code/mono_depth/test.jpg   자동 종료: --frames 60"""
import sys
import time

import cv2
import numpy as np
import open3d as o3d

from common import FLIP, load_camera, open_source, load_depth_model, predict_depth, depth_to_points

K, dist = load_camera()
pipe = load_depth_model(metric=True)
cap = open_source(sys.argv)
frames = int(sys.argv[sys.argv.index("--frames") + 1]) if "--frames" in sys.argv else 0

vis = o3d.visualization.Visualizer()  
assert vis.create_window("Point Cloud", 640, 480), "3D 창을 못 열었다 → 그래픽 드라이버 확인 (터미널에서 glxinfo -B)"
pcd = o3d.geometry.PointCloud()
n, t0 = 0, time.time()
while True:
    ok, frame = cap.read()
    if not ok:
        break
    # frame = cv2.undistort(frame, K, dist)
    depth = predict_depth(pipe, frame)
    points, colors = depth_to_points(depth, K, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), step=2)   # 2픽셀 간격 → 점 1/4
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    pcd.transform(FLIP)
    if n == 0:
        vis.add_geometry(pcd)      # 첫 프레임만 등록
    else:
        vis.update_geometry(pcd)   # 이후엔 내용만 갱신
    vis.poll_events()
    vis.update_renderer()

    depth_vis = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    cv2.imshow("RGB", frame)
    cv2.imshow("Depth", cv2.applyColorMap(depth_vis, cv2.COLORMAP_INFERNO))
    n += 1
    if n % 10 == 0:
        print(f"FPS: {n / (time.time() - t0):.1f}  points: {len(points)}")
    if cv2.waitKey(1) == 27 or n == frames:
        break
cap.release()
cv2.destroyAllWindows()
vis.destroy_window()
