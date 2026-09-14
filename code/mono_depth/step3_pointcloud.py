"""Step 3: test.jpg → 절대 깊이 → 3D 점군(cloud.ply) → Open3D 창. (--no-show: 창 없이 저장만)"""
import sys

import cv2
import open3d as o3d

from common import HERE, FLIP, load_camera, load_depth_model, predict_depth, depth_to_points

K, dist = load_camera()
bgr = cv2.imread(str(HERE / "test.jpg"))
bgr = cv2.undistort(bgr, K, dist)   # ponytail: newCameraMatrix 생략 → K 그대로 사용, 모서리 검은 띠는 깊이 마스크가 거른다
depth = predict_depth(load_depth_model(metric=True), bgr)
points, colors = depth_to_points(depth, K, cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
print("점 개수:", len(points), "/ 전체 픽셀:", depth.size)
print("Z 범위(m): %.2f ~ %.2f" % (points[:, 2].min(), points[:, 2].max()))

pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
pcd.colors = o3d.utility.Vector3dVector(colors)
assert o3d.io.write_point_cloud(str(HERE / "cloud.ply"), pcd)
print("saved cloud.ply")
if "--no-show" not in sys.argv:
    pcd.transform(FLIP)   # 카메라 좌표 그대로 그리면 위아래가 뒤집혀 보인다
    o3d.visualization.draw_geometries([pcd])
