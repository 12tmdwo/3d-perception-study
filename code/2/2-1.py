import numpy as np
import open3d as o3d

depth = np.array([[0.5, 0.5, 0.0],
                  [0.5, 0.5, 0.5],
                  [np.nan, 0.5, 0.5]], dtype=float)
fx = fy = 100.0
cx = cy = 1.0
v, u = np.indices(depth.shape)
print(v)
print(u)
valid = np.isfinite(depth) & (depth > 0) & (depth < 2.0)
print(valid)
z = depth[valid]
print(z)
x = (u[valid] - cx) * z / fx
print(u[valid])
y = (v[valid] - cy) * z / fy
points = np.column_stack((x, y, z))
assert points.shape == (7, 3)
assert np.isfinite(points).all()
assert np.allclose(points[0], [-0.005, -0.005, 0.5])
assert np.any(np.all(np.isclose(points, [0.0, 0.0, 0.5]), axis=1))
print("유효한 점 수:", len(points))
print(points)



cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
assert o3d.io.write_point_cloud("practice_cloud.ply", cloud)
# 화면 표시가 가능한 환경에서 아래 줄의 주석을 해제하세요.
o3d.visualization.draw_geometries([cloud])
